// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Diagnostics;
using System.Text.Json;

namespace InferenceWeb.Tests;

public sealed class CudaPtxBuildSettingsTests
{
    private const string Nvcc130 = "Cuda compilation tools, release 13.0, V13.0.88";
    private const string Nvcc134 = "Cuda compilation tools, release 13.4, V13.4.92";
    private const string Nvcc128 = "Cuda compilation tools, release 12.8, V12.8.93";

    [Theory]
    [InlineData("compute_75", "", false, false, false)]
    [InlineData("compute_80", "", false, false, false)]
    [InlineData("compute_120", "", false, false, true)]
    [InlineData("sm_120a", "", false, false, true)]
    [InlineData("compute_121", "", false, false, false)]
    [InlineData("compute_75", "75,86,120", false, false, true)]
    [InlineData("compute_120", "", true, false, false)]
    [InlineData("compute_120", "", false, true, false)]
    public async Task OptionalVerify8BuildNeverAddsTheCoreToPortableOrSkippedBuilds(string architecture,
        string fatbins, bool skip, bool refresh, bool expected)
    {
        var result = await RunTarget("ResolveNinferVerify8BuildPolicy", new()
        {
            ["_ResolvedCudaArch"] = architecture, ["_NvccExitCode"] = "0",
            ["TensorSharpSkipCudaPtx"] = skip ? "true" : "false", ["TensorSharpUpdateCommittedPtx"] = refresh ? "true" : "false"
        }, new() { ["CudaFatbinArchitectures"] = fatbins }, "_NinferVerify8BuildEnabled,_NinferVerify8Flags");
        Assert.Equal(0, result.ExitCode);
        using var json = JsonDocument.Parse(result.Output);
        Assert.Equal(expected ? "true" : "false", json.RootElement.GetProperty("Properties").GetProperty("_NinferVerify8BuildEnabled").GetString());
    }

    [Theory]
    [InlineData("compute_75", false)]
    [InlineData("compute_120", true)]
    public async Task OptionalVerify8UsesItsOwnFixedCompilerFlagsAndPairedPublishAssets(string architecture, bool expected)
    {
        string directory = NewDirectory(), tools = Path.Combine(directory, "tools"), output = Path.Combine(directory, "verify8");
        Directory.CreateDirectory(tools);
        try
        {
            string compiler = FakeCompilingNvcc(tools);
            var result = await RunTarget("SelectNinferVerify8AttentionContent", new()
            {
                ["CudaArch"] = architecture, ["CudaNvcc"] = compiler, ["NinferVerify8OutputDir"] = output,
                // These global settings must not relax the dedicated core.
                ["CudaEnableFastMath"] = "true", ["CudaNvccOptimizationFlags"] = "-O0"
            }, getProperties: "_NinferVerify8BuildEnabled,_NinferVerify8Flags", getItems: "Content");
            Assert.True(result.ExitCode == 0, result.Output + result.Error);
            using var json = JsonDocument.Parse(result.Output);
            string flags = json.RootElement.GetProperty("Properties").GetProperty("_NinferVerify8Flags").GetString()!;
            Assert.Contains("-std=c++20 -O3 --ftz=false --fmad=true -arch=sm_120a", flags);
            Assert.DoesNotContain("--use_fast_math", flags); Assert.DoesNotContain("-O0", flags);
            var assets = json.RootElement.GetProperty("Items").GetProperty("Content").EnumerateArray()
                .Where(x => x.TryGetProperty("Link", out var link) && link.GetString()!.Contains("tensorsharp_ninfer_prefill_attention.")).ToArray();
            Assert.True(assets.Length == (expected ? 3 : 0), result.Output + result.Error);
            Assert.Equal(expected, File.Exists(Path.Combine(output, "tensorsharp_ninfer_prefill_attention.fatbin")));
            if (expected)
            {
                Assert.All(assets, x => { Assert.Equal("PreserveNewest", x.GetProperty("CopyToOutputDirectory").GetString()); Assert.Equal("PreserveNewest", x.GetProperty("CopyToPublishDirectory").GetString());
                    Assert.Equal("cuda_kernels/" + Path.GetFileName(x.GetProperty("Identity").GetString()), x.GetProperty("Link").GetString()!.Replace('\\', '/')); });
                string ptx = Path.Combine(output, "tensorsharp_ninfer_prefill_attention.ptx");
                Assert.Equal(Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(File.ReadAllBytes(ptx))),
                    File.ReadAllText(Path.Combine(output, "tensorsharp_ninfer_prefill_attention.fatbin.sha256")).Trim(), ignoreCase: true);
            }
        }
        finally { Directory.Delete(directory, true); }
    }

    [Theory]
    [InlineData(true, null, "compute_120", "compute_86", "compute_75")]
    [InlineData(true, "auto", "compute_120", "compute_86", "compute_75")]
    [InlineData(true, "native", "compute_120", "compute_86", "compute_75")]
    [InlineData(true, "compute_75", "compute_120", "compute_86", "compute_75")]
    [InlineData(true, "sm_75", "compute_120", "compute_86", "compute_75")]
    [InlineData(false, "auto", "compute_120", "compute_75", "compute_120")]
    [InlineData(false, "native", "compute_86", "compute_75", "compute_86")]
    [InlineData(false, "auto", "", "compute_86", "compute_86")]
    [InlineData(false, "compute_120", "compute_86", "compute_75", "compute_120")]
    public async Task ActualMsbuildResolutionKeepsRefreshPortableAndOrdinaryBuildsGpuSpecific(
        bool refresh, string? requested, string detected, string fallback, string expected)
    {
        var properties = new Dictionary<string, string>
        {
            ["TensorSharpUpdateCommittedPtx"] = refresh.ToString(),
            ["_DetectedCudaArch"] = detected,
            ["_DetectCudaArchExitCode"] = "0",
            ["CudaArchFallback"] = fallback,
        };
        if (requested != null) properties["CudaArch"] = requested;
        var result = await RunTarget("ResolveCudaPtxArchitecture", properties);
        Assert.True(result.ExitCode == 0, result.Output + result.Error);
        using var json = JsonDocument.Parse(result.Output);
        var actual = json.RootElement.GetProperty("Properties");
        Assert.Equal(expected, actual.GetProperty("_ResolvedCudaArch").GetString());
        Assert.StartsWith(expected + "_", actual.GetProperty("_CudaBuildSettingsKey").GetString());
    }

    [Theory]
    [InlineData("compute_61")]
    [InlineData("sm_61")]
    [InlineData("compute_70")]
    [InlineData("compute_86")]
    [InlineData("compute_120")]
    [InlineData("sm_120")]
    public async Task RefreshRejectsEveryTargetButTheBaselineBeforeInvokingNvcc(string requested)
    {
        var result = await RunTarget("ResolveCudaPtxArchitecture", new()
        {
            ["TensorSharpUpdateCommittedPtx"] = "true", ["CudaArch"] = requested,
        });
        Assert.NotEqual(0, result.ExitCode);
        Assert.Contains("must target compute_75", result.Output + result.Error);
        Assert.Contains("omit TensorSharpUpdateCommittedPtx", result.Output + result.Error);
    }

    [Fact]
    public async Task CopyTargetRefusesAnInconsistentResolvedArchitectureBeforeWritingFiles()
    {
        var result = await RunTarget("UpdateCommittedCudaPtx", new()
        {
            ["TensorSharpUpdateCommittedPtx"] = "true", ["_NvccExitCode"] = "0",
            ["_ResolvedCudaArch"] = "compute_120",
        });
        Assert.NotEqual(0, result.ExitCode);
        Assert.Contains("Refusing to overwrite the portable committed CUDA PTX baseline", result.Output + result.Error);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task IntermediatePtxIsCopiedOnlyWhenNvccWasAvailable(bool nvccAvailable)
    {
        string directory = NewDirectory();
        string input = Path.Combine(directory, "intermediate"), output = Path.Combine(directory, "output");
        Directory.CreateDirectory(input);
        File.WriteAllText(Path.Combine(input, "probe.ptx"), ".target sm_120\n");
        try
        {
            var result = await RunTarget("CopyCudaPtxToOutput", new()
            {
                ["_NvccExitCode"] = nvccAvailable ? "0" : "1", ["PtxOutputDir"] = input, ["OutDir"] = output,
            });
            Assert.True(result.ExitCode == 0, result.Output + result.Error);
            Assert.Equal(nvccAvailable, File.Exists(Path.Combine(output, "cuda_kernels", "probe.ptx")));
        }
        finally { Directory.Delete(directory, recursive: true); }
    }

    [Fact]
    public async Task ActualRefreshRecompilesStaleGpuSpecificPtxEvenWithAPortableStamp()
    {
        string directory = NewDirectory();
        string tools = Path.Combine(directory, "tools"), kernels = Path.Combine(directory, "kernels");
        string intermediate = Path.Combine(directory, "intermediate"), stamps = Path.Combine(directory, "stamps");
        string baseline = Path.Combine(directory, "baseline");
        foreach (string path in new[] { tools, kernels, intermediate, stamps, baseline }) Directory.CreateDirectory(path);
        FakeCompilingNvcc(tools);   // found on PATH as "nvcc"
        File.WriteAllText(Path.Combine(kernels, "probe.cu"), "// The fake toolchain checks build policy, not CUDA arithmetic.\n");
        string stale = Path.Combine(intermediate, "probe.ptx");
        // The stamp of a refresh by nvcc 13.0: the toolkit is part of the build-settings key.
        string stamp = Path.Combine(stamps, "tensorsharp_cuda_compute_75_O3_extradevicevectorization_nvcc13.0.stamp");
        File.WriteAllText(stale, ".target sm_120\n");
        File.WriteAllText(stamp, "arch=compute_75\n");
        File.SetLastWriteTimeUtc(stale, DateTime.UtcNow.AddMinutes(5));
        File.SetLastWriteTimeUtc(stamp, DateTime.UtcNow.AddMinutes(5));
        try
        {
            var result = await RunTarget("CompileCudaPtx", RefreshProperties(directory, kernels, intermediate, stamps, baseline),
                new() { ["PATH"] = tools + Path.PathSeparator + Environment.GetEnvironmentVariable("PATH") });
            Assert.True(result.ExitCode == 0, result.Output + result.Error);
            foreach (string ptx in new[] { Path.Combine(baseline, "probe.ptx"), stale })
            {
                string text = File.ReadAllText(ptx);
                Assert.Matches(@"(?m)^\.target sm_75\r?$", text);
                Assert.Matches(@"(?m)^\.version 9\.0\r?$", text);
            }
        }
        finally { Directory.Delete(directory, recursive: true); }
    }

    [Fact]
    public async Task RefreshCompilesButNeverCommitsTheSm80OnlyQwen35Kernels()
    {
        string directory = NewDirectory();
        string tools = Path.Combine(directory, "tools"), kernels = Path.Combine(directory, "kernels");
        string intermediate = Path.Combine(directory, "intermediate"), stamps = Path.Combine(directory, "stamps");
        string baseline = Path.Combine(directory, "baseline");
        foreach (string path in new[] { tools, kernels, intermediate, stamps, baseline }) Directory.CreateDirectory(path);
        string stub = FakeCompilingNvcc(tools);
        foreach (string kernel in new[] { "probe.cu", "tensorsharp_q35_kernels.cu" })
            File.WriteAllText(Path.Combine(kernels, kernel), "// fake\n");
        try
        {
            var properties = RefreshProperties(directory, kernels, intermediate, stamps, baseline);
            properties["CudaNvcc"] = stub;   // the side-by-side toolkit a refresh points at
            var result = await RunTarget("CompileCudaPtx", properties);
            Assert.True(result.ExitCode == 0, result.Output + result.Error);
            Assert.True(File.Exists(Path.Combine(intermediate, "tensorsharp_q35_kernels.ptx")));
            Assert.True(File.Exists(Path.Combine(baseline, "probe.ptx")));
            Assert.False(File.Exists(Path.Combine(baseline, "tensorsharp_q35_kernels.ptx")));
        }
        finally { Directory.Delete(directory, recursive: true); }
    }

    [Theory]
    [InlineData("// Cuda compilation tools, release 13.0, V13.0.88\r\n\r\n.version 9.0\r\n.target sm_75\r\n.address_size 64\r\n", true)]
    [InlineData("// Cuda compilation tools, release 13.0, V13.0.88\n\n.version 9.0\n.target sm_75\n.address_size 64\n", true)]
    [InlineData("// Cuda compilation tools, release 13.4, V13.4.92\n\n.version 9.4\n.target sm_75\n", false)]
    [InlineData("// Cuda compilation tools, release 13.0, V13.0.88\n\n.version 9.4\n.target sm_75\n", false)]
    [InlineData("// Cuda compilation tools, release 13.0, V13.0.88\n\n.version 9.0\n.target sm_86\n", false)]
    [InlineData("// Cuda compilation tools, release 12.8, V12.8.93\n\n.version 8.7\n.target sm_61\n", false)]
    [InlineData(".version 9.0\n.target sm_75\n", false)]
    public async Task CommittedCopyAcceptsOnlyANvcc130Compute75Header(string ptx, bool accepted)
    {
        string directory = NewDirectory();
        string kernels = Path.Combine(directory, "kernels"), intermediate = Path.Combine(directory, "intermediate");
        string baseline = Path.Combine(directory, "baseline");
        foreach (string path in new[] { kernels, intermediate, baseline }) Directory.CreateDirectory(path);
        File.WriteAllText(Path.Combine(kernels, "probe.cu"), "// fake\n");
        File.WriteAllText(Path.Combine(intermediate, "probe.ptx"), ptx);
        try
        {
            var result = await RunTarget("UpdateCommittedCudaPtx", new()
            {
                ["TensorSharpUpdateCommittedPtx"] = "true", ["_NvccExitCode"] = "0", ["_ResolvedCudaArch"] = "compute_75",
                ["CudaKernelDir"] = kernels, ["PtxOutputDir"] = intermediate, ["CudaCommittedPtxDir"] = baseline,
            });
            Assert.True((result.ExitCode == 0) == accepted, result.Output + result.Error);
            Assert.Equal(accepted, File.Exists(Path.Combine(baseline, "probe.ptx")));
            if (!accepted)
                Assert.Contains("Refusing to commit probe.ptx", result.Output + result.Error);
        }
        finally { Directory.Delete(directory, recursive: true); }
    }

    [Fact]
    public async Task NoNvccFallbackReplacesANewerGpuSpecificDestinationThroughSdkCopyTargets()
    {
        string directory = NewDirectory();
        string baseline = Path.Combine(directory, "baseline"), output = Path.Combine(directory, "output");
        string tools = Path.Combine(directory, "tools");
        foreach (string path in new[] { baseline, tools, Path.Combine(output, "cuda_kernels") }) Directory.CreateDirectory(path);
        string stub = Path.Combine(tools, OperatingSystem.IsWindows() ? "nvcc.cmd" : "nvcc");
        File.WriteAllText(stub, OperatingSystem.IsWindows() ? "@exit /b 1\r\n" : "#!/bin/sh\nexit 1\n");
        if (!OperatingSystem.IsWindows())
            File.SetUnixFileMode(stub, UnixFileMode.UserRead | UnixFileMode.UserWrite | UnixFileMode.UserExecute);
        File.WriteAllText(Path.Combine(baseline, "probe.ptx"), ".target sm_75\n");
        string destination = Path.Combine(output, "cuda_kernels", "probe.ptx");
        File.WriteAllText(destination, ".target sm_120\n");
        File.SetLastWriteTimeUtc(destination, DateTime.UtcNow.AddMinutes(5));
        try
        {
            var result = await RunTarget("_CopySourceItemsToOutputDirectory", new()
            {
                ["CudaCommittedPtxDir"] = baseline, ["OutDir"] = output + Path.DirectorySeparatorChar,
                ["_GetChildProjectCopyToOutputDirectoryItems"] = "false", ["BuildProjectReferences"] = "false",
            }, new() { ["PATH"] = tools + Path.PathSeparator + Environment.GetEnvironmentVariable("PATH") });
            Assert.True(result.ExitCode == 0, result.Output + result.Error);
            Assert.Equal(".target sm_75", File.ReadAllText(destination).Trim());
        }
        finally { Directory.Delete(directory, recursive: true); }
    }

    [Theory]
    [InlineData(Nvcc134, null, "compute_75")]
    [InlineData(Nvcc130, null, "compute_75")]
    [InlineData(Nvcc134, "compute_90", "compute_90")]
    public async Task FallbackArchitectureIsCompute75UnlessPinned(string versionLine, string? pinned, string expected)
    {
        string directory = NewDirectory();
        try
        {
            string tools = FakeVersionNvcc(directory, versionLine);
            var properties = new Dictionary<string, string> { ["CudaArch"] = "compute_86" };
            if (pinned != null) properties["CudaArchFallback"] = pinned;
            var result = await RunTarget("DetectCudaPtxBuildSettings", properties,
                new() { ["PATH"] = tools + Path.PathSeparator + Environment.GetEnvironmentVariable("PATH") },
                "_CudaArchFallback,_NvccMajor,_NvccMinor");
            Assert.True(result.ExitCode == 0, result.Output + result.Error);
            using var json = JsonDocument.Parse(result.Output);
            var actual = json.RootElement.GetProperty("Properties");
            Assert.Equal(expected, actual.GetProperty("_CudaArchFallback").GetString());
            Assert.Equal("13", actual.GetProperty("_NvccMajor").GetString());
        }
        finally { Directory.Delete(directory, recursive: true); }
    }

    [Theory]
    [InlineData(Nvcc128, "reports CUDA 12.8")]
    [InlineData("Cuda compilation tools, release 11.8, V11.8.89", "reports CUDA 11.8")]
    [InlineData("nvcc: something that prints no release line", "no 'release X.Y,' line")]
    public async Task OrdinaryBuildWithAnOlderOrUnreadableNvccFails(string versionLine, string detail)
    {
        string directory = NewDirectory();
        try
        {
            string tools = FakeVersionNvcc(directory, versionLine);
            var result = await RunTarget("ResolveCudaPtxBuildSettings", new() { ["CudaArch"] = "compute_86" },
                new() { ["PATH"] = tools + Path.PathSeparator + Environment.GetEnvironmentVariable("PATH") });
            Assert.NotEqual(0, result.ExitCode);
            Assert.Contains("TensorSharp requires CUDA 13 or newer", result.Output + result.Error);
            Assert.Contains(detail, result.Output + result.Error);
        }
        finally { Directory.Delete(directory, recursive: true); }
    }

    [Fact]
    public async Task OrdinaryBuildWithACuda13NvccResolvesWithTheToolkitInTheStampKey()
    {
        string directory = NewDirectory();
        try
        {
            string tools = FakeVersionNvcc(directory, Nvcc134);
            var result = await RunTarget("ResolveCudaPtxBuildSettings", new() { ["CudaArch"] = "compute_86", ["CudaEmitFatbin"] = "false" },
                new() { ["PATH"] = tools + Path.PathSeparator + Environment.GetEnvironmentVariable("PATH") });
            Assert.True(result.ExitCode == 0, result.Output + result.Error);
            using var json = JsonDocument.Parse(result.Output);
            var actual = json.RootElement.GetProperty("Properties");
            Assert.Equal("compute_86", actual.GetProperty("_ResolvedCudaArch").GetString());
            Assert.EndsWith("_nvcc13.4", actual.GetProperty("_CudaBuildSettingsKey").GetString());
        }
        finally { Directory.Delete(directory, recursive: true); }
    }

    // The native build scripts and CMake take CUDACXX ahead of PATH, and their CUDA 12
    // errors tell the user to set it. The managed build must use the same nvcc, or it
    // fails on an older nvcc on PATH that the native build was told to skip.
    [Fact]
    public async Task OrdinaryBuildTakesTheCudacxxNvccAheadOfAnOlderOneOnPath()
    {
        string directory = NewDirectory();
        try
        {
            string path = FakeVersionNvcc(Path.Combine(directory, "cuda12"), Nvcc128);
            string cudacxx = Path.Combine(FakeVersionNvcc(Path.Combine(directory, "cuda13"), Nvcc134),
                OperatingSystem.IsWindows() ? "nvcc.cmd" : "nvcc");
            var result = await RunTarget("ResolveCudaPtxBuildSettings", new() { ["CudaArch"] = "compute_86", ["CudaEmitFatbin"] = "false" },
                new()
                {
                    ["PATH"] = path + Path.PathSeparator + Environment.GetEnvironmentVariable("PATH"),
                    // Quoted as a shell user would write a path with spaces; the quotes are not part of it.
                    ["CUDACXX"] = "\"" + cudacxx + "\"",
                },
                getProperties: "_ResolvedCudaArch,_CudaBuildSettingsKey,CudaNvcc");
            Assert.True(result.ExitCode == 0, result.Output + result.Error);
            using var json = JsonDocument.Parse(result.Output);
            var actual = json.RootElement.GetProperty("Properties");
            Assert.Equal(cudacxx, actual.GetProperty("CudaNvcc").GetString());
            Assert.EndsWith("_nvcc13.4", actual.GetProperty("_CudaBuildSettingsKey").GetString());
        }
        finally { Directory.Delete(directory, recursive: true); }
    }

    [Fact]
    public async Task CudaNvccPropertyBeatsCudacxx()
    {
        string directory = NewDirectory();
        try
        {
            string cudacxx = Path.Combine(FakeVersionNvcc(Path.Combine(directory, "cuda12"), Nvcc128),
                OperatingSystem.IsWindows() ? "nvcc.cmd" : "nvcc");
            string pinned = Path.Combine(FakeVersionNvcc(Path.Combine(directory, "cuda13"), Nvcc134),
                OperatingSystem.IsWindows() ? "nvcc.cmd" : "nvcc");
            var result = await RunTarget("ResolveCudaPtxBuildSettings",
                new() { ["CudaArch"] = "compute_86", ["CudaEmitFatbin"] = "false", ["CudaNvcc"] = pinned },
                new() { ["CUDACXX"] = cudacxx },
                getProperties: "_ResolvedCudaArch,_CudaBuildSettingsKey,CudaNvcc");
            Assert.True(result.ExitCode == 0, result.Output + result.Error);
            using var json = JsonDocument.Parse(result.Output);
            var actual = json.RootElement.GetProperty("Properties");
            Assert.Equal(pinned, actual.GetProperty("CudaNvcc").GetString());
            Assert.EndsWith("_nvcc13.4", actual.GetProperty("_CudaBuildSettingsKey").GetString());
        }
        finally { Directory.Delete(directory, recursive: true); }
    }

    [Fact]
    public async Task OrdinaryBuildWithACuda12CudacxxFails()
    {
        string directory = NewDirectory();
        try
        {
            string cudacxx = Path.Combine(FakeVersionNvcc(Path.Combine(directory, "cuda12"), Nvcc128),
                OperatingSystem.IsWindows() ? "nvcc.cmd" : "nvcc");
            var result = await RunTarget("ResolveCudaPtxBuildSettings", new() { ["CudaArch"] = "compute_86" },
                new() { ["CUDACXX"] = cudacxx });
            Assert.NotEqual(0, result.ExitCode);
            Assert.Contains("reports CUDA 12.8", result.Output + result.Error);
        }
        finally { Directory.Delete(directory, recursive: true); }
    }

    [Fact]
    public async Task RefreshWithCuda13_4FailsBeforeCompiling()
    {
        string directory = NewDirectory();
        try
        {
            string tools = FakeVersionNvcc(directory, Nvcc134);
            var result = await RunTarget("ResolveCudaPtxBuildSettings",
                new() { ["TensorSharpUpdateCommittedPtx"] = "true", ["CudaPtxStampDir"] = Path.Combine(directory, "stamps") },
                new() { ["PATH"] = tools + Path.PathSeparator + Environment.GetEnvironmentVariable("PATH") });
            Assert.NotEqual(0, result.ExitCode);
            Assert.Contains("must be built by nvcc 13.0 (found 13.4)", result.Output + result.Error);
            Assert.Contains("install-cuda-redist.py --version 13.0.3", result.Output + result.Error);
        }
        finally { Directory.Delete(directory, recursive: true); }
    }

    [Fact]
    public async Task RefreshWithCuda12_8FailsBeforeCompiling()
    {
        string directory = NewDirectory();
        try
        {
            string tools = FakeVersionNvcc(directory, Nvcc128);
            var result = await RunTarget("ResolveCudaPtxBuildSettings",
                new() { ["TensorSharpUpdateCommittedPtx"] = "true", ["CudaPtxStampDir"] = Path.Combine(directory, "stamps") },
                new() { ["PATH"] = tools + Path.PathSeparator + Environment.GetEnvironmentVariable("PATH") });
            Assert.NotEqual(0, result.ExitCode);
            Assert.Contains("TensorSharp requires CUDA 13 or newer", result.Output + result.Error);
        }
        finally { Directory.Delete(directory, recursive: true); }
    }

    [Fact]
    public async Task RefreshWithCuda13_0ResolvesCompute75ThroughCudaNvcc()
    {
        string directory = NewDirectory();
        try
        {
            // The 13.0 toolkit sits beside the PATH one and is selected with -p:CudaNvcc.
            string tools = FakeVersionNvcc(directory, Nvcc130);
            string stub = Path.Combine(tools, OperatingSystem.IsWindows() ? "nvcc.cmd" : "nvcc");
            var result = await RunTarget("ResolveCudaPtxBuildSettings", new()
            {
                ["TensorSharpUpdateCommittedPtx"] = "true", ["CudaNvcc"] = stub,
                // Keep the target's stamp <Delete> away from the real obj/cuda_ptx.
                ["CudaPtxStampDir"] = Path.Combine(directory, "stamps"),
            });
            Assert.True(result.ExitCode == 0, result.Output + result.Error);
            using var json = JsonDocument.Parse(result.Output);
            var actual = json.RootElement.GetProperty("Properties");
            Assert.Equal("compute_75", actual.GetProperty("_ResolvedCudaArch").GetString());
            Assert.Equal("compute_75_O3_extradevicevectorization_nvcc13.0", actual.GetProperty("_CudaBuildSettingsKey").GetString());
        }
        finally { Directory.Delete(directory, recursive: true); }
    }

    [Fact]
    public async Task ArchResolverFallbackMovesUpToTheOldestSupportedTargetNeverTheNewest()
    {
        // nvcc 13 cannot compile compute_61; the resolver must pick compute_75 for it, not
        // the newest architecture the toolkit lists (which no older GPU could load).
        string directory = NewDirectory();
        try
        {
            string tools = Path.Combine(directory, "tools");
            Directory.CreateDirectory(tools);
            string[] archs = { "compute_75", "compute_80", "compute_86", "compute_90", "compute_120", "compute_121" };
            string nvcc, nvidiaSmi;
            if (OperatingSystem.IsWindows())
            {
                nvcc = Path.Combine(tools, "fake-nvcc.cmd");
                File.WriteAllText(nvcc, "@echo off\r\nif \"%~1\"==\"--list-gpu-arch\" (" +
                    string.Join("& ", archs.Select(a => "echo " + a)) + "& exit /b 0)\r\nexit /b 1\r\n");
                nvidiaSmi = Path.Combine(tools, "nvidia-smi.cmd");
                File.WriteAllText(nvidiaSmi, "@exit /b 1\r\n");
            }
            else
            {
                nvcc = Path.Combine(tools, "fake-nvcc");
                File.WriteAllText(nvcc, "#!/bin/sh\nif [ \"$1\" = \"--list-gpu-arch\" ]; then printf '%s\\n' " +
                    string.Join(" ", archs) + "; exit 0; fi\nexit 1\n");
                nvidiaSmi = Path.Combine(tools, "nvidia-smi");
                File.WriteAllText(nvidiaSmi, "#!/bin/sh\nexit 1\n");
                foreach (string script in new[] { nvcc, nvidiaSmi })
                    File.SetUnixFileMode(script, UnixFileMode.UserRead | UnixFileMode.UserWrite | UnixFileMode.UserExecute);
            }

            string native = Path.Combine(RepositoryRoot(), "TensorSharp.Backends.Cuda", "native");
            var start = OperatingSystem.IsWindows()
                ? new ProcessStartInfo("powershell") { ArgumentList = { "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                    Path.Combine(native, "resolve-cuda-arch.ps1"), "-FallbackArch", "compute_61", "-Nvcc", nvcc } }
                : new ProcessStartInfo("sh") { ArgumentList = { Path.Combine(native, "resolve-cuda-arch.sh"), "compute_61", nvcc } };
            start.UseShellExecute = false;
            start.RedirectStandardOutput = true;
            start.RedirectStandardError = true;
            start.Environment.Remove("TENSORSHARP_CUDA_ARCH");
            start.Environment["PATH"] = tools + Path.PathSeparator + Environment.GetEnvironmentVariable("PATH");
            using var process = Process.Start(start)!;
            string output = await process.StandardOutput.ReadToEndAsync();
            Assert.True(process.WaitForExit(60_000), "The resolver did not finish.");
            Assert.Equal("compute_75", output.Trim());
        }
        finally { Directory.Delete(directory, recursive: true); }
    }

    [Theory]
    [InlineData("compute_120", "", "true", "-gencode arch=compute_120a,code=sm_120a -gencode arch=compute_120,code=compute_120")]
    [InlineData("sm_86", "", "true", "-gencode arch=compute_86,code=sm_86 -gencode arch=compute_86,code=compute_86")]
    [InlineData("compute_75", "75,86,120", "true",
        "-gencode arch=compute_75,code=sm_75 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_120a,code=sm_120a -gencode arch=compute_75,code=compute_75")]
    [InlineData("compute_120", "", "false", "")]
    public async Task FatbinCarriesSassForTheTargetsPlusThePtx(string arch, string sassList, string emit, string expected)
    {
        // A list reaches MSBuild from the environment: its command line splits -p values on ',' and ';'.
        var environment = sassList.Length > 0 ? new Dictionary<string, string> { ["CudaFatbinArchitectures"] = sassList } : null;
        var result = await RunTarget("ResolveCudaPtxArchitecture", new() { ["CudaArch"] = arch, ["CudaEmitFatbin"] = emit },
            environment, "_CudaFatbinGencode,_CudaBuildSettingsKey");
        Assert.True(result.ExitCode == 0, result.Output + result.Error);
        using var json = JsonDocument.Parse(result.Output);
        var actual = json.RootElement.GetProperty("Properties");
        Assert.Equal(expected, actual.GetProperty("_CudaFatbinGencode").GetString());
        Assert.Equal(expected.Length > 0, actual.GetProperty("_CudaBuildSettingsKey").GetString()!.Contains("_fatbin"));
    }

    [Fact]
    public async Task TheCommittedBaselineRefreshBuildsNoFatbin()
    {
        var result = await RunTarget("ResolveCudaPtxArchitecture", new() { ["TensorSharpUpdateCommittedPtx"] = "true" },
            getProperties: "_CudaFatbinGencode,_ResolvedCudaArch");
        Assert.True(result.ExitCode == 0, result.Output + result.Error);
        using var json = JsonDocument.Parse(result.Output);
        Assert.Equal("", json.RootElement.GetProperty("Properties").GetProperty("_CudaFatbinGencode").GetString());
        Assert.Equal("compute_75", json.RootElement.GetProperty("Properties").GetProperty("_ResolvedCudaArch").GetString());
    }

    [Fact]
    public async Task FatbinAndItsPtxHashAreCopiedBesideThePtx()
    {
        string directory = NewDirectory();
        string input = Path.Combine(directory, "intermediate"), output = Path.Combine(directory, "output");
        Directory.CreateDirectory(input);
        foreach (string name in new[] { "probe.ptx", "probe.fatbin", "probe.fatbin.sha256" })
            File.WriteAllText(Path.Combine(input, name), name);
        try
        {
            var result = await RunTarget("CopyCudaPtxToOutput", new()
            {
                ["_NvccExitCode"] = "0", ["_CudaFatbinGencode"] = "-gencode arch=compute_120,code=sm_120",
                ["PtxOutputDir"] = input, ["OutDir"] = output,
            });
            Assert.True(result.ExitCode == 0, result.Output + result.Error);
            foreach (string name in new[] { "probe.ptx", "probe.fatbin", "probe.fatbin.sha256" })
                Assert.True(File.Exists(Path.Combine(output, "cuda_kernels", name)), name);
        }
        finally { Directory.Delete(directory, recursive: true); }
    }

    [Fact]
    public void ModuleLoaderPrefersOnlyTheFatbinBuiltFromThisPtx()
    {
        string directory = NewDirectory();
        try
        {
            string ptx = Path.Combine(directory, "probe.ptx");
            byte[] ptxBytes = "//\n.version 9.4\n.target sm_120\n"u8.ToArray();
            File.WriteAllBytes(ptx, ptxBytes);
            byte[] image = { 0x50, 0xed, 0x55, 0xba, 1, 2, 3 };
            File.WriteAllBytes(Path.ChangeExtension(ptx, ".fatbin"), image);
            Assert.Null(TensorSharp.Cuda.CudaModule.TryReadMatchingFatbin(ptx, ptxBytes));   // no sidecar

            string sidecar = Path.ChangeExtension(ptx, ".fatbin") + ".sha256";
            File.WriteAllText(sidecar, Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(ptxBytes)).ToLowerInvariant() + "\n");
            Assert.Equal(image, TensorSharp.Cuda.CudaModule.TryReadMatchingFatbin(ptx, ptxBytes));

            byte[] newerPtx = "//\n.version 9.4\n.target sm_120\n// edited\n"u8.ToArray();
            Assert.Null(TensorSharp.Cuda.CudaModule.TryReadMatchingFatbin(ptx, newerPtx));   // stale fatbin
            Assert.Null(TensorSharp.Cuda.CudaModule.TryReadMatchingFatbin(Path.ChangeExtension(ptx, ".bin"), ptxBytes));
        }
        finally { Directory.Delete(directory, recursive: true); }
    }

    private static Dictionary<string, string> RefreshProperties(
        string directory, string kernels, string intermediate, string stamps, string baseline)
    {
        var properties = new Dictionary<string, string>
        {
            ["TensorSharpUpdateCommittedPtx"] = "true", ["CudaArch"] = "auto", ["CudaArchFallback"] = "compute_120",
            ["CudaKernelDir"] = kernels, ["PtxOutputDir"] = intermediate, ["CudaPtxStampDir"] = stamps,
            ["CudaCommittedPtxDir"] = baseline, ["OutDir"] = Path.Combine(directory, "output"),
        };
        // The target sets this whole nvcc flag only on Windows; presetting it skips the vswhere lookup.
        if (OperatingSystem.IsWindows()) properties["_CompilerBinDir"] = "--compiler-bindir=unused-stub-compiler";
        return properties;
    }

    /// <summary>
    /// A fake nvcc 13.0: --version answers like the real toolkit, and a compile writes the PTX
    /// header nvcc 13.0 writes for the requested -arch (.version 9.0, .target sm_NN). The
    /// committed-baseline copy checks exactly that header, so the refresh goes through.
    /// </summary>
    private static string FakeCompilingNvcc(string tools)
    {
        string stub = Path.Combine(tools, OperatingSystem.IsWindows() ? "nvcc.cmd" : "nvcc");
        // cmd.exe splits a batch file's arguments at '=' as well as at spaces, so the target's
        // unquoted -arch=compute_75 arrives as two, -arch and compute_75; a quoted one stays whole.
        if (OperatingSystem.IsWindows())
            File.WriteAllText(stub, """
                @echo off
                if "%~1"=="--version" (echo Cuda compilation tools, release 13.0, V13.0.88& exit /b 0)
                setlocal enabledelayedexpansion
                :next
                if "%~1"=="" goto done
                set "argument=%~1"
                if "!argument:~0,6!"=="-arch=" set "arch=!argument:~6!"
                if "%~1"=="-arch" goto arch
                if "%~1"=="-o" goto output
                shift
                goto next
                :arch
                shift
                set "arch=%~1"
                shift
                goto next
                :output
                shift
                set "out=%~1"
                shift
                goto next
                :done
                if not defined out exit /b 2
                if not defined arch exit /b 3
                (
                    echo // Cuda compilation tools, release 13.0, V13.0.88
                    echo.
                    echo .version 9.0
                    echo .target !arch:compute_=sm_!
                ) > "%out%"
                exit /b 0
                """.ReplaceLineEndings("\r\n") + "\r\n");
        else
        {
            // A raw literal keeps this source file's line endings, and a tree checked out on Windows
            // has CRLF: "#!/bin/sh\r" cannot exec, so dash keeps searching PATH and runs the real
            // toolkit's nvcc. Like nvcc, the stub rejects a bare argument that is not the kernel.
            File.WriteAllText(stub, """
                #!/bin/sh
                if [ "$1" = "--version" ]; then echo "Cuda compilation tools, release 13.0, V13.0.88"; exit 0; fi
                while [ "$#" -gt 0 ]; do
                    case "$1" in
                        -arch=*) arch=${1#-arch=} ;;
                        -o) shift; output=$1 ;;
                        -*) ;;
                        *.cu) input=$1 ;;
                        *) echo "nvcc error : Don't know what to do with '$1'" >&2; exit 1 ;;
                    esac
                    shift
                done
                [ -n "$output" ] && [ -n "$input" ] || exit 2
                printf '// Cuda compilation tools, release 13.0, V13.0.88\n\n.version 9.0\n.target sm_%s\n' "${arch#compute_}" > "$output"
                """.ReplaceLineEndings("\n") + "\n");
            File.SetUnixFileMode(stub, UnixFileMode.UserRead | UnixFileMode.UserWrite | UnixFileMode.UserExecute);
        }
        // A stub that cannot run is skipped for the next nvcc on PATH, so prove it runs first.
        using (var probe = Process.Start(stub, "--version")!)
            Assert.True(probe.WaitForExit(10_000) && probe.ExitCode == 0, "The fake nvcc does not run by itself.");
        return stub;
    }

    private static string FakeVersionNvcc(string directory, string versionLine)
    {
        string tools = Path.Combine(directory, "tools");
        Directory.CreateDirectory(tools);
        string stub = Path.Combine(tools, OperatingSystem.IsWindows() ? "nvcc.cmd" : "nvcc");
        if (OperatingSystem.IsWindows())
            File.WriteAllText(stub, "@echo off\r\nif \"%~1\"==\"--version\" (echo " + versionLine + "& exit /b 0)\r\nexit /b 1\r\n");
        else
        {
            File.WriteAllText(stub, "#!/bin/sh\nif [ \"$1\" = \"--version\" ]; then echo '" + versionLine + "'; exit 0; fi\nexit 1\n");
            File.SetUnixFileMode(stub, UnixFileMode.UserRead | UnixFileMode.UserWrite | UnixFileMode.UserExecute);
        }
        return tools;
    }

    private static string NewDirectory()
    {
        string directory = Path.Combine(RepositoryRoot(), "artifacts", "tests", "CudaPtxBuildSettings", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        return directory;
    }

    private static string RepositoryRoot()
    {
        for (var directory = new DirectoryInfo(AppContext.BaseDirectory); directory != null; directory = directory.Parent)
            if (File.Exists(Path.Combine(directory.FullName, "TensorSharp.slnx"))) return directory.FullName;
        throw new DirectoryNotFoundException("TensorSharp source tree is required for the CUDA build-settings probes.");
    }

    // Execute the real project targets without restoring or compiling managed
    // projects. Architecture probes need no toolchain; the compilation probes
    // use a fake nvcc and isolated fixture paths, never repository PTX.
    private static async Task<(int ExitCode, string Output, string Error)> RunTarget(
        string target, Dictionary<string, string> properties, Dictionary<string, string>? environment = null,
        string getProperties = "_ResolvedCudaArch,_CudaBuildSettingsKey", string? getItems = null)
    {
        string root = RepositoryRoot();
        string? host = Environment.GetEnvironmentVariable("DOTNET_HOST_PATH");
        var start = new ProcessStartInfo(string.IsNullOrEmpty(host) ? "dotnet" : host)
        {
            WorkingDirectory = root,
            UseShellExecute = false, CreateNoWindow = true,
            RedirectStandardOutput = true, RedirectStandardError = true,
        };
        // CUDACXX picks the nvcc too, so a developer's own setting must not replace the fake ones.
        foreach (string name in new[] { "CudaArch", "CudaArchFallback", "TensorSharpUpdateCommittedPtx", "TENSORSHARP_CUDA_ARCH",
                     "CudaEmitFatbin", "CudaFatbinArchitectures", "CudaNvcc", "CUDACXX" })
            start.Environment.Remove(name);
        if (environment != null)
            foreach ((string name, string value) in environment) start.Environment[name] = value;
        foreach (string argument in new[] { "msbuild", Path.Combine(root, "TensorSharp.Backends.Cuda", "TensorSharp.Backends.Cuda.csproj"),
            "-nologo", "-verbosity:quiet", "-target:" + target, "-getProperty:" + getProperties })
            start.ArgumentList.Add(argument);
        if (getItems != null) start.ArgumentList.Add("-getItem:" + getItems);
        foreach ((string name, string value) in properties) start.ArgumentList.Add("-property:" + name + "=" + value);
        using var process = Process.Start(start)!;
        var output = process.StandardOutput.ReadToEndAsync();
        var error = process.StandardError.ReadToEndAsync();
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(30));
        try { await process.WaitForExitAsync(timeout.Token); }
        catch (OperationCanceledException)
        {
            process.Kill(entireProcessTree: true);
            Assert.Fail("The CUDA build-settings probe did not complete within 30 seconds.");
        }
        return (process.ExitCode, await output, await error);
    }
}
