// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// The three direct-CUDA prefill kernel modules (tensorsharp_flash_prefill, tensorsharp_moe_prefill,
// tensorsharp_ssm_prefill) each carry a ts_<module>_caps probe that reports the source version the
// loaded image was compiled from. The C# launchers refuse an image whose version is not theirs, so a
// committed compute_75 PTX that was not refreshed after a kernel change keeps every route on its
// fallback (the lesson of Dsv4Kernels.DecodesQ5_1Experts). These tests pin the three halves of that
// contract: the source and C# agree on the number, the committed baseline reports it, and a stale
// image is refused.
using System;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using TensorSharp.Cuda;
using Xunit;

namespace InferenceWeb.Tests;

public sealed class CudaPrefillModuleTests
{
    public static TheoryData<string> Modules() => new() { "flash_prefill", "moe_prefill", "ssm_prefill" };

    private static string RepoFile(params string[] parts)
    {
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir != null; dir = dir.Parent)
        {
            string candidate = Path.Combine(new[] { dir.FullName }.Concat(parts).ToArray());
            if (File.Exists(candidate))
                return candidate;
        }
        throw new FileNotFoundException(string.Join("/", parts));
    }

    private static string Source(string module) => RepoFile("TensorSharp.Backends.Cuda", "native", "kernels", $"tensorsharp_{module}.cu");
    private static string CommittedPtx(string module) => RepoFile("TensorSharp.Backends.Cuda", "native", "ptx", $"tensorsharp_{module}.ptx");

    private static int ExpectedVersion(string module) => module switch
    {
        "flash_prefill" => CudaFlashPrefillKernels.CurrentVersion,
        "moe_prefill" => CudaMoePrefillKernels.CurrentVersion,
        "ssm_prefill" => CudaSsmPrefillKernels.CurrentVersion,
        _ => throw new ArgumentOutOfRangeException(nameof(module)),
    };

    private static string CapsKernel(string module) => module switch
    {
        "flash_prefill" => CudaFlashPrefillKernels.CapsKernelName,
        "moe_prefill" => CudaMoePrefillKernels.CapsKernelName,
        "ssm_prefill" => CudaSsmPrefillKernels.CapsKernelName,
        _ => throw new ArgumentOutOfRangeException(nameof(module)),
    };

    private static CudaPrefillModule FromModule(string module, CudaModule loaded) => module switch
    {
        "flash_prefill" => CudaFlashPrefillKernels.FromModule(loaded),
        "moe_prefill" => CudaMoePrefillKernels.FromModule(loaded),
        "ssm_prefill" => CudaSsmPrefillKernels.FromModule(loaded),
        _ => throw new ArgumentOutOfRangeException(nameof(module)),
    };

    private static CudaPrefillModule Create(string module) => module switch
    {
        "flash_prefill" => CudaFlashPrefillKernels.Create(),
        "moe_prefill" => CudaMoePrefillKernels.Create(),
        "ssm_prefill" => CudaSsmPrefillKernels.Create(),
        _ => throw new ArgumentOutOfRangeException(nameof(module)),
    };

    private static bool TryCreate(string module, out CudaPrefillModule kernels, out string reason)
    {
        bool ok;
        switch (module)
        {
            case "flash_prefill": ok = CudaFlashPrefillKernels.TryCreate(out var f, out reason); kernels = f; break;
            case "moe_prefill": ok = CudaMoePrefillKernels.TryCreate(out var m, out reason); kernels = m; break;
            case "ssm_prefill": ok = CudaSsmPrefillKernels.TryCreate(out var s, out reason); kernels = s; break;
            default: throw new ArgumentOutOfRangeException(nameof(module));
        }
        return ok;
    }

    [Theory]
    [MemberData(nameof(Modules))]
    public void SourceVersion_MatchesTheLaunchers_AndTheProbeIsInSourceAndBaseline(string module)
    {
        string text = File.ReadAllText(Source(module));
        string macro = "TS_" + module.ToUpperInvariant() + "_VERSION";
        var define = Regex.Match(text, @"(?m)^[ \t]*#[ \t]*define[ \t]+" + macro + @"[ \t]+(\d+)\b");
        Assert.True(define.Success, $"tensorsharp_{module}.cu defines no {macro}.");
        Assert.True(int.Parse(define.Groups[1].Value) == ExpectedVersion(module),
            $"{macro} is {define.Groups[1].Value} in the source but Cuda*{module}Kernels.CurrentVersion is {ExpectedVersion(module)}: bump both together.");
        Assert.Equal($"ts_{module}_caps", CapsKernel(module));
        Assert.Contains(CapsKernel(module), CudaKernelAvailabilityTests.KernelNames(Source(module)));
        Assert.Contains(CapsKernel(module), CudaKernelAvailabilityTests.PtxEntries(CommittedPtx(module)));

        // The committed PTX must report the same number. CI has no GPU (the CudaTheory below never
        // runs there), so read the literal the probe stores first, at offset 0 of its output
        // (the arch follows at +4):  mov.u32 %rA, <version>;  st.global.u32 [%rdB], %rA;
        // A PR that bumps the source and C# constants but not the PTX fails here on every machine.
        var store = Regex.Matches(File.ReadAllText(CommittedPtx(module)),
            @"mov\.u32\s+(%r\d+),\s*(\d+);\s+st\.global\.u32\s+\[%rd\d+\],\s*\1;");
        Assert.True(store.Count == 1, $"tensorsharp_{module}.ptx: expected one version store at offset 0, found {store.Count}; update this pattern if the probe changed");
        Assert.True(int.Parse(store[0].Groups[2].Value) == ExpectedVersion(module),
            $"the committed tensorsharp_{module}.ptx reports version {store[0].Groups[2].Value} but {macro} is {ExpectedVersion(module)}: " +
            "refresh the committed PTX with the source (nvcc 13.0.88, DEVELOPMENT.md).");
    }

    /// <summary>The committed compute_75 baseline JITs here, and its probe reports the version the
    /// launchers expect (a source bump without a PTX refresh fails here first) and arch 75.</summary>
    [CudaTheory]
    [MemberData(nameof(Modules))]
    public void CommittedBaseline_ReportsTheCurrentVersion_AtCompute75(string module)
    {
        using var allocator = new CudaAllocator();
        allocator.Context.MakeCurrent();
        using var kernels = FromModule(module, CudaModule.LoadFromBytes(File.ReadAllBytes(CommittedPtx(module))));
        Assert.Equal(ExpectedVersion(module), kernels.Version);
        Assert.True(kernels.IsCurrent, kernels.StaleReason);
        Assert.Null(kernels.StaleReason);
        Assert.Equal(75, kernels.ImageArch);
        Assert.Equal(75, kernels.CompiledArch);
        Assert.False(kernels.ArchGuardedKernelsUsable);
    }

    /// <summary>The image this process loads (the build output's cuda_kernels: nvcc's PTX and
    /// fatbin, or the committed baseline without nvcc) is current, and the probe's own arch
    /// reading agrees with the driver's.</summary>
    [CudaTheory]
    [MemberData(nameof(Modules))]
    public void LoadedModule_IsCurrent_AndBothArchReadingsAgree(string module)
    {
        using var allocator = new CudaAllocator();
        allocator.Context.MakeCurrent();
        using (var kernels = Create(module))
        {
            Assert.True(kernels.IsCurrent, kernels.StaleReason);
            Assert.True(kernels.ImageArch >= 75, $"probe arch {kernels.ImageArch}");
            Assert.Equal(kernels.CompiledArch, kernels.ImageArch);
            var device = CudaDevice.GetDevice(allocator.DeviceId);
            Assert.Equal(device.ComputeCapabilityMajor, kernels.DeviceCcMajor);
            Assert.Equal(kernels.ImageArch >= 80 && device.ComputeCapabilityMajor >= 8, kernels.ArchGuardedKernelsUsable);
        }
        Assert.True(TryCreate(module, out var current, out string reason), reason);
        using (current)
        {
            Assert.Null(reason);
            Assert.True(current.IsCurrent);
        }
    }

    /// <summary>An image whose probe reports another version (here: the committed PTX with its
    /// version constant rewritten), or that has no probe at all (a module from before the probe),
    /// is refused with a reason.</summary>
    [CudaTheory]
    [MemberData(nameof(Modules))]
    public void StaleOrProbelessImage_IsRefused(string module)
    {
        using var allocator = new CudaAllocator();
        allocator.Context.MakeCurrent();

        string ptx = File.ReadAllText(CommittedPtx(module));
        // The probe stores the version, then the arch, through one base register:
        //   mov.u32 %rA, <version>;  st.global.u32 [%rdB], %rA;
        var store = new Regex(@"mov\.u32\s+(%r\d+),\s*" + ExpectedVersion(module) + @";(\s+st\.global\.u32\s+\[%rd\d+\],\s*\1;)");
        Assert.True(store.Matches(ptx).Count == 1, "the probe's version store was not found once in the committed PTX; update this test's pattern");
        string stale = store.Replace(ptx, m => $"mov.u32 \t{m.Groups[1].Value}, {ExpectedVersion(module) + 1000};{m.Groups[2].Value}");
        using (var kernels = FromModule(module, CudaModule.LoadFromBytes(System.Text.Encoding.ASCII.GetBytes(stale))))
        {
            Assert.Equal(ExpectedVersion(module) + 1000, kernels.Version);
            Assert.False(kernels.IsCurrent);
            Assert.Contains("stale", kernels.StaleReason);
            Assert.Contains(CapsKernel(module), kernels.StaleReason);
        }

        // Another module's image has no such probe: version 0, never current.
        string other = RepoFile("TensorSharp.Backends.Cuda", "native", "ptx", "tensorsharp_ltx25.ptx");
        using (var probeless = FromModule(module, CudaModule.LoadFromBytes(File.ReadAllBytes(other))))
        {
            Assert.Equal(0, probeless.Version);
            Assert.Equal(0, probeless.CompiledArch);
            Assert.False(probeless.IsCurrent);
            Assert.False(probeless.ArchGuardedKernelsUsable);
            Assert.NotNull(probeless.StaleReason);
        }
    }
}
