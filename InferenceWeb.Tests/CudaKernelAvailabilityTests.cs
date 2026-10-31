// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// The committed direct-CUDA PTX (native/ptx/tensorsharp_kernels.ptx) is what every
// machine without nvcc loads, so it is a tracked artifact that has to keep up with
// the kernel source - and it is a compute_75 baseline, so the kernels whose bodies
// are guarded by __CUDA_ARCH__ >= 800 compile to a trap or to nothing in it. Their
// symbols still resolve, which is why availability must come from the module's
// compile target and never from a successful symbol lookup alone.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using TensorSharp.Cuda;
using Xunit;

namespace InferenceWeb.Tests;

public sealed class CudaKernelAvailabilityTests
{
    private static string RepoFile(params string[] parts)
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir != null)
        {
            string candidate = Path.Combine(new[] { dir.FullName }.Concat(parts).ToArray());
            if (File.Exists(candidate))
                return candidate;
            dir = dir.Parent;
        }
        throw new FileNotFoundException(string.Join("/", parts));
    }

    private static string KernelSource => RepoFile("TensorSharp.Backends.Cuda", "native", "kernels", "tensorsharp_kernels.cu");
    private static string CommittedPtx => RepoFile("TensorSharp.Backends.Cuda", "native", "ptx", "tensorsharp_kernels.ptx");

    /// <summary>
    /// The committed baselines as native/ptx holds them, so a module added there is covered by
    /// every theory over them without editing a list
    /// (<see cref="CommittedPtx_IsExactlyThePortableModules_NeverQ35"/> pins the set).
    /// </summary>
    public static TheoryData<string> CommittedModules()
    {
        var modules = new TheoryData<string>();
        foreach (string module in CommittedModuleNames())
            modules.Add(module);
        return modules;
    }

    private static string[] CommittedModuleNames() =>
        Directory.GetFiles(Path.GetDirectoryName(CommittedPtx)!, "*.ptx")
            .Select(path => Path.GetFileNameWithoutExtension(path))
            .OrderBy(name => name, StringComparer.Ordinal)
            .ToArray();

    /// <summary>
    /// Every committed baseline keeps up with its source. Refresh on Linux with nvcc 13.0:
    /// <c>dotnet build TensorSharp.Backends.Cuda -c Release -p:TensorSharpUpdateCommittedPtx=true -p:CudaNvcc=&lt;13.0&gt;/bin/nvcc</c>.
    /// tensorsharp_q35_kernels has no committed baseline by design (its engine needs nvcc).
    /// </summary>
    [Theory]
    [MemberData(nameof(CommittedModules))]
    public void CommittedPtx_HasAnEntryForEveryKernelInTheSource(string module)
    {
        string source = RepoFile("TensorSharp.Backends.Cuda", "native", "kernels", module + ".cu");
        string ptx = RepoFile("TensorSharp.Backends.Cuda", "native", "ptx", module + ".ptx");
        var missing = KernelNames(source).Except(PtxEntries(ptx)).OrderBy(n => n).ToList();
        Assert.True(missing.Count == 0,
            $"The committed {module}.ptx has no .entry for: " + string.Join(", ", missing) +
            ". Refresh native/ptx with nvcc 13.0 (see DEVELOPMENT.md).");
    }

    [Fact]
    public void CommittedPtx_HasNoEntryTheSourceNoLongerDefines()
    {
        // The direction that can be checked today: everything in the baseline still
        // exists in the source (a renamed kernel would otherwise keep its stale twin
        // alive on no-nvcc machines). Also proves the scanner reads the source,
        // including the macro-generated kernels.
        var source = KernelNames(KernelSource);
        var entries = PtxEntries(CommittedPtx);
        Assert.Contains("ts_quant_matmul_lowk_dp4a_f32_q3_16", source);   // two-level macro expansion
        Assert.Contains("ts_wan_attention_d128_f32", source);             // __launch_bounds__ before the type
        Assert.Contains("ts_vision_attention_mma_d128", source);
        Assert.Contains("ts_quant_matmul_q8_0_mmq_f32", source);          // __launch_bounds__ after the type
        var stale = entries.Except(source).OrderBy(n => n).ToList();
        Assert.True(stale.Count == 0, "PTX entries with no kernel in the source: " + string.Join(", ", stale));
    }

    [NvccFact("tensorsharp_kernels.ptx")]
    public void LocallyBuiltPtx_WhenPresent_HasEveryKernel()
    {
        // With nvcc on PATH the build compiles the source into the test output; that
        // PTX must then cover the whole source. (Without nvcc the output holds the
        // committed baseline, which the theories above cover, and this test skips.)
        string built = Path.Combine(AppContext.BaseDirectory, "cuda_kernels", "tensorsharp_kernels.ptx");
        var missing = KernelNames(KernelSource).Except(PtxEntries(built)).OrderBy(n => n).ToList();
        Assert.True(missing.Count == 0, "The locally built PTX has no .entry for: " + string.Join(", ", missing));
    }

    /// <summary>
    /// The committed baselines are what a driver JITs when there is no nvcc, so each must be
    /// compute_75 PTX from nvcc 13.0 (ISA 9.0): the oldest architecture CUDA 13 compiles, in
    /// the oldest ISA any CUDA 13 driver accepts. Newer PTX fails to load on an older CUDA 13
    /// driver (CUDA_ERROR_UNSUPPORTED_PTX_VERSION).
    /// </summary>
    [Theory]
    [MemberData(nameof(CommittedModules))]
    public void CommittedPtx_IsCompute75FromNvcc130(string module)
    {
        byte[] ptx = File.ReadAllBytes(RepoFile("TensorSharp.Backends.Cuda", "native", "ptx", module + ".ptx"));
        string head = System.Text.Encoding.ASCII.GetString(ptx, 0, Math.Min(ptx.Length, 4096));
        Assert.Contains("Cuda compilation tools, release 13.0,", head);
        Assert.Matches(new Regex(@"(?m)^\.version 9\.0\r?$"), head);
        Assert.Equal(75, CudaModule.ParsePtxTargetArch(ptx));
    }

    [Fact]
    public void CommittedPtx_IsExactlyThePortableModules_NeverQ35()
    {
        // Every module a build without nvcc loads ships a baseline. The qwen35 one never
        // does: its engine's GEMMs need sm_120a SASS (CommittedBaseline=false in the csproj;
        // a refresh never copies it). The three *_prefill modules are the direct-CUDA prefill
        // plan's (flash attention, grouped MoE, recurrent layers); each carries its
        // ts_<module>_caps version probe (CudaPrefillModuleTests).
        Assert.Equal(
            new[]
            {
                "tensorsharp_dsv4_kernels", "tensorsharp_flash_prefill", "tensorsharp_glm_kernels", "tensorsharp_kernels",
                "tensorsharp_ltx25", "tensorsharp_moe_prefill", "tensorsharp_q4e_kernels", "tensorsharp_ssm_prefill",
            },
            CommittedModuleNames());
    }

    /// <summary>
    /// Every committed baseline JIT-compiles on this machine's driver, and each of its entries
    /// resolves and reports compute_75. The header theory above only implies that a CUDA 13
    /// driver accepts them; the glm, q4e and ltx25 modules load only for their own models, so
    /// nothing else proves it for them. Run with CUDA_CACHE_DISABLE=1 so the driver compiles
    /// instead of reusing a cached image.
    /// </summary>
    [CudaTheory]
    [MemberData(nameof(CommittedModules))]
    public void CommittedPtx_JitCompilesOnThisDriver(string module)
    {
        string path = RepoFile("TensorSharp.Backends.Cuda", "native", "ptx", module + ".ptx");
        using var allocator = new CudaAllocator();
        allocator.Context.MakeCurrent();
        using var loaded = CudaModule.LoadFromBytes(File.ReadAllBytes(path));
        var entries = PtxEntries(path);
        Assert.NotEmpty(entries);
        foreach (string entry in entries)
            Assert.Equal(75, loaded.CompiledArchOf(entry));   // resolves the entry, or throws
    }

    [Fact]
    public void CommittedBaseline_IsCompute75_AndNeverEnablesArchGuardedKernels()
    {
        Assert.Equal(75, CudaModule.ParsePtxTargetArch(File.ReadAllBytes(CommittedPtx)));
        Assert.Equal(86, CudaModule.ParsePtxTargetArch(System.Text.Encoding.ASCII.GetBytes("//\n// header\n.version 9.2\n.target sm_86\n.address_size 64\n")));
        Assert.Equal(120, CudaModule.ParsePtxTargetArch(System.Text.Encoding.ASCII.GetBytes(".version 8.7\n.target sm_120a\n")));
        Assert.Equal(0, CudaModule.ParsePtxTargetArch(System.Text.Encoding.ASCII.GetBytes("// .target sm_90 in a comment\n")));
        Assert.Equal(0, CudaModule.ParsePtxTargetArch(Array.Empty<byte>()));

        foreach (int ccMajor in new[] { 7, 8, 9, 10, 12 })
            Assert.False(CudaKernels.ArchGuardedKernelsUsable(75, ccMajor));
        Assert.False(CudaKernels.ArchGuardedKernelsUsable(0, 9));
        Assert.False(CudaKernels.ArchGuardedKernelsUsable(86, 7));
        Assert.True(CudaKernels.ArchGuardedKernelsUsable(80, 8));
        Assert.True(CudaKernels.ArchGuardedKernelsUsable(86, 8));
        Assert.True(CudaKernels.ArchGuardedKernelsUsable(120, 12));
    }

    [CudaFact]
    public void CommittedBaselineModule_DisablesTensorCoreKernels_LocalBuildKeepsThem()
    {
        using var allocator = new CudaAllocator();
        allocator.Context.MakeCurrent();

        // The baseline loads on any GPU (the driver JITs its compute_75 PTX), and its
        // missing optional kernels must not stop it. The sm_80+ kernels it does carry
        // are empty or trap there, so they must read as unavailable. The JIT output is
        // SASS for this GPU, so this also proves the arch comes from what the code was
        // compiled from (75), not from the binary the driver produced.
        using (var baseline = CudaKernels.FromModule(CudaModule.LoadFromBytes(File.ReadAllBytes(CommittedPtx))))
        {
            Assert.Equal(75, baseline.CompiledArch);
            Assert.False(baseline.HasQ80Mmq);
            foreach (int headDim in new[] { 64, 72, 80, 96, 112, 128 })
                Assert.False(baseline.HasVisionAttentionMma(headDim));
        }
        string committedDsv4 = RepoFile("TensorSharp.Backends.Cuda", "native", "ptx", "tensorsharp_dsv4_kernels.ptx");
        using (var baseline = Dsv4Kernels.FromModule(CudaModule.LoadFromBytes(File.ReadAllBytes(committedDsv4))))
            Assert.False(baseline.HasMoeMma);

        // The kernels this process actually uses: a GPU-specific nvcc build keeps
        // its tensor-core kernels on an sm_80+ device.
        var kernels = allocator.Kernels;
        Assert.NotNull(kernels);
        var device = CudaDevice.GetDevice(allocator.DeviceId);
        bool expected = kernels.CompiledArch >= 80 && device.ComputeCapabilityMajor >= 8;
        Assert.Equal(expected, kernels.HasQ80Mmq);
        if (expected)
            Assert.True(kernels.HasVisionAttentionMma(64));
    }

    /// <summary>
    /// What the release archives ship: PTX for compute_75 (the GPU-less runner's fallback) and,
    /// beside it, a fatbin with SASS for sm_75 to sm_120 whose sidecar names that PTX. On an
    /// sm_80+ GPU the driver runs that GPU's SASS, compiled with __CUDA_ARCH__ &gt;= 800, so the
    /// tensor-core kernels carry their real bodies and must stay enabled although the PTX
    /// header says sm_75. Rebuilt from the local nvcc output: its fatbin beside a stub PTX
    /// that targets sm_75. Needs an nvcc build (the committed PTX has no fatbin) for an sm_80+
    /// GPU at or below this one, so the fatbin's code runs here; skips anywhere else.
    /// </summary>
    [CudaFact(NvccOutput = "tensorsharp_kernels.fatbin", TensorCoreModule = "tensorsharp_kernels")]
    public void FatbinBesideALowerPtxTarget_KeepsTensorCoreKernelsOfTheLoadedSass()
    {
        using var allocator = new CudaAllocator();
        allocator.Context.MakeCurrent();
        string built = Path.Combine(AppContext.BaseDirectory, "cuda_kernels");
        int builtArch = CudaModule.ParsePtxTargetArch(File.ReadAllBytes(Path.Combine(built, "tensorsharp_kernels.ptx")));

        string dir = Directory.CreateTempSubdirectory("ts-release-fatbin-").FullName;
        try
        {
            byte[] stub = System.Text.Encoding.ASCII.GetBytes("//\n// release-style PTX header\n.version 9.0\n.target sm_75\n.address_size 64\n");
            foreach (string name in new[] { "tensorsharp_kernels", "tensorsharp_dsv4_kernels" })
            {
                File.WriteAllBytes(Path.Combine(dir, name + ".ptx"), stub);
                File.Copy(Path.Combine(built, name + ".fatbin"), Path.Combine(dir, name + ".fatbin"));
                File.WriteAllText(Path.Combine(dir, name + ".fatbin.sha256"),
                    Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(stub)));
            }

            using (var kernels = CudaKernels.FromModule(CudaModule.LoadFromFile(Path.Combine(dir, "tensorsharp_kernels.ptx"))))
            {
                Assert.Equal(builtArch, kernels.CompiledArch);
                Assert.True(kernels.HasQ80Mmq);
                foreach (int headDim in new[] { 64, 72, 80, 96, 112, 128 })
                    Assert.True(kernels.HasVisionAttentionMma(headDim));
            }
            using (var dsv4 = Dsv4Kernels.FromModule(CudaModule.LoadFromFile(Path.Combine(dir, "tensorsharp_dsv4_kernels.ptx"))))
                Assert.True(dsv4.HasMoeMma);
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    /// <summary>
    /// Every GEMM of the direct-CUDA qwen35 engine needs the block-scaled FP4 MMA, which only
    /// sm_120a / sm_121a SASS carries; any other image of tensorsharp_q35_kernels compiles it
    /// (and the mma / ldmatrix helpers below sm_80) to __trap(). Such a module must be refused
    /// up front - the support check falls back to the managed path - instead of loading and
    /// killing the context on the first prompt.
    /// </summary>
    [CudaFact(NvccOutput = "tensorsharp_q35_kernels.ptx")]   // an nvcc-only module
    public void Q35Module_WithoutFp4TensorCoreCode_IsRefusedBeforeAnyGemm()
    {
        string path = CudaKernels.LocatePtxPath("tensorsharp_q35_kernels.ptx");
        Assert.NotNull(path);
        using var allocator = new CudaAllocator();
        allocator.Context.MakeCurrent();
        var device = CudaDevice.GetDevice(allocator.DeviceId);

        // The PTX alone (what -p:CudaEmitFatbin=false ships) targets compute_NN, never the
        // architecture-specific sm_120a, so it never carries the FP4 code.
        bool archSpecificPtx = Regex.IsMatch(File.ReadAllText(path), @"^\.target sm_12[01]a\b", RegexOptions.Multiline);
        using (var module = CudaModule.LoadFromBytes(File.ReadAllBytes(path)))
            Assert.Equal(archSpecificPtx, Q35Kernels.RunsTensorCoreKernels(module));

        // As the engine loads it (the fatbin when there is one): off a CC 12 GPU it never runs.
        var loaded = CudaModule.LoadFromFile(path);
        bool runs = Q35Kernels.RunsTensorCoreKernels(loaded);
        if (device.ComputeCapabilityMajor != 12)
            Assert.False(runs);
        if (runs)
        {
            using var kernels = Q35Kernels.FromModule(loaded);
        }
        else
        {
            var refused = Assert.Throws<NotSupportedException>(() => Q35Kernels.FromModule(loaded));
            Assert.Contains("sm_120a", refused.Message);
        }

        // The support check runs the same probe without disturbing this thread's context.
        TensorSharp.Cuda.Interop.CudaDriverApi.cuCtxGetCurrent(out IntPtr before);
        Assert.Equal(runs, Q35Kernels.ProbeTensorCoreKernels(path, out string reason));
        Assert.Equal(runs, reason == null);
        TensorSharp.Cuda.Interop.CudaDriverApi.cuCtxGetCurrent(out IntPtr after);
        Assert.Equal(before, after);
    }

    /// <summary>
    /// Every <c>extern "C" __global__</c> kernel the source defines, following local
    /// <c>#include "..."</c> headers and expanding the function-like macros that emit
    /// kernels (TS_LOWK_KERNELS -> TS_LOWK_KERNEL, TS_VISION_ATTENTION_*_KERNEL, ...),
    /// with <c>##</c> token pasting.
    /// </summary>
    internal static HashSet<string> KernelNames(string cuPath)
    {
        string text = Inline(cuPath, new HashSet<string>(StringComparer.OrdinalIgnoreCase));
        text = Regex.Replace(text, @"\\\r?\n", " ");
        text = Regex.Replace(text, @"/\*.*?\*/", " ", RegexOptions.Singleline);
        text = Regex.Replace(text, @"//[^\n]*", "");

        var macros = new Dictionary<string, (string[] Params, string Body)>(StringComparer.Ordinal);
        var code = new List<string>();
        foreach (string line in text.Split('\n'))
        {
            var define = Regex.Match(line, @"^[ \t]*#[ \t]*define[ \t]+(\w+)\(([^)]*)\)(.*)$");
            if (define.Success)
            {
                string[] parameters = define.Groups[2].Value.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
                macros[define.Groups[1].Value] = (parameters, define.Groups[3].Value);
                continue;
            }
            if (!Regex.IsMatch(line, @"^[ \t]*#[ \t]*define"))
                code.Add(line);
        }

        // Only the macros that (transitively) emit a kernel are expanded.
        var kernelMacros = new HashSet<string>(macros.Where(m => m.Value.Body.Contains("__global__")).Select(m => m.Key));
        for (bool grew = true; grew;)
        {
            grew = false;
            foreach (var (name, macro) in macros)
            {
                if (!kernelMacros.Contains(name) && kernelMacros.Any(k => Regex.IsMatch(macro.Body, @"\b" + k + @"\s*\(")))
                    grew = kernelMacros.Add(name);
            }
        }

        string body = string.Join("\n", code);
        for (int round = 0; ; round++)
        {
            Assert.True(round < 10_000, "kernel macro expansion did not terminate");
            Match call = null;
            string macroName = null;
            foreach (string name in kernelMacros)
            {
                var m = Regex.Match(body, @"\b" + name + @"\s*\(");
                if (m.Success && (call == null || m.Index < call.Index)) { call = m; macroName = name; }
            }
            if (call == null)
                break;
            var (args, end) = SplitArgs(body, call.Index + call.Length - 1);
            var (parameters, macroBody) = macros[macroName];
            string expansion = macroBody;
            for (int i = 0; i < parameters.Length && i < args.Count; i++)
                expansion = Regex.Replace(expansion, @"\b" + Regex.Escape(parameters[i]) + @"\b", args[i].Replace("$", "$$"));
            expansion = Regex.Replace(expansion, @"\s*##\s*", "");
            body = body.Substring(0, call.Index) + " " + expansion + " " + body.Substring(end);
        }

        var names = new HashSet<string>(StringComparer.Ordinal);
        foreach (Match declaration in Regex.Matches(body, "extern\\s+\"C\"\\s+__global__"))
        {
            int i = declaration.Index + declaration.Length;
            while (true)
            {
                while (i < body.Length && char.IsWhiteSpace(body[i])) i++;
                var ident = Regex.Match(body.Substring(i, Math.Min(256, body.Length - i)), @"^\w+");
                if (!ident.Success)
                    break;
                i += ident.Length;
                if (ident.Value == "__launch_bounds__")
                {
                    i = SplitArgs(body, body.IndexOf('(', i)).End;
                    continue;
                }
                int j = i;
                while (j < body.Length && char.IsWhiteSpace(body[j])) j++;
                if (j < body.Length && body[j] == '(')
                {
                    names.Add(ident.Value);
                    break;
                }
            }
        }
        return names;
    }

    internal static HashSet<string> PtxEntries(string ptxPath)
        => new(Regex.Matches(File.ReadAllText(ptxPath), @"\.entry\s+(\w+)\s*\(").Select(m => m.Groups[1].Value), StringComparer.Ordinal);

    private static string Inline(string path, HashSet<string> seen)
    {
        string text = File.ReadAllText(path);
        return Regex.Replace(text, "^[ \\t]*#[ \\t]*include[ \\t]*\"([^\"]+)\"", m =>
        {
            string header = Path.Combine(Path.GetDirectoryName(path)!, m.Groups[1].Value);
            return File.Exists(header) && seen.Add(Path.GetFullPath(header)) ? Inline(header, seen) : "";
        }, RegexOptions.Multiline);
    }

    private static (List<string> Args, int End) SplitArgs(string s, int openParen)
    {
        var args = new List<string>();
        var current = new System.Text.StringBuilder();
        int depth = 0;
        for (int i = openParen; i < s.Length; i++)
        {
            char c = s[i];
            if (c == '(')
            {
                if (depth++ > 0) current.Append(c);
            }
            else if (c == ')')
            {
                if (--depth == 0)
                {
                    args.Add(current.ToString().Trim());
                    return (args, i + 1);
                }
                current.Append(c);
            }
            else if (c == ',' && depth == 1)
            {
                args.Add(current.ToString().Trim());
                current.Clear();
            }
            else
            {
                current.Append(c);
            }
        }
        throw new InvalidDataException("Unbalanced parentheses in kernel source.");
    }
}
