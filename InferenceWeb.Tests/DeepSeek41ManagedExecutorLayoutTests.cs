// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp.Cuda;
using TensorSharp.Models;
using TensorSharp.Models.Architecture;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

/// <summary>
/// What the two managed V4.1 executors -- <c>--backend cuda</c>'s direct-CUDA engine and
/// <c>--backend cpu</c>'s pure C# one -- accept and refuse before a weight is read: both Engram
/// layouts (TensorSharp's per-layer one and llama.cpp's combined one, which unsloth's
/// DeepSeek-V4.1-Flash-1017 GGUFs use) but not a header that mixes them or carries only part of
/// one, and routed-expert quantizations the direct-CUDA engine cannot run where its split puts
/// them. Header-only synthetic GGUFs and descriptors: no weights, native library or GPU.
/// </summary>
public sealed class DeepSeek41ManagedExecutorLayoutTests : IDisposable
{
    private readonly EnvScope _env = new();
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "dsv41-layout-" + Guid.NewGuid().ToString("N"));

    public DeepSeek41ManagedExecutorLayoutTests()
    {
        _env.ClearSpeculationVars();
        foreach (string name in new[] { "TS_DSV41_YOCO_KEEP", "TS_DSV41_ENGRAM_DEVICE", "TS_DSV41_ALLOW_NON_CUDA_GPU",
                     "TENSORSHARP_TP_DEGREE", "TENSORSHARP_LAYER_SPLIT_DEGREE" })
            _env.Set(name, null);
        _env.Set("KV_CACHE_DTYPE", "f16");
        Directory.CreateDirectory(_dir);
    }

    public void Dispose()
    {
        _env.Dispose();
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { /* best effort */ }
    }

    /// <summary><c>Both</c> carries both key sets (TensorSharp's wins, as in the native loader);
    /// the last four mix the layouts or carry part of the combined one.</summary>
    public enum Layout
    {
        TensorSharp, Combined, Both,
        CombinedTableOnly, CombinedKeysOnly, TensorSharpWithTable, CombinedWithPerLayerTable,
    }

    /// <summary>A deepseek41 header in one Engram layout, keys only as the native loader names them
    /// (dsv41_engram_gguf.h). The combined one carries what the 1017 files do: the <c>ple</c> hash
    /// geometry and the raw pad token that the token map compresses.</summary>
    private string Write(Layout layout)
    {
        var kv = new List<SyntheticGguf.Kv>
        {
            new SyntheticGguf.Str { Key = "general.architecture", V = "deepseek41" },
            new SyntheticGguf.U32 { Key = "deepseek41.block_count", V = 43 },
        };
        SyntheticGguf.AddByteTokenizer(kv, 16, 0, 1);
        kv.Add(new SyntheticGguf.U32Arr { Key = "deepseek41.engram.token_map", V = Enumerable.Range(0, 16).Select(i => (uint)(i % 8)).ToArray() });
        if (layout is Layout.TensorSharp or Layout.Both or Layout.TensorSharpWithTable)
        {
            kv.Add(new SyntheticGguf.U32 { Key = "deepseek41.engram.pad_id", V = 2 });
            kv.Add(new SyntheticGguf.U32 { Key = "deepseek41.engram.max_ngram_size", V = 3 });
            kv.Add(new SyntheticGguf.U32 { Key = "deepseek41.engram.head_count", V = 1 });
            kv.Add(new SyntheticGguf.U32 { Key = "deepseek41.engram.key_length", V = 32 });
            kv.Add(new SyntheticGguf.U32Arr { Key = "deepseek41.engram.layer_ids", V = new uint[] { 1, 14 } });
        }
        if (layout is Layout.Combined or Layout.Both or Layout.CombinedKeysOnly or Layout.CombinedWithPerLayerTable)
        {
            kv.Add(new SyntheticGguf.U32 { Key = "deepseek41.ple.ngram_size", V = 3 });
            kv.Add(new SyntheticGguf.U32 { Key = "deepseek41.ple.heads_per_ngram", V = 1 });
            if (layout != Layout.CombinedKeysOnly)
                kv.Add(new SyntheticGguf.U32Arr { Key = "deepseek41.ple.layers", V = new uint[] { 1, 14 } });
            kv.Add(new SyntheticGguf.U64Arr { Key = "deepseek41.ple.layer_multipliers", V = new ulong[] { 7, 9, 11, 13, 15, 17 } });
            kv.Add(new SyntheticGguf.U32Arr { Key = "deepseek41.ple.head_vocab_sizes", V = new uint[] { 11, 13, 17, 19 } });
            kv.Add(new SyntheticGguf.U64Arr { Key = "deepseek41.ple.head_offsets", V = new ulong[] { 0, 11, 24, 41 } });
            kv.Add(new SyntheticGguf.U32 { Key = "deepseek41.embedding_length_per_layer_input", V = 32 });
            kv.Add(new SyntheticGguf.U32 { Key = "deepseek41.engram.pad_token_id", V = 3 });
        }
        var tensors = new List<SyntheticGguf.Tensor>();
        if (layout is Layout.CombinedTableOnly or Layout.TensorSharpWithTable)
            tensors.Add(SyntheticGguf.Gen("per_layer_token_embd.weight", 0.1f, 32, 60));
        if (layout == Layout.CombinedWithPerLayerTable)
            tensors.Add(SyntheticGguf.Gen("blk.1.engram_embd.weight", 0.1f, 32, 24));
        string path = Path.Combine(_dir, layout + ".gguf");
        SyntheticGguf.Write(path, kv, tensors);
        return path;
    }

    /// <summary>
    /// llama.cpp's combined layout reaches both managed executors, which read it through
    /// Dsv41EngramData as the native loader does. They used to refuse it from the header.
    /// </summary>
    [Theory]
    [InlineData(BackendType.Cuda)]
    [InlineData(BackendType.Cpu)]
    public void CombinedEngramLayoutReachesTheManagedExecutors(BackendType backend)
    {
        string path = Write(Layout.Combined);
        DeepSeek41Architecture.ValidateLoad(path, backend, null);
        using var header = new GgufFile(path);
        Assert.True(DeepSeek41Architecture.UsesCombinedEngramLayout(header));
        DeepSeek41Architecture.ValidateLoad("not-opened.gguf", backend, null, header: header);
        DeepSeek41Architecture.Descriptor.ApplyNativeTunables(new ModelCreateContext(path, backend, header));
    }

    /// <summary>ggml_cuda and ggml_cpu go through the native loader, which reads both layouts.</summary>
    [Theory]
    [InlineData(BackendType.GgmlCuda)]
    [InlineData(BackendType.GgmlCpu)]
    public void CombinedEngramLayoutStillReachesTheNativeLoader(BackendType backend)
    {
        string path = Write(Layout.Combined);
        DeepSeek41Architecture.ValidateLoad(path, backend, null);
        using var header = new GgufFile(path);
        DeepSeek41Architecture.ValidateLoad(path, backend, null, header: header);
    }

    /// <summary>Regression guard: TensorSharp's layout, including a header that carries both key
    /// sets (the native loader then reads TensorSharp's), still reaches the managed executors.</summary>
    [Theory]
    [InlineData(BackendType.Cuda, Layout.TensorSharp)]
    [InlineData(BackendType.Cpu, Layout.TensorSharp)]
    [InlineData(BackendType.Cuda, Layout.Both)]
    [InlineData(BackendType.Cpu, Layout.Both)]
    public void TensorSharpEngramLayoutReachesTheManagedExecutors(BackendType backend, Layout layout)
    {
        string path = Write(layout);
        DeepSeek41Architecture.ValidateLoad(path, backend, null);
        using var header = new GgufFile(path);
        Assert.False(DeepSeek41Architecture.UsesCombinedEngramLayout(header));
        DeepSeek41Architecture.ValidateLoad(path, backend, null, header: header);
    }

    /// <summary>
    /// A header that carries part of the combined layout, or both layouts' tables, is refused by
    /// the managed executors before anything is mapped, naming what is wrong. Without this they
    /// fail inside the executor, with "Missing or invalid DeepSeek V4.1 GGUF metadata:
    /// deepseek41.engram.pad_id" for a partial one.
    /// </summary>
    [Theory]
    [InlineData(BackendType.Cuda, Layout.CombinedTableOnly, "without the deepseek41.ple.layers list")]
    [InlineData(BackendType.Cpu, Layout.CombinedTableOnly, "without the deepseek41.ple.layers list")]
    [InlineData(BackendType.Cuda, Layout.CombinedKeysOnly, "without the deepseek41.ple.layers list")]
    [InlineData(BackendType.Cpu, Layout.CombinedKeysOnly, "without the deepseek41.ple.layers list")]
    [InlineData(BackendType.Cuda, Layout.TensorSharpWithTable, "both Engram layouts: TensorSharp's per-layer metadata")]
    [InlineData(BackendType.Cpu, Layout.TensorSharpWithTable, "both Engram layouts: TensorSharp's per-layer metadata")]
    [InlineData(BackendType.Cuda, Layout.CombinedWithPerLayerTable, "both Engram layouts: llama.cpp's combined metadata")]
    [InlineData(BackendType.Cpu, Layout.CombinedWithPerLayerTable, "both Engram layouts: llama.cpp's combined metadata")]
    public void AMixedOrPartialEngramLayoutIsRefusedFromTheHeader(BackendType backend, Layout layout, string problem)
    {
        string path = Write(layout);
        var error = Assert.Throws<NotSupportedException>(() => DeepSeek41Architecture.ValidateLoad(path, backend, null));
        Assert.Contains(problem, error.Message);
        string option = backend == BackendType.Cuda ? "cuda (the direct-CUDA engine)" : "cpu (the pure C# executor)";
        Assert.Contains($"--backend {option} reads either TensorSharp's per-layer Engram layout or llama.cpp's combined one",
            error.Message);
        Assert.DoesNotContain("Missing or invalid", error.Message);

        // The header the caller already holds gives the same answer without reopening the file.
        using var header = new GgufFile(path);
        Assert.Equal(layout == Layout.CombinedWithPerLayerTable, DeepSeek41Architecture.UsesCombinedEngramLayout(header));
        var again = Assert.Throws<NotSupportedException>(() =>
            DeepSeek41Architecture.ValidateLoad("not-opened.gguf", backend, null, header: header));
        Assert.Equal(error.Message, again.Message);

        // The native loader judges these itself.
        DeepSeek41Architecture.ValidateLoad(path, BackendType.GgmlCuda, null);
    }

    /// <summary>The refusal is the descriptor's pre-load hook, so ModelBase.Create stops before the
    /// factory builds an executor -- this header has no weights for one to read.</summary>
    [Theory]
    [InlineData(BackendType.Cuda)]
    [InlineData(BackendType.Cpu)]
    public void TheRegistryLoadStopsBeforeTheExecutorIsBuilt(BackendType backend)
    {
        string path = Write(Layout.CombinedTableOnly);
        using (var probe = new GgufFile(path))
        {
            var context = new ModelCreateContext(path, backend, probe);
            Assert.Contains("without the deepseek41.ple.layers list", Assert.Throws<NotSupportedException>(() =>
                DeepSeek41Architecture.Descriptor.ApplyNativeTunables(context)).Message);
        }
        var error = Assert.Throws<NotSupportedException>(() => ModelBase.Create(path, backend));
        Assert.Contains("without the deepseek41.ple.layers list", error.Message);
    }

    /// <summary>
    /// The real 1017 UD-IQ2_XXS header (shard 1 of 6, metadata only) passes every executor's
    /// pre-load checks in llama.cpp's combined layout. Set TS_TEST_DSV41_1017_HEADER to that shard.
    /// </summary>
    [ModelFact(Dsv41EngramDataTests.Header1017Var)]
    public void TheReal1017HeaderReachesEveryExecutor()
    {
        string path = Environment.GetEnvironmentVariable(Dsv41EngramDataTests.Header1017Var);
        using var header = GgufFile.OpenWithoutSiblingShards(path);
        Assert.True(DeepSeek41Architecture.UsesCombinedEngramLayout(header));
        foreach (BackendType backend in new[] { BackendType.Cuda, BackendType.Cpu, BackendType.GgmlCuda, BackendType.GgmlCpu })
        {
            DeepSeek41Architecture.ValidateLoad(path, backend, null);
            DeepSeek41Architecture.ValidateLoad("not-opened.gguf", backend, null, header: header);
        }
    }

    // ---- routed-expert types on the direct-CUDA engine ----

    private const int Embd = 512, Ff = 256;

    private static CudaWeightDesc Expert(GgmlTensorType type, string name, int ne0, int ne1)
        => new() { GgmlType = (int)type, Name = name, Ne0 = ne0, Ne1 = ne1, Ne2 = 4 };

    private static Dsv4CudaEngine.LayerDesc Layer(int il, GgmlTensorType gateUp, GgmlTensorType down, string prefix = null,
        int embd = Embd, int ff = Ff)
    {
        prefix ??= $"blk.{il}.";
        return new Dsv4CudaEngine.LayerDesc
        {
            GateExps = Expert(gateUp, prefix + "ffn_gate_exps.weight", embd, ff),
            UpExps = Expert(gateUp, prefix + "ffn_up_exps.weight", embd, ff),
            DownExps = Expert(down, prefix + "ffn_down_exps.weight", ff, embd),
        };
    }

    private static Dsv4CudaEngine.ModelDesc Model(params Dsv4CudaEngine.LayerDesc[] layers) => new()
    {
        NLayer = layers.Length,
        Layers = layers,
        HeadDim = 512,
        NExpertUsed = 6,
        HostMatMul = DeepSeek4CudaExecutor.HostMatMul.Instance,
    };

    /// <summary>The check against a kernel module built from the current source, which carries every
    /// expert decoder (the engine passes the loaded module's own markers).</summary>
    private static void Validate(Dsv4CudaEngine.ModelDesc m, int nCpuMoe)
        => Dsv4CudaEngine.ValidateRoutedExpertTypes(m, nCpuMoe, decodesQ5_1: true, decodesLowBitIq: true);

    private static bool Accepts(Dsv4CudaEngine.ModelDesc m, int nCpuMoe)
    {
        try
        {
            Validate(m, nCpuMoe);
            return true;
        }
        catch (NotSupportedException)
        {
            return false;
        }
    }

    /// <summary>
    /// The routed-expert types of unsloth's 1017 UD-IQ2_XXS file against both places the engine can
    /// run them: the GPU expert kernels, shared with GLM and Qwen3.8, run all of them; the host matmul
    /// behind <c>--n-cpu-moe</c> runs neither IQ1_S, IQ2_S nor IQ2_XXS. docs/models/deepseek41.md
    /// quotes this table.
    /// </summary>
    [Theory]
    [InlineData(GgmlTensorType.IQ1_S, true, false)]
    [InlineData(GgmlTensorType.IQ2_S, true, false)]
    [InlineData(GgmlTensorType.IQ2_XS, true, true)]
    [InlineData(GgmlTensorType.IQ2_XXS, true, false)]
    [InlineData(GgmlTensorType.IQ3_XXS, true, true)]
    [InlineData(GgmlTensorType.Q2_K, true, true)]
    public void The1017ExpertTypesAgainstTheGpuKernelsAndTheHostMatmul(GgmlTensorType type, bool gpu, bool host)
    {
        Assert.Equal(gpu, CudaMoe.SupportsExpertType((int)type));
        Assert.Equal(host, DeepSeek4CudaExecutor.HostMatMul.Instance.SupportsType((int)type, Embd));
        Assert.Equal(host, DeepSeek4CudaExecutor.HostMatMul.Instance.SupportsType((int)type, Ff));
        var m = Model(Layer(0, type, type));
        Assert.Equal(gpu, Accepts(m, 0));
        Assert.Equal(host, Accepts(m, 1));
    }

    /// <summary>
    /// The 1017's routed experts at its widths (5120-wide gate/up, 2304-wide down) and counts: gate/up
    /// IQ1_S in 10 layers, IQ2_S in 15 and IQ2_XXS in 15; down IQ2_S in 15, IQ3_XXS in 15, IQ2_XS in 6,
    /// IQ2_XXS in 3 and Q2_K in 1. (The counts are the file's; which layer pairs which gate/up with which
    /// down is this test's, the per-layer listing not being at hand.) GPU-resident they all run; offloaded,
    /// the host refuses the three low-bit IQ layouts by name; and a kernel module older than their decoders
    /// refuses the first such layer by name.
    /// </summary>
    [Fact]
    public void The1017ExpertMatrix_RunsGpuResident_AndIsRefusedOffloaded()
    {
        var gateUp = Enumerable.Repeat(GgmlTensorType.IQ1_S, 10).Concat(Enumerable.Repeat(GgmlTensorType.IQ2_S, 15))
            .Concat(Enumerable.Repeat(GgmlTensorType.IQ2_XXS, 15)).ToArray();
        var down = Enumerable.Repeat(GgmlTensorType.IQ2_S, 15).Concat(Enumerable.Repeat(GgmlTensorType.IQ3_XXS, 15))
            .Concat(Enumerable.Repeat(GgmlTensorType.IQ2_XS, 6)).Concat(Enumerable.Repeat(GgmlTensorType.IQ2_XXS, 3))
            .Append(GgmlTensorType.Q2_K).ToArray();
        Assert.Equal(40, gateUp.Length);
        Assert.Equal(40, down.Length);
        var m = Model(Enumerable.Range(0, 40).Select(il => Layer(il, gateUp[il], down[il], embd: 5120, ff: 2304)).ToArray());

        Validate(m, 0);

        var error = Assert.Throws<NotSupportedException>(() => Validate(m, 1));
        Assert.Contains("cannot run 3 routed-expert tensor(s)", error.Message);
        Assert.Contains("blk.0.ffn_gate_exps.weight is IQ1_S (offloaded by --n-cpu-moe)", error.Message);
        Assert.Contains("blk.0.ffn_up_exps.weight is IQ1_S (offloaded by --n-cpu-moe)", error.Message);
        Assert.Contains("blk.0.ffn_down_exps.weight is IQ2_S (offloaded by --n-cpu-moe)", error.Message);
        Assert.DoesNotContain("blk.1.", error.Message);
        Assert.True(ModelLoadRefusal.TryDescribe(error, out _));

        // Every layer offloaded: all 80 gate/up tensors and the 15 IQ2_S and 3 IQ2_XXS down ones.
        error = Assert.Throws<NotSupportedException>(() => Validate(m, int.MaxValue));
        Assert.Contains("cannot run 98 routed-expert tensor(s)", error.Message);
        Assert.Contains("; and 92 more", error.Message);

        var stale = Assert.Throws<NotSupportedException>(() =>
            Dsv4CudaEngine.ValidateRoutedExpertTypes(m, 0, decodesQ5_1: true, decodesLowBitIq: false));
        Assert.StartsWith("[dsv4-cuda] blk.0.ffn_gate_exps.weight is IQ1_S, and the loaded tensorsharp_dsv4_kernels module",
            stale.Message);
        Assert.True(ModelLoadRefusal.TryDescribe(stale, out _));
    }

    /// <summary>
    /// On a kernel module built from source older than an expert decoder (a committed PTX baseline not
    /// yet regenerated, which a machine without nvcc loads), whose kernels would read those rows as Q8_0,
    /// the engine refuses the GPU-resident layers that need it by tensor name: IQ1_S and IQ2_S without the
    /// low-bit IQ marker, Q5_1 without its own, the drafter's stages as the trunk's. IQ2_XXS passes either
    /// way (such a module keeps it on the warp-per-row kernels, which predate the marker), and an
    /// offloaded layer is the host matmul's to judge.
    /// </summary>
    [Fact]
    public void AKernelModuleOlderThanADecoder_RefusesItsLayersByName()
    {
        void Stale(Dsv4CudaEngine.ModelDesc m, int nCpuMoe, string start, bool decodesQ5_1 = true)
        {
            var error = Assert.Throws<NotSupportedException>(() =>
                Dsv4CudaEngine.ValidateRoutedExpertTypes(m, nCpuMoe, decodesQ5_1, decodesLowBitIq: false));
            Assert.StartsWith(start, error.Message);
            Assert.Contains("Rebuild TensorSharp.Backends.Cuda with a CUDA 13 nvcc on PATH, or run this model with --backend ggml_cuda",
                error.Message);
            Validate(m, nCpuMoe);
        }

        var trunk = Model(Layer(0, GgmlTensorType.IQ2_XXS, GgmlTensorType.IQ3_XXS),
                          Layer(1, GgmlTensorType.IQ2_XXS, GgmlTensorType.IQ2_S),
                          Layer(2, GgmlTensorType.IQ1_S, GgmlTensorType.IQ2_XS));
        Stale(trunk, 0, "[dsv4-cuda] blk.1.ffn_down_exps.weight is IQ2_S, and the loaded tensorsharp_dsv4_kernels module " +
            "was built from kernel source older than its IQ2_S expert decoder");

        // IQ2_XXS alone runs on such a module.
        var iq2xxs = Model(Layer(0, GgmlTensorType.IQ2_XXS, GgmlTensorType.IQ2_XXS),
                           Layer(1, GgmlTensorType.IQ2_XXS, GgmlTensorType.IQ3_XXS));
        Dsv4CudaEngine.ValidateRoutedExpertTypes(iq2xxs, 0, decodesQ5_1: false, decodesLowBitIq: false);

        // Offloaded, layer 0's IQ1_S gate/up never meets the module (a host matmul, were there one for
        // IQ1_S, would run it); without a host matmul the engine refuses the offload itself.
        var offloaded = Model(Layer(0, GgmlTensorType.IQ1_S, GgmlTensorType.Q2_K), Layer(1, GgmlTensorType.Q2_K, GgmlTensorType.Q2_K));
        offloaded.HostMatMul = null;
        Dsv4CudaEngine.ValidateRoutedExpertTypes(offloaded, 1, decodesQ5_1: true, decodesLowBitIq: false);
        Stale(offloaded, 0, "[dsv4-cuda] blk.0.ffn_gate_exps.weight is IQ1_S, and");

        // The drafter's stages, named after their stage when the loader gave them no name.
        var drafted = Model(Layer(0, GgmlTensorType.Q2_K, GgmlTensorType.Q2_K));
        drafted.Dspark = new Dsv4CudaEngine.DsparkDesc
        {
            Stages = new[]
            {
                new Dsv4CudaEngine.LayerDesc
                {
                    GateExps = Expert(GgmlTensorType.IQ2_XXS, null, Embd, Ff),
                    UpExps = Expert(GgmlTensorType.IQ2_XXS, null, Embd, Ff),
                    DownExps = Expert(GgmlTensorType.IQ2_S, null, Ff, Embd),
                },
            },
        };
        Stale(drafted, int.MaxValue, "[dsv4-cuda] dspark.0.ffn_down_exps.weight is IQ2_S, and");

        // Q5_1 keys on its own marker.
        var q51 = Model(Layer(0, GgmlTensorType.Q4_K, GgmlTensorType.Q5_1));
        Dsv4CudaEngine.ValidateRoutedExpertTypes(q51, 0, decodesQ5_1: true, decodesLowBitIq: false);
        Stale(q51, 0, "[dsv4-cuda] blk.0.ffn_down_exps.weight is Q5_1, and", decodesQ5_1: false);
    }

    [Fact]
    public void UnsupportedGpuResidentExpertTypesAreRefusedByTensorName()
    {
        var m = Model(Layer(0, GgmlTensorType.IQ1_M, GgmlTensorType.TQ2_0),
                      Layer(1, GgmlTensorType.IQ2_XXS, GgmlTensorType.Q2_K),
                      Layer(2, GgmlTensorType.IQ1_S, GgmlTensorType.IQ1_M));
        var error = Assert.Throws<NotSupportedException>(() => Validate(m, 0));
        Assert.Contains("cannot run 4 routed-expert tensor(s)", error.Message);
        Assert.Contains("blk.0.ffn_gate_exps.weight is IQ1_M (GPU-resident)", error.Message);
        Assert.Contains("blk.0.ffn_up_exps.weight is IQ1_M (GPU-resident)", error.Message);
        Assert.Contains("blk.0.ffn_down_exps.weight is TQ2_0 (GPU-resident)", error.Message);
        Assert.Contains("blk.2.ffn_down_exps.weight is IQ1_M (GPU-resident)", error.Message);
        Assert.DoesNotContain("blk.1.", error.Message);
        Assert.DoesNotContain("blk.2.ffn_gate_exps", error.Message);
        Assert.Contains("The GPU expert kernels implement Q8_0, Q6_K, Q5_K, Q5_1, Q4_K, Q3_K, Q2_K, Q2_0, IQ4_XS, IQ4_NL, IQ3_S, " +
            "IQ3_XXS, IQ2_S, IQ2_XS, IQ2_XXS, IQ1_S, MXFP4;", error.Message);
        Assert.Contains("Use --backend ggml_cuda", error.Message);
        Assert.True(ModelLoadRefusal.TryDescribe(error, out _));
    }

    [Fact]
    public void SupportedExpertTypesPassOnTheGpuAndTheHost()
    {
        var m = Model(Layer(0, GgmlTensorType.Q4_0, GgmlTensorType.Q5_0),
                      Layer(1, GgmlTensorType.Q2_K, GgmlTensorType.Q3_K),
                      Layer(2, GgmlTensorType.IQ2_XXS, GgmlTensorType.MXFP4),
                      Layer(3, GgmlTensorType.Q8_0, GgmlTensorType.IQ3_S),
                      Layer(4, GgmlTensorType.IQ1_S, GgmlTensorType.IQ2_S),
                      Layer(5, GgmlTensorType.IQ2_XS, GgmlTensorType.IQ3_XXS),
                      Layer(6, GgmlTensorType.IQ4_XS, GgmlTensorType.Q5_1));
        // Layer 0's types have only a host kernel, so they pass offloaded and are refused resident.
        Validate(m, 1);
        var error = Assert.Throws<NotSupportedException>(() => Validate(m, 0));
        Assert.Contains("cannot run 3 routed-expert tensor(s)", error.Message);
        Assert.Contains("blk.0.ffn_gate_exps.weight is Q4_0 (GPU-resident)", error.Message);
        Assert.Contains("blk.0.ffn_down_exps.weight is Q5_0 (GPU-resident)", error.Message);
    }

    /// <summary>An offloaded layer is checked against the host matmul instead: IQ2_XXS has GPU
    /// kernels but no host one, and Q3_K the same.</summary>
    [Fact]
    public void OffloadedLayersAreCheckedAgainstTheHostMatmul()
    {
        var m = Model(Layer(0, GgmlTensorType.IQ2_XXS, GgmlTensorType.Q2_K),
                      Layer(1, GgmlTensorType.Q4_K, GgmlTensorType.Q3_K),
                      Layer(2, GgmlTensorType.IQ2_XXS, GgmlTensorType.Q2_K));
        Validate(m, 0);
        var error = Assert.Throws<NotSupportedException>(() => Validate(m, 2));
        Assert.Contains("blk.0.ffn_gate_exps.weight is IQ2_XXS (offloaded by --n-cpu-moe)", error.Message);
        Assert.Contains("blk.1.ffn_down_exps.weight is Q3_K (offloaded by --n-cpu-moe)", error.Message);
        Assert.DoesNotContain("blk.2.", error.Message);
        // Without a host matmul the offload is refused by the engine's own check, not here.
        m.HostMatMul = null;
        Validate(m, 2);
    }

    [Fact]
    public void GateAndUpMustShareOneType()
    {
        var layer = Layer(0, GgmlTensorType.Q4_K, GgmlTensorType.Q4_K);
        layer.UpExps = Expert(GgmlTensorType.Q5_K, "blk.0.ffn_up_exps.weight", Embd, Ff);
        var error = Assert.Throws<NotSupportedException>(() => Validate(Model(layer), 0));
        Assert.Contains("blk.0.ffn_gate_exps.weight (Q4_K) and blk.0.ffn_up_exps.weight (Q5_K) differ", error.Message);
    }

    [Fact]
    public void DrafterStagesAreAlwaysGpuResident()
    {
        var m = Model(Layer(0, GgmlTensorType.IQ2_XS, GgmlTensorType.IQ2_XS));
        m.Dspark = new Dsv4CudaEngine.DsparkDesc
        {
            Stages = new[] { Layer(0, GgmlTensorType.IQ2_XXS, GgmlTensorType.IQ1_M, "mtp.0.") },
        };
        var error = Assert.Throws<NotSupportedException>(() => Validate(m, int.MaxValue));
        Assert.Contains("cannot run 1 routed-expert tensor(s)", error.Message);
        Assert.Contains("mtp.0.ffn_down_exps.weight is IQ1_M (GPU-resident)", error.Message);
    }

    [Fact]
    public void LongListsNameTheFirstTensorsAndCountTheRest()
    {
        var layers = Enumerable.Range(0, 40).Select(il => Layer(il, GgmlTensorType.IQ1_M, GgmlTensorType.TQ2_0)).ToArray();
        var error = Assert.Throws<NotSupportedException>(() => Validate(Model(layers), 0));
        Assert.Contains("cannot run 120 routed-expert tensor(s)", error.Message);
        Assert.Contains("blk.1.ffn_up_exps.weight is IQ1_M", error.Message);
        Assert.Contains("; and 114 more", error.Message);
        Assert.DoesNotContain("blk.2.", error.Message);
    }

    /// <summary>
    /// What a refusal for want of VRAM advises (the engine's own refusal is in Dsv4CudaEngineLoadTests): the
    /// offload of the leading layers that would make the model fit only when the host matmul runs every
    /// routed expert it would move, <c>--cpu-moe</c> only when it runs every layer's, and otherwise the first
    /// tensor in the way and what can work instead. It used to advise both offloads whatever the layouts.
    /// </summary>
    [Fact]
    public void AVramRefusalAdvisesOnlyOffloadsTheHostRuns()
    {
        var hostable = Model(Layer(0, GgmlTensorType.Q2_K, GgmlTensorType.Q2_K), Layer(1, GgmlTensorType.IQ2_XS, GgmlTensorType.IQ3_XXS));
        string advice = Dsv4CudaEngine.OffloadAdvice(hostable, 1, 3L << 30);
        Assert.StartsWith(". Re-run with --n-cpu-moe 1 (moves the routed experts of the first 1 layer(s), ", advice);
        Assert.EndsWith(" to system RAM) or --cpu-moe to offload every layer.", advice);

        // Layer 2 runs on the GPU only: the leading two layers still offload, every layer does not.
        var partly = Model(Layer(0, GgmlTensorType.Q2_K, GgmlTensorType.Q2_K), Layer(1, GgmlTensorType.IQ2_XS, GgmlTensorType.IQ3_XXS),
                           Layer(2, GgmlTensorType.IQ1_S, GgmlTensorType.IQ2_S));
        advice = Dsv4CudaEngine.OffloadAdvice(partly, 2, 3L << 30);
        Assert.StartsWith(". Re-run with --n-cpu-moe 2 ", advice);
        Assert.EndsWith(" to system RAM).", advice);
        Assert.DoesNotContain("--cpu-moe", advice);

        advice = Dsv4CudaEngine.OffloadAdvice(partly, 3, 3L << 30);
        Assert.StartsWith(". Offloading routed experts cannot make it fit: the first 3 layer(s) would move to system RAM, " +
            "and blk.2.ffn_gate_exps.weight is IQ1_S, which the host matmul behind --n-cpu-moe does not implement.", advice);
        Assert.Contains("--backend ggml_cuda", advice);
        Assert.DoesNotContain("Re-run with", advice);

        // Without a host matmul no offload runs at all (the engine would refuse it with this reason).
        partly.HostMatMul = null;
        advice = Dsv4CudaEngine.OffloadAdvice(partly, 1, 3L << 30);
        Assert.Contains("and no host matmul was supplied to run them", advice);
        Assert.DoesNotContain("Re-run with", advice);
    }

    /// <summary>An explicit --n-cpu-moe is checked before the engine touches CUDA, so this refusal
    /// needs no device.</summary>
    [Fact]
    public void TheEngineRefusesBeforeAnyDeviceWorkWithAnExplicitOffload()
    {
        var m = Model(Layer(0, GgmlTensorType.IQ1_M, GgmlTensorType.Q2_K));
        var error = Assert.Throws<NotSupportedException>(() => new Dsv4CudaEngine(m, 1, 0));
        Assert.Contains("blk.0.ffn_gate_exps.weight is IQ1_M (GPU-resident)", error.Message);
    }

    /// <summary>
    /// The test-only route override: unset, the engine routes as GLM and Qwen3.8 do (per-slot up to 16
    /// tokens, then tensor cores, register-staged or warp-per-row kernels); set, every block takes the
    /// forced route at any ubatch, and a route the module, the layouts or the widths cannot take throws
    /// rather than running another one. No device: the module's two capabilities are arguments.
    /// </summary>
    [Fact]
    public void AForcedExpertRoute_RunsOnlyWhereItCan()
    {
        const int IQ1_S = (int)GgmlTensorType.IQ1_S, IQ2_S = (int)GgmlTensorType.IQ2_S, IQ2_XXS = (int)GgmlTensorType.IQ2_XXS;
        const int Q8_0 = (int)GgmlTensorType.Q8_0;
        Assert.Null(Dsv4CudaEngine.ForceMoeRoute);
        foreach (int nt in new[] { 1, 16, 17, 256 })
            foreach (bool mma in new[] { false, true })
                foreach (bool caps in new[] { false, true })
                    Assert.Equal(CudaMoe.SelectRoute(nt, Dsv4CudaEngine.MaxBatchedDecodeRows, mma, caps, IQ2_XXS, IQ2_S, 5120, 2304),
                        Dsv4CudaEngine.SelectMoeRoute(nt, mma, caps, IQ2_XXS, IQ2_S, 5120, 2304));
        Assert.Equal(CudaMoe.ExpertRoute.TensorCore, Dsv4CudaEngine.SelectMoeRoute(17, true, true, IQ1_S, IQ2_S, 5120, 2304));
        Assert.Equal(CudaMoe.ExpertRoute.Staged, Dsv4CudaEngine.SelectMoeRoute(17, false, true, IQ1_S, IQ2_S, 5120, 2304));
        try
        {
            foreach (CudaMoe.ExpertRoute route in Enum.GetValues<CudaMoe.ExpertRoute>())
            {
                Dsv4CudaEngine.ForceMoeRoute = route;
                foreach (int nt in new[] { 1, 16, 17, 256 })
                    Assert.Equal(route, Dsv4CudaEngine.SelectMoeRoute(nt, true, true, IQ1_S, IQ2_S, 5120, 2304));
            }

            Dsv4CudaEngine.ForceMoeRoute = CudaMoe.ExpertRoute.TensorCore;
            var error = Assert.Throws<InvalidOperationException>(() =>
                Dsv4CudaEngine.SelectMoeRoute(256, false, true, Q8_0, Q8_0, 5120, 2304));
            Assert.Contains("no tensor-core body", error.Message);
            Assert.Contains("do not decode IQ2_XXS gate/up", Assert.Throws<InvalidOperationException>(() =>
                Dsv4CudaEngine.SelectMoeRoute(256, true, false, IQ2_XXS, Q8_0, 5120, 2304)).Message);
            Assert.Contains("not multiples of its k-step", Assert.Throws<InvalidOperationException>(() =>
                Dsv4CudaEngine.SelectMoeRoute(256, true, true, Q8_0, Q8_0, 5120, 2368)).Message);

            Dsv4CudaEngine.ForceMoeRoute = CudaMoe.ExpertRoute.Staged;
            Assert.Contains("do not decode IQ1_S gate/up", Assert.Throws<InvalidOperationException>(() =>
                Dsv4CudaEngine.SelectMoeRoute(256, true, false, IQ1_S, Q8_0, 5120, 2304)).Message);
            Assert.Contains("register-staged kernels do not take widths 5120 and 128", Assert.Throws<InvalidOperationException>(() =>
                Dsv4CudaEngine.SelectMoeRoute(256, true, true, Q8_0, Q8_0, 5120, 128)).Message);

            // Warp-per-row and per-slot decode every layout the engine admits, at any width.
            Dsv4CudaEngine.ForceMoeRoute = CudaMoe.ExpertRoute.Warp;
            Assert.Equal(CudaMoe.ExpertRoute.Warp, Dsv4CudaEngine.SelectMoeRoute(256, false, false, IQ2_XXS, Q8_0, 5120, 128));
            Dsv4CudaEngine.ForceMoeRoute = CudaMoe.ExpertRoute.PerSlot;
            Assert.Equal(CudaMoe.ExpertRoute.PerSlot, Dsv4CudaEngine.SelectMoeRoute(256, false, false, IQ2_XXS, Q8_0, 5120, 128));
        }
        finally
        {
            Dsv4CudaEngine.ForceMoeRoute = null;
        }
    }

    [Fact]
    public void TypeNamesMatchGgml()
    {
        foreach (GgmlTensorType type in Enum.GetValues<GgmlTensorType>().Where(t => (int)t < 100))
            Assert.Equal(type.ToString(), Dsv4CudaEngine.GgmlTypeName((int)type));
        Assert.Equal("type 99", Dsv4CudaEngine.GgmlTypeName(99));
    }
}
