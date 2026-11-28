// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// The managed side of minimax-m3 over a tiny synthetic checkpoint: the metadata
// parse, the tensor validation (indexer on every layer or only the MoE ones, the
// NVFP4 per-expert scales, the F32 routing bias), the cache fingerprint, the host
// contracts an instance without the native executor reports, the queued image
// rows and the pure helpers the executor wiring uses. Native execution is covered
// on a GPU host against llama.cpp (eng/validation/minimax-m3-parity.cpp).
using System.IO;
using System.Text.RegularExpressions;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Models.Architecture;
using TensorSharp.Runtime.Scheduling.PrefixCache;
using static InferenceWeb.Tests.SyntheticGguf;
using GgufTensor = InferenceWeb.Tests.SyntheticGguf.Tensor;

namespace InferenceWeb.Tests;

[Collection(MiniMaxM3EnvironmentCollection.Name)]
public sealed class MiniMaxM3ModelTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-minimax-m3-" + Guid.NewGuid().ToString("N"));
    private readonly KvCacheDtype _restoreDtype = KvCacheDtypeConfig.Current;
    private readonly bool _restoreExplicit = KvCacheDtypeConfig.IsExplicitlySet;
    private readonly string _restoreMsa = Environment.GetEnvironmentVariable(MiniMaxM3Architecture.MsaModeVariable);
    private readonly string _restoreContext = Environment.GetEnvironmentVariable("MAX_CONTEXT");

    public MiniMaxM3ModelTests()
    {
        Directory.CreateDirectory(_dir);
        Environment.SetEnvironmentVariable(MiniMaxM3Architecture.MsaModeVariable, null);
        Environment.SetEnvironmentVariable("MAX_CONTEXT", null);
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, false);
        MoeCpuOffloadConfig.Reset();
    }

    public void Dispose()
    {
        Environment.SetEnvironmentVariable(MiniMaxM3Architecture.MsaModeVariable, _restoreMsa);
        Environment.SetEnvironmentVariable("MAX_CONTEXT", _restoreContext);
        KvCacheDtypeConfig.RestoreForTests(_restoreDtype, _restoreExplicit);
        MoeCpuOffloadConfig.Reset();
        Directory.Delete(_dir, true);
    }

    private string Write(MiniMaxM3SyntheticModelBuilder builder)
        => builder.Write(Path.Combine(_dir, Guid.NewGuid().ToString("N") + ".gguf"));

    /// <summary>Parse and validate without the native executor (BackendType.Cpu only maps the file).</summary>
    private static MiniMaxM3Model Parse(string path)
        => new(path, BackendType.Cpu, 1, null, 1, null, loadExecutor: false);

    private MiniMaxM3Model Parse(MiniMaxM3SyntheticModelBuilder builder) => Parse(Write(builder));

    [Fact]
    public void ParsesTheCheckpointGeometry()
    {
        using var model = Parse(new MiniMaxM3SyntheticModelBuilder());
        Assert.Equal("minimax-m3", model.Config.Architecture);
        Assert.Equal(4, model.Config.NumLayers);
        Assert.Equal(MiniMaxM3SyntheticModelBuilder.Hidden, model.Config.HiddenSize);
        Assert.Equal(MiniMaxM3SyntheticModelBuilder.Heads, model.Config.NumHeads);
        Assert.Equal(MiniMaxM3SyntheticModelBuilder.KvHeads, model.Config.NumKVHeads);
        Assert.Equal(MiniMaxM3SyntheticModelBuilder.HeadDim, model.Config.HeadDim);
        Assert.Equal(MiniMaxM3SyntheticModelBuilder.Experts, model.Config.NumExperts);
        Assert.Equal(MiniMaxM3SyntheticModelBuilder.Used, model.Config.NumExpertsUsed);
        Assert.Equal(MiniMaxM3SyntheticModelBuilder.Vocab, model.Config.VocabSize);
        Assert.Equal(1e7f, model.Config.RopeBase);
        Assert.Equal(4096, model.MaxContextLength);
        // The executor's caches are F16 whatever the process default says.
        Assert.Equal(KvCacheDtype.F16, model.KvCacheDtype);
        Assert.False(model.UsesNativeExecutor);
    }

    [Fact]
    public void FingerprintCarriesTheIndexerGeometryAndItsLayers()
    {
        using var m31 = Parse(new MiniMaxM3SyntheticModelBuilder());
        Assert.Equal("minimax-m3|layers=4|heads=4:2:32|rope=16:10000000:1|msa=2:32:2:16:1:1111|attn=sparse|dtype=f16|exec=none",
            m31.KVStateFingerprint);
        // MiniMax-M3 proper: no indexer on the leading dense layers, so those layers attend densely.
        using var m3 = Parse(new MiniMaxM3SyntheticModelBuilder { IndexerOnDenseLayers = false });
        Assert.Contains("msa=2:32:2:16:1:0011|", m3.KVStateFingerprint, StringComparison.Ordinal);
        Assert.Equal(m31.KVStateFingerprint, m31.GetPrefixCacheCapabilities().NamespaceFingerprint);
        Assert.NotEqual(m31.KVStateFingerprint, m3.KVStateFingerprint);
    }

    [Fact]
    public void ForcedDenseAttentionGetsItsOwnCacheNamespace()
    {
        string path = Write(new MiniMaxM3SyntheticModelBuilder());
        string sparse;
        using (var model = Parse(path)) sparse = model.KVStateFingerprint;
        Environment.SetEnvironmentVariable(MiniMaxM3Architecture.MsaModeVariable, "dense");
        using var dense = Parse(path);
        Assert.Contains("|attn=dense|", dense.KVStateFingerprint, StringComparison.Ordinal);
        Assert.NotEqual(sparse, dense.KVStateFingerprint);
        Environment.SetEnvironmentVariable(MiniMaxM3Architecture.MsaModeVariable, "kernel");
        using var kernel = Parse(path);
        Assert.Equal(sparse, kernel.KVStateFingerprint);
    }

    [Fact]
    public void ExpertScalesAreOptionalAsInLlamaCpp()
    {
        using var model = Parse(new MiniMaxM3SyntheticModelBuilder { ExpertScales = false });
        Assert.Equal("minimax-m3", model.Config.Architecture);
    }

    /// <summary>A broken checkpoint (keyed by the words its refusal must contain) and how to break it.</summary>
    private static readonly Dictionary<string, Action<List<GgufTensor>>> Breakages = new()
    {
        ["layer 2 carries 3 of the 4 indexer tensors"] = MiniMaxM3SyntheticModelBuilder.Drop("blk.2.indexer.k_proj.weight"),
        ["layer 0 carries 3 of the 4 indexer tensors"] = MiniMaxM3SyntheticModelBuilder.Drop("blk.0.indexer.q_norm.weight"),
        ["blk.1.attn_k.weight"] = MiniMaxM3SyntheticModelBuilder.Reshape("blk.1.attn_k.weight", 64, 32),
        ["blk.3.ffn_down_exps.weight"] = MiniMaxM3SyntheticModelBuilder.Drop("blk.3.ffn_down_exps.weight"),
        ["blk.2.ffn_up_exps.scale"] = MiniMaxM3SyntheticModelBuilder.Reshape("blk.2.ffn_up_exps.scale", 2, 2),
        ["blk.3.ffn_gate_shexp.weight"] = MiniMaxM3SyntheticModelBuilder.Drop("blk.3.ffn_gate_shexp.weight"),
        ["blk.0.ffn_down.weight"] = MiniMaxM3SyntheticModelBuilder.Drop("blk.0.ffn_down.weight"),
        ["output.weight"] = MiniMaxM3SyntheticModelBuilder.Drop("output.weight"),
        ["blk.2.exp_probs_b.bias must be F32"] = MiniMaxM3SyntheticModelBuilder.Retype("blk.2.exp_probs_b.bias", GgmlType.F16),
        ["blk.3.ffn_gate_exps.scale must be F32"] = MiniMaxM3SyntheticModelBuilder.Retype("blk.3.ffn_gate_exps.scale", GgmlType.F16),
    };

    public static IEnumerable<object[]> BrokenCheckpoints => Breakages.Keys.Select(k => new object[] { k });

    [Theory]
    [MemberData(nameof(BrokenCheckpoints))]
    public void BrokenTensorsAreRefusedByName(string expected)
    {
        var builder = new MiniMaxM3SyntheticModelBuilder { EditTensors = Breakages[expected] };
        var error = Assert.Throws<InvalidDataException>(() => Parse(builder));
        Assert.Contains(expected, error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void MoeLayersWithoutAnIndexerAreRefused()
    {
        var builder = new MiniMaxM3SyntheticModelBuilder
        {
            EditTensors = list => list.RemoveAll(t => t.Name.StartsWith("blk.3.indexer.", StringComparison.Ordinal)),
        };
        var error = Assert.Throws<InvalidDataException>(() => Parse(builder));
        Assert.Contains("MoE layer 3 has no sparse-attention indexer", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void IndexerHeadsMustMatchTheKvHeads()
    {
        var error = Assert.Throws<InvalidDataException>(() => Parse(new MiniMaxM3SyntheticModelBuilder { IndexerHeads = 4 }));
        Assert.Contains("4 index heads", error.Message, StringComparison.Ordinal);
        Assert.Contains("2 KV heads", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void SoftmaxRoutingIsRefused()
    {
        var error = Assert.Throws<NotSupportedException>(() => Parse(new MiniMaxM3SyntheticModelBuilder { GatingFunc = 1 }));
        Assert.Contains("sigmoid", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void InvalidIndexerBlockIsRefused()
    {
        var builder = new MiniMaxM3SyntheticModelBuilder
        {
            EditMetadata = MiniMaxM3SyntheticModelBuilder.SetU32("minimax-m3.attention.indexer.block_size", 0),
        };
        var error = Assert.Throws<InvalidDataException>(() => Parse(builder));
        Assert.Contains("block 0", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void CreateRefusesTheManagedCpuBackendBeforeConstruction()
    {
        string path = Write(new MiniMaxM3SyntheticModelBuilder());
        var error = Assert.Throws<NotSupportedException>(() => ModelBase.Create(path, BackendType.Cpu));
        Assert.Equal(MiniMaxM3Architecture.ManagedCpuRefusal, error.Message);
    }

    [Fact]
    public void AnInstanceWithoutTheExecutorRefusesToForward()
    {
        using var model = Parse(new MiniMaxM3SyntheticModelBuilder());
        Assert.Throws<InvalidOperationException>(() => model.Forward(new[] { 1, 2, 3 }));
        Assert.Throws<InvalidOperationException>(() => model.SpecForward(new[] { 1 }, null, new float[model.Config.VocabSize], false));
        Assert.False(model.SupportsPerSequenceFusedForward);
        Assert.False(model.BatchedForwardAvailable);
        Assert.False(model.SpeculationProfitable);
        Assert.False(model.SpecTrunkFollowsBoundCache);
        Assert.True(model.SpecVerifyPersistsAcceptedKv);
        Assert.Throws<NotSupportedException>(() => model.ForwardBatch(null));
        Assert.Equal(AudioInputSupport.MiniMaxM3Message, AudioInputSupport.UnsupportedReasonFor(model));
    }

    [Fact]
    public void PrefixCacheContractRewindsAnywhereAndNeverShifts()
    {
        using var model = Parse(new MiniMaxM3SyntheticModelBuilder());
        var caps = model.GetPrefixCacheCapabilities();
        Assert.Equal(TruncationKind.Any, caps.Truncation);
        Assert.Equal(1, caps.TruncationGranularity);
        Assert.Equal(int.MaxValue, caps.RewindCapTokens);
        Assert.Equal(PageSupport.None, caps.Pages);
        Assert.False(caps.Persistable);
        // Without the executor there are no slots to retain or copy.
        Assert.Equal(EndStateSupport.None, caps.EndState);
        Assert.False(caps.CanCaptureCopy);
        Assert.False(model.TryCaptureCopy("r", "k", out _));
        Assert.True(model.SupportsKVCacheTruncation);
        Assert.True(model.CanTruncateKVCache(100, 37));
    }

    [Fact]
    public void ImageRowsQueueAtAbsolutePositionsAndAResetDropsThem()
    {
        using var model = Parse(new MiniMaxM3SyntheticModelBuilder());
        int hidden = model.Config.HiddenSize;
        Assert.Throws<ArgumentException>(() => model.QueueNativeVisionRows(new float[hidden * 3 - 1], 3, 5));
        Assert.Throws<ArgumentOutOfRangeException>(() => model.QueueNativeVisionRows(new float[hidden], 0, 5));
        Assert.Throws<ArgumentOutOfRangeException>(() => model.QueueNativeVisionRows(new float[hidden], 1, -1));
        Assert.False(model.HasPendingVisionRows);
        Assert.Null(model.BatchedFusedDecodeDeclineReason);

        model.QueueNativeVisionRows(new float[hidden * 3], 3, 5);
        Assert.True(model.HasPendingVisionRows);
        // The batched paths step aside, and say why.
        Assert.Contains("image rows", model.BatchedFusedDecodeDeclineReason, StringComparison.Ordinal);
        model.ResetKVCache();
        Assert.False(model.HasPendingVisionRows);

        model.QueueNativeVisionRows(new float[hidden], 1, 0);
        Assert.True(model.TryTruncateKVCache(0));
        Assert.False(model.HasPendingVisionRows);
    }

    [Fact]
    public void GpuCountFollowsTheLayerSplitAndRefusesWhatItCannotHonour()
    {
        Assert.Equal(1, MiniMaxM3Model.ResolveNativeGpuCount(1, null, 8, "CUDA"));
        Assert.Equal(8, MiniMaxM3Model.ResolveNativeGpuCount(8, null, 8, "CUDA"));
        Assert.Equal(3, MiniMaxM3Model.ResolveNativeGpuCount(1, "20,20,20", 8, "CUDA"));
        Assert.Equal(1, MiniMaxM3Model.ResolveNativeGpuCount(1, null, null, "CPU"));
        var short_ = Assert.Throws<TensorSharp.Runtime.ModelLoadRefusedException>(() =>
            MiniMaxM3Model.ResolveNativeGpuCount(8, null, 4, "CUDA"));
        Assert.Contains("sees only 4", short_.Message, StringComparison.Ordinal);
        Assert.Throws<TensorSharp.Runtime.ModelLoadRefusedException>(() =>
            MiniMaxM3Model.ResolveNativeGpuCount(17, null, 32, "CUDA"));
    }

    [Fact]
    public void CpuMoeLayerCountMatchesLlamaCppAndNamesADenseOnlyCount()
    {
        Assert.Equal(0, MiniMaxM3Model.ResolveCpuMoeLayers(60, 3, null));
        MoeCpuOffloadConfig.SetLayers(20);
        Assert.Equal(20, MiniMaxM3Model.ResolveCpuMoeLayers(60, 3, null));
        Assert.Equal(-1, MiniMaxM3Model.ResolveCpuMoeLayers(60, 3, "auto"));
        MoeCpuOffloadConfig.SetAllLayers();
        Assert.Equal(60, MiniMaxM3Model.ResolveCpuMoeLayers(60, 3, null));

        MoeCpuOffloadConfig.Reset();
        MoeCpuOffloadConfig.SetLayers(3);
        var stderr = Console.Error;
        var captured = new StringWriter();
        Console.SetError(captured);
        try { Assert.Equal(3, MiniMaxM3Model.ResolveCpuMoeLayers(60, 3, null)); }
        finally { Console.SetError(stderr); }
        Assert.Contains("--n-cpu-moe 3 moves no experts", captured.ToString(), StringComparison.Ordinal);
    }

    [Fact]
    public void SoloPrefillStepsJoinAShortRemainder()
    {
        int Plan(long p0, long n) => 512;
        Assert.Equal(8192, MiniMaxM3Model.SoloPrefillStep(0, 20000, Plan));
        Assert.Equal(10000, MiniMaxM3Model.SoloPrefillStep(0, 10000, Plan));
        Assert.Equal(0, MiniMaxM3Model.SoloPrefillStep(0, 20000, (_, _) => 0));
        Assert.Equal(0, MiniMaxM3Model.SoloPrefillStep(0, 0, Plan));
    }

    [Fact]
    public void ShouldPrefaultSkipsOnlyOffloadedExperts()
    {
        // The policy is the model's load-time snapshot (ModelBase.MoeOffload): set before the load, it holds for the model;
        // a later change of the process default reaches the next load only.
        MoeCpuOffloadConfig.SetLayers(3);
        using var model = new PrefaultProbe(Write(new MiniMaxM3SyntheticModelBuilder()));
        MoeCpuOffloadConfig.SetLayers(5);
        Assert.False(model.Prefaults("blk.2.ffn_up_exps.weight"));
        Assert.True(model.Prefaults("blk.3.ffn_up_exps.weight"));
        Assert.True(model.Prefaults("blk.2.ffn_up_shexp.weight"));
        Assert.True(model.Prefaults("blk.0.attn_q.weight"));
        Assert.Equal(3, model.MoeOffload.CpuMoeLayers);
    }

    /// <summary>Both directions between the options the native MiniMax-M3 sources read (env_long, env_flag,
    /// opt_get or getenv of a literal name) and what the managed side forwards: a name missing from
    /// NativeOptionNames, set through Environment.SetEnvironmentVariable (tests, launchers, in-process
    /// hosts), is silently ignored on Linux, whose native getenv never sees it; a forwarded name nothing
    /// reads is a stale spelling.</summary>
    [Fact]
    public void NativeOptionNamesMatchWhatTheNativeSourcesRead()
    {
        string native = Path.Combine(LoraHostFixtures.RepoRoot(), "TensorSharp.GGML.Native");
        var files = Directory.EnumerateFiles(native, "ggml_ops_minimax_m3*")
            .Concat(Directory.EnumerateFiles(native, "ggml_ops_minimax_msa*")).ToArray();
        Assert.Contains(files, f => Path.GetFileName(f) == "ggml_ops_minimax_m3_exec.cpp");
        Assert.Contains(files, f => Path.GetFileName(f) == "ggml_ops_minimax_m3_dspark.inc");
        var read = new SortedSet<string>(StringComparer.Ordinal);
        foreach (string file in files)
            foreach (Match m in Regex.Matches(File.ReadAllText(file), @"\b(?:env_long|env_flag|opt_get|getenv)\(""([A-Z][A-Z0-9_]*)"""))
                read.Add(m.Groups[1].Value);
        Assert.Contains(MiniMaxM3Architecture.MsaModeVariable, read);
        Assert.Contains("TS_MINIMAX_M3_DSPARK_GRAPHS", read);

        var forwarded = new HashSet<string>(MiniMaxM3Model.NativeOptionNames, StringComparer.Ordinal);
        Assert.Equal(MiniMaxM3Model.NativeOptionNames.Length, forwarded.Count);   // no name listed twice
        Assert.Empty(read.Where(n => !forwarded.Contains(n) && !MiniMaxM3Model.NativeLoadArgumentNames.Contains(n)));
        Assert.Empty(MiniMaxM3Model.NativeOptionNames.Where(n => !read.Contains(n)));
        Assert.Empty(MiniMaxM3Model.NativeLoadArgumentNames.Where(n => !read.Contains(n) || forwarded.Contains(n)));
    }

    private sealed class PrefaultProbe : MiniMaxM3Model
    {
        public PrefaultProbe(string path) : base(path, BackendType.Cpu, 1, null, 1, null, loadExecutor: false) { }
        public bool Prefaults(string name) => ShouldPrefaultWeight(new TensorSharp.Runtime.GgufTensorInfo { Name = name });
    }
}
