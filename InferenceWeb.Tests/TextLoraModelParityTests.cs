// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Models.Lora;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;
using Xunit;

namespace InferenceWeb.Tests;

/// <summary>
/// A text LoRA applied at run time must be the model with merged weights, <c>W + s * B @ A</c>
/// (PR #282 review items 2 and 8). Tiny synthetic mistral3, hunyuan-dense and Nemotron-H GGUFs
/// carry a rank-4/8 adapter on every projection the family adapts and the LM head, named as the
/// GGUF stores them (attn_q/k/v and ffn_gate/up, which the model fuses at load, so the fused
/// composites are exercised), and a twin GGUF holds the merged weights. Prompt prefill, decode
/// tokens and the engine's batched forward must match the twin; with the adapter deselected the
/// model is the base model bit for bit.
/// </summary>
public sealed class TextLoraModelParityTests : IDisposable
{
    private const int Layers = 2;
    private readonly string _dir = Directory.CreateTempSubdirectory("ts-lora-parity-").FullName;
    private readonly KvCacheDtype _restoreKvDtype = KvCacheDtypeConfig.Current;
    private readonly bool _restoreKvExplicit = KvCacheDtypeConfig.IsExplicitlySet;

    // The adapter and its merged twin are compared through the same K/V cache: F32, pinned. Left to
    // the model-aligned default, each model picks its cache from its own weights, and the two sides
    // of a row differ there: the twin's merged weights are written as F32, so it gets an F32 cache,
    // while a base with F16 layers or projections gets F16. The comparison would then include the
    // cache's rounding, not just the adapter's: an F16-projection Nemotron row measured 2.7e-4
    // against its 1e-4 bound on managed cpu. An F16 cache on both sides is no substitute: two
    // F16-rounded caches whose F32 inputs differ by rounding put the all-F32 rows' LoRA-vs-merged at
    // 1.2e-5..2.5e-5 on direct cuda, against 6e-7..1.1e-6 with an F32 cache and a 1e-5 bound. (Until
    // KvCacheDtypeConfig's default became per-model, a process-wide default that only ever went up
    // decided both from whatever the process had loaded before, so the rows were order-dependent.)
    public TextLoraModelParityTests() => KvCacheDtypeConfig.Set(KvCacheDtype.F32);

    public void Dispose()
    {
        KvCacheDtypeConfig.RestoreForTests(_restoreKvDtype, _restoreKvExplicit);
        try { Directory.Delete(_dir, recursive: true); } catch { /* best effort */ }
    }

    // Managed cpu stores layer 1 as F16 (its q/k/v and gate/up fuse as quantized weights,
    // layer 0's as F32). The GGML lanes keep every matrix F32: their F16 matmul rounds the
    // activations, which would blur the comparison with the F32 twin.
    [Theory]
    [InlineData("mistral3")]
    [InlineData("hunyuan-dense")]
    public void LoraEqualsMergedTwin_Cpu(string arch) => RunParity(arch, BackendType.Cpu, halfLayer: 1, tolerance: 1e-3f);

    [GgmlTheory(BackendType.GgmlCpu)]
    [InlineData("mistral3")]
    [InlineData("hunyuan-dense")]
    public void LoraEqualsMergedTwin_GgmlCpu(string arch) => RunParity(arch, BackendType.GgmlCpu, halfLayer: null, tolerance: 2e-3f);

    [GgmlTheory(BackendType.GgmlCuda)]
    [InlineData("mistral3")]
    [InlineData("hunyuan-dense")]
    public void LoraEqualsMergedTwin_GgmlCuda(string arch) => RunParity(arch, BackendType.GgmlCuda, halfLayer: null, tolerance: 5e-3f);

    // Direct cuda runs the managed forward on device tensors: F32 matrices and the adapter's
    // factors through cuBLAS, F16 ones through the CUDA F16 kernel. With every matrix F32 the
    // adapter matched the twin to 2.0e-6. With layer 1 F16 (fused as quantized weights, as on
    // managed cpu) it measured 5.4e-4, the F16 kernel's own rounding: with no adapter at all,
    // that base and an F32 copy of its weights differ by up to 3.7e-4.
    [CudaTheory]
    [InlineData("mistral3", null, 1e-5f)]
    [InlineData("hunyuan-dense", null, 1e-5f)]
    [InlineData("mistral3", 1, 2e-3f)]
    [InlineData("hunyuan-dense", 1, 2e-3f)]
    public void LoraEqualsMergedTwin_Cuda(string arch, int? halfLayer, float tolerance)
        => RunParity(arch, BackendType.Cuda, halfLayer, tolerance);

    private void RunParity(string arch, BackendType backend, int? halfLayer, float tolerance)
    {
        var fixture = Fixture.Write(_dir, arch, halfLayer);
        AssertMatchesTwin(fixture.BasePath, fixture.TwinPath, fixture.Adapter, backend, tolerance, arch, promptLength: 12,
            "blk.0.attn_qkv.weight", "blk.1.ffn_gate_up.weight");
    }

    /// <summary>
    /// Prefill and six decode tokens with <paramref name="adapter"/> selected equal the merged
    /// twin's logits within <paramref name="tolerance"/>, the adapter moves them well past it, and
    /// the base model is bit-identical before binding, while bound but not selected, and after.
    /// </summary>
    private static void AssertMatchesTwin(string basePath, string twinPath, TextLoraAdapter adapter,
        BackendType backend, float tolerance, string label, int promptLength, params string[] appliedWeights)
    {
        int[] prompt = Enumerable.Range(0, promptLength).Select(i => 65 + (i * 7) % 50).ToArray();

        using var twin = ModelBase.Create(twinPath, backend);
        twin.ResetKVCache();
        var expected = new List<float[]> { (float[])twin.ForwardRefill(prompt).Clone() };
        var tokens = new List<int>();
        for (int step = 0; step < 6; step++)
        {
            tokens.Add(ArgMax(expected[^1]));
            expected.Add((float[])twin.Forward(new[] { tokens[^1] }).Clone());
        }

        using var model = ModelBase.Create(basePath, backend);
        model.ResetKVCache();
        float[] baseBefore = (float[])model.ForwardRefill(prompt).Clone();

        var plan = model.PlanTextLora(adapter);
        Assert.True(plan.IsApplicable, plan.RefusalMessage);
        foreach (string weight in appliedWeights)
            Assert.Contains(weight, plan.AppliedWeights);
        Assert.True(model.TryBindAdapter(adapter, out string? error), error);

        // Bound but not selected: still the base model, bit for bit.
        model.ResetKVCache();
        Assert.Equal(baseBefore, model.ForwardRefill(prompt));

        model.SelectStepAdapter(adapter);
        try
        {
            model.ResetKVCache();
            var actual = new List<float[]> { (float[])model.ForwardRefill(prompt).Clone() };
            foreach (int token in tokens)
                actual.Add((float[])model.Forward(new[] { token }).Clone());

            float moved = MaxAbsDiff(baseBefore, actual[0]);
            Assert.True(moved > 20 * tolerance, $"the adapter must move the logits well past the tolerance (moved {moved})");
            float[] diffs = expected.Select((e, i) => MaxAbsDiff(e, actual[i])).ToArray();
            Assert.True(diffs.Max() <= tolerance,
                $"{label} on {backend}: max |lora - merged| per forward = [{string.Join(", ", diffs)}], adapter moved {moved}");
        }
        finally
        {
            model.SelectStepAdapter(null);
        }

        model.ResetKVCache();
        Assert.Equal(baseBefore, model.ForwardRefill(prompt));
    }

    /// <summary>
    /// The engine path: several adapter requests batched through the family's ForwardBatch
    /// (whose audit runs on every step) produce the merged twin's greedy tokens.
    /// </summary>
    [Theory]
    [InlineData("mistral3")]
    [InlineData("hunyuan-dense")]
    public async Task EngineBatchedAdapterRequests_ProduceTheMergedTwinsTokens(string arch)
    {
        var fixture = Fixture.Write(_dir, arch, halfLayer: 1);
        await AssertEngineMatchesTwin(fixture.BasePath, fixture.TwinPath, fixture.Adapter, BackendType.Cpu);
    }

    /// <summary>The same on direct cuda, within 0.01 of the twin's top logit: the F16 layer
    /// rounds there (by up to 5.4e-4 against the F32 twin) and could break a near tie.</summary>
    [CudaTheory]
    [InlineData("mistral3")]
    [InlineData("hunyuan-dense")]
    public async Task EngineBatchedAdapterRequests_ProduceTheMergedTwinsTokens_Cuda(string arch)
    {
        var fixture = Fixture.Write(_dir, arch, halfLayer: 1);
        await AssertEngineMatchesTwin(fixture.BasePath, fixture.TwinPath, fixture.Adapter, BackendType.Cuda, nearTop: 0.01f);
    }

    /// <summary>
    /// Three adapter requests through <see cref="InferenceEngine"/> produce the merged twin's
    /// greedy tokens: the twin, fed each request's prompt and the engine's tokens before it, has
    /// every engine token as its top pick. With <paramref name="nearTop"/> (where the batched and
    /// single-sequence kernels round differently) an engine token need only be within that
    /// margin of the twin's top logit.
    /// </summary>
    private static async Task AssertEngineMatchesTwin(string basePath, string twinPath, TextLoraAdapter adapter,
        BackendType backend, float? nearTop = null)
    {
        const int maxNewTokens = 8;
        var prompts = new List<int[]>
        {
            Enumerable.Range(0, 19).Select(i => 65 + (i * 7) % 50).ToArray(),
            Enumerable.Range(0, 27).Select(i => 97 + (i * 5) % 26).ToArray(),
            Enumerable.Range(0, 13).Select(i => 48 + (i * 3) % 10).ToArray(),
        };

        int[][] outputs;
        using (var model = ModelBase.Create(basePath, backend))
        {
            var batchedModel = (IBatchedPagedModel)model;
            Assert.True(batchedModel.BatchedForwardAvailable || batchedModel.SupportsPerSequenceFusedForward);
            var cfg = new SchedulerConfig
            {
                MaxNumBatchedTokens = 256, MaxNumRunningSequences = 4, MaxPrefillChunkSize = 8,
                NumBlocks = 32, BlockSize = 16, EnablePrefixCaching = true,
            };
            using var engine = new InferenceEngine(model, cfg, NullLogger.Instance);
            var handles = prompts.Select((p, i) => engine.SubmitRequest(new SequenceState(
                $"lora{i}", p, maxNewTokens, cfg.BlockSize, SamplingConfig.Greedy, adapter: adapter))).ToList();
            outputs = await Task.WhenAll(handles.Select(async h =>
            {
                var tokens = new List<int>();
                await foreach (int t in h.Tokens.ReadAllAsync())
                    tokens.Add(t);
                await h.Completion;
                return tokens.ToArray();
            }));
            Assert.Null(((IAdapterAwareModel)model).StepAdapter);
        }

        using var twin = ModelBase.Create(twinPath, backend);
        for (int i = 0; i < prompts.Count; i++)
        {
            Assert.True(outputs[i].Length > 0, $"request {i} produced nothing");
            twin.ResetKVCache();
            float[] logits = twin.ForwardRefill(prompts[i]);
            for (int j = 0; j < outputs[i].Length; j++)
            {
                int token = outputs[i][j], top = ArgMax(logits);
                Assert.True(nearTop is { } margin ? logits[token] >= logits[top] - margin : token == top,
                    $"request {i}, token {j}: the engine picked {token} ({logits[token]}), the merged twin {top} ({logits[top]})");
                if (j + 1 < outputs[i].Length)
                    logits = twin.Forward(new[] { token });
            }
        }
    }

    // ------------------------------------------------------------------ Nemotron-H

    // Nemotron-H hybrids with F16 projections, so decode tokens take the paths that run a weight
    // outside LinearForward: on GGML, TryLinearAddInto (attn_output, ffn_down, ssm_out,
    // ffn_down_shexp, ffn_latent_out) and the MoE decode residual, which must decline for an
    // adapted weight; LinearForwardInto (ffn_latent_out, and ssm_in into the native Mamba2 decode
    // on ggml_cuda), which must apply the delta itself. "dense" is Mamba2 + attention + dense
    // ReLU^2 (nemotron_h); "moe" and "moe-latent" put a MoE layer with shared experts (and the
    // latent projections) in place of the dense FFN. Routed experts are never adapted.
    //
    // Tolerances: managed cpu matches the twin to ~2e-6. The GGML lanes round activations to F16
    // for the F16 base weights (the twin's merged weights are F32), and the MoE's Q8_0 routed
    // experts re-quantize those slightly different activations; with no adapter at all, the base
    // and an F32 copy of its weights differ by up to 1.1e-3 (dense) and 1.2e-2 (MoE) on
    // ggml_cpu, and LoRA-vs-merged measured the same. On ggml_cuda LoRA-vs-merged measured
    // 5.1e-3 (dense), 1.8e-2 (moe) and 9.2e-3 (moe-latent). The adapter moves the logits by
    // 1.5-2.0, over 20 times every tolerance below.
    [Theory]
    [InlineData("dense")]
    [InlineData("moe-latent")]
    public void Nemotron_LoraEqualsMergedTwin_Cpu(string kind) => RunNemotronParity(kind, BackendType.Cpu, tolerance: 1e-4f);

    [GgmlTheory(BackendType.GgmlCpu)]
    [InlineData("dense", 5e-3f)]
    [InlineData("moe", 3e-2f)]
    [InlineData("moe-latent", 3e-2f)]
    public void Nemotron_LoraEqualsMergedTwin_GgmlCpu(string kind, float tolerance) => RunNemotronParity(kind, BackendType.GgmlCpu, tolerance);

    [GgmlTheory(BackendType.GgmlCuda)]
    [InlineData("dense", 1e-2f)]
    [InlineData("moe", 4e-2f)]
    [InlineData("moe-latent", 4e-2f)]
    public void Nemotron_LoraEqualsMergedTwin_GgmlCuda(string kind, float tolerance) => RunNemotronParity(kind, BackendType.GgmlCuda, tolerance);

    // Direct cuda: with F32 projections LoRA-vs-merged measured 1.8e-6 (dense), 7.7e-7 (moe)
    // and 8.3e-7 (moe-latent). With the F16 projections, which the CUDA F16 kernel rounds, it
    // measured 1.3e-3, 1.1e-2 and 5.9e-3, where with no adapter the base and an F32 copy of its
    // projections differ by up to 9.2e-4, 1.2e-2 and 1.0e-2.
    [CudaTheory]
    [InlineData("dense", "F32", 1e-5f)]
    [InlineData("moe", "F32", 1e-5f)]
    [InlineData("moe-latent", "F32", 1e-5f)]
    [InlineData("dense", "F16", 5e-3f)]
    [InlineData("moe", "F16", 3e-2f)]
    [InlineData("moe-latent", "F16", 3e-2f)]
    public void Nemotron_LoraEqualsMergedTwin_Cuda(string kind, string linearType, float tolerance)
        => RunNemotronParity(kind, BackendType.Cuda, tolerance, Enum.Parse<SyntheticGguf.GgmlType>(linearType));

    /// <summary>
    /// 12 tokens: past the 8 that take the native Mamba2 prefill and the batched-by-expert MoE
    /// prefill on the GGML backends. The delta reaches ssm_in through LinearForward before either
    /// Mamba2 prefill path.
    /// </summary>
    internal const int NemotronPromptLength = 12;

    private void RunNemotronParity(string kind, BackendType backend, float tolerance,
        SyntheticGguf.GgmlType linearType = SyntheticGguf.GgmlType.F16)
    {
        var fixture = NemotronFixture.Write(_dir, kind, linearType);
        var applied = new List<string> { "blk.1.attn_qkv.weight", "blk.0.ssm_in.weight", "blk.3.ssm_out.weight", "output.weight" };
        if (kind == "dense")
            applied.Add("blk.2.ffn_down.weight");
        else
            applied.Add("blk.2.ffn_down_shexp.weight");
        if (kind == "moe-latent")
            applied.Add("blk.2.ffn_latent_out.weight");
        AssertMatchesTwin(fixture.BasePath, fixture.TwinPath, fixture.Adapter, backend, tolerance, "nemotron " + kind,
            NemotronPromptLength, applied.ToArray());
    }

    /// <summary>
    /// The engine path for Nemotron: adapter requests batched through ForwardBatch (Mamba2 slots
    /// per sequence, the per-token MoE) produce the merged twin's greedy tokens; exactly on
    /// managed cpu, and on ggml_cpu, where each sequence's Mamba2 decode token also runs ssm_out
    /// through TryLinearAddInto (which must decline for the adapted weight), within 0.05 of the
    /// twin's top logit (batched and single-sequence kernels round F16 activations differently);
    /// on direct cuda within the same 0.05, where the F16 kernel rounds against the F32 twin.
    /// The managed rows run in every lane, the ggml_cuda one included.
    /// </summary>
    [Theory]
    [InlineData("dense")]
    [InlineData("moe-latent")]
    public Task Nemotron_EngineBatchedAdapterRequests_ProduceTheMergedTwinsTokens_Cpu(string kind)
        => RunNemotronEngine(kind, BackendType.Cpu);

    [GgmlTheory(BackendType.GgmlCpu)]
    [InlineData("dense")]
    [InlineData("moe-latent")]
    public Task Nemotron_EngineBatchedAdapterRequests_ProduceTheMergedTwinsTokens_GgmlCpu(string kind)
        => RunNemotronEngine(kind, BackendType.GgmlCpu);

    [CudaTheory]
    [InlineData("dense")]
    [InlineData("moe-latent")]
    public Task Nemotron_EngineBatchedAdapterRequests_ProduceTheMergedTwinsTokens_Cuda(string kind)
        => RunNemotronEngine(kind, BackendType.Cuda);

    private async Task RunNemotronEngine(string kind, BackendType backend)
    {
        var fixture = NemotronFixture.Write(_dir, kind);
        await AssertEngineMatchesTwin(fixture.BasePath, fixture.TwinPath, fixture.Adapter, backend,
            nearTop: backend == BackendType.Cpu ? null : 0.05f);
    }

    // ------------------------------------------------------------------ fixture

    /// <summary>
    /// A Nemotron-H base GGUF (layout SAFS for "dense", SAMS for the MoE kinds: Mamba2,
    /// attention, FFN, Mamba2), an adapter on every projection Nemotron adapts, named and shaped
    /// as the GGUF stores them (attn_q/k/v fuse into attn_qkv at load; ffn_latent_down/up are the
    /// latent in/out projections), and the merged twin. The dense adapter is a llama.cpp GGUF,
    /// the MoE ones safetensors.
    /// </summary>
    private sealed record NemotronFixture(string BasePath, string TwinPath, TextLoraAdapter Adapter)
    {
        private const float Alpha = 16f;
        private static readonly Regex s_adapted = new(
            @"^(output|blk\.\d+\.(attn_[qkv]|attn_output|ffn_up|ffn_down|ssm_in|ssm_out|ffn_gate_inp|ffn_(up|down)_shexp|ffn_latent_(down|up)))\.weight$",
            RegexOptions.CultureInvariant);

        public static NemotronFixture Write(string dir, string kind, SyntheticGguf.GgmlType linearType = SyntheticGguf.GgmlType.F16)
        {
            NemotronSyntheticModelBuilder Builder(Func<string, float[], float[]> transform) => new()
            {
                Layout = kind == "dense" ? "SAFS" : "SAMS",
                Latent = kind == "moe-latent",
                LinearType = linearType,
                Transform = transform,
            };
            var builder = Builder(null!);
            string basePath = builder.Write(Path.Combine(dir, $"nemotron-{kind}-base.gguf"));

            var targets = new List<(string Target, float[] A, float[] B, int Rank, int In, int Out)>();
            using (var gguf = new GgufFile(basePath))
            {
                foreach (string name in gguf.Tensors.Keys.Where(n => s_adapted.IsMatch(n)).OrderBy(n => n, StringComparer.Ordinal))
                {
                    var shape = gguf.Tensors[name].Shape;
                    int inDim = (int)shape[0], outDim = (int)shape[1];
                    int rank = targets.Count % 2 == 0 ? 4 : 8;      // modules of different ranks
                    var (a, b) = TextLoraFactors.Random(2000 + targets.Count, rank, inDim, outDim, scale: 0.15f);
                    targets.Add((name, a, b, rank, inDim, outDim));
                }
            }

            var byName = targets.ToDictionary(t => t.Target);
            string twinPath = Builder((name, stored) => byName.TryGetValue(name, out var t)
                ? TextLoraFactors.Merge(stored, t.A, t.B, t.Rank, t.In, t.Out, Alpha / t.Rank)
                : null!).Write(Path.Combine(dir, $"nemotron-{kind}-merged.gguf"));
            string adapterPath = kind == "dense"
                ? TextLoraFactors.WriteGguf(Path.Combine(dir, $"nemotron-{kind}-adapter.gguf"), builder.Architecture, Alpha, targets)
                : TextLoraFactors.WriteSafetensors(Path.Combine(dir, $"nemotron-{kind}-adapter.safetensors"), targets, Alpha);
            return new NemotronFixture(basePath, twinPath, TextLoraLoader.Load("nemotron-" + kind, adapterPath));
        }
    }

    private sealed record Fixture(string BasePath, string TwinPath, TextLoraAdapter Adapter)
    {
        private const int Hidden = DenseDecoderSyntheticModelBuilder.Hidden;
        private const int QDim = DenseDecoderSyntheticModelBuilder.NumHeads * DenseDecoderSyntheticModelBuilder.HeadDim;
        private const int KDim = DenseDecoderSyntheticModelBuilder.NumKvHeads * DenseDecoderSyntheticModelBuilder.HeadDim;
        private const int Ff = DenseDecoderSyntheticModelBuilder.FfnLength;
        private const float Alpha = 16f;

        public static Fixture Write(string dir, string arch, int? halfLayer)
        {
            var targets = new List<(string Target, float[] A, float[] B, int Rank, int In, int Out)>();
            int seed = 0;
            void Add(string target, int inDim, int outDim)
            {
                int rank = seed % 2 == 0 ? 4 : 8;      // modules of different ranks
                var (a, b) = TextLoraFactors.Random(1000 + seed++, rank, inDim, outDim, scale: 0.15f);
                targets.Add((target, a, b, rank, inDim, outDim));
            }
            for (int l = 0; l < Layers; l++)
            {
                string p = $"blk.{l}.";
                Add(p + "attn_q.weight", Hidden, QDim);
                Add(p + "attn_k.weight", Hidden, KDim);
                Add(p + "attn_v.weight", Hidden, KDim);
                Add(p + "attn_output.weight", QDim, Hidden);
                Add(p + "ffn_gate.weight", Hidden, Ff);
                Add(p + "ffn_up.weight", Hidden, Ff);
                Add(p + "ffn_down.weight", Ff, Hidden);
            }
            Add("output.weight", Hidden, DenseDecoderSyntheticModelBuilder.ByteTokens + (arch == "mistral3" ? 4 : 0));

            DenseDecoderSyntheticModelBuilder Builder(Func<string, float[], float[]> transform) => arch == "mistral3"
                ? new DenseDecoderSyntheticModelBuilder { Architecture = "mistral3", LayerCount = Layers, HalfPrecisionLayer = halfLayer, Transform = transform }
                : new DenseDecoderSyntheticModelBuilder
                {
                    Architecture = "hunyuan-dense", PreTokenizer = "hunyuan-dense", IncludeMistralControlTokens = false,
                    IncludeQkNorms = true, RopeBase = 10000f, LayerCount = Layers, HalfPrecisionLayer = halfLayer, Transform = transform,
                };

            string basePath = Builder(null!).Write(Path.Combine(dir, arch + "-base.gguf"));
            var byName = targets.ToDictionary(t => t.Target);
            string twinPath = Builder((name, stored) => byName.TryGetValue(name, out var t)
                ? TextLoraFactors.Merge(stored, t.A, t.B, t.Rank, t.In, t.Out, Alpha / t.Rank)
                : null!).Write(Path.Combine(dir, arch + "-merged.gguf"));

            // Both containers end to end: mistral3 loads the llama.cpp GGUF adapter, hunyuan-dense
            // the safetensors one.
            string adapterPath = arch == "mistral3"
                ? TextLoraFactors.WriteGguf(Path.Combine(dir, arch + "-adapter.gguf"), arch, Alpha, targets)
                : TextLoraFactors.WriteSafetensors(Path.Combine(dir, arch + "-adapter.safetensors"), targets, Alpha);
            return new Fixture(basePath, twinPath, TextLoraLoader.Load(arch, adapterPath));
        }
    }

    private static float MaxAbsDiff(float[] a, float[] b)
    {
        Assert.Equal(a.Length, b.Length);
        float max = 0;
        for (int i = 0; i < a.Length; i++) max = Math.Max(max, Math.Abs(a[i] - b[i]));
        return max;
    }

    private static int ArgMax(float[] logits)
    {
        int best = 0;
        for (int v = 1; v < logits.Length; v++) if (logits[v] > logits[best]) best = v;
        return best;
    }
}
