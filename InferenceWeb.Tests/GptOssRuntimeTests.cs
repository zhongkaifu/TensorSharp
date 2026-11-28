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
using System.Threading.Tasks;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// GPT-OSS on the shared decoder runtime (<see cref="GptOssRuntimeModel"/>) over the synthetic fixtures
/// (<see cref="GptOssSyntheticModelBuilder"/>: alternating sliding-window and full attention with sinks and biases, YaRN rotary,
/// MXFP4 or Q8_0 routed experts with biases and the clamped SwiGLU): the legacy model's logits, the pinned GGML backend and the
/// direct CUDA executor against the managed one, and every mechanism the runtime claims for the family (placements, batched
/// slots, continuous batching, host-served experts, checkpoints, speculation).
/// </summary>
[Collection("MoeCpuOffloadConfig")]   // the host-experts test sets the process-wide expert-offload policy
public sealed class GptOssRuntimeTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-gptoss-runtime-" + Guid.NewGuid().ToString("N"));
    private readonly ITestOutputHelper _output;

    public GptOssRuntimeTests(ITestOutputHelper output)
    {
        _output = output;
        Directory.CreateDirectory(_dir);
    }

    public void Dispose()
    {
        MoeCpuOffloadConfig.Reset();
        try { Directory.Delete(_dir, recursive: true); } catch { }
    }

    private string Fixture(string name)
    {
        string path = Path.Combine(_dir, name + ".gguf");
        return File.Exists(path) ? path : GptOssLegacySynthetic.Get(name).Write(path);
    }

    private static GptOssRuntimeModel Create(string path, BackendType backend = BackendType.Cpu, int tp = 1, int stages = 1,
        string draft = null)
    {
        ITensorParallelGroup group = tp > 1 ? new SimulatedDeviceGroup(tp * stages) : null;
        return new GptOssRuntimeModel(path, backend, tp, group, stages, draft);
    }

    /// <summary>How far two F32 runs of these fixtures' random layers part when only their summation order differs; a wrong
    /// recipe part (a sink, a bias, the window, the clamp, a route) moves these logits by 0.1 to 1.</summary>
    private const double NoiseTolerance = 0.025;

    /// <summary>A q8_0 cache's rows from the f16 cache's on the sink-sensitive MXFP4 fixture: up to 0.118 on ggml_cpu and 0.113
    /// on ggml_cuda (A40), with 1.5x headroom. The fixture amplifies the cache's rounding; on gpt-oss-20b a q8_0 cache moved the
    /// mean NLL by -0.0004 on ggml_cuda (384 tokens) and +0.008 on ggml_cpu (160 tokens).</summary>
    private const double QuantizedKvTolerance = 0.18;

    // ---------------------------------------------------------------- the forward

    [Fact]
    public void TheFactoryCreatesTheRuntime()
    {
        using ModelBase model = ModelBase.Create(Fixture("mxfp4"), BackendType.Cpu);
        var runtime = Assert.IsType<GptOssRuntimeModel>(model);
        Assert.Equal(4, model.Config.NumLayers);
        Assert.True(runtime.IsSlidingWindowLayerForTests(0));
        Assert.False(runtime.IsSlidingWindowLayerForTests(1));
    }

    /// <summary>The legacy model's logits, frozen before it was deleted (<see cref="GptOssLegacySynthetic"/>): the same math on the
    /// managed backend, every case's prompt row and decode rows (the same top token wherever the frozen top two are half a logit
    /// apart). Before the freeze the two models ran side by side here: relL2 1.8e-3 to 1.5e-2 on every row.</summary>
    [Theory]
    [InlineData("mxfp4")]
    [InlineData("q8-experts")]
    [InlineData("long")]
    [InlineData("tp")]
    public void MatchesTheFrozenLegacyLogits(string name)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture(name);
        List<float[]> frozen = GptOssLegacySynthetic.Rows(name, path);
        using var model = Create(path);
        List<float[]> actual = GptOssLegacySynthetic.Run(model, GptOssLegacySynthetic.Get(name));
        Assert.Equal(frozen.Count, actual.Count);
        for (int i = 0; i < frozen.Count; i++)
        {
            double rel = RelL2(actual[i], frozen[i]);
            _output.WriteLine($"{name} row {i}: relL2 {rel:E3}");
            Assert.True(rel <= NoiseTolerance, $"{name} row {i}: relL2 {rel:E3} against the legacy model");
            if (Margin(frozen[i]) > 0.5f) Assert.Equal(Argmax(frozen[i]), Argmax(actual[i]));
        }
    }

    /// <summary>The pinned GGML backend's rows are the managed backend's (ggml_cpu's F32 products round apart; ggml_cuda's
    /// quantize the activations of its quantized products: on the MXFP4 fixture its rows part from the managed ones by up to 0.22,
    /// and the legacy model's ggml_cuda rows were the runtime's bit for bit on every row of every fixture, A40).</summary>
    [Theory]
    [InlineData("mxfp4")]
    [InlineData("q8-experts")]
    [InlineData("long")]
    public void PinnedGgmlBackendMatchesTheManagedBackend(string name)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture(name);
        GptOssLegacySynthetic.Case c = GptOssLegacySynthetic.Get(name);
        List<float[]> managed, ggml;
        using (var model = Create(path)) managed = GptOssLegacySynthetic.Run(model, c);
        using (var model = Create(path, TestGates.PinnedGgmlBackend)) ggml = GptOssLegacySynthetic.Run(model, c);
        double tolerance = TestGates.PinnedGgmlBackend == BackendType.GgmlCpu ? NoiseTolerance
            : name is "mxfp4" or "long" ? 10 * NoiseTolerance : 2 * NoiseTolerance;
        for (int i = 0; i < managed.Count; i++)
        {
            double rel = RelL2(ggml[i], managed[i]);
            _output.WriteLine($"{name} {TestGates.PinnedGgmlBackend} row {i}: relL2 {rel:E3}");
            Assert.True(rel <= tolerance, $"{name} {TestGates.PinnedGgmlBackend} row {i}: relL2 {rel:E3} > {tolerance:E1}");
        }
    }

    /// <summary>The direct CUDA executor against the managed backend, and its runs reproduce bit for bit.</summary>
    [CudaTheory]
    [InlineData("mxfp4")]
    [InlineData("q8-experts")]
    [InlineData("long")]
    public void DirectCudaMatchesTheManagedBackend(string name)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture(name);
        GptOssLegacySynthetic.Case c = GptOssLegacySynthetic.Get(name);
        List<float[]> managed, first, again;
        using (var model = Create(path)) managed = GptOssLegacySynthetic.Run(model, c);
        using (var model = Create(path, BackendType.Cuda))
        {
            first = GptOssLegacySynthetic.Run(model, c);
            again = GptOssLegacySynthetic.Run(model, c);
        }
        for (int i = 0; i < managed.Count; i++)
        {
            double rel = RelL2(first[i], managed[i]);
            _output.WriteLine($"{name} cuda row {i}: relL2 {rel:E3}");
            Assert.True(rel <= 2 * NoiseTolerance, $"{name} cuda row {i}: relL2 {rel:E3}");
            Assert.Equal(first[i], again[i]);
        }
    }

    /// <summary>A prompt in one chunk and in pieces (the sliding-window rings carried across them) gives the same logits.</summary>
    [Theory]
    [InlineData("mxfp4", "managed")]
    [InlineData("mxfp4", "ggml")]
    [InlineData("long", "managed")]
    [InlineData("long", "ggml")]
    public void ChunkedPrefillMatchesOneChunk(string name, string executor)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        BackendType backend = executor == "ggml" ? TestGates.PinnedGgmlBackend : BackendType.Cpu;
        using var model = Create(Fixture(name), backend);
        int[] prompt = GptOssLegacySynthetic.Get(name).Prompt;
        float[] whole = (float[])model.ForwardRefill(prompt).Clone();
        model.ResetKVCache();
        float[] pieces = null;
        int cut1 = prompt.Length / 3, cut2 = cut1 + 1;
        foreach (int[] chunk in new[] { prompt[..cut1], prompt[cut1..cut2], prompt[cut2..] })
            pieces = model.Forward(chunk);
        // ggml-cpu's quantized products round apart by the rows of a chunk: the legacy model's chunks of the "long" fixture parted
        // from its whole prompt by the same 5.8e-3 (its first chunk of 27 rows and of 40 alike; 0 with a first chunk of 10).
        double tolerance = backend is BackendType.Cpu ? 1e-5 : backend == BackendType.GgmlCpu ? 1e-2 : 3e-2;
        Assert.True(RelL2(pieces, whole) <= tolerance, $"{name} {backend} chunked: relL2 {RelL2(pieces, whole):E3}");
    }

    // ---------------------------------------------------------------- mechanisms

    public static TheoryData<string, int, int> Placements() => new()
    {
        { "tp", 2, 1 },
        { "mxfp4", 1, 2 },
        { "tp", 1, 2 },
        { "long", 1, 3 },
        { "tp", 2, 2 },
    };

    /// <summary>Tensor parallelism (per-rank graphs over simulated devices: each rank owns its heads, their biases and sinks,
    /// and a slice of every expert's width with its biases), whole-layer stages and their hybrid (every stage's layers as
    /// per-rank graphs over its own group) answer as one device.</summary>
    [Theory]
    [MemberData(nameof(Placements))]
    public void PlacementMatchesOneDevice(string name, int tp, int stages)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture(name);
        GptOssLegacySynthetic.Case c = GptOssLegacySynthetic.Get(name);
        List<float[]> reference;
        using (var one = Create(path)) reference = Script(one, c);
        using var placed = Create(path, BackendType.Cpu, tp, stages);
        Assert.Equal(tp, placed.Mesh.TensorParallel);
        Assert.Equal(stages, placed.Mesh.Stages);
        List<float[]> actual = Script(placed, c);
        Assert.True(placed.GraphRuns > 0, $"{name} tp={tp} stages={stages}: no graph ran");
        double tolerance = tp == 1 ? 1e-6 : NoiseTolerance;
        for (int i = 0; i < reference.Count; i++)
        {
            double rel = RelL2(actual[i], reference[i]);
            _output.WriteLine($"{name} tp={tp} stages={stages} row {i}: relL2 {rel:E3}");
            Assert.True(rel <= tolerance, $"{name} tp={tp} stages={stages} row {i}: relL2 {rel:E3} > {tolerance:E2}");
        }
    }

    [Theory]
    [InlineData("mxfp4")]
    [InlineData("long")]
    public void SlotBatchesMatchEachSequenceAlone(string name)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture(name);
        SlotParity.AssertBatchedSlotsMatchAlone(() => Create(path), 1e-5, $"gpt-oss {name}");
    }

    /// <summary>The engine's greedy streams of three requests served together (batched steps over their slots, prefill
    /// chunks, decode quanta, the prefix cache) equal each request's alone.</summary>
    [Theory]
    [InlineData("mxfp4")]
    [InlineData("long")]
    public async Task ConcurrentRequestsAreBatched(string name)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture(name);
        const int maxNewTokens = 8;
        var prompts = new List<int[]>
        {
            Enumerable.Range(0, 19).Select(i => 65 + (i * 7) % 50).ToArray(),
            Enumerable.Range(0, 41).Select(i => 97 + (i * 5) % 26).ToArray(),
            Enumerable.Range(0, 13).Select(i => 48 + (i * 3) % 10).ToArray(),
        };
        var expected = new List<int[]>();
        using (var reference = Create(path))
        {
            foreach (int[] prompt in prompts)
            {
                reference.ResetKVCache();
                float[] logits = reference.ForwardRefill(prompt);
                var produced = new List<int>();
                for (int i = 0; i < maxNewTokens; i++)
                {
                    int token = Argmax(logits);
                    if (reference.Tokenizer.EosTokenIds.Contains(token)) break;
                    produced.Add(token);
                    if (i + 1 < maxNewTokens) logits = reference.Forward(new[] { token });
                }
                expected.Add(produced.ToArray());
            }
        }

        using var model = Create(path);
        var cfg = new SchedulerConfig
        {
            MaxNumBatchedTokens = 256,
            MaxNumRunningSequences = 4,
            MaxPrefillChunkSize = 8,
            NumBlocks = 32,
            BlockSize = 16,
            EnablePrefixCaching = true,
            DecodeQuantumTokens = 2,
        };
        using var engine = new InferenceEngine(model, cfg, NullLogger.Instance);
        var handles = prompts.Select((p, i) => engine.SubmitRequest(
            new SequenceState($"req{i}", p, maxNewTokens, cfg.BlockSize, SamplingConfig.Greedy))).ToList();
        var outputs = await Task.WhenAll(handles.Select(async h =>
        {
            var tokens = new List<int>();
            await foreach (int t in h.Tokens.ReadAllAsync())
                tokens.Add(t);
            await h.Completion;
            return tokens.ToArray();
        })).WaitAsync(TimeSpan.FromMinutes(3));

        Assert.True(model.BatchedForwardCount > 0, "the engine must serve these requests in batched steps over their slots");
        for (int i = 0; i < prompts.Count; i++)
            Assert.Equal(expected[i], outputs[i]);
    }

    [Theory]
    [InlineData("mxfp4", "managed")]
    [InlineData("mxfp4", "ggml")]
    [InlineData("long", "managed")]
    public void CheckpointsContinueExactly(string name, string executor)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        BackendType backend = executor == "ggml" ? TestGates.PinnedGgmlBackend : BackendType.Cpu;
        string path = Fixture(name);
        CheckpointChecks.AssertCheckpointsContinueExactly(() => Create(path, backend), backend == BackendType.GgmlCuda ? 1e-2 : 1e-5,
            $"gpt-oss {name} {backend}", deviceCopies: executor == "ggml");
    }

    /// <summary>
    /// GPT-OSS's harmony template drops the analysis channel when it re-renders a past answer, so a conversation's next turn
    /// parts from its retained cache right after the previous prompt: continuing it rewinds the whole previous answer, well
    /// past a sliding window. The rings keep every position the next rows' windows read while the cache holds fewer positions
    /// than a ring's cells, so the rewind is exact: a cache rewound past a long answer continues as one that never held it.
    /// </summary>
    [Theory]
    [InlineData("long", "managed")]
    [InlineData("long", "ggml")]
    [InlineData("mxfp4", "ggml")]
    public void ARewindPastALongAnswerContinuesAsACacheThatNeverHeldIt(string name, string executor)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        BackendType backend = executor == "ggml" ? TestGates.PinnedGgmlBackend : BackendType.Cpu;
        string path = Fixture(name);
        int[] prompt = GptOssLegacySynthetic.Tokens(40, 3), answer = GptOssLegacySynthetic.Tokens(56, 11);
        int[] nextTurn = GptOssLegacySynthetic.Tokens(10, 29);
        List<float[]> Continue(GptOssRuntimeModel model)
        {
            var rows = new List<float[]>();
            foreach (int t in nextTurn) rows.Add((float[])model.Forward(new[] { t }).Clone());
            return rows;
        }
        List<float[]> expected;
        using (var fresh = Create(path, backend))
        {
            fresh.ResetKVCache();
            fresh.ForwardRefill(prompt);
            expected = Continue(fresh);
        }
        using var rewound = Create(path, backend);
        rewound.ResetKVCache();
        rewound.ForwardRefill(prompt);
        foreach (int t in answer.Take(8)) rewound.Forward(new[] { t });      // decode rows, then a chunk: both rewound
        rewound.Forward(answer.Skip(8).ToArray());
        Assert.True(rewound.CanTruncateKVCache(prompt.Length + answer.Length, prompt.Length), $"{name} {backend}: the rewind was refused");
        Assert.True(rewound.TryTruncateKVCache(prompt.Length), $"{name} {backend}: the rewind failed");
        List<float[]> actual = Continue(rewound);
        for (int i = 0; i < expected.Count; i++)
        {
            double rel = RelL2(actual[i], expected[i]);
            _output.WriteLine($"{name} {backend} step {i}: relL2 {rel:E3}");
            Assert.True(rel <= 1e-6, $"{name} {backend} step {i}: relL2 {rel:E3}");
        }
    }

    /// <summary>
    /// unsloth's gpt-oss requants quantize Q, K and V with different types (Q2_K: IQ4_NL q and k, Q5_0 v; Q4_K_M: a Q8_0 v on
    /// half the layers). The legacy model decided fusion for the whole model: it once fused the uniform half of the layers
    /// under a model-wide "separate" name table (the first forward aborted), and its unconditional bias fusion orphaned every
    /// layer's q, k and v biases (garbage on CUDA). The runtime fuses layer by layer and the biases with their weights: a
    /// uniform layer reads attn_qkv and its [q | k | v] bias, a mixed one its three projections and their own biases, and the
    /// model answers as the managed backend's.
    /// </summary>
    [Fact]
    public void MixedProjectionTypesFuseLayerByLayerWithTheirBiases()
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Path.Combine(Path.GetTempPath(), $"ts-gptoss-mixedqkv-{Guid.NewGuid():N}.gguf");
        new GptOssSyntheticModelBuilder { SinkCenter = 2f, SinkSpread = 1f, F32Values = l => l % 2 == 1 }.Write(path);
        try
        {
            GptOssLegacySynthetic.Case c = GptOssLegacySynthetic.Get("mxfp4");
            List<float[]> managed;
            using (var model = Create(path, BackendType.Cpu)) managed = Script(model, c);
            using var ggml = Create(path, TestGates.PinnedGgmlBackend);
            for (int l = 0; l < ggml.Config.NumLayers; l++)
            {
                bool fused = l % 2 == 0;
                Assert.Equal(fused, ggml.HasWeightForTests($"blk.{l}.attn_qkv.weight"));
                Assert.Equal(fused, ggml.HasWeightForTests($"blk.{l}.attn_qkv.bias"));
                foreach (string bias in new[] { "attn_q.bias", "attn_k.bias", "attn_v.bias" })
                    Assert.Equal(!fused, ggml.HasWeightForTests($"blk.{l}.{bias}"));
                Assert.True(ggml.HasWeightForTests($"blk.{l}.attn_output.bias"));
            }
            List<float[]> actual = Script(ggml, c);
            // ggml-cpu and the managed backend quantize this fixture's activations each its own way, and how far apart their
            // rows land depends on the host's kernels: 1.3e-2 at most on x64 (AVX-512 and AVX2), 2.7e-2 at its last decode row on
            // the ARM64 runner and with the managed portable 128-bit kernels (DOTNET_EnableAVX2=0), so a CPU run takes twice the
            // summation-order bound (held to it, this failed on every ARM64 run); a misfused layer or orphaned bias is garbage.
            double tolerance = TestGates.PinnedGgmlBackend switch
            {
                BackendType.GgmlCuda => 10 * NoiseTolerance,
                BackendType.GgmlCpu => 2 * NoiseTolerance,
                _ => NoiseTolerance,
            };
            for (int i = 0; i < managed.Count; i++)
            {
                double rel = RelL2(actual[i], managed[i]);
                _output.WriteLine($"mixed q/k/v {TestGates.PinnedGgmlBackend} row {i}: relL2 {rel:E3}");
                Assert.True(rel <= tolerance, $"mixed q/k/v {TestGates.PinnedGgmlBackend} row {i}: relL2 {rel:E3}");
            }
        }
        finally
        {
            File.Delete(path);
        }
    }

    /// <summary>
    /// A block-quantized K/V request: q4_0 becomes f16 everywhere (gpt-oss-20b lost 0.48 nat a token over a q4_0 cache), q8_0
    /// stays on the GGML backends, whose graphs run the rings over it, and becomes f16 on the tensor executors. The q8_0 cache
    /// answers as the f16 one within its rounding.
    /// </summary>
    [Theory]
    [InlineData("managed")]
    [InlineData("ggml")]
    public void BlockQuantizedKvRequestsKeepWhatTheFamilyReads(string executor)
    {
        BackendType backend = executor == "ggml" ? TestGates.PinnedGgmlBackend : BackendType.Cpu;
        string path = Fixture("mxfp4");
        GptOssLegacySynthetic.Case c = GptOssLegacySynthetic.Get("mxfp4");
        try
        {
            KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.Q4_0, true);
            using (var q4 = Create(path, backend))
                Assert.Equal(KvCacheDtype.F16, q4.KvCacheDtype);
            KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F16, true);
            List<float[]> f16;
            using (var model = Create(path, backend)) f16 = Script(model, c);
            KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.Q8_0, true);
            using var q8 = Create(path, backend);
            Assert.Equal(executor == "ggml" ? KvCacheDtype.Q8_0 : KvCacheDtype.F16, q8.KvCacheDtype);
            List<float[]> actual = Script(q8, c);
            var rels = actual.Select((row, i) => RelL2(row, f16[i])).ToList();
            _output.WriteLine($"{backend} {q8.KvCacheDtype} rows from f16: {string.Join(" ", rels.Select(r => r.ToString("E2")))}");
            for (int i = 0; i < f16.Count; i++)
                Assert.True(rels[i] <= QuantizedKvTolerance, $"{backend} {q8.KvCacheDtype} row {i}: relL2 {rels[i]:E3} from f16");
        }
        finally
        {
            KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        }
    }

    /// <summary>Greedy speculation (scripted drafts, accepted and rejected) emits exactly the plain greedy tokens: GPT-OSS now
    /// has a speculative trunk (n-gram drafts, DFlash drafters).</summary>
    [Theory]
    [InlineData("mxfp4", "managed")]
    [InlineData("long", "managed")]
    [InlineData("mxfp4", "ggml")]
    public void SpeculativeDecodingMatchesPlainGreedy(string name, string executor)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        BackendType backend = executor == "ggml" ? TestGates.PinnedGgmlBackend : BackendType.Cpu;
        string path = Fixture(name);
        // ggml_cuda's verify rows take its multi-row products, decode rows its matvec ones: a near tie (2.3e-2 relative on the
        // MXFP4 fixture, A40) may part them.
        SpeculativeRuntimeChecks.AssertMatchesPlainGreedy(() => Create(path, backend), $"gpt-oss {name} {backend}",
            tieTolerance: backend == BackendType.Cpu ? 0 : backend == BackendType.GgmlCuda ? 3e-2 : 1e-2);
    }

    /// <summary>Routed experts kept in host memory (--cpu-moe) answer as the device's: on the managed backend (where host memory
    /// is the device's: its expert host's biased rows) on one device and over whole-layer stages, and on the pinned GGML backend
    /// (GGML's host MoE segment: the biased gate and up products, the clamped SwiGLU, the biased down product on the host).</summary>
    [Theory]
    [InlineData("mxfp4", "managed", 1)]
    [InlineData("mxfp4", "managed", 2)]
    [InlineData("q8-experts", "managed", 1)]
    [InlineData("mxfp4", "ggml", 1)]
    [InlineData("q8-experts", "ggml", 1)]
    public void HostExpertsMatchDeviceExperts(string name, string executor, int stages)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        BackendType backend = executor == "ggml" ? TestGates.PinnedGgmlBackend : BackendType.Cpu;
        string path = Fixture(name);
        GptOssLegacySynthetic.Case c = GptOssLegacySynthetic.Get(name);
        List<float[]> reference;
        MoeCpuOffloadConfig.Reset();
        using (var device = Create(path, backend)) reference = Script(device, c);
        MoeCpuOffloadConfig.SetAllLayers();
        try
        {
            ModelBase.ValidateExpertOffloadRequest(GptOssArchitecture.Descriptor, backend, 1);
            using var offloaded = Create(path, backend, 1, stages);
            List<float[]> actual = Script(offloaded, c);
            Assert.True(offloaded.GraphRuns > 0, $"{name} {backend}: no graph ran");
            // GGML's host seams counted the routes they computed (the managed backend's expert host runs its routes where the
            // device is, host memory: through the executor's biased products).
            if (backend != BackendType.Cpu)
            {
                ulong[] routes = offloaded.RoutingCountsForTests(0);
                Assert.True(routes != null && routes.Any(r => r > 0), $"{name} {backend}: no route ran on the host");
            }
            // GGML's host segment quantizes the activations its device products may keep in F32.
            double tolerance = backend == BackendType.Cpu ? NoiseTolerance : backend == BackendType.GgmlCpu ? NoiseTolerance : 2 * NoiseTolerance;
            for (int i = 0; i < reference.Count; i++)
            {
                double rel = RelL2(actual[i], reference[i]);
                _output.WriteLine($"--cpu-moe {name} {backend} {stages} stage(s) row {i}: relL2 {rel:E3}");
                Assert.True(rel <= tolerance, $"--cpu-moe {name} {backend}, {stages} stage(s), row {i}: relL2 {rel:E3} > {tolerance:E1}");
            }
        }
        finally
        {
            MoeCpuOffloadConfig.Reset();
        }
    }

    /// <summary>The same on the direct CUDA executor: its expert host computes every host route with the host kernel (the biased
    /// rows, the clamped SwiGLU).</summary>
    [CudaTheory]
    [InlineData("mxfp4")]
    [InlineData("q8-experts")]
    public void HostExpertsOnDirectCudaMatchDeviceExperts(string name)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture(name);
        GptOssLegacySynthetic.Case c = GptOssLegacySynthetic.Get(name);
        List<float[]> reference;
        MoeCpuOffloadConfig.Reset();
        using (var device = Create(path, BackendType.Cuda)) reference = Script(device, c);
        MoeCpuOffloadConfig.SetAllLayers();
        try
        {
            using var offloaded = Create(path, BackendType.Cuda);
            List<float[]> actual = Script(offloaded, c);
            (long hostRoutes, long streamed) = offloaded.ExpertHostCountsForTests;
            var executor = (TensorSharp.Cuda.CudaGraphExecutor)offloaded.GraphExecutorForTests;
            _output.WriteLine($"{name}: {hostRoutes} host routes, {streamed} streamed steps, {executor.DoorbellCalls} doorbell calls, "
                + $"{executor.HostExpertFallbacks} fallbacks");
            Assert.True(hostRoutes > 0, $"{name}: no host route");
            // The decode rows' host experts ring the doorbell (the step stays on the device), the biased experts on the MoE
            // engine; the prompt chunks stream their experts to the device.
            Assert.True(executor.DoorbellCalls > 0, $"{name}: no doorbell call");
            Assert.Equal(0, executor.HostExpertFallbacks);
            for (int i = 0; i < reference.Count; i++)
            {
                double rel = RelL2(actual[i], reference[i]);
                _output.WriteLine($"--cpu-moe {name} cuda row {i}: relL2 {rel:E3}");
                Assert.True(rel <= 2 * NoiseTolerance, $"--cpu-moe {name} cuda row {i}: relL2 {rel:E3}");
            }
        }
        finally
        {
            MoeCpuOffloadConfig.Reset();
        }
    }

    /// <summary>The offload is served on every backend, on one device and over stages, and refused with --tp: the per-rank
    /// graphs keep the experts on the devices.</summary>
    [Fact]
    public void ExpertOffloadIsRefusedUnderTensorParallelism()
    {
        MoeCpuOffloadConfig.Reset();
        MoeCpuOffloadConfig.SetAllLayers();
        try
        {
            var tp = Assert.Throws<NotSupportedException>(() => ModelBase.ValidateExpertOffloadRequest(GptOssArchitecture.Descriptor, BackendType.GgmlCuda, 2));
            Assert.Contains("--tp", tp.Message, StringComparison.Ordinal);
            foreach (BackendType backend in Enum.GetValues<BackendType>())
                ModelBase.ValidateExpertOffloadRequest(GptOssArchitecture.Descriptor, backend, 1);
        }
        finally
        {
            MoeCpuOffloadConfig.Reset();
        }
    }

    // ---------------------------------------------------------------- helpers

    /// <summary>A prompt chunk, decode steps, a second chunk and more steps: every step's logits.</summary>
    private static List<float[]> Script(ModelBase model, GptOssLegacySynthetic.Case c)
    {
        model.ResetKVCache();
        var rows = new List<float[]> { (float[])model.ForwardRefill(c.Prompt).Clone() };
        foreach (int t in c.Decode.Take(3)) rows.Add((float[])model.Forward(new[] { t }).Clone());
        rows.Add((float[])model.Forward(GptOssLegacySynthetic.Tokens(9, 4)).Clone());
        foreach (int t in c.Decode.Skip(3)) rows.Add((float[])model.Forward(new[] { t }).Clone());
        return rows;
    }

    /// <summary>The top logit's lead over the runner-up.</summary>
    private static float Margin(float[] x)
    {
        float first = float.NegativeInfinity, second = float.NegativeInfinity;
        foreach (float v in x)
        {
            if (v > first) (first, second) = (v, first);
            else if (v > second) second = v;
        }
        return first - second;
    }

    private static int Argmax(float[] x)
    {
        int best = 0;
        for (int i = 1; i < x.Length; i++) if (x[i] > x[best]) best = i;
        return best;
    }

    private static double RelL2(float[] a, float[] b)
    {
        double num = 0, den = 0;
        for (int i = 0; i < a.Length; i++)
        {
            double d = a[i] - b[i];
            num += d * d;
            den += (double)b[i] * b[i];
        }
        return Math.Sqrt(num / Math.Max(den, 1e-30));
    }
}
