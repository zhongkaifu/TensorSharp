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
using TensorSharp.Runtime.Speculative;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// Gemma 4 on the shared decoder runtime (<see cref="Gemma4RuntimeModel"/>) over the synthetic E2B/E4B layout
/// (<see cref="Gemma4DenseSyntheticModelBuilder"/>: per-layer embeddings, KV sharing, a 208-token window, proportional global
/// RoPE) and the 26B-A4B layout (<see cref="Gemma4MoeSyntheticModelBuilder"/>: fused gate|up experts with per-expert scales,
/// V = K global layers, a 32-token window): the legacy model's frozen logits, the pinned GGML backend against the managed
/// one, and every mechanism the runtime claims for the family (placements, batched slots, continuous batching, routed
/// experts on the host, checkpoints, speculation through n-gram drafts and through the gemma4-assistant head). The real
/// checkpoints were held to the legacy model on an A40 by eng/validation/Gemma4RuntimeAB.
/// </summary>
[Collection("MoeCpuOffloadConfig")]   // the host-experts test sets the process-wide expert-offload policy
public sealed class Gemma4RuntimeTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-gemma4-runtime-" + Guid.NewGuid().ToString("N"));
    private readonly ITestOutputHelper _output;

    public Gemma4RuntimeTests(ITestOutputHelper output)
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
        return File.Exists(path) ? path : Gemma4LegacySynthetic.Get(name).Write(path);
    }

    private string Assistant() => new Gemma4AssistantSyntheticModelBuilder().Write(Path.Combine(_dir, "assistant.gguf"));

    private static Gemma4RuntimeModel Create(string path, BackendType backend = BackendType.Cpu, int tp = 1, int stages = 1)
    {
        ITensorParallelGroup group = tp > 1 ? new SimulatedDeviceGroup(tp * stages) : null;
        return new Gemma4RuntimeModel(path, backend, tp, group, stages);
    }

    // ---------------------------------------------------------------- the forward

    [Fact]
    public void TheFactoryCreatesTheRuntime()
    {
        using ModelBase model = ModelBase.Create(Fixture("dense"), BackendType.Cpu);
        Assert.IsType<Gemma4RuntimeModel>(model);
        Assert.Equal(Gemma4DenseSyntheticModelBuilder.Pattern.Length, model.Config.NumLayers);
    }

    /// <summary>
    /// How far two F32 runs of these fixtures' random layers part when only their summation order differs: the legacy
    /// suite's own batched-against-single tolerance on the dense file (Gemma4DenseBatchedForwardParityTests, 0.025 on the
    /// managed backend). The runtime and the legacy model agree exactly while every attention reads at most three keys
    /// (prompts of one or two tokens and their first steps) and part by 2e-3 to 2e-2 relL2 past that (measured on the
    /// managed backend); a wrong recipe part (a rotary, a norm, a mask, a route) moves these logits by 0.1 to 1.
    /// </summary>
    private const double NoiseTolerance = 0.025;

    /// <summary>The legacy model's logits, frozen before it was deleted (<see cref="Gemma4LegacySynthetic"/>): the same math
    /// on the managed backend, every case's prompt row and decode rows (the same top token wherever the frozen top two are
    /// half a logit apart).</summary>
    [Theory]
    [InlineData("dense")]
    [InlineData("dense-long")]
    [InlineData("moe")]
    [InlineData("moe-kv2")]
    public void MatchesTheFrozenLegacyLogits(string name)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture(name);
        List<float[]> frozen = Gemma4LegacySynthetic.Rows(name, path);
        using var model = Create(path);
        List<float[]> actual = Gemma4LegacySynthetic.Run(model, Gemma4LegacySynthetic.Get(name));
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
    /// take TF32 and its quantized products quantize the activations).</summary>
    [Theory]
    [InlineData("dense")]
    [InlineData("dense-long")]
    [InlineData("moe")]
    public void PinnedGgmlBackendMatchesTheManagedBackend(string name)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture(name);
        Gemma4LegacySynthetic.Case c = Gemma4LegacySynthetic.Get(name);
        List<float[]> managed, ggml;
        using (var model = Create(path)) managed = Gemma4LegacySynthetic.Run(model, c);
        using (var model = Create(path, TestGates.PinnedGgmlBackend)) ggml = Gemma4LegacySynthetic.Run(model, c);
        double tolerance = TestGates.PinnedGgmlBackend == BackendType.GgmlCpu ? NoiseTolerance : 2 * NoiseTolerance;
        for (int i = 0; i < managed.Count; i++)
        {
            double rel = RelL2(ggml[i], managed[i]);
            _output.WriteLine($"{name} {TestGates.PinnedGgmlBackend} row {i}: relL2 {rel:E3}");
            Assert.True(rel <= tolerance, $"{name} {TestGates.PinnedGgmlBackend} row {i}: relL2 {rel:E3} > {tolerance:E1}");
        }
    }

    /// <summary>A prompt in one chunk and in pieces (across the sliding windows' rings) gives the same logits.</summary>
    [Theory]
    [InlineData("dense-long", "managed")]
    [InlineData("dense-long", "ggml")]
    [InlineData("moe", "managed")]
    [InlineData("moe", "ggml")]
    public void ChunkedPrefillMatchesOneChunk(string name, string executor)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        BackendType backend = executor == "ggml" ? TestGates.PinnedGgmlBackend : BackendType.Cpu;
        using var model = Create(Fixture(name), backend);
        int[] prompt = Gemma4LegacySynthetic.Get(name).Prompt;
        float[] whole = (float[])model.ForwardRefill(prompt).Clone();
        model.ResetKVCache();
        float[] pieces = null;
        int cut1 = prompt.Length / 3, cut2 = cut1 + 1;
        foreach (int[] chunk in new[] { prompt[..cut1], prompt[cut1..cut2], prompt[cut2..] })
            pieces = model.Forward(chunk);
        double tolerance = backend is BackendType.Cpu ? 1e-5 : backend == BackendType.GgmlCpu ? 5e-4 : 3e-2;
        Assert.True(RelL2(pieces, whole) <= tolerance, $"{name} {backend} chunked: relL2 {RelL2(pieces, whole):E3}");
    }

    // ---------------------------------------------------------------- mechanisms

    public static TheoryData<string, int, int> Placements() => new()
    {
        { "dense", 2, 1 },
        { "dense", 1, 2 },
        { "dense", 1, 3 },
        { "moe-kv2", 2, 1 },
        { "moe-kv2", 1, 2 },
        { "moe-kv2", 1, 3 },
    };

    /// <summary>Tensor parallelism (per-rank graphs over simulated devices) and whole-layer stages (the per-layer embeddings
    /// carried from stage to stage, the KV-shared layers on their sources' stage) answer as one device; a MoE layer's fused
    /// gate|up experts split by halves. A stage's layers run the one device's ops; the ranks' partial sums meet in another
    /// order (noise-level apart on these fixtures, <see cref="NoiseTolerance"/>). The hybrid is a declared gap: the graph-only
    /// family has no graphs for tensor-parallel stages yet.</summary>
    [Theory]
    [MemberData(nameof(Placements))]
    public void PlacementMatchesOneDevice(string name, int tp, int stages)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture(name);
        Gemma4LegacySynthetic.Case c = Gemma4LegacySynthetic.Get(name);
        List<float[]> reference;
        using (var one = Create(path)) reference = Script(one, c);
        using var placed = Create(path, BackendType.Cpu, tp, stages);
        Assert.Equal(tp, placed.Mesh.TensorParallel);
        Assert.Equal(stages, placed.Mesh.Stages);
        List<float[]> actual = Script(placed, c);
        Assert.True(placed.GraphRuns > 0, $"{name} tp={tp} stages={stages}: no graph ran");
        double tolerance = tp == 1 ? 1e-6 : NoiseTolerance;
        if (stages > 1)
            for (int l = 0; l < placed.Config.NumLayers; l++)
                Assert.Equal(placed.Mesh.StageOf(placed.KvSourceForTests(l)), placed.Mesh.StageOf(l));
        for (int i = 0; i < reference.Count; i++)
        {
            double rel = RelL2(actual[i], reference[i]);
            Assert.True(rel <= tolerance, $"{name} tp={tp} stages={stages} row {i}: relL2 {rel:E2} > {tolerance:E0}");
        }
    }

    /// <summary>The stage planner starts no stage between a KV-shared layer and the layer whose cache it reads: the dense
    /// file's layers 6 and 7 read layers 4 and 5, so its last stage holds layers 4 to 7 however many stages there are.</summary>
    [Theory]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    public void StagesKeepKvSharedLayersWithTheirSources(int stages)
    {
        using var placed = Create(Fixture("dense"), BackendType.Cpu, 1, stages);
        Assert.Equal(stages, placed.Mesh.Stages);
        for (int l = 4; l < 8; l++) Assert.Equal(stages - 1, placed.Mesh.StageOf(l));
    }

    [Theory]
    [InlineData("dense")]
    [InlineData("moe")]
    public void SlotBatchesMatchEachSequenceAlone(string name)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture(name);
        SlotParity.AssertBatchedSlotsMatchAlone(() => Create(path), 1e-5, $"gemma4 {name}");
    }

    /// <summary>The engine's greedy streams of three requests served together (batched steps over their slots, prefill
    /// chunks, decode quanta, the prefix cache) equal each request's alone.</summary>
    [Theory]
    [InlineData("dense")]
    [InlineData("moe")]
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
        Assert.True(((IBatchedPagedModel)model).SupportsPerSequenceFusedForward);
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

    /// <summary>Routed experts kept in host memory (--cpu-moe) answer as the device's on the managed backend (where host
    /// memory is the device's) and on the pinned GGML backend (GGML's host MoE segment), on one device and over whole-layer
    /// stages: GeGLU over the fused gate|up stack, the host kernel's dot products quantizing the activations to the weights'
    /// dot type (noise-level apart).</summary>
    [Theory]
    [InlineData("managed", 1)]
    [InlineData("managed", 2)]
    [InlineData("ggml", 1)]
    public void HostExpertsMatchDeviceExperts(string executor, int stages)
        => AssertHostExpertsMatchDevice(executor == "ggml" ? TestGates.PinnedGgmlBackend : BackendType.Cpu, stages);

    /// <summary>The same on the direct CUDA executor: its expert host computes the host's routes with the host kernel at
    /// Gemma's activation and the device's (none without a tier) through its ops.</summary>
    [CudaFact]
    public void HostExpertsOnDirectCudaMatchDeviceExperts() => AssertHostExpertsMatchDevice(BackendType.Cuda, 1);

    private void AssertHostExpertsMatchDevice(BackendType backend, int stages)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture("moe-kv2");
        Gemma4LegacySynthetic.Case c = Gemma4LegacySynthetic.Get("moe-kv2");
        List<float[]> reference;
        MoeCpuOffloadConfig.Reset();
        using (var device = Create(path, backend)) reference = Script(device, c);
        MoeCpuOffloadConfig.SetAllLayers();
        try
        {
            using var offloaded = Create(path, backend, 1, stages);
            List<float[]> actual = Script(offloaded, c);
            double tolerance = backend == BackendType.Cpu || backend == BackendType.GgmlCpu ? NoiseTolerance : 2 * NoiseTolerance;
            for (int i = 0; i < reference.Count; i++)
            {
                double rel = RelL2(actual[i], reference[i]);
                _output.WriteLine($"--cpu-moe {backend} {stages} stage(s) row {i}: relL2 {rel:E3}");
                Assert.True(rel <= tolerance, $"--cpu-moe {backend}, {stages} stage(s), row {i}: relL2 {rel:E2}");
            }
        }
        finally
        {
            MoeCpuOffloadConfig.Reset();
        }
    }

    [Theory]
    [InlineData("dense", "managed")]
    [InlineData("dense", "ggml")]
    [InlineData("moe", "managed")]
    public void CheckpointsContinueExactly(string name, string executor)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        BackendType backend = executor == "ggml" ? TestGates.PinnedGgmlBackend : BackendType.Cpu;
        string path = Fixture(name);
        CheckpointChecks.AssertCheckpointsContinueExactly(() => Create(path, backend), backend == BackendType.GgmlCuda ? 1e-2 : 1e-5,
            $"gemma4 {name} {backend}", deviceCopies: executor == "ggml");
    }

    /// <summary>Greedy speculation (scripted drafts, accepted and rejected, so verify windows roll the sliding-window rings
    /// back) emits exactly the plain greedy tokens.</summary>
    [Theory]
    [InlineData("dense")]
    [InlineData("moe")]
    public void SpeculativeDecodingMatchesPlainGreedy(string name)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture(name);
        SpeculativeRuntimeChecks.AssertMatchesPlainGreedy(() => Create(path), $"gemma4 {name}");
    }

    /// <summary>The gemma4-assistant head drafts on the runtime: its blocks query the trunk's donor caches (through the
    /// KV-sharing map) and chain their state through the hidden row, and greedy speculation with it emits plain greedy's
    /// tokens.</summary>
    [Theory]
    [InlineData("managed")]
    [InlineData("ggml")]
    public void AssistantHeadSpeculationMatchesPlainGreedy(string executor)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        BackendType backend = executor == "ggml" ? TestGates.PinnedGgmlBackend : BackendType.Cpu;
        string path = Fixture("dense"), draft = Assistant();
        Gemma4RuntimeModel CreateWithHead()
        {
            Gemma4RuntimeModel model = Create(path, backend);
            model.LoadMtpDraftWeights(draft);
            return model;
        }
        using (var probe = CreateWithHead())
        {
            Assert.Equal(DraftHeadKind.PerToken, probe.DraftHeadKind);
            Assert.True(probe.DraftHeadResumesAfterGap);
        }
        SpeculativeRuntimeChecks.AssertDraftHeadMatchesPlainGreedy(CreateWithHead, $"gemma4-assistant {backend}",
            tieTolerance: backend == BackendType.Cpu ? 0 : 1e-2, reference: () => Create(path, backend));
    }

    /// <summary>The head's step reads only the trunk's caches and the row it is handed: the same token and row at the same
    /// trunk length draft the same logits however often they are asked (a replayed step), and a step after more trunk rows
    /// reads them.</summary>
    [Fact]
    public void AssistantHeadStepsAreStateless()
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        using var model = Create(Fixture("dense"));
        model.LoadMtpDraftWeights(new Gemma4AssistantSyntheticModelBuilder { MtpPrefix = true }.Write(Path.Combine(_dir, "assistant-mtp.gguf")));
        IDraftHead head = model;
        int vocab = model.Config.VocabSize, hidden = model.Config.HiddenSize;
        model.ForwardRefill(Gemma4LegacySynthetic.Tokens(29, 1));
        var row = Enumerable.Range(0, hidden).Select(i => MathF.Sin(i * 0.37f)).ToArray();
        float[] Draft()
        {
            var logits = new float[vocab];
            head.DraftStep(17, row, model.CacheSeqLen, logits, new float[hidden]);
            return logits;
        }
        float[] first = Draft(), again = Draft();
        Assert.Equal(first, again);
        model.Forward(new[] { 9, 10, 11 });
        Assert.True(RelL2(Draft(), first) > 1e-4, "a draft after more trunk rows must read them");
    }

    // ---------------------------------------------------------------- helpers

    /// <summary>A prompt chunk, decode steps, a second chunk and more steps: every step's logits.</summary>
    private static List<float[]> Script(ModelBase model, Gemma4LegacySynthetic.Case c)
    {
        model.ResetKVCache();
        var rows = new List<float[]> { (float[])model.ForwardRefill(c.Prompt).Clone() };
        foreach (int t in c.Decode.Take(3)) rows.Add((float[])model.Forward(new[] { t }).Clone());
        rows.Add((float[])model.Forward(Gemma4LegacySynthetic.Tokens(9, 4)).Clone());
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
