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
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;
using Xunit;

namespace InferenceWeb.Tests;

/// <summary>
/// <see cref="Inkling2RuntimeModel"/> against <see cref="Inkling2Reference"/>, the double-precision forward written from
/// llama.cpp's inkling2 graphs, on the synthetic F32 checkpoints of <see cref="Inkling2SyntheticModelBuilder"/>: a prompt past
/// the sliding window and the relative extent, then decode steps, for the plain and the looped stack.
/// </summary>
public sealed class Inkling2RuntimeTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-inkling2-runtime-" + Guid.NewGuid().ToString("N"));
    private readonly Xunit.Abstractions.ITestOutputHelper _output;

    public Inkling2RuntimeTests(Xunit.Abstractions.ITestOutputHelper output)
    {
        _output = output;
        Directory.CreateDirectory(_dir);
    }

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { }
    }

    private static double RelL2(float[] actual, double[] expected)
    {
        double num = 0, den = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            double d = actual[i] - expected[i];
            num += d * d;
            den += expected[i] * expected[i];
        }
        return Math.Sqrt(num / Math.Max(den, 1e-30));
    }

    [Fact]
    public void TheFactoryCreatesTheRuntime()
    {
        var builder = new Inkling2SyntheticModelBuilder();
        string path = builder.Write(Path.Combine(_dir, "factory.gguf"));
        using ModelBase model = ModelBase.Create(path, BackendType.Cpu);
        Assert.IsType<Inkling2RuntimeModel>(model);
        Assert.Equal(builder.Slots, model.Config.NumLayers);
        Assert.Equal(builder.Vocab, model.Config.VocabSize);
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void MatchesTheReferenceForward(bool looped) => AssertMatchesReference(BackendType.Cpu, looped, looped ? 5e-5 : 2e-5);

    /// <summary>
    /// The pinned GGML backend, the reference run along the routes the device picked (<see cref="AssertMatchesReference"/>).
    /// ggml_cpu computes in F32 (relL2 ~1e-6 plain, ~2e-5 looped). The GPU backends' F32 products of a prompt chunk round
    /// their operands (ggml-cuda's cuBLAS handles take TF32 tensor-op math, docs/embeddings.md; ggml-metal stages
    /// kernel_mul_mm_f32_f32's tiles in half from 17 rows, docs/design/diffusion-runtime.md; ggml-vulkan's matrix kernels
    /// likewise): along the same routes the plain stack agrees to ~1e-3..9e-3, and the synthetic loop's three gated passes
    /// over random weights amplify that to ~1e-1.
    /// </summary>
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void PinnedGgmlBackendMatchesTheReferenceForward(bool looped)
        => AssertMatchesReference(TestGates.PinnedGgmlBackend, looped,
            TestGates.PinnedGgmlBackend == BackendType.GgmlCpu ? 1e-4 : looped ? 0.3 : 1e-2);

    /// <summary>The route check of <see cref="AssertMatchesReference"/> fails on a routing bug: a device that reads exp_probs_b
    /// negated (the reference reads the file's own) scores and routes outside what rounding explains by more than 10x, from
    /// its first layer on (<see cref="SelectionScorePerturbation"/> measured RMS 8e-2..1.3e-1 at every layer run).</summary>
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void RouteCheckCatchesAWrongExpertBias(bool looped)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        var builder = new Inkling2SyntheticModelBuilder { Looped = looped };
        var reference = Inkling2Reference.Load(builder.Write(Path.Combine(_dir, $"inkling2-true-{looped}.gguf")));
        string slipped = new Inkling2SyntheticModelBuilder { Looped = looped, ExpertBiasScale = -1f }
            .Write(Path.Combine(_dir, $"inkling2-slipped-{looped}.gguf"));
        int[] tokens = Enumerable.Range(0, 17).Select(i => 3 + (i * 7) % 55).ToArray();
        using var model = Create(slipped, TestGates.PinnedGgmlBackend);
        var routes = new DeviceRouteLog(model, builder.ExpertsUsed, builder.Experts);
        model.ForwardRefill(tokens);
        Assert.Equal(tokens.Length * builder.Slots, routes.Ids.Count);
        double ratio = RouteCheck(reference, tokens, routes, looped, out _, out string worst);
        _output.WriteLine($"{TestGates.PinnedGgmlBackend} looped={looped}, negated bias: {worst}");
        Assert.True(ratio >= 10, $"a negated selection bias stayed within 10x of the rounding tolerances: {worst}");
    }

    /// <summary>
    /// What a GGML backend's rounding moves a router's selection scores (sigmoid probability plus exp_probs_b, every expert of
    /// every routing) by, along the same routes as the reference's, per layer run r (the runtime's layer index): the largest
    /// RMS over a run's routings and the largest single deviation. Derived from the backends' rounding, not fitted to one
    /// prompt: <see cref="SelectionScorePerturbation"/> forced each backend along the reference's routes over 12 fixtures
    /// (seeds 1-12) x 3 prompts (17, 24 and 40 random tokens, then 8 decode steps), and these are the largest values
    /// ggml_cuda, ggml_vulkan and ggml_metal gave at each run (2026-11-21, ggml d9fe7bf; RTX PRO 4500, M5 Pro), rounded up.
    /// They grow with depth (the products' TF32 and half-staged rounding, ~30x through the loop's random gates); the
    /// checks allow <see cref="ScoreHeadroom"/> times them.
    /// </summary>
    private static readonly double[] PlainScoreRms = { 2.2e-4, 3.6e-4, 6.2e-4, 8.5e-4, 1.4e-3, 1.6e-3, 1.9e-3, 3.4e-3 };
    private static readonly double[] PlainScoreMost = { 1.6e-3, 2.7e-3, 3.8e-3, 5.5e-3, 1.2e-2, 1.2e-2, 1.7e-2, 2.6e-2 };
    private static readonly double[] LoopScoreRms =
    {
        2.2e-4, 3.6e-4, 6.2e-4, 8.5e-4, 1.4e-3, 1.6e-3, 2.0e-3, 2.9e-3, 5.0e-3, 4.9e-3, 7.0e-3, 9.0e-3, 1.4e-2, 1.5e-2, 1.7e-2, 2.2e-2,
    };
    private static readonly double[] LoopScoreMost =
    {
        1.6e-3, 2.7e-3, 3.8e-3, 5.5e-3, 1.2e-2, 1.2e-2, 1.6e-2, 2.4e-2, 3.5e-2, 4.1e-2, 6.5e-2, 6.1e-2, 9.2e-2, 1.1e-1, 1.1e-1, 1.4e-1,
    };
    private const double ScoreHeadroom = 4;

    /// <summary>Measures the score deviations <see cref="RouteCheck"/> allows (<see cref="PlainScoreRms"/>) on the pinned GGML
    /// backend (Category Bench: a sample, not a check): per layer run, the largest RMS and the largest |device - reference| of
    /// the selection scores over the sample, the device forced along the reference's routes; then the smallest RMS a device
    /// reading exp_probs_b negated gives (unforced, the reference along its routes), the slip the check must catch.</summary>
    [Theory]
    [Trait("Category", "Bench")]
    [InlineData(false)]
    [InlineData(true)]
    public void SelectionScorePerturbation(bool looped)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        int runs = new Inkling2SyntheticModelBuilder { Looped = looped }.Slots;
        var rms = new double[runs];
        var most = new double[runs];
        var slipRms = Enumerable.Repeat(double.PositiveInfinity, runs).ToArray();
        for (int seed = 1; seed <= 12; seed++)
        {
            var builder = new Inkling2SyntheticModelBuilder { Looped = looped, Seed = seed };
            string path = builder.Write(Path.Combine(_dir, $"inkling2-seed{seed}-{looped}.gguf"));
            var reference = Inkling2Reference.Load(path);
            foreach (int length in new[] { 17, 24, 40 })
            {
                var rng = new Random(seed * 100 + length);
                int[] prompt = Enumerable.Range(0, length).Select(_ => 3 + rng.Next(builder.Vocab - 3)).ToArray();
                int[] decode = Enumerable.Range(0, 8).Select(_ => 3 + rng.Next(builder.Vocab - 3)).ToArray();
                int[] all = prompt.Concat(decode).ToArray();
                var log = new Inkling2Reference.Counters { RoutingLog = new() };
                reference.Logits(all, log);
                var chosen = log.RoutingLog.ToDictionary(r => (r.Position, r.Run));
                using (var model = Create(path, TestGates.PinnedGgmlBackend))
                {
                    model.ForcedRoutesForTests = (layer, request, position) => chosen[(position, layer)].Chosen;
                    var routes = new DeviceRouteLog(model, builder.ExpertsUsed, builder.Experts);
                    model.ForwardRefill(prompt);
                    foreach (int token in decode) model.Forward(new[] { token });
                    Assert.Equal(all.Length * builder.Slots, routes.Scores.Count);
                    foreach ((int run, double r, double m) in ScoreDeviation(log.RoutingLog, routes, runs))
                    {
                        rms[run] = Math.Max(rms[run], r);
                        most[run] = Math.Max(most[run], m);
                    }
                }
                if (length != 17) continue;
                string slipped = new Inkling2SyntheticModelBuilder { Looped = looped, Seed = seed, ExpertBiasScale = -1f }
                    .Write(Path.Combine(_dir, $"inkling2-seed{seed}-slipped-{looped}.gguf"));
                using (var model = Create(slipped, TestGates.PinnedGgmlBackend))
                {
                    var routes = new DeviceRouteLog(model, builder.ExpertsUsed, builder.Experts);
                    model.ForwardRefill(prompt);
                    var along = new Inkling2Reference.Counters { RoutingLog = new() };
                    reference.RouteOverride = (position, run) => routes.Ids[(position, run)];
                    reference.Logits(prompt, along);
                    reference.RouteOverride = null;
                    foreach ((int run, double r, _) in ScoreDeviation(along.RoutingLog, routes, runs)) slipRms[run] = Math.Min(slipRms[run], r);
                }
            }
        }
        for (int run = 0; run < runs; run++)
            _output.WriteLine($"{TestGates.PinnedGgmlBackend} looped={looped} run {run,2}: largest score RMS {rms[run]:E2}, largest |deviation| {most[run]:E2}; " +
                $"negated bias: smallest RMS {slipRms[run]:E2}");
        _output.WriteLine($"{TestGates.PinnedGgmlBackend} looped={looped}: largest score RMS {rms.Max():E2}, largest |deviation| {most.Max():E2}");
    }

    /// <summary>Per layer run, the RMS and the largest |device - reference| of the selection scores over every routing's
    /// experts, the reference's routings <paramref name="routings"/> (logged along the same routes as the device's).</summary>
    private static IEnumerable<(int Run, double Rms, double Most)> ScoreDeviation(List<Inkling2Reference.Routing> routings, DeviceRouteLog routes, int runs)
    {
        var sum = new double[runs];
        var count = new long[runs];
        var most = new double[runs];
        foreach (Inkling2Reference.Routing r in routings)
        {
            float[] device = routes.Scores[(r.Position, r.Run)];
            for (int e = 0; e < device.Length; e++)
            {
                double d = device[e] - r.Scores[e];
                sum[r.Run] += d * d;
                count[r.Run]++;
                most[r.Run] = Math.Max(most[r.Run], Math.Abs(d));
            }
        }
        for (int run = 0; run < runs; run++)
            if (count[run] > 0) yield return (run, Math.Sqrt(sum[run] / count[run]), most[run]);
    }

    /// <summary>Each (position, layer run)'s experts and selection scores as a model's steps picked them on its device
    /// (DeviceRoutesForTests), one sequence's rows.</summary>
    private sealed class DeviceRouteLog
    {
        public readonly Dictionary<(int Position, int Run), int[]> Ids = new();
        public readonly Dictionary<(int Position, int Run), float[]> Scores = new();

        public DeviceRouteLog(StandardDecoderModel model, int k, int experts)
        {
            model.DeviceRoutesForTests = r =>
            {
                Assert.Single(r.Starts);
                int rows = r.Counts[0];
                Assert.Equal(rows * k, r.Ids.Length);
                Assert.Equal(rows * experts, r.Scores.Length);
                for (int i = 0; i < rows; i++)
                {
                    Ids[(r.Starts[0] + i, r.Layer)] = r.Ids[(i * k)..((i + 1) * k)];
                    Scores[(r.Starts[0] + i, r.Layer)] = r.Scores[(i * experts)..((i + 1) * experts)];
                }
            };
        }
    }

    /// <summary>
    /// The reference along a device's routes over <paramref name="tokens"/> (its logits out), and how far the device's routing
    /// lies outside its rounding, as the worst ratio to the measured deviations times <see cref="ScoreHeadroom"/> (at most 1:
    /// within; <paramref name="worst"/> names it). Per layer run: the RMS and the largest deviation of the device's selection
    /// scores from the reference's, and the largest inversion of a device route against the reference's scores (an expert
    /// left out that the reference ranks above one taken), allowed twice the largest deviation: a device that takes the top-k
    /// of its own scores (checked exactly) inverts two experts only by both scores' deviations. That is the tie a backend's
    /// rounding may split: this fixture's prompt has a route whose third and fourth scores are 1.35e-4 apart (position 7,
    /// layer 4), which every GPU backend's prompt chunk had flipped, moving every later row by 5e-2 (the loop by up to 0.7).
    /// </summary>
    private static double RouteCheck(Inkling2Reference reference, int[] tokens, DeviceRouteLog routes, bool looped,
        out double[][] logits, out string worst)
    {
        var log = new Inkling2Reference.Counters { RoutingLog = new() };
        reference.RouteOverride = (position, run) => routes.Ids[(position, run)];
        try
        {
            logits = reference.Logits(tokens, log);
        }
        finally
        {
            reference.RouteOverride = null;
        }
        double[] rmsTolerance = looped ? LoopScoreRms : PlainScoreRms, mostTolerance = looped ? LoopScoreMost : PlainScoreMost;
        int runs = rmsTolerance.Length;
        var sum = new double[runs];
        var count = new long[runs];
        var most = new double[runs];
        var inversion = Enumerable.Repeat(double.NegativeInfinity, runs).ToArray();
        foreach (Inkling2Reference.Routing r in log.RoutingLog)
        {
            Assert.InRange(r.Run, 0, runs - 1);
            float[] device = routes.Scores[(r.Position, r.Run)];
            int[] left = Enumerable.Range(0, device.Length).Where(e => Array.IndexOf(r.Chosen, e) < 0).ToArray();
            Assert.True(r.Chosen.Min(e => device[e]) >= left.Max(e => device[e]),
                $"position {r.Position}, run {r.Run}: the device's route [{string.Join(",", r.Chosen)}] is not the top-k of its own scores");
            for (int e = 0; e < device.Length; e++)
            {
                double d = device[e] - r.Scores[e];
                sum[r.Run] += d * d;
                count[r.Run]++;
                most[r.Run] = Math.Max(most[r.Run], Math.Abs(d));
            }
            inversion[r.Run] = Math.Max(inversion[r.Run], left.Max(e => r.Scores[e]) - r.Chosen.Min(e => r.Scores[e]));
        }
        double ratio = double.NegativeInfinity;
        worst = "no routings";
        for (int run = 0; run < runs; run++)
        {
            if (count[run] == 0) continue;
            double rms = Math.Sqrt(sum[run] / count[run]);
            foreach ((string what, double value, double tolerance) in new[]
            {
                ("score RMS", rms, ScoreHeadroom * rmsTolerance[run]),
                ("largest score deviation", most[run], ScoreHeadroom * mostTolerance[run]),
                ("route inversion", inversion[run], 2 * ScoreHeadroom * mostTolerance[run]),
            })
                if (value / tolerance > ratio)
                {
                    ratio = value / tolerance;
                    worst = $"run {run} {what} {value:E2} against {tolerance:E2} ({ratio:F2}x)";
                }
        }
        return ratio;
    }

    /// <summary>Greedy speculative decoding (scripted drafts, partly accepted) emits exactly the plain greedy tokens: a verify
    /// window keeps every layer's convolution inputs (K, V, the attention and MoE outputs) and the commit takes the accepted
    /// rows' windows; the rejected rows' ring cells are written again later.</summary>
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void SpeculativeDecodingMatchesPlainGreedy(bool looped)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = new Inkling2SyntheticModelBuilder { Looped = looped }.Write(Path.Combine(_dir, $"inkling2-spec-{looped}.gguf"));
        SpeculativeRuntimeChecks.AssertMatchesPlainGreedy(() =>
        {
            var model = new Inkling2RuntimeModel(path, BackendType.Cpu);
            model.GraphLoweringEnabled = true;
            return model;
        }, $"inkling2 looped={looped}");
    }

    /// <summary>A fixture whose vocabulary holds the token ids of the shared slot, engine and checkpoint scripts, with conv
    /// windows of 4.5 KB a layer (192 channels x 6 columns), as the real model's are large (80 KB): the GGML graph builder
    /// keeps a state leaf under 4 KB in per-run compute memory (uploaded, then downloaded), not in the device copy a slot
    /// copy on the device reads and writes.</summary>
    private string ScriptFixture(bool looped)
        => new Inkling2SyntheticModelBuilder { Looped = looped, Vocab = 125, VocabPadded = 128, ConvTaps = 7 }
            .Write(Path.Combine(_dir, $"inkling2-script-{looped}.gguf"));

    private static Inkling2RuntimeModel Create(string path, BackendType backend = BackendType.Cpu)
    {
        var model = new Inkling2RuntimeModel(path, backend);
        model.GraphLoweringEnabled = true;
        return model;
    }

    /// <summary>Several requests' slots in batched steps (each slot its own attention rows, rings and conv windows) give
    /// each request's rows alone.</summary>
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void SlotBatchesMatchEachSequenceAlone(bool looped)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = ScriptFixture(looped);
        SlotParity.AssertBatchedSlotsMatchAlone(() => Create(path), looped ? 5e-5 : 1e-5, $"inkling2 looped={looped}");
    }

    /// <summary>
    /// The same on the pinned GGML backend, plain stack, both runs routed as the reference routes each request
    /// (ForcedRoutesForTests: the routers' top-k replaced by those ids, the weights still the device's): a batched prefill's
    /// rows and a lone request's run different kernels, whose rounding (see
    /// <see cref="PinnedGgmlBackendMatchesTheReferenceForward"/>) splits a near-tied route one way in one run and the other in
    /// the other, so their own routes would compare two different expert choices. With the routes forced the requests'
    /// rows still take sparse, per-row experts in batched steps, so ids mixed up between sequences would show.
    /// </summary>
    [Fact]
    public void SlotBatchesMatchEachSequenceAloneOnThePinnedGgmlBackend()
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = ScriptFixture(looped: false);
        var reference = Inkling2Reference.Load(path);
        var routes = new Dictionary<(int Request, int Position, int Run), int[]>();
        for (int s = 0; s < SlotParity.Prompts.Length; s++)
        {
            var log = new Inkling2Reference.Counters { RoutingLog = new() };
            reference.Logits(SlotParity.Prompts[s].Concat(SlotParity.DecodeTokens[s]).ToArray(), log);
            foreach (Inkling2Reference.Routing r in log.RoutingLog) routes[(s, r.Position, r.Run)] = r.Chosen;
        }
        int alone = -1;
        SlotParity.AssertBatchedSlotsMatchAlone(() =>
        {
            Inkling2RuntimeModel model = Create(path, TestGates.PinnedGgmlBackend);
            model.ForcedRoutesForTests = (layer, request, position) =>
                routes[(request == null ? alone : Array.IndexOf(SlotParity.RequestIds, request), position, layer)];
            return model;
        }, TestGates.PinnedGgmlBackend == BackendType.GgmlCpu ? 1e-4 : 2.5e-2, $"inkling2 {TestGates.PinnedGgmlBackend}", s => alone = s);
    }

    /// <summary>The engine's greedy streams of three requests served together (batched steps over their slots, prefill
    /// chunks, decode quanta) equal each request's alone.</summary>
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public async Task ConcurrentRequestsAreBatched(bool looped)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = ScriptFixture(looped);
        const int maxNewTokens = 8;
        var prompts = new List<int[]>
        {
            Enumerable.Range(0, 19).Select(i => 65 + (i * 7) % 50).ToArray(),
            Enumerable.Range(0, 27).Select(i => 97 + (i * 5) % 26).ToArray(),
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
                    int token = Array.IndexOf(logits, logits.Max());
                    if (reference.Tokenizer.EosTokenIds.Contains(token)) break;   // the engine's stream ends without it
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

    /// <summary>A shared-prefix checkpoint (attention rows, sliding rings and every layer's conv windows), cloned or written
    /// to a file and read back, continues exactly as a fresh run; on a GGML backend also through the device copy program.</summary>
    [Theory]
    [InlineData("ggml")]
    [InlineData("managed")]
    public void CheckpointsContinueExactly(string executor)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        BackendType backend = executor == "ggml" ? TestGates.PinnedGgmlBackend : BackendType.Cpu;
        string path = ScriptFixture(looped: true);
        CheckpointChecks.AssertCheckpointsContinueExactly(() => Create(path, backend), backend == BackendType.GgmlCuda ? 1e-2 : 1e-5,
            $"inkling2 {backend}", deviceCopies: executor == "ggml");
    }

    /// <summary>
    /// The model on <paramref name="backend"/> against the reference: a prompt chunk, then decode steps. On a GGML backend the
    /// reference runs along the routes the device picked (DeviceRoutesForTests), which must be its own top-k up to the ties
    /// the backend's rounding can split (<see cref="RouteCheck"/>): top-k routing is discontinuous.
    /// </summary>
    private void AssertMatchesReference(BackendType backend, bool looped, double tolerance, int promptLength = 17)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        var builder = new Inkling2SyntheticModelBuilder { Looped = looped };
        string path = builder.Write(Path.Combine(_dir, $"inkling2-{looped}.gguf"));
        var reference = Inkling2Reference.Load(path);
        int[] prompt = Enumerable.Range(0, promptLength).Select(i => 3 + (i * 7) % 55).ToArray();
        int[] decode = { 11, 12, 13, 14, 15, 16, 17, 18 };
        int[] all = prompt.Concat(decode).ToArray();

        using var model = new Inkling2RuntimeModel(path, backend);
        model.GraphLoweringEnabled = true;
        DeviceRouteLog routes = backend == BackendType.Cpu ? null : new DeviceRouteLog(model, builder.ExpertsUsed, builder.Experts);
        _output.WriteLine($"graph refusal: {model.GraphRefusalForTests ?? "none"}");
        var actual = new List<(int Position, float[] Logits)> { (prompt.Length - 1, (float[])model.ForwardRefill(prompt).Clone()) };
        for (int t = 0; t < decode.Length; t++)
            actual.Add((prompt.Length + t, (float[])model.Forward(new[] { decode[t] }).Clone()));
        Assert.True(model.GraphRuns > 0, "no graph runs");
        double[][] expected;
        if (routes == null)
        {
            expected = reference.Logits(all);
        }
        else
        {
            Assert.Equal(all.Length * builder.Slots, routes.Ids.Count);
            double ratio = RouteCheck(reference, all, routes, looped, out expected, out string worst);
            _output.WriteLine($"{backend} looped={looped}: routes against the reference, worst {worst}");
            Assert.True(ratio <= 1, $"the device's routes lie outside its rounding: {worst}");
        }
        double[] rel = actual.Select(a => RelL2(a.Logits, expected[a.Position])).ToArray();
        _output.WriteLine($"{backend} looped={looped}: relL2 per row {string.Join(" ", rel.Select(r => r.ToString("E1")))}");
        for (int i = 0; i < rel.Length; i++)
            Assert.True(rel[i] <= tolerance, $"row {actual[i].Position}: relL2 {rel[i]:E2} > {tolerance:E1}");
    }
}
