// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Muse-Glimmer serving on the shared runtime (stage S3 of docs/design/muse-glimmer-unified.md): the unified step's per-request
// slots, the sliding-window rings, the prefix cache, checkpoints and checkpoint files, continuous batching and the caches'
// growth, on the runtime family (MuseGlimmerArchitecture.CreateRuntimeModel) and every executor this machine has (managed,
// the pinned GGML backend, MLX). The fixture is the untied one of the frozen legacy goldens with a window of 8 keys, and the
// rings are the tests' small geometry (RingGeometryForTests: 8 rows a step, so 16 cells), so every script laps its rings
// several times: the legacy bug class (docs §4.6, memory muse-glimmer-kv-grow-loop) lived where a cache grew, a ring wrapped
// or a snapshot was taken past either.
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp;
using TensorSharp.Cli;
using TensorSharp.Models;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Scheduling.PrefixCache;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

public sealed class MuseGlimmerRuntimeServingTests(ITestOutputHelper output) : IDisposable
{
    private const int Window = 8, SmallRingStepRows = 8, SmallRingCells = 16;

    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-mg-serving-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { } catch (UnauthorizedAccessException) { }
    }

    /// <summary>The goldens' untied fixture (8 blocks: 6 sliding-window, 2 full NoPE; a real logit scale and a head that reaches
    /// the softcap; block outputs small enough for the post norms' epsilon to show) with a window of <paramref name="window"/>.</summary>
    private string Fixture(int window = Window)
    {
        Directory.CreateDirectory(_dir);
        return new MuseGlimmerSyntheticModelBuilder
        {
            BlockCount = 8, SlidingWindow = window, LogitScale = 0.19612f, Softcap = 20f, HeadScale = 3f, BlockOutputScale = 0.004f,
        }.Write(Path.Combine(_dir, $"muse-glimmer-w{window}.gguf"));
    }

    private static StandardDecoderModel Create(string path, BackendType backend = BackendType.Cpu, int tp = 1, int stages = 1)
        => (StandardDecoderModel)MuseGlimmerArchitecture.CreateRuntimeModel(path, backend, tp,
            tp > 1 ? new SimulatedDeviceGroup(tp * stages) : null, stages);

    private static BackendType BackendOf(string executor) => executor switch
    {
        "managed" => BackendType.Cpu,
        "ggml" => TestGates.PinnedGgmlBackend,
        "mlx" => BackendType.Mlx,
        _ => throw new ArgumentOutOfRangeException(nameof(executor)),
    };

    /// <summary>Rings of 16 cells for the models constructed on this flow (the geometry is read at construction).</summary>
    private static IDisposable SmallRings()
    {
        StandardDecoderModel.RingGeometryForTests.Value = (SmallRingStepRows, SmallRingStepRows);
        return new RingScope();
    }

    private sealed class RingScope : IDisposable
    {
        public void Dispose() => StandardDecoderModel.RingGeometryForTests.Value = null;
    }

    /// <summary>Backends whose products round activations (ggml_cpu's F16 dot products, ggml_cuda, ggml_vulkan, direct CUDA) or
    /// whose batched rows take other kernels than lone ones (MLX): their batched and lone rows agree to a bound, not bit for bit.</summary>
    private static double RowBound(BackendType backend)
        => backend == BackendType.Cpu ? 1e-5 : backend is BackendType.GgmlCpu or BackendType.GgmlCuda or BackendType.GgmlVulkan or BackendType.Cuda
            ? 2.5e-2 : 5e-3;

    // ---------------------------------------------------------------- the family's serving declarations

    /// <summary>The runtime family serves on the unified step: every request in a slot of its own, the sliding-window layers in
    /// rings (16 cells here), class P prefix pages with truncation bounded by the rings, end states copied or donated, and
    /// persistable checkpoints (format 6: the rings whole).</summary>
    [Fact]
    public void TheFamilyServesOnSlotsWithRingsAndPersistableCheckpoints()
    {
        using var rings = SmallRings();
        using var _ = KvCacheDtypeConfig.ForCurrentFlowInTests(KvCacheDtype.F32);
        using StandardDecoderModel model = Create(Fixture());
        for (int l = 0; l < model.Config.NumLayers; l++)
            Assert.Equal(l % 4 < 3 ? SmallRingCells : 0, model.RingCellsForTests(l));
        Assert.Equal(SmallRingStepRows, model.MaxSequenceStepRows);
        Assert.True(ExecutionCapabilities.FromModel(model).SupportsUnifiedBatching, "the family must serve on the unified step");
        var slots = (IBatchedPagedModel)model;
        Assert.True(slots.SupportsPrefixCheckpoints);
        Assert.True(slots.SupportsRetainedCacheSerialization);
        PrefixCacheCapabilities caps = model.GetPrefixCacheCapabilities();
        output.WriteLine($"prefix cache: class {caps.Class}, truncation {caps.Truncation}, end state {caps.EndState}, persistable {caps.Persistable}");
        Assert.Equal(FamilyClass.P, caps.Class);
        Assert.Equal(TruncationKind.Any, caps.Truncation);
        Assert.Equal(EndStateSupport.CopyAndDonate, caps.EndState);
        Assert.True(caps.Persistable);
        Assert.Equal(int.MaxValue, caps.RewindCapTokens);
    }

    // ---------------------------------------------------------------- slots and continuous batching

    /// <summary>Three requests' slots in batched steps over wrapped rings (prompts of 19, 6 and 13 rows in ring steps, then
    /// decode, a lone step on one slot between) reproduce each request run alone, in every local placement on the managed
    /// backend.</summary>
    [Theory]
    [InlineData("managed", 1, 1)]
    [InlineData("managed", 2, 1)]
    [InlineData("managed", 1, 2)]
    [InlineData("ggml", 1, 1)]
    public void SlotBatchesOverRingsMatchEachSequenceAlone(string executor, int tp, int stages)
        => CompareSlotBatches(BackendOf(executor), tp, stages);

    [MlxFact]
    public void SlotBatchesOverRingsMatchEachSequenceAloneOnMlx() => CompareSlotBatches(BackendType.Mlx, 1, 1);

    private void CompareSlotBatches(BackendType backend, int tp, int stages)
    {
        using var rings = SmallRings();
        using var _ = KvCacheDtypeConfig.ForCurrentFlowInTests(KvCacheDtype.F32);
        string path = Fixture();
        SlotParity.AssertBatchedSlotsMatchAlone(() => Create(path, backend, tp, stages), RowBound(backend),
            $"muse-glimmer tp={tp} stages={stages} {backend}");
    }

    /// <summary>Four requests served together by the engine (prompts in 8-row chunks and decode tokens of several requests in
    /// one step, every ring lapped) give the greedy tokens each gives served alone by the same engine (4/4), up to a near-tie
    /// where the backend's batched rows round apart from lone ones.</summary>
    [Theory]
    [InlineData("managed")]
    [InlineData("ggml")]
    public Task ConcurrentRequestsOverRingsMatchSequentialOnes(string executor) => AssertConcurrentMatchesSequential(BackendOf(executor));

    [MlxFact]
    public Task ConcurrentRequestsOverRingsMatchSequentialOnesOnMlx() => AssertConcurrentMatchesSequential(BackendType.Mlx);

    private async Task AssertConcurrentMatchesSequential(BackendType backend)
    {
        using var rings = SmallRings();
        using var _ = KvCacheDtypeConfig.ForCurrentFlowInTests(KvCacheDtype.F32);
        string path = Fixture();
        const int maxNew = 24;
        int[][] prompts =
        {
            Enumerable.Range(0, 37).Select(i => 65 + (i * 7) % 50).ToArray(),
            Enumerable.Range(0, 21).Select(i => 97 + (i * 5) % 26).ToArray(),
            Enumerable.Range(0, 45).Select(i => 40 + (i * 11) % 150).ToArray(),
            Enumerable.Range(0, 13).Select(i => 48 + (i * 3) % 10).ToArray(),
        };
        var cfg = new SchedulerConfig
        {
            MaxNumBatchedTokens = 256, MaxNumRunningSequences = 4, MaxPrefillChunkSize = 8, NumBlocks = 64, BlockSize = 16,
            EnablePrefixCaching = true, DecodeQuantumTokens = 2, StopRepetition = false,
        };

        // Sequential: one request at a time, each on a fresh engine and model (no other request's cache to reuse). One model
        // lives at a time: disposing a GGML model clears the backend's process-wide host-buffer wrappers and reuse buffers
        // (ModelBase.Dispose), which another live model's cached graphs bind.
        var sequential = new int[prompts.Length][];
        var margins = new double[prompts.Length][];
        for (int r = 0; r < prompts.Length; r++)
        {
            using (StandardDecoderModel model = Create(path, backend))
            using (var engine = new InferenceEngine(model, cfg, NullLogger.Instance))
                sequential[r] = (await Collect(engine.SubmitRequest(new SequenceState($"seq{r}", prompts[r], maxNew, cfg.BlockSize, SamplingConfig.Greedy))))
                    .ToArray();
            margins[r] = GreedyMargins(Create(path, backend), prompts[r], sequential[r]);
        }

        using StandardDecoderModel together = Create(path, backend);
        using (var engine = new InferenceEngine(together, cfg, NullLogger.Instance))
        {
            var gate = new ComputeGate();
            gate.Close();
            engine.ComputeGate = gate;
            var handles = prompts.Select((p, r) => engine.SubmitRequest(new SequenceState($"req{r}", p, maxNew, cfg.BlockSize, SamplingConfig.Greedy)))
                .ToArray();
            gate.Open();
            int[][] concurrent = (await Task.WhenAll(handles.Select(Collect)).WaitAsync(TimeSpan.FromMinutes(3))).Select(t => t.ToArray()).ToArray();
            Assert.True(together.BatchedForwardCount > 0, "the engine must serve the four requests in batched steps over their slots");
            int identical = 0;
            for (int r = 0; r < prompts.Length; r++)
            {
                Assert.True(concurrent[r].Length > 0, $"request {r} produced nothing");
                int at = Enumerable.Range(0, Math.Min(concurrent[r].Length, sequential[r].Length))
                    .FirstOrDefault(t => concurrent[r][t] != sequential[r][t], -1);
                if (at < 0 && concurrent[r].Length == sequential[r].Length)
                {
                    identical++;
                    continue;
                }
                Assert.True(at >= 0 && backend != BackendType.Cpu && margins[r][at] <= 1e-2,
                    $"{backend} request {r} parts from its sequential stream at token {at} (top-two margin {(at >= 0 ? margins[r][at] : double.NaN):E2})");
            }
            output.WriteLine($"{backend}: {identical}/{prompts.Length} concurrent streams identical to the sequential ones " +
                $"({together.BatchedForwardCount} batched steps)");
        }
    }

    private static async Task<List<int>> Collect(InferenceRequestHandle handle)
    {
        var tokens = new List<int>();
        await foreach (int t in handle.Tokens.ReadAllAsync())
            tokens.Add(t);
        await handle.Completion;
        return tokens;
    }

    /// <summary>The relative top-two gap of each greedy step of <paramref name="produced"/> after <paramref name="prompt"/>, read
    /// alone on a fresh model (disposed here).</summary>
    private static double[] GreedyMargins(StandardDecoderModel model, int[] prompt, int[] produced)
    {
        using (model)
        {
            var margins = new double[produced.Length];
            float[] logits = model.ForwardRefill(prompt);
            for (int i = 0; i < produced.Length; i++)
            {
                margins[i] = TopTwoMargin(logits);
                if (i + 1 < produced.Length) logits = model.Forward(new[] { produced[i] });
            }
            return margins;
        }
    }

    private static double TopTwoMargin(float[] logits)
    {
        float first = float.NegativeInfinity, second = float.NegativeInfinity, scale = 0;
        foreach (float v in logits)
        {
            scale = Math.Max(scale, Math.Abs(v));
            if (v > first) { second = first; first = v; }
            else if (v > second) second = v;
        }
        return (first - second) / Math.Max(scale, 1e-30f);
    }

    // ---------------------------------------------------------------- checkpoints and checkpoint files

    /// <summary>A checkpoint of wrapped rings (23 positions, 16 cells) - cloned into a request's slot, copied on the device, into
    /// a pooled slot, and written to a checkpoint file another model instance reads back - continues exactly as a fresh run.</summary>
    [Theory]
    [InlineData("managed")]
    [InlineData("ggml")]
    public void CheckpointsOfWrappedRingsContinueExactly(string executor) => AssertCheckpoints(BackendOf(executor), executor == "ggml");

    [MlxFact]
    public void CheckpointsOfWrappedRingsContinueExactlyOnMlx() => AssertCheckpoints(BackendType.Mlx, deviceCopies: false);

    private void AssertCheckpoints(BackendType backend, bool deviceCopies)
    {
        using var rings = SmallRings();
        using var _ = KvCacheDtypeConfig.ForCurrentFlowInTests(KvCacheDtype.F32);
        string path = Fixture();
        CheckpointChecks.AssertCheckpointsContinueExactly(() => Create(path, backend), 1e-5, $"muse-glimmer {backend}", deviceCopies);
    }

    // ---------------------------------------------------------------- multi-turn prefix reuse

    /// <summary>
    /// Two conversations served in parallel for three turns (every turn's prompt the previous prompt, its reply and a new user
    /// message; the rings lapped from the first turn): each later turn reuses its conversation's whole previous turn (the
    /// finished request's slot donated to it: every position but the reply's last token, which the turn forwards with its new
    /// message), and answers exactly as a cold run of the same conversation does.
    /// </summary>
    [Theory]
    [InlineData("managed")]
    [InlineData("ggml")]
    public Task FollowUpTurnsReuseTheirConversationAndAnswerLikeAColdRun(string executor) => AssertMultiTurn(BackendOf(executor));

    [MlxFact]
    public Task FollowUpTurnsReuseTheirConversationAndAnswerLikeAColdRunOnMlx() => AssertMultiTurn(BackendType.Mlx);

    private async Task AssertMultiTurn(BackendType backend)
    {
        using var rings = SmallRings();
        using var _ = KvCacheDtypeConfig.ForCurrentFlowInTests(KvCacheDtype.F32);
        string path = Fixture();
        const int conversations = 2, turns = 3, maxNew = 12;
        var prompts = Enumerable.Range(0, conversations).Select(FirstPrompt).ToArray();
        var served = new List<(int Turn, int Conversation, List<int> Prompt, List<int> Reply, int Reused, int Carried)>();
        using var timeout = new CancellationTokenSource(TimeSpan.FromMinutes(3));
        using (StandardDecoderModel model = Create(path, backend))
        {
            SchedulerConfig config = MultiTurnConfig(conversations);
            using var engine = new InferenceEngine(model, config, NullLogger.Instance);
            var carried = new int[conversations];
            for (int turn = 0; turn < turns; turn++)
            {
                var requests = prompts.Select((p, i) => new SequenceState($"c{i}t{turn}", p.ToList(), maxNew, config.BlockSize,
                    SamplingConfig.Greedy, cacheScope: $"conversation{i}")).ToArray();
                InferenceCompletion[] done = await RunTogether(engine, requests, timeout.Token);
                for (int i = 0; i < conversations; i++)
                {
                    served.Add((turn, i, prompts[i], requests[i].OutputTokens.ToList(), done[i].PrefixCacheReusedTokens, carried[i]));
                    carried[i] = prompts[i].Count + requests[i].OutputTokens.Count;
                    prompts[i] = prompts[i].Concat(requests[i].OutputTokens).Concat(NextUser(i, turn)).ToList();
                }
            }
        }
        // The cold runs once the serving model is gone (one GGML model at a time: see AssertConcurrentMatchesSequential).
        foreach (var (turn, i, prompt, reply, reused, carriedBefore) in served)
        {
            List<int> cold = Cold(path, backend, prompt, maxNew, timeout.Token);
            output.WriteLine($"{backend} turn {turn} conversation {i}: prompt {prompt.Count}, reused {reused} (previous turn {carriedBefore} positions), " +
                $"reply {reply.Count} tokens, cold reply {(cold.SequenceEqual(reply) ? "identical" : "DIFFERENT")}");
            AssertSameReply(cold, reply, path, backend, prompt, $"{backend} turn {turn} conversation {i}");
            if (turn > 0)
                Assert.True(reused >= carriedBefore - 1,
                    $"{backend} turn {turn} conversation {i}: reused {reused} of its previous turn's {carriedBefore} positions");
        }
    }

    /// <summary>
    /// A follow-up whose history leaves the cached conversation a few positions before its end (an edited last reply): within
    /// the rings' slack past their window the finished slot rewinds to the divergence and the turn reuses everything before
    /// it; further back the rings have overwritten what the next row reads, so the rewind is refused (the runtime's
    /// RingsKeepTruncation) and the turn reuses no more than an exact prefix the cache still holds. Either way the reply is a
    /// cold run's.
    /// </summary>
    [Theory]
    [InlineData("managed", 5)]
    [InlineData("managed", 12)]
    [InlineData("ggml", 5)]
    [InlineData("ggml", 12)]
    public async Task EditedFollowUpsRewindWithinTheRingsOrReprefill(string executor, int edited)
    {
        BackendType backend = BackendOf(executor);
        using var rings = SmallRings();
        using var _ = KvCacheDtypeConfig.ForCurrentFlowInTests(KvCacheDtype.F32);
        string path = Fixture();
        const int maxNew = 16;
        using var timeout = new CancellationTokenSource(TimeSpan.FromMinutes(3));
        List<int> second;
        SequenceState turn2;
        // The served turns first; the cold run once the serving model is gone (one GGML model at a time).
        using (StandardDecoderModel model = Create(path, backend))
        {
            SchedulerConfig config = MultiTurnConfig(1);
            using var engine = new InferenceEngine(model, config, NullLogger.Instance);
            List<int> first = FirstPrompt(0);
            var turn1 = new SequenceState("e0", first.ToList(), maxNew, config.BlockSize, SamplingConfig.Greedy, cacheScope: "edited");
            await RunTogether(engine, new[] { turn1 }, timeout.Token);
            Assert.Equal(maxNew, turn1.OutputTokens.Count);
            // The client sends the reply back with its last `edited` tokens replaced: the history leaves the cache there.
            List<int> reply = turn1.OutputTokens.Take(maxNew - edited).Concat(Enumerable.Range(0, edited).Select(k => 200 + k)).ToList();
            second = first.Concat(reply).Concat(NextUser(0, 0)).ToList();
            int divergence = first.Count + maxNew - edited;
            turn2 = new SequenceState("e1", second.ToList(), maxNew, config.BlockSize, SamplingConfig.Greedy, cacheScope: "edited");
            InferenceCompletion done = (await RunTogether(engine, new[] { turn2 }, timeout.Token))[0];
            int cached = first.Count + maxNew - 1;
            // A rewind keeps what the next row's window reads, (divergence - 8, divergence), while that is inside the ring's last 16
            // cells: divergence - 7 >= cached - 16 (RingsKeepTruncation).
            bool keeps = divergence - Window + 1 >= cached - SmallRingCells;
            Assert.Equal(keeps, model.CanTruncateKVCache(cached, divergence));
            output.WriteLine($"{backend}: edited the last {edited} reply tokens; the cache held {cached} positions, the history leaves it at " +
                $"{divergence}; reused {done.PrefixCacheReusedTokens} ({(keeps ? "a rewind within the rings' slack" : "past the slack: refused")})");
            Assert.True(done.PrefixCacheReusedTokens <= divergence, "reused past where the history leaves the cache");
            if (keeps)
                Assert.Equal(divergence, done.PrefixCacheReusedTokens);
            else
                Assert.True(done.PrefixCacheReusedTokens < divergence, "a refused rewind cannot reuse up to the divergence");
        }
        AssertSameReply(Cold(path, backend, second, maxNew, timeout.Token), turn2.OutputTokens, path, backend, second,
            $"{backend} edited follow-up ({edited})");
    }

    private static SchedulerConfig MultiTurnConfig(int conversations) => new()
    {
        BlockSize = 16, NumBlocks = 1024, MaxNumRunningSequences = Math.Max(2, conversations), MaxNumBatchedTokens = 512,
        SoloPrefillChunkSize = 512, EnablePrefixCaching = true, StopRepetition = false,
    };

    /// <summary>Submits <paramref name="requests"/> so they are admitted in one step, and waits for them and for the step that
    /// keeps the last one's state.</summary>
    private static async Task<InferenceCompletion[]> RunTogether(InferenceEngine engine, SequenceState[] requests, CancellationToken token)
    {
        var gate = new ComputeGate();
        gate.Close();
        engine.ComputeGate = gate;
        var handles = requests.Select(r => engine.SubmitRequest(r)).ToArray();
        gate.Open();
        InferenceCompletion[] done = await Task.WhenAll(handles.Select(h => h.Completion)).WaitAsync(token);
        await engine.WhenStepSettledAsync().WaitAsync(token);
        return done;
    }

    /// <summary>The reply a fresh model with no cache gives for <paramref name="prompt"/>.</summary>
    private static List<int> Cold(string path, BackendType backend, List<int> prompt, int maxNew, CancellationToken token)
    {
        using StandardDecoderModel model = Create(path, backend);
        var config = new SchedulerConfig { BlockSize = 16, NumBlocks = 64, MaxNumRunningSequences = 1, StopRepetition = false };
        using var session = new CliInferenceSession(model, config, NullLogger.Instance);
        return session.Generate(prompt, maxNew, SamplingConfig.Greedy, cancellationToken: token, enablePrefixCache: false).Tokens;
    }

    /// <summary>The served reply is the cold one, or (on a backend whose prompt chunks round apart from a whole prompt's) leaves
    /// it only where the cold run's greedy step is a near-tie.</summary>
    private static void AssertSameReply(List<int> cold, IReadOnlyList<int> served, string path, BackendType backend, List<int> prompt, string label)
    {
        int at = Enumerable.Range(0, Math.Min(cold.Count, served.Count)).FirstOrDefault(t => cold[t] != served[t], -1);
        if (at < 0 && cold.Count == served.Count) return;
        Assert.True(at >= 0 && backend != BackendType.Cpu, $"{label}: the reply differs from a cold run's ({served.Count} vs {cold.Count} tokens)");
        double margin = GreedyMargins(Create(path, backend), prompt.ToArray(), cold.ToArray())[at];
        Assert.True(margin <= 1e-2, $"{label}: the reply leaves a cold run's at token {at}, where its top-two margin is {margin:E2}");
    }

    // A 20-token shared opening, then a user message of the conversation's own (already past the 16-cell rings).
    private static List<int> FirstPrompt(int conversation)
        => Enumerable.Range(10, 20).Concat(Enumerable.Range(0, 20).Select(k => 40 + (conversation * 13 + k * 7) % 190)).ToList();

    private static IEnumerable<int> NextUser(int conversation, int turn)
        => Enumerable.Range(0, 9).Select(k => 30 + (conversation * 17 + turn * 29 + k * 11) % 200);

    // ---------------------------------------------------------------- the caches' growth (the legacy bug class)

    /// <summary>
    /// The full layers' cache grows mid-sequence (the primary starting at 256 cells: 256 -> 512 inside a prefill that keeps 200
    /// rows, 512 -> 1024 on a decode step, then to 4096 in a prefill that also laps the window's ring, 768 cells at the default
    /// geometry), then decodes on the wrapped rings: every forward gives exactly the logits of a cache that never grew, as the
    /// legacy probe's grow fix demanded (MuseGlimmerKvGrowTests). The same plan through a request's slot (the unified step),
    /// the slot reserved for 1 token and grown by the steps, gives a presized slot's logits.
    /// </summary>
    [Theory]
    [InlineData("managed")]
    [InlineData("ggml")]
    public void CachesGrowingMidSequenceGiveThePresizedLogits(string executor) => AssertGrowthIsExact(BackendOf(executor));

    [MlxFact]
    public void CachesGrowingMidSequenceGiveThePresizedLogitsOnMlx() => AssertGrowthIsExact(BackendType.Mlx);

    private void AssertGrowthIsExact(BackendType backend)
    {
        using var _ = KvCacheDtypeConfig.ForCurrentFlowInTests(KvCacheDtype.F16);
        string path = Fixture(window: 64);
        int[] tokens = Enumerable.Range(0, 2600).Select(i => (int)(((uint)i * 2654435761u >> 5) % MuseGlimmerSyntheticModelBuilder.ByteTokens)).ToArray();
        var plan = new List<(int From, int To)> { (0, 200), (200, 400) };
        for (int p = 400; p < 550; p++) plan.Add((p, p + 1));
        plan.Add((550, 2450));
        for (int p = 2450; p < 2510; p++) plan.Add((p, p + 1));

        List<float[]> Primary(int initial)
        {
            StandardDecoderModel.InitialCacheLengthForTests.Value = initial;
            try
            {
                using StandardDecoderModel model = Create(path, backend);
                Assert.Equal(initial, model.PrimaryCacheCapacityForTests);
                var rows = plan.Select(s => (float[])model.Forward(tokens[s.From..s.To]).Clone()).ToList();
                Assert.True(model.RingCellsForTests(0) > 0 && model.RingCellsForTests(0) < 2450, "the window's ring must wrap");
                output.WriteLine($"{backend} primary from {initial} cells: {model.PrimaryCacheCapacityForTests} cells at the end, rings of {model.RingCellsForTests(0)}");
                return rows;
            }
            finally { StandardDecoderModel.InitialCacheLengthForTests.Value = null; }
        }
        List<float[]> Slot(int reserve)
        {
            using StandardDecoderModel model = Create(path, backend);
            var unified = (IUnifiedBatchModel)model;
            unified.EnsureSequenceSlot("r", reserve);
            int held = model.SlotCapacityForTests("r")!.Value;
            var rows = new List<float[]>();
            var logits = new float[1][];
            foreach (var (from, to) in plan)
            {
                unified.EnsureSequenceSlot("r", reserve, to);
                Assert.True(unified.TryForwardUnified(new[] { "r" }, new[] { tokens[from..to] }, new[] { from }, logits));
                rows.Add((float[])logits[0].Clone());
            }
            output.WriteLine($"{backend} slot reserved for {reserve}: {held} -> {model.SlotCapacityForTests("r")} cells");
            if (reserve == 1) Assert.True(model.SlotCapacityForTests("r") > held, "the slot must grow");
            return rows;
        }

        AssertIdentical(Primary(4096), Primary(256), plan, $"{backend} primary grown vs presized");
        AssertIdentical(Slot(4096), Slot(1), plan, $"{backend} slot grown vs presized");
    }

    private static void AssertIdentical(List<float[]> presized, List<float[]> grown, List<(int From, int To)> plan, string label)
    {
        for (int i = 0; i < plan.Count; i++)
        {
            float max = 0;
            for (int v = 0; v < presized[i].Length; v++) max = Math.Max(max, Math.Abs(presized[i][v] - grown[i][v]));
            Assert.True(max == 0f, $"{label}: forward {i} (to position {plan[i].To}) differs by {max:E3}: a growth lost or moved rows");
        }
    }

    /// <summary>A rewind keeps what the next row's window reads, or is refused: a stack of rings and linear caches truncates within
    /// a wrapped ring's slack past its window, never to positions it overwrote, and the cache truncated there continues as a fresh
    /// run does.</summary>
    [Theory]
    [InlineData("managed")]
    [InlineData("ggml")]
    public void RingsRewindOnlyWhereTheyKeepTheWindow(string executor)
    {
        BackendType backend = BackendOf(executor);
        using var rings = SmallRings();
        using var _ = KvCacheDtypeConfig.ForCurrentFlowInTests(KvCacheDtype.F32);
        string path = Fixture();
        int[] tokens = Enumerable.Range(0, 40).Select(i => 40 + (i * 11) % 150).ToArray();
        using StandardDecoderModel model = Create(path, backend);
        model.ForwardRefill(tokens);
        // 40 positions in a 16-cell ring keep 24..39; position t's window reads (t - 8, t).
        Assert.True(model.CanTruncateKVCache(40, 32));
        Assert.True(model.CanTruncateKVCache(40, 31));    // reads 24..30, the oldest position kept
        Assert.False(model.CanTruncateKVCache(40, 30));   // reads 23..29: 23 is overwritten
        Assert.True(model.CanTruncateKVCache(40, 0));
        Assert.False(model.TryTruncateKVCache(20));
        Assert.True(model.TryTruncateKVCache(31));
        float[] continued = (float[])model.Forward(new[] { 77 }).Clone();

        using StandardDecoderModel fresh = Create(path, backend);
        fresh.ForwardRefill(tokens.Take(31).ToArray());
        float[] expected = fresh.Forward(new[] { 77 });
        double num = 0, den = 0;
        for (int v = 0; v < expected.Length; v++) { double d = continued[v] - expected[v]; num += d * d; den += (double)expected[v] * expected[v]; }
        double rel = Math.Sqrt(num / Math.Max(den, 1e-30));
        Assert.True(rel <= RowBound(backend), $"{backend}: truncated ring vs fresh run: relL2 {rel:E2}");
    }
}
