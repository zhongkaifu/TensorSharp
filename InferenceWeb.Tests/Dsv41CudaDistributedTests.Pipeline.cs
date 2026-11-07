// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// Deferred boundary sends (TS_DSV4_ASYNC_SEND, Dsv4CudaEngine.Distributed.cs): a sending stage queues a non-final
// ubatch's device-to-host copies, records an event and returns; the wait and the send happen during the next ubatch,
// once its first GPU's layers are queued (or at the next send on a one-GPU stage). The last ubatch of a forward sends
// synchronously. These tests prove that the switch changes when the host sends and nothing else:
//  - every logit of a script with prompts of 1 to 5 ubatches (odd and even counts, exact multiples and remainders),
//    decode steps after each, a prompt fed in three chunks, refused truncations and resets is byte-identical with the
//    switch at 1, 2 and 0, at three cuts of 2 nodes x 2 GPUs, on 3 nodes x 2 GPUs (a middle stage that both receives and
//    sends) and on 3 nodes x 1 GPU over TCP (the model's stage link, whose middle node now keeps one buffer per
//    direction); and identical to the single-process engine;
//  - with the switch on, the payload of ubatch k really leaves after ubatch k + 1's first layers were queued (the
//    stage's layer counters at each send), and with it off it never does;
//  - a 960-token prompt (15 ubatches) answers byte-identically in every mode, run three times each, interleaved;
//  - nothing waits in the send staging between forwards;
//  - a stage that fails while a payload waits drops it, fails the pipeline, and tears down.
using System.Diagnostics;
using System.Runtime.InteropServices;
using TensorSharp;
using TensorSharp.Cuda;
using TensorSharp.Models;

namespace InferenceWeb.Tests;

public sealed partial class Dsv41CudaDistributedTests
{
    private const string AsyncSendVariable = "TS_DSV4_ASYNC_SEND";

    /// <summary>One forward of the deferred-send script, a reset before it, or (Tokens null) a truncation the stages
    /// must refuse without changing anything.</summary>
    private readonly record struct Step(string Name, int[] Tokens, bool Reset = false, int TruncateTo = -1);

    /// <summary>
    /// Ubatch 64. Prompts of 5 ubatches (4 x 64 + 44), 2 (an exact even multiple), 1 and 3 (64 + 64 + 22) appended
    /// to one sequence with decode steps after each and a refused truncation between them (head 623); then a reset
    /// and a 264-token prompt fed in three chunks of 2, 1 and 3 ubatches (70, 64, 130) with decode steps; then a reset
    /// and a 4-ubatch prompt (256) with decode steps. 26 logit vectors.
    /// </summary>
    private static readonly Step[] PipelineSteps = BuildPipelineSteps();

    private static Step[] BuildPipelineSteps()
    {
        var steps = new List<Step>();
        int seed = 7100;
        void Prompt(string name, int count, bool reset = false) => steps.Add(new(name, Tokens(seed++, count), reset));
        void Decode(string after, int count)
        {
            for (int i = 0; i < count; i++)
                steps.Add(new($"decode {i + 1} after {after}", Tokens(seed++, 1)));
        }

        Prompt("5 ubatches (300)", 300);
        Decode("5 ubatches", 3);
        Prompt("2 ubatches (128)", 128);
        Decode("2 ubatches", 2);
        Prompt("1 ubatch (37)", 37);
        Decode("1 ubatch", 1);
        steps.Add(new("refused truncation to 400", null, TruncateTo: 400));
        Prompt("3 ubatches (150)", 150);
        Decode("3 ubatches", 2);
        Prompt("chunk 1 of 3, 2 ubatches (70)", 70, reset: true);
        Prompt("chunk 2 of 3, 1 ubatch (64)", 64);
        steps.Add(new("refused truncation to 64", null, TruncateTo: 64));
        Prompt("chunk 3 of 3, 3 ubatches (130)", 130);
        Decode("the chunked prompt", 3);
        Prompt("4 ubatches (256)", 256, reset: true);
        Decode("4 ubatches", 2);
        return steps.ToArray();
    }

    private static IEnumerable<Step> PipelineForwards => PipelineSteps.Where(s => s.Tokens != null);

    private static int UbatchesOf(int tokens) => (tokens + Ubatch - 1) / Ubatch;

    /// <summary>Payloads a sending stage defers over the script: every ubatch of a forward but its last.</summary>
    private static long ExpectedDeferred => PipelineForwards.Sum(s => (long)UbatchesOf(s.Tokens.Length) - 1);

    private static long ScriptUbatches => PipelineForwards.Sum(s => (long)UbatchesOf(s.Tokens.Length));

    private static readonly System.Collections.Concurrent.ConcurrentDictionary<string, float[][]> PipelineReferences = new();

    /// <summary>The script on the single-process, single-device engine. Truncations are skipped: the stages refuse
    /// them, so the reference is the sequence that never truncated.</summary>
    private static float[][] PipelineReference(string path) => PipelineReferences.GetOrAdd(path, p =>
    {
        using ModelBase model = ModelBase.Create(p, BackendType.Cuda);
        var rows = new List<float[]>();
        foreach (Step step in PipelineForwards)
        {
            if (step.Reset)
                model.ResetKVCache();
            rows.Add((float[])model.Forward(step.Tokens).Clone());
        }
        return rows.ToArray();
    });

    private static void AssertSameBytes(string what, float[][] expected, float[][] actual)
    {
        Step[] forwards = PipelineForwards.ToArray();
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i].Length, actual[i].Length);
            for (int j = 0; j < expected[i].Length; j++)
            {
                if (BitConverter.SingleToInt32Bits(expected[i][j]) != BitConverter.SingleToInt32Bits(actual[i][j]))
                    Assert.Fail($"{what}: {forwards[i].Name}, logit {j}: {expected[i][j]:R} vs {actual[i][j]:R} " +
                        $"(max |d| over the vector {expected[i].Zip(actual[i], (a, b) => MathF.Abs(a - b)).Max():G4})");
            }
        }
    }

    /// <summary>What a stage's layer counters said at one of its sends: the payload's sequence number and how many
    /// ubatches the stage's first and last layers had been queued for.</summary>
    private readonly record struct SendPoint(int Stage, long Sequence, long FirstLayerRuns, long LastLayerRuns);

    private sealed record PipelineRun(float[][] Logits, StageCounters[] Stages, List<SendPoint> Sends, double PrefillMs,
        bool PendingBetweenForwards);

    /// <summary>The script on in-process stages (<see cref="LocalStages"/>) with TS_DSV4_ASYNC_SEND=<paramref name="mode"/>
    /// (0 synchronous, 1 deferred to the next hand-over, 2 deferred to the next send); every stage's sends are sampled
    /// through <see cref="Dsv4CudaEngine.BeforeBoundarySend"/>.</summary>
    private async Task<PipelineRun> PipelineOnLocalStages(string path, int[] boundaries, int localDevices, int mode)
    {
        _env.Set(AsyncSendVariable, mode.ToString());
        bool asyncSend = mode != 0;
        var sends = new List<SendPoint>();
        double prefillMs = 0;
        bool pending = false;
        try
        {
            LocalRun run = await LocalStages(path, boundaries, localDevices, drive: (executors, forward) =>
            {
                Dsv4CudaEngine[] byStage = executors.Select(e => e.Engine).OrderBy(e => e.OwnedLayers.First).ToArray();
                Assert.All(byStage.Take(byStage.Length - 1), e => Assert.Equal(asyncSend, e.DefersBoundarySends));
                Assert.False(byStage[^1].DefersBoundarySends);   // the last stage never sends
                Dsv4CudaEngine.BeforeBoundarySend = (stage, sequence) =>
                {
                    var (first, end) = byStage[stage].OwnedLayers;
                    lock (sends)
                        sends.Add(new SendPoint(stage, sequence, byStage[stage].LayerRuns[first], byStage[stage].LayerRuns[end - 1]));
                };
                var rows = new List<float[]>();
                foreach (Step step in PipelineSteps)
                {
                    if (step.Reset)
                        foreach (var ex in executors)
                            ex.Reset();
                    if (step.Tokens == null)
                    {
                        foreach (var ex in executors)
                            Assert.False(ex.Truncate(step.TruncateTo), $"a stage truncated to {step.TruncateTo}");
                        continue;
                    }
                    long start = Stopwatch.GetTimestamp();
                    rows.Add(forward(step.Tokens));
                    if (step.Tokens.Length > Ubatch)
                        prefillMs += Stopwatch.GetElapsedTime(start).TotalMilliseconds;
                    pending |= byStage.Any(e => e.HasPendingBoundarySend);
                }
                return rows.ToArray();
            });
            return new PipelineRun(run.Logits, run.Stages, sends, prefillMs, pending);
        }
        finally
        {
            Dsv4CudaEngine.BeforeBoundarySend = null;
        }
    }

    public static IEnumerable<object[]> PipelineShapes() => new[]
    {
        // name, topology, boundaries, GPUs per node
        new object[] { "2x2-span (the 1017's 2 x 2 path)", "span", new[] { 0, 3, 5, 7, 8 }, 2 },
        new object[] { "2x2-kv-rows-and-selection", "default", new[] { 0, 2, 3, 5, 6 }, 2 },
        new object[] { "2x2-after-engram", "default", new[] { 0, 1, 2, 4, 6 }, 2 },
        new object[] { "3x2-span (a middle stage that receives and sends)", "span", new[] { 0, 1, 3, 5, 6, 7, 8 }, 2 },
        new object[] { "3x1-span", "span", new[] { 0, 5, 7, 8 }, 1 },
    };

    /// <summary>
    /// The deferred send changes when a payload leaves and nothing else: every logit of the script is byte-identical
    /// with TS_DSV4_ASYNC_SEND at 1 (send at the next hand-over), 2 (at the next send) and 0 (synchronous), and equal to
    /// the single-process engine's. The counters agree between the runs; deferred, each sending stage held back every
    /// ubatch of a forward but the last, sent ubatch k only after ubatch k + 1's first layers were queued (its first
    /// layer had run k + 2 ubatches at the send of sequence k) and left nothing in the staging between forwards;
    /// synchronous, nothing was held back.
    /// </summary>
    [CudaTheory]
    [MemberData(nameof(PipelineShapes))]
    public async Task DeferredSends_AnswerAsSynchronousSends_ToTheBit(string name, string topology, int[] boundaries, int localDevices)
    {
        string path = topology == "span" ? Dsv41SyntheticModelBuilder.Fixture(Span) : Dsv41SyntheticModelBuilder.Fixture();
        PipelineRun on = await PipelineOnLocalStages(path, boundaries, localDevices, mode: 1);
        PipelineRun atSend = await PipelineOnLocalStages(path, boundaries, localDevices, mode: 2);
        PipelineRun off = await PipelineOnLocalStages(path, boundaries, localDevices, mode: 0);

        // Primary: the switch on against the switch off, in one build.
        AssertSameBytes($"[{name}] deferred (hand-over) vs synchronous sends", off.Logits, on.Logits);
        AssertSameBytes($"[{name}] deferred (next send) vs synchronous sends", off.Logits, atSend.Logits);
        // Secondary: against one process on one device.
        AssertSameBytes($"[{name}] deferred sends vs single-process", PipelineReference(path), on.Logits);
        AssertSameBytes($"[{name}] synchronous sends vs single-process", PipelineReference(path), off.Logits);

        int stages = on.Stages.Length;
        long ubatches = ScriptUbatches;
        foreach (PipelineRun deferred in new[] { on, atSend })
        for (int s = 0; s < stages; s++)
        {
            StageCounters a = deferred.Stages[s], b = off.Stages[s];
            Assert.Equal((b.First, b.End, b.Sent, b.Received), (a.First, a.End, a.Sent, a.Received));
            Assert.Equal((b.BytesSent, b.BytesReceived, b.RowsSent, b.RowsReceived), (a.BytesSent, a.BytesReceived, a.RowsSent, a.RowsReceived));
            Assert.Equal((b.SelectionsSent, b.MasksSent, b.MasksReceived, b.MaskBytesReceived),
                (a.SelectionsSent, a.MasksSent, a.MasksReceived, a.MaskBytesReceived));
            Assert.Equal(b.SourcesSent, a.SourcesSent);
            Assert.Equal(b.LayerRuns, a.LayerRuns);
            Assert.Equal(s < stages - 1 ? ubatches : 0, a.Sent);
            Assert.Equal(s > 0 ? ubatches : 0, a.Received);
            Assert.Equal(s < stages - 1 ? ExpectedDeferred : 0, a.Deferred);
            Assert.Equal(0, b.Deferred);
        }
        Assert.False(on.PendingBetweenForwards, "a payload waited in the send staging after a forward returned");
        Assert.False(atSend.PendingBetweenForwards, "a payload waited in the send staging after a forward returned (mode 2)");
        Assert.False(off.PendingBetweenForwards);

        // When each payload left: synchronous sends leave with the stage's layers at their own ubatch (the first layer
        // ran sequence + 1 ubatches); a deferred one leaves after the next ubatch's first layers were queued.
        foreach (var (deferred, mode) in new[] { (on, 1), (atSend, 2) })
        for (int s = 0; s < stages - 1; s++)
        {
            SendPoint[] onSends = deferred.Sends.Where(p => p.Stage == s).OrderBy(p => p.Sequence).ToArray();
            SendPoint[] offSends = off.Sends.Where(p => p.Stage == s).OrderBy(p => p.Sequence).ToArray();
            Assert.Equal(Enumerable.Range(0, (int)ubatches).Select(i => (long)i), onSends.Select(p => p.Sequence));
            Assert.Equal(Enumerable.Range(0, (int)ubatches).Select(i => (long)i), offSends.Select(p => p.Sequence));
            Assert.All(offSends, p => Assert.Equal((p.Sequence + 1, p.Sequence + 1), (p.FirstLayerRuns, p.LastLayerRuns)));
            long late = onSends.Count(p => p.FirstLayerRuns == p.Sequence + 2);
            long onTime = onSends.Count(p => p.FirstLayerRuns == p.Sequence + 1);
            Assert.Equal(ExpectedDeferred, late);
            Assert.Equal(ubatches - ExpectedDeferred, onTime);
            // In mode 1 with two GPUs the deferred payload leaves at the next ubatch's first hand-over: its last layer has
            // not been queued for the next ubatch yet. With one GPU, or in mode 2, it leaves at the next send, after all of them.
            Assert.All(onSends.Where(p => p.FirstLayerRuns == p.Sequence + 2),
                p => Assert.Equal(localDevices > 1 && mode == 1 ? p.Sequence + 1 : p.Sequence + 2, p.LastLayerRuns));
        }
        _output.WriteLine($"[{name}] {ubatches} ubatches per sending stage, {ExpectedDeferred} deferred; byte-identical on/off and " +
            $"to one process. Multi-ubatch forwards: {on.PrefillMs:F0} ms deferred to the hand-over, {atSend.PrefillMs:F0} ms " +
            $"deferred to the next send, {off.PrefillMs:F0} ms synchronous " +
            "(one GPU shared by every stage: not a speed measurement)");
    }

    /// <summary>
    /// A 960-token prompt (15 ubatches, the longest the tests' 1024-token context holds) on 2 x 2 GPUs, three times per
    /// mode with the modes interleaved (1, 0, 2, 1, 0, 2, ...): every run's logits (the prompt and two decode steps after
    /// it) are byte-identical to the first. The wall times are printed, not asserted: every stage and logical device
    /// shares this one GPU, so they show only that the deferred send costs nothing measurable here.
    /// </summary>
    [CudaFact]
    public async Task DeferredSends_OverALongPrompt_AnswerToTheBit()
    {
        string path = Dsv41SyntheticModelBuilder.Fixture(Span);
        int[] prompt = Tokens(7300, 960), steps = Tokens(7301, 2);
        var times = new Dictionary<int, List<double>> { [0] = new(), [1] = new(), [2] = new() };
        float[][] first = null;
        for (int round = 0; round < 3; round++)
        {
            foreach (int mode in new[] { 1, 0, 2 })
            {
                _env.Set(AsyncSendVariable, mode.ToString());
                double ms = 0;
                LocalRun run = await LocalStages(path, new[] { 0, 3, 5, 7, 8 }, localDevices: 2, drive: (executors, forward) =>
                {
                    var rows = new List<float[]>();
                    for (int repeat = 0; repeat < 2; repeat++)
                    {
                        foreach (var ex in executors)
                            ex.Reset();
                        long start = Stopwatch.GetTimestamp();
                        rows.Add(forward(prompt));
                        if (repeat == 1)
                            ms = Stopwatch.GetElapsedTime(start).TotalMilliseconds;   // the second, warm forward
                        foreach (int t in steps)
                            rows.Add(forward(new[] { t }));
                    }
                    return rows.ToArray();
                });
                times[mode].Add(ms);
                Assert.Equal(mode == 0 ? 0 : UbatchesOf(prompt.Length) * 2 - 2, run.Stages[0].Deferred);
                if (first == null)
                    first = run.Logits;
                else
                    for (int i = 0; i < first.Length; i++)
                        Assert.True(MemoryMarshal.AsBytes(first[i].AsSpan()).SequenceEqual(MemoryMarshal.AsBytes(run.Logits[i].AsSpan())),
                            $"mode {mode}, round {round}, row {i}: differs from the first run");
            }
        }
        double Median(List<double> v) => v.OrderBy(x => x).ElementAt(v.Count / 2);
        _output.WriteLine($"[long prompt 2x2-span] 960-token forward (15 ubatches), median of 3 warm runs: synchronous " +
            $"{Median(times[0]):F1} ms, deferred to the hand-over {Median(times[1]):F1} ms, deferred to the next send " +
            $"{Median(times[2]):F1} ms; all runs byte-identical (one shared GPU: not a speed measurement). Raw: " +
            string.Join("; ", times.Select(kv => $"mode {kv.Key}: {string.Join(", ", kv.Value.Select(v => v.ToString("F1")))}")));
    }

    /// <summary>
    /// The same over TCP on 127.0.0.1 with three nodes of one GPU each: the model's stage link carries the payloads, and
    /// its middle node receives each ubatch before it sends the previous one, through its own buffer per direction.
    /// Every logit is byte-identical with the switch on and off and to one process; the warm-up's one-token forward
    /// sends synchronously.
    /// </summary>
    [CudaFact]
    public async Task DeferredSends_OverTcp_ThroughAMiddleNode_AnswerToTheBit()
    {
        string path = Dsv41SyntheticModelBuilder.Fixture(Span);
        Task<float[][]> Drive(DeepSeek4Model driver) => Task.Run(() =>
        {
            var rows = new List<float[]>();
            foreach (Step step in PipelineForwards)
            {
                if (step.Reset)
                    driver.ResetKVCache();
                rows.Add((float[])driver.Forward(step.Tokens).Clone());
            }
            return rows.ToArray();
        });
        _env.Set(AsyncSendVariable, "1");
        Run on = await Distributed(new[] { 0, 5, 7, 8 }, path, Drive);
        _env.Set(AsyncSendVariable, "0");
        Run off = await Distributed(new[] { 0, 5, 7, 8 }, path, Drive);

        AssertSameBytes("[3x1 tcp] deferred vs synchronous sends", off.Logits, on.Logits);
        AssertSameBytes("[3x1 tcp] deferred sends vs single-process", PipelineReference(path), on.Logits);
        long ubatches = ScriptUbatches + 1;   // + the warm-up's one-token forward
        StageCounters[] a = on.Stages, b = off.Stages;
        Assert.Equal(3, a.Length);
        for (int s = 0; s < 3; s++)
        {
            Assert.Equal((b[s].Sent, b[s].Received, b[s].BytesSent, b[s].BytesReceived, b[s].RowsSent, b[s].RowsReceived),
                (a[s].Sent, a[s].Received, a[s].BytesSent, a[s].BytesReceived, a[s].RowsSent, a[s].RowsReceived));
            Assert.Equal(s < 2 ? ubatches : 0, a[s].Sent);
            Assert.Equal(s > 0 ? ubatches : 0, a[s].Received);
            Assert.Equal(s < 2 ? ExpectedDeferred : 0, a[s].Deferred);
            Assert.Equal(0, b[s].Deferred);
        }
        // The middle node sent on group 4's rows and the mask it received, as many as with synchronous sends.
        Assert.Equal((a[1].RowsReceived, a[1].MasksReceived), (a[1].RowsSent, a[1].MasksSent));
        _output.WriteLine($"[3x1 tcp] middle node received {a[1].Received} and sent {a[1].Sent} ({a[1].Deferred} deferred); " +
            $"send wall {a[0].SendMs:F1} / {b[0].SendMs:F1} ms (on / off) on the first node");
    }

    /// <summary>
    /// A stage that fails while one of its payloads waits in the send staging: the middle node of 3 x 2 GPUs fails just
    /// before it copies ubatch 2's payload out of its receive staging, when ubatch 1's payload is still unsent. The
    /// failure ends the forward on every stage within the deadline, the middle stage drops the waiting payload (nothing
    /// is pending afterwards) and every stage tears down.
    /// </summary>
    [CudaFact]
    public async Task AStageFailingWithAPayloadWaiting_DropsIt_AndFailsThePipeline()
    {
        string path = Dsv41SyntheticModelBuilder.Fixture(Span);
        _env.Set(AsyncSendVariable, "1");
        Exception failure = null;
        bool pendingAtFailure = false, pendingAfter = true;
        double seconds = 0;
        try
        {
            await LocalStages(path, new[] { 0, 1, 3, 5, 6, 7, 8 }, localDevices: 2, drive: (executors, forward) =>
            {
                Dsv4CudaEngine middle = executors.Select(e => e.Engine).OrderBy(e => e.OwnedLayers.First).ElementAt(1);
                Dsv4CudaEngine.BeforeBoundaryCopies = (stage, sequence, stream) =>
                {
                    if (stage != 1 || sequence != 2)
                        return;
                    pendingAtFailure = middle.HasPendingBoundarySend;
                    throw new IOException("injected middle-stage failure with ubatch 1's payload unsent");
                };
                var sw = Stopwatch.StartNew();
                failure = Record.Exception(() => forward(Tokens(7200, 300)));
                seconds = sw.Elapsed.TotalSeconds;
                pendingAfter = middle.HasPendingBoundarySend;
                return Array.Empty<float[]>();
            });
        }
        finally
        {
            Dsv4CudaEngine.BeforeBoundaryCopies = null;
        }
        Assert.NotNull(failure);
        Assert.IsNotType<TimeoutException>(failure);
        Assert.Contains("injected middle-stage failure", failure.ToString());
        Assert.True(pendingAtFailure, "ubatch 1's payload had already left when ubatch 2 arrived: the failure did not hit the deferred window");
        Assert.False(pendingAfter, "the failed stage kept its unsent payload");
        Assert.True(seconds < 30, $"the pipeline took {seconds:F1} s to fail");
        _output.WriteLine($"pipeline failed {seconds * 1000:F0} ms after the injected failure; the waiting payload was dropped");
    }
}
