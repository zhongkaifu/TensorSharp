// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Issue #376 (T1-7), the shared-row checkpoints' less travelled paths (the step's review). A conversation's branches: once a
// branch with a prompt of its own past a checkpoint, or the conversation's next turn, has written the rows past it, an adopter
// of the checkpoint copies it (taken over, two live branches had each taken the other's end state over at every turn, each
// turn prefilling its branch again), while retries of the checkpoint's prompt still take the slot over. Also: a steal that
// uses up a longer checkpoint over the same rows, a steal refused because one of the payloads it would use up is not the
// adopter's, a steal from the checkpoint that took the rows over when their slot was freed (a hand-over heir), a QSA
// checkpoint whose length is not a whole number of blocks (its incomplete block's pooled cell holds the owner's value when the
// slot is taken over), and two turns of one conversation adopting the same checkpoint at once. Each adopter continues bit for
// bit as a fresh run of the same tokens.
using Microsoft.Extensions.Logging.Abstractions;

using InferenceWeb.Tests.PrefixCache.Fakes;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Scheduling.PrefixCache;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

public sealed class SharedRowCheckpointReviewTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-shared-rows-review-" + Guid.NewGuid().ToString("N"));
    private readonly ITestOutputHelper _output;

    public SharedRowCheckpointReviewTests(ITestOutputHelper output)
    {
        _output = output;
        Directory.CreateDirectory(_dir);
    }

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    private string Fixture(string family)
    {
        string marker = Path.Combine(_dir, family + ".path");
        if (File.Exists(marker)) return File.ReadAllText(marker);
        string path = family switch
        {
            "qwen35" => new Qwen35DenseSyntheticModelBuilder { Layers = 8, UntiedOutput = true }.Write(Path.Combine(_dir, "q35-dense.gguf")),
            "qwen4exp" => Qwen4ExpSyntheticModelBuilder.Write(Path.Combine(_dir, "q4e.gguf"), indexerTopK: 8, qsa: true),
            "mistral3" => new DenseDecoderSyntheticModelBuilder { Architecture = "mistral3" }.Write(Path.Combine(_dir, "mistral3.gguf")),
            _ => throw new ArgumentException(family),
        };
        File.WriteAllText(marker, path);
        return path;
    }

    private StandardDecoderModel Create(string family, RecordingPayloadSink sink = null, bool slotBound = true)
    {
        using IDisposable dtype = KvCacheDtypeConfig.ForCurrentFlowInTests(KvCacheDtype.F32);
        var model = (StandardDecoderModel)ModelBase.Create(Fixture(family), BackendType.Cpu);
        model.SlotBoundPoliciesForTests = slotBound;
        model.SlotInvariantChecksForTests = true;
        model.AttachPrefixCache(sink ?? new RecordingPayloadSink());
        return model;
    }

    private static int[] Tokens(int count, int salt) => Enumerable.Range(0, count).Select(i => 3 + (i * 37 + i / 7 + salt * 11) % 250).ToArray();

    private static readonly int[] Next = Tokens(3, 5);

    private static int ArgMax(float[] row)
    {
        int best = 0;
        for (int i = 1; i < row.Length; i++)
            if (row[i] > row[best]) best = i;
        return best;
    }

    private static float[] Forward(StandardDecoderModel model, string id, int[] tokens, int at)
    {
        var logits = new float[1][];
        Assert.True(((IUnifiedBatchModel)model).TryForwardUnified(new[] { id }, new[] { tokens }, new[] { at }, logits), $"{id} did not run");
        return (float[])logits[0].Clone();
    }

    /// <summary><paramref name="prompt"/> on a new slot of <paramref name="id"/>, in two chunks.</summary>
    private static void RunPrompt(StandardDecoderModel model, string id, int[] prompt)
    {
        Assert.Equal(0, ((IUnifiedBatchModel)model).EnsureSequenceSlot(id, 64));
        int half = prompt.Length / 2;
        Forward(model, id, prompt[..half], 0);
        Forward(model, id, prompt[half..], half);
    }

    /// <summary>Three tokens, a two-row verify window keeping one row, then greedy decode steps: every row's logits.</summary>
    private static List<float[]> Continuation(StandardDecoderModel model, string id, int at, int decodes = 3)
    {
        var rows = new List<float[]>();
        float[] last = Forward(model, id, Next, at);
        rows.Add(last);
        int length = at + Next.Length, next = ArgMax(last);
        int vocab = model.Config.VocabSize;
        var window = new float[2 * vocab];
        var batched = (IBatchedSpeculativeTarget)model;
        Assert.True(batched.TryBatchedSpecVerifyLogits(new[] { id }, new[] { length }, 2, new[] { next, next }, window, null));
        batched.CommitBatchedSpecVerify(new[] { id }, new[] { length }, new[] { 1 });
        rows.Add(window.AsSpan(0, vocab).ToArray());
        length++;
        next = ArgMax(rows[^1]);
        for (int t = 0; t < decodes; t++)
        {
            last = Forward(model, id, new[] { next }, length);
            rows.Add(last);
            next = ArgMax(last);
            length++;
        }
        return rows;
    }

    /// <summary>A fresh run of <paramref name="prefix"/> (in the same two chunks of its first <paramref name="promptLength"/>
    /// tokens and the rest in one), then the continuation at its end.</summary>
    private List<float[]> Reference(string family, int[] prefix, int promptLength, int decodes = 3)
    {
        using StandardDecoderModel reference = Create(family);
        RunPrompt(reference, "ref", prefix[..promptLength]);
        if (prefix.Length > promptLength) Forward(reference, "ref", prefix[promptLength..], promptLength);
        return Continuation(reference, "ref", prefix.Length, decodes);
    }

    private void AssertSameRows(List<float[]> expected, List<float[]> actual, string what)
    {
        Assert.Equal(expected.Count, actual.Count);
        int apart = 0;
        for (int i = 0; i < expected.Count; i++)
            if (!expected[i].AsSpan().SequenceEqual(actual[i])) apart++;
        _output.WriteLine($"{what}: {apart} of {expected.Count} rows apart");
        Assert.True(apart == 0, $"{what}: {apart} of {expected.Count} rows differ from a fresh run's");
    }

    private static MaterializeRequest Clone(string payload, string target, int tokens, IReadOnlyList<string> consumable)
        => new(MaterializeOp.Clone, payload, target, tokens, tokens, 0, consumable);

    /// <summary>
    /// Request "a": <paramref name="prompt"/>, checkpoint "cp1" at its end, ten more rows, checkpoint "cp2", then 300 rows and
    /// three decode steps past it, and its end state retained as "E". Both checkpoints share a's rows.
    /// </summary>
    private static (ulong Cp1, ulong Cp2) TwoCheckpointsAndAnEndState(StandardDecoderModel model, int[] prompt, int[] more, bool retain = true)
    {
        var cache = (IPrefixCacheModel)model;
        RunPrompt(model, "a", prompt);
        Assert.True(cache.TryCaptureCheckpoint("a", "cp1", mayShareRows: true, out _));
        Forward(model, "a", more, prompt.Length);
        Assert.True(cache.TryCaptureCheckpoint("a", "cp2", mayShareRows: true, out _));
        ulong cp1 = model.RowChecksumForTests("cp1"), cp2 = model.RowChecksumForTests("cp2");
        int length = prompt.Length + more.Length;
        int next = ArgMax(Forward(model, "a", Tokens(300, 9), length));
        length += 300;
        for (int t = 0; t < 3; t++)
        {
            next = ArgMax(Forward(model, "a", new[] { next }, length));
            length++;
        }
        Assert.Equal(cp1, model.RowChecksumForTests("cp1"));
        Assert.Equal(cp2, model.RowChecksumForTests("cp2"));
        if (retain) Assert.True(model.RetainSequenceCacheAs("a", "E"));
        return (cp1, cp2);
    }

    /// <summary>Adopting the shorter of two checkpoints over an end state's rows uses up the end state and the longer
    /// checkpoint (both reported), and its adopter continues as a fresh run; where only the end state is the adopter's to use
    /// up, the shorter checkpoint is copied and both stay whole.</summary>
    [Theory]
    [InlineData("qwen35")]
    [InlineData("qwen4exp")]
    [InlineData("mistral3")]
    public void AStealUsesUpTheLongerCheckpointsOverTheRowsOrCopiesWhereOneIsNotTheAdopters(string family)
    {
        int[] prompt = Tokens(40, 0), more = Tokens(10, 3);
        List<float[]> fromCp1 = Reference(family, prompt, prompt.Length);

        // The longer checkpoint is not the adopter's to use up: a copy, everything stays.
        var sink = new RecordingPayloadSink();
        using (StandardDecoderModel model = Create(family, sink))
        {
            var cache = (IPrefixCacheModel)model;
            var (cp1, cp2) = TwoCheckpointsAndAnEndState(model, prompt, more);
            IReadOnlyList<string> sharers = cache.RowSharers("cp1");
            Assert.Equal(new[] { "E", "cp2" }, sharers.OrderBy(k => k, StringComparer.Ordinal));
            int copies = model.SlotCopies;
            Assert.True(cache.TryMaterialize(Clone("cp1", "b", prompt.Length, new[] { "E" })));
            Assert.Equal(0, model.RowSteals);
            Assert.Equal(copies + 1, model.SlotCopies);
            Assert.Empty(sink.Reports);
            AssertSameRows(fromCp1, Continuation(model, "b", prompt.Length), $"{family}: a copy of the shorter checkpoint");
            Assert.Equal(cp1, model.RowChecksumForTests("cp1"));
            Assert.Equal(cp2, model.RowChecksumForTests("cp2"));
            model.CheckSlotShareInvariants();
        }

        // Both are the adopter's: the end state's slot is taken over, the longer checkpoint goes first.
        sink = new RecordingPayloadSink();
        using (StandardDecoderModel model = Create(family, sink))
        {
            var cache = (IPrefixCacheModel)model;
            var (cp1, _) = TwoCheckpointsAndAnEndState(model, prompt, more);
            Tensor[][] rows = model.SlotCachesForTests("E");
            Assert.True(cache.TryMaterialize(Clone("cp1", "b", prompt.Length, cache.RowSharers("cp1"))));
            Assert.Equal(1, model.RowSteals);
            Assert.Same(rows, model.SlotCachesForTests("b"));
            Assert.Equal(new[] { "E", "cp2" }, sink.Reports.Select(r => r.Key).OrderBy(k => k, StringComparer.Ordinal));
            Assert.All(sink.Reports, r => Assert.Equal(InvalidationReason.TrimmedByModel, r.Reason));
            Assert.Null(model.RetainedSharesRowsForTests("cp2"));
            Assert.True(model.RetainedSharesRowsForTests("cp1"));
            AssertSameRows(fromCp1, Continuation(model, "b", prompt.Length), $"{family}: the shorter checkpoint's steal");
            Assert.Equal(cp1, model.RowChecksumForTests("cp1"));
            Assert.Equal(0, model.SharedCheckpointsInvalidatedByWrite);
            model.CheckSlotShareInvariants();
        }
    }

    /// <summary>The slot holding two checkpoints' rows is released unretained: the longer checkpoint takes the rows over (a
    /// hand-over), and an adopter of the shorter one takes them over from it in turn, continuing as a fresh run; an adopter of
    /// the heir itself continues from the heir's length.</summary>
    [Theory]
    [InlineData("qwen35")]
    [InlineData("qwen4exp")]
    [InlineData("mistral3")]
    public void TheShorterCheckpointIsTakenOverFromTheCheckpointThatTookTheRowsOver(string family)
    {
        int[] prompt = Tokens(40, 0), more = Tokens(10, 3);
        List<float[]> fromCp1 = Reference(family, prompt, prompt.Length);
        List<float[]> fromCp2 = Reference(family, prompt.Concat(more).ToArray(), prompt.Length);

        var sink = new RecordingPayloadSink();
        using (StandardDecoderModel model = Create(family, sink))
        {
            var cache = (IPrefixCacheModel)model;
            var (cp1, cp2) = TwoCheckpointsAndAnEndState(model, prompt, more, retain: false);
            Tensor[][] rows = model.SlotCachesForTests("a");
            model.OnSequenceReleased("a");
            Assert.Equal(1, model.RowHandOvers);
            Assert.False(model.RetainedSharesRowsForTests("cp2"));   // the heir holds the rows now
            Assert.True(model.RetainedSharesRowsForTests("cp1"));
            Assert.Same(rows, model.SlotCachesForTests("cp2"));
            Assert.Equal(cp1, model.RowChecksumForTests("cp1"));
            Assert.Equal(cp2, model.RowChecksumForTests("cp2"));
            model.CheckSlotShareInvariants();

            Assert.Equal(new[] { "cp2" }, cache.RowSharers("cp1"));
            Assert.True(cache.TryMaterialize(Clone("cp1", "b", prompt.Length, cache.RowSharers("cp1"))));
            Assert.Equal(1, model.RowSteals);
            Assert.Same(rows, model.SlotCachesForTests("b"));
            Assert.Equal(new[] { ("cp2", InvalidationReason.TrimmedByModel) }, sink.Reports);
            AssertSameRows(fromCp1, Continuation(model, "b", prompt.Length), $"{family}: taken over from the heir");
            model.CheckSlotShareInvariants();
        }

        // The heir donated as an ordinary retained slot: its adopter continues from its length.
        using (StandardDecoderModel model = Create(family))
        {
            var cache = (IPrefixCacheModel)model;
            TwoCheckpointsAndAnEndState(model, prompt, more, retain: false);
            model.OnSequenceReleased("a");
            int length = prompt.Length + more.Length;
            Assert.True(cache.TryMaterialize(new MaterializeRequest(MaterializeOp.Donate, "cp2", "c", length, length)));
            AssertSameRows(fromCp2, Continuation(model, "c", length), $"{family}: the heir donated");
            Assert.True(model.RetainedSharesRowsForTests("cp1"));   // still over the rows, which c writes past only
            model.CheckSlotShareInvariants();
        }
    }

    /// <summary>Qwen3.8's QSA pools blocks of four cells into each block's last cell: a checkpoint inside a block leaves that
    /// block's pooled cell to its owner, which completes the block with its own rows; the adopter that takes the slot over at
    /// the checkpoint completes the block with other rows and continues as a fresh run (the incomplete block's pooled cell is
    /// never read before it is written again).</summary>
    [Theory]
    [InlineData(41)]
    [InlineData(42)]
    [InlineData(43)]
    public void AQsaCheckpointInsideABlockIsTakenOverAsAFreshRun(int promptLength)
    {
        int[] prompt = Tokens(promptLength, 0);
        List<float[]> expected = Reference("qwen4exp", prompt, prompt.Length, decodes: 8);
        var sink = new RecordingPayloadSink();
        using StandardDecoderModel model = Create("qwen4exp", sink);
        var cache = (IPrefixCacheModel)model;
        RunPrompt(model, "a", prompt);
        Assert.True(cache.TryCaptureCheckpoint("a", "cp", mayShareRows: true, out _));
        ulong sum = model.RowChecksumForTests("cp");
        int length = prompt.Length;
        int next = ArgMax(Forward(model, "a", Tokens(300, 9), length));
        length += 300;
        for (int t = 0; t < 3; t++)
        {
            next = ArgMax(Forward(model, "a", new[] { next }, length));
            length++;
        }
        Assert.Equal(sum, model.RowChecksumForTests("cp"));
        Assert.True(model.RetainSequenceCacheAs("a", "E"));
        Assert.True(cache.TryMaterialize(Clone("cp", "b", prompt.Length, cache.RowSharers("cp"))));
        Assert.Equal(1, model.RowSteals);
        AssertSameRows(expected, Continuation(model, "b", prompt.Length, decodes: 8), $"qwen4exp, a checkpoint of {promptLength} rows");
        Assert.Equal(0, model.SharedCheckpointsInvalidatedByWrite);
        model.CheckSlotShareInvariants();
    }

    /// <summary>
    /// Two turns of one conversation adopt the same prompt-end checkpoint at once (a retry sent twice): one takes the end
    /// state's slot over, the other copies the checkpoint's rows while it runs (or takes the next end state over), and both
    /// answer as the same turns do with copied checkpoints; the tree and the slots stay consistent.
    /// </summary>
    [Fact]
    public async Task TwoTurnsAdoptingOneCheckpointAtOnceAnswerAsWithCopiedCheckpoints()
    {
        int[] a = Tokens(300, 1);
        int[] b = null;
        (List<int[]> expected, StandardDecoderModel copies) = await Turns(slotBound: false);
        copies.Dispose();
        (List<int[]> answers, StandardDecoderModel model) = await Turns(slotBound: true);
        using (model)
        {
            _output.WriteLine($"steals {model.RowSteals}, shared {model.SharedCheckpoints}, copies {model.SlotCopies}, hand-overs {model.RowHandOvers}");
            for (int i = 0; i < expected.Count; i++)
                Assert.True(expected[i].AsSpan().SequenceEqual(answers[i]),
                    $"turn {i} answered [{string.Join(",", answers[i])}] where copied checkpoints gave [{string.Join(",", expected[i])}]");
            Assert.True(model.RowSteals >= 1, "no turn took a slot over");
            Assert.Equal(0, model.SharedCheckpointsInvalidatedByWrite);
            model.CheckSlotShareInvariants();
        }

        async Task<(List<int[]>, StandardDecoderModel)> Turns(bool slotBound)
        {
            StandardDecoderModel model = Create("qwen35", slotBound: slotBound);
            model.RetainCopyMaxBytesForTests = 0;
            using var engine = new InferenceEngine(model, new SchedulerConfig
            {
                MaxNumBatchedTokens = 512,
                MaxNumRunningSequences = 4,
                NumBlocks = 128,
                BlockSize = 16,
                EnablePrefixCaching = true,
                StopRepetition = false,
            }, NullLogger.Instance);
            var answers = new List<int[]>();
            answers.Add(await Serve(engine, new SequenceState("A", a, 1, 16, SamplingConfig.Greedy, cacheScope: "conversation")));
            b ??= a.Concat(new[] { 3 + (answers[0][0] + 1) % 250, 18 }).Concat(Tokens(60, 2)).ToArray();
            answers.Add(await Serve(engine, new SequenceState("B", b, 1, 16, SamplingConfig.Greedy, cacheScope: "conversation")));
            var c1 = new SequenceState("C1", b, 12, 16, SamplingConfig.Greedy, cacheScope: "conversation");
            var c2 = new SequenceState("C2", b, 12, 16, SamplingConfig.Greedy, cacheScope: "conversation");
            Task<int[]> t1 = Serve(engine, c1), t2 = Serve(engine, c2);
            answers.Add(await t1);
            answers.Add(await t2);
            _output.WriteLine($"{(slotBound ? "shared" : "copies")}: reused {c1.PrefixCacheReusedTokens}, {c2.PrefixCacheReusedTokens}");
            PrefixTree tree = engine.RadixCache!.Tree;
            Assert.Empty(PrefixTreeInvariantChecker.CheckOnce(tree));
            var held = new HashSet<string>(model.RetainedPayloadKeys, StringComparer.Ordinal);
            Assert.All(tree.PayloadKeys, key => Assert.Contains(key, held));
            return (answers, model);
        }
    }

    /// <summary>Once a branch (a prompt of its own past the checkpoint) or the conversation's next turn (the end state donated)
    /// has written the rows past a checkpoint, an adopter of the checkpoint copies it and that branch's end state stays whole;
    /// after a retry of the checkpoint's prompt the next retry takes the slot over again.</summary>
    [Theory]
    [InlineData("qwen35")]
    [InlineData("mistral3")]
    public void AnAdopterCopiesACheckpointWhoseRowsPastItABranchOrTheNextTurnWrote(string family)
    {
        int[] prompt = Tokens(40, 0);
        List<float[]> expected = Reference(family, prompt, prompt.Length);
        foreach (string taker in new[] { "branch", "next turn", "retry" })
        {
            var sink = new RecordingPayloadSink();
            using StandardDecoderModel model = Create(family, sink);
            var cache = (IPrefixCacheModel)model;
            RunPrompt(model, "a", prompt);
            Assert.True(cache.TryCaptureCheckpoint("a", "cp", mayShareRows: true, out _));
            Forward(model, "a", Tokens(20, 9), prompt.Length);
            Assert.True(model.RetainSequenceCacheAs("a", "E"));
            int length = prompt.Length + 20;
            if (taker == "next turn")
            {
                // The end state donated to the conversation's next turn, which runs on past it.
                Assert.True(cache.TryMaterialize(new MaterializeRequest(MaterializeOp.Donate, "E", "t", length, length)));
                Forward(model, "t", Tokens(30, 4), length);
            }
            else
            {
                Assert.True(cache.TryMaterialize(new MaterializeRequest(MaterializeOp.Clone, "cp", "t", prompt.Length, prompt.Length, 0,
                    cache.RowSharers("cp"), EndsPrompt: taker == "retry")));
                Assert.Equal(1, model.RowSteals);
                Forward(model, "t", Tokens(30, 4), prompt.Length);
            }
            Assert.True(model.RetainSequenceCacheAs("t", "T"));
            ulong branch = model.RowChecksumForTests("T");
            int reports = sink.Reports.Count;
            IReadOnlyList<string> sharers = cache.RowSharers("cp");
            Assert.True(cache.TryMaterialize(new MaterializeRequest(MaterializeOp.Clone, "cp", "b", prompt.Length, prompt.Length, 0,
                sharers, EndsPrompt: true)));
            if (taker == "retry")
            {
                Assert.Equal(new[] { "T" }, sharers);
                Assert.Equal(2, model.RowSteals);
            }
            else
            {
                Assert.Empty(sharers);
                Assert.Equal(taker == "branch" ? 1 : 0, model.RowSteals);
                Assert.Equal(reports, sink.Reports.Count);
                Assert.Equal(branch, model.RowChecksumForTests("T"));   // the branch's end state is whole
            }
            AssertSameRows(expected, Continuation(model, "b", prompt.Length), $"{family}: after the {taker}");
            Assert.Equal(0, model.SharedCheckpointsInvalidatedByWrite);
            model.CheckSlotShareInvariants();
        }
    }

    /// <summary>
    /// Two branches of one conversation fork at a prompt-end checkpoint and both stay alive, their turns interleaved (a client
    /// that forks a transcript into two threads, or a branching chat UI): each branch's next turn resumes its own end state, as
    /// with copied checkpoints - a branch's first turn does not take the other branch's end state over.
    /// </summary>
    [Fact]
    public async Task TwoLiveBranchesOfOneConversationEachResumeTheirOwnEndState()
    {
        int[] a = Tokens(300, 1);
        (List<int> expected, StandardDecoderModel copies) = await Turns(slotBound: false);
        copies.Dispose();
        (List<int> reused, StandardDecoderModel model) = await Turns(slotBound: true);
        using (model)
        {
            _output.WriteLine($"reused: shared [{string.Join(", ", reused)}], copies [{string.Join(", ", expected)}]; steals {model.RowSteals}, copies {model.SlotCopies}");
            Assert.Equal(expected, reused);
            model.CheckSlotShareInvariants();
        }

        async Task<(List<int>, StandardDecoderModel)> Turns(bool slotBound)
        {
            StandardDecoderModel model = Create("qwen35", slotBound: slotBound);
            model.RetainCopyMaxBytesForTests = 0;
            using var engine = new InferenceEngine(model, new SchedulerConfig
            {
                MaxNumBatchedTokens = 512,
                MaxNumRunningSequences = 4,
                NumBlocks = 256,
                BlockSize = 16,
                EnablePrefixCaching = true,
                StopRepetition = false,
            }, NullLogger.Instance);
            var reused = new List<int>();
            async Task<int[]> Turn(string id, int[] prompt, int max)
            {
                var sequence = new SequenceState(id, prompt, max, 16, SamplingConfig.Greedy, cacheScope: "conversation");
                int[] answer = await Serve(engine, sequence);
                reused.Add(sequence.PrefixCacheReusedTokens);
                return answer;
            }
            int[] first = await Turn("A", a, 1);
            // Two branches diverge right after A's prompt (each with an answer other than A's), each with a long turn of its own.
            int[] f1 = a.Concat(new[] { 3 + (first[0] + 1) % 250, 18 }).Concat(Tokens(120, 2)).ToArray();
            int[] f2 = a.Concat(new[] { 3 + (first[0] + 2) % 250, 18 }).Concat(Tokens(120, 4)).ToArray();
            int[] r1 = await Turn("F1", f1, 4);
            int[] r2 = await Turn("F2", f2, 4);
            // Each branch's next turn continues its own transcript: its answer and a new user turn.
            await Turn("F1b", f1.Concat(r1).Concat(Tokens(40, 6)).ToArray(), 4);
            await Turn("F2b", f2.Concat(r2).Concat(Tokens(40, 7)).ToArray(), 4);
            return (reused, model);
        }
    }

    private static async Task<int[]> Serve(InferenceEngine engine, SequenceState sequence)
    {
        var handle = engine.SubmitRequest(sequence);
        var tokens = new List<int>();
        await foreach (int token in handle.Tokens.ReadAllAsync()) tokens.Add(token);
        await handle.Completion.WaitAsync(TimeSpan.FromMinutes(2));
        await engine.WhenStepSettledAsync().WaitAsync(TimeSpan.FromMinutes(1));
        return tokens.ToArray();
    }
}
