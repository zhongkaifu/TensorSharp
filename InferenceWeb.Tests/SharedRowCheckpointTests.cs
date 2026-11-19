// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Issue #376 (T1-7): where recorded steps bind slots (the CUDA executor), a request's automatic prompt-end checkpoint keeps only
// its recurrent states and shares the request's cache rows below its length; a conversation's next turn that adopts it takes
// the end state's slot over (a steal) instead of copying 3.3 GB at 47K tokens. Whatever happens to the slot holding the rows -
// retained and taken over, copied from, evicted, released, grown - an adopter of the checkpoint continues exactly as a fresh
// run of the same tokens, and the checkpoint's rows never change while it lasts (checksums); a write below them invalidates it
// (reported to the prefix cache), never corrupts it. The managed CPU executor runs the policies through the model's seam
// (SlotBoundPoliciesForTests), with the share invariants checked before every step.
using Microsoft.Extensions.Logging.Abstractions;

using InferenceWeb.Tests.PrefixCache.Fakes;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Scheduling.PrefixCache;
using TensorSharp.Runtime.Speculative;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

public sealed class SharedRowCheckpointTests : IDisposable
{
    private const long GB = 1L << 30;
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-shared-rows-" + Guid.NewGuid().ToString("N"));
    private readonly ITestOutputHelper _output;

    public SharedRowCheckpointTests(ITestOutputHelper output)
    {
        _output = output;
        Directory.CreateDirectory(_dir);
    }

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    private string Fixture(string family) => family switch
    {
        "qwen35" => new Qwen35DenseSyntheticModelBuilder { Layers = 8, UntiedOutput = true }.Write(Path.Combine(_dir, "q35-dense.gguf")),
        "qwen35-nextn" => new Qwen35DenseSyntheticModelBuilder { Layers = 8, UntiedOutput = true, NextN = true }
            .Write(Path.Combine(_dir, "q35-dense-nextn.gguf")),
        "qwen35moe" => new Qwen35MoeSyntheticModelBuilder { Layers = 8, UniformWeightType = SyntheticGguf.GgmlType.BF16 }
            .Write(Path.Combine(_dir, "q35-moe.gguf")),
        "qwen4exp" => Qwen4ExpSyntheticModelBuilder.Write(Path.Combine(_dir, "q4e.gguf"), indexerTopK: 8, qsa: true),
        "mistral3" => new DenseDecoderSyntheticModelBuilder { Architecture = "mistral3" }.Write(Path.Combine(_dir, "mistral3.gguf")),
        _ => throw new ArgumentException(family),
    };

    /// <summary>A model of <paramref name="family"/> on the managed CPU executor with the slot-bound policies of the CUDA
    /// executor (or on direct CUDA, which has them) and the share invariants checked before every step;
    /// <paramref name="slotBound"/> false: today's policies on the managed executor, copied checkpoints on direct CUDA.</summary>
    private StandardDecoderModel Create(string family, RecordingPayloadSink sink = null, bool slotBound = true, BackendType backend = BackendType.Cpu)
    {
        string path = File.Exists(Path.Combine(_dir, family + ".path")) ? File.ReadAllText(Path.Combine(_dir, family + ".path")) : null;
        if (path == null)
        {
            path = Fixture(family);
            File.WriteAllText(Path.Combine(_dir, family + ".path"), path);
        }
        using IDisposable dtype = KvCacheDtypeConfig.ForCurrentFlowInTests(backend == BackendType.Cpu ? KvCacheDtype.F32 : KvCacheDtype.F16);
        var model = (StandardDecoderModel)ModelBase.Create(path, backend);
        if (backend == BackendType.Cpu) model.SlotBoundPoliciesForTests = slotBound;
        else model.SharedRowsDisabledForTests = !slotBound;
        model.SlotInvariantChecksForTests = true;
        model.AttachPrefixCache(sink ?? new RecordingPayloadSink());
        return model;
    }

    private static int[] Tokens(int count, int salt) => Enumerable.Range(0, count).Select(i => 3 + (i * 37 + i / 7 + salt * 11) % 250).ToArray();

    private static readonly int[] Prompt = Tokens(40, 0);
    private static readonly int[] Next = Tokens(3, 5);
    private const int Appended = 300;   // more than 256 rows past the checkpoint, inside a fresh 512-cell slot
    private const int Grows = 600;      // past a fresh 512-cell slot: it grows

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

    /// <summary>The prompt on a new slot of <paramref name="id"/>, in two chunks.</summary>
    private static void RunPrompt(StandardDecoderModel model, string id, int reserve = 64)
    {
        Assert.Equal(0, ((IUnifiedBatchModel)model).EnsureSequenceSlot(id, reserve));
        Forward(model, id, Prompt[..20], 0);
        Forward(model, id, Prompt[20..], 20);
    }

    /// <summary>What a request continuing at <paramref name="at"/> computes: three tokens, a two-row verify window keeping one
    /// row, two greedy decode steps; every row's logits.</summary>
    private static List<float[]> Continuation(StandardDecoderModel model, string id, int at, int decodes = 2)
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

    private void AssertSameRows(List<float[]> expected, List<float[]> actual, string what)
    {
        Assert.Equal(expected.Count, actual.Count);
        int apart = 0;
        for (int i = 0; i < expected.Count; i++)
            if (!expected[i].AsSpan().SequenceEqual(actual[i])) apart++;
        _output.WriteLine($"{what}: {apart} of {expected.Count} rows apart");
        Assert.True(apart == 0, $"{what}: {apart} of {expected.Count} rows differ from a fresh run's");
    }

    /// <summary>A fresh run: the prompt, then the continuation at its end.</summary>
    private List<float[]> Reference(string family, BackendType backend = BackendType.Cpu, int decodes = 2)
    {
        using StandardDecoderModel reference = Create(family, backend: backend);
        RunPrompt(reference, "ref");
        return Continuation(reference, "ref", Prompt.Length, decodes);
    }

    /// <summary>Request "a": the prompt, its checkpoint "cp" sharing its rows, then <see cref="Appended"/> rows and three decode
    /// steps past it. The checkpoint's checksum when it was taken.</summary>
    private static ulong ShareAndAppend(StandardDecoderModel model, int reserve = 64, int appended = Appended, int decodes = 3,
        bool checksums = true)
    {
        RunPrompt(model, "a", reserve);
        var cache = (IPrefixCacheModel)model;
        Assert.True(cache.TryCaptureCheckpoint("a", "cp", mayShareRows: true, out PayloadFootprint footprint));
        Assert.Equal(Prompt.Length, footprint.Tokens);
        Assert.Same(model.SlotCachesForTests("a"), model.SlotCachesForTests("cp"));
        Assert.True(model.RetainedSharesRowsForTests("cp"));
        ulong sum = checksums ? model.RowChecksumForTests("cp") : 0;
        int length = Prompt.Length;
        int next = ArgMax(Forward(model, "a", Tokens(appended, 9), length));
        length += appended;
        for (int t = 0; t < decodes; t++)
        {
            next = ArgMax(Forward(model, "a", new[] { next }, length));
            length++;
        }
        if (checksums) Assert.Equal(sum, model.RowChecksumForTests("cp"));   // the owner wrote past it only
        Assert.Equal(0, model.SharedCheckpointsInvalidatedByWrite);
        return sum;
    }

    private static MaterializeRequest Clone(string payload, string target, IReadOnlyList<string> consumable)
        => new(MaterializeOp.Clone, payload, target, Prompt.Length, Prompt.Length, 0, consumable);

    public enum Fate { Steal, StealTwice, Copy, Evicted, Released, Grown, GrownRolling }

    /// <summary>Every fate of the slot holding a checkpoint's rows, for a recurrent dense family (Qwen3.5: GDN plus attention),
    /// a MoE one, Qwen3.8 Flash Next's (QSA past its budget, PLE, HC) and an attention-only one (Mistral 3): the checkpoint's
    /// adopter continues exactly as a fresh run, and its rows never changed.</summary>
    [Theory]
    [MemberData(nameof(Cases))]
    public void AnAdopterOfASharedCheckpointContinuesAsAFreshRunWhateverHappenedToItsRowsSlot(string family, Fate fate)
    {
        List<float[]> expected = Reference(family);
        var sink = new RecordingPayloadSink();
        using StandardDecoderModel model = Create(family, sink);
        var cache = (IPrefixCacheModel)model;
        if (fate == Fate.GrownRolling) model.DeviceRoomForTests = () => (64 * GB, 128 * GB);   // slots sized to room: rolling growth
        ulong sum = ShareAndAppend(model, appended: fate switch { Fate.GrownRolling => 2100, Fate.Grown => Grows, _ => Appended });
        Tensor[][] rows = model.SlotCachesForTests("a");

        switch (fate)
        {
            case Fate.Steal:
            case Fate.StealTwice:
            case Fate.Grown:
            case Fate.GrownRolling:
            {
                if (fate is Fate.Grown or Fate.GrownRolling)
                {
                    Assert.True(model.RowRepoints > 0, "the owner did not grow (or its checkpoint was not taken along)");
                    Assert.Equal(fate == Fate.GrownRolling ? 1 : 0, Math.Sign(model.RollingGrowths));
                }
                else Assert.Equal(0, model.RowRepoints);
                Assert.True(model.RetainSequenceCacheAs("a", "E"));
                IReadOnlyList<string> sharers = cache.RowSharers("cp");
                Assert.Equal(new[] { "E" }, sharers);
                int copies = model.SlotCopies;
                Assert.True(cache.TryMaterialize(Clone("cp", "b", sharers)));
                Assert.Equal(1, model.RowSteals);
                Assert.Equal(copies, model.SlotCopies);   // nothing copied
                Assert.Same(rows, model.SlotCachesForTests("b"));
                Assert.Contains(("E", InvalidationReason.TrimmedByModel), sink.Reports);
                Assert.True(model.RetainedSharesRowsForTests("cp"));   // the checkpoint stays, now over the adopter's rows
                Assert.Equal(sum, model.RowChecksumForTests("b"));     // the adopter starts at the checkpoint's very state
                AssertSameRows(expected, Continuation(model, "b", Prompt.Length), $"{family} {fate}");
                Assert.Equal(sum, model.RowChecksumForTests("cp"));
                if (fate == Fate.StealTwice)
                {
                    // The adopter's end state is taken over again by the next adopter of the same checkpoint.
                    Assert.True(model.RetainSequenceCacheAs("b", "E2"));
                    Assert.True(cache.TryMaterialize(Clone("cp", "c", cache.RowSharers("cp"))));
                    Assert.Equal(2, model.RowSteals);
                    Assert.Same(rows, model.SlotCachesForTests("c"));
                    AssertSameRows(expected, Continuation(model, "c", Prompt.Length), $"{family} {fate} (second)");
                }
                break;
            }
            case Fate.Copy:
            {
                // Not the request's to consume (another conversation's adopter): a copy, and the end state stays.
                Assert.True(model.RetainSequenceCacheAs("a", "E"));
                ulong owner = model.RowChecksumForTests("E");
                int copies = model.SlotCopies;
                Assert.True(cache.TryMaterialize(Clone("cp", "b", null)));
                Assert.Equal(0, model.RowSteals);
                Assert.Equal(copies + 1, model.SlotCopies);
                Assert.NotSame(rows, model.SlotCachesForTests("b"));
                Assert.Same(rows, model.SlotCachesForTests("E"));
                AssertSameRows(expected, Continuation(model, "b", Prompt.Length), $"{family} {fate}");
                Assert.Equal(sum, model.RowChecksumForTests("cp"));
                Assert.Equal(owner, model.RowChecksumForTests("E"));
                break;
            }
            case Fate.Evicted:
            case Fate.Released:
            {
                // The slot holding the rows goes (evicted from the prefix cache, or its request released unretained): the
                // checkpoint takes the rows over, and its adopter takes them in turn.
                if (fate == Fate.Evicted)
                {
                    Assert.True(model.RetainSequenceCacheAs("a", "E"));
                    model.DiscardRetainedCaches(new[] { "E" }, ReleaseReason.Evicted);
                }
                else model.OnSequenceReleased("a");
                Assert.Equal(1, model.RowHandOvers);
                Assert.False(model.RetainedSharesRowsForTests("cp"));
                Assert.Same(rows, model.SlotCachesForTests("cp"));
                Assert.Equal(0, model.SlotPoolForTests.Pooled);   // the rows went to the checkpoint, not the pool
                Assert.Equal(sum, model.RowChecksumForTests("cp"));
                Assert.True(cache.TryMaterialize(new MaterializeRequest(MaterializeOp.Donate, "cp", "b", Prompt.Length, Prompt.Length)));
                Assert.Same(rows, model.SlotCachesForTests("b"));
                AssertSameRows(expected, Continuation(model, "b", Prompt.Length), $"{family} {fate}");
                break;
            }
        }
        Assert.Equal(0, model.SharedCheckpointsInvalidatedByWrite);
        model.CheckSlotShareInvariants();
    }

    public static IEnumerable<object[]> Cases()
    {
        foreach (string family in new[] { "qwen35", "qwen35moe", "qwen4exp", "mistral3" })
            foreach (Fate fate in Enum.GetValues<Fate>())
                yield return new object[] { family, fate };
    }

    /// <summary>A shared checkpoint is charged its states only, a copy where it may not share (a public or explicit boundary,
    /// or executors whose steps do not bind slots), and at most four share one slot's rows (the oldest goes, reported).</summary>
    [Fact]
    public void ACheckpointSharesItsRowsOnlyWhereItMayAndAtMostFourShareASlot()
    {
        var sink = new RecordingPayloadSink();
        using StandardDecoderModel model = Create("qwen35", sink);
        var cache = (IPrefixCacheModel)model;
        RunPrompt(model, "a");
        Assert.True(cache.TryCaptureCheckpoint("a", "cp0", mayShareRows: true, out PayloadFootprint shared));
        Assert.True(cache.TryCaptureCheckpoint("a", "copy", mayShareRows: false, out PayloadFootprint copied));
        Assert.Equal(1, model.SharedCheckpoints);
        Assert.False(model.RetainedSharesRowsForTests("copy"));
        Assert.NotSame(model.SlotCachesForTests("a"), model.SlotCachesForTests("copy"));
        Assert.True(shared.Bytes.HostKv > 0);
        Assert.True(shared.Bytes.HostKv < copied.Bytes.HostKv / 2, $"a shared checkpoint charged {shared.Bytes.HostKv} B against a copy's {copied.Bytes.HostKv}");

        int length = Prompt.Length;
        for (int i = 1; i <= 4; i++)
        {
            Forward(model, "a", new[] { 7 + i }, length++);
            Assert.True(cache.TryCaptureCheckpoint("a", $"cp{i}", mayShareRows: true, out _));
        }
        Assert.Equal(new[] { ("cp0", InvalidationReason.ModelBudgetEviction) }, sink.Reports);
        Assert.Null(model.RetainedSharesRowsForTests("cp0"));
        for (int i = 1; i <= 4; i++) Assert.True(model.RetainedSharesRowsForTests($"cp{i}"));

        using StandardDecoderModel elsewhere = Create("qwen35", slotBound: false);
        RunPrompt(elsewhere, "a");
        Assert.True(((IPrefixCacheModel)elsewhere).TryCaptureCheckpoint("a", "cp", mayShareRows: true, out _));
        Assert.False(elsewhere.RetainedSharesRowsForTests("cp"));
        Assert.Equal(0, elsewhere.SharedCheckpoints);
    }

    /// <summary>A checkpoint takes the states made ahead while the engine was idle (instead of allocating them in its request's
    /// tail), its adopter continues as a fresh run, and making device room gives a spare set back first.</summary>
    [Fact]
    public void ACheckpointTakesTheSpareStatesMadeAheadAndMakingRoomGivesThemBackFirst()
    {
        List<float[]> expected = Reference("qwen35");
        using StandardDecoderModel model = Create("qwen35");
        var cache = (IPrefixCacheModel)model;
        RunPrompt(model, "a");
        model.MakeSpareCheckpointStatesForTests();
        Assert.Equal(1, model.SpareCheckpointStatesMade);
        var spare = model.SpareCheckpointStatesForTests;
        Assert.NotNull(spare);
        Assert.True(cache.TryCaptureCheckpoint("a", "cp", mayShareRows: true, out _));
        Assert.Same(spare, model.RetainedStatesForTests("cp"));
        Assert.Null(model.SpareCheckpointStatesForTests);
        Assert.True(model.RetainSequenceCacheAs("a", "E"));
        Assert.True(cache.TryMaterialize(Clone("cp", "b", cache.RowSharers("cp"))));
        AssertSameRows(expected, Continuation(model, "b", Prompt.Length), "a checkpoint of spare states");

        model.MakeSpareCheckpointStatesForTests();
        Assert.NotNull(model.SpareCheckpointStatesForTests);
        model.DeviceRoomForTests = () => (0, 16 * GB);
        model.RetainedPayloadBytesForTests = _ => 0;
        model.EnsureRecordingRoomForTests(GB);
        Assert.Null(model.SpareCheckpointStatesForTests);
        Assert.Equal(1, model.DeviceRoomStages[2]);
    }

    /// <summary>A write below a checkpoint (its rows' slot truncated to zero, or rewound below it) invalidates it first, reported
    /// to the prefix cache: its rows are never rewritten under it.</summary>
    [Theory]
    [InlineData("qwen35", 0)]
    [InlineData("mistral3", 0)]
    [InlineData("mistral3", 36)]   // an attention-only family rewinds below the checkpoint's 40 rows
    public void AWriteBelowACheckpointInvalidatesItFirst(string family, int truncateTo)
    {
        var sink = new RecordingPayloadSink();
        using StandardDecoderModel model = Create(family, sink);
        RunPrompt(model, "a");
        Assert.True(((IPrefixCacheModel)model).TryCaptureCheckpoint("a", "cp", mayShareRows: true, out _));
        Forward(model, "a", new[] { 9 }, Prompt.Length);
        model.BindSequenceCache("a");
        Assert.True(model.TryTruncateKVCache(truncateTo));
        model.RestorePrimaryCache();
        Assert.Equal(1, model.SharedCheckpointsInvalidatedByWrite);
        Assert.Equal(new[] { ("cp", InvalidationReason.TrimmedByModel) }, sink.Reports);
        Assert.Null(model.RetainedSharesRowsForTests("cp"));
        Forward(model, "a", Tokens(3, 4), truncateTo);   // the owner runs on, alone
        model.CheckSlotShareInvariants();
    }

    /// <summary>A checkpoint evicted from the prefix cache frees its states only: it is never pooled, and its rows' slot runs on
    /// with its rows untouched; the allocator's reclaimer releases checkpoints' states before their rows' slot, then the slot
    /// whole.</summary>
    [Fact]
    public void ACheckpointIsNeverPooledAndTheReclaimerFreesItsStatesBeforeItsRowsSlot()
    {
        var sink = new RecordingPayloadSink();
        using StandardDecoderModel model = Create("qwen35", sink);
        ShareAndAppend(model);
        ulong owner = model.RowChecksumForTests("a");
        int pooled = model.SlotPoolForTests.Pooled;
        model.DiscardRetainedCaches(new[] { "cp" }, ReleaseReason.Evicted);
        Assert.Equal(pooled, model.SlotPoolForTests.Pooled);
        Assert.Null(model.RetainedSharesRowsForTests("cp"));
        Assert.Equal(owner, model.RowChecksumForTests("a"));
        Forward(model, "a", new[] { 5 }, Prompt.Length + Appended + 3);

        Assert.True(((IPrefixCacheModel)model).TryCaptureCheckpoint("a", "cp2", mayShareRows: true, out _));
        Assert.True(model.RetainSequenceCacheAs("a", "E"));
        RunPrompt(model, "run");
        Assert.True(model.ReclaimForAllocator(1L << 40) > 0);
        Assert.Equal(new[] { ("cp2", InvalidationReason.NativeSlotReclaimed), ("E", InvalidationReason.NativeSlotReclaimed) }, sink.Reports);
        Assert.Empty(model.RetainedPayloadKeys);
        Forward(model, "run", new[] { 5 }, Prompt.Length);   // the freed slots' steps are forgotten at this entry
        model.CheckSlotShareInvariants();
    }

    /// <summary>While a request runs on a checkpoint's rows, another request adopting the checkpoint copies it (there is no end
    /// state to take over), and the running one is untouched; without device room for the copy the model says so, so the
    /// request waits instead of prefilling.</summary>
    [Fact]
    public void ACheckpointWhoseRowsARequestRunsOnIsCopiedOrTheAdopterWaitsForRoom()
    {
        List<float[]> expected = Reference("qwen35");
        List<float[]> ownerExpected;
        using (StandardDecoderModel alone = Create("qwen35"))
        {
            ShareAndAppend(alone);
            ownerExpected = Continuation(alone, "a", Prompt.Length + Appended + 3);
        }
        using StandardDecoderModel model = Create("qwen35");
        var cache = (IPrefixCacheModel)model;
        ShareAndAppend(model);
        Assert.Null(cache.RowSharers("cp"));   // the rows' slot runs a request

        model.DeviceRoomForTests = () => (0, 16 * GB);
        Assert.False(cache.TryMaterialize(Clone("cp", "b", null)));
        Assert.True(cache.LastMaterializeWantedRoom);
        model.DeviceRoomForTests = null;

        Assert.True(cache.TryMaterialize(Clone("cp", "b", null)));
        Assert.False(cache.LastMaterializeWantedRoom);
        Assert.NotSame(model.SlotCachesForTests("a"), model.SlotCachesForTests("b"));
        AssertSameRows(expected, Continuation(model, "b", Prompt.Length), "the copy's adopter");
        AssertSameRows(ownerExpected, Continuation(model, "a", Prompt.Length + Appended + 3), "the running owner");

        // A rebind of the checkpoint (the holder lifecycle's donation) copies too: a checkpoint never runs a request.
        Assert.True(model.TryRebindRetainedCache("cp", "c"));
        Assert.NotSame(model.SlotCachesForTests("a"), model.SlotCachesForTests("c"));
        Assert.Null(model.RetainedSharesRowsForTests("cp"));   // donated: used up
        AssertSameRows(expected, Continuation(model, "c", Prompt.Length), "the rebind's copy");
    }

    /// <summary>A request on a taken-over slot that moves into a larger pooled slot for its reservation (instead of growing its
    /// own) takes the checkpoints over its rows along: the copy holds them, and the slot it left goes back to the pool.</summary>
    [Fact]
    public void ARequestMovedIntoAPooledSlotTakesTheCheckpointsOverItsRowsAlong()
    {
        List<float[]> expected = Reference("qwen35");
        using StandardDecoderModel model = Create("qwen35");
        var cache = (IPrefixCacheModel)model;
        ShareAndAppend(model);
        ulong sum = model.RowChecksumForTests("cp");
        Assert.True(model.RetainSequenceCacheAs("a", "E"));
        Assert.Equal(0, ((IUnifiedBatchModel)model).EnsureSequenceSlot("big", 1500));   // a 1,536-cell slot, then pooled
        model.OnSequenceReleased("big");
        Assert.True(cache.TryMaterialize(Clone("cp", "b", cache.RowSharers("cp"))));
        Tensor[][] taken = model.SlotCachesForTests("b");
        Assert.Equal(Prompt.Length, ((IUnifiedBatchModel)model).EnsureSequenceSlot("b", 1000));
        Assert.Equal(1, model.PooledSlotMoves);
        Assert.Equal(1, model.RowRepoints);
        Assert.NotSame(taken, model.SlotCachesForTests("b"));
        Assert.Same(model.SlotCachesForTests("b"), model.SlotCachesForTests("cp"));
        Assert.Equal(sum, model.RowChecksumForTests("cp"));
        Assert.Equal(1, model.SlotPoolForTests.Pooled);   // the slot it left, no checkpoint over it
        AssertSameRows(expected, Continuation(model, "b", Prompt.Length), "the moved request");
        Assert.Equal(sum, model.RowChecksumForTests("cp"));
    }

    /// <summary>A short finished slot retained as a copy (into a pooled slot) takes its checkpoints along, so the request's own
    /// slot still goes back to the pool, with its recordings, for the next request.</summary>
    [Fact]
    public void AShortSlotRetainedAsACopyTakesItsCheckpointsAlong()
    {
        List<float[]> expected = Reference("qwen35");
        using StandardDecoderModel model = Create("qwen35");
        var cache = (IPrefixCacheModel)model;
        RunPrompt(model, "a");
        RunPrompt(model, "pooled");
        model.OnSequenceReleased("pooled");
        Assert.Equal(1, model.SlotPoolForTests.Pooled);
        Assert.True(cache.TryCaptureCheckpoint("a", "cp", mayShareRows: true, out _));
        Forward(model, "a", new[] { 9 }, Prompt.Length);
        Tensor[][] own = model.SlotCachesForTests("a");
        Assert.True(cache.TryCaptureDonate("a", "E", Prompt.Length + 1, out _));
        Assert.True(model.LastRetentionCopied);
        Assert.Same(model.SlotCachesForTests("E"), model.SlotCachesForTests("cp"));
        model.OnSequenceReleased("a");
        Assert.Equal(1, model.SlotPoolForTests.Pooled);   // the request's own slot, back in the pool
        Assert.Equal(0, model.RowHandOvers);
        Assert.NotSame(own, model.SlotCachesForTests("cp"));
        Assert.True(cache.TryMaterialize(Clone("cp", "b", cache.RowSharers("cp"))));
        Assert.Equal(1, model.RowSteals);
        AssertSameRows(expected, Continuation(model, "b", Prompt.Length), "the copy's checkpoint");
    }

    /// <summary>
    /// The issue's reproducer behind an engine, small: A (a long prompt, one token), B (A's history, a diverging turn, a new
    /// user turn: it adopts A's prompt-end checkpoint), C (B's prompt again, decoding: it adopts B's prompt-end checkpoint). Each
    /// adoption takes the end state's slot over: no slot is copied, the invariants hold at every step, no checkpoint is
    /// invalidated by a write, and every answer - speculating over the taken-over slots, with n-gram or MTP drafts - equals
    /// plain decoding's with copied checkpoints (the same turns, chunks and reuse).
    /// </summary>
    [Theory]
    [InlineData("qwen35", null)]
    [InlineData("qwen35", SpeculatorRegistry.NGram)]
    [InlineData("qwen35-nextn", SpeculatorRegistry.DraftHead)]
    public Task TheReproducersTurnsTakeTheirEndStatesSlotsOverAndAnswerAsWithCopiedCheckpoints(string family, string speculator)
        => Reproducer(family, speculator, BackendType.Cpu);

    internal async Task Reproducer(string family, string speculator, BackendType backend)
    {
        int[] a = Tokens(300, 1);
        int[] b = null, c = null;
        // Copied checkpoints, plain decoding: the same turns, the same chunks, the same reuse.
        (List<int[]> expected, List<int> expectedReuse, StandardDecoderModel copies) = await Turns(slotBound: false, speculate: false);
        copies.Dispose();
        (List<int[]> answers, List<int> reused, StandardDecoderModel model) = await Turns(slotBound: true, speculate: speculator != null);
        using (model)
        {
            _output.WriteLine($"{family} {speculator ?? "plain"} {backend}: reused {string.Join(", ", reused)} (copies: " +
                $"{string.Join(", ", expectedReuse)}); steals {model.RowSteals}, shared {model.SharedCheckpoints}, copies {model.SlotCopies}, " +
                $"verify steps {model.BatchedSpecVerifySteps}");
            for (int i = 0; i < 3; i++)
                Assert.True(expected[i].AsSpan().SequenceEqual(answers[i]),
                    $"turn {i} answered [{string.Join(",", answers[i])}] where copied checkpoints and plain decoding gave [{string.Join(",", expected[i])}]");
            Assert.Equal(expectedReuse, reused);
            Assert.Equal(a.Length, reused[1]);   // B: A's prompt-end checkpoint
            Assert.Equal(c.Length, reused[2]);   // C: its whole prompt
            Assert.True(model.SharedCheckpoints >= 2, $"{model.SharedCheckpoints} checkpoints shared their rows");
            Assert.Equal(2, model.RowSteals);
            Assert.Equal(0, model.SlotCopies);
            Assert.Equal(0, model.SharedCheckpointsInvalidatedByWrite);
            if (speculator != null) Assert.True(model.BatchedSpecVerifySteps > 0, "C did not speculate");
        }

        async Task<(List<int[]>, List<int>, StandardDecoderModel)> Turns(bool slotBound, bool speculate)
        {
            StandardDecoderModel model = Create(family, slotBound: slotBound, backend: backend);
            model.RetainCopyMaxBytesForTests = 0;   // long slots: retained by donation, as the 27B's at 47K
            using var engine = new InferenceEngine(model, new SchedulerConfig
            {
                MaxNumBatchedTokens = 512,
                MaxNumRunningSequences = 4,
                NumBlocks = 128,
                BlockSize = 16,
                EnablePrefixCaching = true,
                StopRepetition = false,
                Speculation = speculate
                    ? new SpeculationOptions { Enabled = true, SpeculatorName = speculator, MaxDraftTokens = 3, MinDraftProb = 0 }
                    : new SpeculationOptions { Enabled = false },
            }, NullLogger.Instance);
            var answers = new List<int[]>();
            var reused = new List<int>();
            foreach (string id in new[] { "A", "B", "C" })
            {
                (int[] prompt, int max) = id switch { "A" => (a, 1), "B" => (b, 1), _ => (c, 16) };
                var sequence = new SequenceState(id, prompt, max, 16, SamplingConfig.Greedy, cacheScope: "conversation");
                answers.Add(await Serve(engine, sequence));
                reused.Add(sequence.PrefixCacheReusedTokens);
                if (id == "A" && b == null)
                {
                    // B: A's history with an answer other than A's (as a turn whose thinking was stripped), then a user turn: it
                    // diverges from A's end state at A's prompt's end, where A's prompt-end checkpoint is.
                    int other = 3 + (answers[0][0] + 1) % 250;
                    b = a.Concat(new[] { other, 18 }).Concat(Tokens(60, 2)).ToArray();
                    c = b;
                }
            }
            PrefixTree tree = engine.RadixCache!.Tree;
            Assert.Empty(PrefixTreeInvariantChecker.CheckOnce(tree));
            var held = new HashSet<string>(model.RetainedPayloadKeys, StringComparer.Ordinal);
            Assert.All(tree.PayloadKeys, key => Assert.Contains(key, held));
            return (answers, reused, model);
        }
    }

    // ---------------------------------------------------------------- on the device

    /// <summary>On direct CUDA (the CUDA executor's own slot-bound policies, F16 caches, slots sized to the device's room): the
    /// adopter of a checkpoint whose rows' slot was taken over, copied from or evicted continues bit for bit as a fresh run; a
    /// steal keeps the slot's recordings, so the adopter's decode steps replay what its rows' last request recorded.</summary>
    [CudaTheory]
    [InlineData("qwen35", Fate.Steal)]
    [InlineData("qwen35", Fate.Copy)]
    [InlineData("qwen35", Fate.Evicted)]
    [InlineData("qwen35moe", Fate.Steal)]
    [InlineData("qwen4exp", Fate.Steal)]
    [InlineData("mistral3", Fate.Steal)]
    public void OnDirectCuda_AnAdopterOfASharedCheckpointContinuesAsAFreshRun(string family, Fate fate)
    {
        const int decodes = 16;
        List<float[]> expected = Reference(family, BackendType.Cuda, decodes);
        var sink = new RecordingPayloadSink();
        using StandardDecoderModel model = Create(family, sink, backend: BackendType.Cuda);
        var cache = (IPrefixCacheModel)model;
        ShareAndAppend(model, decodes: 48, checksums: false);
        Tensor[][] rows = model.SlotCachesForTests("a");
        switch (fate)
        {
            case Fate.Steal:
            {
                // Qwen3.5's decode steps record on the CUDA executor (the issue's family); where a family's do, the adopter
                // replays them.
                bool recorded = model.DecodeRecordings > 0;
                if (family == "qwen35") Assert.True(recorded, "the owner's decode steps recorded nothing");
                Assert.True(model.RetainSequenceCacheAs("a", "E"));
                Assert.True(cache.TryMaterialize(Clone("cp", "b", cache.RowSharers("cp"))));
                Assert.Equal(1, model.RowSteals);
                Assert.Same(rows, model.SlotCachesForTests("b"));
                int adopted = model.RecordingsAdopted;
                AssertSameRows(expected, Continuation(model, "b", Prompt.Length, decodes), $"{family} {fate} on direct CUDA");
                if (recorded)
                    Assert.True(model.RecordingsAdopted > adopted, "the adopter recorded its decode steps again (the taken-over slot's recordings went)");
                _output.WriteLine($"{family}: owner recordings {model.DecodeRecordings}, adopted {model.RecordingsAdopted - adopted}");
                break;
            }
            case Fate.Copy:
                Assert.True(model.RetainSequenceCacheAs("a", "E"));
                Assert.True(cache.TryMaterialize(Clone("cp", "b", null)));
                Assert.NotSame(rows, model.SlotCachesForTests("b"));
                AssertSameRows(expected, Continuation(model, "b", Prompt.Length, decodes), $"{family} {fate} on direct CUDA");
                break;
            case Fate.Evicted:
                Assert.True(model.RetainSequenceCacheAs("a", "E"));
                model.DiscardRetainedCaches(new[] { "E" }, ReleaseReason.Evicted);
                Assert.Equal(1, model.RowHandOvers);
                Assert.True(cache.TryMaterialize(new MaterializeRequest(MaterializeOp.Donate, "cp", "b", Prompt.Length, Prompt.Length)));
                Assert.Same(rows, model.SlotCachesForTests("b"));
                AssertSameRows(expected, Continuation(model, "b", Prompt.Length, decodes), $"{family} {fate} on direct CUDA");
                break;
        }
        Assert.Equal(0, model.SharedCheckpointsInvalidatedByWrite);
        model.CheckSlotShareInvariants();
    }

    /// <summary>On direct CUDA a new request short of device room takes over the oldest retained slot that holds it
    /// (ReclaimRetainedSlot) - never one whose rows a checkpoint shares, nor a checkpoint.</summary>
    [CudaFact]
    public void OnDirectCuda_ANewRequestShortOfRoomNeverTakesOverASharedSlot()
    {
        var sink = new RecordingPayloadSink();
        using StandardDecoderModel model = Create("qwen35", sink, backend: BackendType.Cuda);
        ShareAndAppend(model, checksums: false);
        Assert.True(model.RetainSequenceCacheAs("a", "E"));
        // A device with no room, where releasing the conversations frees nothing: the pooled spares go (making room for
        // recordings), and the new request then finds only the shared slot and its checkpoint to take over.
        model.DeviceRoomForTests = () => (0, 16 * GB);
        model.RetainedPayloadBytesForTests = _ => 0;
        model.EnsureRecordingRoomForTests(GB);
        Assert.Equal(0, model.SlotPoolForTests.Pooled);
        var refused = Assert.Throws<InvalidOperationException>(() => ((IUnifiedBatchModel)model).EnsureSequenceSlot("new", 64));
        Assert.Contains("refused before any of it ran", refused.Message);
        Assert.Equal(0, model.RetainedSlotsReclaimed);
        Assert.Equal(new[] { "E", "cp" }, model.RetainedPayloadKeys.OrderBy(k => k, StringComparer.Ordinal));
        Assert.Empty(sink.Reports);
        model.DeviceRoomForTests = null;
        model.CheckSlotShareInvariants();
    }

    /// <summary>The reproducer's turns on direct CUDA: each next turn takes the end state's slot over, nothing is copied, and the
    /// answers - speculating with n-gram or MTP drafts over the taken-over slots - are plain decoding's with copied checkpoints.</summary>
    [CudaTheory]
    [InlineData("qwen35", null)]
    [InlineData("qwen35", SpeculatorRegistry.NGram)]
    [InlineData("qwen35-nextn", SpeculatorRegistry.DraftHead)]
    public Task OnDirectCuda_TheReproducersTurnsTakeTheirEndStatesSlotsOverAndAnswerAsWithCopiedCheckpoints(string family, string speculator)
        => Reproducer(family, speculator, BackendType.Cuda);

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
