// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Issue #376: where slots are sized to the device's room (the CUDA executor on one device), a slot is made, copied or grown to
// the class of its request's reservation where memory allows without releasing a conversation, else to the largest capacity
// that fits, and only for the cells its step needs are finished conversations released. A reservation is fitted once, a slot
// grows a tensor at a time, spares are the smallest class and a finished slot is pooled only with room beside it. The managed
// CPU executor runs the policies through the model's seams (SlotBoundPoliciesOnFlowForTests, DeviceRoomForTests); the CUDA
// tests run them on the device.
using InferenceWeb.Tests.PrefixCache.Fakes;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Scheduling.PrefixCache;

namespace InferenceWeb.Tests;

public sealed class SlotTargetTests : IDisposable
{
    private const long GB = 1L << 30;
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-slot-target-" + Guid.NewGuid().ToString("N"));
    private readonly int[] _prompt = Enumerable.Range(0, 20).Select(i => 40 + i % 7).ToArray();

    public SlotTargetTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    /// <summary>A Qwen3.5 dense fixture with a 131,072-token window on the managed CPU executor, with the slot-bound policies
    /// from its construction and a device of <paramref name="free"/> bytes free of 32 GB.</summary>
    private Qwen35RuntimeModel Create(Func<long> free, RecordingPayloadSink sink = null)
    {
        using var dtype = KvCacheDtypeConfig.ForCurrentFlowInTests(KvCacheDtype.F32);
        string path = new Qwen35DenseSyntheticModelBuilder { Layers = 8, UntiedOutput = true, ContextLength = 131072 }
            .Write(Path.Combine(_dir, "q35-dense.gguf"));
        bool policies = StandardDecoderModel.SlotBoundPoliciesOnFlowForTests.Value;
        StandardDecoderModel.SlotBoundPoliciesOnFlowForTests.Value = true;
        Qwen35RuntimeModel model;
        try
        {
            model = new Qwen35RuntimeModel(path, BackendType.Cpu, 1, null, 1);
        }
        finally
        {
            StandardDecoderModel.SlotBoundPoliciesOnFlowForTests.Value = policies;
        }
        Assert.True(model.SlotBoundPoliciesForTests);
        model.DeviceRoomForTests = () => (free(), 32 * GB);
        if (sink != null) model.AttachPrefixCache(sink);
        return model;
    }

    private static int ArgMax(float[] row)
    {
        int best = 0;
        for (int i = 1; i < row.Length; i++)
            if (row[i] > row[best]) best = i;
        return best;
    }

    /// <summary>The largest whole 1,024 below <paramref name="below"/> whose fresh slot wants no more than <paramref name="free"/>.</summary>
    private static int LargestFitting(Qwen35RuntimeModel model, long free, int below)
    {
        int fit = 0;
        for (int c = 1024; c < below; c += 1024)
            if (model.SlotRoomNeedForTests(c) <= free) fit = c;
        return fit;
    }

    /// <summary>The class of the reservation and an eighth where memory allows; else the largest whole 1,024 that fits, with no
    /// conversation released for either; only the cells the step needs release one, the oldest first.</summary>
    [Fact]
    public void TheClassWhereMemoryAllows_ElseTheLargestThatFits_ConversationsOnlyForTheRequiredCells()
    {
        var sink = new RecordingPayloadSink();
        long free = 32 * GB;
        using Qwen35RuntimeModel model = Create(() => free, sink);
        var unified = (IUnifiedBatchModel)model;
        foreach (string key in new[] { "old", "older" })
        {
            Assert.Equal(0, unified.EnsureSequenceSlot(key, 64));
            Assert.True(unified.TryForwardUnified(new[] { key }, new[] { _prompt }, new[] { 0 }, new float[1][]));
            Assert.True(model.RetainSequenceCacheAs(key, key));
        }

        int cls = model.SlotCapacityClass(20000 + 2500);
        Assert.Equal(24576, cls);
        free = model.SlotRoomNeedForTests(cls);
        Assert.Equal(cls, model.SlotTargetForTests(21, 20000));

        free = model.SlotRoomNeedForTests(cls) - 1;
        int fit = LargestFitting(model, free, cls);
        Assert.Equal(23552, fit);
        Assert.Equal(fit, model.SlotTargetForTests(21, 20000));
        free = model.SlotRoomNeedForTests(9000);
        Assert.Equal(8192, model.SlotTargetForTests(21, 20000));
        Assert.Empty(sink.Reports);

        // The required cells' 1,024 do not fit, their 256 do: no conversation goes for the slack.
        free = model.SlotRoomNeedForTests(1024) - 1;
        Assert.Equal(256, model.SlotTargetForTests(21, 20000));
        Assert.Empty(sink.Reports);

        // Not even those fit: the oldest conversation goes for the required cells' 1,024, and only it.
        long tight = model.SlotRoomNeedForTests(256) - 1, roomy = model.SlotRoomNeedForTests(1024);
        model.DeviceRoomForTests = () => (model.RetainedPayloadKeys.Count < 2 ? roomy : tight, 32 * GB);
        Assert.Equal(1024, model.SlotTargetForTests(21, 20000));
        Assert.Equal(new[] { ("old", InvalidationReason.NativeSlotReclaimed) }, sink.Reports);

        // Nothing left that makes room: every conversation released, no target.
        model.DeviceRoomForTests = () => (tight, 32 * GB);
        Assert.Equal(0, model.SlotTargetForTests(21, 20000));
        Assert.Empty(model.RetainedPayloadKeys);
        Assert.Equal(("older", InvalidationReason.NativeSlotReclaimed), sink.Reports[^1]);
    }

    /// <summary>A conversation is released whole: the room it leaves past the required cells is taken for the slot (the largest
    /// whole 1,024 it holds, up to the class, nothing more released), not left for a growth at the request's next step.</summary>
    [Fact]
    public void TheRoomAReleasedConversationLeavesPastTheRequiredCellsIsTaken()
    {
        var sink = new RecordingPayloadSink();
        using Qwen35RuntimeModel model = Create(() => 32 * GB, sink);
        var unified = (IUnifiedBatchModel)model;
        foreach (string key in new[] { "old", "older" })
        {
            Assert.Equal(0, unified.EnsureSequenceSlot(key, 64));
            Assert.True(unified.TryForwardUnified(new[] { key }, new[] { _prompt }, new[] { 0 }, new float[1][]));
            Assert.True(model.RetainSequenceCacheAs(key, key));
        }
        long tight = model.SlotRoomNeedForTests(256) - 1, after = model.SlotRoomNeedForTests(9000);
        model.DeviceRoomForTests = () => (model.RetainedPayloadKeys.Count < 2 ? after : tight, 32 * GB);
        int fit = LargestFitting(model, after, model.SlotCapacityClass(20000 + 2500));
        Assert.Equal(8192, fit);
        Assert.Equal(fit, model.SlotTargetForTests(21, 20000));
        Assert.Equal(new[] { ("old", InvalidationReason.NativeSlotReclaimed) }, sink.Reports);
    }

    /// <summary>A new request's slot is its reservation's class where memory allows, else the largest that fits, and the request
    /// runs in it: its reservation is fitted once, and the steps after its first do not ask the device again while the slot
    /// holds what they need.</summary>
    [Fact]
    public void ANewSlotIsFittedOnce_AndItsLaterStepsDoNotAskAgain()
    {
        long free = 32 * GB;
        int reads = 0;
        using Qwen35RuntimeModel model = Create(() => { reads++; return free; });
        var unified = (IUnifiedBatchModel)model;
        Assert.Equal(0, unified.EnsureSequenceSlot("a", 6000, _prompt.Length + 1));
        Assert.Equal(model.SlotCapacityClass(6000 + 1024), model.SlotCapacityForTests("a"));
        Assert.Equal(7168, model.SlotCapacityForTests("a"));

        free = model.SlotRoomNeedForTests(4096);
        Assert.Equal(0, unified.EnsureSequenceSlot("b", 6000, _prompt.Length + 1));
        Assert.Equal(LargestFitting(model, free, 7168), model.SlotCapacityForTests("b"));
        Assert.Equal(4096, model.SlotCapacityForTests("b"));
        Assert.Equal(2, model.FittedReservationsForTests);

        var logits = new float[1][];
        Assert.True(unified.TryForwardUnified(new[] { "b" }, new[] { _prompt }, new[] { 0 }, logits));
        int length = _prompt.Length, next = ArgMax(logits[0]);
        int before = reads;
        object caches = model.SlotCachesForTests("b");
        for (int t = 0; t < 5; t++)
        {
            Assert.Equal(length, unified.EnsureSequenceSlot("b", 6000, length + 1));
            Assert.True(unified.TryForwardUnified(new[] { "b" }, new[] { new[] { next } }, new[] { length }, logits));
            next = ArgMax(logits[0]);
            length++;
        }
        Assert.Equal(before, reads);
        Assert.Same(caches, model.SlotCachesForTests("b"));
    }

    /// <summary>A conversation's next turn that adds no more than its slot's class holds runs in the slot its finished turn
    /// donated: nothing allocated, the same caches.</summary>
    [Fact]
    public void AContinuationWithinTheClassAllocatesNothing()
    {
        var sink = new RecordingPayloadSink();
        using Qwen35RuntimeModel model = Create(() => 32 * GB, sink);
        var unified = (IUnifiedBatchModel)model;
        Assert.Equal(0, unified.EnsureSequenceSlot("turn1", 3000, _prompt.Length + 1));
        Assert.Equal(4096, model.SlotCapacityForTests("turn1"));
        Assert.True(unified.TryForwardUnified(new[] { "turn1" }, new[] { _prompt }, new[] { 0 }, new float[1][]));
        object caches = model.SlotCachesForTests("turn1");
        Assert.True(model.RetainSequenceCacheAs("turn1", "conversation"));
        Assert.True(model.TryRebindRetainedCache("conversation", "turn2"));

        Assert.Equal(_prompt.Length, unified.EnsureSequenceSlot("turn2", 3900, _prompt.Length + 40));
        Assert.Same(caches, model.SlotCachesForTests("turn2"));
        Assert.Equal(0, model.RollingGrowths);
        Assert.True(unified.TryForwardUnified(new[] { "turn2" }, new[] { Enumerable.Repeat(9, 30).ToArray() }, new[] { _prompt.Length },
            new float[1][]));
    }

    /// <summary>A slot fitted below its reservation grows when a step needs more, a tensor at a time, to the class of what its
    /// request reserved: its rows are kept, and its logits are those of a run whose slot held everything from the start.</summary>
    [Fact]
    public void AGrownSlotKeepsItsRows_AndAnswersAsARunThatNeverGrew()
    {
        int[] prompt = Enumerable.Range(0, 1016).Select(i => 3 + (i * 37 + i / 7) % 250).ToArray();
        List<float[]> Run(bool tight, out Qwen35RuntimeModel kept)
        {
            long free = 32 * GB;
            var model = Create(() => free);
            var unified = (IUnifiedBatchModel)model;
            if (tight) free = model.SlotRoomNeedForTests(1024);
            Assert.Equal(0, unified.EnsureSequenceSlot("a", 3000, prompt.Length + 1));
            Assert.Equal(tight ? 1024 : 4096, model.SlotCapacityForTests("a"));
            free = 32 * GB;
            var rows = new List<float[]>();
            var logits = new float[1][];
            Assert.True(unified.TryForwardUnified(new[] { "a" }, new[] { prompt }, new[] { 0 }, logits));
            rows.Add((float[])logits[0].Clone());
            int length = prompt.Length, next = ArgMax(logits[0]);
            for (int t = 0; t < 16; t++)
            {
                unified.EnsureSequenceSlot("a", 3000, length + 1);
                Assert.True(unified.TryForwardUnified(new[] { "a" }, new[] { new[] { next } }, new[] { length }, logits));
                rows.Add((float[])logits[0].Clone());
                next = ArgMax(logits[0]);
                length++;
            }
            kept = model;
            return rows;
        }

        List<float[]> grown = Run(tight: true, out Qwen35RuntimeModel grownModel);
        List<float[]> whole = Run(tight: false, out Qwen35RuntimeModel wholeModel);
        using (grownModel)
        using (wholeModel)
        {
            Assert.Equal(1, grownModel.RollingGrowths);
            Assert.Equal(4096, grownModel.SlotCapacityForTests("a"));
            Assert.Equal(0, wholeModel.RollingGrowths);
            Assert.Equal(whole.Count, grown.Count);
            for (int i = 0; i < whole.Count; i++)
                Assert.True(whole[i].AsSpan().SequenceEqual(grown[i]), $"step {i} differs after the slot grew");
        }
    }

    /// <summary>On direct CUDA (no allocation is zero there): a slot fitted to a short device grows a tensor at a time when its
    /// decode crosses its capacity, zero-filling only the cells past its length, and its logits are those of a run whose slot
    /// held everything from the start, bit for bit.</summary>
    [CudaFact]
    public void OnDirectCuda_AGrowthZeroesOnlyThePastLengthCells_AndAnswersAsARunThatNeverGrew()
    {
        using var dtype = KvCacheDtypeConfig.ForCurrentFlowInTests(KvCacheDtype.F16);
        string path = new Qwen35DenseSyntheticModelBuilder { Layers = 8, UntiedOutput = true, ContextLength = 16384 }
            .Write(Path.Combine(_dir, "q35-cuda.gguf"));
        int[] prompt = Enumerable.Range(0, 2040).Select(i => 3 + (i * 37 + i / 7) % 250).ToArray();
        List<float[]> Run(bool tight, out int growths, out long zeroedPastLength, out long expected)
        {
            using var model = new Qwen35RuntimeModel(path, BackendType.Cuda, 1, null, 1);
            var unified = (IUnifiedBatchModel)model;
            long free = 32 * GB;
            model.DeviceRoomForTests = () => (free, 32 * GB);
            if (tight) free = model.SlotRoomNeedForTests(2048);
            Assert.Equal(0, unified.EnsureSequenceSlot("a", 3000, prompt.Length + 1));
            Assert.Equal(tight ? 2048 : 4096, model.SlotCapacityForTests("a"));
            free = 32 * GB;
            var rows = new List<float[]>();
            var logits = new float[1][];
            Assert.True(unified.TryForwardUnified(new[] { "a" }, new[] { prompt }, new[] { 0 }, logits));
            rows.Add((float[])logits[0].Clone());
            int length = prompt.Length, next = ArgMax(logits[0]);
            zeroedPastLength = expected = 0;
            for (int t = 0; t < 16; t++)
            {
                long before = model.CacheZeroFillBytes;
                int capacity = model.SlotCapacityForTests("a") ?? 0;
                unified.EnsureSequenceSlot("a", 3000, length + 1);
                if (model.SlotCapacityForTests("a") != capacity)
                {
                    zeroedPastLength = model.CacheZeroFillBytes - before;
                    expected = model.CacheBytesPerCellForTests * ((model.SlotCapacityForTests("a") ?? 0) - length);
                }
                Assert.True(unified.TryForwardUnified(new[] { "a" }, new[] { new[] { next } }, new[] { length }, logits));
                rows.Add((float[])logits[0].Clone());
                next = ArgMax(logits[0]);
                length++;
            }
            growths = model.RollingGrowths;
            model.OnSequenceReleased("a");
            return rows;
        }

        List<float[]> grown = Run(tight: true, out int growths, out long zeroed, out long expected);
        List<float[]> whole = Run(tight: false, out int noGrowths, out _, out _);
        Assert.Equal(1, growths);
        Assert.Equal(0, noGrowths);
        Assert.True(expected > 0);
        Assert.Equal(expected, zeroed);
        for (int i = 0; i < whole.Count; i++)
            Assert.True(whole[i].AsSpan().SequenceEqual(grown[i]), $"step {i} differs after the slot grew");
    }

    /// <summary>A growth whose allocation fails part way fails the request with what happened; the slot, its caches of two
    /// capacities, is released, and the next request runs.</summary>
    [Fact]
    public void AGrowthThatFailsPartWayReleasesTheSlot()
    {
        long free = 32 * GB;
        using Qwen35RuntimeModel model = Create(() => free);
        var unified = (IUnifiedBatchModel)model;
        free = model.SlotRoomNeedForTests(1024);
        Assert.Equal(0, unified.EnsureSequenceSlot("a", 3000, _prompt.Length + 1));
        Assert.Equal(1024, model.SlotCapacityForTests("a"));
        Assert.True(unified.TryForwardUnified(new[] { "a" }, new[] { _prompt }, new[] { 0 }, new float[1][]));
        free = 32 * GB;
        model.RollingGrowthFailAfterForTests = 3;
        var failure = Assert.Throws<InvalidOperationException>(() => unified.EnsureSequenceSlot("a", 3000, 2000));
        Assert.Contains("after 3 of its cache tensors", failure.Message);
        Assert.Null(model.SlotCapacityForTests("a"));
        Assert.False(unified.TryForwardUnified(new[] { "a" }, new[] { new[] { 7 } }, new[] { _prompt.Length }, new float[1][]));
        model.OnSequenceReleased("a");
        Assert.Equal(0, model.SlotPoolForTests.Pooled);

        model.RollingGrowthFailAfterForTests = null;
        Assert.Equal(0, unified.EnsureSequenceSlot("b", 64, _prompt.Length + 1));
        Assert.True(unified.TryForwardUnified(new[] { "b" }, new[] { _prompt }, new[] { 0 }, new float[1][]));
    }

    /// <summary>A finished request's slot is pooled only while the device keeps as much again free beside its headroom and a
    /// prompt step; a spare is at least the smallest class, and no larger than the largest slot a finished request's copy fits
    /// (a long conversation's slot is donated, not copied).</summary>
    [Fact]
    public void PooledSlotsNeedRoom_AndSparesAreSmall()
    {
        long free = 32 * GB;
        using Qwen35RuntimeModel model = Create(() => free);
        var unified = (IUnifiedBatchModel)model;
        Assert.Equal(0, unified.EnsureSequenceSlot("big", 20000, _prompt.Length + 1));
        Assert.Equal(24576, model.SlotCapacityForTests("big"));
        Assert.True(unified.TryForwardUnified(new[] { "big" }, new[] { _prompt }, new[] { 0 }, new float[1][]));
        Assert.Equal(1024, model.SpareCapacityForTests);

        free = GB + GB / 16;   // the 24,576-cell slot (192 MiB of K/V) and the 1 GB headroom do not fit
        model.OnSequenceReleased("big");
        Assert.Equal(0, model.SlotPoolForTests.Pooled);

        free = 32 * GB;
        Assert.Equal(0, unified.EnsureSequenceSlot("small", 64, _prompt.Length + 1));
        Assert.True(unified.TryForwardUnified(new[] { "small" }, new[] { _prompt }, new[] { 0 }, new float[1][]));
        model.OnSequenceReleased("small");
        Assert.Equal(new[] { model.SlotCapacityClass(64 + 1024) }, model.PooledCapacitiesForTests);
    }
}
