// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Issue #376: where recorded steps bind slots (the CUDA executor), a finished request's slot is donated to the prefix cache and
// the conversation's next turn runs on it; only a short slot is copied, into a pooled slot that holds it (nothing allocated).
// Retained slots keep the small buffers an in-place verify window binds, so the next turn's windows replay what the finished
// request recorded; a device short of memory takes them back before it releases a conversation. The managed CPU executor
// runs the policies through the model's seam (SlotBoundPoliciesForTests); the CUDA test runs them on the device.
using InferenceWeb.Tests.PrefixCache.Fakes;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Models;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Scheduling.PrefixCache;

namespace InferenceWeb.Tests;

public sealed class RetainedSlotDonationTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-retained-donation-" + Guid.NewGuid().ToString("N"));
    private readonly int[] _prompt = Enumerable.Range(0, 20).Select(i => 40 + i % 7).ToArray();

    public RetainedSlotDonationTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    /// <summary>A Qwen3.5 dense fixture; on the managed CPU executor with the slot-bound policies of the CUDA executor
    /// (<paramref name="slotBound"/>), on direct CUDA with its own.</summary>
    private Qwen35RuntimeModel Create(BackendType backend = BackendType.Cpu, bool slotBound = true)
    {
        using var dtype = KvCacheDtypeConfig.ForCurrentFlowInTests(backend == BackendType.Cpu ? KvCacheDtype.F32 : KvCacheDtype.F16);
        string path = new Qwen35DenseSyntheticModelBuilder { Layers = 8, UntiedOutput = true }.Write(Path.Combine(_dir, "q35-dense.gguf"));
        var model = new Qwen35RuntimeModel(path, backend, 1, null, 1);
        if (backend == BackendType.Cpu) model.SlotBoundPoliciesForTests = slotBound;
        model.AttachPrefixCache(new RecordingPayloadSink());
        return model;
    }

    private static int ArgMax(float[] row)
    {
        int best = 0;
        for (int i = 1; i < row.Length; i++)
            if (row[i] > row[best]) best = i;
        return best;
    }

    /// <summary>A request's prompt and <paramref name="steps"/> decode steps; its length and next token.</summary>
    private (int Length, int Next) Run(Qwen35RuntimeModel model, string id, int reserve = 64, int steps = 3)
    {
        var unified = (IUnifiedBatchModel)model;
        var logits = new float[1][];
        Assert.Equal(0, unified.EnsureSequenceSlot(id, reserve));
        Assert.True(unified.TryForwardUnified(new[] { id }, new[] { _prompt }, new[] { 0 }, logits));
        int length = _prompt.Length, next = ArgMax(logits[0]);
        for (int t = 0; t < steps; t++)
        {
            Assert.True(unified.TryForwardUnified(new[] { id }, new[] { new[] { next } }, new[] { length }, logits));
            next = ArgMax(logits[0]);
            length++;
        }
        return (length, next);
    }

    /// <summary>One verify window of two rows on <paramref name="id"/>'s slot, keeping its first row; its rows' logits.</summary>
    private static int Window(Qwen35RuntimeModel model, string id, ref int length, int next) => Window(model, id, ref length, next, out _);

    private static int Window(Qwen35RuntimeModel model, string id, ref int length, int next, out float[] logits)
    {
        var batched = (IBatchedSpeculativeTarget)model;
        int vocab = model.Config.VocabSize;
        logits = new float[2 * vocab];
        Assert.True(batched.TryBatchedSpecVerifyLogits(new[] { id }, new[] { length }, 2, new[] { next, next }, logits, null));
        batched.CommitBatchedSpecVerify(new[] { id }, new[] { length }, new[] { 1 });
        length++;
        return ArgMax(logits.AsSpan(0, vocab).ToArray());
    }

    /// <summary>The logits after <paramref name="tokens"/> continue <paramref name="id"/>'s slot from <paramref name="at"/>.</summary>
    private static float[] Continue(Qwen35RuntimeModel model, string id, int[] tokens, int at)
    {
        var logits = new float[1][];
        Assert.True(((IUnifiedBatchModel)model).TryForwardUnified(new[] { id }, new[] { tokens }, new[] { at }, logits));
        return (float[])logits[0].Clone();
    }

    /// <summary>A pooled slot of <paramref name="reserve"/> tokens a finished request left (made beside the running ones).</summary>
    private void PoolASlot(Qwen35RuntimeModel model, int reserve = 64)
    {
        Run(model, "pooled", reserve, steps: 0);
        model.OnSequenceReleased("pooled");
    }

    /// <summary>A finished slot whose rows are past the copy bound is donated, even beside a pooled slot that holds it: the
    /// prefix cache keeps the request's own caches, the next turn runs on them, and it continues as the request would have.
    /// Nothing is allocated or copied (the pooled slot stays).</summary>
    [Fact]
    public void ALongFinishedSlotIsDonatedAndTheNextTurnRunsOnIt()
    {
        float[] expected;
        int length, next;
        using (Qwen35RuntimeModel reference = Create())
        {
            (length, next) = Run(reference, "a");
            expected = Continue(reference, "a", new[] { next, 9, 10 }, length);
        }

        using Qwen35RuntimeModel model = Create();
        (length, next) = Run(model, "a");
        PoolASlot(model);
        model.RetainCopyMaxBytesForTests = 0;   // every slot's rows are past it
        Tensor[][] caches = model.SlotCachesForTests("a");
        var cache = (IPrefixCacheModel)model;
        Assert.True(cache.TryCaptureDonate("a", "kept", length, out PayloadFootprint footprint));
        Assert.Equal(length, footprint.Tokens);
        Assert.Equal(0, model.RetainedByCopy);
        Assert.False(model.LastRetentionCopied);
        Assert.Same(caches, model.SlotCachesForTests("kept"));
        Assert.Null(model.SlotCachesForTests("a"));
        Assert.Equal(1, model.SlotPoolForTests.Pooled);

        Assert.True(model.TryRebindRetainedCache("kept", "turn2"));
        Assert.Same(caches, model.SlotCachesForTests("turn2"));
        float[] actual = Continue(model, "turn2", new[] { next, 9, 10 }, length);
        Assert.True(expected.AsSpan().SequenceEqual(actual), "the donated slot's next turn answers otherwise than the request would have");
    }

    /// <summary>A short finished slot is copied into a pooled slot that holds it, as a whole slot of its capacity: the request
    /// keeps its own slot (its release returns it to the pool, where the next request adopts its recordings), and the copy
    /// continues as the request would have.</summary>
    [Fact]
    public void AShortFinishedSlotIsCopiedIntoAPooledSlotThatHoldsIt()
    {
        float[] expected;
        int length, next;
        using (Qwen35RuntimeModel reference = Create())
        {
            (length, next) = Run(reference, "a");
            expected = Continue(reference, "a", new[] { next, 9, 10 }, length);
        }

        using Qwen35RuntimeModel model = Create();
        (length, next) = Run(model, "a");
        PoolASlot(model);
        Tensor[][] caches = model.SlotCachesForTests("a");
        var cache = (IPrefixCacheModel)model;
        Assert.True(cache.TryCaptureDonate("a", "kept", length, out _));
        Assert.Equal(1, model.RetainedByCopy);
        Assert.True(model.LastRetentionCopied);
        Assert.Equal(0, model.SlotPoolForTests.Pooled);
        Assert.NotSame(caches, model.SlotCachesForTests("kept"));
        Assert.Same(caches, model.SlotCachesForTests("a"));
        model.OnSequenceReleased("a");
        Assert.Equal(1, model.SlotPoolForTests.Pooled);

        Assert.True(model.TryRebindRetainedCache("kept", "turn2"));
        float[] actual = Continue(model, "turn2", new[] { next, 9, 10 }, length);
        Assert.True(expected.AsSpan().SequenceEqual(actual), "the retained copy's next turn answers otherwise than the request would have");
    }

    /// <summary>A short finished slot is donated where no pooled slot holds it (none, or only a smaller one): a copy would
    /// allocate a slot beside the one the request ran in.</summary>
    [Fact]
    public void AShortSlotWithoutAPooledSlotThatHoldsItIsDonated()
    {
        using Qwen35RuntimeModel model = Create();
        var cache = (IPrefixCacheModel)model;
        (int length, _) = Run(model, "a", reserve: 1500);   // a 1,536-cell slot
        Tensor[][] caches = model.SlotCachesForTests("a");
        Assert.True(cache.TryCaptureDonate("a", "kept", length, out _));
        Assert.Equal(0, model.RetainedByCopy);
        Assert.Same(caches, model.SlotCachesForTests("kept"));

        (length, _) = Run(model, "b", reserve: 1500);
        PoolASlot(model, reserve: 64);                       // a 512-cell slot
        caches = model.SlotCachesForTests("b");
        Assert.True(cache.TryCaptureDonate("b", "kept-b", length, out _));
        Assert.Equal(0, model.RetainedByCopy);
        Assert.False(model.LastRetentionCopied);
        Assert.Same(caches, model.SlotCachesForTests("kept-b"));
        Assert.Equal(1, model.SlotPoolForTests.Pooled);
    }

    /// <summary>Executors whose recorded steps do not bind slots keep today's retention: the slot is donated whatever its size
    /// and whatever the pool holds.</summary>
    [Fact]
    public void ElsewhereAFinishedSlotIsDonatedAsBefore()
    {
        using Qwen35RuntimeModel model = Create(slotBound: false);
        (int length, _) = Run(model, "a");
        PoolASlot(model);
        Tensor[][] caches = model.SlotCachesForTests("a");
        Assert.True(((IPrefixCacheModel)model).TryCaptureDonate("a", "kept", length, out _));
        Assert.Equal(0, model.RetainedByCopy);
        Assert.Same(caches, model.SlotCachesForTests("kept"));
    }

    /// <summary>A retained slot's states keep the buffers an in-place window binds (its conv history and saved inputs) where
    /// recorded steps bind slots, and free the per-row verify buffer, which an in-place window never allocates; a later window
    /// of the same width then binds the same buffers. Released without keeping, every buffer goes.</summary>
    [Fact]
    public void AnInPlaceWindowsBuffersStayWhenItsStateIsRetained()
    {
        var allocator = new CpuAllocator(BlasEnum.DotNet);
        var executor = new TensorSharp.Models.Graph.ManagedGraphExecutor(allocator);
        using var state = new Qwen35RuntimeModel.DeviceGdnState(allocator, executor, _ => IntPtr.Zero,
            convRows: 3, channels: 48, vHeads: 2, dv: 8, dk: 8, backingRows: 8);
        Assert.True(state.WindowInPlace(4));
        Assert.True(state.EnsureSpecBuffers(4));
        Tensor history = state.ConvHistory, inputs = state.SavedInputs;
        Assert.NotNull(history);
        Assert.NotNull(inputs);
        Assert.Null(state.Verify);
        long spec = state.SpecBufferBytes;
        Assert.Equal(history.Storage.ByteLength + inputs.Storage.ByteLength, spec);

        state.ReleaseSpecBuffers(keepInPlaceBuffers: true);
        Assert.Same(history, state.ConvHistory);
        Assert.Same(inputs, state.SavedInputs);
        Assert.False(state.EnsureSpecBuffers(4), "a window of the same width bound new buffers after the retention kept them");

        // A window wider than the in-place rows keeps a state after each row: that buffer goes, the in-place ones stay.
        state.EnsureSpecBuffers(16);
        Assert.NotNull(state.Verify);
        history = state.ConvHistory;
        state.ReleaseSpecBuffers(keepInPlaceBuffers: true);
        Assert.Null(state.Verify);
        Assert.Same(history, state.ConvHistory);
        Assert.Same(inputs, state.SavedInputs);

        state.ReleaseSpecBuffers(keepInPlaceBuffers: false);
        Assert.Null(state.ConvHistory);
        Assert.Null(state.SavedInputs);
        Assert.Equal(0, state.SpecBufferBytes);
    }

    /// <summary>The model's retention keeps what its windows bind beside the per-row buffer (the managed CPU executor's windows
    /// keep a state after each row: that buffer goes, the conv history stays) where recorded steps bind slots, and frees
    /// every verify buffer elsewhere.</summary>
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void ARetainedSlotKeepsItsWindowBuffersWhereRecordedStepsBindSlots(bool slotBound)
    {
        using Qwen35RuntimeModel model = Create(slotBound: slotBound);
        (int length, int next) = Run(model, "a", steps: 1);
        next = Window(model, "a", ref length, next);
        Assert.True(model.RetainSequenceCacheAs("a", "kept"));
        var states = model.RetainedStatesForTests("kept").SelectMany(l => l ?? Array.Empty<StandardDecoderModel.RecurrentState>())
            .OfType<Qwen35RuntimeModel.DeviceGdnState>().ToArray();
        Assert.NotEmpty(states);
        Assert.All(states, s => Assert.Null(s.Verify));
        if (slotBound) Assert.All(states, s => Assert.NotNull(s.ConvHistory));
        else Assert.All(states, s => Assert.Null(s.ConvHistory));
    }

    /// <summary>The allocator's reclaimer (rung 3) takes back the window buffers retained slots keep before it releases a
    /// conversation, where they cover what is asked; short of that, it leaves them to the conversations that stay and releases
    /// conversations, the oldest first.</summary>
    [Fact]
    public void TheReclaimerTakesRetainedWindowBuffersBeforeConversations()
    {
        using Qwen35RuntimeModel model = Create();
        foreach (string key in new[] { "k0", "k1" })
        {
            (int length, int next) = Run(model, key, steps: 1);
            Window(model, key, ref length, next);
            Assert.True(model.RetainSequenceCacheAs(key, key));
        }
        Run(model, "run", steps: 0);   // the engine's thread: the reclaimer acts here
        long Buffers(string key) => model.RetainedStatesForTests(key).SelectMany(l => l ?? Array.Empty<StandardDecoderModel.RecurrentState>())
            .Sum(s => s?.SpecBufferBytes ?? 0);
        long k0 = Buffers("k0"), k1 = Buffers("k1");
        Assert.True(k0 > 0 && k1 > 0);

        // Asked for less than k0's buffers: they go (the oldest), k1's stay, no conversation is released.
        Assert.Equal(k0, model.ReclaimForAllocator(k0 - 1));
        Assert.Equal(1, model.DeviceRoomStages[4]);
        Assert.Equal(0, Buffers("k0"));
        Assert.Equal(k1, Buffers("k1"));
        Assert.Equal(new[] { "k0", "k1" }, model.RetainedPayloadKeys.OrderBy(k => k, StringComparer.Ordinal));

        // Asked for more than every buffer: the oldest conversation goes, k1 keeps its buffers.
        Assert.True(model.ReclaimForAllocator(k1 + 1) > k1);
        Assert.Equal(1, model.DeviceRoomStages[4]);
        Assert.Equal(1, model.DeviceRoomStages[5]);
        Assert.Equal(new[] { "k1" }, model.RetainedPayloadKeys);
        Assert.Equal(k1, Buffers("k1"));
    }

    /// <summary>
    /// On direct CUDA a conversation's next turn on its donated slot replays the recordings its first turn made: its decode
    /// steps and its verify windows are adopted where they are (none recorded again, none moved from another slot), and it
    /// answers as the first turn continued without the prefix cache, bit for bit.
    /// </summary>
    [CudaFact]
    public void ADonatedSlotsNextTurnReplaysItsRecordedWindowsOnDirectCuda()
    {
        (List<float[]> Rows, int Adopted, int Relocated, int Recorded) Conversation(bool retain)
        {
            using Qwen35RuntimeModel model = Create(BackendType.Cuda);
            model.RetainCopyMaxBytesForTests = 0;   // donated whatever its size
            var rows = new List<float[]>();
            (int length, int next) = Run(model, "turn1", reserve: 320, steps: 2);
            for (int w = 0; w < 6; w++) next = Window(model, "turn1", ref length, next);
            for (int t = 0; t < 4; t++)
            {
                float[] row = Continue(model, "turn1", new[] { next }, length++);
                next = ArgMax(row);
            }
            string id = "turn1";
            if (retain)
            {
                Assert.True(((IPrefixCacheModel)model).TryCaptureDonate("turn1", "kept", length, out _));
                Assert.Equal(0, model.RetainedByCopy);
                Assert.True(model.TryRebindRetainedCache("kept", "turn2"));
                id = "turn2";
            }
            int adopted = model.RecordingsAdopted, relocated = model.RecordingsRelocated, recorded = model.DecodeRecordings;
            // The next turn: its new prompt rows, then windows and decode steps of the widths the first turn recorded.
            float[] prompt = Continue(model, id, new[] { next, 11, 12 }, length);
            length += 3;
            rows.Add(prompt);
            next = ArgMax(prompt);
            for (int w = 0; w < 4; w++)
            {
                next = Window(model, id, ref length, next, out float[] window);
                rows.Add(window);
            }
            for (int t = 0; t < 4; t++)
            {
                float[] row = Continue(model, id, new[] { next }, length++);
                rows.Add(row);
                next = ArgMax(row);
            }
            return (rows, model.RecordingsAdopted - adopted, model.RecordingsRelocated - relocated, model.DecodeRecordings - recorded);
        }

        var continued = Conversation(retain: false);
        var donated = Conversation(retain: true);
        Assert.True(donated.Adopted > 0, "the next turn adopted none of the recordings its first turn parked");
        Assert.Equal(0, donated.Relocated);
        Assert.Equal(0, donated.Recorded);
        Assert.Equal(continued.Rows.Count, donated.Rows.Count);
        for (int i = 0; i < continued.Rows.Count; i++)
            Assert.True(continued.Rows[i].AsSpan().SequenceEqual(donated.Rows[i]), $"row {i} of the next turn differs on the donated slot");
    }
}
