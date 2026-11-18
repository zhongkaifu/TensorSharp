// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Issue #376: on one direct-CUDA device the shared runtime makes room before it allocates a slot, in a fixed order (pooled
// slots, parked recordings, the oldest finished conversations, the cached graphs last), shows the prefix cache the device's
// deficit, lets the CUDA allocator ask it for memory, and refuses a request whose slot can never fit before any of it runs.
// The managed CPU executor runs the same code through the model's device stand-in (DeviceRoomForTests); the CUDA tests run it
// on the device.
using Microsoft.Extensions.Logging.Abstractions;

using InferenceWeb.Tests.PrefixCache.Fakes;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Scheduling.PrefixCache;

namespace InferenceWeb.Tests;

public sealed class DeviceRoomTests : IDisposable
{
    private const long GB = 1L << 30;
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-device-room-" + Guid.NewGuid().ToString("N"));
    private readonly int[] _prompt = Enumerable.Range(0, 20).Select(i => 40 + i % 7).ToArray();

    public DeviceRoomTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    private Qwen35RuntimeModel Create(BackendType backend = BackendType.Cpu)
    {
        using var dtype = KvCacheDtypeConfig.ForCurrentFlowInTests(backend == BackendType.Cpu ? KvCacheDtype.F32 : KvCacheDtype.F16);
        string path = new Qwen35DenseSyntheticModelBuilder { Layers = 8, UntiedOutput = true }.Write(Path.Combine(_dir, "q35-dense.gguf"));
        return new Qwen35RuntimeModel(path, backend, 1, null, 1);
    }

    private void Prompt(Qwen35RuntimeModel model, string id, int reserve = 64)
    {
        var unified = (IUnifiedBatchModel)model;
        Assert.Equal(0, unified.EnsureSequenceSlot(id, reserve));
        Assert.True(unified.TryForwardUnified(new[] { id }, new[] { _prompt }, new[] { 0 }, new float[1][]));
    }

    private void Continue(Qwen35RuntimeModel model, string id)
        => Assert.True(((IUnifiedBatchModel)model).TryForwardUnified(new[] { id }, new[] { new[] { 7 } }, new[] { _prompt.Length },
            new float[1][]), $"{id} did not continue");

    /// <summary>A slot's room comes from the pooled slots first (no conversation lost), then from the conversations the prefix
    /// cache keeps, the oldest first, each reported as reclaimed and only until there is room; a running request's slot is
    /// never touched.</summary>
    [Fact]
    public void MakingRoomForASlotGivesUpPooledSlotsThenTheOldestConversations()
    {
        using Qwen35RuntimeModel model = Create();
        var sink = new RecordingPayloadSink();
        model.AttachPrefixCache(sink);
        Prompt(model, "run");
        string[] keys = { "k0", "k1", "k2" };
        foreach (string key in keys)
        {
            Prompt(model, key);
            Assert.True(model.RetainSequenceCacheAs(key, key));
        }
        // Two finished requests' slots in the pool (made together: a new request takes a pooled slot).
        Prompt(model, "p1");
        Prompt(model, "p2");
        model.OnSequenceReleased("p1");
        model.OnSequenceReleased("p2");
        Assert.Equal(2, model.SlotPoolForTests.Pooled);

        // Room for the new request's 2,048-cell slot (beside the headroom and a request's recordings) only once the pool is
        // empty and one conversation is left.
        var seen = new List<(int Pooled, int Retained)>();
        model.DeviceRoomForTests = () =>
        {
            (int pooled, int retained) = (model.SlotPoolForTests.Pooled, model.RetainedPayloadKeys.Count);
            seen.Add((pooled, retained));
            return (pooled == 0 && retained <= 1 ? 8 * GB : 2 * GB, 16 * GB);
        };
        Assert.Equal(0, ((IUnifiedBatchModel)model).EnsureSequenceSlot("new", 2048));

        Assert.Equal(new[] { ("k0", InvalidationReason.NativeSlotReclaimed), ("k1", InvalidationReason.NativeSlotReclaimed) }, sink.Reports);
        Assert.Equal(new[] { "k2" }, model.RetainedPayloadKeys);
        Assert.Equal(0, model.SlotPoolForTests.Pooled);
        Assert.DoesNotContain(seen, s => s.Pooled > 0 && s.Retained < keys.Length);   // no conversation went while a pooled slot was left
        Assert.Equal(1, model.DeviceRoomStages[2]);
        Assert.Equal(1, model.DeviceRoomStages[5]);
        Assert.Equal(0, model.DeviceRoomStages[6]);
        Continue(model, "run");
        Assert.True(((IUnifiedBatchModel)model).TryForwardUnified(new[] { "new" }, new[] { _prompt }, new[] { 0 }, new float[1][]));
    }

    /// <summary>A prompt-end checkpoint's copy releases the oldest conversation the prefix cache keeps rather than being
    /// declined (declined, the conversation's next turn prefilled its whole prompt again); it is declined only where releasing
    /// them makes no room.</summary>
    [Fact]
    public void ACheckpointCopyReleasesTheOldestConversationRatherThanBeingDeclined()
    {
        using Qwen35RuntimeModel model = Create();
        var sink = new RecordingPayloadSink();
        var cache = (IPrefixCacheModel)model;
        cache.AttachPrefixCache(sink);
        foreach (string key in new[] { "k0", "k1" })
        {
            Prompt(model, key);
            Assert.True(model.RetainSequenceCacheAs(key, key));
        }
        Prompt(model, "run");
        model.DeviceRoomForTests = () => (model.RetainedPayloadKeys.Count <= 1 ? 8 * GB : 2 * GB, 16 * GB);

        Assert.True(cache.TryCaptureCopy("run", "checkpoint", out _));
        Assert.Equal(new[] { ("k0", InvalidationReason.NativeSlotReclaimed) }, sink.Reports);
        Assert.Equal(new[] { "checkpoint", "k1" }, model.RetainedPayloadKeys.OrderBy(k => k, StringComparer.Ordinal));
        Assert.Equal(0, model.CheckpointsDeclinedForRoom);

        model.DeviceRoomForTests = () => (2 * GB, 16 * GB);
        Assert.False(cache.TryCaptureCopy("run", "checkpoint2", out _));
        Assert.Equal(1, model.CheckpointsDeclinedForRoom);
        Continue(model, "run");
    }

    /// <summary>A new request whose slot does not fit waits beside a running one (the scheduler re-queues it); alone, it is
    /// refused with the numbers before any of it runs where its slot does not fit beside the headroom, and runs where it does
    /// (narrower prompt steps, decode that records only where it finds room).</summary>
    [Fact]
    public void ANewRequestWaitsBesideARunningOneAndIsRefusedWhenItsSlotCannotFitAlone()
    {
        using Qwen35RuntimeModel model = Create();
        var unified = (IUnifiedBatchModel)model;
        Prompt(model, "run");
        model.DeviceRoomForTests = () => (GB / 2, 16 * GB);

        Assert.Throws<SequenceSlotUnavailableException>(() => unified.EnsureSequenceSlot("second", 2048));
        Assert.Equal(1, model.SlotWaits);
        Assert.False(model.HasFusedSequenceCache("second"));
        Continue(model, "run");

        model.OnSequenceReleased("run");
        var refused = Assert.Throws<InvalidOperationException>(() => unified.EnsureSequenceSlot("second", 2048));
        Assert.Contains("2,048 cells", refused.Message);
        Assert.Contains("refused before any of it ran", refused.Message);
        Assert.Equal(1, model.SlotRefusals);
        Assert.False(model.HasFusedSequenceCache("second"));

        // Beside its headroom, though short of a full prompt step's and its recordings' room: it runs.
        model.DeviceRoomForTests = () => (GB + GB / 2, 16 * GB);
        Assert.Equal(0, unified.EnsureSequenceSlot("second", 2048));
        Assert.True(unified.TryForwardUnified(new[] { "second" }, new[] { _prompt }, new[] { 0 }, new float[1][]));
        Assert.Equal(1, model.SlotRefusals);
    }

    /// <summary>The prefix cache is shown what the device lacks of its headroom (the larger of 1 GB and 1/32 of it), and
    /// nothing while the device holds it.</summary>
    [Fact]
    public void TheDeviceDeficitIsWhatTheDeviceLacksOfItsHeadroom()
    {
        using Qwen35RuntimeModel model = Create();
        Assert.Equal(0, model.DeviceSlotDeficitBytes);   // a device the model does not manage
        model.DeviceRoomForTests = () => (2 * GB, 16 * GB);
        Assert.Equal(0, model.DeviceSlotDeficitBytes);
        model.DeviceRoomForTests = () => (GB, 16 * GB);
        Assert.Equal(0, model.DeviceSlotDeficitBytes);
        model.DeviceRoomForTests = () => (GB / 4, 16 * GB);
        Assert.Equal(3 * GB / 4, model.DeviceSlotDeficitBytes);
        model.DeviceRoomForTests = () => (GB, 64 * GB);
        Assert.Equal(GB, model.DeviceSlotDeficitBytes);
    }

    /// <summary>The CUDA allocator's reclaimer (rung 3 of its out-of-memory ladder) gives up pooled slots and then the
    /// conversations the prefix cache keeps, each reported, but nothing before a step ran, off the engine's thread, while a
    /// graph is captured, from inside itself or while slots are being changed; a running request's slot never.</summary>
    [Fact]
    public void TheAllocatorsReclaimerActsOnlyOnTheEngineThreadOutsideCapturesAndSlotChanges()
    {
        using Qwen35RuntimeModel model = Create();
        var sink = new ReenteringSink();
        model.AttachPrefixCache(sink);
        const long all = 1L << 40;
        Assert.Equal(0, model.ReclaimForAllocator(all));   // no step yet: no engine thread

        Prompt(model, "run");
        Prompt(model, "k0");
        Assert.True(model.RetainSequenceCacheAs("k0", "k0"));
        Prompt(model, "pooled");
        model.OnSequenceReleased("pooled");
        Assert.Equal(1, model.SlotPoolForTests.Pooled);

        long offThread = -1;
        var other = new Thread(() => offThread = model.ReclaimForAllocator(all));
        other.Start();
        other.Join();
        Assert.Equal(0, offThread);
        model.GraphCaptureForTests = () => true;
        Assert.Equal(0, model.ReclaimForAllocator(all));
        model.GraphCaptureForTests = null;
        using (model.MutatingSlotsForTests())
            Assert.Equal(0, model.ReclaimForAllocator(all));
        Assert.Equal(1, model.SlotPoolForTests.Pooled);
        Assert.Equal(new[] { "k0" }, model.RetainedPayloadKeys);
        Assert.Empty(sink.Reports);

        sink.Inner = () => model.ReclaimForAllocator(all);
        Assert.True(model.ReclaimForAllocator(all) > 0);
        Assert.Equal(new[] { ("k0", InvalidationReason.NativeSlotReclaimed) }, sink.Reports);
        Assert.Equal(new[] { 0L }, sink.InnerResults);   // asked again from inside itself: nothing
        Assert.Equal(0, model.SlotPoolForTests.Pooled);
        Assert.Empty(model.RetainedPayloadKeys);
        Continue(model, "run");   // the freed slots' steps are forgotten at this entry
    }

    /// <summary>A host warms the kernels on its loading thread (StartupModelLoader, the CLI), and the warm-up makes its pooled
    /// slot there (WarmPooledSlot, through EnsureSequenceSlot); the engine's own worker thread runs every later step. The
    /// reclaimer acts on the thread that runs the steps: bound to the first thread that made a slot, it never released anything
    /// in a server (rung 3 went straight to the last resort).</summary>
    [Fact]
    public void TheReclaimerActsOnTheThreadThatRunsTheStepsAfterAWarmUpOnTheLoadingThread()
    {
        using Qwen35RuntimeModel model = Create();
        Exception failed = null;
        var loader = new Thread(() =>
        {
            try { model.WarmUpKernels(); }
            catch (Exception ex) { failed = ex; }
        });
        loader.Start();
        loader.Join();
        Assert.Null(failed);
        Assert.True(model.SlotPoolForTests.Pooled > 0, "the warm-up made no pooled slot (it did not go through EnsureSequenceSlot)");

        var sink = new RecordingPayloadSink();
        model.AttachPrefixCache(sink);   // the engine attaches its prefix cache after the warm-up
        Prompt(model, "run");
        Prompt(model, "k0");
        Assert.True(model.RetainSequenceCacheAs("k0", "k0"));
        Assert.True(model.ReclaimForAllocator(1L << 40) > 0, "the reclaimer stayed bound to the loading thread");
        Assert.Contains(("k0", InvalidationReason.NativeSlotReclaimed), sink.Reports);
        Continue(model, "run");
    }

    /// <summary>Behind an engine, the conversations a new request's room released are dropped from the prefix tree as the
    /// model reported them: the tree stays consistent and names no payload the model no longer holds.</summary>
    [Fact]
    public async Task ConversationsReleasedForRoomLeaveThePrefixTreeConsistent()
    {
        using Qwen35RuntimeModel model = Create();
        var config = new SchedulerConfig
        {
            MaxNumBatchedTokens = 256,
            MaxNumRunningSequences = 4,
            NumBlocks = 64,
            BlockSize = 16,
            EnablePrefixCaching = true,
            StopRepetition = false,
        };
        using var engine = new InferenceEngine(model, config, NullLogger.Instance);
        static async Task Serve(InferenceEngine engine, string id, int[] prompt, string scope)
        {
            var handle = engine.SubmitRequest(new SequenceState(id, prompt, 4, 16, SamplingConfig.Greedy, cacheScope: scope));
            await foreach (int _ in handle.Tokens.ReadAllAsync()) { }
            await handle.Completion.WaitAsync(TimeSpan.FromMinutes(2));
            await engine.WhenStepSettledAsync().WaitAsync(TimeSpan.FromMinutes(1));
        }
        for (int c = 0; c < 3; c++)
            await Serve(engine, $"c{c}", Enumerable.Range(0, 40).Select(i => 30 + (i * 7 + c * 11) % 200).ToArray(), $"s{c}");
        Assert.True(model.RetainedPayloadKeys.Count >= 3, $"{model.RetainedPayloadKeys.Count} payloads kept");

        model.DeviceRoomForTests = () => (model.RetainedPayloadKeys.Count <= 1 ? 8 * GB : 2 * GB, 16 * GB);
        await Serve(engine, "long", Enumerable.Range(0, 600).Select(i => 30 + (i * 13) % 200).ToArray(), "s3");

        Assert.True(model.DeviceRoomStages[5] > 0, "no conversation was released for the long request's slot");
        PrefixTree tree = engine.RadixCache!.Tree;
        Assert.Empty(PrefixTreeInvariantChecker.CheckOnce(tree));
        var held = new HashSet<string>(model.RetainedPayloadKeys, StringComparer.Ordinal);
        Assert.All(tree.PayloadKeys, key => Assert.Contains(key, held));
    }

    /// <summary>A sink that asks the reclaimer again from inside its report (re-entry), recording what it got.</summary>
    private sealed class ReenteringSink : IPrefixPayloadSink
    {
        internal List<(string Key, InvalidationReason Reason)> Reports { get; } = new();
        internal List<long> InnerResults { get; } = new();
        internal Func<long> Inner { get; set; }

        public void OnPayloadInvalidated(string payloadKey, InvalidationReason reason)
        {
            Reports.Add((payloadKey, reason));
            if (Inner != null) InnerResults.Add(Inner());
        }
    }

    // ---------------------------------------------------------------- on the device

    /// <summary>On one direct-CUDA device the same order holds with the device's own allocator under the stand-in: pooled
    /// slots, then the oldest conversations; the running request continues.</summary>
    [CudaFact]
    public void MakingRoomForASlotGivesUpPooledSlotsThenTheOldestConversationsOnDirectCuda()
    {
        using Qwen35RuntimeModel model = Create(BackendType.Cuda);
        var sink = new RecordingPayloadSink();
        model.AttachPrefixCache(sink);
        Prompt(model, "run");
        foreach (string key in new[] { "k0", "k1" })
        {
            Prompt(model, key);
            Assert.True(model.RetainSequenceCacheAs(key, key));
        }
        Prompt(model, "p1");
        model.OnSequenceReleased("p1");
        Assert.Equal(1, model.SlotPoolForTests.Pooled);
        model.DeviceRoomForTests = () =>
            (model.SlotPoolForTests.Pooled == 0 && model.RetainedPayloadKeys.Count <= 1 ? 8 * GB : 2 * GB, 16 * GB);
        Assert.Equal(0, ((IUnifiedBatchModel)model).EnsureSequenceSlot("new", 2048));
        Assert.Equal(new[] { ("k0", InvalidationReason.NativeSlotReclaimed) }, sink.Reports);
        Assert.Equal(1, model.DeviceRoomStages[2]);
        Assert.Equal(1, model.DeviceRoomStages[5]);
        Continue(model, "run");
    }
}

/// <summary>Fills the device, so it runs alone (no other test may allocate on the GPU meanwhile).</summary>
[CollectionDefinition(Name, DisableParallelization = true)]
public sealed class DeviceFillingCollection
{
    public const string Name = "Device-filling CUDA tests";
}

[Collection(DeviceFillingCollection.Name)]
public sealed class DeviceRoomReclaimerCudaTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-device-room-cuda-" + Guid.NewGuid().ToString("N"));
    private readonly int[] _prompt = Enumerable.Range(0, 20).Select(i => 40 + i % 7).ToArray();

    public DeviceRoomReclaimerCudaTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    private void Prompt(Qwen35RuntimeModel model, string id)
    {
        var unified = (IUnifiedBatchModel)model;
        Assert.Equal(0, unified.EnsureSequenceSlot(id, 64));
        Assert.True(unified.TryForwardUnified(new[] { id }, new[] { _prompt }, new[] { 0 }, new float[1][]));
    }

    private void Continue(Qwen35RuntimeModel model, string id)
        => Assert.True(((IUnifiedBatchModel)model).TryForwardUnified(new[] { id }, new[] { new[] { 7 } }, new[] { _prompt.Length },
            new float[1][]), $"{id} did not continue");

    /// <summary>An allocation no site sized that finds the device full climbs the allocator's ladder to the model's reclaimer
    /// (rung 3), which releases the conversations the prefix cache keeps; the allocation then succeeds without the last
    /// resort, and the running request continues.</summary>
    [CudaFact]
    public void AnAllocationThatFindsTheDeviceFullReleasesRetainedConversationsThroughTheReclaimer()
    {
        Qwen35RuntimeModel model;
        using (KvCacheDtypeConfig.ForCurrentFlowInTests(KvCacheDtype.F16))
            model = new Qwen35RuntimeModel(new Qwen35DenseSyntheticModelBuilder { Layers = 8, UntiedOutput = true }
                .Write(Path.Combine(_dir, "q35-dense.gguf")), BackendType.Cuda, 1, null, 1);
        using var owned = model;
        var sink = new RecordingPayloadSink();
        model.AttachPrefixCache(sink);
        var unified = (IUnifiedBatchModel)model;
        Prompt(model, "run");
        string[] keys = { "k0", "k1", "k2", "k3" };
        foreach (string key in keys)
        {
            Assert.Equal(0, unified.EnsureSequenceSlot(key, 4096));
            Assert.True(unified.TryForwardUnified(new[] { key }, new[] { _prompt }, new[] { 0 }, new float[1][]));
            Assert.True(model.RetainSequenceCacheAs(key, key));
        }
        var cuda = (TensorSharp.Cuda.CudaAllocator)model.Mesh.Allocator(0, 0);
        Assert.Contains(cuda.RegisteredReclaimers, r => r.GetType().Name == "DeviceRoomReclaimer");
        cuda.TrimPool();
        var ballast = new List<Tensor>();
        try
        {
            // Fill the device to within 16 MiB of full, in 256 MiB pieces and then smaller ones, with the reclaimer away (a
            // piece that found no room must not release the conversations).
            model.DetachPrefixCache();
            Assert.DoesNotContain(cuda.RegisteredReclaimers, r => r.GetType().Name == "DeviceRoomReclaimer");
            for (long piece = 256L << 20; piece >= 1L << 20; piece /= 2)
                while (cuda.GetMemoryInfo().freeBytes > piece + (16L << 20))
                    ballast.Add(new Tensor(cuda, DType.UInt8, piece));
            model.AttachPrefixCache(sink);
            Assert.Equal(keys, model.RetainedPayloadKeys.OrderBy(k => k, StringComparer.Ordinal));
            TensorSharp.Cuda.CudaAllocatorStats before = cuda.GetStats();
            long asked = 48L << 20;   // past the free memory and the pool's cache, within what the conversations hold
            using (var tensor = new Tensor(cuda, DType.UInt8, asked))
            {
                TensorSharp.Cuda.CudaAllocatorStats after = cuda.GetStats();
                Assert.True(after.OomReclaims > before.OomReclaims, $"rung 3 did not make the room: {after}");
                Assert.Equal(before.FullDrains, after.FullDrains);
            }
            Assert.NotEmpty(sink.Reports);
            Assert.All(sink.Reports, r => Assert.Equal(InvalidationReason.NativeSlotReclaimed, r.Reason));
            Assert.Equal(sink.Reports[0].Key, keys[0]);   // the oldest first
        }
        finally
        {
            foreach (Tensor t in ballast) t.Dispose();
        }
        Continue(model, "run");
    }
}
