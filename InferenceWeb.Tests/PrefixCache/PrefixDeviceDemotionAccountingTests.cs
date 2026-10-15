// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using InferenceWeb.Tests.PrefixCache.Fakes;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp.Runtime.Paged;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Scheduling.PrefixCache;

namespace InferenceWeb.Tests.PrefixCache;

public sealed class PrefixDeviceDemotionAccountingTests
{
    [Theory]
    [InlineData("state")]
    [InlineData("pin")]
    [InlineData("donation")]
    public void DemotionPreservesHostPayloadAndPageProtectionWhileUpdatingEveryResourceTotal(string protection)
    {
        var host = new FakePageHost(8);
        var tree = Tk.Tree(Tk.Caps(), host: host, pageHostBytes: 64);
        int scope = Tk.Scope(tree);
        var key = Tk.Key(tree, Tk.Seq(1, 40));
        var payload = Tk.Holder(tree, host: 100, device: 200, state: 300, tokens: 24);
        payload.DeviceDirty = true;
        payload.Footprint = payload.Footprint with { CapacityTokens = 32 };
        var node = Tk.Put(tree, key, 24, scope, payload: payload);
        tree.AttachPages(node, Tk.Pages(host, 3));
        LockReceipt path = tree.AcquirePath(node), state = default;
        if (protection == "state") state = tree.AcquireState(node);
        else if (protection == "pin") tree.Pin(node);
        else tree.MarkDonationPending(node);
        var previous = payload.Footprint;
        var current = previous with { Bytes = previous.Bytes with { DeviceKv = 0 } };
        var before = tree.Cached; long version = tree.Version;
        Assert.True(tree.TryApplyDeviceDemotion(payload.Key, previous, current));
        Assert.Equal(before with { DeviceKv = 0 }, tree.Cached);
        Assert.Equal(tree.Cached, tree.Protected);
        Assert.Equal(tree.Cached, tree.Scopes[scope].Bytes);
        Assert.Equal(current, payload.Footprint);
        Assert.False(payload.DeviceDirty);
        Assert.Same(payload, node.EndState);
        Assert.Equal(1, tree.PayloadKeys.Count);
        Assert.Equal(version + 1, tree.Version);
        Assert.True(tree.PendingReclaim.IsZero);
        Tk.Valid(tree, new(TransactionOpen: protection == "donation"));
        if (protection == "state") tree.Release(ref state);
        else if (protection == "pin") tree.Unpin(node);
        else tree.CancelDonation(node);
        Assert.Equal(new ResourceVector { PoolPages = 3, HostKv = 192 }, tree.Protected);
        tree.Release(ref path);
        Assert.True(tree.Protected.IsZero);
        Assert.Equal(tree.Cached, tree.Evictable);
        Tk.Valid(tree, new(Quiescent: true));
        var reclaimed = tree.DetachEndState(node, ReleaseReason.Evicted);
        Assert.Equal(new ResourceVector { HostKv = 100, StateSnapshot = 300 }, reclaimed);
        Assert.Equal(reclaimed, tree.PendingReclaim);
        tree.DrainReclaimQueue(null);
        Assert.True(tree.PendingReclaim.IsZero);
        Tk.Valid(tree);
    }

    [Fact]
    public void StaleGenerationShapeAndResourceChangesCannotAlterCurrentPayloadOrMakeNegativeCredit()
    {
        var tree = Tk.Tree(Tk.Caps(pages: PageSupport.None)); int scope = Tk.Scope(tree);
        var key = Tk.Key(tree, Tk.Seq(1, 40));
        var original = Tk.Holder(tree, host: 100, device: 200, state: 300, tokens: 24);
        var node = Tk.Put(tree, key, 24, scope, payload: original);
        var previous = original.Footprint;
        var current = previous with { Bytes = previous.Bytes with { DeviceKv = 0 } };
        tree.DetachEndState(node, ReleaseReason.Rollback, enqueue: false);
        var replacement = Tk.Holder(tree, host: 100, device: 200, state: 300, tokens: 24);
        tree.AttachEndState(node, replacement);
        var cached = tree.Cached; long version = tree.Version;
        Assert.False(tree.TryApplyDeviceDemotion(original.Key, previous, current));
        Assert.False(tree.TryApplyDeviceDemotion(replacement.Key, previous with { CapacityTokens = 25 }, current));
        foreach (var invalid in new[]
        {
            current with { Tokens = 23 }, current with { CapacityTokens = 25 }, current with { PositionDelta = 1 },
            current with { Bytes = current.Bytes with { DeviceKv = -1 } },
            current with { Bytes = current.Bytes with { DeviceKv = 201 } },
            current with { Bytes = current.Bytes with { HostKv = 101 } },
            current with { Bytes = current.Bytes with { StateSnapshot = 0 } },
            current with { Bytes = current.Bytes with { PoolPages = 1 } },
            current with { Bytes = current.Bytes with { NativeSlot = 1 } },
        }) Assert.False(tree.TryApplyDeviceDemotion(replacement.Key, previous, invalid));
        Assert.Equal(cached, tree.Cached); Assert.Equal(version, tree.Version);
        Assert.Equal(previous, replacement.Footprint);
        Assert.True(tree.TryApplyDeviceDemotion(replacement.Key, previous, current));
        Assert.False(tree.TryApplyDeviceDemotion(replacement.Key, previous, current)); // Duplicate old generation.
        long updatedVersion = tree.Version;
        Assert.True(tree.TryApplyDeviceDemotion(replacement.Key, current, current)); // Idempotent current state.
        Assert.Equal(updatedVersion, tree.Version);
        Assert.Equal(cached with { DeviceKv = 0 }, tree.Cached);
        Tk.Valid(tree, new(Quiescent: true));
    }

    [Fact]
    public void DeferredCoordinatorReportRefreshesBeforeReadAndInvalidationWinsOverDelayedDemotion()
    {
        using var model = OracleFakes.R(8);
        var config = new SchedulerConfig { BlockSize = 8, NumBlocks = 128, EnablePrefixCaching = true };
        var pool = new BlockPool(128, 8, 0); using var storage = pool.Storage;
        var scheduler = new ContinuousBatchScheduler(config, pool);
        var cache = new PrefixCacheCoordinator(model, pool, scheduler, model.GetPrefixCacheCapabilities(), NullLogger.Instance);
        try
        {
            int scope = Tk.Scope(cache.Tree);
            var payload = Tk.Holder(cache.Tree, host: 100, device: 200, state: 300, tokens: 32);
            payload.DeviceDirty = true;
            var node = Tk.Put(cache.Tree, Tk.Key(cache.Tree, Tk.Seq(1, 48)), 32, scope, payload: payload);
            var previous = payload.Footprint;
            var current = previous with { Bytes = previous.Bytes with { DeviceKv = 0 } };
            cache.OnPayloadDeviceDemoted(payload.Key, previous, current);
            Assert.Equal(200, cache.Tree.Cached.DeviceKv); // Callback never mutates the tree.
            cache.Drain();
            Assert.Equal(0, cache.Tree.Cached.DeviceKv);
            Assert.True(cache.Tree.TryGetNodeByKey(payload.Key, out var currentNode));
            Assert.Same(node, currentNode); Assert.Same(payload, node.EndState);
            Tk.Valid(cache.Tree, new(Quiescent: true));

            cache.OnPayloadDeviceDemoted(payload.Key, current, current);
            cache.OnPayloadInvalidated(payload.Key, InvalidationReason.TrimmedByModel);
            cache.Drain();
            Assert.False(cache.Tree.TryGetNodeByKey(payload.Key, out _));
            Assert.True(cache.Tree.Cached.IsZero);
            Assert.True(cache.Tree.PendingReclaim.IsZero);
            Tk.Valid(cache.Tree, new(Quiescent: true));
        }
        finally { cache.Reset(); cache.Detach(); }
    }
}
