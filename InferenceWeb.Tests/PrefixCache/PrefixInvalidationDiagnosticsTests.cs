// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System;
using System.Collections.Generic;
using System.Linq;
using InferenceWeb.Tests.PrefixCache.Fakes;
using Microsoft.Extensions.Logging;
using TensorSharp.Runtime.Paged;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Scheduling.PrefixCache;

namespace InferenceWeb.Tests.PrefixCache;

public sealed class PrefixInvalidationDiagnosticsTests
{
    [Fact]
    public void ReasonsRemainFifoAndMetadataIsCapturedBeforeWorkerDetach()
    {
        using var fixture = new Fixture(debug: true);
        var a = fixture.Put(1, 16);
        var b = fixture.Put(100, 24);
        var c = fixture.Put(200, 32);
        long firstNodeId = a.Id;
        long version = fixture.Cache.Tree.Version;
        var events = new[]
        {
            (a.EndState!.Key, InvalidationReason.ModelBudgetEviction),
            ("unknown-key", InvalidationReason.NativeSlotReclaimed),
            (b.EndState!.Key, InvalidationReason.BackendRecreated),
            (c.EndState!.Key, InvalidationReason.TrimmedByModel),
        };
        foreach (var (key, reason) in events) fixture.Cache.OnPayloadInvalidated(key, reason);
        Assert.Equal(version, fixture.Cache.Tree.Version);
        Assert.Equal(3, fixture.Cache.Tree.PayloadKeys.Count);
        Assert.Empty(fixture.Logger.Entries); // Model callbacks neither log nor mutate the tree.
        fixture.Logger.BeforeLog = fields =>
        {
            var key = (string)fields["PayloadKey"]!;
            if (key != "unknown-key") Assert.True(fixture.Cache.Tree.TryGetNodeByKey(key, out _));
        };
        long queries = fixture.Model.StateMemberCalls;
        fixture.Cache.Drain();
        Assert.Equal(events.Select(e => e.Item1), fixture.Logger.Entries.Select(e => (string)e["PayloadKey"]!));
        Assert.Equal(events.Select(e => e.Item2), fixture.Logger.Entries.Select(e => (InvalidationReason)e["InvalidationReason"]!));
        Assert.Equal(new[] { true, false, true, true }, fixture.Logger.Entries.Select(e => (bool)e["PayloadIndexed"]!));
        Assert.Equal(firstNodeId, fixture.Logger.Entries[0]["NodeId"]);
        Assert.Equal(fixture.Scope, fixture.Logger.Entries[0]["ScopeIndex"]);
        Assert.Equal(16, fixture.Logger.Entries[0]["NodeDepth"]);
        Assert.Equal(16, fixture.Logger.Entries[0]["PayloadTokens"]);
        Assert.Null(fixture.Logger.Entries[1]["NodeId"]);
        Assert.Null(fixture.Logger.Entries[1]["PayloadTokens"]);
        Assert.DoesNotContain(fixture.Logger.Entries[0].Keys, key =>
            key.Contains("Physical", StringComparison.Ordinal) || key is "IsCheckpoint" or "TriggerPool");
        Assert.Equal(queries, fixture.Model.StateMemberCalls); // Logging queries only tree metadata.
        Assert.Equal(0, fixture.Model.ObservedReleaseCalls); // The model already released these notifications.
        Assert.Empty(fixture.Cache.Tree.PayloadKeys);
        Assert.Equal(3, fixture.Cache.Tree.Counters.Invalidations);
        Tk.Valid(fixture.Cache.Tree, new(Quiescent: true));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void UnknownAndNullNotificationsDoNotBlockTheFollowingValidInvalidation(bool debug)
    {
        using var fixture = new Fixture(debug);
        var victim = fixture.Put(1, 16);
        string key = victim.EndState!.Key;
        fixture.Cache.OnPayloadInvalidated(null!, InvalidationReason.NativeSlotReclaimed);
        fixture.Cache.OnPayloadInvalidated(string.Empty, InvalidationReason.BackendRecreated);
        fixture.Cache.OnPayloadInvalidated("not-indexed", InvalidationReason.ModelBudgetEviction);
        fixture.Cache.OnPayloadInvalidated(key, InvalidationReason.TrimmedByModel);
        Assert.True(fixture.Cache.Tree.TryGetNodeByKey(key, out _));
        fixture.Cache.Drain();
        Assert.False(fixture.Cache.Tree.TryGetNodeByKey(key, out _));
        Assert.Equal(1, fixture.Cache.Tree.Counters.Invalidations);
        Assert.True(fixture.Cache.Tree.Cached.IsZero);
        Assert.True(fixture.Cache.Tree.PendingReclaim.IsZero);
        Assert.Equal(debug ? 4 : 0, fixture.Logger.Entries.Count);
        if (debug)
        {
            Assert.All(fixture.Logger.Entries.Take(3), row => Assert.Equal(false, row["PayloadIndexed"]));
            Assert.Equal(true, fixture.Logger.Entries[3]["PayloadIndexed"]);
        }
        Tk.Valid(fixture.Cache.Tree, new(Quiescent: true));
    }

    [Fact]
    public void DebugToggleDoesNotChangeInvalidationDemotionOrReleaseBehavior()
    {
        Assert.Equal(Lifecycle(debug: false), Lifecycle(debug: true));
    }

    private static (ResourceVector Cached, long Invalidations, long Queries, int Releases) Lifecycle(bool debug)
    {
        using var fixture = new Fixture(debug);
        var victim = fixture.Put(1, 16);
        var survivor = fixture.Put(100, 24);
        string key = victim.EndState!.Key;
        var previous = victim.EndState.Footprint;
        fixture.Cache.OnPayloadDeviceDemoted(key, previous, previous with { Bytes = previous.Bytes with { DeviceKv = 0 } });
        fixture.Cache.OnPayloadInvalidated(key, InvalidationReason.NativeSlotReclaimed);
        var retained = survivor.EndState!.Footprint;
        fixture.Cache.OnPayloadDeviceDemoted(survivor.EndState.Key, retained,
            retained with { Bytes = retained.Bytes with { DeviceKv = 0 } });
        long queries = fixture.Model.StateMemberCalls;
        fixture.Cache.Drain();
        Assert.False(fixture.Cache.Tree.TryGetNodeByKey(key, out _));
        Assert.Equal(0, fixture.Cache.Tree.Cached.DeviceKv);
        Assert.Same(survivor, fixture.Cache.Tree.TryGetNodeByKey(survivor.EndState!.Key, out var found) ? found : null);
        Assert.Equal(queries, fixture.Model.StateMemberCalls);
        Assert.Equal(0, fixture.Model.ObservedReleaseCalls);
        Tk.Valid(fixture.Cache.Tree, new(Quiescent: true));
        return (fixture.Cache.Tree.Cached, fixture.Cache.Tree.Counters.Invalidations,
            fixture.Model.StateMemberCalls, fixture.Model.ObservedReleaseCalls);
    }

    [Fact]
    public void ReportsRaisedByReleaseRemainDeferredAndCannotReviveAnInvalidatedPayload()
    {
        using var fixture = new Fixture(debug: true);
        var trigger = fixture.Put(1, 16);
        var victim = fixture.Put(100, 24);
        string victimKey = victim.EndState!.Key;
        var previous = victim.EndState.Footprint;
        fixture.Model.DuringRelease = () =>
        {
            fixture.Cache.OnPayloadDeviceDemoted(victimKey, previous,
                previous with { Bytes = previous.Bytes with { DeviceKv = 0 } });
            fixture.Cache.OnPayloadInvalidated(victimKey, InvalidationReason.BackendRecreated);
            Assert.True(fixture.Cache.Tree.TryGetNodeByKey(victimKey, out _));
        };
        // Exercise the full eviction path: DetachEndState alone intentionally leaves a
        // tombstone; real eviction also collects an unlocked, payload-less leaf.
        Assert.True(fixture.Cache.Tree.Evict(ResourceClass.HostKv, 1,
            ReleaseReason.Evicted, EvictionTier.Ordinary));
        Assert.False(trigger.InTree);
        Assert.True(fixture.Cache.Tree.TryGetNodeByKey(victimKey, out _));
        Tk.Valid(fixture.Cache.Tree);
        fixture.Cache.Drain();
        Assert.Equal(1, fixture.Model.ObservedReleaseCalls);
        Assert.True(fixture.Cache.Tree.TryGetNodeByKey(victimKey, out _));
        Assert.Empty(fixture.Logger.Entries);
        fixture.Cache.Drain();
        Assert.False(fixture.Cache.Tree.TryGetNodeByKey(victimKey, out _));
        Assert.Equal(1, fixture.Model.ObservedReleaseCalls);
        Assert.Equal(InvalidationReason.BackendRecreated, Assert.Single(fixture.Logger.Entries)["InvalidationReason"]);
        Assert.True(fixture.Cache.Tree.Cached.IsZero);
        Assert.True(fixture.Cache.Tree.PendingReclaim.IsZero);
        Tk.Valid(fixture.Cache.Tree, new(Quiescent: true));
    }

    [Fact]
    public void FailedDebugProviderCannotLoseOrBlockModelInvalidations()
    {
        using var fixture = new Fixture(debug: true);
        var victim = fixture.Put(1, 16);
        string key = victim.EndState!.Key;
        fixture.Logger.FailLog = true;
        fixture.Cache.OnPayloadInvalidated("unknown", InvalidationReason.NativeSlotReclaimed);
        fixture.Cache.OnPayloadInvalidated(key, InvalidationReason.TrimmedByModel);
        fixture.Cache.Drain();
        Assert.Equal(2, fixture.Logger.LogAttempts);
        Assert.False(fixture.Cache.Tree.TryGetNodeByKey(key, out _));
        Assert.Equal(1, fixture.Cache.Tree.Counters.Invalidations);
        Tk.Valid(fixture.Cache.Tree, new(Quiescent: true));
    }

    private sealed class Fixture : IDisposable
    {
        internal readonly ReportingModel Model = new();
        internal readonly CaptureLogger Logger;
        internal readonly BlockPool Pool = new(128, 8, 0);
        internal readonly PrefixCacheCoordinator Cache;
        internal readonly int Scope;

        internal Fixture(bool debug)
        {
            Logger = new(debug);
            var config = new SchedulerConfig { BlockSize = 8, NumBlocks = 128, EnablePrefixCaching = true };
            var scheduler = new ContinuousBatchScheduler(config, Pool);
            Cache = new(Model, Pool, scheduler, Model.GetPrefixCacheCapabilities(), Logger);
            Scope = Tk.Scope(Cache.Tree);
        }

        internal RadixNode Put(int first, int length)
            => Tk.Put(Cache.Tree, Tk.Key(Cache.Tree, Tk.Seq(first, 48)), length, Scope,
                payload: Tk.Holder(Cache.Tree, host: 100, device: 200, state: 300, tokens: length));

        public void Dispose()
        {
            Model.DuringRelease = null;
            Cache.Reset();
            Cache.Detach();
            Pool.Storage.Dispose();
            Model.Dispose();
        }
    }

    private sealed class ReportingModel : OracleModel, IPrefixCacheModel
    {
        internal int ObservedReleaseCalls;
        internal Action? DuringRelease;
        internal ReportingModel() : base(new OracleTraits
        {
            Name = "invalidation-reporting", Class = FamilyClass.R,
            EndState = EndStateSupport.CopyAndDonate, CanCaptureCopy = true,
            AdoptPrimaryOnDisplacement = true, DeviceDirtyOnForward = true,
        }, blockSize: 8) { }

        public new void ReleasePayloads(ReadOnlySpan<string> keys, ReleaseReason reason)
        {
            ObservedReleaseCalls++;
            base.ReleasePayloads(keys, reason);
            DuringRelease?.Invoke();
        }
    }

    private sealed class CaptureLogger(bool enabled) : ILogger
    {
        internal List<Dictionary<string, object?>> Entries { get; } = new();
        internal Action<IReadOnlyDictionary<string, object?>>? BeforeLog;
        internal bool FailLog;
        internal int LogAttempts;
        public IDisposable? BeginScope<TState>(TState state) where TState : notnull => null;
        public bool IsEnabled(LogLevel level) => enabled && level == LogLevel.Debug;
        public void Log<TState>(LogLevel level, EventId id, TState state, Exception? exception,
            Func<TState, Exception?, string> formatter)
        {
            if (!IsEnabled(level)) return;
            LogAttempts++;
            if (FailLog) throw new InvalidOperationException("debug provider failed");
            var fields = ((IEnumerable<KeyValuePair<string, object?>>)state!).ToDictionary(x => x.Key, x => x.Value);
            BeforeLog?.Invoke(fields);
            Entries.Add(fields);
        }
    }
}
