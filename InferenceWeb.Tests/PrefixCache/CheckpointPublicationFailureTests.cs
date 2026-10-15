// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using InferenceWeb.Tests.PrefixCache.Fakes;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp.Memory;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Paged;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Scheduling.PrefixCache;

namespace InferenceWeb.Tests.PrefixCache;

/// <summary>Actual coordinator/tree transactions and budget-owned oracle state.
/// Injected allocation/free failures do not imply native allocator qualification.</summary>
public sealed class CheckpointPublicationFailureTests
{
    [Fact]
    public void PublicationObjectAllocationFailureNeverCapturesPhysicalState()
    {
        using var fixture = new Fixture();
        ulong source = fixture.Model.ActiveState;
        var failure = new OutOfMemoryException("injected pre-capture metadata allocation failure");
        fixture.Cache.BeforeCheckpointPublicationAllocation = () => throw failure;
        Assert.Same(failure, Assert.Throws<OutOfMemoryException>(() => fixture.Capture()));
        Assert.Equal(0, fixture.Model.CaptureCalls);
        Assert.Equal(0, fixture.Model.PhysicalReleases);
        Assert.Empty(fixture.Model.RetainedPayloadKeys);
        Assert.Equal(0, fixture.Model.CommittedBytes);
        Assert.Equal(source, fixture.Model.ActiveState);
        fixture.AssertTreeEmptyAndValid();
    }

    [Fact]
    public void TreeCapacityOomAfterCaptureReleasesTheCopyAndRetryRestoresExactState()
    {
        using var fixture = new Fixture();
        ulong source = fixture.Model.ActiveState;
        var failure = new OutOfMemoryException("injected tree publication capacity growth failure");
        fixture.Cache.Tree.BeforeEndStatePublicationCapacity = () => throw failure;
        Assert.Same(failure, Assert.Throws<OutOfMemoryException>(() => fixture.Capture()));
        Assert.Equal(1, fixture.Model.CaptureCalls);
        Assert.Equal(1, fixture.Model.PhysicalReleases);
        Assert.Empty(fixture.Model.RetainedPayloadKeys);
        Assert.Equal(0, fixture.Model.CommittedBytes);
        Assert.Equal(source, fixture.Model.ActiveState);
        fixture.AssertTreeEmptyAndValid();

        fixture.Cache.Tree.BeforeEndStatePublicationCapacity = null;
        fixture.Capture();
        string key = Assert.Single(fixture.Model.RetainedPayloadKeys);
        Assert.Equal(BudgetedModel.CopyBytes, fixture.Model.CommittedBytes);
        Assert.True(fixture.Cache.Tree.TryGetNodeByKey(key, out var node));
        Assert.Equal(16, node.Depth);
        Assert.True(fixture.Model.TryMaterialize(new(MaterializeOp.Clone, key, "restored", 16, 16)));
        fixture.Model.BindSequenceCache("restored");
        float[] restored = fixture.Model.Forward([149, 150]);
        using var cold = OracleFakes.R(8);
        float[] expected = cold.Forward(fixture.Sequence.PromptTokens.Take(16).Concat(new[] { 149, 150 }).ToArray());
        Assert.Equal(expected, restored);
        fixture.Model.OnSequenceReleased("restored");
        Assert.Empty(PrefixTreeInvariantChecker.CheckOnce(fixture.Cache.Tree, new(Quiescent: true)));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void RefusedRollbackKeepsTheRealModelOwnerChargedUntilPhysicalRelease(bool throws)
    {
        using var fixture = new Fixture();
        ulong source = fixture.Model.ActiveState;
        fixture.Model.RefuseFree = !throws;
        fixture.Model.ThrowFree = throws;
        var failure = new OutOfMemoryException("injected post-copy tree allocation failure");
        fixture.Cache.Tree.BeforeEndStatePublicationCapacity = () => throw failure;
        if (throws) Assert.Throws<IOException>(() => fixture.Capture());
        else Assert.Same(failure, Assert.Throws<OutOfMemoryException>(() => fixture.Capture()));

        string key = Assert.Single(fixture.Model.RetainedPayloadKeys);
        Assert.Equal(1, fixture.Model.CaptureCalls);
        Assert.Equal(0, fixture.Model.PhysicalReleases);
        Assert.Equal(BudgetedModel.CopyBytes, fixture.Model.CommittedBytes);
        Assert.True(fixture.Model.CanMaterialize(key, 16, 16));
        Assert.Equal(source, fixture.Model.ActiveState);
        fixture.AssertTreeEmptyAndValid();

        fixture.Model.RefuseFree = fixture.Model.ThrowFree = false;
        ((IPrefixCacheModel)fixture.Model).ReleasePayloads([key], ReleaseReason.Invalidated);
        Assert.Empty(fixture.Model.RetainedPayloadKeys);
        Assert.Equal(1, fixture.Model.PhysicalReleases);
        Assert.Equal(0, fixture.Model.CommittedBytes);
    }

    private sealed class Fixture : IDisposable
    {
        internal readonly BudgetedModel Model = new();
        internal readonly BlockPool Pool = new(128, 8, 64);
        internal readonly SequenceState Sequence = new("producer", Enumerable.Range(1, 20).ToArray(), 3, 8,
            SamplingConfig.Greedy, sharedPrefixTokens: 16, cacheScope: "test-chat", publicCheckpointBoundaries: [16]);
        internal readonly PrefixCacheCoordinator Cache;
        internal Fixture()
        {
            var config = new SchedulerConfig
            {
                BlockSize = 8, NumBlocks = 128, MaxNumRunningSequences = 1,
                MaxNumBatchedTokens = 64, SoloPrefillChunkSize = 64,
                EnablePrefixCaching = true, StopRepetition = false,
            };
            var scheduler = new ContinuousBatchScheduler(config, Pool);
            Cache = new(Model, Pool, scheduler, Model.GetPrefixCacheCapabilities(), NullLogger.Instance);
            foreach (var block in Pool.AllocateNew(3)!) Sequence.BlockTable.AppendBlock(block);
            Model.BindSequenceCache(Sequence.RequestId);
            Model.Forward(Sequence.PromptTokens.Take(16).ToArray());
            Sequence.SetComputedTokensForPrefixAdoption(16);
        }
        internal void Capture() => Cache.CaptureCheckpoint(Sequence);
        internal void AssertTreeEmptyAndValid()
        {
            Assert.Empty(Cache.Tree.PayloadKeys);
            Assert.True(Cache.Tree.Cached.IsZero);
            Assert.True(Cache.Tree.Protected.IsZero);
            Assert.True(Cache.Tree.PendingReclaim.IsZero);
            Assert.False(Cache.Tree.Root.HasPayload);
            Assert.Empty(PrefixTreeInvariantChecker.CheckOnce(Cache.Tree, new(Quiescent: true)));
        }
        public void Dispose()
        {
            Model.RefuseFree = Model.ThrowFree = false;
            Cache.Reset(); Cache.Detach();
            Model.OnSequenceReleased(Sequence.RequestId);
            Sequence.BlockTable.ReleaseAll(Pool);
            Model.Dispose();
            Assert.Equal(Pool.NumBlocks, Pool.NumFreeBlocks);
        }
    }

    private sealed class BudgetedModel : OracleModel, IPrefixCacheModel
    {
        // Synthetic complete-state physical ownership, backed by a real quota
        // child which survives root disposal. No device/heap-byte claim.
        internal const long CopyBytes = 4096;
        private readonly MemoryBudget _budget = new([new("test-copy", 1 << 20)]);
        private readonly Dictionary<string, BudgetReservation> _owners = new(StringComparer.Ordinal);
        internal int CaptureCalls, PhysicalReleases;
        internal bool RefuseFree, ThrowFree;
        internal long CommittedBytes => _budget.Snapshot().Single().Committed;
        internal BudgetedModel() : base(new OracleTraits
        {
            Name = "budgeted-publication-recurrent", Class = FamilyClass.R,
            EndState = EndStateSupport.CopyAndDonate, CanCaptureCopy = true,
            AdoptPrimaryOnDisplacement = true, DeviceDirtyOnForward = true,
            Truncation = TruncationKind.None, Pages = PageSupport.None,
        }, 8) { }

        bool IPrefixCacheModel.TryCaptureCopy(string requestId, string payloadKey, out PayloadFootprint footprint)
        {
            ++CaptureCalls;
            _owners.EnsureCapacity(_owners.Count + 1);
            using var root = _budget.Reserve([new("test-copy", CopyBytes)]);
            var child = root.TryTake([new("test-copy", CopyBytes)])!;
            bool published = false;
            try
            {
                if (!TryCaptureCopy(requestId, payloadKey, out footprint)) return false;
                child.Commit();
                _owners.Add(payloadKey, child);
                footprint = footprint with { Bytes = new ResourceVector { StateSnapshot = CopyBytes } };
                published = true;
                return true;
            }
            finally { if (!published) child.Dispose(); }
        }

        void IPrefixCacheModel.ReleasePayloads(ReadOnlySpan<string> payloadKeys, ReleaseReason reason)
        {
            if (ThrowFree) throw new IOException("injected physical copy release failure");
            if (RefuseFree) return; // Physical state and its quota remain owned.
            ReleasePayloads(payloadKeys, reason);
            foreach (string key in payloadKeys)
                if (_owners.Remove(key, out var owner)) { owner.Dispose(); ++PhysicalReleases; }
        }

        public new void Dispose()
        {
            RefuseFree = ThrowFree = false;
            ((IPrefixCacheModel)this).ReleasePayloads(_owners.Keys.ToArray(), ReleaseReason.Reset);
            base.Dispose();
            Assert.Equal(0, CommittedBytes);
        }
    }
}
