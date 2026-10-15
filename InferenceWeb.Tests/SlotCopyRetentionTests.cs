using TensorSharp.Models;

namespace InferenceWeb.Tests;

public sealed class SlotCopyRetentionTests
{
    private sealed class State(int head, long bytes, byte[] kv, float[] recurrent)
    {
        public int Head = head;
        public long OwnedBytes = bytes;
        public byte[] Kv = kv;
        public float[] Recurrent = recurrent;
    }

    private sealed class Store : SlotCopyRetention.IStore
    {
        public readonly Dictionary<int, State> Slots = new()
        {
            [0] = new(57, 4096, [1, 3, 7, 11], [0.25f, -0.5f, 2.0f]),
        };
        public readonly List<int> Freed = [];
        public int Active = 0;
        public int AllocCalls;
        public int CopyCalls;
        public int LastAllocated = -1;
        public int LastExpectedHead = -1;
        public long DestinationBytes = 6144;
        public bool RefuseAlloc, RefuseCopy, RefuseFree;
        public Exception? CopyError, ByteError, FreeError;
        public Action? BeforeAlloc;
        public long OwnedBytes => Slots.Values.Sum(slot => slot.OwnedBytes);

        public int Alloc()
        {
            AllocCalls++;
            BeforeAlloc?.Invoke();
            if (RefuseAlloc) return -1;
            LastAllocated = Slots.Keys.Max() + 1;
            Slots.Add(LastAllocated, new(0, DestinationBytes, [0, 0, 0, 0], [0, 0, 0]));
            return LastAllocated;
        }

        public bool CopyInto(int src, int dst, int expectedHead)
        {
            CopyCalls++;
            LastExpectedHead = expectedHead;
            if (CopyError != null) throw CopyError;
            if (!Slots.TryGetValue(src, out var source) || !Slots.TryGetValue(dst, out var destination)
                || source.Head != expectedHead)
                return false;
            if (RefuseCopy)
            {
                destination.Kv[0] = 99; // A refused copy may already have written its owned destination.
                return false;
            }
            destination.Kv = (byte[])source.Kv.Clone();
            destination.Recurrent = (float[])source.Recurrent.Clone();
            destination.Head = expectedHead;
            return true;
        }

        public long Bytes(int slot)
        {
            if (slot == LastAllocated && ByteError != null) throw ByteError;
            return Slots.TryGetValue(slot, out var state) ? state.OwnedBytes : 0;
        }

        public bool Free(int slot)
        {
            Freed.Add(slot);
            if (FreeError != null) throw FreeError;
            return !RefuseFree && Slots.Remove(slot);
        }
    }

    private sealed class PublicationComparer : IEqualityComparer<string>
    {
        public bool Throw;
        public bool Equals(string? left, string? right) => StringComparer.Ordinal.Equals(left, right);
        public int GetHashCode(string value)
            => Throw ? throw new OutOfMemoryException("injected metadata publication failure") : StringComparer.Ordinal.GetHashCode(value);
    }

    [Fact]
    public void CaptureReportsActualDestinationBytesAndCopiesTheWholeStateAtTheExactHead()
    {
        var store = new Store();
        var retained = new OrderedDictionary<string, SlotRetention.RetainedSlot>();
        Assert.True(SlotCopyRetention.Capture(store, retained, "public-seed", 0, 57, out long bytes));
        var seed = retained["public-seed"];
        Assert.Equal(57, seed.Tokens);
        Assert.True(seed.IsCopy);
        Assert.NotEqual(0, seed.Slot);
        Assert.Equal(6144, bytes); // Not the source's4096-byte charge or an estimated geometry.
        Assert.Equal(57, store.LastExpectedHead);
        Assert.Equal(store.Slots[0].Kv, store.Slots[seed.Slot].Kv);
        Assert.Equal(store.Slots[0].Recurrent, store.Slots[seed.Slot].Recurrent);
        Assert.NotSame(store.Slots[0].Kv, store.Slots[seed.Slot].Kv);
        Assert.NotSame(store.Slots[0].Recurrent, store.Slots[seed.Slot].Recurrent);
        Assert.Equal(0, store.Active);
        Assert.Empty(store.Freed);
    }

    [Fact]
    public void PublicSeedStaysImmutableWhenItsFormerActiveSourceContinues()
    {
        var store = new Store();
        var retained = new OrderedDictionary<string, SlotRetention.RetainedSlot>();
        Assert.True(SlotCopyRetention.Capture(store, retained, "seed", 0, 57, out _));
        int seed = retained["seed"].Slot;
        store.Slots[0].Head++;
        store.Slots[0].Kv[0] = 77;
        store.Slots[0].Recurrent[0] = 9;
        Assert.Equal(57, store.Slots[seed].Head);
        Assert.Equal(1, store.Slots[seed].Kv[0]);
        Assert.Equal(0.25f, store.Slots[seed].Recurrent[0]);
        Assert.Equal(0, store.Active);
    }

    [Fact]
    public void ThreeConcurrentClonesOwnIndependentSlotsAndCancellingOnePreservesTheOthers()
    {
        var store = new Store();
        var retained = new OrderedDictionary<string, SlotRetention.RetainedSlot>();
        Assert.True(SlotCopyRetention.Capture(store, retained, "seed", 0, 57, out _));
        int seed = retained["seed"].Slot;
        var requests = new Dictionary<string, int>();
        foreach (string id in new[] { "a", "b", "c" })
            Assert.True(SlotCopyRetention.Clone(store, requests, id, seed, 57));
        Assert.Equal(3, requests.Values.Distinct().Count());
        Assert.DoesNotContain(seed, requests.Values);
        Assert.DoesNotContain(0, requests.Values);
        store.Slots[requests["a"]].Kv[0] = 88;
        store.Slots[requests["a"]].Recurrent[0] = 12;
        Assert.True(store.Free(requests["a"]));
        requests.Remove("a");
        foreach (int surviving in new[] { 0, seed, requests["b"], requests["c"] })
        {
            Assert.Equal(57, store.Slots[surviving].Head);
            Assert.Equal(1, store.Slots[surviving].Kv[0]);
            Assert.Equal(0.25f, store.Slots[surviving].Recurrent[0]);
        }
        Assert.Equal(0, store.Active);
        Assert.Equal(4096 + 3 * 6144, store.OwnedBytes);
    }

    [Fact]
    public void BothMetadataCollectionsHaveCapacityBeforeAnyNewPhysicalSlotIsAllocated()
    {
        var store = new Store();
        var retained = new OrderedDictionary<string, SlotRetention.RetainedSlot>();
        store.BeforeAlloc = () => Assert.True(retained.EnsureCapacity(0) >= retained.Count + 1);
        Assert.True(SlotCopyRetention.Capture(store, retained, "seed", 0, 57, out _));
        var requests = new Dictionary<string, int>();
        store.BeforeAlloc = () => Assert.True(requests.EnsureCapacity(0) >= requests.Count + 1);
        Assert.True(SlotCopyRetention.Clone(store, requests, "request", retained["seed"].Slot, 57));
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData(" ")]
    public void InvalidKeysDeclineBeforeAllocating(string? key)
    {
        var store = new Store();
        Assert.False(SlotCopyRetention.Capture(store, new(), key!, 0, 57, out long bytes));
        Assert.Equal(0, bytes);
        Assert.False(SlotCopyRetention.Clone(store, new(), key!, 0, 57));
        Assert.Equal(0, store.AllocCalls);
        Assert.Empty(store.Freed);
    }

    [Theory]
    [InlineData(-1, 57)]
    [InlineData(0, -1)]
    [InlineData(99, 57)]
    public void NegativeHeadOrInvalidUnknownSourceDeclinesBeforeAllocating(int source, int tokens)
    {
        var store = new Store();
        Assert.False(SlotCopyRetention.Capture(store, new(), "seed", source, tokens, out long bytes));
        Assert.False(SlotCopyRetention.Clone(store, new(), "request", source, tokens));
        Assert.Equal(0, bytes);
        Assert.Equal(0, store.AllocCalls);
        Assert.Equal(0, store.CopyCalls);
    }

    [Fact]
    public void ExistingKeysAreNeverReplacedOrReallocated()
    {
        var store = new Store();
        var retained = new OrderedDictionary<string, SlotRetention.RetainedSlot> { ["seed"] = new(0, 57) };
        var requests = new Dictionary<string, int> { ["request"] = 0 };
        Assert.False(SlotCopyRetention.Capture(store, retained, "seed", 0, 57, out long bytes));
        Assert.False(SlotCopyRetention.Clone(store, requests, "request", 0, 57));
        Assert.Equal(new SlotRetention.RetainedSlot(0, 57), retained["seed"]);
        Assert.Equal(0, requests["request"]);
        Assert.Equal(0, bytes);
        Assert.Equal(0, store.AllocCalls);
    }

    [Fact]
    public void NonnegativeEmptyHeadStillCopiesARealOwnedSlot()
    {
        var store = new Store(); store.Slots[0].Head = 0;
        var retained = new OrderedDictionary<string, SlotRetention.RetainedSlot>();
        Assert.True(SlotCopyRetention.Capture(store, retained, "empty-seed", 0, 0, out long bytes));
        Assert.Equal(0, retained["empty-seed"].Tokens);
        Assert.Equal(6144, bytes);
    }

    [Fact]
    public void ExactHeadMismatchFreesOnlyTheNewDestination()
    {
        var store = new Store();
        var retained = new OrderedDictionary<string, SlotRetention.RetainedSlot>();
        Assert.False(SlotCopyRetention.Capture(store, retained, "wrong-head", 0, 58, out long bytes));
        Assert.Equal(0, bytes);
        Assert.Empty(retained);
        Assert.Single(store.Freed);
        Assert.NotEqual(0, store.Freed[0]);
        Assert.Single(store.Slots);
        Assert.Equal(57, store.Slots[0].Head);
        Assert.Equal(4096, store.OwnedBytes);
    }

    [Fact]
    public void CopyRefusalAfterPartialWriteCleansBothCaptureAndCloneDestinations()
    {
        var store = new Store { RefuseCopy = true };
        var retained = new OrderedDictionary<string, SlotRetention.RetainedSlot>();
        var requests = new Dictionary<string, int>();
        Assert.False(SlotCopyRetention.Capture(store, retained, "seed", 0, 57, out long bytes));
        Assert.False(SlotCopyRetention.Clone(store, requests, "request", 0, 57));
        Assert.Equal(0, bytes);
        Assert.Empty(retained); Assert.Empty(requests);
        Assert.Equal(2, store.Freed.Count);
        Assert.Single(store.Slots);
        Assert.Equal(1, store.Slots[0].Kv[0]);
        Assert.Equal(0, store.Active);
    }

    [Fact]
    public void AllocationRefusalDoesNotFreeOrReclaimAnyExistingOwner()
    {
        var store = new Store { RefuseAlloc = true };
        var retained = new OrderedDictionary<string, SlotRetention.RetainedSlot> { ["existing"] = new(0, 57) };
        var requests = new Dictionary<string, int>();
        Assert.False(SlotCopyRetention.Capture(store, retained, "new-seed", 0, 57, out long bytes));
        Assert.False(SlotCopyRetention.Clone(store, requests, "request", 0, 57));
        Assert.Equal(0, bytes);
        Assert.Single(retained); Assert.Empty(requests);
        Assert.Empty(store.Freed);
        Assert.Equal(0, store.CopyCalls);
        Assert.Equal(4096, store.OwnedBytes);
    }

    [Fact]
    public void NonpositiveActualDestinationBytesCannotBePublished()
    {
        var store = new Store { DestinationBytes = 0 };
        var retained = new OrderedDictionary<string, SlotRetention.RetainedSlot>();
        var requests = new Dictionary<string, int>();
        Assert.False(SlotCopyRetention.Capture(store, retained, "seed", 0, 57, out long bytes));
        Assert.False(SlotCopyRetention.Clone(store, requests, "request", 0, 57));
        Assert.Equal(0, bytes);
        Assert.Empty(retained); Assert.Empty(requests);
        Assert.Equal(2, store.Freed.Count);
        Assert.Single(store.Slots);
    }

    [Fact]
    public void CopyExceptionFreesTheDestinationAndPreservesTheOriginalFailure()
    {
        var failure = new OutOfMemoryException("injected copy failure");
        var store = new Store { CopyError = failure };
        Assert.Same(failure, Assert.Throws<OutOfMemoryException>(() =>
            SlotCopyRetention.Capture(store, new(), "seed", 0, 57, out _)));
        Assert.Same(failure, Assert.Throws<OutOfMemoryException>(() =>
            SlotCopyRetention.Clone(store, new(), "request", 0, 57)));
        Assert.Equal(2, store.Freed.Count);
        Assert.Single(store.Slots);
    }

    [Fact]
    public void ByteQueryExceptionAfterCopyCannotLeakAnUnpublishedOwner()
    {
        var store = new Store { ByteError = new InvalidOperationException("injected bytes failure") };
        Assert.Throws<InvalidOperationException>(() => SlotCopyRetention.Capture(store, new(), "seed", 0, 57, out _));
        Assert.Single(store.Freed);
        Assert.Single(store.Slots);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void UnexpectedMetadataPublicationOomFreesThePhysicalOwner(bool capture)
    {
        var comparer = new PublicationComparer();
        var retained = new OrderedDictionary<string, SlotRetention.RetainedSlot>(comparer);
        var requests = new Dictionary<string, int>(comparer);
        var store = new Store { BeforeAlloc = () => comparer.Throw = true };
        if (capture)
            Assert.Throws<OutOfMemoryException>(() => SlotCopyRetention.Capture(store, retained, "seed", 0, 57, out _));
        else
            Assert.Throws<OutOfMemoryException>(() => SlotCopyRetention.Clone(store, requests, "request", 0, 57));
        comparer.Throw = false;
        Assert.Empty(retained); Assert.Empty(requests);
        Assert.Single(store.Freed);
        Assert.Single(store.Slots);
        Assert.Equal(4096, store.OwnedBytes);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void FailedCleanupExplicitlyReportsTheUnpublishedPayloadAndBudgetOwner(bool capture)
    {
        var store = new Store { RefuseCopy = true, RefuseFree = true };
        var retained = new OrderedDictionary<string, SlotRetention.RetainedSlot>();
        var requests = new Dictionary<string, int>();
        var error = Assert.Throws<InvalidOperationException>(() =>
        {
            if (capture) SlotCopyRetention.Capture(store, retained, "seed", 0, 57, out _);
            else SlotCopyRetention.Clone(store, requests, "request", 0, 57);
        });
        Assert.Contains("Copied slot " + store.LastAllocated, error.Message);
        Assert.Contains("unpublished payload and budget ownership remain with the store", error.Message);
        Assert.Empty(retained); Assert.Empty(requests);
        Assert.True(store.Slots.ContainsKey(store.LastAllocated));
        Assert.Equal(4096 + 6144, store.OwnedBytes);
        Assert.Equal(0, store.Active);
    }

    [Fact]
    public void FailedCleanupKeepsTheOriginalCopyExceptionAsContext()
    {
        var copyFailure = new OutOfMemoryException("copy failed");
        var store = new Store { CopyError = copyFailure, RefuseFree = true };
        var error = Assert.Throws<InvalidOperationException>(() => SlotCopyRetention.Clone(store, new(), "request", 0, 57));
        Assert.Same(copyFailure, error.InnerException);
        Assert.Equal(2, store.Slots.Count);
    }

    [Fact]
    public void CleanupExceptionIsAlsoAnExplicitUnpublishedOwnershipFailure()
    {
        var freeFailure = new InvalidOperationException("store refused physical teardown");
        var store = new Store { RefuseCopy = true, FreeError = freeFailure };
        var error = Assert.Throws<InvalidOperationException>(() => SlotCopyRetention.Clone(store, new(), "request", 0, 57));
        Assert.Same(freeFailure, error.InnerException);
        Assert.Contains("budget ownership remain", error.Message);
        Assert.Equal(4096 + 6144, store.OwnedBytes);
    }
}
