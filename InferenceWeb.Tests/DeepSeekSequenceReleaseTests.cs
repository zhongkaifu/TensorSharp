// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp.Models;

namespace InferenceWeb.Tests;

public sealed class DeepSeekSequenceReleaseTests
{
    [Fact]
    public void AdoptedActiveSlotIsReclaimedAtCapacityWithoutAnotherAllocation()
    {
        var native = new Slots(2, 0, (0, 12), (1, 7));
        native.Failed.Add(0);
        Assert.Equal(-1, native.Allocate()); // The original replacement-primary path cannot succeed.
        var requests = new Dictionary<string, int> { ["ended"] = 0, ["healthy"] = 1 };
        int primary = -1;
        string active = "ended";

        DeepSeek4Model.ReleaseNativeSequence(requests, "ended", ref primary, ref active, native);

        Assert.Equal(1, native.AllocationCalls);
        Assert.Equal(new[] { "reset:0" }, native.Calls);
        Assert.Equal(0, primary);
        Assert.Null(active);
        Assert.False(requests.ContainsKey("ended"));
        Assert.Equal(1, requests["healthy"]);
        Assert.Equal(0, native.Positions[0]);
        Assert.Equal(7, native.Positions[1]);
        Assert.DoesNotContain(0, native.Failed);
        Assert.Equal(2, native.Positions.Count);
        native.Positions[0] = 3; // Subsequent primary use must survive duplicate release notification.
        DeepSeek4Model.ReleaseNativeSequence(requests, "ended", ref primary, ref active, native);
        Assert.Equal(3, native.Positions[0]);
        Assert.Single(native.Calls);
    }

    [Fact]
    public void InactiveReleaseFreesCapacityAndPreservesTheLiveAdoptedRequest()
    {
        var native = new Slots(2, 1, (0, 12), (1, 7));
        var requests = new Dictionary<string, int> { ["ended"] = 0, ["live"] = 1 };
        int primary = -1;
        string active = "live";
        DeepSeek4Model.ReleaseNativeSequence(requests, "ended", ref primary, ref active, native);
        Assert.Equal(new[] { "free:0" }, native.Calls);
        Assert.Equal(-1, primary);
        Assert.Equal("live", active);
        Assert.Equal(1, native.Active);
        Assert.Equal(7, native.Positions[1]);
        Assert.Single(requests);
        Assert.True(native.Allocate() >= 0);
    }

    [Fact]
    public void ExistingPrimaryIsSelectedBeforeTheCompletedActiveSlotIsFreed()
    {
        var native = new Slots(3, 1, (0, 9), (1, 12), (2, 7));
        var requests = new Dictionary<string, int> { ["ended"] = 1, ["healthy"] = 2 };
        int primary = 0;
        string active = "ended";
        DeepSeek4Model.ReleaseNativeSequence(requests, "ended", ref primary, ref active, native);
        Assert.Equal(new[] { "select:0", "free:1" }, native.Calls);
        Assert.Null(active);
        Assert.Equal(0, native.Active);
        Assert.Equal(9, native.Positions[0]);
        Assert.Equal(7, native.Positions[2]);
        Assert.False(native.Positions.ContainsKey(1));
        Assert.Single(requests);
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void NativeRefusalRetainsOwnershipForARealRetry(bool refuseSelect)
    {
        var native = new Slots(2, 1, (0, 9), (1, 12))
        {
            RefuseSelect = refuseSelect,
            RefuseFree = !refuseSelect,
        };
        var requests = new Dictionary<string, int> { ["ended"] = 1 };
        int primary = 0;
        string active = "ended";
        Assert.Throws<InvalidOperationException>(() =>
            DeepSeek4Model.ReleaseNativeSequence(requests, "ended", ref primary, ref active, native));
        Assert.Equal(1, requests["ended"]);
        Assert.True(native.Positions.ContainsKey(1));
        Assert.Equal(refuseSelect ? "ended" : null, active);
        Assert.Equal(refuseSelect ? 1 : 0, native.Active);

        native.RefuseSelect = native.RefuseFree = false;
        DeepSeek4Model.ReleaseNativeSequence(requests, "ended", ref primary, ref active, native);
        Assert.Empty(requests);
        Assert.Null(active);
        Assert.False(native.Positions.ContainsKey(1));
        Assert.Equal(9, native.Positions[0]);
    }

    [Fact]
    public void ResetExceptionDoesNotForgetTheAdoptedSlot()
    {
        var native = new Slots(1, 0, (0, 12)) { ThrowReset = true };
        var requests = new Dictionary<string, int> { ["ended"] = 0 };
        int primary = -1;
        string active = "ended";
        Assert.Throws<InvalidOperationException>(() =>
            DeepSeek4Model.ReleaseNativeSequence(requests, "ended", ref primary, ref active, native));
        Assert.Equal(-1, primary);
        Assert.Equal("ended", active);
        Assert.Equal(0, requests["ended"]);
        native.ThrowReset = false;
        DeepSeek4Model.ReleaseNativeSequence(requests, "ended", ref primary, ref active, native);
        Assert.Equal(0, primary);
        Assert.Null(active);
        Assert.Empty(requests);
        Assert.Equal(0, native.Positions[0]);
    }

    [Fact]
    public void SilentNativeResetRefusalKeepsAdoptedRequestUntilCheckedRetrySucceeds()
    {
        var native = new Slots(1, 0, (0, 12)) { RefuseReset = true };
        var requests = new Dictionary<string, int> { ["ended"] = 0 };
        int primary = -1;
        string active = "ended";
        Assert.Throws<InvalidOperationException>(() =>
            DeepSeek4Model.ReleaseNativeSequence(requests, "ended", ref primary, ref active, native));
        Assert.Equal(0, native.Positions[0]); // Native Reset partially mutated, then returned failure.
        Assert.Contains(0, native.Failed);
        Assert.Equal(-1, primary);
        Assert.Equal("ended", active);
        Assert.Equal(0, requests["ended"]);
        native.RefuseReset = false;
        DeepSeek4Model.ReleaseNativeSequence(requests, "ended", ref primary, ref active, native);
        Assert.Equal(0, primary);
        Assert.Null(active);
        Assert.Empty(requests);
        Assert.Empty(native.Failed);
    }

    private sealed class Slots : DeepSeek4Model.INativeSlotRelease
    {
        private readonly int _capacity;
        public Dictionary<int, int> Positions { get; }
        public HashSet<int> Failed { get; } = new();
        public List<string> Calls { get; } = new();
        public int Active { get; private set; }
        public int AllocationCalls { get; private set; }
        public bool RefuseSelect { get; set; }
        public bool RefuseFree { get; set; }
        public bool ThrowReset { get; set; }
        public bool RefuseReset { get; set; }

        public Slots(int capacity, int active, params (int Slot, int Position)[] slots)
        {
            _capacity = capacity;
            Active = active;
            Positions = slots.ToDictionary(s => s.Slot, s => s.Position);
        }

        public int Allocate()
        {
            AllocationCalls++;
            if (Positions.Count >= _capacity) return -1;
            int id = Enumerable.Range(0, _capacity).First(i => !Positions.ContainsKey(i));
            Positions.Add(id, 0);
            return id;
        }

        public bool Reset()
        {
            Calls.Add($"reset:{Active}");
            if (ThrowReset) throw new InvalidOperationException("Injected reset failure");
            Positions[Active] = 0;
            if (RefuseReset) { Failed.Add(Active); return false; }
            Failed.Remove(Active);
            return true;
        }

        public bool Select(int slot)
        {
            Calls.Add($"select:{slot}");
            if (RefuseSelect || !Positions.ContainsKey(slot)) return false;
            Active = slot;
            return true;
        }

        public bool Free(int slot)
        {
            Calls.Add($"free:{slot}");
            if (RefuseFree || slot == Active) return false;
            Failed.Remove(slot);
            return Positions.Remove(slot);
        }
    }
}
