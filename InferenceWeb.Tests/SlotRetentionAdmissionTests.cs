// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// The shared slot retention (TensorSharp.Models.SlotRetention, used by the direct-CUDA Qwen 3.8 Flash Next engine,
// GLM and DeepSeek) admitting a new request when device memory holds no more slots, on a fake slot store: a finished
// conversation's retained slot is given up for it, a slot a running request owns never is, a retained slot the native
// side refuses to free is passed over for the next, and reuse is untouched while memory suffices.
using TensorSharp.Models;

namespace InferenceWeb.Tests;

public sealed class SlotRetentionAdmissionTests
{
    [Fact]
    public void ARetainedConversationsSlot_IsGivenUpForANewRequest_WhenMemoryHoldsNoMore()
    {
        // Four slots fit: three running requests and one finished conversation kept for its next turn.
        var store = new FakeSlots(capacity: 4, active: 6, 4, 5, 6, 7);
        var retained = new OrderedDictionary<string, SlotRetention.RetainedSlot> { ["pc:done"] = new(7, 300) };
        var reclaimed = new List<string>();

        int slot = SlotRetention.AllocReclaiming(retained, store, reclaimed);

        Assert.True(slot >= 0, "the new request was refused although a retained slot could be given up");
        Assert.Equal(new[] { "pc:done" }, reclaimed);
        Assert.Empty(retained);
        Assert.DoesNotContain(7, store.Live);
        Assert.Contains(slot, store.Live);
    }

    [Fact]
    public void ASlotARunningRequestOwns_IsNeverGivenUp()
    {
        // Memory full with running requests only: nothing retained, so nothing is freed and the request waits.
        var store = new FakeSlots(capacity: 3, active: 6, 4, 5, 6);
        var retained = new OrderedDictionary<string, SlotRetention.RetainedSlot>();
        var reclaimed = new List<string>();

        Assert.Equal(-1, SlotRetention.AllocReclaiming(retained, store, reclaimed));
        Assert.Empty(reclaimed);
        Assert.Equal(new[] { 4, 5, 6 }, store.Live.OrderBy(s => s));
        Assert.Empty(store.Freed);

        // With a retained conversation as well, only its slot goes; the running requests' slots stay.
        var withRetained = new FakeSlots(capacity: 4, active: 6, 4, 5, 6, 7);
        var kept = new OrderedDictionary<string, SlotRetention.RetainedSlot> { ["pc:done"] = new(7, 300) };
        Assert.True(SlotRetention.AllocReclaiming(kept, withRetained, reclaimed) >= 0);
        Assert.Equal(new[] { 7 }, withRetained.Freed);
        Assert.Contains(4, withRetained.Live);
        Assert.Contains(5, withRetained.Live);
        Assert.Contains(6, withRetained.Live);
    }

    [Fact]
    public void ARetainedSlotTheStoreRefusesToFree_IsPassedOver_AndTheNextOldestGoes()
    {
        // The oldest retained slot is the native side's selection, which it refuses to free; stopping there left the
        // younger conversation's memory held while the new request was refused.
        var store = new FakeSlots(capacity: 4, active: 7, 4, 5, 7, 8);
        var retained = new OrderedDictionary<string, SlotRetention.RetainedSlot>
        {
            ["pc:older"] = new(7, 300),
            ["pc:younger"] = new(8, 200),
        };
        var reclaimed = new List<string>();

        int slot = SlotRetention.AllocReclaiming(retained, store, reclaimed);

        Assert.True(slot >= 0);
        Assert.Equal(new[] { "pc:younger" }, reclaimed);
        Assert.True(retained.ContainsKey("pc:older"), "the refused slot must stay retained for a later retry");
        Assert.Contains(7, store.Live);
        Assert.DoesNotContain(8, store.Live);
    }

    [Fact]
    public void ReuseIsUntouched_WhileMemorySuffices()
    {
        // Room for a new slot: the retained conversation is kept, and its next turn can still take its slot.
        var store = new FakeSlots(capacity: 5, active: 6, 4, 5, 6, 7);
        var retained = new OrderedDictionary<string, SlotRetention.RetainedSlot> { ["pc:done"] = new(7, 300) };
        var reclaimed = new List<string>();

        int slot = SlotRetention.AllocReclaiming(retained, store, reclaimed);

        Assert.True(slot >= 0);
        Assert.Empty(reclaimed);
        Assert.True(retained.ContainsKey("pc:done"));
        Assert.Contains(7, store.Live);

        var requests = new Dictionary<string, int>();
        Assert.True(SlotRetention.DonateSlot(requests, retained, "pc:done", "next-turn", out var donated));
        Assert.Equal(7, donated.Slot);
        Assert.Equal(300, donated.Tokens);
        Assert.Equal(7, requests["next-turn"]);
        Assert.Empty(retained);
    }

    /// <summary>A slot store whose device memory holds <c>capacity</c> slots and that refuses to free its selection.</summary>
    private sealed class FakeSlots : SlotRetention.ISlotStore
    {
        private readonly int _capacity;
        private int _next = 100;
        public readonly HashSet<int> Live;
        public readonly List<int> Freed = new();
        public int Active;

        public FakeSlots(int capacity, int active, params int[] live)
        {
            _capacity = capacity;
            Active = active;
            Live = new HashSet<int>(live);
        }

        public int Alloc()
        {
            if (Live.Count >= _capacity)
                return -1;
            Live.Add(_next);
            return _next++;
        }

        public bool Select(int slot)
        {
            if (!Live.Contains(slot))
                return false;
            Active = slot;
            return true;
        }

        public bool Free(int slot)
        {
            if (slot == Active || !Live.Remove(slot))
                return false;
            Freed.Add(slot);
            return true;
        }

        public int ActiveHead() => 0;
        public bool RewindActive(int tokens) => true;
        public bool ResetActive() => true;
        public bool SetReclaimable(int[] slots) => true;
        public int TakeReclaimed(int[] buffer) => 0;
    }
}
