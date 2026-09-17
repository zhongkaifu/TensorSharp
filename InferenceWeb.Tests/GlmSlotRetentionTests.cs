// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// GLM's managed slot retention for the radix prefix cache (DESIGN §6.5.2, DEC-39),
// on a fake slot store: the ownership moves must never leave a retained slot
// active, never lose a slot on a refusal, and never change the native selection
// they found unless the move itself requires it. Unused by the engine until M5f.
using TensorSharp.Models;

namespace InferenceWeb.Tests;

public sealed class GlmSlotRetentionTests
{
    [Fact]
    public void RetainingTheActiveSlot_SelectsThePrimaryAndKeysTheSlotByThePayload()
    {
        var store = new Slots(active: 1, (0, 0), (1, 40));
        var requests = new Dictionary<string, int> { ["req"] = 1 };
        var retained = new Dictionary<string, GlmDsaModel.RetainedGlmSlot>();
        int primary = 0; string active = "req";
        Assert.True(GlmDsaModel.RetainSlot(requests, retained, "req", "pc:1:1", 40, canRewind: true, ref primary, ref active, store));
        Assert.Equal(new GlmDsaModel.RetainedGlmSlot(1, 40), retained["pc:1:1"]);
        Assert.Empty(requests);
        Assert.Null(active);
        Assert.Equal(0, store.Active);
        Assert.Equal(0, primary);
    }

    [Fact]
    public void RetainingARequestThatAdoptedThePrimary_AllocatesAFreshPrimary_OrRefusesUntouched()
    {
        var store = new Slots(active: 0, (0, 40)) { AllocFails = true };
        var requests = new Dictionary<string, int> { ["req"] = 0 };
        var retained = new Dictionary<string, GlmDsaModel.RetainedGlmSlot>();
        int primary = -1; string active = "req";
        Assert.False(GlmDsaModel.RetainSlot(requests, retained, "req", "pc:1:1", 40, true, ref primary, ref active, store));
        Assert.Equal(0, requests["req"]);
        Assert.Equal("req", active);
        Assert.Equal(-1, primary);
        Assert.Equal(0, store.Active);
        Assert.Empty(retained);

        store.AllocFails = false;
        Assert.True(GlmDsaModel.RetainSlot(requests, retained, "req", "pc:1:1", 40, true, ref primary, ref active, store));
        Assert.True(primary > 0);
        Assert.Equal(primary, store.Active);
        Assert.Equal(0, store.Heads[primary]);
        Assert.Equal(0, retained["pc:1:1"].Slot);
    }

    [Fact]
    public void ARefusedRewind_FreesTheFreshPrimaryItAllocated_AndLeavesTheSlotAsItWas()
    {
        var store = new Slots(active: 0, (0, 40)) { RefuseRewind = true };
        var requests = new Dictionary<string, int> { ["req"] = 0 };
        var retained = new Dictionary<string, GlmDsaModel.RetainedGlmSlot>();
        int primary = -1; string active = "req";
        Assert.False(GlmDsaModel.RetainSlot(requests, retained, "req", "pc:1:1", 36, canRewind: true, ref primary, ref active, store));
        Assert.Equal(new[] { 0 }, store.Heads.Keys);   // the fresh primary was freed again
        Assert.Equal(40, store.Heads[0]);
        Assert.Equal(-1, primary);
        Assert.Equal("req", active);
        Assert.Equal(0, requests["req"]);
    }

    [Fact]
    public void RetainingAnInactiveSlot_ReadsItThroughASelectAndRestoresTheSelection()
    {
        var store = new Slots(active: 0, (0, 5), (1, 40), (2, 9));
        var requests = new Dictionary<string, int> { ["done"] = 1, ["running"] = 2 };
        var retained = new Dictionary<string, GlmDsaModel.RetainedGlmSlot>();
        int primary = 0; string active = null;

        // Longer than asked: rewound where the architecture can (glm-dsa)...
        Assert.True(GlmDsaModel.RetainSlot(requests, retained, "done", "pc:1:1", 36, canRewind: true, ref primary, ref active, store));
        Assert.Equal(36, store.Heads[1]);
        Assert.Equal(0, store.Active);
        Assert.Equal(new GlmDsaModel.RetainedGlmSlot(1, 36), retained["pc:1:1"]);

        // ...refused where it cannot (glm5next), and shorter than asked is always refused.
        Assert.False(GlmDsaModel.RetainSlot(requests, retained, "running", "pc:1:2", 8, canRewind: false, ref primary, ref active, store));
        Assert.False(GlmDsaModel.RetainSlot(requests, retained, "running", "pc:1:2", 10, canRewind: true, ref primary, ref active, store));
        Assert.Equal(9, store.Heads[2]);
        Assert.Equal(2, requests["running"]);
        Assert.Equal(0, store.Active);

        // A key already in use consumes nothing.
        Assert.False(GlmDsaModel.RetainSlot(requests, retained, "running", "pc:1:1", 9, true, ref primary, ref active, store));
        Assert.Equal(2, requests["running"]);
    }

    [Fact]
    public void DonateThenReturn_MovesTheSlotBothWays_AndABoundDonationCannotReturn()
    {
        var store = new Slots(active: 0, (0, 0), (1, 40));
        var requests = new Dictionary<string, int>();
        var retained = new Dictionary<string, GlmDsaModel.RetainedGlmSlot> { ["pc:1:1"] = new(1, 40) };
        int primary = 0; string active = null;

        Assert.True(GlmDsaModel.CanDonate(retained, "pc:1:1", 40, 40, canRewind: false));
        Assert.True(GlmDsaModel.CanDonate(retained, "pc:1:1", 40, 36, canRewind: true));
        Assert.False(GlmDsaModel.CanDonate(retained, "pc:1:1", 40, 36, canRewind: false));
        Assert.False(GlmDsaModel.CanDonate(retained, "pc:1:1", 41, 41, canRewind: true));

        Assert.True(GlmDsaModel.DonateSlot(requests, retained, "pc:1:1", "next", out var donated));
        Assert.Equal(1, donated.Slot);
        Assert.Equal(1, requests["next"]);
        Assert.Empty(retained);
        Assert.False(GlmDsaModel.DonateSlot(requests, retained, "pc:1:1", "again", out _));

        Assert.True(GlmDsaModel.ReturnSlot(requests, retained, "next", "pc:1:1", ref primary, ref active, store));
        Assert.Equal(new GlmDsaModel.RetainedGlmSlot(1, 40), retained["pc:1:1"]);
        Assert.Equal(0, store.Active);

        Assert.True(GlmDsaModel.DonateSlot(requests, retained, "pc:1:1", "bound", out _));
        active = "bound";
        store.Active = 1;
        Assert.False(GlmDsaModel.ReturnSlot(requests, retained, "bound", "pc:1:1", ref primary, ref active, store));
        Assert.Equal(1, requests["bound"]);
    }

    [Fact]
    public void ConvertPrimary_RetainsTheLivePrimaryAndSelectsAFreshOne_OrRefusesUntouched()
    {
        var store = new Slots(active: 0, (0, 40), (1, 7));
        var retained = new Dictionary<string, GlmDsaModel.RetainedGlmSlot>();
        int primary = 0; string active = null;
        Assert.False(GlmDsaModel.ConvertPrimarySlot(retained, "pc:1:1", 39, ref primary, ref active, store));
        store.AllocFails = true;
        Assert.False(GlmDsaModel.ConvertPrimarySlot(retained, "pc:1:1", 40, ref primary, ref active, store));
        string busy = "req";
        store.AllocFails = false;
        Assert.False(GlmDsaModel.ConvertPrimarySlot(retained, "pc:1:1", 40, ref primary, ref busy, store));
        Assert.Equal(0, primary);
        Assert.Empty(retained);

        Assert.True(GlmDsaModel.ConvertPrimarySlot(retained, "pc:1:1", 40, ref primary, ref active, store));
        Assert.Equal(new GlmDsaModel.RetainedGlmSlot(0, 40), retained["pc:1:1"]);
        Assert.NotEqual(0, primary);
        Assert.Equal(primary, store.Active);
        Assert.Equal(40, store.Heads[0]);
    }

    [Fact]
    public void Release_FreesKnownSlots_IgnoresUnknownKeys_AndKeepsARefusedSlotForARetry()
    {
        var store = new Slots(active: 0, (0, 0), (1, 40), (2, 30));
        var retained = new Dictionary<string, GlmDsaModel.RetainedGlmSlot> { ["a"] = new(1, 40), ["b"] = new(2, 30) };
        store.RefuseFree.Add(2);
        GlmDsaModel.ReleaseSlots(retained, new[] { "a", "missing", "b" }, store);
        Assert.False(store.Heads.ContainsKey(1));
        Assert.True(retained.ContainsKey("b"));
        store.RefuseFree.Clear();
        GlmDsaModel.ReleaseSlots(retained, new[] { "b" }, store);
        Assert.Empty(retained);
    }

    private sealed class Slots : GlmDsaModel.IGlmSlotStore
    {
        public readonly Dictionary<int, int> Heads;
        public readonly HashSet<int> RefuseFree = new();
        public int Active;
        public bool AllocFails;
        public bool RefuseRewind;
        private int _next = 100;

        public Slots(int active, params (int Slot, int Head)[] slots)
        {
            Active = active;
            Heads = slots.ToDictionary(s => s.Slot, s => s.Head);
        }

        public int Alloc()
        {
            if (AllocFails) return -1;
            Heads[_next] = 0;
            return _next++;
        }

        public bool Select(int slot)
        {
            if (!Heads.ContainsKey(slot)) return false;
            Active = slot;
            return true;
        }

        // Like the native side: the active slot cannot be freed (a fresh primary is never selected before it is freed).
        public bool Free(int slot) => slot != Active && !RefuseFree.Contains(slot) && Heads.Remove(slot);
        public int ActiveHead() => Heads[Active];

        public bool RewindActive(int tokens)
        {
            if (RefuseRewind || tokens > Heads[Active]) return false;
            Heads[Active] = tokens;
            return true;
        }
    }
}
