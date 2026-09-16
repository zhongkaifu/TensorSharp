// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp.Models;

namespace InferenceWeb.Tests;

public sealed class DeepSeekSlotRetentionTests
{
    [Fact]
    public void AdoptedPrimaryRetainsAndRebindsWithoutResetCopyOrAllocation()
    {
        var native = new Slots(0, (0, 4411));
        var requests = new Dictionary<string, int> { ["old"] = 0 };
        var retained = new Dictionary<string, int>();
        string active = "old", selected = null;
        int primary = -1;
        Assert.True(Retain(requests, retained, "old", ref active, ref selected, native));
        Assert.Null(active);
        Assert.Equal("old", selected);
        DeepSeek4Model.ReleaseNativeSequence(requests, "old", ref primary, ref active, native);
        Assert.Equal(4411, native.Heads[0]);
        Assert.True(Rebind(requests, retained, "old", "new", ref active, ref selected, native));
        Assert.Equal(0, requests["new"]);
        Assert.Equal("new", active);
        Assert.Null(selected);
        Assert.Empty(retained);
        Assert.False(Rebind(requests, retained, "old", "other", ref active, ref selected, native));
        Assert.Empty(native.Mutations);
    }

    [Theory]
    [InlineData(false, true, 9)]
    [InlineData(true, false, 9)]
    [InlineData(true, true, 0)]
    public void FailedEmptyOrOverBudgetHolderKeepsItsOriginalOwner(bool healthy, bool fits, int head)
    {
        var native = new Slots(0, (0, head)) { Fits = fits };
        if (!healthy) native.Failed.Add(0);
        var requests = new Dictionary<string, int> { ["old"] = 0 };
        var retained = new Dictionary<string, int>();
        string active = "old", selected = null;
        Assert.False(Retain(requests, retained, "old", ref active, ref selected, native));
        Assert.Equal(0, requests["old"]);
        Assert.Equal("old", active);
        Assert.Null(selected);
        Assert.Empty(retained);
        Assert.Empty(native.Mutations);
    }

    [Fact]
    public void KeyCollisionDoesNotConsumeEitherHolder()
    {
        var native = new Slots(1, (0, 11), (1, 23));
        var requests = new Dictionary<string, int> { ["new"] = 1 };
        var retained = new Dictionary<string, int> { ["old"] = 0 };
        string active = "new", selected = null;
        Assert.False(Rebind(requests, retained, "old", "new", ref active, ref selected, native));
        Assert.Equal(0, retained["old"]);
        Assert.Equal(1, requests["new"]);
        native.Failed.Add(0);
        Assert.False(Rebind(requests, retained, "old", "other", ref active, ref selected, native));
        Assert.Single(retained);
        Assert.Single(requests);
        Assert.Empty(native.Mutations);
    }

    [Fact]
    public void SelectedRetainedSlotWithNoPrimaryIsReclaimedAndItsGraphArenasReleased()
    {
        var native = new Slots(0, (0, 4411), (1, 23));
        var retained = new Dictionary<string, int> { ["old"] = 0 };
        string active = null, selected = "old";
        int primary = -1;
        Discard(retained, "old", ref primary, ref active, ref selected, native);
        Assert.Equal(new[] { "graphs:0", "reset:0" }, native.Mutations);
        Assert.Equal(0, primary);
        Assert.Null(selected);
        Assert.Null(active);
        Assert.Empty(retained);
        Assert.Equal(0, native.Heads[0]);
        Assert.Equal(23, native.Heads[1]);
    }

    [Fact]
    public void ExistingPrimaryIsSelectedBeforeFreeAndUnrelatedStateSurvives()
    {
        var native = new Slots(1, (0, 17), (1, 4411), (2, 23));
        var retained = new Dictionary<string, int> { ["old"] = 1 };
        string active = null, selected = "old";
        int primary = 0;
        Discard(retained, "old", ref primary, ref active, ref selected, native);
        Assert.Equal(new[] { "select:0", "free:1" }, native.Mutations);
        Assert.Equal(17, native.Heads[0]);
        Assert.Equal(23, native.Heads[2]);
        Assert.False(native.Heads.ContainsKey(1));
    }

    [Fact]
    public void InactiveEvictionDoesNotSelectOrResetAnotherRequest()
    {
        var native = new Slots(1, (0, 4411), (1, 23));
        var retained = new Dictionary<string, int> { ["old"] = 0 };
        string active = "live", selected = null;
        int primary = -1;
        Discard(retained, "old", ref primary, ref active, ref selected, native);
        Assert.Equal(new[] { "free:0" }, native.Mutations);
        Assert.Equal("live", active);
        Assert.Equal(1, native.Active);
        Assert.Equal(23, native.Heads[1]);
    }

    [Theory]
    [InlineData("graphs")]
    [InlineData("reset")]
    public void FailedSelectedReclamationIsNotPublishedAsHealthyPrimary(string failure)
    {
        var native = new Slots(0, (0, 4411)) { Failure = failure };
        var retained = new Dictionary<string, int> { ["old"] = 0 };
        string active = null, selected = "old";
        int primary = -1;
        Assert.Throws<InvalidOperationException>(() =>
            Discard(retained, "old", ref primary, ref active, ref selected, native));
        Assert.Equal(-1, primary);
        Assert.Equal("old", selected);
        Assert.Equal(0, retained["old"]);
        if (failure == "reset")
        {
            var requests = new Dictionary<string, int>();
            Assert.False(Rebind(requests, retained, "old", "new", ref active, ref selected, native));
        }
        native.Failure = null;
        Discard(retained, "old", ref primary, ref active, ref selected, native);
        Assert.Equal(0, primary);
        Assert.Empty(retained);
        Assert.Empty(native.Failed);
    }

    [Theory]
    [InlineData("select")]
    [InlineData("free")]
    public void NativeEvictionRefusalKeepsOwnershipForRetry(string failure)
    {
        var native = new Slots(1, (0, 17), (1, 4411)) { Failure = failure };
        var retained = new Dictionary<string, int> { ["old"] = 1 };
        string active = null, selected = "old";
        int primary = 0;
        Assert.Throws<InvalidOperationException>(() =>
            Discard(retained, "old", ref primary, ref active, ref selected, native));
        Assert.Equal(1, retained["old"]);
        Assert.Equal(failure == "select" ? "old" : null, selected);
        native.Failure = null;
        Discard(retained, "old", ref primary, ref active, ref selected, native);
        Assert.Empty(retained);
        Assert.Equal(17, native.Heads[0]);
    }

    private static bool Retain(Dictionary<string, int> requests, Dictionary<string, int> retained,
        string key, ref string active, ref string selected, Slots native)
        => DeepSeek4Model.RetainNativeSequence(requests, retained, key, ref active, ref selected, 1UL << 30, native);
    private static bool Rebind(Dictionary<string, int> requests, Dictionary<string, int> retained,
        string oldKey, string newKey, ref string active, ref string selected, Slots native)
        => DeepSeek4Model.RebindNativeSequence(requests, retained, oldKey, newKey, ref active, ref selected, native);
    private static void Discard(Dictionary<string, int> retained, string key, ref int primary,
        ref string active, ref string selected, Slots native)
        => DeepSeek4Model.DiscardRetainedNativeSequence(retained, key, ref primary, ref active, ref selected, native);

    private sealed class Slots : DeepSeek4Model.INativeSlotRetention
    {
        public readonly Dictionary<int, int> Heads;
        public readonly HashSet<int> Failed = new();
        public readonly List<string> Mutations = new();
        public int Active;
        public bool Fits = true;
        public string Failure;
        public Slots(int active, params (int Slot, int Head)[] slots)
        { Active = active; Heads = slots.ToDictionary(s => s.Slot, s => s.Head); }
        public bool Status(int slot, out int head, out bool healthy)
        { healthy = !Failed.Contains(slot); return Heads.TryGetValue(slot, out head); }
        public bool CanRetain(int slot, int count, ulong budget) => Fits && budget > 0;
        public bool Select(int slot)
        { Mutations.Add($"select:{slot}"); if (Failure == "select") return false; Active = slot; return Heads.ContainsKey(slot); }
        public bool Free(int slot)
        { Mutations.Add($"free:{slot}"); return Failure != "free" && slot != Active && Heads.Remove(slot); }
        public bool ReleaseGraphs(int slot)
        { Mutations.Add($"graphs:{slot}"); return Failure != "graphs"; }
        public bool Reset()
        {
            Mutations.Add($"reset:{Active}");
            Heads[Active] = 0;
            if (Failure == "reset") { Failed.Add(Active); return false; }
            Failed.Remove(Active);
            return true;
        }
    }
}
