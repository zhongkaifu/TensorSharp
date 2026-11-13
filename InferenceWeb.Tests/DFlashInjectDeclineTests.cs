using System.Collections;
using System.Reflection;
using System.Runtime.CompilerServices;
using TensorSharp.Models;

namespace InferenceWeb.Tests;

/// <summary>
/// A DFlash/DSpine context inject that fails on a discrete GPU (device memory, as on a Qwen3.8-27B DSpine server
/// after a 16K prompt) ends drafting for that request only: it decodes plainly, one warning per model names the
/// consequence, and the next request on that ring drafts again. Before, the failed inject fell back to the per-op
/// ops, which replaced the ring's device copy under the persisted drafter graphs and crashed the server with an
/// illegal memory access. The native half (the decline itself, under a forced allocation failure) is
/// GgmlOpsDFlashInjectTest.
/// </summary>
[Collection(ConsoleErrorCollection.Name)]
public class DFlashInjectDeclineTests
{
    private const string Warning = "[dflash] the drafter's context inject failed";
    private const BindingFlags Instance = BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public;

    private sealed class DeclineTestModel : Qwen35RuntimeModel
    {
        // Never called: the tests build only the drafter's request state, without a GGUF or a backend.
        public DeclineTestModel() : base("", BackendType.Cpu) { }
    }

    private static DeclineTestModel ModelOn(BackendType backend)
    {
        var model = (DeclineTestModel)RuntimeHelpers.GetUninitializedObject(typeof(DeclineTestModel));
        typeof(ModelBase).GetField("_backend", Instance)!.SetValue(model, backend);
        return model;
    }

    private static Type RingType => typeof(ModelBase).GetNestedType("DFlashRing", BindingFlags.NonPublic)!;

    private static object Ring(string? owner, bool declined, long lastUsed = 0)
    {
        object ring = Activator.CreateInstance(RingType, nonPublic: true)!;
        RingType.GetField("Owner")!.SetValue(ring, owner);
        RingType.GetField("DraftingDeclined")!.SetValue(ring, declined);
        RingType.GetField("LastUsed")!.SetValue(ring, lastUsed);
        return ring;
    }

    private static bool Declined(object ring) => (bool)RingType.GetField("DraftingDeclined")!.GetValue(ring)!;

    private static (bool[] results, int warnings) Decline(ModelBase model, int calls)
    {
        TextWriter original = Console.Error;
        var captured = new StringWriter();
        Console.SetError(captured);
        var results = new bool[calls];
        try
        {
            for (int i = 0; i < calls; i++)
                results[i] = model.DeclineDraftingAfterFailedInject("DFlash inject: failed to allocate persist buffer.");
        }
        finally
        {
            Console.SetError(original);
        }
        int warnings = captured.ToString().Split('\n').Count(line => line.Contains(Warning, StringComparison.Ordinal));
        return (results, warnings);
    }

    [Theory]
    [InlineData(BackendType.GgmlCuda)]
    [InlineData(BackendType.GgmlVulkan)]
    public void OnADiscreteGpu_AFailedInjectEndsDraftingForTheRequest_WarnedOncePerModel(BackendType backend)
    {
        var model = ModelOn(backend);

        var (results, warnings) = Decline(model, calls: 3);

        Assert.All(results, Assert.True);
        Assert.True(model.DFlashDraftingDeclined);
        Assert.Equal(1, warnings);

        // The request ends: the next one drafts again, and a later failure is not warned again.
        model.ReleaseDFlashRing("r1");
        Assert.False(model.DFlashDraftingDeclined);
        Assert.Equal(0, Decline(model, calls: 1).warnings);
        Assert.True(model.DFlashDraftingDeclined);
    }

    [Theory]
    [InlineData(BackendType.GgmlCpu)]
    [InlineData(BackendType.GgmlMetal)]
    [InlineData(BackendType.Cpu)]
    public void ElsewhereTheRequestKeepsDrafting_ThroughThePerOpInject(BackendType backend)
    {
        var model = ModelOn(backend);

        var (results, warnings) = Decline(model, calls: 2);

        Assert.All(results, Assert.False);
        Assert.False(model.DFlashDraftingDeclined);
        Assert.Equal(0, warnings);
    }

    [Fact]
    public void TheDeclineIsTheBoundRingsRequest_AndLeavesWithIt()
    {
        var model = ModelOn(BackendType.GgmlCuda);
        object mine = Ring("a", declined: false), other = Ring("b", declined: false);
        var pool = (IList)Activator.CreateInstance(typeof(List<>).MakeGenericType(RingType))!;
        pool.Add(mine);
        pool.Add(other);
        typeof(ModelBase).GetField("_dflashRingPool", Instance)!.SetValue(model, pool);
        typeof(ModelBase).GetField("_dflashActiveRing", Instance)!.SetValue(model, mine);

        Decline(model, calls: 1);

        Assert.True(Declined(mine));
        Assert.False(Declined(other));
        Assert.True(model.DFlashDraftingDeclined);

        // "a" ends: its ring returns to the pool without the decline. (Releasing the active ring rebinds the model's
        // own ring, which this fixture has none of, so "a" is released while "b"'s ring is the bound one.)
        typeof(ModelBase).GetField("_dflashActiveRing", Instance)!.SetValue(model, other);
        model.ReleaseDFlashRing("a");
        Assert.False(Declined(mine));
        Assert.Null(RingType.GetField("Owner")!.GetValue(mine));
    }

    [Fact]
    public void ARingTakenFromItsLeastRecentlyUsedOwner_DropsThatOwnersDecline()
    {
        var model = ModelOn(BackendType.GgmlCuda);
        object modelRing = Ring("busy", declined: false);
        var pool = (IList)Activator.CreateInstance(typeof(List<>).MakeGenericType(RingType))!;
        object oldest = Ring("stale", declined: true, lastUsed: 1);
        pool.Add(oldest);
        for (int i = 1; i < 8; i++) pool.Add(Ring("live" + i, declined: false, lastUsed: 10 + i));   // the pool is full
        typeof(ModelBase).GetField("_dflashModelRing", Instance)!.SetValue(model, modelRing);
        typeof(ModelBase).GetField("_dflashActiveRing", Instance)!.SetValue(model, modelRing);
        typeof(ModelBase).GetField("_dflashRingPool", Instance)!.SetValue(model, pool);

        object chosen = typeof(ModelBase).GetMethod("FindOrAssignDFlashRing", Instance)!.Invoke(model, new object[] { "new" })!;

        Assert.Same(oldest, chosen);
        Assert.Equal("new", RingType.GetField("Owner")!.GetValue(chosen));
        Assert.False(Declined(chosen));
    }
}

/// <summary>Tests that swap <see cref="Console.Error"/> to read what a model warns; they must not interleave.</summary>
[CollectionDefinition(Name, DisableParallelization = true)]
public sealed class ConsoleErrorCollection
{
    public const string Name = "Console.Error capture";
}
