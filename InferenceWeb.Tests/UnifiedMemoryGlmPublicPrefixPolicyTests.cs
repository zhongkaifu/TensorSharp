// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Reflection;
using System.Runtime.CompilerServices;
using TensorSharp.Memory;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;

namespace InferenceWeb.Tests;

/// <summary>Exercise the real session's configuration and early refusal paths
/// without constructing a model, registering native callbacks or observing a
/// device. A valid capture/copy still requires native and end-to-end tests.</summary>
public sealed class UnifiedMemoryGlmPublicPrefixPolicyTests
{
    private const BindingFlags Private = BindingFlags.Instance | BindingFlags.NonPublic;

    private static void Set(object target, string field, object? value)
        => target.GetType().GetField(field, Private)!.SetValue(target, value);

    private static GlmUnifiedMemoryModelSession Session(int plannedCopies = 3)
    {
        var session = (GlmUnifiedMemoryModelSession)RuntimeHelpers.GetUninitializedObject(typeof(GlmUnifiedMemoryModelSession));
        Set(session, "_options", new UnifiedMemoryModelOptions { MaximumRunningRequests = 3, RetainedConversationCount = 2 });
        Set(session, "_prefixCopies", plannedCopies); Set(session, "_context", 1024); Set(session, "_prefill", 32);
        Set(session, "<Budget>k__BackingField", new MemoryBudget([
            new(UnifiedMemoryModelSession.HostPool, 1L << 30), new(UnifiedMemoryModelSession.DevicePool, 1L << 30)]));
        return session;
    }

    private static SequenceState Sequence(int shared = 4) => new("policy", [11, 22, 33, 44, 55, 66],
        2, 4, SamplingConfig.Greedy, sharedPrefixTokens: shared, cacheScope: "private-chat",
        publicCheckpointBoundaries: [2, 4]);

    private static object? Capture(GlmUnifiedMemoryModelSession session, int tokens)
        => typeof(GlmUnifiedMemoryModelSession).GetMethod("EnterPrefixCapture", Private)!.Invoke(session, [tokens]);

    [Theory]
    [InlineData(false, 0, 0)]
    [InlineData(false, 8, 0)]
    [InlineData(true, 0, 0)]
    [InlineData(true, 1, 1)]
    [InlineData(true, 2, 2)]
    [InlineData(true, 3, 3)]
    [InlineData(true, 8, 3)]
    public void SchedulerUsesTheEnabledBudgetClampedToActualPlannedCapacity(bool enabled, int requested, int expected)
    {
        var session = Session();
        var original = new SchedulerConfig
        {
            BlockSize = 16, EnablePrefixCaching = enabled,
            ExecutionOptionsOverride = new ExecutionOptions { PrefixCheckpointBudget = requested },
        };
        var configured = session.ConfigureScheduler(original);
        Assert.Equal(enabled, configured.EnablePrefixCaching);
        Assert.Equal(expected, configured.ExecutionOptionsOverride!.PrefixCheckpointBudget);
        Assert.Equal(expected, (int)typeof(GlmUnifiedMemoryModelSession).GetField("_effectivePrefixCopies", Private)!.GetValue(session)!);
        Assert.Equal(64 * (3 + 2 + expected + 1), configured.NumBlocks);
        Assert.Equal(requested, original.ExecutionOptionsOverride.PrefixCheckpointBudget);
        Assert.Empty(session.Budget.Snapshot().Where(pool => pool.Reserved != 0 || pool.Committed != 0));
    }

    [Theory]
    [InlineData(-1)]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(3)]
    [InlineData(5)]
    [InlineData(6)]
    public void UndeclaredOrPrivateBoundariesDeclineBeforeAnyBudgetOrNativeWork(int tokens)
    {
        var session = Session(); Set(session, "_effectivePrefixCopies", 3);
        Set(session, "_executingSequence", Sequence());
        var before = session.Budget.Snapshot().ToArray();
        // _model/_geometry/_native deliberately remain absent. Getting past
        // the boundary guard would throw rather than silently count as a pass.
        Assert.Null(Capture(session, tokens));
        Assert.Equal(before, session.Budget.Snapshot().ToArray());
    }

    [Fact]
    public void APrivatePromptWithoutAPublicPrefixNeverCapturesItsPromptEnd()
    {
        var session = Session(); Set(session, "_effectivePrefixCopies", 3);
        Set(session, "_executingSequence", Sequence(shared: 0));
        Assert.Null(Capture(session, 6));
        Assert.Empty(session.Budget.Snapshot().Where(pool => pool.Reserved != 0 || pool.Committed != 0));
    }

    [Fact]
    public void DisabledBudgetAndMissingExecutionDeclineEvenAtADeclaredBoundary()
    {
        var session = Session(); Set(session, "_executingSequence", Sequence());
        Assert.Null(Capture(session, 4));
        Set(session, "_effectivePrefixCopies", 3); Set(session, "_executingSequence", null);
        Assert.Null(Capture(session, 4));
        Assert.Empty(session.Budget.Snapshot().Where(pool => pool.Reserved != 0 || pool.Committed != 0));
    }

    [Fact]
    public void AFullRetainedPlusTransientCapacityDeclinesBeforeReservationOrNativeEntry()
    {
        var session = Session(); Set(session, "_effectivePrefixCopies", 1);
        Set(session, "_executingSequence", Sequence());
        var model = (GlmDsaModel)RuntimeHelpers.GetUninitializedObject(typeof(GlmDsaModel));
        Set(model, "_nativeSync", new object());
        Set(model, "_retainedSlots", new OrderedDictionary<string, SlotRetention.RetainedSlot>
        {
            ["old-public"] = new(1, 2, IsCopy: true),
            ["replacement-public"] = new(2, 4, IsCopy: true),
            ["conversation"] = new(3, 6),
        });
        Set(session, "_model", model);
        Assert.Equal(2, model.RetainedCopiedSlotCount);
        Assert.Null(Capture(session, 4));
        Assert.Equal(2, model.RetainedCopiedSlotCount);
        Assert.Empty(session.Budget.Snapshot().Where(pool => pool.Reserved != 0 || pool.Committed != 0));
    }
}
