using System;
using TensorSharp.Server.Hosting;

namespace InferenceWeb.Tests;

public class StartupWarmupStatusTests
{
    [Fact]
    public void AResultBelongsOnlyToItsOriginalModel()
    {
        var model = new object();
        var store = new StartupWarmupStatusStore();
        var status = BothModes();
        store.Set(model, status);
        Assert.Same(status, store.Get(model));
        Assert.Null(store.Get(null));
        Assert.Null(store.Get(new object()));
    }

    [Fact]
    public void CompletionRequiresBothModesAndDoesNotClaimPrefixRetention()
    {
        var stages = new[] { new StartupWarmupStatus.Stage(false, true, 100),
            new StartupWarmupStatus.Stage(true, true, 200) };
        var status = new StartupWarmupStatus(stages, false);
        stages[0] = new(false, false, 400);
        Assert.True(status.Completed);
        Assert.False(status.PrefixRetentionRequested);
        Assert.Equal(300, status.ElapsedMilliseconds);
        Assert.False(new StartupWarmupStatus(stages, true).Completed);
        Assert.False(new StartupWarmupStatus(new[] { stages[1] }, true).Completed);
        Assert.False(new StartupWarmupStatus(new[] { stages[1], stages[1] }, true).Completed);
    }

    [Fact]
    public void JointStatusRequiresFreshNativeWidthProofAndLeavesBothModeCompletionIndependent()
    {
        var serial = new[] { new StartupWarmupStatus.Stage(false, true, 1), new StartupWarmupStatus.Stage(true, true, 1) };
        var joint = new[] { new StartupWarmupStatus.JointStage(2, true, true, true, 7, 100, 1, "accepted"),
            new StartupWarmupStatus.JointStage(3, true, true, true, 8, 150, 2, "accepted") };
        var status = new StartupWarmupStatus(serial, true, joint, true);
        Assert.True(status.Completed); Assert.True(status.JointCompleted); Assert.True(status.JointWarmupRequested);
        Assert.False(status.JointCleanupPending); Assert.Equal(3, status.JointElapsedMilliseconds);
        joint[1] = joint[1] with { CleanupCompleted = false, CleanupPending = true, Detail = "cleanup pending" };
        Assert.True(status.JointCompleted); // Defensive snapshot.
        var failed = new StartupWarmupStatus(serial, true, joint, true);
        Assert.True(failed.Completed); Assert.False(failed.JointCompleted); Assert.True(failed.JointCleanupPending);
        Assert.False(new StartupWarmupStatus(serial, true, new[] { joint[0], joint[0] }, true).JointCompleted);
        Assert.False(new StartupWarmupStatus(serial, true,
            new[] { joint[0], joint[1] with { CleanupCompleted = true, NativeAcceptedExecutionId = 0 } }, true).JointCompleted);
        Assert.False(new StartupWarmupStatus(serial, true,
            new[] { joint[0], joint[1] with { CleanupCompleted = true, NativeAcceptedExecutionId = 9 } }, true).JointCompleted);
        var skipped = new StartupWarmupStatus(serial, true, jointWarmupSkipReason: "joint decode disabled");
        Assert.True(skipped.Completed); Assert.False(skipped.JointWarmupRequested);
        Assert.Equal("joint decode disabled", skipped.JointWarmupSkipReason);
        Assert.False(BothModes().JointWarmupRequested); Assert.False(BothModes().JointCompleted);
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void ExplicitSingleSlotPolicyCompletesOnlyItsPreferredStage(bool preferred)
    {
        var stages = new[] { new StartupWarmupStatus.Stage(preferred, true, 30) };
        var status = Single(stages, preferred);
        stages[0] = new(!preferred, false, 90);
        Assert.True(status.Completed);
        Assert.Equal(preferred, status.PreferredThinking);
        Assert.Equal(StartupWarmupPolicy.SinglePreferredVariantReason, status.SingleVariantReason);
        Assert.Equal("default-chat-prefix", status.Scope);
        Assert.Equal(30, status.ElapsedMilliseconds);
        Assert.False(status.JointWarmupRequested); Assert.False(status.JointCompleted);
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("unrecognized-policy")]
    public void OneCompletedStageCannotClaimReadinessWithoutTheExactPolicy(string? reason)
        => Assert.False(new StartupWarmupStatus([new(true, true, 1)], true, null, false, null,
            false, preferredThinking: true, singleVariantReason: reason).Completed);

    [Fact]
    public void ACompletedUnpreferredVariantDoesNotSatisfySingleSlotReadiness()
        => Assert.False(Single([new(false, true, 1)], preferred: true).Completed);

    [Fact]
    public void AFailedPreferredStageDoesNotSatisfySingleSlotReadiness()
        => Assert.False(Single([new(true, false, 1)], preferred: true).Completed);

    [Fact]
    public void SingleSlotStatusRejectsAStaleExtraStage()
        => Assert.False(Single([new(true, true, 1), new(false, true, 1)], preferred: true).Completed);

    [Fact]
    public void KernelOnlyInitializationCannotReportASingleRetainedPrefixStage()
        => Assert.False(new StartupWarmupStatus([new(true, true, 1)], true, null, false, null,
            true, preferredThinking: true, singleVariantReason: StartupWarmupPolicy.SinglePreferredVariantReason).Completed);

    [Fact]
    public void SinglePrefixPolicyRequiresPrefixRetentionToBeRequested()
        => Assert.False(new StartupWarmupStatus([new(true, true, 1)], false, null, false, null,
            false, preferredThinking: true, singleVariantReason: StartupWarmupPolicy.SinglePreferredVariantReason).Completed);

    [Fact]
    public void ExistingConstructorsKeepTwoModeKernelAndJointSemantics()
    {
        var stages = new[] { new StartupWarmupStatus.Stage(true, true, 1), new StartupWarmupStatus.Stage(false, true, 2) };
        var status = new StartupWarmupStatus(stages, false, null, false, null, true);
        Assert.True(status.Completed); Assert.True(status.PreferredThinking); Assert.Null(status.SingleVariantReason);
        Assert.Equal("kernel-initialization", status.Scope);
        Assert.NotNull(typeof(StartupWarmupStatus).GetConstructor([typeof(IEnumerable<StartupWarmupStatus.Stage>), typeof(bool),
            typeof(IEnumerable<StartupWarmupStatus.JointStage>), typeof(bool), typeof(string), typeof(bool)]));
    }

    private static StartupWarmupStatus Single(IEnumerable<StartupWarmupStatus.Stage> stages, bool preferred)
        => new(stages, true, null, false, null, false, preferredThinking: preferred,
            singleVariantReason: StartupWarmupPolicy.SinglePreferredVariantReason);

    private static StartupWarmupStatus BothModes() => new(new[] {
        new StartupWarmupStatus.Stage(false, true, 1),
        new StartupWarmupStatus.Stage(true, true, 1) }, true);
}
