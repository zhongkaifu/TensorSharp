using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Scheduling.PrefixCache;
using TensorSharp.Server.Hosting;

namespace InferenceWeb.Tests;

public sealed class StartupWarmupPolicyTests
{
    [Theory]
    [InlineData(null, true)]
    [InlineData("0", false)]
    [InlineData("1", true)]
    [InlineData("true", true)]
    [InlineData("TRUE", true)]
    [InlineData("TrUe", true)]
    [InlineData("false", false)]
    [InlineData("FALSE", false)]
    [InlineData("FaLsE", false)]
    public void PreferredThinkingUsesTheWebUiDefaultAndExplicitStrictBooleanValues(string? setting, bool expected)
        => Assert.Equal(expected, StartupWarmupPolicy.ResolvePreferredThinking(setting));

    [Theory]
    [InlineData("")]
    [InlineData(" ")]
    [InlineData("\t")]
    [InlineData(" true")]
    [InlineData("false ")]
    [InlineData("2")]
    [InlineData("yes")]
    [InlineData("on")]
    [InlineData("01")]
    [InlineData("-1")]
    public void InvalidPreferredThinkingFailsInsteadOfSilentlyChangingTheWarmupMode(string setting)
        => Assert.Throws<FormatException>(() => StartupWarmupPolicy.ResolvePreferredThinking(setting));

    private static PrefixCacheCapabilities CopyCapabilities(FamilyClass family = FamilyClass.R)
        => new()
        {
            Class = family,
            NamespaceFingerprint = "independent-startup-policy-fixture",
            EndState = EndStateSupport.CopyAndDonate,
            CanCaptureCopy = true,
        };

    [Theory]
    [InlineData(FamilyClass.P)]
    [InlineData(FamilyClass.S)]
    [InlineData(FamilyClass.R)]
    [InlineData(FamilyClass.N)]
    public void ActualCopyAndPublicBudgetSelectTheRealChatPrefixForEveryFamily(FamilyClass family)
    {
        Assert.False(StartupWarmupPolicy.UseKernelInitializationOnly(
            true, true, CopyCapabilities(family), 4));
    }

    [Fact]
    public void DonateOnlyStateCannotPrepareACopiedSharedPublicPrefix()
    {
        var capabilities = CopyCapabilities() with
        {
            EndState = EndStateSupport.DonateOnly,
            CanCaptureCopy = false,
        };
        Assert.True(StartupWarmupPolicy.UseKernelInitializationOnly(true, true, capabilities, 4));
    }

    [Fact]
    public void CopyFlagIsRequiredEvenWhenTheEndStateSupportsCopies()
    {
        var capabilities = CopyCapabilities() with { CanCaptureCopy = false };
        Assert.True(StartupWarmupPolicy.UseKernelInitializationOnly(true, true, capabilities, 4));
    }

    [Fact]
    public void RequestedCacheDoesNotSubstituteForAnActuallyAttachedCache()
    {
        var requested = new SchedulerConfig
        {
            EnablePrefixCaching = true,
            ExecutionOptionsOverride = new ExecutionOptions { PrefixCheckpointBudget = 4 },
        };
        Assert.True(requested.EnablePrefixCaching);
        Assert.True(StartupWarmupPolicy.UseKernelInitializationOnly(true,
            prefixCacheActive: false, CopyCapabilities(), requested.ResolveExecutionOptions().PrefixCheckpointBudget));
    }

    [Fact]
    public void MissingCopyContractSelectsOnlyKernelInitialization()
    {
        Assert.True(StartupWarmupPolicy.UseKernelInitializationOnly(true, true, null, 4));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    public void NonpositiveEffectivePublicBudgetCannotPreparePublicCheckpoints(int budget)
    {
        Assert.True(StartupWarmupPolicy.UseKernelInitializationOnly(true, true, CopyCapabilities(), budget));
    }

    [Theory]
    [InlineData(1)]
    [InlineData(4)]
    [InlineData(int.MaxValue)]
    public void APositiveEffectivePublicBudgetAllowsCopiedPrefixWarmup(int budget)
    {
        Assert.False(StartupWarmupPolicy.UseKernelInitializationOnly(true, true, CopyCapabilities(), budget));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void NonUnifiedExecutionPreservesItsExistingWarmupSelection(bool active)
    {
        Assert.False(StartupWarmupPolicy.UseKernelInitializationOnly(false, active, null, 0));
    }

    [Fact]
    public void SessionOverrideWinsOverTheRequestedPublicCheckpointBudget()
    {
        var requested = new SchedulerConfig
        {
            EnablePrefixCaching = true,
            ExecutionOptionsOverride = new ExecutionOptions { PrefixCheckpointBudget = 4 },
        };
        var effective = new SchedulerConfig
        {
            EnablePrefixCaching = requested.EnablePrefixCaching,
            ExecutionOptionsOverride = new ExecutionOptions { PrefixCheckpointBudget = 0 },
        };
        Assert.Equal(4, requested.ResolveExecutionOptions().PrefixCheckpointBudget);
        Assert.Equal(0, effective.ResolveExecutionOptions().PrefixCheckpointBudget);
        Assert.True(StartupWarmupPolicy.UseKernelInitializationOnly(true, true, CopyCapabilities(),
            effective.ResolveExecutionOptions().PrefixCheckpointBudget));
    }

    [Theory]
    [InlineData(FamilyClass.P)]
    [InlineData(FamilyClass.S)]
    [InlineData(FamilyClass.R)]
    [InlineData(FamilyClass.N)]
    public void OneActualCopySlotWarmsOnlyThePreferredVariantForEveryFamily(FamilyClass family)
        => Assert.True(StartupWarmupPolicy.UseSinglePreferredVariant(true, true, CopyCapabilities(family), 1));

    [Theory]
    [InlineData(false, true, true, 1)]
    [InlineData(true, false, true, 1)]
    [InlineData(true, true, false, 1)]
    [InlineData(true, true, true, 0)]
    [InlineData(true, true, true, 2)]
    [InlineData(true, true, true, int.MaxValue)]
    public void OtherExecutionOrCopyBudgetsPreserveBothModeSelection(bool unified, bool attached, bool copies, int budget)
        => Assert.False(StartupWarmupPolicy.UseSinglePreferredVariant(unified, attached,
            CopyCapabilities() with { CanCaptureCopy = copies }, budget));

    [Fact]
    public void MissingCopyCapabilitiesCannotSelectASingleReusableVariant()
        => Assert.False(StartupWarmupPolicy.UseSinglePreferredVariant(true, true, null, 1));
}
