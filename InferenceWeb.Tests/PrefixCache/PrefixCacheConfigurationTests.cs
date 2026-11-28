// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Speculative;

namespace InferenceWeb.Tests.PrefixCache;

[Collection(EngineEnvironmentCollection.Name)]

public class PrefixCacheConfigurationTests
{
    [Fact]
    public void PrefixCachingIsOnByDefaultAndSurvivesASpeculationChange()
    {
        Assert.True(SchedulerConfig.Default.EnablePrefixCaching);
        Assert.True(SchedulerConfig.Default.WithSpeculation(SpeculationOptions.Disabled).EnablePrefixCaching);
        Assert.False(new SchedulerConfig { EnablePrefixCaching = false }
            .WithSpeculation(SpeculationOptions.Disabled).EnablePrefixCaching);
    }

    /// <summary>Only a value the user gave counts as a chosen lone-prompt step: unset, a model with its own
    /// steps (Muse Spark's pipelined prefill) sizes them.</summary>
    [Fact]
    public void TheSoloPrefillChunkIsChosenOnlyWhenTheEnvironmentSetsIt()
    {
        using var env = new EnvScope();
        env.Set("TS_SCHED_SOLO_PREFILL_CHUNK", null);
        var unset = SchedulerConfig.FromEnvironment();
        Assert.False(unset.SoloPrefillChunkSizeChosen);
        Assert.Equal(8192, unset.SoloPrefillChunkSize);
        env.Set("TS_SCHED_SOLO_PREFILL_CHUNK", "32768");
        var set = SchedulerConfig.FromEnvironment();
        Assert.True(set.SoloPrefillChunkSizeChosen);
        Assert.Equal(32768, set.SoloPrefillChunkSize);
    }

    /// <summary>The mixed-step prompt bound comes from the environment alone: unset or 0, none.</summary>
    [Fact]
    public void TheMixedStepPrefillBudgetComesFromTheEnvironment()
    {
        using var env = new EnvScope();
        env.Set("TS_SCHED_MIXED_PREFILL_TOKENS", null);
        var unset = SchedulerConfig.FromEnvironment();
        Assert.Equal(0, unset.MixedStepPrefillBudget);
        Assert.Equal(0, unset.WithSpeculation(SpeculationOptions.Disabled).MixedStepPrefillBudget);
        env.Set("TS_SCHED_MIXED_PREFILL_TOKENS", "0");
        Assert.Equal(0, SchedulerConfig.FromEnvironment().MixedStepPrefillBudget);
        env.Set("TS_SCHED_MIXED_PREFILL_TOKENS", "512");
        var bound = SchedulerConfig.FromEnvironment();
        Assert.Equal(512, bound.MixedStepPrefillBudget);
        Assert.Equal(512, bound.WithSpeculation(SpeculationOptions.Disabled).MixedStepPrefillBudget);
    }

    [Fact]
    public void TheOffSwitchTurnsPrefixCachingOff()
    {
        using var env = new EnvScope();
        env.Set("TS_SCHED_PREFIX_CACHE", "0");
        Assert.False(SchedulerConfig.FromEnvironment().EnablePrefixCaching);
    }

    /// <summary>The radix cache is the only prefix cache. A deployment still asking for the
    /// removed legacy mode hears so instead of running on something it did not ask for.</summary>
    [Theory]
    [InlineData("legacy")]
    [InlineData("tree")]
    public void TheRemovedModeVariable_ErrorsNamingWhatReplacesIt(string value)
    {
        using var env = new EnvScope();
        env.Set("TS_PREFIX_CACHE_MODE", value);
        var ex = Assert.Throws<ArgumentException>(() => SchedulerConfig.FromEnvironment());
        Assert.Contains("TS_PREFIX_CACHE_MODE", ex.Message);
        Assert.Contains("TS_SCHED_PREFIX_CACHE", ex.Message);
    }
}
