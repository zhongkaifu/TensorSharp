// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Scheduling.PrefixCache;
using TensorSharp.Runtime.Speculative;

namespace InferenceWeb.Tests.PrefixCache;

public class PrefixCacheConfigurationTests
{
    [Fact]
    public void SchedulerDefaultsToRadixAndKeepsModeWhenSpeculationChanges()
    {
        Assert.Equal(PrefixCacheMode.Tree, SchedulerConfig.Default.PrefixCacheMode);
        Assert.Equal(PrefixCacheMode.Tree, SchedulerConfig.Default.WithSpeculation(SpeculationOptions.Disabled).PrefixCacheMode);
        Assert.Equal(PrefixCacheMode.Legacy, new SchedulerConfig { PrefixCacheMode = PrefixCacheMode.Legacy }
            .WithSpeculation(SpeculationOptions.Disabled).PrefixCacheMode);
    }

    [Theory]
    [InlineData(null, PrefixCacheMode.Tree)]
    [InlineData("tree", PrefixCacheMode.Tree)]
    [InlineData(" TREE ", PrefixCacheMode.Tree)]
    [InlineData("legacy", PrefixCacheMode.Legacy)]
    public void EnvironmentChoosesModeWithoutOverridingTheOffSwitch(string? value, PrefixCacheMode expected)
    {
        string? savedMode = Environment.GetEnvironmentVariable("TS_PREFIX_CACHE_MODE");
        string? savedEnabled = Environment.GetEnvironmentVariable("TS_SCHED_PREFIX_CACHE");
        try
        {
            Environment.SetEnvironmentVariable("TS_PREFIX_CACHE_MODE", value);
            Environment.SetEnvironmentVariable("TS_SCHED_PREFIX_CACHE", "0");
            var config = SchedulerConfig.FromEnvironment();
            Assert.Equal(expected, config.PrefixCacheMode);
            Assert.False(config.EnablePrefixCaching);
        }
        finally
        {
            Environment.SetEnvironmentVariable("TS_PREFIX_CACHE_MODE", savedMode);
            Environment.SetEnvironmentVariable("TS_SCHED_PREFIX_CACHE", savedEnabled);
        }
    }

    [Theory]
    [InlineData("shadow")]
    [InlineData("tre")]
    public void UnsupportedModeFailsInsteadOfSilentlyUsingLegacy(string value)
    {
        string? savedMode = Environment.GetEnvironmentVariable("TS_PREFIX_CACHE_MODE");
        try
        {
            Environment.SetEnvironmentVariable("TS_PREFIX_CACHE_MODE", value);
            Assert.Throws<ArgumentException>(() => SchedulerConfig.FromEnvironment());
        }
        finally { Environment.SetEnvironmentVariable("TS_PREFIX_CACHE_MODE", savedMode); }
    }
}
