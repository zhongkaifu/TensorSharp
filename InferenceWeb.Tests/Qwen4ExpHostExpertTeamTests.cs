// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp.Models;
using Xunit;

namespace InferenceWeb.Tests;

/// <summary>The direct-CUDA Qwen 3.8 Flash Next executor's default host-expert team
/// (<see cref="Qwen4ExpCudaExecutor.HostExpertTeamThreads"/>).</summary>
public sealed class Qwen4ExpHostExpertTeamTests
{
    /// <summary>All CPUs but three, at least half, at most 32, on every machine size: from 66 CPUs, where half passes
    /// the cap, the cap holds instead of throwing.</summary>
    [Theory]
    [InlineData(1, 1)]
    [InlineData(2, 1)]
    [InlineData(4, 2)]
    [InlineData(14, 11)]
    [InlineData(35, 32)]
    [InlineData(64, 32)]
    [InlineData(65, 32)]
    [InlineData(66, 32)]
    [InlineData(80, 32)]
    [InlineData(192, 32)]
    public void TheHostExpertTeam_IsAllButThreeCpus_AtLeastHalf_AtMost32(int cpus, int threads)
        => Assert.Equal(threads, Qwen4ExpCudaExecutor.HostExpertTeamThreads(cpus));
}
