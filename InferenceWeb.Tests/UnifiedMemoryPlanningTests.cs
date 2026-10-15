// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using TensorSharp.Models;
using Xunit;

namespace InferenceWeb.Tests;

public sealed class UnifiedMemoryPlanningTests
{
    private const long MiB = 1L << 20;

    [Fact]
    public void SessionCacheEnablesBoundedFileReadsWithoutLegacyEnvironment()
    {
        Assert.True(ModelBase.HasExpertFileStagingPolicy(8192 * MiB, null));
        Assert.True(ModelBase.HasExpertFileStagingPolicy(8192 * MiB, "0"));
        Assert.True(ModelBase.HasExpertFileStagingPolicy(0, "0"));
        Assert.True(ModelBase.HasExpertFileStagingPolicy(null, "8192"));
        Assert.False(ModelBase.HasExpertFileStagingPolicy(null, null));
        Assert.False(ModelBase.HasExpertFileStagingPolicy(null, long.MaxValue.ToString()));
    }

    [Theory]
    [InlineData(16384, 4096, 3072, 32768, 2048, 32768, 9216)]
    [InlineData(8192, 4096, 3072, 32768, 2048, 32768, 0)]
    [InlineData(16384, 4096, 3072, 4096, 2048, 32768, 4096)]
    [InlineData(16384, 4096, 3072, 32768, 2048, 6144, 6144)]
    [InlineData(16384, 4096, 3072, 32768, 0, 32768, 0)]
    [InlineData(16384, 4096, 3072, 32768, 2048, 0, 0)]
    public void CacheUsesOnlyHardwareSlackAfterDenseAndExecutionPeaks(long available,
        long dense, long execution, long source, long minimum, long ceiling, long expected)
        => Assert.Equal(expected * MiB, UnifiedMemoryModelSession.SelectExpertCacheBytes(
            available * MiB, dense * MiB, execution * MiB, source * MiB, minimum * MiB, ceiling * MiB));

    [Fact]
    public void AllocationRoundingCannotCrossTheSharedCeiling()
        => Assert.Equal(2 * MiB, UnifiedMemoryModelSession.SelectExpertCacheBytes(5 * MiB + 123,
            MiB, 2 * MiB, 8 * MiB, MiB, long.MaxValue));

    [Fact]
    public void CapacityArithmeticDoesNotWrapForLargeInputs()
    {
        Assert.Equal(0, UnifiedMemoryModelSession.SelectExpertCacheBytes(long.MaxValue,
            long.MaxValue, long.MaxValue, long.MaxValue, 1, long.MaxValue));
        Assert.Throws<ArgumentOutOfRangeException>(() => UnifiedMemoryModelSession.SelectExpertCacheBytes(
            100, -1, 0, 1, 1, 1));
    }

    [Fact]
    public void UnsupportedBackendIsRejectedBeforeFileOrNativeAccess()
        => Assert.Throws<NotSupportedException>(() => UnifiedMemoryModelSession.Create(
            "not-opened.gguf", null, BackendType.Cpu));

    [Fact]
    public void InvalidPolicyIsRejectedBeforeHardwareOrFileAccess()
    {
        foreach (var options in new[]
        {
            new UnifiedMemoryModelOptions { ContextTokens = -1 },
            new UnifiedMemoryModelOptions { PrefillChunkTokens = 513 },
            new UnifiedMemoryModelOptions { PrefillChunkTokens = 0 },
            new UnifiedMemoryModelOptions { ExpertCachePrefillRows = 0 },
            new UnifiedMemoryModelOptions { ExpertCachePrefillRows = 129 },
            new UnifiedMemoryModelOptions { MaximumHostBytes = 0 },
            new UnifiedMemoryModelOptions { MaximumDeviceBytes = -1 },
            new UnifiedMemoryModelOptions { MaximumExpertCacheBytes = -1 }
        }) Assert.Throws<ArgumentOutOfRangeException>(() => UnifiedMemoryModelSession.Create(
            "not-opened.gguf", null, BackendType.GgmlCuda, options));
    }
}
