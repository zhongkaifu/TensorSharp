// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using TensorSharp.Models;
using TensorSharp.Server;
using TensorSharp.Server.Hosting;

namespace InferenceWeb.Tests;

/// <summary>Host option resolution only: no model files, native allocator or GPU
/// execution. Process environment changes are isolated and restored.</summary>
[Collection(EngineEnvironmentCollection.Name)]
public sealed class UnifiedMemoryHostOptionsTests : IDisposable
{
    private readonly Dictionary<string, string> _previous = new();
    public UnifiedMemoryHostOptionsTests()
    {
        foreach (string name in new[] { "TS_SCHED_PREFILL_CHUNK", "TS_PER_SEQ_FUSED", "TS_BATCHED_FUSED_DECODE",
            "TS_SCHED_PREFIX_CACHE", "TS_PREFIX_CHECKPOINTS_MAX" })
        {
            _previous.Add(name, Environment.GetEnvironmentVariable(name));
            Environment.SetEnvironmentVariable(name, null);
        }
    }
    public void Dispose()
    {
        foreach (var pair in _previous) Environment.SetEnvironmentVariable(pair.Key, pair.Value);
    }

    [Fact]
    public void ExplicitUnifiedHostSelectsThePlannedGlmTierWithoutChangingApiDefaultsOrEnvironment()
    {
        var expected = new UnifiedMemoryModelOptions { EnableGlmSelectedExperts = true, EnableAdaptiveGlmPrefill = true,
            AdaptiveGlmPrefillChunkTokens = 1024 };
        Assert.Equal(expected, UnifiedMemorySessionHandle.ResolveHostOptions());
        Assert.Equal(512, expected.PrefillChunkTokens);
        Assert.Equal(1024, expected.ResolveGlmPrefillChunkTokens());
        Assert.Null(new UnifiedMemoryModelOptions().AdaptiveGlmPrefillChunkTokens);
        Assert.Equal(0, expected.ContextTokens);
        Assert.Equal(3, expected.MaximumRunningRequests);
        Assert.True(expected.EnableGlmSelectedExperts);
        Assert.False(new UnifiedMemoryModelOptions().EnableGlmSelectedExperts);
        Assert.True(expected.EnableAdaptiveGlmPrefill);
        Assert.False(new UnifiedMemoryModelOptions().EnableAdaptiveGlmPrefill);
        Assert.False(expected.EnableAsyncExpertPrefill);
        Assert.False(expected.EnableExpertCacheCapacityGrowth);
        Assert.All(_previous.Keys, name => Assert.Null(Environment.GetEnvironmentVariable(name)));
    }

    [Theory]
    [InlineData("1", 1)]
    [InlineData("16", 16)]
    [InlineData("128", 128)]
    [InlineData("512", 512)]
    public void ExplicitPrefillCapConstrainsBothCommonAndAdaptiveGlmModelLoadOptions(string value, int expected)
    {
        Environment.SetEnvironmentVariable("TS_SCHED_PREFILL_CHUNK", value);
        var options = UnifiedMemorySessionHandle.ResolveHostOptions();
        Assert.Equal(new UnifiedMemoryModelOptions { PrefillChunkTokens = expected,
            EnableGlmSelectedExperts = true, EnableAdaptiveGlmPrefill = true,
            AdaptiveGlmPrefillChunkTokens = expected }, options);
        Assert.Equal(expected, options.ResolveGlmPrefillChunkTokens());
        Assert.Equal(value, Environment.GetEnvironmentVariable("TS_SCHED_PREFILL_CHUNK"));
        // A later load receives a fresh snapshot, rather than mutating a prior
        // session's immutable options or resetting the operator's environment.
        Environment.SetEnvironmentVariable("TS_SCHED_PREFILL_CHUNK", "32");
        Assert.Equal(32, UnifiedMemorySessionHandle.ResolveHostOptions().PrefillChunkTokens);
        Assert.Equal(expected, options.PrefillChunkTokens);
        Assert.Equal(expected, options.AdaptiveGlmPrefillChunkTokens);
    }

    [Theory]
    [InlineData("0")]
    [InlineData("-1")]
    [InlineData("513")]
    [InlineData("2147483648")]
    [InlineData("abc")]
    [InlineData("1.5")]
    [InlineData(" 16 ")]
    [InlineData("+16")]
    public void InvalidPrefillRefusesBeforeAnyArchitectureOrNativeLoad(string value)
    {
        Environment.SetEnvironmentVariable("TS_SCHED_PREFILL_CHUNK", value);
        foreach (string architecture in new[] { "qwen4exp", "glm5next", "glm5-next" })
        {
            var error = Assert.Throws<ArgumentException>(() => UnifiedMemorySessionHandle.CreateOwned(
                Path.Combine(Path.GetTempPath(), "missing-" + architecture + ".gguf"), null, BackendType.GgmlCuda));
            Assert.Contains("TS_SCHED_PREFILL_CHUNK", error.Message);
            Assert.Contains("1 through 512", error.Message);
        }
        Assert.Equal(value, Environment.GetEnvironmentVariable("TS_SCHED_PREFILL_CHUNK"));
    }

    [Theory]
    [InlineData("TS_PER_SEQ_FUSED", "0")]
    [InlineData("TS_PER_SEQ_FUSED", "false")]
    [InlineData("TS_PER_SEQ_FUSED", "FALSE")]
    [InlineData("TS_BATCHED_FUSED_DECODE", "0")]
    [InlineData("TS_BATCHED_FUSED_DECODE", "false")]
    [InlineData("TS_BATCHED_FUSED_DECODE", "FALSE")]
    public void UnsupportedFusedOptOutRefusesAtTheSharedPreloadGuard(string variable, string value)
    {
        Environment.SetEnvironmentVariable(variable, value);
        foreach (string architecture in new[] { "qwen4exp", "glm5next", "glm5-next" })
        {
            var error = Assert.Throws<NotSupportedException>(() => UnifiedMemorySessionHandle.CreateOwned(
                Path.Combine(Path.GetTempPath(), "missing-" + architecture + ".gguf"), null, BackendType.GgmlCuda));
            Assert.Contains(variable, error.Message);
            Assert.Contains("requires", error.Message);
        }
        Assert.Equal(value, Environment.GetEnvironmentVariable(variable));
    }

    [Theory]
    [InlineData("--prefill-chunk-size", "16", 16)]
    [InlineData("--prefill-chunk-size=128", null, 128)]
    [InlineData("--PREFILL-CHUNK-SIZE", "512", 512)]
    public void ActualCliPassOverridesInheritedCapAndReachesUnifiedLoadOptions(
        string flag, string? value, int expected)
    {
        Environment.SetEnvironmentVariable("TS_SCHED_PREFILL_CHUNK", "513");
        string[] args = value == null
            ? ["--unified-memory", "--backend", "ggml_cuda", flag]
            : ["--unified-memory", "--backend", "ggml_cuda", flag, value];
        Assert.True(ServerOptionsBuilder.ApplyContinuousBatchingCliFlag(args));
        Assert.True(ServerOptionsBuilder.Build(args, Path.GetTempPath()).UnifiedMemoryEnabled);
        Assert.Equal(expected, UnifiedMemorySessionHandle.ResolveHostOptions().PrefillChunkTokens);
        Assert.Equal(expected, UnifiedMemorySessionHandle.ResolveHostOptions().AdaptiveGlmPrefillChunkTokens);
    }

    [Fact]
    public void ActualCliCapBeyondUnifiedCeilingRefusesBeforeModelLoading()
    {
        string[] args = ["--unified-memory", "--backend", "ggml_cuda", "--prefill-chunk-size=513"];
        Assert.True(ServerOptionsBuilder.ApplyContinuousBatchingCliFlag(args));
        Assert.True(ServerOptionsBuilder.Build(args, Path.GetTempPath()).UnifiedMemoryEnabled);
        var error = Assert.Throws<ArgumentException>(() => UnifiedMemorySessionHandle.CreateOwned(
            Path.Combine(Path.GetTempPath(), "missing-cli-input.gguf"), null, BackendType.GgmlCuda));
        Assert.Contains("1 through 512", error.Message);
    }

    [Theory]
    [InlineData("1")]
    [InlineData("true")]
    [InlineData("TRUE")]
    public void ExplicitEnabledFusedPathsPreserveDefaults(string value)
    {
        Environment.SetEnvironmentVariable("TS_PER_SEQ_FUSED", value);
        Environment.SetEnvironmentVariable("TS_BATCHED_FUSED_DECODE", value);
        Assert.Equal(new UnifiedMemoryModelOptions { EnableGlmSelectedExperts = true, EnableAdaptiveGlmPrefill = true,
                AdaptiveGlmPrefillChunkTokens = 1024 },
            UnifiedMemorySessionHandle.ResolveHostOptions());
    }

    [Theory]
    [InlineData("0", "4", 0)]
    [InlineData("false", "8", 0)]
    [InlineData("1", "0", 0)]
    [InlineData("true", "1", 1)]
    [InlineData("1", "4", 4)]
    [InlineData("1", "8", 8)]
    public void PrefixPlacementHonorsTheActualHostCacheAndOperatorCheckpointBudget(
        string enabled, string copies, int expected)
    {
        Environment.SetEnvironmentVariable("TS_SCHED_PREFIX_CACHE", enabled);
        Environment.SetEnvironmentVariable("TS_PREFIX_CHECKPOINTS_MAX", copies);
        var options = UnifiedMemorySessionHandle.ResolveHostOptions();
        Assert.Equal(expected, options.PrefixCheckpointCount);
        Environment.SetEnvironmentVariable("TS_PREFIX_CHECKPOINTS_MAX", "2");
        Assert.Equal(expected, options.PrefixCheckpointCount); // Immutable load forecast.
        Assert.Equal(enabled, Environment.GetEnvironmentVariable("TS_SCHED_PREFIX_CACHE"));
    }
}
