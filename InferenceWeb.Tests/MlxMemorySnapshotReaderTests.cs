// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using TensorSharp.MLX;

namespace InferenceWeb.Tests;

public sealed class MlxMemorySnapshotReaderTests
{
    [Fact]
    public void GenuineZeroIsSuccessfulWithoutReadingErrorOrEvaluatingAnything()
    {
        int queries = 0;
        int Zero(ref nuint value) { queries++; value = 0; return 0; }
        var result = MlxMemorySnapshotReader.Read(Zero, Zero, Zero,
            () => throw new InvalidOperationException("Success must not consume another operation's diagnostic."));
        Assert.True(result.Success);
        Assert.Equal(new MlxMemorySnapshot(0, 0, 0), result.Snapshot);
        Assert.Null(result.Error);
        Assert.Equal(3, queries);
    }

    [Theory]
    [InlineData(0, "active")]
    [InlineData(1, "cache")]
    [InlineData(2, "peak")]
    public void AnyFailedCounterInvalidatesTheEntireSampleAndStopsFurtherQueries(int failureIndex, string name)
    {
        int queries = 0, errors = 0;
        int Query(ref nuint value) { value = (nuint)(++queries * 100); return queries - 1 == failureIndex ? -7 : 0; }
        var result = MlxMemorySnapshotReader.Read(Query, Query, Query, () => { errors++; return "native failure"; });
        Assert.False(result.Success);
        Assert.Equal(default, result.Snapshot); // earlier successful counters cannot look like a complete sample
        Assert.Equal(failureIndex + 1, queries);
        Assert.Equal(1, errors);
        Assert.Contains(name, result.Error);
        Assert.Contains("-7", result.Error);
        Assert.Contains("native failure", result.Error);
    }

    [Fact]
    public void MissingNativeEntryPointMeansUnknownAndDoesNotTryLaterCounters()
    {
        int calls = 0;
        int Missing(ref nuint value) { calls++; throw new EntryPointNotFoundException("mlx_get_active_memory"); }
        int Unexpected(ref nuint value) => throw new Exception("Must not query after failure.");
        var result = MlxMemorySnapshotReader.Read(Missing, Unexpected, Unexpected, () => "unused");
        Assert.False(result.Success);
        Assert.Equal(1, calls);
        Assert.Contains("unknown", result.Error);
        Assert.Contains("mlx_get_active_memory", result.Error);
    }

    [Fact]
    public void EmptyNativeDiagnosticStillReportsCounterAndStatus()
    {
        int Failed(ref nuint value) => 12;
        var result = MlxMemorySnapshotReader.Read(Failed, Failed, Failed, () => "");
        Assert.False(result.Success);
        Assert.Contains("active", result.Error);
        Assert.Contains("12", result.Error);
    }

    [Fact]
    public void SuccessfulIndependentGaugesArePreservedWithoutSummingOrInferringOwnership()
    {
        int Active(ref nuint value) { value = 300; return 0; }
        int Cache(ref nuint value) { value = 200; return 0; }
        int Peak(ref nuint value) { value = 700; return 0; }
        var result = MlxMemorySnapshotReader.Read(Active, Cache, Peak, () => "unused");
        Assert.True(result.Success);
        Assert.Equal(new MlxMemorySnapshot(300, 200, 700), result.Snapshot);
    }
}
