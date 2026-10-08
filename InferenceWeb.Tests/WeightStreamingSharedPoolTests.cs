// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using TensorSharp.Memory;
using TensorSharp.Models;
using TensorSharp.GGML;

namespace InferenceWeb.Tests;

public sealed class WeightStreamingSharedPoolTests
{
    [Theory]
    [InlineData(8, 64)]
    [InlineData(1, 35)]
    public void HostAndDeviceSharePool_PlannerIncludesBothPhysicalAllocations(int weightType, long width)
    {
        var budget = Budget();
        var options = Options(budget);
        using var staging = new StreamingHostBuffer(options, 1024);
        // N=2 needs 1024 device bytes plus a distinct 64-byte host output tile.
        // Only 1024 shared bytes remain, so independent per-pool checks would
        // wrongly admit it. N=1 with three output rows requires 768+64 instead.
        long BackendPayload(long k, int rows, int tokens) => Payload(k, rows, tokens, weightType);
        var layout = WeightStreamingExecutor.SelectTileLayout(options, width, 100, 8, BackendPayload, weightType);
        Assert.Equal((3, 1), layout);
        using (var output = new StreamingHostBuffer(options, layout.WeightRows * layout.TokenRows * sizeof(float)))
        using (var device = budget.Reserve(options.DevicePools.Select(pool =>
                   new MemoryCharge(pool, BackendPayload(width, layout.WeightRows, layout.TokenRows)))))
        {
            device.Commit();
            Assert.Equal(1856, budget.Snapshot().Single(pool => pool.Pool == "shared").Committed);
            Assert.Equal(768, budget.Snapshot().Single(pool => pool.Pool == "gpu").Committed);
            Assert.All(budget.Snapshot(), pool => Assert.True(pool.Available >= 0));
        }
        Assert.Equal(1024, budget.Snapshot().Single(pool => pool.Pool == "shared").Committed);
        Assert.Equal(0, budget.Snapshot().Single(pool => pool.Pool == "gpu").Committed);
    }

    [Theory]
    [InlineData(8, 64)]
    [InlineData(1, 35)]
    public void SharedCapacityTakenAfterPlanning_DeviceRefusalReleasesTemporaryHostCredit(int weightType, long width)
    {
        var budget = Budget();
        var options = Options(budget);
        using var staging = new StreamingHostBuffer(options, 1024);
        long BackendPayload(long k, int rows, int tokens) => Payload(k, rows, tokens, weightType);
        var layout = WeightStreamingExecutor.SelectTileLayout(options, width, 100, 8, BackendPayload, weightType);
        using var competitor = budget.Reserve([new MemoryCharge("shared", 256)]);
        competitor.Commit();
        // Reproduce the executor's immediate host reservation followed by the
        // device reservation. A competing owner can invalidate the plan; failure
        // must return pressure and unwind output staging, with no partial charge
        // to the independent GPU constraint or release of the competitor.
        Assert.Throws<MemoryPressureException>(() =>
        {
            using var output = new StreamingHostBuffer(options, layout.WeightRows * layout.TokenRows * sizeof(float));
            using var device = budget.Reserve(options.DevicePools.Select(pool =>
                new MemoryCharge(pool, BackendPayload(width, layout.WeightRows, layout.TokenRows))));
        });
        Assert.Equal(1280, budget.Snapshot().Single(pool => pool.Pool == "shared").Committed);
        Assert.Equal(0, budget.Snapshot().Single(pool => pool.Pool == "gpu").Committed);
        Assert.All(budget.Snapshot(), pool => Assert.Equal(0, pool.Reserved));
        competitor.Dispose();
        Assert.Equal(layout, WeightStreamingExecutor.SelectTileLayout(options, width, 100, 8, BackendPayload, weightType));
    }

    [Theory]
    [InlineData(8, 64, 2)]
    [InlineData(1, 35, 1)]
    public void FileReadTileLimitUsesTheFormatRowBytes(int weightType, long width, int expectedRows)
    {
        var budget = new MemoryBudget([new MemoryCharge("shared", 2048), new MemoryCharge("gpu", 4096)]);
        var options = new WeightStreamingOptions(budget, "shared", ["shared", "gpu"], tileBytes: 136, tokenTileRows: 1);
        using var staging = new StreamingHostBuffer(options, options.TileBytes);
        long BackendPayload(long k, int rows, int tokens) => Payload(k, rows, tokens, weightType);
        var layout = WeightStreamingExecutor.SelectTileLayout(options, width, 100, 1, BackendPayload, weightType);
        // Two Q8 rows consume 136 bytes; two odd-width F16 rows consume 140.
        // Device quota is ample: only the fixed file-read tile limits rows.
        Assert.Equal((expectedRows, 1), layout);
        long rowBytes = weightType == 1 ? width * 2 : width / 32 * 34;
        Assert.True(layout.WeightRows * rowBytes <= options.TileBytes);
        Assert.True((layout.WeightRows + 1) * rowBytes > options.TileBytes);
        Assert.Equal(192, budget.Snapshot().Single(pool => pool.Pool == "shared").Committed);
        Assert.Equal(0, budget.Snapshot().Single(pool => pool.Pool == "gpu").Reserved);
        Assert.Equal(0, budget.Snapshot().Single(pool => pool.Pool == "gpu").Committed);
    }

    [Theory]
    [InlineData(8, GgmlWeightStreamingArithmetic.ResidentCuda, 65535)]
    [InlineData(8, GgmlWeightStreamingArithmetic.FullPrecision, 65536)]
    [InlineData(1, GgmlWeightStreamingArithmetic.ResidentCuda, 65536)]
    public void ActiveTokenLimitDoesNotRejectALargerLogicalOperation(int type,
        GgmlWeightStreamingArithmetic arithmetic, int expectedTokens)
    {
        const int logicalTokens = 65536;
        var budget = new MemoryBudget([new MemoryCharge("host", 1L << 32), new MemoryCharge("gpu", 1L << 32)]);
        var options = new WeightStreamingOptions(budget, "host", ["gpu"], tileBytes: 64, tokenTileRows: logicalTokens);
        int largestQueriedBatch = 0;
        long Query(long width, int rows, int activeTokens)
        {
            // Model the actual native Q8 grid.y constraint. Its original N
            // remains 65536, including after the active workspace is capped.
            Assert.True(activeTokens <= expectedTokens);
            largestQueriedBatch = Math.Max(largestQueriedBatch, activeTokens);
            return Payload(width, rows, activeTokens, type);
        }
        Assert.Equal((1, expectedTokens), WeightStreamingExecutor.SelectTileLayout(options, 32, 128,
            logicalTokens, Query, type, arithmetic));
        Assert.Equal(expectedTokens, largestQueriedBatch);
        Assert.All(budget.Snapshot(), pool => Assert.Equal(0, pool.Reserved + pool.Committed));
    }

    [Fact]
    public void ResidentQuantizedInputIndexLimitShrinksTheBatchBeforeNativeQuery()
    {
        const long width = 131072;
        var budget = new MemoryBudget([new MemoryCharge("host", 1L << 32), new MemoryCharge("gpu", 1L << 40)]);
        var options = new WeightStreamingOptions(budget, "host", ["gpu"], tileBytes: 1 << 20, tokenTileRows: 65535);
        long Query(long k, int rows, int tokens)
        {
            Assert.True((long)tokens * (k / 32) * 9 <= int.MaxValue,
                "The pinned quantizer's int stride cannot represent this active batch.");
            return Payload(k, rows, tokens, 8);
        }
        Assert.Equal((7, 32767), WeightStreamingExecutor.SelectTileLayout(options, width, 128, 65536,
            Query, 8, GgmlWeightStreamingArithmetic.ResidentCuda));
        Assert.All(budget.Snapshot(), pool => Assert.Equal(0, pool.Reserved + pool.Committed));
    }

    [Fact]
    public void UnsupportedNativeDeviceIsNotConvertedIntoBudgetPressure()
    {
        var budget = new MemoryBudget([new MemoryCharge("host", 0), new MemoryCharge("gpu", 0)]);
        var options = new WeightStreamingOptions(budget, "host", ["gpu"], tileBytes: 64, tokenTileRows: 65536);
        var unsupported = new InvalidOperationException("The initialized device does not support resident arithmetic.");
        Assert.Same(unsupported, Assert.Throws<InvalidOperationException>(() =>
            WeightStreamingExecutor.SelectTileLayout(options, 32, 128, 65536,
                (_, _, _) => throw unsupported, 8, GgmlWeightStreamingArithmetic.ResidentCuda)));
    }

    [Fact]
    public void UnexpectedNativeQueryFailureIsNotSilentlyRetriedAsASmallerTile()
    {
        var budget = new MemoryBudget([new MemoryCharge("host", 1 << 20), new MemoryCharge("gpu", 1 << 20)]);
        var options = new WeightStreamingOptions(budget, "host", ["gpu"], tileBytes: 1024);
        var unexpected = new InvalidOperationException("Native device query failed.");
        int calls = 0;
        long Query(long _, int rows, int tokens)
        {
            calls++;
            if (rows == 1 && tokens == 1) return 1024;
            throw unexpected;
        }
        Assert.Same(unexpected, Assert.Throws<InvalidOperationException>(() =>
            WeightStreamingExecutor.SelectTileLayout(options, 32, 128, 32,
                Query, 8, GgmlWeightStreamingArithmetic.ResidentCuda)));
        Assert.Equal(2, calls);
    }

    private static MemoryBudget Budget()
        => new([new MemoryCharge("shared", 2048), new MemoryCharge("gpu", 1024)]);

    private static WeightStreamingOptions Options(MemoryBudget budget)
        => new(budget, "shared", ["shared", "gpu"], tileBytes: 1024, tokenTileRows: 8);

    // Independent fake backend with aligned input, weight and output regions.
    // Native integration tests validate the actual CUDA workspace separately.
    private static long Payload(long width, int rows, int tokens, int weightType)
        => Align(width * tokens * sizeof(float)) + Align((weightType == 1 ? width * 2 : width / 32 * 34) * rows)
            + Align((long)rows * tokens * sizeof(float));
    private static long Align(long bytes) => (bytes + 255) / 256 * 256;
}
