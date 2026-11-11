// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp.Models;

namespace InferenceWeb.Tests;

/// <summary>
/// Four concurrent Flash Next requests on ggml_cuda (RTX 5090, 32 GB) aborted the server inside ggml: every request's
/// holder was the whole 131072-token window (3.6 GB of K/V and indexer keys), the 19.1 GB expert tier left ~4.6 GB, the
/// batched arena failed on every step, and the fourth request's step could not create a cuBLAS handle. Now holders start
/// small (Qwen4ExpFreshHolderCapacityTests), a request that does not fit waits (the Metal admission rule, applied to one
/// CUDA GPU), the expert tier keeps VRAM for TS_Q4E_LANES sequences, and a per-request backend that cannot get a cuBLAS
/// handle fails that request instead of the process. These pin the sizing rules.
/// </summary>
public class Qwen4ExpGgmlCudaMemoryTests
{
    [Theory]
    [InlineData(null, 1)]
    [InlineData("", 1)]
    [InlineData("4", 4)]
    [InlineData("0", 1)]
    [InlineData("12", 12)]
    [InlineData("20", 16)]
    [InlineData("-2", 1)]
    [InlineData("two", 1)]
    public void TheLaneCount_ComesFromTsQ4eLanes_ClampedToOneToSixteen(string value, int expected)
        => Assert.Equal(expected, ExpertPlacementPlanner.GgmlCudaLanes(value));

    [Fact]
    public void OneLane_ReservesNothing_AndEachFurtherLane_AHolderAndItsState()
    {
        const long holder = 226L << 20, extra = 256L << 20;
        Assert.Equal(0, ExpertPlacementPlanner.ConcurrentSequenceReserveBytes(1, holder, extra));
        Assert.Equal(3 * (holder + extra), ExpertPlacementPlanner.ConcurrentSequenceReserveBytes(4, holder, extra));
        Assert.Throws<ArgumentOutOfRangeException>(() => ExpertPlacementPlanner.ConcurrentSequenceReserveBytes(0, holder, extra));
    }

    /// <summary>Flash Next's 12 attention layers at 8192 rows: 2 KV heads x 256 dims of K and V plus 128 indexer dims
    /// each, as F16 (226 MiB) and as Q8_0 (34 bytes a 32-value block).</summary>
    [Fact]
    public void CacheBytes_FollowTheStorageType()
    {
        long elements = 12L * 8192 * (2 * 2 * 256 + 128);
        Assert.Equal(elements * 2, Qwen4ExpModel.KvBytes(KvCacheDtype.F16, elements));
        Assert.Equal(226_492_416, Qwen4ExpModel.KvBytes(KvCacheDtype.F16, elements));
        Assert.Equal(elements * 4, Qwen4ExpModel.KvBytes(KvCacheDtype.F32, elements));
        Assert.Equal(elements / 32 * 34, Qwen4ExpModel.KvBytes(KvCacheDtype.Q8_0, elements));
        Assert.Equal(elements / 32 * 18, Qwen4ExpModel.KvBytes(KvCacheDtype.Q4_0, elements));
    }
}
