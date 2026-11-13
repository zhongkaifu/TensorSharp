// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using TensorSharp.MLX;

namespace InferenceWeb.Tests;

/// <summary>MLX's default wired and allocator ceiling leaves macOS room: wired memory is never paged, and a ceiling
/// near the RAM hung a 48 GiB Mac twice (49.1 of 51.5 GB wired, watchdog panics).</summary>
public sealed class MlxMemoryCeilingTests
{
    [Theory]
    [InlineData(49152, 34406)]     // 48 GiB: 70% (the old default wired 43008)
    [InlineData(131072, 91750)]    // 128 GiB: 70%
    [InlineData(16384, 11468)]     // 16 GiB: 70%
    [InlineData(8192, 5734)]       // 8 GiB (a phone): 70%
    [InlineData(4096, 2867)]
    [InlineData(0, 0)]
    public void TheCeilingLeavesMacOsAtLeastAQuarterOrEightGiB(long physMb, long expected)
    {
        long ceiling = MlxNative.DefaultMlxMemoryCeilingMb(physMb);
        Assert.Equal(expected, ceiling);
        Assert.True(physMb - ceiling >= System.Math.Min(8192, physMb / 4));
    }
}
