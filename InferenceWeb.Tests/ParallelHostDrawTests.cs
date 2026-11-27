// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Large Philox draws and LTX rotary tables run in chunks on several cores; every value depends only on its own counter or
// token, so the result must equal small sequential pieces of it bit for bit.
using TensorSharp.Models.Ltx25;
using TensorSharp.Models.SharedDiffusion;

namespace InferenceWeb.Tests;

public sealed class ParallelHostDrawTests
{
    [Theory]
    [InlineData(0L, 0UL, 0u)]
    [InlineData(3206401772581117598L, 1UL, 7u)]
    public void ALargePhiloxDrawEqualsItsSequentialPieces(long seed, ulong stream, uint draw)
    {
        const int count = 200_003, piece = 1_000;
        float[] whole = PhiloxNoise.Draw(count, seed, stream, draw);
        for (int start = 0; start < count; start += piece)
        {
            int n = Math.Min(piece, count - start);
            float[] part = PhiloxNoise.Draw(n, seed, stream, draw, (uint)start);
            for (int i = 0; i < n; i++)
                Assert.Equal(BitConverter.SingleToInt32Bits(part[i]), BitConverter.SingleToInt32Bits(whole[start + i]));
        }
    }

    [Theory]
    [InlineData(Ltx25FrequencyPrecision.Float64)]
    [InlineData(Ltx25FrequencyPrecision.Float32)]
    public void ALargeRotaryTableEqualsItsSequentialRows(Ltx25FrequencyPrecision precision)
    {
        // A 1024x1024x17 refinement's grid: 3072 tokens, far past the parallel threshold at 4096 wide.
        Ltx25PositionGrid grid = Ltx25Geometry.VideoPositions(3, 32, 32, 24);
        Ltx25Rope whole = Ltx25Geometry.BuildRope(grid, 4096, 32, precision);
        int half = 4096 / 2, rows = 8;
        for (int token = 0; token < grid.Tokens; token += rows)
        {
            int n = Math.Min(rows, grid.Tokens - token);
            var slice = new Ltx25PositionGrid(n, grid.Axes, grid.Boundaries.AsSpan(token * grid.Axes * 2, n * grid.Axes * 2).ToArray());
            Ltx25Rope part = Ltx25Geometry.BuildRope(slice, 4096, 32, precision);
            for (int i = 0; i < n * half; i++)
            {
                Assert.Equal(BitConverter.SingleToInt32Bits(part.Cosine[i]), BitConverter.SingleToInt32Bits(whole.Cosine[token * half + i]));
                Assert.Equal(BitConverter.SingleToInt32Bits(part.Sine[i]), BitConverter.SingleToInt32Bits(whole.Sine[token * half + i]));
            }
        }
    }

    [Fact]
    public void ALargeRotaryTableStillRejectsANonfinitePhase()
    {
        var boundaries = new float[4096 * 3 * 2];
        boundaries[^1] = float.MaxValue;
        boundaries[^2] = float.MaxValue;
        Assert.Throws<ArgumentException>(() => Ltx25Geometry.BuildRope(new Ltx25PositionGrid(4096, 3, boundaries), 4096, 32));
    }
}
