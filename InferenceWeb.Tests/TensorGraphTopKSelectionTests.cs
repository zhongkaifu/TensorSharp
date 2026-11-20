// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using TensorSharp.Graph;

namespace InferenceWeb.Tests;

/// <summary>
/// The tensor executors' host top-k (TensorGraphExecutor.TopKRows: the managed CPU's and MLX's TopK) selects a row's few picks
/// in one pass where it sorted the row whole: the same columns in the same order (largest first, a tie to the lower column, a
/// NaN below every number), for one pick over a vocabulary and for a router's several, on rows with ties and NaNs.
/// </summary>
public sealed class TensorGraphTopKSelectionTests
{
    private static int[] Sorted(float[] v, long rows, long cols, int k)
    {
        var picked = new int[rows * k];
        var order = new int[cols];
        for (long r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++) order[c] = c;
            long b = r * cols;
            Array.Sort(order, (i, j) => v[b + j].CompareTo(v[b + i]) is var cmp && cmp != 0 ? cmp : i.CompareTo(j));
            Array.Copy(order, 0, picked, r * k, k);
        }
        return picked;
    }

    [Theory]
    [InlineData(1, 4096)]
    [InlineData(1, 202048)]
    [InlineData(2, 64)]
    [InlineData(8, 128)]
    [InlineData(32, 1000)]
    [InlineData(33, 1000)]
    public void SelectionPicksWhatAWholeSortPicks(int k, int cols)
    {
        var random = new Random(k * 7919 + cols);
        const int rows = 5;
        var v = new float[rows * cols];
        for (int i = 0; i < v.Length; i++)
            // Coarse values give ties; a few NaNs and infinities sit among them.
            v[i] = random.Next(64) switch
            {
                0 => float.NaN,
                1 => float.NegativeInfinity,
                2 => float.PositiveInfinity,
                _ => random.Next(40) / 4f,
            };
        // One row of equal values, one with its largest value last.
        for (int c = 0; c < cols; c++) v[cols + c] = 1f;
        v[3 * cols + cols - 1] = 1000f;
        var picked = new int[rows * k];
        TensorGraphExecutor.TopKRows(v, rows, cols, k, picked);
        Assert.Equal(Sorted(v, rows, cols, k), picked);
    }
}
