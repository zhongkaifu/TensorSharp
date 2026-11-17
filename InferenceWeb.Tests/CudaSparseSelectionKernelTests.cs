// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Linq;
using TensorSharp;
using TensorSharp.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// The graph executor's kernels for a sparse attention's selection on a recorded step (a QSA layer past its budget): the
/// wide top-k (radix select, then sorted) against the host sort with its ties, the lightning indexer against a double
/// reference, ggml_top_k_mask's cells, and the row gather/scatter of F16 cache rows read in place.
/// </summary>
public sealed class CudaSparseSelectionKernelTests
{
    private readonly ITestOutputHelper _output;

    public CudaSparseSelectionKernelTests(ITestOutputHelper output) => _output = output;

    private static IntPtr P(Tensor t) => ((CudaStorage)t.Storage).DevicePtrAtElement(t.StorageOffset);

    private static void Upload(params Tensor[] tensors)
    {
        foreach (Tensor t in tensors) ((CudaStorage)t.Storage).EnsureDeviceCurrent();
    }

    private static void Written(params Tensor[] tensors)
    {
        foreach (Tensor t in tensors) ((CudaStorage)t.Storage).MarkDeviceModified();
    }

    private static CudaGraphKernels Kernels(CudaAllocator allocator)
    {
        allocator.Context.MakeCurrent();
        Assert.True(CudaGraphKernels.TryCreate(out CudaGraphKernels kernels, out string reason), reason);
        return kernels;
    }

    /// <summary>Largest first, a tie to the lower index (the host executor's sort).</summary>
    private static int[] HostTopK(float[] v, int row, int cols, int k)
        => Enumerable.Range(0, cols).OrderByDescending(c => v[row * cols + c]).ThenBy(c => c).Take(k).ToArray();

    [CudaTheory]
    [InlineData(3, 3000, 512, false)]   // QSA blocks: many picks over a few thousand
    [InlineData(2, 9000, 37, true)]     // wider than the narrow kernel, ties and -inf
    [InlineData(1, 600, 600, true)]     // every column
    [InlineData(4, 512, 8, true)]       // a router's width: the narrow kernel
    public void TopKMatchesTheHostSort(int rows, int cols, int k, bool ties)
    {
        using var allocator = new CudaAllocator();
        using CudaGraphKernels kernels = Kernels(allocator);
        var rng = new Random(5);
        var x = new float[rows * cols];
        for (int i = 0; i < x.Length; i++)
        {
            float v = (float)(rng.NextDouble() * 4 - 2);
            if (ties) v = MathF.Round(v * 8) / 8;                // many equal values
            if (ties && i % 7 == 0) v = float.NegativeInfinity;   // masked blocks
            x[i] = v;
        }
        using var xt = Tensor.FromArray(allocator, x);
        using var ids = new Tensor(allocator, DType.Int32, rows * k);
        Upload(xt, ids);
        allocator.Context.MakeCurrent();
        if (k > 32 || cols > CudaGraphKernels.TopKMaxColumns)
            kernels.LaunchTopKWide(P(xt), P(ids), rows, cols, k, allocator.Stream.Handle);
        else
            kernels.LaunchTopK(P(xt), P(ids), rows, cols, k, allocator.Stream.Handle);
        allocator.Stream.Synchronize();
        Written(ids);
        int[] got = ids.GetElementsAsInt(rows * k);
        for (int r = 0; r < rows; r++)
            Assert.Equal(HostTopK(x, r, cols, k), got.AsSpan(r * k, k).ToArray());
    }

    /// <summary>The two-stage pick (a greedy verify's token a row) against the wide top-k at k = 1, which it replaces for
    /// wide rows: the same index for every row, ties (the lower index), -0 against +0, a maximum in the last slice.</summary>
    [CudaTheory]
    [InlineData(4, 248320, false)]   // Qwen3.8's vocabulary
    [InlineData(3, 151643, true)]
    [InlineData(2, 4097, true)]      // one column past a slice
    [InlineData(5, 9000, true)]
    public void ArgmaxMatchesTheWideTopK(int rows, int cols, bool ties)
    {
        using var allocator = new CudaAllocator();
        using CudaGraphKernels kernels = Kernels(allocator);
        var rng = new Random(13);
        var x = new float[rows * cols];
        for (int i = 0; i < x.Length; i++)
        {
            float v = (float)(rng.NextDouble() * 8 - 4);
            if (ties) v = MathF.Round(v * 2) / 2;
            x[i] = v;
        }
        // Row 0's maximum twice (the lower index wins), row 1's in the last column, row 2 all zeros of both signs.
        x[17] = 100f; x[cols - 5] = 100f;
        if (rows > 1) x[2L * cols - 1 > int.MaxValue ? 0 : 2 * cols - 1] = 1000f;
        if (rows > 2) for (int c = 0; c < cols; c++) x[2 * cols + c] = c % 3 == 0 ? -0f : 0f;
        using var xt = Tensor.FromArray(allocator, x);
        using var wide = new Tensor(allocator, DType.Int32, rows);
        using var two = new Tensor(allocator, DType.Int32, rows);
        using var part = new Tensor(allocator, DType.Float32, CudaGraphKernels.ArgmaxPartBytes(rows, cols) / 4);
        Upload(xt, wide, two, part);
        allocator.Context.MakeCurrent();
        kernels.LaunchTopKWide(P(xt), P(wide), rows, cols, 1, allocator.Stream.Handle);
        kernels.LaunchArgmax(P(xt), P(two), P(part), rows, cols, allocator.Stream.Handle);
        allocator.Stream.Synchronize();
        Written(wide, two);
        int[] a = wide.GetElementsAsInt(rows), b = two.GetElementsAsInt(rows);
        Assert.Equal(a, b);
        Assert.Equal(17, b[0]);
        if (rows > 1) Assert.Equal(cols - 1, b[1]);
        if (rows > 2) Assert.Equal(0, b[2]);
        for (int r = 3; r < rows; r++) Assert.Equal(HostTopK(x, r, cols, 1)[0], b[r]);
    }

    /// <summary>The two-stage few picks (a draft step's ten over the draft vocabulary) against the wide top-k they replace for
    /// wide rows: the same indices in the same order for every row - ties to the lower index, -0 against +0, picks crowding
    /// one slice, a row whose picks straddle the last slice.</summary>
    [CudaTheory]
    [InlineData(1, 151643, 10, true)]   // Qwen3.8's draft vocabulary
    [InlineData(4, 248320, 16, false)]
    [InlineData(3, 4097, 2, true)]       // one column past a slice
    [InlineData(5, 9000, 7, true)]
    public void FewPicksMatchTheWideTopK(int rows, int cols, int k, bool ties)
    {
        using var allocator = new CudaAllocator();
        using CudaGraphKernels kernels = Kernels(allocator);
        var rng = new Random(29);
        var x = new float[rows * cols];
        for (int i = 0; i < x.Length; i++)
        {
            float v = (float)(rng.NextDouble() * 8 - 4);
            if (ties) v = MathF.Round(v * 2) / 2;
            x[i] = v;
        }
        // Row 0: its largest values all in one slice, one of them twice; the last row's in the last columns.
        for (int j = 0; j < k; j++) x[100 + j] = 50f + j % 3;
        x[cols - 5] = 52f;
        int last = rows - 1;
        if (rows > 1) for (int j = 0; j < k; j++) x[(long)last * cols + cols - 1 - j] = 70f - j;
        if (rows > 2) for (int c = 0; c < cols; c++) x[cols + c] = c % 3 == 0 ? -0f : 0f;
        using var xt = Tensor.FromArray(allocator, x);
        using var wide = new Tensor(allocator, DType.Int32, rows * k);
        using var few = new Tensor(allocator, DType.Int32, rows * k);
        using var part = new Tensor(allocator, DType.Float32, CudaGraphKernels.TopKFewPartBytes(rows, cols, k) / 4);
        Upload(xt, wide, few, part);
        allocator.Context.MakeCurrent();
        kernels.LaunchTopKWide(P(xt), P(wide), rows, cols, k, allocator.Stream.Handle);
        kernels.LaunchTopKFew(P(xt), P(few), P(part), rows, cols, k, allocator.Stream.Handle);
        allocator.Stream.Synchronize();
        Written(wide, few);
        int[] a = wide.GetElementsAsInt(rows * k), b = few.GetElementsAsInt(rows * k);
        Assert.Equal(a, b);
        for (int r = 0; r < rows; r++) Assert.Equal(HostTopK(x, r, cols, k), b.AsSpan(r * k, k).ToArray());
    }

    [CudaTheory]
    [InlineData(128, 4, 1, 700, 1)]
    [InlineData(64, 8, 3, 257, 3)]
    [InlineData(128, 32, 2, 1100, 1)]   // one mask row for every query row
    public void LightningIndexerMatchesTheReference(int d, int heads, int rows, int n, int maskRows)
    {
        using var allocator = new CudaAllocator();
        using CudaGraphKernels kernels = Kernels(allocator);
        var rng = new Random(9);
        float[] R(int count) => Enumerable.Range(0, count).Select(_ => (float)(rng.NextDouble() * 2 - 1)).ToArray();
        float[] q = R(rows * heads * d), keys = R(n * d), w = R(rows * heads);
        float[] mask = Enumerable.Range(0, maskRows * n).Select(i => i % 5 == 0 ? float.NegativeInfinity : 0f).ToArray();
        using var qt = Tensor.FromArray(allocator, q);
        using var kt = Tensor.FromArray(allocator, keys);
        using var wt = Tensor.FromArray(allocator, w);
        using var mt = Tensor.FromArray(allocator, mask);
        using var output = new Tensor(allocator, DType.Float32, rows * n);
        Upload(qt, kt, wt, mt, output);
        allocator.Context.MakeCurrent();
        kernels.LaunchLightningIndexer(P(qt), P(kt), P(wt), P(mt), P(output), d, heads, n, rows, maskRows, allocator.Stream.Handle);
        allocator.Stream.Synchronize();
        Written(output);
        float[] got = output.GetElementsAsFloat(rows * n);
        double worst = 0;
        for (int r = 0; r < rows; r++)
            for (int j = 0; j < n; j++)
            {
                double sum = 0;
                for (int h = 0; h < heads; h++)
                {
                    double dot = 0;
                    for (int i = 0; i < d; i++) dot += q[(r * heads + h) * d + i] * (double)keys[j * d + i];
                    if (dot > 0) sum += w[r * heads + h] * dot;
                }
                float m = mask[(r % maskRows) * n + j];
                float g = got[r * n + j];
                if (float.IsNegativeInfinity(m)) { Assert.True(float.IsNegativeInfinity(g), $"row {r} key {j}: {g}, not -inf"); continue; }
                worst = Math.Max(worst, Math.Abs(g - sum) / (1 + Math.Abs(sum)));
            }
        _output.WriteLine($"worst relative error {worst:E2}");
        Assert.True(worst < 1e-5, $"worst relative error {worst:E2}");
    }

    [CudaTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void TopKMaskKeepsThePicksCells(bool blocks)
    {
        using var allocator = new CudaAllocator();
        using CudaGraphKernels kernels = Kernels(allocator);
        const int rows = 2, ratio = 4, nPick = 50, top = 9, nExtra = 3;
        int nkv = blocks ? nPick * ratio + 5 : nPick + 5, cells = blocks ? ratio : 1;
        var rng = new Random(13);
        float[] mask = Enumerable.Range(0, nkv).Select(_ => (float)rng.NextDouble()).ToArray();   // one row for both
        int[] picks = Enumerable.Range(0, rows * top).Select(i => i % top == 0 ? -1 : rng.Next(nPick + 3)).ToArray();   // some out of range
        float[] score = Enumerable.Range(0, rows * nPick).Select(i => i % 6 == 0 ? float.NegativeInfinity : 1f).ToArray();
        int[] table = Enumerable.Range(0, nPick * ratio).Select(i => i % 11 == 0 ? nkv + 2 : i).ToArray();   // some cells past the keys
        int[] extra = Enumerable.Range(0, rows * nExtra).Select(i => i == 1 ? -3 : nkv - 1 - i).ToArray();
        using var mt = Tensor.FromArray(allocator, mask);
        using var pt = Tensor.FromArray(allocator, picks);
        using var st = Tensor.FromArray(allocator, score);
        using var bt = Tensor.FromArray(allocator, table);
        using var et = Tensor.FromArray(allocator, extra);
        using var output = new Tensor(allocator, DType.Float32, rows * nkv);
        Upload(mt, pt, st, bt, et, output);
        allocator.Context.MakeCurrent();
        kernels.LaunchTopKMask(P(mt), P(pt), P(st), blocks ? P(bt) : IntPtr.Zero, P(et), P(output), rows, nkv, 1, top, nPick, cells,
            blocks ? nPick : 0, nExtra, allocator.Stream.Handle);
        allocator.Stream.Synchronize();
        Written(output);
        float[] got = output.GetElementsAsFloat(rows * nkv);
        var want = Enumerable.Repeat(float.NegativeInfinity, rows * nkv).ToArray();
        for (int r = 0; r < rows; r++)
        {
            void Keep(int c) { if (c >= 0 && c < nkv) want[r * nkv + c] = mask[c]; }
            for (int t = 0; t < top; t++)
            {
                int pick = picks[r * top + t];
                if (pick < 0 || pick >= nPick || float.IsNegativeInfinity(score[r * nPick + pick])) continue;
                if (!blocks) { Keep(pick); continue; }
                for (int c = 0; c < ratio; c++) Keep(table[pick * ratio + c]);
            }
            for (int e = 0; e < nExtra; e++) Keep(extra[r * nExtra + e]);
        }
        Assert.Equal(want, got);
    }

    [CudaTheory]
    [InlineData(true)]
    [InlineData(false)]
    public void RowsGatherAndScatterInPlace(bool half)
    {
        using var allocator = new CudaAllocator();
        using CudaGraphKernels kernels = Kernels(allocator);
        const int d = 24, pitch = 2 * d, capacity = 40, n = 5;
        var rng = new Random(17);
        // A cache of [capacity, 2d] rows (values an F16 cache holds exactly); the gather reads the second half in place.
        float[] cache = Enumerable.Range(0, capacity * pitch).Select(_ => (float)(System.Half)(rng.NextDouble() * 2 - 1)).ToArray();
        int[] rows = { 3, 39, 0, 17, 3 };
        using var c32 = Tensor.FromArray(allocator, cache);
        using var store = half ? new Tensor(allocator, DType.Float16, capacity * pitch) : null;
        Tensor src = store ?? c32;
        using var ids = Tensor.FromArray(allocator, rows);
        using var gathered = new Tensor(allocator, DType.Float32, n * d);
        Upload(c32, ids, gathered);
        if (store != null)
        {
            Upload(store);
            allocator.Context.MakeCurrent();
            kernels.LaunchConvert(P(c32), P(store), cache.Length, true, allocator.Stream.Handle);
        }
        IntPtr secondHalf = ((CudaStorage)src.Storage).DevicePtrAtElement(src.StorageOffset + d);
        allocator.Context.MakeCurrent();
        kernels.LaunchGetRows(secondHalf, P(ids), P(gathered), d, n, capacity, 1, pitch, half, allocator.Stream.Handle);
        allocator.Stream.Synchronize();
        Written(gathered);
        float[] got = gathered.GetElementsAsFloat(n * d);
        for (int i = 0; i < n; i++)
            for (int c = 0; c < d; c++)
                Assert.Equal(cache[rows[i] * pitch + d + c], got[i * d + c]);

        // Scatter rows of F32 into the first half of rows 7 and 2 (an index past the rows writes nothing).
        float[] fresh = Enumerable.Range(0, 3 * d).Select(_ => (float)(System.Half)(rng.NextDouble() * 2 - 1)).ToArray();
        using var ft = Tensor.FromArray(allocator, fresh);
        using var to = Tensor.FromArray(allocator, new[] { 7, 2, capacity });
        Upload(ft, to);
        allocator.Context.MakeCurrent();
        kernels.LaunchSetRows(P(ft), P(to), P(src), 3, d, pitch, capacity, half, allocator.Stream.Handle);
        if (store != null) kernels.LaunchConvert(P(store), P(c32), cache.Length, false, allocator.Stream.Handle);
        allocator.Stream.Synchronize();
        Written(c32);
        float[] after = c32.GetElementsAsFloat(cache.Length);
        float[] want = (float[])cache.Clone();
        for (int c = 0; c < d; c++) { want[7 * pitch + c] = fresh[c]; want[2 * pitch + c] = fresh[d + c]; }
        Assert.Equal(want, after);
    }
}
