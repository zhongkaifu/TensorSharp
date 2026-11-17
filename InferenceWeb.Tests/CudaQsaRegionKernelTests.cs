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
/// The CUDA executor's kernels for a QSA region (GraphFusion.QsaAttention) against the ops they replace: the block scores
/// against the gathered pooled keys' lightning indexer (bit for bit), the cell lists against the top-k, top-k mask and mask
/// listing over the step's plan (exactly: ties, rows that see no more blocks than they pick, every tail length), the prompt
/// attention over the listed cells against a double reference, and the flash kernel over listed cells without a mask against
/// the same cells under a mask of zeros (bit for bit: the decode rows' path).
/// </summary>
public sealed class CudaQsaRegionKernelTests
{
    private readonly ITestOutputHelper _output;

    public CudaQsaRegionKernelTests(ITestOutputHelper output) => _output = output;

    private static IntPtr P(Tensor t) => ((CudaStorage)t.Storage).DevicePtrAtElement(t.StorageOffset);

    private static void Upload(params Tensor[] tensors)
    {
        foreach (Tensor t in tensors) ((CudaStorage)t.Storage).EnsureDeviceCurrent();
    }

    private static void Written(params Tensor[] tensors)
    {
        foreach (Tensor t in tensors) ((CudaStorage)t.Storage).MarkDeviceModified();
    }

    /// <summary>The graph kernels with the region's (sm_80) kernels, or null (logged) where this image or GPU has none.</summary>
    private CudaGraphKernels RegionKernels(CudaAllocator allocator)
    {
        allocator.Context.MakeCurrent();
        Assert.True(CudaGraphKernels.TryCreate(out CudaGraphKernels kernels, out string reason), reason);
        if (kernels.TakesQsaAttention) return kernels;
        kernels.Dispose();
        _output.WriteLine("no QSA region kernels for this GPU");
        return null;
    }

    /// <summary>The rows' KV indices: start .. start + rows - 1.</summary>
    private static int[] Positions(int start, int rows) => Enumerable.Range(start, rows).ToArray();

    [CudaTheory]
    [InlineData(128, 2, 1, 256, 4, 200)]     // a decode row (the synthetic fixture's indexer)
    [InlineData(128, 4, 7, 512, 4, 300)]     // a verify window's rows
    [InlineData(128, 4, 40, 301, 4, 0)]      // a prompt from the first cell, a padded last block
    [InlineData(64, 8, 33, 1030, 8, 900)]    // narrower keys than the lanes, a wider ratio
    public void ScoresMatchTheLightningIndexerOverGatheredKeys(int d, int heads, int rows, int window, int ratio, int start)
    {
        using var allocator = new CudaAllocator();
        using CudaGraphKernels kernels = RegionKernels(allocator);
        if (kernels == null) return;
        int cap = window + 9, pools = (window + ratio - 1) / ratio, complete = (start + rows) / ratio;
        var rng = new Random(d + rows);
        // An indexer cache of [raw | pooled] rows holding values an F16 cache holds exactly.
        float[] cache = Enumerable.Range(0, cap * 2 * d).Select(_ => (float)(System.Half)(rng.NextDouble() * 2 - 1)).ToArray();
        float[] q = Enumerable.Range(0, rows * heads * d).Select(_ => (float)(rng.NextDouble() * 2 - 1)).ToArray();
        int[] pos = Positions(start, rows);
        // The plan's inputs: each block's pooled row (padded blocks read row 0), its visibility, the head weights.
        int[] pooledRows = Enumerable.Range(0, pools).Select(b => b < complete ? b * ratio + ratio - 1 : 0).ToArray();
        var visible = new float[rows * pools];
        for (int r = 0; r < rows; r++)
            for (int b = 0; b < pools; b++)
                visible[r * pools + b] = b < complete && (b + 1) * ratio - 1 <= pos[r] ? 0f : float.NegativeInfinity;
        float weight = 1f / MathF.Sqrt(d);
        float[] w = Enumerable.Repeat(weight, rows * heads).ToArray();

        using var c32 = Tensor.FromArray(allocator, cache);
        using var c16 = new Tensor(allocator, DType.Float16, cache.Length);
        using var qt = Tensor.FromArray(allocator, q);
        using var pt = Tensor.FromArray(allocator, pos);
        using var ids = Tensor.FromArray(allocator, pooledRows);
        using var vt = Tensor.FromArray(allocator, visible);
        using var wt = Tensor.FromArray(allocator, w);
        using var gathered = new Tensor(allocator, DType.Float32, pools * d);
        using var byOps = new Tensor(allocator, DType.Float32, rows * pools);
        using var byRegion = new Tensor(allocator, DType.Float32, rows * pools);
        Upload(c32, c16, qt, pt, ids, vt, wt, gathered, byOps, byRegion);
        IntPtr stream = allocator.Stream.Handle;
        allocator.Context.MakeCurrent();
        kernels.LaunchConvert(P(c32), P(c16), cache.Length, true, stream);
        // The ops: the pooled half gathered (rows 2d apart, at offset d), then the indexer under the visibility mask.
        kernels.LaunchGetRows(((CudaStorage)c16.Storage).DevicePtrAtElement(c16.StorageOffset + d), P(ids), P(gathered), d, pools, cap, 1,
            2L * d, true, stream);
        kernels.LaunchLightningIndexer(P(qt), P(gathered), P(wt), P(vt), P(byOps), d, heads, pools, rows, rows, stream);
        kernels.LaunchQsaScores(P(qt), P(c16), 2L * d, P(pt), P(byRegion), weight, d, heads, rows, pools, ratio, stream);
        allocator.Stream.Synchronize();
        Written(byOps, byRegion);
        float[] want = byOps.GetElementsAsFloat(rows * pools), got = byRegion.GetElementsAsFloat(rows * pools);
        for (int i = 0; i < want.Length; i++)
            if (BitConverter.SingleToInt32Bits(want[i]) != BitConverter.SingleToInt32Bits(got[i]))
                Assert.Fail($"row {i / pools} block {i % pools}: {got[i]:R} from the region's kernel, {want[i]:R} from the ops");
        Assert.Contains(want, float.IsNegativeInfinity);
    }

    /// <summary>Scores as the region's selection sees them: -inf past a row's visible blocks, the rest drawn as
    /// <paramref name="mode"/> says - 0 distinct values, 1 coarse values (many ties, many zeros: rectified sums), 2 one value.</summary>
    private static float[] Scores(Random rng, int[] pos, int pools, int ratio, int mode)
    {
        var scores = new float[pos.Length * pools];
        for (int r = 0; r < pos.Length; r++)
            for (int b = 0; b < pools; b++)
            {
                float v = mode switch
                {
                    0 => (float)rng.NextDouble() * 8,
                    1 => MathF.Max(0f, MathF.Round((float)rng.NextDouble() * 6 - 2)),
                    _ => 0.75f,
                };
                scores[r * pools + b] = b < (pos[r] + 1) / ratio ? v : float.NegativeInfinity;
            }
        return scores;
    }

    [CudaTheory]
    [InlineData(14, 0, 256, 4, 2, 1)]        // the synthetic fixture: the first rows see no more blocks than they pick
    [InlineData(40, 0, 256, 4, 2, 2)]        // a prompt from the first cell, every score tied
    [InlineData(8, 3000, 3072, 4, 512, 1)]   // the real width: 512 picks of ~750 blocks (the wide top-k), ties
    [InlineData(5, 700, 1024, 8, 40, 0)]     // a wider ratio, the wide top-k past 32 picks
    [InlineData(3, 50, 256, 4, 64, 1)]       // fewer visible blocks than picks on every row: dense
    [InlineData(9, 1500, 2048, 4, 30, 2)]    // one value everywhere: the lowest blocks
    public void SelectionListsTheCellsTheMaskedSelectionKeeps(int rows, int start, int window, int ratio, int picks, int mode)
    {
        using var allocator = new CudaAllocator();
        using CudaGraphKernels kernels = RegionKernels(allocator);
        if (kernels == null) return;
        int pools = (window + ratio - 1) / ratio, complete = (start + rows) / ratio, stride = CudaGraphKernels.QsaCellStride(picks, ratio);
        Assert.True(picks <= pools);
        var rng = new Random(rows * 31 + mode);
        int[] pos = Positions(start, rows);
        float[] scores = Scores(rng, pos, pools, ratio, mode);
        // The plan the ops read (QsaPlanFor): a block's cells (padded blocks' past the keys), each row's tail, the causal mask.
        int[] blocks = Enumerable.Range(0, pools * ratio).Select(i => i / ratio < complete ? i : window).ToArray();
        var tail = new int[rows * (ratio - 1)];
        var mask = new float[rows * window];
        for (int r = 0; r < rows; r++)
        {
            int p = pos[r], tailStart = (p + 1) / ratio * ratio;
            for (int t = 0; t < ratio - 1; t++) tail[r * (ratio - 1) + t] = tailStart + t <= p ? tailStart + t : window;
            for (int c = 0; c < window; c++) mask[r * window + c] = c <= p ? 0f : float.NegativeInfinity;
        }

        using var st = Tensor.FromArray(allocator, scores);
        using var pt = Tensor.FromArray(allocator, pos);
        using var bt = Tensor.FromArray(allocator, blocks);
        using var tt = Tensor.FromArray(allocator, tail);
        using var mt = Tensor.FromArray(allocator, mask);
        using var picked = new Tensor(allocator, DType.Int32, rows * picks);
        using var masked = new Tensor(allocator, DType.Float32, rows * window);
        using var opCells = new Tensor(allocator, DType.Int32, rows * window);
        using var opCount = new Tensor(allocator, DType.Int32, rows);
        using var cells = new Tensor(allocator, DType.Int32, rows * stride);
        using var count = new Tensor(allocator, DType.Int32, rows);
        Upload(st, pt, bt, tt, mt, picked, masked, opCells, opCount, cells, count);
        IntPtr stream = allocator.Stream.Handle;
        allocator.Context.MakeCurrent();
        // The ops as the executor runs them: the device top-k (the wide kernel for many picks or wide rows), the top-k mask,
        // the mask's cells.
        if (picks > 32 || pools > CudaGraphKernels.TopKMaxColumns) kernels.LaunchTopKWide(P(st), P(picked), rows, pools, picks, stream);
        else kernels.LaunchTopK(P(st), P(picked), rows, pools, picks, stream);
        kernels.LaunchTopKMask(P(mt), P(picked), P(st), P(bt), P(tt), P(masked), rows, window, rows, picks, pools, ratio, pools, ratio - 1, stream);
        kernels.LaunchMaskCells(P(masked), P(opCells), P(opCount), rows, window, stream);
        kernels.LaunchQsaSelect(P(st), P(pt), P(cells), P(count), rows, pools, ratio, picks, stride, stream);
        allocator.Stream.Synchronize();
        Written(opCells, opCount, cells, count);
        int[] wantCount = opCount.GetElementsAsInt(rows), gotCount = count.GetElementsAsInt(rows);
        int[] wantCells = opCells.GetElementsAsInt(rows * window), gotCells = cells.GetElementsAsInt(rows * stride);
        for (int r = 0; r < rows; r++)
        {
            Assert.True(wantCount[r] == gotCount[r], $"row {r} (position {pos[r]}): {gotCount[r]} cells listed, the ops keep {wantCount[r]}");
            Assert.True(wantCount[r] <= stride, $"row {r}: the ops keep {wantCount[r]} cells, past the list's {stride}");
            Assert.Equal(wantCells.AsSpan(r * window, wantCount[r]).ToArray(), gotCells.AsSpan(r * stride, gotCount[r]).ToArray());
            int visible = (pos[r] + 1) / ratio;
            int expected = visible <= picks ? pos[r] + 1 : picks * ratio + pos[r] + 1 - visible * ratio;
            Assert.Equal(expected, gotCount[r]);
        }
    }

    [CudaTheory]
    [InlineData(9, 4, 2, 600, 11)]       // the synthetic fixture's heads: a short prompt chunk
    [InlineData(20, 16, 1, 3000, 2051)]  // a full tile of query heads, the real budget's cells
    [InlineData(12, 12, 1, 2100, 700)]   // twelve query heads of one KV head
    [InlineData(10, 8, 4, 400, 1)]       // rows of a single cell
    public void PromptAttentionOverListedCellsMatchesTheReference(int rows, int heads, int kvHeads, int cap, int maxCells)
    {
        const int d = CudaGraphKernels.QsaAttentionHeadDim;
        using var allocator = new CudaAllocator();
        using CudaGraphKernels kernels = RegionKernels(allocator);
        if (kernels == null) return;
        int group = heads / kvHeads, stride = maxCells;
        var rng = new Random(rows + heads);
        float[] qv = Enumerable.Range(0, rows * heads * d).Select(_ => (float)(rng.NextDouble() * 2 - 1)).ToArray();
        float[] kh = Enumerable.Range(0, kvHeads * cap * d).Select(_ => (float)(System.Half)(rng.NextDouble() * 2 - 1)).ToArray();
        float[] vh = Enumerable.Range(0, kvHeads * cap * d).Select(_ => (float)(System.Half)(rng.NextDouble() * 2 - 1)).ToArray();
        // Each row's cells: an ascending random subset of the cache, of 1 .. maxCells cells (the last row the most).
        var lists = new int[rows * stride];
        var counts = new int[rows];
        for (int t = 0; t < rows; t++)
        {
            int n = t == rows - 1 ? maxCells : 1 + rng.Next(maxCells);
            int[] chosen = Enumerable.Range(0, cap).OrderBy(_ => rng.Next()).Take(n).OrderBy(c => c).ToArray();
            counts[t] = n;
            chosen.CopyTo(lists, t * stride);
        }
        float scale = 1f / MathF.Sqrt(d);
        using var q = Tensor.FromArray(allocator, qv);
        using var k32 = Tensor.FromArray(allocator, kh);
        using var v32 = Tensor.FromArray(allocator, vh);
        using var k = new Tensor(allocator, DType.Float16, kh.Length);
        using var v = new Tensor(allocator, DType.Float16, vh.Length);
        using var ct = Tensor.FromArray(allocator, lists);
        using var nt = Tensor.FromArray(allocator, counts);
        using var output = new Tensor(allocator, DType.Float32, rows * heads * d);
        Upload(q, k32, v32, k, v, ct, nt, output);
        IntPtr stream = allocator.Stream.Handle;
        allocator.Context.MakeCurrent();
        kernels.LaunchConvert(P(k32), P(k), kh.Length, true, stream);
        kernels.LaunchConvert(P(v32), P(v), vh.Length, true, stream);
        kernels.LaunchQsaAttention(P(q), P(k), P(v), P(ct), P(nt), P(output), cap, rows, heads, kvHeads, scale, stride, stream);
        allocator.Stream.Synchronize();
        Written(output);
        float[] got = output.GetElementsAsFloat(rows * heads * d);

        double num = 0, den = 0;
        for (int t = 0; t < rows; t++)
            for (int h = 0; h < heads; h++)
            {
                int kvh = h / group, n = counts[t];
                var w = new double[n];
                double mx = double.NegativeInfinity;
                for (int j = 0; j < n; j++)
                {
                    int c = lists[t * stride + j];
                    double dot = 0;
                    for (int x = 0; x < d; x++) dot += qv[(t * heads + h) * d + x] * (double)kh[(kvh * cap + c) * d + x];
                    w[j] = dot * scale;
                    mx = Math.Max(mx, w[j]);
                }
                double sum = 0;
                for (int j = 0; j < n; j++) { w[j] = Math.Exp(w[j] - mx); sum += w[j]; }
                for (int x = 0; x < d; x++)
                {
                    double want = 0;
                    for (int j = 0; j < n; j++) want += w[j] * (double)vh[(kvh * cap + lists[t * stride + j]) * d + x];
                    want /= sum;
                    double diff = got[(t * heads + h) * d + x] - want;
                    num += diff * diff;
                    den += want * want;
                }
            }
        double rel = Math.Sqrt(num / den);
        _output.WriteLine($"{rows} rows, {heads}/{kvHeads} heads, up to {maxCells} cells of {cap}: relative L2 {rel:E2}");
        Assert.True(rel <= 5e-3, $"relative L2 {rel:E2}");
    }

    /// <summary>The flash kernel over listed cells without a mask (the region's decode rows: every kept cell's mask value
    /// was 0) gives what it gives under a mask of zeros, bit for bit, for a decode row and a verify window's rows.</summary>
    [CudaTheory]
    [InlineData(1, 16, 2, 3000, 7)]
    [InlineData(4, 4, 2, 600, 2)]
    public void ListedCellsWithoutAMaskMatchAMaskOfZeros(int rows, int heads, int kvHeads, int window, int every)
    {
        const int d = 256;
        using var allocator = new CudaAllocator();
        using CudaGraphKernels kernels = RegionKernels(allocator);
        CudaFlashPrefillKernels flash = allocator.FlashPrefillKernels;
        if (kernels == null || flash == null || !flash.RunsAttention(d)) return;
        int cap = window + 5;
        var rng = new Random(rows * 7 + every);
        float[] qv = Enumerable.Range(0, rows * heads * d).Select(_ => (float)(rng.NextDouble() * 2 - 1)).ToArray();
        float[] kh = Enumerable.Range(0, kvHeads * cap * d).Select(_ => (float)(System.Half)(rng.NextDouble() * 2 - 1)).ToArray();
        float[] vh = Enumerable.Range(0, kvHeads * cap * d).Select(_ => (float)(System.Half)(rng.NextDouble() * 2 - 1)).ToArray();
        // Row z keeps every few cells in runs (as a selection's blocks), its mask 0 there and -inf elsewhere.
        var mask = new float[rows * window];
        int kept = int.MaxValue;
        for (int z = 0; z < rows; z++)
        {
            int n = 0;
            for (int c = 0; c < window; c++)
            {
                bool keep = ((c + z) / 4) % every == 0;
                mask[z * window + c] = keep ? 0f : float.NegativeInfinity;
                n += keep ? 1 : 0;
            }
            kept = Math.Min(kept, n);
        }
        float scale = 1f / MathF.Sqrt(d);
        int splits = flash.Splits(heads / kvHeads, kvHeads, Math.Min(kept, window));
        using var q = Tensor.FromArray(allocator, qv);
        using var k32 = Tensor.FromArray(allocator, kh);
        using var v32 = Tensor.FromArray(allocator, vh);
        using var k = new Tensor(allocator, DType.Float16, kh.Length);
        using var v = new Tensor(allocator, DType.Float16, vh.Length);
        using var mt = Tensor.FromArray(allocator, mask);
        using var cells = new Tensor(allocator, DType.Int32, rows * window);
        using var count = new Tensor(allocator, DType.Int32, rows);
        using var withMask = new Tensor(allocator, DType.Float32, rows * heads * d);
        using var without = new Tensor(allocator, DType.Float32, rows * heads * d);
        using Tensor partial = splits > 1 ? new Tensor(allocator, DType.Float32, CudaFlashPrefillKernels.PartialFloats(rows, heads, d, splits)) : null;
        Upload(q, k32, v32, k, v, mt, cells, count, withMask, without);
        if (partial != null) Upload(partial);
        IntPtr stream = allocator.Stream.Handle, part = partial == null ? IntPtr.Zero : P(partial);
        allocator.Context.MakeCurrent();
        kernels.LaunchConvert(P(k32), P(k), kh.Length, true, stream);
        kernels.LaunchConvert(P(v32), P(v), vh.Length, true, stream);
        kernels.LaunchMaskCells(P(mt), P(cells), P(count), rows, window, stream);
        foreach ((Tensor output, IntPtr cellMask) in new[] { (withMask, P(mt)), (without, IntPtr.Zero) })
        {
            if (rows == 1)
                flash.LaunchAttention(P(q), P(k), P(v), cap, 1, window, 0, heads, kvHeads, d, scale, splits, part, P(output), stream, IntPtr.Zero,
                    P(cells), P(count), cellMask);
            else
                flash.LaunchDecodeRows(P(q), P(k), P(v), cap, rows, window, 0, heads, kvHeads, d, scale, splits, part, P(output), stream,
                    IntPtr.Zero, P(cells), P(count), cellMask, window);
        }
        allocator.Stream.Synchronize();
        Written(withMask, without);
        float[] a = withMask.GetElementsAsFloat(rows * heads * d), b = without.GetElementsAsFloat(rows * heads * d);
        Assert.Equal(a.Select(BitConverter.SingleToInt32Bits), b.Select(BitConverter.SingleToInt32Bits));
    }
}
