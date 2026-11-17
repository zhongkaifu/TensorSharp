// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using TensorSharp;
using TensorSharp.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// The tensor-core flash prefill attention (ts_flash_prefill_f16_d128 / _d256) against a double-precision host
/// reference over the same F16 cache: causal rows that are the last nq of the window, GQA groups folded into the row
/// tiles, keys below a floor masked, and split-K partials for a few rows over a long window. F16 operands and short
/// F16 accumulation runs keep it within a few 1e-3.
/// </summary>
public sealed class CudaFlashPrefillAttentionTests
{
    private readonly ITestOutputHelper _output;

    public CudaFlashPrefillAttentionTests(ITestOutputHelper output) => _output = output;

    [CudaTheory]
    [InlineData(256, 24, 4, 130, 130, 0)]   // a prompt chunk: rows past one 128-row tile, no history
    [InlineData(256, 24, 4, 17, 300, 0)]    // a chunk after a history
    [InlineData(256, 8, 1, 3, 2100, 0)]     // a verify window over a long history: split over the keys
    [InlineData(128, 16, 2, 40, 90, 33)]    // 128-wide heads, keys below a floor masked
    [InlineData(256, 24, 4, 1, 3700, 0)]    // a decode row over a long window
    public void RowsMatchTheReference(int d, int heads, int kvHeads, int nq, int len, int floor)
    {
        using var allocator = new CudaAllocator();
        CudaFlashPrefillKernels flash = allocator.FlashPrefillKernels;
        if (flash == null || !flash.RunsAttention(d))
        {
            _output.WriteLine("no flash prefill attention for this GPU");
            return;
        }
        int cap = len + 37, group = heads / kvHeads;
        var rng = new Random(3);
        float[] qv = new float[heads * nq * d];
        for (int i = 0; i < qv.Length; i++) qv[i] = (float)(rng.NextDouble() * 2 - 1);
        // Values an F16 cache holds exactly (rounded on the host, converted on the device without loss).
        var kh = new float[kvHeads * cap * d];
        var vh = new float[kvHeads * cap * d];
        for (int i = 0; i < kh.Length; i++)
        {
            kh[i] = (float)(System.Half)(rng.NextDouble() * 2 - 1);
            vh[i] = (float)(System.Half)(rng.NextDouble() * 2 - 1);
        }
        float scale = 1f / MathF.Sqrt(d);

        using var q = Tensor.FromArray(allocator, qv);
        using var k32 = Tensor.FromArray(allocator, kh);
        using var v32 = Tensor.FromArray(allocator, vh);
        using var k = new Tensor(allocator, DType.Float16, kvHeads * cap * d);
        using var v = new Tensor(allocator, DType.Float16, kvHeads * cap * d);
        foreach (Tensor t in new[] { k32, v32, k, v })
            ((CudaStorage)t.Storage).EnsureDeviceCurrent();
        allocator.Context.MakeCurrent();
        allocator.Kernels.LaunchConvertF32F16(((CudaStorage)k32.Storage).DevicePtrAtElement(0), ((CudaStorage)k.Storage).DevicePtrAtElement(0),
            kh.Length, allocator.Stream.Handle);
        allocator.Kernels.LaunchConvertF32F16(((CudaStorage)v32.Storage).DevicePtrAtElement(0), ((CudaStorage)v.Storage).DevicePtrAtElement(0),
            vh.Length, allocator.Stream.Handle);
        using var output = new Tensor(allocator, DType.Float32, nq * heads * d);
        int splits = flash.Splits(nq * group, kvHeads, len);
        using Tensor partial = splits > 1 ? new Tensor(allocator, DType.Float32, CudaFlashPrefillKernels.PartialFloats(nq, heads, d, splits)) : null;
        foreach (Tensor t in new[] { q, output })
            ((CudaStorage)t.Storage).EnsureDeviceCurrent();
        IntPtr P(Tensor t) => ((CudaStorage)t.Storage).DevicePtrAtElement(t.StorageOffset);
        allocator.Context.MakeCurrent();
        flash.LaunchAttention(P(q), P(k), P(v), cap, nq, len, floor, heads, kvHeads, d, scale, splits,
            partial == null ? IntPtr.Zero : P(partial), P(output), allocator.Stream.Handle);
        allocator.Stream.Synchronize();
        ((CudaStorage)output.Storage).MarkDeviceModified();
        float[] got = output.GetElementsAsFloat(nq * heads * d);

        double num = 0, den = 0;
        var w = new double[len];
        for (int h = 0; h < heads; h++)
        {
            int kvh = h / group;
            for (int t = 0; t < nq; t++)
            {
                int end = len - nq + t;   // last visible key
                double mx = double.NegativeInfinity;
                for (int j = floor; j <= end; j++)
                {
                    double dot = 0;
                    for (int x = 0; x < d; x++) dot += qv[(h * nq + t) * d + x] * (double)kh[(kvh * cap + j) * d + x];
                    w[j] = dot * scale;
                    mx = Math.Max(mx, w[j]);
                }
                double sum = 0;
                for (int j = floor; j <= end; j++) { w[j] = Math.Exp(w[j] - mx); sum += w[j]; }
                for (int x = 0; x < d; x++)
                {
                    double want = 0;
                    for (int j = floor; j <= end; j++) want += w[j] * (double)vh[(kvh * cap + j) * d + x];
                    want /= sum;
                    double diff = got[(t * heads + h) * d + x] - want;
                    num += diff * diff;
                    den += want * want;
                }
            }
        }
        double rel = Math.Sqrt(num / den);
        _output.WriteLine($"d {d}, {heads}/{kvHeads} heads, {nq} rows over {len} keys (floor {floor}), {splits} splits: relative L2 {rel:E2}");
        Assert.True(rel <= 5e-3, $"relative L2 {rel:E2}");
    }

    /// <summary>A decode row over the cells a device mask keeps (a sparse attention's selection): the cells listed by
    /// ts_graph_mask_cells, the flash kernel reading those rows alone with their mask values added.</summary>
    [CudaTheory]
    [InlineData(256, 16, 2, 3000, 7)]   // about a seventh of the window kept: split over the listed cells
    [InlineData(128, 8, 8, 600, 2)]
    [InlineData(256, 4, 1, 90, 1)]      // every cell
    public void ListedCellsMatchTheReference(int d, int heads, int kvHeads, int window, int every)
    {
        using var allocator = new CudaAllocator();
        CudaFlashPrefillKernels flash = allocator.FlashPrefillKernels;
        if (flash == null || !flash.RunsAttention(d))
        {
            _output.WriteLine("no flash prefill attention for this GPU");
            return;
        }
        allocator.Context.MakeCurrent();
        Assert.True(CudaGraphKernels.TryCreate(out CudaGraphKernels graph, out string reason), reason);
        using (graph)
        {
            int cap = window + 11, group = heads / kvHeads;
            var rng = new Random(23);
            float[] qv = new float[heads * d];
            for (int i = 0; i < qv.Length; i++) qv[i] = (float)(rng.NextDouble() * 2 - 1);
            var kh = new float[kvHeads * cap * d];
            var vh = new float[kvHeads * cap * d];
            for (int i = 0; i < kh.Length; i++)
            {
                kh[i] = (float)(System.Half)(rng.NextDouble() * 2 - 1);
                vh[i] = (float)(System.Half)(rng.NextDouble() * 2 - 1);
            }
            // Kept cells: every few, in runs, with small mask values; the rest -inf.
            var mask = new float[window];
            for (int j = 0; j < window; j++)
                mask[j] = (j / 4) % every == 0 ? (float)(rng.NextDouble() * 0.5 - 0.25) : float.NegativeInfinity;
            float scale = 1f / MathF.Sqrt(d);
            using var q = Tensor.FromArray(allocator, qv);
            using var k32 = Tensor.FromArray(allocator, kh);
            using var v32 = Tensor.FromArray(allocator, vh);
            using var mt = Tensor.FromArray(allocator, mask);
            using var k = new Tensor(allocator, DType.Float16, kvHeads * cap * d);
            using var v = new Tensor(allocator, DType.Float16, kvHeads * cap * d);
            using var cells = new Tensor(allocator, DType.Int32, window);
            using var count = new Tensor(allocator, DType.Int32, 1);
            using var output = new Tensor(allocator, DType.Float32, heads * d);
            int kept = 0;
            foreach (float m in mask) if (!float.IsNegativeInfinity(m)) kept++;
            int splits = flash.Splits(group, kvHeads, kept);
            using Tensor partial = splits > 1 ? new Tensor(allocator, DType.Float32, CudaFlashPrefillKernels.PartialFloats(1, heads, d, splits)) : null;
            foreach (Tensor t in new[] { q, k32, v32, mt, k, v, cells, count, output })
                ((CudaStorage)t.Storage).EnsureDeviceCurrent();
            IntPtr P(Tensor t) => ((CudaStorage)t.Storage).DevicePtrAtElement(t.StorageOffset);
            IntPtr stream = allocator.Stream.Handle;
            allocator.Context.MakeCurrent();
            allocator.Kernels.LaunchConvertF32F16(P(k32), P(k), kh.Length, stream);
            allocator.Kernels.LaunchConvertF32F16(P(v32), P(v), vh.Length, stream);
            graph.LaunchMaskCells(P(mt), P(cells), P(count), 1, window, stream);
            flash.LaunchAttention(P(q), P(k), P(v), cap, 1, window, 0, heads, kvHeads, d, scale, splits,
                partial == null ? IntPtr.Zero : P(partial), P(output), stream, IntPtr.Zero, P(cells), P(count), P(mt));
            allocator.Stream.Synchronize();
            foreach (Tensor t in new[] { output, count }) ((CudaStorage)t.Storage).MarkDeviceModified();
            Assert.Equal(kept, count.GetElementsAsInt(1)[0]);
            float[] got = output.GetElementsAsFloat(heads * d);

            double num = 0, den = 0;
            var w = new double[window];
            for (int h = 0; h < heads; h++)
            {
                int kvh = h / group;
                double mx = double.NegativeInfinity;
                for (int j = 0; j < window; j++)
                {
                    if (float.IsNegativeInfinity(mask[j])) continue;
                    double dot = 0;
                    for (int x = 0; x < d; x++) dot += qv[h * d + x] * (double)kh[(kvh * cap + j) * d + x];
                    w[j] = dot * scale + mask[j];
                    mx = Math.Max(mx, w[j]);
                }
                double sum = 0;
                for (int j = 0; j < window; j++)
                    if (!float.IsNegativeInfinity(mask[j])) { w[j] = Math.Exp(w[j] - mx); sum += w[j]; }
                for (int x = 0; x < d; x++)
                {
                    double want = 0;
                    for (int j = 0; j < window; j++)
                        if (!float.IsNegativeInfinity(mask[j])) want += w[j] * (double)vh[(kvh * cap + j) * d + x];
                    want /= sum;
                    double diff = got[h * d + x] - want;
                    num += diff * diff;
                    den += want * want;
                }
            }
            double rel = Math.Sqrt(num / den);
            _output.WriteLine($"d {d}, {heads}/{kvHeads} heads, {kept} of {window} cells, {splits} splits: relative L2 {rel:E2}");
            Assert.True(rel <= 5e-3, $"relative L2 {rel:E2}");
        }
    }

    /// <summary>
    /// A verify window's rows in one launch against one one-row launch a row: row z attends the window less the rows - 1 - z
    /// keys after it (the length read from a device block, as a recorded step reads it) in the launch's shared row tiles
    /// (ts_flash_decode_window_f16_d*), or its own row of a selection's cells on the row grid (ts_flash_decode_rows_f16_d*),
    /// with the window's key splits. Bit for bit: speculative decoding answers as plain greedy decoding only if a window's
    /// row rounds as the decode row at its position. Rows whose lengths split their keys at different tile boundaries (a
    /// share rounded up past a 32-key tile between the window's first and last row) keep their own splits.
    /// </summary>
    [CudaTheory]
    [InlineData(256, 24, 4, 8, 4100, 0, 0)]   // Qwen3.8-27B's heads, an 8-row window after 4K: shares straddle a tile
    [InlineData(256, 24, 4, 8, 4097, 0, 0)]
    [InlineData(128, 16, 2, 8, 2049, 0, 0)]
    [InlineData(256, 24, 4, 8, 513, 0, 0)]
    [InlineData(256, 24, 4, 8, 20, 0, 0)]     // one split, keys shorter than a tile for the first rows
    [InlineData(256, 16, 2, 4, 2100, 0, 0)]   // split over the keys
    [InlineData(256, 32, 4, 8, 700, 0, 0)]
    [InlineData(128, 8, 1, 3, 300, 40, 0)]    // keys below a floor masked
    [InlineData(256, 16, 2, 4, 3000, 0, 7)]   // a selection: each row's own cells
    [InlineData(256, 4, 1, 2, 90, 0, 1)]
    public void WindowRowsMatchOneRowLaunches(int d, int heads, int kvHeads, int rows, int len, int floor, int every)
    {
        using var allocator = new CudaAllocator();
        CudaFlashPrefillKernels flash = allocator.FlashPrefillKernels;
        if (flash == null || !flash.RunsAttention(d))
        {
            _output.WriteLine("no flash prefill attention for this GPU");
            return;
        }
        allocator.Context.MakeCurrent();
        Assert.True(CudaGraphKernels.TryCreate(out CudaGraphKernels graph, out string reason), reason);
        using (graph)
        {
            bool selection = every > 0;
            int cap = len + 19, group = heads / kvHeads;
            var rng = new Random(41);
            float[] qv = new float[rows * heads * d];
            for (int i = 0; i < qv.Length; i++) qv[i] = (float)(rng.NextDouble() * 2 - 1);
            var kh = new float[kvHeads * cap * d];
            var vh = new float[kvHeads * cap * d];
            for (int i = 0; i < kh.Length; i++)
            {
                kh[i] = (float)(System.Half)(rng.NextDouble() * 2 - 1);
                vh[i] = (float)(System.Half)(rng.NextDouble() * 2 - 1);
            }
            // Each row's own kept cells (rotated runs, small mask values), the rest -inf.
            var mask = new float[rows * len];
            for (int z = 0; z < rows; z++)
                for (int j = 0; j < len; j++)
                    mask[z * len + j] = selection && ((j / 4) + z) % every == 0 ? (float)(rng.NextDouble() * 0.5 - 0.25) : float.NegativeInfinity;
            float scale = 1f / MathF.Sqrt(d);
            int splits = flash.Splits(group, kvHeads, len);
            using var q = Tensor.FromArray(allocator, qv);
            using var k32 = Tensor.FromArray(allocator, kh);
            using var v32 = Tensor.FromArray(allocator, vh);
            using var mt = Tensor.FromArray(allocator, mask);
            using var dyn = Tensor.FromArray(allocator, new[] { len, 0, 0, 0 });
            using var k = new Tensor(allocator, DType.Float16, kvHeads * cap * d);
            using var v = new Tensor(allocator, DType.Float16, kvHeads * cap * d);
            using var cells = new Tensor(allocator, DType.Int32, rows * len);
            using var count = new Tensor(allocator, DType.Int32, rows);
            using var together = new Tensor(allocator, DType.Float32, rows * heads * d);
            using var alone = new Tensor(allocator, DType.Float32, rows * heads * d);
            using Tensor partial = splits > 1 ? new Tensor(allocator, DType.Float32, CudaFlashPrefillKernels.PartialFloats(rows, heads, d, splits)) : null;
            foreach (Tensor t in new[] { q, k32, v32, mt, dyn, k, v, cells, count, together, alone })
                ((CudaStorage)t.Storage).EnsureDeviceCurrent();
            IntPtr P(Tensor t) => ((CudaStorage)t.Storage).DevicePtrAtElement(t.StorageOffset);
            IntPtr At(Tensor t, long element) => ((CudaStorage)t.Storage).DevicePtrAtElement(t.StorageOffset + element);
            IntPtr stream = allocator.Stream.Handle;
            IntPtr none = IntPtr.Zero;
            allocator.Context.MakeCurrent();
            allocator.Kernels.LaunchConvertF32F16(P(k32), P(k), kh.Length, stream);
            allocator.Kernels.LaunchConvertF32F16(P(v32), P(v), vh.Length, stream);
            if (selection)
            {
                graph.LaunchMaskCells(P(mt), P(cells), P(count), rows, len, stream);
                flash.LaunchDecodeRows(P(q), P(k), P(v), cap, rows, len, 0, heads, kvHeads, d, scale, splits,
                    partial == null ? none : P(partial), P(together), stream, none, P(cells), P(count), P(mt), len);
            }
            else
                // The length from the device block (a recorded step's), the scalar one wrong on purpose.
                flash.LaunchDecodeRows(P(q), P(k), P(v), cap, rows, 1, floor, heads, kvHeads, d, scale, splits,
                    partial == null ? none : P(partial), P(together), stream, P(dyn));
            for (int z = 0; z < rows; z++)
            {
                long at = (long)z * heads * d;
                if (selection)
                    flash.LaunchAttention(At(q, at), P(k), P(v), cap, 1, len, 0, heads, kvHeads, d, scale, splits,
                        partial == null ? none : P(partial), At(alone, at), stream, none, At(cells, (long)z * len), At(count, z), At(mt, (long)z * len));
                else
                    flash.LaunchAttention(At(q, at), P(k), P(v), cap, 1, len - (rows - 1 - z), floor, heads, kvHeads, d, scale, splits,
                        partial == null ? none : P(partial), At(alone, at), stream);
            }
            allocator.Stream.Synchronize();
            foreach (Tensor t in new[] { together, alone }) ((CudaStorage)t.Storage).MarkDeviceModified();
            float[] a = together.GetElementsAsFloat(rows * heads * d), b = alone.GetElementsAsFloat(rows * heads * d);
            int differ = 0;
            for (int i = 0; i < a.Length; i++)
                if (BitConverter.SingleToInt32Bits(a[i]) != BitConverter.SingleToInt32Bits(b[i])) differ++;
            int zeros = 0;
            foreach (float x in a) if (x == 0f) zeros++;
            _output.WriteLine($"d {d}, {heads}/{kvHeads} heads, {rows} rows over {len} keys (floor {floor}, every {every}), {splits} splits: {differ} of {a.Length} differ");
            Assert.Equal(0, differ);
            Assert.True(zeros < a.Length / 4, $"{zeros} of {a.Length} outputs are zero");
        }
    }

    /// <summary>
    /// A verify window whose rows' lengths fall in two 256-key buckets (CudaGraphExecutor's decode bucket) against the decode
    /// row at each row's position, launched as the executor launches a decode row: unrecorded with the splits of its own
    /// length's bucket window, and recorded (the length read from its device block, the launch holding the whole window's
    /// splits). Both window kernels (the shared row tiles and the row grid), recorded and not, bit for bit. Each row had run
    /// the splits of the window's last row's bucket, so the rows before a bucket's end split their keys as the next bucket's
    /// (Qwen3.8-27B on an RTX PRO 4000: 2 splits at 256 keys, 4 past them) and a speculating stream rounded apart from plain
    /// decoding wherever a window straddled one.
    /// </summary>
    [CudaTheory]
    [InlineData(256, 24, 4, 8, 260, false, false)]   // Qwen3.8-27B's heads, rows of 253..260 keys
    [InlineData(256, 24, 4, 8, 260, true, false)]
    [InlineData(256, 24, 4, 8, 260, true, true)]
    [InlineData(256, 24, 4, 16, 1030, true, false)]  // 16 rows of 1015..1030 keys
    [InlineData(256, 24, 4, 16, 1030, false, true)]
    [InlineData(128, 16, 2, 5, 2050, false, false)]  // 2046..2050 keys
    [InlineData(128, 16, 2, 5, 2050, true, true)]
    [InlineData(256, 8, 2, 3, 300, true, false)]     // one bucket: every row the window's splits
    public void WindowRowsSplitTheirKeysAsTheDecodeRowAtTheirPosition(int d, int heads, int kvHeads, int rows, int len, bool recorded,
        bool rowGrid)
    {
        using var allocator = new CudaAllocator();
        CudaFlashPrefillKernels flash = allocator.FlashPrefillKernels;
        if (flash == null || !flash.RunsAttention(d))
        {
            _output.WriteLine("no flash prefill attention for this GPU");
            return;
        }
        const int bucket = 256;
        int group = heads / kvHeads, windowKeys = (len / bucket + 2) * bucket, cap = windowKeys;
        var rng = new Random(43);
        float[] qv = new float[rows * heads * d];
        for (int i = 0; i < qv.Length; i++) qv[i] = (float)(rng.NextDouble() * 2 - 1);
        var kh = new float[kvHeads * cap * d];
        var vh = new float[kvHeads * cap * d];
        for (int i = 0; i < kh.Length; i++)
        {
            kh[i] = (float)(System.Half)(rng.NextDouble() * 2 - 1);
            vh[i] = (float)(System.Half)(rng.NextDouble() * 2 - 1);
        }
        float scale = 1f / MathF.Sqrt(d);
        int Bucketed(int keys) => flash.Splits(group, kvHeads, CudaFlashPrefillKernels.BucketKeys(windowKeys, keys, bucket));
        // A recorded launch holds the whole window's splits; an unrecorded one is launched with its length's bucket's.
        int launched = recorded ? flash.Splits(group, kvHeads, windowKeys) : Bucketed(len);
        var lengths = new int[rows + 1];
        for (int z = 0; z < rows; z++) lengths[z + 1] = len - (rows - 1 - z);
        lengths[0] = len;
        var dynHost = new int[(rows + 1) * 4];
        for (int block = 0; block <= rows; block++) dynHost[4 * block] = lengths[block];
        using var q = Tensor.FromArray(allocator, qv);
        using var k32 = Tensor.FromArray(allocator, kh);
        using var v32 = Tensor.FromArray(allocator, vh);
        using var dyn = Tensor.FromArray(allocator, dynHost);
        using var k = new Tensor(allocator, DType.Float16, kvHeads * cap * d);
        using var v = new Tensor(allocator, DType.Float16, kvHeads * cap * d);
        using var together = new Tensor(allocator, DType.Float32, rows * heads * d);
        using var alone = new Tensor(allocator, DType.Float32, rows * heads * d);
        using var aloneRecorded = new Tensor(allocator, DType.Float32, rows * heads * d);
        int widest = flash.Splits(group, kvHeads, windowKeys);
        using var partial = new Tensor(allocator, DType.Float32, CudaFlashPrefillKernels.PartialFloats(rows, heads, d, widest));
        using var rowPartial = new Tensor(allocator, DType.Float32, CudaFlashPrefillKernels.PartialFloats(1, heads, d, widest));
        foreach (Tensor t in new[] { q, k32, v32, dyn, k, v, together, alone, aloneRecorded, partial, rowPartial })
            ((CudaStorage)t.Storage).EnsureDeviceCurrent();
        IntPtr P(Tensor t) => ((CudaStorage)t.Storage).DevicePtrAtElement(t.StorageOffset);
        IntPtr At(Tensor t, long element) => ((CudaStorage)t.Storage).DevicePtrAtElement(t.StorageOffset + element);
        IntPtr stream = allocator.Stream.Handle;
        allocator.Context.MakeCurrent();
        allocator.Kernels.LaunchConvertF32F16(P(k32), P(k), kh.Length, stream);
        allocator.Kernels.LaunchConvertF32F16(P(v32), P(v), vh.Length, stream);
        flash.RowGridForTests = rowGrid;
        try
        {
            // The window's launch: its length from the step's block when recorded (the scalar one wrong on purpose).
            flash.LaunchDecodeRows(P(q), P(k), P(v), cap, rows, recorded ? 1 : len, 0, heads, kvHeads, d, scale, launched,
                launched > 1 ? P(partial) : IntPtr.Zero, P(together), stream, recorded ? P(dyn) : IntPtr.Zero, windowKeys: windowKeys,
                bucket: bucket);
        }
        finally
        {
            flash.RowGridForTests = false;
        }
        var splitsOf = new int[rows];
        for (int z = 0; z < rows; z++)
        {
            long at = (long)z * heads * d;
            int lenZ = lengths[z + 1];
            splitsOf[z] = Bucketed(lenZ);
            // The decode row unrecorded (its length's bucket splits), then recorded (the length its block holds).
            flash.LaunchAttention(At(q, at), P(k), P(v), cap, 1, lenZ, 0, heads, kvHeads, d, scale, splitsOf[z],
                splitsOf[z] > 1 ? P(rowPartial) : IntPtr.Zero, At(alone, at), stream);
            flash.LaunchAttention(At(q, at), P(k), P(v), cap, 1, 1, 0, heads, kvHeads, d, scale, widest,
                widest > 1 ? P(rowPartial) : IntPtr.Zero, At(aloneRecorded, at), stream, At(dyn, 4L * (z + 1)), windowKeys: windowKeys,
                bucket: bucket);
        }
        allocator.Stream.Synchronize();
        foreach (Tensor t in new[] { together, alone, aloneRecorded }) ((CudaStorage)t.Storage).MarkDeviceModified();
        int n = rows * heads * d;
        float[] a = together.GetElementsAsFloat(n), b = alone.GetElementsAsFloat(n), c = aloneRecorded.GetElementsAsFloat(n);
        int Differ(float[] x, float[] y)
        {
            int count = 0;
            for (int i = 0; i < x.Length; i++)
                if (BitConverter.SingleToInt32Bits(x[i]) != BitConverter.SingleToInt32Bits(y[i])) count++;
            return count;
        }
        int differ = Differ(a, b), recordedDiffer = Differ(c, b);
        _output.WriteLine($"d {d}, {heads}/{kvHeads} heads, {rows} rows of {lengths[1]}..{len} keys, row splits [{string.Join(",", splitsOf)}], " +
            $"launched {launched} ({(recorded ? "recorded" : "unrecorded")}, {(rowGrid ? "row grid" : "row tiles")}): {differ} of {n} differ " +
            $"from the decode rows ({recordedDiffer} between the decode rows recorded and not)");
        Assert.Equal(0, recordedDiffer);
        Assert.Equal(0, differ);
    }
}
