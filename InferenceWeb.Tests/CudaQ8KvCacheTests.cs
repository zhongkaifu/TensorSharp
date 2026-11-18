// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using TensorSharp;
using TensorSharp.Cuda;
using TensorSharp.Graph;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>ggml's q8_0 rows on the host (block_q8_0: an F16 scale d, then 32 int8 codes): quantize_row_q8_0_ref, the value
/// a CUDA flash kernel's tile holds for a code (RN_F16(code * d)), and q8_0 tensors on a CUDA allocator.</summary>
internal static class Q8Reference
{
    public const int Block = 32, BlockBytes = 34;

    public static long Bytes(long values) => values / Block * BlockBytes;

    /// <summary>ggml's quantize_row_q8_0_ref: amax over the block, d = amax / 127, id = d ? 1 / d : 0, code = roundf(x * id)
    /// (half away from zero), d stored as F16 (round to nearest even).</summary>
    public static byte[] Quantize(ReadOnlySpan<float> x)
    {
        var q = new byte[Bytes(x.Length)];
        for (int b = 0; b < x.Length / Block; b++)
        {
            float amax = 0f;
            for (int j = 0; j < Block; j++)
            {
                float v = MathF.Abs(x[b * Block + j]);
                amax = amax > v ? amax : v;
            }
            float d = amax / 127f;
            float id = d != 0f ? 1f / d : 0f;
            ushort bits = BitConverter.HalfToUInt16Bits((System.Half)d);
            q[b * BlockBytes] = (byte)bits;
            q[b * BlockBytes + 1] = (byte)(bits >> 8);
            for (int j = 0; j < Block; j++)
                q[b * BlockBytes + 2 + j] = (byte)(sbyte)MathF.Round(x[b * Block + j] * id, MidpointRounding.AwayFromZero);
        }
        return q;
    }

    /// <summary>Value <paramref name="i"/> of q8_0 bytes as the flash kernels dequantize it: RN_F16(code * d) (code * d is exact in
    /// F32).</summary>
    public static float Value(byte[] q, long i)
    {
        long b = i / Block;
        int j = (int)(i % Block);
        System.Half d = BitConverter.UInt16BitsToHalf((ushort)(q[b * BlockBytes] | (q[b * BlockBytes + 1] << 8)));
        sbyte code = (sbyte)q[b * BlockBytes + 2 + j];
        return (float)(System.Half)(code * (float)d);
    }

    public static float[] Dequantize(byte[] q, long values)
    {
        var x = new float[values];
        for (long i = 0; i < values; i++) x[i] = Value(q, i);
        return x;
    }

    /// <summary>Random q8_0 blocks: codes over [-127, 127]; scales mostly 0.002-0.032, some zero blocks and some whose products
    /// are F16 subnormals.</summary>
    public static byte[] Random(Random rng, long values)
    {
        var q = new byte[Bytes(values)];
        for (long b = 0; b < values / Block; b++)
        {
            double pick = rng.NextDouble();
            float d = pick < 0.04 ? 0f : pick < 0.08 ? (float)(rng.NextDouble() * 4e-6) : (float)(0.002 + rng.NextDouble() * 0.03);
            ushort bits = BitConverter.HalfToUInt16Bits((System.Half)d);
            q[b * BlockBytes] = (byte)bits;
            q[b * BlockBytes + 1] = (byte)(bits >> 8);
            for (int j = 0; j < Block; j++) q[b * BlockBytes + 2 + j] = (byte)(sbyte)rng.Next(-127, 128);
        }
        return q;
    }

    /// <summary>A q8_0 tensor of <paramref name="sizes"/> holding <paramref name="bytes"/> on the device.</summary>
    public static Tensor Upload(CudaAllocator allocator, byte[] bytes, params long[] sizes)
    {
        var t = new Tensor(allocator, DType.Q8_0, sizes);
        Assert.Equal(bytes.Length, t.Storage.ByteLength);
        Marshal.Copy(bytes, 0, t.Storage.PtrAtElementForOverwrite(0), bytes.Length);
        ((CudaStorage)t.Storage).EnsureDeviceCurrent();
        return t;
    }

    public static byte[] Download(Tensor t)
    {
        var bytes = new byte[t.Storage.ByteLength];
        Marshal.Copy(t.Storage.PtrAtElementForRead(0), bytes, 0, bytes.Length);
        return bytes;
    }
}

/// <summary>
/// A q8_0 K/V cache on direct CUDA (ggml's block_q8_0 rows): the flash kernels over it (ts_flash_*_q8_d*) answer as the F16
/// kernels over the F16 cache of its values, bit for bit, in every launch the executor makes (decode rows recorded and not,
/// verify windows in shared row tiles and on the row grid, a selection's cells, short and prompt chunks); and every way a row
/// is written into it - set_rows by device ids, the host-indexed writes of a run of rows and of scattered rows - holds ggml's
/// quantize_row_q8_0 bytes for the row's values (zero blocks and halfway codes included).
/// </summary>
public sealed class CudaQ8KvCacheTests
{
    private readonly ITestOutputHelper _output;

    public CudaQ8KvCacheTests(ITestOutputHelper output) => _output = output;

    private static IntPtr P(Tensor t) => ((CudaStorage)t.Storage).DevicePtrAtElement(t.StorageOffset);

    /// <summary>The _q8 flash kernels against the F16 ones over the dequantized cache. Their bodies are sm_80 code: where the
    /// module this process loads has none (the committed compute_75 PTX of a build without nvcc, an nvcc build for an sm_7x GPU)
    /// the gate skips the test, and a module that has them must run them (a stale image fails here).</summary>
    [CudaTheory(TensorCoreModule = "tensorsharp_flash_prefill")]
    [InlineData("decode", 256, 24, 4, 1, 3700, 0)]             // Qwen3.8-27B's heads, a decode row over a long window
    [InlineData("decode", 128, 16, 2, 1, 2100, 33)]            // 128-wide heads (8-byte rows), keys below a floor
    [InlineData("decode-recorded", 256, 24, 4, 1, 1030, 0)]    // the length read on the device, the launch over the capacity
    [InlineData("window", 256, 24, 4, 8, 4100, 0)]             // an 8-row verify window in shared row tiles
    [InlineData("window", 128, 16, 2, 5, 2050, 0)]
    [InlineData("window-recorded", 256, 24, 4, 8, 1030, 0)]
    [InlineData("window-grid", 256, 24, 4, 16, 4097, 0)]       // the row grid (a launch a row in one grid)
    [InlineData("verify", 256, 24, 4, 3, 2100, 0)]             // a short chunk's rows on the prefill kernel
    [InlineData("prompt", 256, 24, 4, 130, 1100, 0)]           // a prompt chunk on the prompt kernel
    [InlineData("prompt", 128, 16, 2, 300, 300, 40)]
    [InlineData("selection", 256, 16, 2, 4, 3000, 0)]          // each row's own cells of a selection
    public void Q8CacheEqualsF16KernelOverDequantizedCache(string mode, int d, int heads, int kvHeads, int rows, int len, int floor)
    {
        using var allocator = new CudaAllocator();
        CudaFlashPrefillKernels flash = allocator.FlashPrefillKernels;
        Assert.True(flash != null && flash.RunsAttentionQ8(d),
            $"the loaded tensorsharp_flash_prefill module carries sm_80 code but runs no q8_0 attention for {d}-wide heads " +
            $"(current image: {flash?.IsCurrent.ToString() ?? "none loaded"})");
        allocator.Context.MakeCurrent();
        Assert.True(CudaGraphKernels.TryCreate(out CudaGraphKernels graph, out string reason), reason);
        using (graph)
        {
            const int bucket = 256;
            bool recorded = mode.EndsWith("-recorded", StringComparison.Ordinal);
            int group = heads / kvHeads;
            // An odd capacity where the launch is over the written keys; a whole bucket window where it is over the capacity.
            int cap = recorded ? (len / bucket + 2) * bucket : len + 19;
            int nq = mode is "decode" or "decode-recorded" ? 1 : rows;
            var rng = new Random(d * 31 + rows * 7 + len);
            float[] qv = new float[nq * heads * d];
            for (int i = 0; i < qv.Length; i++) qv[i] = (float)(rng.NextDouble() * 2 - 1);
            long values = (long)kvHeads * cap * d;
            byte[] kq = Q8Reference.Random(rng, values), vq = Q8Reference.Random(rng, values);
            float[] kh = Q8Reference.Dequantize(kq, values), vh = Q8Reference.Dequantize(vq, values);
            float scale = 1f / MathF.Sqrt(d);
            bool selection = mode == "selection";
            var mask = new float[selection ? rows * len : 1];
            if (selection)
                for (int z = 0; z < rows; z++)
                    for (int j = 0; j < len; j++)
                        mask[z * len + j] = ((j / 4) + z) % 7 == 0 ? (float)(rng.NextDouble() * 0.5 - 0.25) : float.NegativeInfinity;

            using var q = Tensor.FromArray(allocator, qv);
            using var k32 = Tensor.FromArray(allocator, kh);
            using var v32 = Tensor.FromArray(allocator, vh);
            using var k16 = new Tensor(allocator, DType.Float16, values);
            using var v16 = new Tensor(allocator, DType.Float16, values);
            using Tensor k8 = Q8Reference.Upload(allocator, kq, values), v8 = Q8Reference.Upload(allocator, vq, values);
            using var mt = Tensor.FromArray(allocator, mask);
            using var dyn = Tensor.FromArray(allocator, new[] { len, 0, 0, 0 });
            using var cells = new Tensor(allocator, DType.Int32, Math.Max(1, selection ? rows * len : 1));
            using var count = new Tensor(allocator, DType.Int32, Math.Max(1, rows));
            using var outF16 = new Tensor(allocator, DType.Float32, nq * heads * d);
            using var outQ8 = new Tensor(allocator, DType.Float32, nq * heads * d);
            int splits = mode switch
            {
                "decode" => flash.Splits(group, kvHeads, len),
                "decode-recorded" or "window-recorded" => flash.Splits(group, kvHeads, cap),
                "window" or "window-grid" => flash.Splits(group, kvHeads, CudaFlashPrefillKernels.BucketKeys(cap, len, bucket)),
                "selection" => flash.Splits(group, kvHeads, len),
                _ => flash.Splits(nq * group, kvHeads, len),
            };
            using Tensor partial = splits > 1 ? new Tensor(allocator, DType.Float32, CudaFlashPrefillKernels.PartialFloats(nq, heads, d, splits)) : null;
            foreach (Tensor t in new[] { q, k32, v32, k16, v16, mt, dyn, cells, count, outF16, outQ8 })
                ((CudaStorage)t.Storage).EnsureDeviceCurrent();
            IntPtr stream = allocator.Stream.Handle, none = IntPtr.Zero, part = partial == null ? none : P(partial);
            allocator.Context.MakeCurrent();
            allocator.Kernels.LaunchConvertF32F16(P(k32), P(k16), values, stream);
            allocator.Kernels.LaunchConvertF32F16(P(v32), P(v16), values, stream);
            if (selection) graph.LaunchMaskCells(P(mt), P(cells), P(count), rows, len, stream);

            void Run(Tensor k, Tensor v, Tensor output, bool q8)
            {
                switch (mode)
                {
                    case "decode":
                        flash.LaunchAttention(P(q), P(k), P(v), cap, 1, len, floor, heads, kvHeads, d, scale, splits, part, P(output), stream, q8: q8);
                        break;
                    case "decode-recorded":
                        flash.LaunchAttention(P(q), P(k), P(v), cap, 1, 1, floor, heads, kvHeads, d, scale, splits, part, P(output), stream, P(dyn),
                            windowKeys: cap, bucket: bucket, q8: q8);
                        break;
                    case "window":
                    case "window-grid":
                        flash.RowGridForTests = mode == "window-grid";
                        try
                        {
                            flash.LaunchDecodeRows(P(q), P(k), P(v), cap, rows, len, floor, heads, kvHeads, d, scale, splits, part, P(output), stream,
                                windowKeys: cap, bucket: bucket, q8: q8);
                        }
                        finally
                        {
                            flash.RowGridForTests = false;
                        }
                        break;
                    case "window-recorded":
                        flash.LaunchDecodeRows(P(q), P(k), P(v), cap, rows, 1, floor, heads, kvHeads, d, scale, splits, part, P(output), stream, P(dyn),
                            windowKeys: cap, bucket: bucket, q8: q8);
                        break;
                    case "selection":
                        flash.LaunchDecodeRows(P(q), P(k), P(v), cap, rows, len, 0, heads, kvHeads, d, scale, splits, part, P(output), stream, none,
                            P(cells), P(count), P(mt), len, q8: q8);
                        break;
                    default:   // "verify", "prompt"
                        flash.LaunchAttention(P(q), P(k), P(v), cap, nq, len, floor, heads, kvHeads, d, scale, splits, part, P(output), stream,
                            prompt: mode == "prompt", q8: q8);
                        break;
                }
            }
            Run(k16, v16, outF16, q8: false);
            Run(k8, v8, outQ8, q8: true);
            allocator.Stream.Synchronize();
            foreach (Tensor t in new[] { outF16, outQ8 }) ((CudaStorage)t.Storage).MarkDeviceModified();
            float[] want = outF16.GetElementsAsFloat(nq * heads * d), got = outQ8.GetElementsAsFloat(nq * heads * d);
            int differ = 0, zeros = 0;
            for (int i = 0; i < want.Length; i++)
            {
                if (BitConverter.SingleToInt32Bits(want[i]) != BitConverter.SingleToInt32Bits(got[i])) differ++;
                if (got[i] == 0f) zeros++;
            }
            _output.WriteLine($"{mode}: d {d}, {heads}/{kvHeads} heads, {nq} rows over {len} keys (cap {cap}, floor {floor}), {splits} splits: " +
                $"{differ} of {want.Length} outputs differ from the F16 kernel over the dequantized cache");
            Assert.Equal(0, differ);
            Assert.True(zeros < want.Length / 4, $"{zeros} of {want.Length} outputs are zero");
        }
    }

    /// <summary>F32 rows of <paramref name="heads"/> heads with the blocks quantization is touchy about: a zero block, blocks whose
    /// codes fall halfway (amax 127: d = 1, so x.5 values round away from zero), and the rest random.</summary>
    private static float[] TouchyRows(Random rng, int heads, int count, int d)
    {
        var x = new float[heads * count * d];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 6 - 3);
        for (int h = 0; h < heads; h++)
            for (int r = 0; r < count; r++)
            {
                int row = (h * count + r) * d;
                if ((h + r) % 3 == 0)
                    for (int j = 0; j < 32; j++) x[row + j] = 0f;
                if ((h + r) % 2 == 0)
                {
                    int b = row + 32 * ((r % (d / 32 - 1)) + 1);
                    x[b] = (r & 1) == 0 ? 127f : -127f;
                    for (int j = 1; j < 32; j++) x[b + j] = (j % 4 == 0 ? -1 : 1) * (0.5f + (j % 9));
                }
            }
        return x;
    }

    /// <summary>The q8_0 cache bytes after writing <paramref name="src"/> [heads][count][d] rows at <paramref name="ids"/> into
    /// <paramref name="start"/> (rows outside [0, cap) skipped), with the reference quantization.</summary>
    private static byte[] Expected(byte[] start, float[] src, int[] ids, int heads, int cap, int d)
    {
        var bytes = (byte[])start.Clone();
        int count = ids.Length;
        long rowBytes = Q8Reference.Bytes(d);
        for (int h = 0; h < heads; h++)
            for (int i = 0; i < count; i++)
            {
                if (ids[i] < 0 || ids[i] >= cap) continue;
                byte[] row = Q8Reference.Quantize(src.AsSpan((h * count + i) * d, d));
                Array.Copy(row, 0, bytes, (h * (long)cap + ids[i]) * rowBytes, rowBytes);
            }
        return bytes;
    }

    private static void AssertSameBytes(byte[] want, byte[] got, int d, string what)
    {
        Assert.Equal(want.Length, got.Length);
        long rowBytes = Q8Reference.Bytes(d);
        for (int i = 0; i < want.Length; i++)
            if (want[i] != got[i])
                Assert.Fail($"{what}: byte {i} (row {i / rowBytes}, block {i % rowBytes / Q8Reference.BlockBytes}) is {got[i]:x2}, ggml's " +
                    $"quantize_row_q8_0 gives {want[i]:x2}");
    }

    /// <summary>ts_graph_set_rows_q8_0 by device ids (one past the cache skipped), from a first row, and from the first row a
    /// recorded step's dynamic block holds (its KV write position, read on the device: the scalar one wrong on purpose), against
    /// ggml's quantize_row_q8_0 on the host.</summary>
    [CudaTheory]
    [InlineData(256, 4)]
    [InlineData(128, 2)]
    public void SetRowsKernelWritesGgmlQuantizedRows(int d, int heads)
    {
        using var allocator = new CudaAllocator();
        allocator.Context.MakeCurrent();
        Assert.True(CudaGraphKernels.TryCreate(out CudaGraphKernels kernels, out string reason), reason);
        using (kernels)
        {
            const int cap = 40;
            var rng = new Random(d + heads);
            int[] ids = { 3, 7, 0, 39, 12, 40 };
            int count = ids.Length;
            float[] src = TouchyRows(rng, heads, count, d);
            byte[] start = Enumerable.Range(0, (int)Q8Reference.Bytes((long)heads * cap * d)).Select(i => (byte)(i * 7 + 1)).ToArray();
            long rowBytes = Q8Reference.Bytes(d);
            foreach (string mode in new[] { "ids", "first", "dyn" })
            {
                int first = mode == "dyn" ? 2 : 30, dynFirst = 17;
                int[] rows = mode == "ids" ? ids : Enumerable.Range(mode == "dyn" ? dynFirst : first, count).ToArray();
                using Tensor cache = Q8Reference.Upload(allocator, start, heads, cap, d);
                using var s = Tensor.FromArray(allocator, src);
                using var idx = Tensor.FromArray(allocator, ids);
                using var dyn = Tensor.FromArray(allocator, new[] { 0, dynFirst, 0, 0 });
                foreach (Tensor t in new[] { s, idx, dyn }) ((CudaStorage)t.Storage).EnsureDeviceCurrent();
                allocator.Context.MakeCurrent();
                kernels.LaunchSetRowsQ8(P(s), d, (long)count * d, mode == "ids" ? P(idx) : IntPtr.Zero, first, P(cache), rowBytes, cap * rowBytes, cap, d,
                    count, heads, allocator.Stream.Handle, mode == "dyn" ? P(dyn) : IntPtr.Zero);
                allocator.Stream.Synchronize();
                ((CudaStorage)cache.Storage).MarkDeviceModified();
                AssertSameBytes(Expected(start, src, rows, heads, cap, d), Q8Reference.Download(cache), d, mode);
            }
        }
    }

    /// <summary>The executor's ggml_set_rows into a q8_0 cache leaf: rows on the device (a recorded step's ids: TrySetRowsOnDevice),
    /// a run of host rows (TryWriteRows) and scattered host rows (the run copies through TryConvert), each the reference
    /// bytes.</summary>
    [CudaTheory]
    [InlineData("device")]
    [InlineData("run")]
    [InlineData("scattered")]
    public void ExecutorRowWritesHoldGgmlQuantizedRows(string ids)
    {
        const int d = 256, heads = 4, cap = 64;
        int[] rows = ids == "scattered" ? new[] { 9, 10, 2, 33, 34, 35, 60 } : Enumerable.Range(17, 7).ToArray();
        int count = rows.Length;
        using var allocator = new CudaAllocator();
        var rng = new Random(count * 13 + ids.Length);
        float[] src = TouchyRows(rng, heads, count, d);
        byte[] start = Enumerable.Range(0, (int)Q8Reference.Bytes((long)heads * cap * d)).Select(i => (byte)(i * 5 + 3)).ToArray();
        using Tensor cache = Q8Reference.Upload(allocator, start, heads, cap, d);
        using var executor = new CudaGraphExecutor(allocator);
        bool device = ids == "device";
        var g = new GraphProgramBuilder(record: true);
        int cacheNode = g.Leaf(GraphLeafRole.State, GraphFormat.Q8_0, new long[] { d, cap, heads }, GraphIdentity.Of(cache.Storage),
            cache.Storage.ByteLength, handle: cache);
        int srcNode = g.Leaf(GraphLeafRole.Input, GraphFormat.F32, new long[] { d, count, heads }, IntPtr.Zero, src.Length * 4L);
        int rowsNode = g.Leaf(GraphLeafRole.Input, device ? GraphFormat.I32 : GraphFormat.I64, new long[] { count }, IntPtr.Zero, count * (device ? 4L : 8L));
        g.Effect(g.Op(GraphOp.SetRows, stackalloc[] { cacheNode, srcNode, rowsNode }));
        int probe = g.Op(GraphOp.Scale, stackalloc[] { srcNode }, default, stackalloc[] { 1f });
        g.Output(probe, src.Length);
        using IGraphPlan plan = executor.Compile(g.ToProgram(replayable: true, reorder: false));
        long[] hostRows = rows.Select(r => (long)r).ToArray();
        using var deviceRows = Tensor.FromArray(allocator, rows);
        ((CudaStorage)deviceRows.Storage).EnsureDeviceCurrent();
        GCHandle srcPin = GCHandle.Alloc(src, GCHandleType.Pinned), rowsPin = GCHandle.Alloc(hostRows, GCHandleType.Pinned);
        try
        {
            bool ran = device
                ? plan.Run(new[] { srcNode, rowsNode }, new[] { srcPin.AddrOfPinnedObject(), IntPtr.Zero }, new[] { src.Length * 4L, count * 4L },
                    new object[] { null, deviceRows })
                : plan.Run(new[] { srcNode, rowsNode }, new[] { srcPin.AddrOfPinnedObject(), rowsPin.AddrOfPinnedObject() },
                    new[] { src.Length * 4L, count * 8L });
            Assert.True(ran, plan.LastError);
        }
        finally
        {
            srcPin.Free();
            rowsPin.Free();
        }
        allocator.Stream.Synchronize();
        AssertSameBytes(Expected(start, src, rows, heads, cap, d), Q8Reference.Download(cache), d, $"{ids} rows");
    }
}
