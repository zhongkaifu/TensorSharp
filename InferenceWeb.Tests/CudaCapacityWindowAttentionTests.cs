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
/// A recorded decode row's or verify window's flash attention launched over its slot's whole capacity (the splits of the
/// capacity's window, the bucket's run on the device from the length it reads there) against the launch over the 256-key
/// bucket window around its length that a recorded step made before: the same bits at lengths around the buckets' edges
/// (1, 255, 256, 257, 1,000, 4,097) and one short of a capacity that is not a whole bucket - a decode row, verify windows in
/// one launch, and a window's rows run one at a time. Every row runs the splits of its own length's bucket (the decode row's
/// at its position), so a window across a bucket's edge splits its earlier rows as the bucket before. An unrecorded launch
/// with the bucket's splits gives them too.
/// </summary>
public sealed class CudaCapacityWindowAttentionTests
{
    private const int Bucket = 256, Cap = 4400;
    private static readonly int[] Lengths = { 1, 255, 256, 257, 1000, 4097, Cap - 1 };

    private readonly ITestOutputHelper _output;

    public CudaCapacityWindowAttentionTests(ITestOutputHelper output) => _output = output;

    private static IntPtr P(Tensor t) => ((CudaStorage)t.Storage).DevicePtrAtElement(t.StorageOffset);

    private static IntPtr At(Tensor t, long element) => ((CudaStorage)t.Storage).DevicePtrAtElement(t.StorageOffset + element);

    /// <summary><paramref name="rows"/>: 1 a decode row, more a verify window in one launch, negative a window of that many
    /// rows run one row at a time.</summary>
    [CudaTheory]
    [InlineData(256, 24, 4, 1)]    // Qwen3.8-27B's heads, a decode row
    [InlineData(256, 16, 2, 4)]    // a 4-row verify window
    [InlineData(128, 8, 1, 8)]     // 128-wide heads, an 8-row window
    [InlineData(256, 24, 4, -3)]   // a 3-row window's rows launched alone
    public void CapacityLaunchesGiveTheBucketLaunchesBits(int d, int heads, int kvHeads, int rows)
    {
        using var allocator = new CudaAllocator();
        CudaFlashPrefillKernels flash = allocator.FlashPrefillKernels;
        if (flash == null || !flash.RunsAttention(d))
        {
            _output.WriteLine("no flash prefill attention for this GPU");
            return;
        }
        bool alone = rows < 0;
        int n = Math.Abs(rows), group = heads / kvHeads;
        var rng = new Random(d + heads + rows);
        float[] qv = new float[n * heads * d];
        for (int i = 0; i < qv.Length; i++) qv[i] = (float)(rng.NextDouble() * 2 - 1);
        // Values an F16 cache holds exactly (rounded on the host, converted on the device without loss).
        var kh = new float[kvHeads * Cap * d];
        var vh = new float[kvHeads * Cap * d];
        for (int i = 0; i < kh.Length; i++)
        {
            kh[i] = (float)(System.Half)(rng.NextDouble() * 2 - 1);
            vh[i] = (float)(System.Half)(rng.NextDouble() * 2 - 1);
        }
        float scale = 1f / MathF.Sqrt(d);
        using var q = Tensor.FromArray(allocator, qv);
        using var k32 = Tensor.FromArray(allocator, kh);
        using var v32 = Tensor.FromArray(allocator, vh);
        using var k = new Tensor(allocator, DType.Float16, kvHeads * Cap * d);
        using var v = new Tensor(allocator, DType.Float16, kvHeads * Cap * d);
        using var dyn = new Tensor(allocator, DType.Int32, 4L * n);
        using var bucketOut = new Tensor(allocator, DType.Float32, (long)n * heads * d);
        using var capacityOut = new Tensor(allocator, DType.Float32, (long)n * heads * d);
        using var eagerOut = new Tensor(allocator, DType.Float32, (long)n * heads * d);
        using var partial = new Tensor(allocator, DType.Float32, CudaFlashPrefillKernels.PartialFloats(n, heads, d, 64));
        foreach (Tensor t in new[] { q, k32, v32, k, v, dyn, bucketOut, capacityOut, eagerOut, partial })
            ((CudaStorage)t.Storage).EnsureDeviceCurrent();
        IntPtr stream = allocator.Stream.Handle, none = IntPtr.Zero;
        allocator.Context.MakeCurrent();
        allocator.Kernels.LaunchConvertF32F16(P(k32), P(k), kh.Length, stream);
        allocator.Kernels.LaunchConvertF32F16(P(v32), P(v), vh.Length, stream);
        int Fewer(int z) => n - 1 - z;
        bool exercised = false;
        foreach (int len in Lengths.Where(l => l >= n))
        {
            // The bucket window a recorded step attended before, and its splits; the capacity window's.
            int window = CudaFlashPrefillKernels.BucketKeys(Cap, len, Bucket);
            int bucketSplits = flash.Splits(group, kvHeads, window), capacitySplits = flash.Splits(group, kvHeads, Cap);
            exercised |= capacitySplits > bucketSplits;
            // The step's parameter block (its length), and each row's when its rows run alone.
            var blocks = new int[4 * n];
            for (int z = 0; z < n; z++)
            {
                int rowLen = alone ? len - Fewer(z) : len;
                blocks[4 * z] = rowLen;
                blocks[4 * z + 1] = rowLen - 1;
            }
            dyn.SetElementsAsInt(blocks);
            ((CudaStorage)dyn.Storage).EnsureDeviceCurrent();
            allocator.Context.MakeCurrent();
            // The scalar lengths are wrong on purpose where the device's are read.
            if (alone)
            {
                for (int z = 0; z < n; z++)
                {
                    long at = (long)z * heads * d;
                    IntPtr rowDyn = At(dyn, 4L * z);
                    // Each row its own length's splits, as the decode row at its position (the executor's rows run alone).
                    int rowSplits = flash.Splits(group, kvHeads, CudaFlashPrefillKernels.BucketKeys(window, len - Fewer(z), Bucket));
                    flash.LaunchAttention(At(q, at), P(k), P(v), Cap, 1, 1, 0, heads, kvHeads, d, scale, bucketSplits, P(partial),
                        At(bucketOut, at), stream, rowDyn, windowKeys: window, bucket: Bucket);
                    flash.LaunchAttention(At(q, at), P(k), P(v), Cap, 1, 1, 0, heads, kvHeads, d, scale, capacitySplits, P(partial),
                        At(capacityOut, at), stream, rowDyn, windowKeys: Cap, bucket: Bucket);
                    flash.LaunchAttention(At(q, at), P(k), P(v), Cap, 1, len - Fewer(z), 0, heads, kvHeads, d, scale, rowSplits,
                        P(partial), At(eagerOut, at), stream);
                }
            }
            else if (n == 1)
            {
                flash.LaunchAttention(P(q), P(k), P(v), Cap, 1, 1, 0, heads, kvHeads, d, scale, bucketSplits, P(partial), P(bucketOut),
                    stream, P(dyn));
                flash.LaunchAttention(P(q), P(k), P(v), Cap, 1, 1, 0, heads, kvHeads, d, scale, capacitySplits, P(partial), P(capacityOut),
                    stream, P(dyn), windowKeys: Cap, bucket: Bucket);
                flash.LaunchAttention(P(q), P(k), P(v), Cap, 1, len, 0, heads, kvHeads, d, scale, bucketSplits, P(partial), P(eagerOut),
                    stream);
            }
            else
            {
                // Every launch with the bucket rule, as the executor launches a window: each row its own length's splits.
                flash.LaunchDecodeRows(P(q), P(k), P(v), Cap, n, 1, 0, heads, kvHeads, d, scale, bucketSplits, P(partial), P(bucketOut),
                    stream, P(dyn), windowKeys: window, bucket: Bucket);
                flash.LaunchDecodeRows(P(q), P(k), P(v), Cap, n, 1, 0, heads, kvHeads, d, scale, capacitySplits, P(partial),
                    P(capacityOut), stream, P(dyn), windowKeys: Cap, bucket: Bucket);
                flash.LaunchDecodeRows(P(q), P(k), P(v), Cap, n, len, 0, heads, kvHeads, d, scale, bucketSplits, P(partial), P(eagerOut),
                    stream, windowKeys: window, bucket: Bucket);
            }
            allocator.Stream.Synchronize();
            foreach (Tensor t in new[] { bucketOut, capacityOut, eagerOut }) ((CudaStorage)t.Storage).MarkDeviceModified();
            float[] want = bucketOut.GetElementsAsFloat(n * heads * d);
            float[] got = capacityOut.GetElementsAsFloat(n * heads * d), eager = eagerOut.GetElementsAsFloat(n * heads * d);
            _output.WriteLine($"{len} keys: the bucket window {window} takes {bucketSplits} splits, the capacity window's launch holds {capacitySplits}");
            Assert.True(want.Count(x => x == 0f) < want.Length / 4, $"{len} keys: most outputs are zero");
            for (int i = 0; i < want.Length; i++)
            {
                if (BitConverter.SingleToInt32Bits(want[i]) != BitConverter.SingleToInt32Bits(got[i]))
                    Assert.Fail($"{len} keys: output {i} (row {i / (heads * d)}) is {got[i]:R} over the capacity window, {want[i]:R} over the bucket's");
                if (BitConverter.SingleToInt32Bits(want[i]) != BitConverter.SingleToInt32Bits(eager[i]))
                    Assert.Fail($"{len} keys: output {i} (row {i / (heads * d)}) is {eager[i]:R} unrecorded, {want[i]:R} recorded");
            }
        }
        Assert.True(exercised, "no length ran fewer splits than the capacity window's launch holds");
    }
}
