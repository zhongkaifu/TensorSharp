// Direct test of the global KV-cache growth copy (ModelBase.CopyCacheRows):
//
//   using var srcK = _kvCacheK[l].Narrow(1, 0, _cacheSeqLen);
//   using var dstK = newK.Narrow(1, 0, _cacheSeqLen);
//   Ops.Copy(dstK, srcK);
//
// Both operands are strided views of [heads, capacity, headDim] tensors with
// DIFFERENT dim-0 strides (old capacity vs new capacity). For CUDA this lands on
// the strided device-copy path. If that copy is wrong, every global-attention
// read after the first resize (at seq pos 2048 on CUDA) sees scrambled history —
// which would garble generation right when the sequence crosses the cap.
using System;
using System.Linq;
using System.Reflection;
using TensorSharp;
using TensorSharp.Cuda;
using TensorSharp.Models;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

public class CudaKvCacheResizeTests
{
    private readonly ITestOutputHelper _output;
    public CudaKvCacheResizeTests(ITestOutputHelper output) => _output = output;

    [CudaTheory]
    [InlineData(8, 2048, 4096, 32, 2000)]   // the real first-grow shape (cross 2048)
    [InlineData(4, 64, 128, 16, 50)]        // tiny
    public void ResizeCopy_PreservesStridedHistory(int heads, int oldCap, int newCap, int headDim, int seqLen)
    {
        using var allocator = new CudaAllocator();

        static float KVal(int pos, int h, int d) => MathF.Sin(0.0021f * pos + 0.31f * h + 0.07f * d);

        using var oldK = new Tensor(allocator, DType.Float32, heads, oldCap, headDim);
        using var newK = new Tensor(allocator, DType.Float32, heads, newCap, headDim);

        var oldFlat = new float[heads * oldCap * headDim];
        for (int h = 0; h < heads; h++)
            for (int p = 0; p < oldCap; p++)
                for (int d = 0; d < headDim; d++)
                    oldFlat[(h * oldCap + p) * headDim + d] = p < seqLen ? KVal(p, h, d) : -123456f;
        oldK.SetElementsAsFloat(oldFlat);

        // Sentinel-fill the destination so we can tell copied vs untouched.
        var newFlat = new float[heads * newCap * headDim];
        Array.Fill(newFlat, -987654f);
        newK.SetElementsAsFloat(newFlat);

        using (var srcK = oldK.Narrow(1, 0, seqLen))
        using (var dstK = newK.Narrow(1, 0, seqLen))
            Ops.Copy(dstK, srcK);

        float[] got = newK.GetElementsAsFloat(heads * newCap * headDim);

        double worst = 0; int badH = -1, badP = -1, badD = -1;
        for (int h = 0; h < heads; h++)
            for (int p = 0; p < seqLen; p++)
                for (int d = 0; d < headDim; d++)
                {
                    float expected = KVal(p, h, d);
                    float actual = got[(h * newCap + p) * headDim + d];
                    double err = Math.Abs(expected - actual);
                    if (err > worst) { worst = err; badH = h; badP = p; badD = d; }
                }

        _output.WriteLine($"[resize] heads={heads} {oldCap}->{newCap} hd={headDim} seqLen={seqLen} worstErr={worst:E3} at h={badH},p={badP},d={badD}");
        Assert.True(worst < 1e-5, $"Resize copy corrupted history: worstErr={worst:E3} at h={badH} pos={badP} d={badD}");
    }

    /// <summary>
    /// A q8_0 cache's growth and slot copies (ModelBase.CopyCacheRows) run on the device: one pitched byte copy of each head's
    /// first rows into a cache of another capacity, the rest of the target as it was, and the host mirror never written (the
    /// host route had downloaded both caches at their whole capacity and uploaded the copy again).
    /// </summary>
    [CudaTheory]
    [InlineData(4, 300, 520, 256, 257)]
    [InlineData(2, 64, 40, 128, 33)]
    [InlineData(1, 512, 1024, 512, 511)]   // an extra key cache's single head, wider rows
    public void Q8CacheRowsCopyOnTheDevice(int heads, int sourceCap, int targetCap, int d, int rows)
    {
        using var allocator = new CudaAllocator();
        var rng = new Random(heads + d);
        byte[] from = Q8Reference.Random(rng, (long)heads * sourceCap * d);
        byte[] start = Enumerable.Repeat((byte)0x5A, (int)Q8Reference.Bytes((long)heads * targetCap * d)).ToArray();
        using Tensor source = Q8Reference.Upload(allocator, from, heads, sourceCap, d);
        using Tensor target = Q8Reference.Upload(allocator, start, heads, targetCap, d);
        typeof(ModelBase).GetMethod("CopyCacheRows", BindingFlags.NonPublic | BindingFlags.Static)!
            .Invoke(null, new object[] { source, target, rows });
        Assert.False(((CudaStorage)target.Storage).IsHostDirty, "the copy went through the target's host mirror");
        allocator.Stream.Synchronize();
        byte[] got = Q8Reference.Download(target);
        long rowBytes = Q8Reference.Bytes(d);
        for (int h = 0; h < heads; h++)
            for (long r = 0; r < targetCap; r++)
                for (long b = 0; b < rowBytes; b++)
                {
                    byte want = r < rows ? from[(h * (long)sourceCap + r) * rowBytes + b] : (byte)0x5A;
                    byte at = got[(h * (long)targetCap + r) * rowBytes + b];
                    if (at != want) Assert.Fail($"head {h} row {r} byte {b}: {at:x2}, want {want:x2}");
                }
    }

    /// <summary>
    /// Zero fills of a q8_0 (and q4_0) cache are byte memsets on the device (CudaKernelOps.TryFill, never the host fallback): the
    /// whole tensor, and the rows from <paramref name="zeroFrom"/> to the capacity of every head (a fresh copy or growth target
    /// whose first rows a copy writes) with the rows before them left as they were. Zero bytes are blocks of d = 0: every value
    /// 0, finite. A non-zero value is declined.
    /// </summary>
    [CudaTheory]
    [InlineData(DType.Q8_0, 4, 520, 256, 0)]
    [InlineData(DType.Q8_0, 4, 520, 256, 257)]
    [InlineData(DType.Q8_0, 2, 40, 128, 33)]
    [InlineData(DType.Q8_0, 1, 1024, 512, 1000)]
    [InlineData(DType.Q4_0, 4, 256, 256, 100)]
    public void BlockQuantizedCacheFillsAreDeviceMemsets(DType dtype, int heads, int cap, int d, int zeroFrom)
    {
        using var allocator = new CudaAllocator();
        long values = (long)heads * cap * d;
        int blockBytes = dtype == DType.Q8_0 ? 34 : 18;
        long rowBytes = d / 32 * blockBytes, bytes = values / 32 * blockBytes;
        var start = new byte[bytes];
        new Random(cap).NextBytes(start);
        using var cache = new Tensor(allocator, dtype, heads, cap, d);
        Assert.Equal(bytes, cache.Storage.ByteLength);
        System.Runtime.InteropServices.Marshal.Copy(start, 0, cache.Storage.PtrAtElementForOverwrite(0), start.Length);
        ((CudaStorage)cache.Storage).EnsureDeviceCurrent();
        Assert.False(CudaKernelOps.TryFill(cache, 1f), "a non-zero fill of a block-quantized tensor has no memset");
        using (Tensor rows = zeroFrom == 0 ? cache.CopyRef() : cache.Narrow(1, zeroFrom, cap - zeroFrom))
            Assert.True(CudaKernelOps.TryFill(rows, 0f), "the fill took the host fallback");
        Assert.False(((CudaStorage)cache.Storage).IsHostDirty);
        allocator.Stream.Synchronize();
        var got = new byte[bytes];
        System.Runtime.InteropServices.Marshal.Copy(cache.Storage.PtrAtElementForRead(0), got, 0, got.Length);
        for (int h = 0; h < heads; h++)
            for (long r = 0; r < cap; r++)
                for (long b = 0; b < rowBytes; b++)
                {
                    long at = (h * (long)cap + r) * rowBytes + b;
                    byte want = r < zeroFrom ? start[at] : (byte)0;
                    if (got[at] != want) Assert.Fail($"head {h} row {r} byte {b}: {got[at]:x2}, want {want:x2} (zero from row {zeroFrom})");
                }
        // Ops.Fill dispatches to the same memsets.
        Ops.Fill(cache, 0f);
        allocator.Stream.Synchronize();
        System.Runtime.InteropServices.Marshal.Copy(cache.Storage.PtrAtElementForRead(0), got, 0, got.Length);
        Assert.True(got.All(x => x == 0), "Ops.Fill left non-zero bytes");
    }
}
