using System.Runtime.InteropServices;
using TensorSharp;
using TensorSharp.GGML;

namespace InferenceWeb.Tests;

// Coverage for the GGML Copy op on the non-Float32 KV-cache layouts the
// model-level EnsureCacheCapacity path drives through Ops.Copy. The shape we
// exercise here mirrors that path exactly: [heads, capacity, head_dim] caches
// narrowed on the token dim to [heads, cacheSeqLen, head_dim], which leaves
// the inner head_dim*cacheSeqLen slice contiguous per head but the outer head
// stride untouched at the original capacity*head_dim. Pre-fix this combination
// tripped the "copy expects Float32 tensors only" guard and the strided
// fallback's float* cursor.
public class GgmlCopyDtypeTests
{
    [Fact]
    public void Copy_Float16_ContiguousTensors_RoundTripsExactBits()
    {
        var context = new GgmlContext(new[] { 0 }, GgmlBackendType.Cpu);
        var allocator = new GgmlAllocator(context, 0);

        using var src = new Tensor(allocator, DType.Float16, 2, 3, 4);
        using var dst = new Tensor(allocator, DType.Float16, 2, 3, 4);

        float[] expected = SeedHalfTensor(src, baseValue: 0.125f);
        Ops.Copy(dst, src);

        float[] actual = ReadHalfTensorAsFloats(dst, expected.Length);
        Assert.Equal(expected, actual);
    }

    [Fact]
    public void Copy_Float16_NarrowedKvCacheLayout_PreservesEachHeadIndependently()
    {
        const int heads = 3;
        const int oldCapacity = 8;
        const int newCapacity = 16;
        const int headDim = 5;
        const int cacheSeqLen = 5;

        var context = new GgmlContext(new[] { 0 }, GgmlBackendType.Cpu);
        var allocator = new GgmlAllocator(context, 0);

        using var oldCache = new Tensor(allocator, DType.Float16, heads, oldCapacity, headDim);
        using var newCache = new Tensor(allocator, DType.Float16, heads, newCapacity, headDim);

        // Distinct seed per head/token/dim so any byte-level slicing error is
        // visible: head index dominates, then token, then dim — matches the
        // contiguous order in storage.
        for (int h = 0; h < heads; h++)
            for (int t = 0; t < oldCapacity; t++)
                for (int d = 0; d < headDim; d++)
                    oldCache.SetElementAsFloat((float)((h + 1) * 1000 + t * 10 + d) * 0.0078125f, h, t, d);

        // Zero-fill the destination via SetElementAsFloat — Ops.Fill on a
        // strided F16 tensor isn't supported here and we want a clean baseline
        // we can verify against.
        for (int h = 0; h < heads; h++)
            for (int t = 0; t < newCapacity; t++)
                for (int d = 0; d < headDim; d++)
                    newCache.SetElementAsFloat(0f, h, t, d);

        using var src = oldCache.Narrow(1, 0, cacheSeqLen);
        using var dst = newCache.Narrow(1, 0, cacheSeqLen);
        Ops.Copy(dst, src);

        for (int h = 0; h < heads; h++)
        {
            for (int t = 0; t < cacheSeqLen; t++)
            {
                for (int d = 0; d < headDim; d++)
                {
                    float expected = (float)((System.Half)((float)((h + 1) * 1000 + t * 10 + d) * 0.0078125f));
                    Assert.Equal(expected, newCache.GetElementAsFloat(h, t, d));
                }
            }
            // Untouched tail of the destination stays at zero — this verifies
            // we didn't run off the end of the per-head slice.
            for (int t = cacheSeqLen; t < newCapacity; t++)
                for (int d = 0; d < headDim; d++)
                    Assert.Equal(0f, newCache.GetElementAsFloat(h, t, d));
        }
    }

    [Fact]
    public void Copy_CrossDtype_Throws()
    {
        var context = new GgmlContext(new[] { 0 }, GgmlBackendType.Cpu);
        var allocator = new GgmlAllocator(context, 0);

        using var src = new Tensor(allocator, DType.Float32, 4);
        using var dst = new Tensor(allocator, DType.Float16, 4);

        Assert.Throws<System.InvalidOperationException>(() => Ops.Copy(dst, src));
    }

    private static float[] SeedHalfTensor(Tensor t, float baseValue)
    {
        long total = 1;
        for (int d = 0; d < t.DimensionCount; d++)
            total *= t.Sizes[d];
        float[] values = new float[total];
        for (long i = 0; i < total; i++)
            values[i] = baseValue * (i + 1) - baseValue * 0.5f;
        // Index linearly through the contiguous storage.
        unsafe
        {
            ushort* p = (ushort*)t.Storage.PtrAtElement(t.StorageOffset).ToPointer();
            for (long i = 0; i < total; i++)
            {
                System.Half h = (System.Half)values[i];
                p[i] = System.BitConverter.HalfToUInt16Bits(h);
                values[i] = (float)h;     // round-trip to F16 so the assertion is exact
            }
        }
        return values;
    }

    private static float[] ReadHalfTensorAsFloats(Tensor t, long length)
    {
        float[] result = new float[length];
        unsafe
        {
            ushort* p = (ushort*)t.Storage.PtrAtElement(t.StorageOffset).ToPointer();
            for (long i = 0; i < length; i++)
                result[i] = (float)System.BitConverter.UInt16BitsToHalf(p[i]);
        }
        return result;
    }
}
