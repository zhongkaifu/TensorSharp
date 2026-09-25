using TensorSharp;
using TensorSharp.GGML;

namespace InferenceWeb.Tests;

/// <summary>
/// Coverage for the Float32 strided path of the GGML Copy op.
///
/// Copy used to walk EVERY strided Float32 tensor one element at a time. The
/// shapes that path actually sees are dominated by views whose innermost run is
/// contiguous on both sides — the prefill QKV split narrows the fused
/// [seq, q|k|v] projection on dim 1, and a KV-cache resize narrows
/// [heads, capacity, head_dim] on the token dim — so those now memcpy per outer
/// index instead. These tests pin the results against an independent
/// element-wise reference for every view shape the codebase produces, including
/// the genuinely element-strided ones (transpose / permute) that must KEEP the
/// element loop.
/// </summary>
public class GgmlCopyStridedFloat32Tests
{
    private static (GgmlAllocator alloc, GgmlContext ctx) NewCpuAllocator()
    {
        var context = new GgmlContext(new[] { 0 }, GgmlBackendType.Cpu);
        return (new GgmlAllocator(context, 0), context);
    }

    private static void Seed(Tensor t)
    {
        // Distinct, exactly-representable value per logical position so any
        // slicing or stride error shows up as a wrong number, not a rounding.
        long n = t.ElementCount();
        var flat = t.Sizes.ToArray();
        int dims = flat.Length;
        var idx = new long[dims];
        for (long i = 0; i < n; i++)
        {
            long rem = i;
            for (int d = dims - 1; d >= 0; d--) { idx[d] = rem % flat[d]; rem /= flat[d]; }
            float v = 0f;
            for (int d = 0; d < dims; d++) v = v * 64f + (idx[d] + 1);
            t.SetElementAsFloat(v * 0.25f, idx);
        }
    }

    /// <summary>Element-wise reference copy, independent of the op under test.</summary>
    private static float[] ReadAll(Tensor t)
    {
        long n = t.ElementCount();
        var sizes = t.Sizes.ToArray();
        int dims = sizes.Length;
        var idx = new long[dims];
        var result = new float[n];
        for (long i = 0; i < n; i++)
        {
            long rem = i;
            for (int d = dims - 1; d >= 0; d--) { idx[d] = rem % sizes[d]; rem /= sizes[d]; }
            result[i] = t.GetElementAsFloat(idx);
        }
        return result;
    }

    [GgmlFact(BackendType.GgmlCpu)]
    public void Copy_NarrowOnInnerDim_MatchesElementwiseReference()
    {
        // The prefill QKV split: fused [seq, qDim+kDim+vDim] narrowed on dim 1.
        // Each row's segment is contiguous; rows are strided by the fused width.
        var (alloc, _) = NewCpuAllocator();
        using var fused = new Tensor(alloc, DType.Float32, 7, 30);
        Seed(fused);

        foreach (var (offset, width) in new[] { (0, 12), (12, 6), (18, 12) })
        {
            using var view = fused.Narrow(1, offset, width);
            using var dst = new Tensor(alloc, DType.Float32, 7, width);
            Ops.Copy(dst, view);

            float[] actual = ReadAll(dst);
            for (int r = 0; r < 7; r++)
                for (int c = 0; c < width; c++)
                    Assert.Equal(fused.GetElementAsFloat(r, offset + c), actual[r * width + c]);
        }
    }

    [GgmlFact(BackendType.GgmlCpu)]
    public void Copy_NarrowOnMiddleDim_KvCacheResizeLayout_MatchesReference()
    {
        // [heads, capacity, head_dim] narrowed on the token dim, into a larger
        // capacity: contiguous per head, strided across heads on both sides.
        const int heads = 3, oldCap = 8, newCap = 16, headDim = 5, seqLen = 5;
        var (alloc, _) = NewCpuAllocator();
        using var oldCache = new Tensor(alloc, DType.Float32, heads, oldCap, headDim);
        using var newCache = new Tensor(alloc, DType.Float32, heads, newCap, headDim);
        Seed(oldCache);
        Ops.Fill(newCache, 0f);

        using (var srcView = oldCache.Narrow(1, 0, seqLen))
        using (var dstView = newCache.Narrow(1, 0, seqLen))
            Ops.Copy(dstView, srcView);

        for (int h = 0; h < heads; h++)
        {
            for (int t = 0; t < seqLen; t++)
                for (int d = 0; d < headDim; d++)
                    Assert.Equal(oldCache.GetElementAsFloat(h, t, d), newCache.GetElementAsFloat(h, t, d));
            // Rows past the copied prefix must be untouched.
            for (int t = seqLen; t < newCap; t++)
                for (int d = 0; d < headDim; d++)
                    Assert.Equal(0f, newCache.GetElementAsFloat(h, t, d));
        }
    }

    [GgmlFact(BackendType.GgmlCpu)]
    public void Copy_TransposedView_StaysOnTheElementPath_AndIsCorrect()
    {
        // A transpose has stride 1 on NEITHER trailing dim of the destination
        // pairing, so the memcpy shortcut must not engage. Correctness is what
        // the test pins; the point is that the fast path did not silently take
        // a view it cannot handle.
        var (alloc, _) = NewCpuAllocator();
        using var src = new Tensor(alloc, DType.Float32, 5, 9);
        Seed(src);

        using var t = src.Transpose();
        using var dst = new Tensor(alloc, DType.Float32, 9, 5);
        Ops.Copy(dst, t);

        for (int r = 0; r < 9; r++)
            for (int c = 0; c < 5; c++)
                Assert.Equal(src.GetElementAsFloat(c, r), dst.GetElementAsFloat(r, c));
    }

    [GgmlFact(BackendType.GgmlCpu)]
    public void Copy_NarrowedOuterDim_ContiguousInner_MatchesReference()
    {
        // Narrow on dim 0 leaves the whole tail contiguous - the fully
        // contiguous-run case the extent scan should collapse to one memcpy.
        var (alloc, _) = NewCpuAllocator();
        using var src = new Tensor(alloc, DType.Float32, 6, 4, 3);
        Seed(src);

        using var view = src.Narrow(0, 2, 3);
        using var dst = new Tensor(alloc, DType.Float32, 3, 4, 3);
        Ops.Copy(dst, view);

        for (int a = 0; a < 3; a++)
            for (int b = 0; b < 4; b++)
                for (int c = 0; c < 3; c++)
                    Assert.Equal(src.GetElementAsFloat(a + 2, b, c), dst.GetElementAsFloat(a, b, c));
    }

    [GgmlFact(BackendType.GgmlCpu)]
    public void Copy_ContiguousToContiguous_IsUnchanged()
    {
        var (alloc, _) = NewCpuAllocator();
        using var src = new Tensor(alloc, DType.Float32, 4, 7);
        using var dst = new Tensor(alloc, DType.Float32, 4, 7);
        Seed(src);
        Ops.Copy(dst, src);
        Assert.Equal(ReadAll(src), ReadAll(dst));
    }
}
