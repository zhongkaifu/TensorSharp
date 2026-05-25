using AdvUtils;
using TensorSharp;
using TensorSharp.Cpu;

namespace InferenceWeb.Tests;

/// <summary>
/// Exercises the CPU and (managed-CPU-backed) GGML Gather / Scatter / ScatterAdd / ScatterFill
/// fast paths added for 2D contiguous tensors. Each scenario is compared against a
/// hand-rolled scalar reference implementation to confirm the new path produces
/// byte-identical results across (dim=0, dim=1) and the "uniform-row index" memcpy shortcut.
/// </summary>
public class GatherScatterTests
{
    private readonly IAllocator _allocator = new CpuAllocator(BlasEnum.DotNet);

    private static Tensor MakeIndicesTensor(IAllocator allocator, int rows, int cols, float[] flat)
    {
        var t = new Tensor(allocator, DType.Float32, rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                t.SetElementAsFloat(flat[i * cols + j], i, j);
        return t;
    }

    private static float[] GatherRef(float[,] src, int[,] indices, int dim)
    {
        int N = indices.GetLength(0);
        int D = indices.GetLength(1);
        float[] result = new float[N * D];
        for (int i = 0; i < N; i++)
            for (int j = 0; j < D; j++)
            {
                int idx = indices[i, j];
                result[i * D + j] = dim == 0 ? src[idx, j] : src[i, idx];
            }
        return result;
    }

    [Fact]
    public void Gather_2DContig_Dim0_UniformIndices_MatchesReference()
    {
        // Embedding-style: all indices in a row are equal -> exercises memcpy fast path.
        float[,] src = new float[5, 4];
        for (int i = 0; i < 5; i++)
            for (int j = 0; j < 4; j++)
                src[i, j] = i * 10 + j;

        int[,] indices = new int[,] { { 3, 3, 3, 3 }, { 0, 0, 0, 0 }, { 4, 4, 4, 4 } };
        float[] idxFlat = { 3, 3, 3, 3, 0, 0, 0, 0, 4, 4, 4, 4 };

        using var srcT = Tensor.FromArray(_allocator, src);
        using var idxT = MakeIndicesTensor(_allocator, 3, 4, idxFlat);
        using var resT = Ops.Gather(null, srcT, 0, idxT);

        float[] expected = GatherRef(src, indices, 0);
        float[] actual = resT.GetElementsAsFloat(12);
        Assert.Equal(expected, actual);
    }

    [Fact]
    public void Gather_2DContig_Dim0_VariedIndices_MatchesReference()
    {
        // Non-uniform indices per row -> exercises column-wise fast path.
        float[,] src = new float[5, 4];
        for (int i = 0; i < 5; i++)
            for (int j = 0; j < 4; j++)
                src[i, j] = i * 10 + j;

        int[,] indices = new int[,] { { 0, 1, 2, 3 }, { 4, 3, 2, 1 } };
        float[] idxFlat = { 0, 1, 2, 3, 4, 3, 2, 1 };

        using var srcT = Tensor.FromArray(_allocator, src);
        using var idxT = MakeIndicesTensor(_allocator, 2, 4, idxFlat);
        using var resT = Ops.Gather(null, srcT, 0, idxT);

        float[] expected = GatherRef(src, indices, 0);
        float[] actual = resT.GetElementsAsFloat(8);
        Assert.Equal(expected, actual);
    }

    [Fact]
    public void Gather_2DContig_Dim1_MatchesReference()
    {
        // dim=1: result[i, j] = src[i, indices[i, j]]
        float[,] src = new float[3, 5];
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 5; j++)
                src[i, j] = i * 10 + j;

        int[,] indices = new int[,] { { 4, 0, 2 }, { 1, 3, 1 }, { 0, 4, 2 } };
        float[] idxFlat = { 4, 0, 2, 1, 3, 1, 0, 4, 2 };

        using var srcT = Tensor.FromArray(_allocator, src);
        using var idxT = MakeIndicesTensor(_allocator, 3, 3, idxFlat);
        using var resT = Ops.Gather(null, srcT, 1, idxT);

        float[] expected = GatherRef(src, indices, 1);
        float[] actual = resT.GetElementsAsFloat(9);
        Assert.Equal(expected, actual);
    }

    [Fact]
    public void Scatter_2DContig_Dim0_VariedIndices_MatchesReference()
    {
        // result[indices[i, j], j] = src[i, j]
        float[,] srcVals = new float[2, 4];
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 4; j++)
                srcVals[i, j] = (i + 1) * 100 + j;

        int[,] indices = new int[,] { { 0, 1, 2, 3 }, { 4, 3, 2, 1 } };
        float[] idxFlat = { 0, 1, 2, 3, 4, 3, 2, 1 };

        // Reference
        float[] expected = new float[5 * 4];
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 4; j++)
                expected[indices[i, j] * 4 + j] = srcVals[i, j];

        using var srcT = Tensor.FromArray(_allocator, srcVals);
        using var idxT = MakeIndicesTensor(_allocator, 2, 4, idxFlat);
        using var resT = new Tensor(_allocator, DType.Float32, 5, 4);
        Ops.Fill(resT, 0f);
        Ops.Scatter(resT, srcT, 0, idxT);

        float[] actual = resT.GetElementsAsFloat(20);
        Assert.Equal(expected, actual);
    }

    [Fact]
    public void Scatter_2DContig_Dim1_MatchesReference()
    {
        // result[i, indices[i, j]] = src[i, j]
        float[,] srcVals = new float[3, 3];
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                srcVals[i, j] = (i + 1) * 100 + j;

        int[,] indices = new int[,] { { 4, 0, 2 }, { 1, 3, 0 }, { 2, 4, 1 } };
        float[] idxFlat = { 4, 0, 2, 1, 3, 0, 2, 4, 1 };

        float[] expected = new float[3 * 5];
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                expected[i * 5 + indices[i, j]] = srcVals[i, j];

        using var srcT = Tensor.FromArray(_allocator, srcVals);
        using var idxT = MakeIndicesTensor(_allocator, 3, 3, idxFlat);
        using var resT = new Tensor(_allocator, DType.Float32, 3, 5);
        Ops.Fill(resT, 0f);
        Ops.Scatter(resT, srcT, 1, idxT);

        float[] actual = resT.GetElementsAsFloat(15);
        Assert.Equal(expected, actual);
    }

    [Fact]
    public void ScatterAdd_2DContig_Dim0_VariedIndices_MatchesReference()
    {
        float[,] srcVals = new float[2, 4];
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 4; j++)
                srcVals[i, j] = 1f;

        int[,] indices = new int[,] { { 0, 1, 0, 1 }, { 1, 0, 1, 0 } };
        float[] idxFlat = { 0, 1, 0, 1, 1, 0, 1, 0 };

        float[] expected = new float[2 * 4];
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 4; j++)
                expected[indices[i, j] * 4 + j] += srcVals[i, j];

        using var srcT = Tensor.FromArray(_allocator, srcVals);
        using var idxT = MakeIndicesTensor(_allocator, 2, 4, idxFlat);
        using var resT = new Tensor(_allocator, DType.Float32, 2, 4);
        Ops.Fill(resT, 0f);
        Ops.ScatterAdd(resT, srcT, 0, idxT);

        float[] actual = resT.GetElementsAsFloat(8);
        Assert.Equal(expected, actual);
    }

    [Fact]
    public void ScatterAdd_2DContig_Dim0_UniformIndices_AccumulatesViaVectorPath()
    {
        // Each row's indices are all equal so the uniform-row vectorized add fires.
        // Use D >= Vector<float>.Count to exercise the SIMD loop.
        int N = 4;
        int D = 32;
        float[,] srcVals = new float[N, D];
        for (int i = 0; i < N; i++)
            for (int j = 0; j < D; j++)
                srcVals[i, j] = i + 1;

        int[,] indices = new int[N, D];
        for (int i = 0; i < N; i++)
            for (int j = 0; j < D; j++)
                indices[i, j] = i % 2; // rows 0,2 -> idx 0; rows 1,3 -> idx 1

        float[] idxFlat = new float[N * D];
        for (int i = 0; i < N; i++)
            for (int j = 0; j < D; j++)
                idxFlat[i * D + j] = indices[i, j];

        float[] expected = new float[2 * D];
        for (int i = 0; i < N; i++)
            for (int j = 0; j < D; j++)
                expected[indices[i, j] * D + j] += srcVals[i, j];

        using var srcT = Tensor.FromArray(_allocator, srcVals);
        using var idxT = MakeIndicesTensor(_allocator, N, D, idxFlat);
        using var resT = new Tensor(_allocator, DType.Float32, 2, D);
        Ops.Fill(resT, 0f);
        Ops.ScatterAdd(resT, srcT, 0, idxT);

        float[] actual = resT.GetElementsAsFloat(2 * D);
        Assert.Equal(expected, actual);
    }

    [Fact]
    public void ScatterFill_2DContig_Dim0_MatchesReference()
    {
        int[,] indices = new int[,] { { 0, 1, 2 }, { 2, 1, 0 } };
        float[] idxFlat = { 0, 1, 2, 2, 1, 0 };

        float[] expected = new float[3 * 3];
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 3; j++)
                expected[indices[i, j] * 3 + j] = 7.5f;

        using var idxT = MakeIndicesTensor(_allocator, 2, 3, idxFlat);
        using var resT = new Tensor(_allocator, DType.Float32, 3, 3);
        Ops.Fill(resT, 0f);
        Ops.ScatterFill(resT, 7.5f, 0, idxT);

        float[] actual = resT.GetElementsAsFloat(9);
        Assert.Equal(expected, actual);
    }

    [Fact]
    public void ScatterFill_2DContig_Dim1_MatchesReference()
    {
        int[,] indices = new int[,] { { 4, 0, 2 }, { 1, 3, 0 }, { 2, 4, 1 } };
        float[] idxFlat = { 4, 0, 2, 1, 3, 0, 2, 4, 1 };

        float[] expected = new float[3 * 5];
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                expected[i * 5 + indices[i, j]] = -1.25f;

        using var idxT = MakeIndicesTensor(_allocator, 3, 3, idxFlat);
        using var resT = new Tensor(_allocator, DType.Float32, 3, 5);
        Ops.Fill(resT, 0f);
        Ops.ScatterFill(resT, -1.25f, 1, idxT);

        float[] actual = resT.GetElementsAsFloat(15);
        Assert.Equal(expected, actual);
    }

    [Fact]
    public void Gather_2DContig_Dim0_OutOfRangeIndex_Throws()
    {
        // Verifies the fast path still validates indices.
        float[,] src = new float[3, 2] { { 0, 1 }, { 2, 3 }, { 4, 5 } };
        float[] idxFlat = { 0, 0, 99, 99 }; // out-of-range row in second row

        using var srcT = Tensor.FromArray(_allocator, src);
        using var idxT = MakeIndicesTensor(_allocator, 2, 2, idxFlat);

        Assert.Throws<IndexOutOfRangeException>(() => Ops.Gather(null, srcT, 0, idxT));
    }
}
