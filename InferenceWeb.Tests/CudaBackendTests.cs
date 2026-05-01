using TensorSharp;
using TensorSharp.Cuda;
using TensorSharp.Models;
using System.Runtime.InteropServices;

namespace InferenceWeb.Tests;

public class CudaBackendTests
{
    [Fact]
    public void CudaAddmm_MatchesCpuForContiguousRhs()
    {
        if (!CudaBackend.IsAvailable())
            return;

        using var allocator = new CudaAllocator();
        using var a = Tensor.FromArray(allocator, new float[,] { { 1, 2, 3 }, { 4, 5, 6 } });
        using var b = Tensor.FromArray(allocator, new float[,] { { 7, 8 }, { 9, 10 }, { 11, 12 } });
        using var c = new Tensor(allocator, DType.Float32, 2, 2);

        Ops.Addmm(c, 0, c, 1, a, b);

        AssertClose(new[] { 58f, 64f, 139f, 154f }, c.GetElementsAsFloat(4));
    }

    [Fact]
    public void CudaStagedHostEmbeddingCopy_PreservesDeviceResidentTensorData()
    {
        if (!CudaBackend.IsAvailable())
            return;

        using var allocator = new CudaAllocator();
        using var tensor = new Tensor(allocator, DType.Float32, 4, 4);
        FillSequential(tensor, scale: 1f, offset: 0f);
        Ops.Mul(tensor, tensor, 2f);

        using var staged = new Tensor(allocator, DType.Float32, 1, 4);
        staged.SetElementsAsFloat(new[] { 100f, 101f, 102f, 103f });
        using (var row = tensor.Narrow(0, 1, 1))
            Ops.Copy(row, staged);

        AssertClose(new[]
        {
            0f, 2f, 4f, 6f,
            100f, 101f, 102f, 103f,
            16f, 18f, 20f, 22f,
            24f, 26f, 28f, 30f,
        }, tensor.GetElementsAsFloat(16));
    }

    [Fact]
    public void CudaAddmm_MatchesCpuForTransposedWeightView()
    {
        if (!CudaBackend.IsAvailable())
            return;

        using var allocator = new CudaAllocator();
        using var input = Tensor.FromArray(allocator, new float[,] { { 1, 2, 3 }, { 4, 5, 6 } });
        using var weights = Tensor.FromArray(allocator, new float[,] { { 7, 9, 11 }, { 8, 10, 12 } });
        using var weightsT = weights.Transpose();
        using var output = new Tensor(allocator, DType.Float32, 2, 2);

        Ops.Addmm(output, 0, output, 1, input, weightsT);

        AssertClose(new[] { 58f, 64f, 139f, 154f }, output.GetElementsAsFloat(4));
    }

    [Fact]
    public void CudaAddmmBatch_MatchesCpuForTransposedRightOperand()
    {
        if (!CudaBackend.IsAvailable())
            return;

        const int batch = 3;
        const int rows = 5;
        const int shared = 7;
        const int cols = 4;

        using var allocator = new CudaAllocator();
        using var lhs = new Tensor(allocator, DType.Float32, batch, rows, shared);
        using var rhs = new Tensor(allocator, DType.Float32, batch, cols, shared);
        FillSequential(lhs, scale: 0.125f, offset: -1.5f);
        FillSequential(rhs, scale: -0.0625f, offset: 2.0f);

        using var rhsT = rhs.Transpose(1, 2);
        using var output = new Tensor(allocator, DType.Float32, batch, rows, cols);
        Ops.AddmmBatch(output, 0, output, 1, lhs, rhsT);

        float[] actual = output.GetElementsAsFloat((int)output.ElementCount());
        float[] expected = new float[actual.Length];
        float[] lhsData = lhs.GetElementsAsFloat((int)lhs.ElementCount());
        float[] rhsData = rhs.GetElementsAsFloat((int)rhs.ElementCount());

        for (int b = 0; b < batch; b++)
        {
            for (int r = 0; r < rows; r++)
            {
                for (int c = 0; c < cols; c++)
                {
                    float sum = 0;
                    for (int k = 0; k < shared; k++)
                    {
                        float l = lhsData[(b * rows + r) * shared + k];
                        float rr = rhsData[(b * cols + c) * shared + k];
                        sum += l * rr;
                    }
                    expected[(b * rows + r) * cols + c] = sum;
                }
            }
        }

        AssertClose(expected, actual, 1e-4f);
    }

    [Fact]
    public void CudaAddmmBatch_LargeAttentionShape_MatchesCpuForTransposedRightOperand()
    {
        if (!CudaBackend.IsAvailable())
            return;

        const int batch = 2;
        const int rows = 257;
        const int shared = 64;
        const int cols = 263;

        using var allocator = new CudaAllocator();
        using var lhs = new Tensor(allocator, DType.Float32, batch, rows, shared);
        using var rhs = new Tensor(allocator, DType.Float32, batch, cols, shared);
        FillSinusoidal(lhs, 0.013f);
        FillSinusoidal(rhs, -0.011f);

        using var rhsT = rhs.Transpose(1, 2);
        using var output = new Tensor(allocator, DType.Float32, batch, rows, cols);
        Ops.AddmmBatch(output, 0, output, 1, lhs, rhsT);

        float[] actual = output.GetElementsAsFloat((int)output.ElementCount());
        float[] lhsData = lhs.GetElementsAsFloat((int)lhs.ElementCount());
        float[] rhsData = rhs.GetElementsAsFloat((int)rhs.ElementCount());

        for (int b = 0; b < batch; b++)
        {
            for (int r = 0; r < rows; r += 31)
            {
                for (int c = 0; c < cols; c += 37)
                {
                    float expected = 0;
                    for (int k = 0; k < shared; k++)
                    {
                        float l = lhsData[(b * rows + r) * shared + k];
                        float rr = rhsData[(b * cols + c) * shared + k];
                        expected += l * rr;
                    }

                    Assert.InRange(MathF.Abs(expected - actual[(b * rows + r) * cols + c]), 0.0f, 5e-4f);
                }
            }
        }
    }

    [Fact]
    public void CudaSoftmax_WithNegativeInfinityMask_MatchesExpected()
    {
        if (!CudaBackend.IsAvailable())
            return;

        using var allocator = new CudaAllocator();
        using var logits = Tensor.FromArray(allocator, new float[,]
        {
            { 1, float.NegativeInfinity, 3, float.NegativeInfinity },
            { float.NegativeInfinity, -2, -1, float.NegativeInfinity },
        });

        using var probs = Ops.Softmax(null, logits);
        float[] actual = probs.GetElementsAsFloat(8);

        AssertClose(new[] { 1f / (1f + MathF.Exp(2)), 0f, MathF.Exp(2) / (1f + MathF.Exp(2)), 0f },
            actual[..4], 1e-5f);
        AssertClose(new[] { 0f, 1f / (1f + MathF.Exp(1)), MathF.Exp(1) / (1f + MathF.Exp(1)), 0f },
            actual[4..], 1e-5f);
    }

    [Fact]
    public void CudaSoftmax_LargeRows_MatchesReference()
    {
        if (!CudaBackend.IsAvailable())
            return;

        const int rows = 3;
        const int cols = 2304;
        float[,] values = new float[rows, cols];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                values[r, c] = MathF.Sin((r + 1) * (c + 3) * 0.017f) * 4f - c * 0.0001f;

        using var allocator = new CudaAllocator();
        using var logits = Tensor.FromArray(allocator, values);
        using var probs = Ops.Softmax(null, logits);
        float[] actual = probs.GetElementsAsFloat(rows * cols);

        for (int r = 0; r < rows; r++)
        {
            float max = float.NegativeInfinity;
            for (int c = 0; c < cols; c++)
                max = MathF.Max(max, values[r, c]);
            float sum = 0;
            for (int c = 0; c < cols; c++)
                sum += MathF.Exp(values[r, c] - max);
            for (int c = 0; c < cols; c++)
            {
                float expected = MathF.Exp(values[r, c] - max) / sum;
                Assert.InRange(MathF.Abs(expected - actual[r * cols + c]), 0.0f, 1e-6f);
            }
        }
    }

    [Fact]
    public void CudaFallbackOps_PreserveTensorSemantics()
    {
        if (!CudaBackend.IsAvailable())
            return;

        using var allocator = new CudaAllocator();
        using var x = Tensor.FromArray(allocator, new float[,] { { 1, 2, 3 }, { 4, 5, 6 } });
        using var y = new Tensor(allocator, DType.Float32, 2, 3);

        Ops.Add(y, x, 2.0f);
        Ops.SiLU(y, y);

        float[] actual = y.GetElementsAsFloat(6);
        for (int i = 0; i < actual.Length; i++)
        {
            float v = i + 3;
            float expected = v / (1.0f + MathF.Exp(-v));
            Assert.InRange(MathF.Abs(actual[i] - expected), 0.0f, 1e-5f);
        }
    }

    [Fact]
    public void CudaElementwiseAndScalarOps_MatchExpected()
    {
        if (!CudaBackend.IsAvailable())
            return;

        using var allocator = new CudaAllocator();
        using var x = Tensor.FromArray(allocator, new float[,] { { 1, -2, 3 }, { -4, 5, -6 } });
        using var y = Tensor.FromArray(allocator, new float[,] { { 0.5f, 2, -1 }, { 4, -0.25f, 3 } });
        using var z = Tensor.FromArray(allocator, new float[,] { { 2, -3, 0.5f }, { 1.5f, -2, 4 } });
        using var w = Tensor.FromArray(allocator, new float[,] { { -1, 0.25f, 2 }, { -0.5f, 3, -2 } });

        using var add = Ops.Add(null, x, y);
        AssertClose(new[] { 1.5f, 0f, 2f, 0f, 4.75f, -3f }, add.GetElementsAsFloat(6));

        using var mul = Ops.Mul(null, x, y);
        AssertClose(new[] { 0.5f, -4f, -3f, -16f, -1.25f, -18f }, mul.GetElementsAsFloat(6));

        using var scaled = Ops.Mul(null, x, 0.25f);
        AssertClose(new[] { 0.25f, -0.5f, 0.75f, -1f, 1.25f, -1.5f }, scaled.GetElementsAsFloat(6));

        using var reverseSub = Ops.Sub(null, 10f, x);
        AssertClose(new[] { 9f, 12f, 7f, 14f, 5f, 16f }, reverseSub.GetElementsAsFloat(6));

        using var reverseDiv = Ops.Div(null, 12f, x);
        AssertClose(new[] { 12f, -6f, 4f, -3f, 2.4f, -2f }, reverseDiv.GetElementsAsFloat(6), 1e-5f);

        using var addMul = Ops.AddMul(null, x, y, z);
        AssertClose(new[] { 2f, -8f, 2.5f, 2f, 5.5f, 6f }, addMul.GetElementsAsFloat(6), 1e-5f);

        using var addDiv = Ops.AddDiv(null, x, y, z);
        AssertClose(new[] { 1.25f, -2.6666667f, 1f, -1.3333333f, 5.125f, -5.25f }, addDiv.GetElementsAsFloat(6), 1e-5f);

        using var addMulV = Ops.AddMulV(null, x, y, 0.5f);
        AssertClose(new[] { 1.25f, -1f, 2.5f, -2f, 4.875f, -4.5f }, addMulV.GetElementsAsFloat(6), 1e-5f);

        using var mulMulAdd = Ops.MulMulAdd(null, x, y, z, w);
        AssertClose(new[] { -1.5f, -4.75f, -2f, -16.75f, -7.25f, -26f }, mulMulAdd.GetElementsAsFloat(6), 1e-5f);
    }

    [Fact]
    public void CudaRmsNormSoftmaxAndIndexSelect_MatchExpected()
    {
        if (!CudaBackend.IsAvailable())
            return;

        using var allocator = new CudaAllocator();
        using var x = Tensor.FromArray(allocator, new float[,] { { 1, 2, 3, 4 }, { -1, -2, -3, -4 } });
        using var alpha = Tensor.FromArray(allocator, new float[] { 1, 0.5f, 2, -1 });
        using var norm = Ops.RMSNorm(null, x, alpha, null, 1e-5f);

        float[] actualNorm = norm.GetElementsAsFloat(8);
        float inv0 = 1.0f / MathF.Sqrt((1 + 4 + 9 + 16) / 4.0f + 1e-5f);
        AssertClose(new[] { 1 * inv0, 2 * inv0 * 0.5f, 3 * inv0 * 2, 4 * inv0 * -1 }, actualNorm[..4], 1e-4f);

        using var logits = Tensor.FromArray(allocator, new float[,] { { 1, 2, 3 }, { -1, -2, -3 } });
        using var probs = Ops.Softmax(null, logits);
        float[] actualProbs = probs.GetElementsAsFloat(6);
        AssertClose(SoftmaxRow(1, 2, 3), actualProbs[..3], 1e-5f);
        AssertClose(SoftmaxRow(-1, -2, -3), actualProbs[3..], 1e-5f);

        using var indices = Tensor.FromArray(allocator, new int[] { 1, 0 });
        using var selected = Ops.IndexSelect(null, x, indices);
        AssertClose(new[] { -1f, -2f, -3f, -4f, 1f, 2f, 3f, 4f }, selected.GetElementsAsFloat(8));
    }

    [Fact]
    public void CudaRoPE_MatchesReference()
    {
        if (!CudaBackend.IsAvailable())
            return;

        using var allocator = new CudaAllocator();
        using var x = Tensor.FromArray(allocator, new float[,] { { 1, 2, 3, 4 }, { 5, 6, 7, 8 } });
        using var rope = Ops.RoPE(null, x, seqLen: 2, rowOffset: 1);

        float[] expected = new float[8];
        float[] input = { 1, 2, 3, 4, 5, 6, 7, 8 };
        for (int row = 0; row < 2; row++)
        {
            int m = row + 1;
            for (int pair = 0; pair < 2; pair++)
            {
                float theta = MathF.Pow(500000.0f, -2.0f * pair / 4.0f);
                float c = MathF.Cos(theta * m);
                float s = MathF.Sin(theta * m);
                float left = input[row * 4 + pair * 2];
                float right = input[row * 4 + pair * 2 + 1];
                expected[row * 4 + pair * 2] = left * c - right * s;
                expected[row * 4 + pair * 2 + 1] = right * c + left * s;
            }
        }

        AssertClose(expected, rope.GetElementsAsFloat(8), 1e-4f);
    }

    [Fact]
    public void CudaQuantizedMatmulAndRows_Q8_0MatchExpected()
    {
        if (!CudaBackend.IsAvailable())
            return;

        byte[] weights = CreateQ8_0Rows(new[]
        {
            Enumerable.Range(1, 32).Select(i => (sbyte)i).ToArray(),
            Enumerable.Range(1, 32).Select(i => (sbyte)(-i)).ToArray(),
            Enumerable.Range(0, 32).Select(i => (sbyte)(i % 5)).ToArray(),
        });

        IntPtr host = Marshal.AllocHGlobal(weights.Length);
        try
        {
            Marshal.Copy(weights, 0, host, weights.Length);

            using var allocator = new CudaAllocator();
            using var input = Tensor.FromArray(allocator, CreateQuantInput());
            using var output = new Tensor(allocator, DType.Float32, 2, 3);

            Assert.True(CudaQuantizedOps.TryAddmmQuantizedToFloat32(output, input, host, host, 8, 32, 3, weights.Length));
            AssertClose(
                new[] { 528f, -528f, 61f, 11440f, -11440f, 1022f },
                output.GetElementsAsFloat(6),
                1e-3f);

            using var indices = Tensor.FromArray(allocator, new int[] { 2, 0 });
            using var rows = new Tensor(allocator, DType.Float32, 2, 32);
            Assert.True(CudaQuantizedOps.TryGetRowsQuantizedToFloat32(rows, host, host, 8, 32, 3, weights.Length, indices));
            float[] actualRows = rows.GetElementsAsFloat(64);
            Assert.Equal(0f, actualRows[0]);
            Assert.Equal(4f, actualRows[4]);
            Assert.Equal(1f, actualRows[32]);
            Assert.Equal(32f, actualRows[63]);

            CudaQuantizedOps.ReleaseQuantizedWeight(allocator, host);
        }
        finally
        {
            Marshal.FreeHGlobal(host);
        }
    }

    [Fact]
    public void CudaQuantizedWeight_CanRunAfterHostCopyReleased()
    {
        if (!CudaBackend.IsAvailable())
            return;

        byte[] weights = CreateQ8_0Rows(new[]
        {
            Enumerable.Range(1, 32).Select(i => (sbyte)i).ToArray(),
            Enumerable.Range(1, 32).Select(i => (sbyte)(-i)).ToArray(),
        });

        using var qw = new QuantizedWeight(weights, 8, 32, 2);
        using var allocator = new CudaAllocator();
        IntPtr cacheKey = qw.EnsureDeviceCacheKey();
        CudaQuantizedOps.PreloadQuantizedWeight(allocator, cacheKey, qw.Data, qw.GgmlType, qw.Ne0, qw.Ne1, qw.RawBytes);
        qw.ReleaseHostData();
        Assert.False(qw.HasHostData);

        using var input = Tensor.FromArray(allocator, CreateQuantInput());
        using var output = new Tensor(allocator, DType.Float32, 2, 2);
        Assert.True(CudaQuantizedOps.TryAddmmQuantizedToFloat32(output, input, cacheKey, IntPtr.Zero, 8, 32, 2, weights.Length));
        AssertClose(new[] { 528f, -528f, 11440f, -11440f }, output.GetElementsAsFloat(4), 1e-3f);

        using var indices = Tensor.FromArray(allocator, new int[] { 1, 0 });
        using var rows = new Tensor(allocator, DType.Float32, 2, 32);
        Assert.True(CudaQuantizedOps.TryGetRowsQuantizedToFloat32(rows, cacheKey, IntPtr.Zero, 8, 32, 2, weights.Length, indices));
        float[] actualRows = rows.GetElementsAsFloat(64);
        Assert.Equal(-1f, actualRows[0]);
        Assert.Equal(-32f, actualRows[31]);
        Assert.Equal(1f, actualRows[32]);
        Assert.Equal(32f, actualRows[63]);

        CudaQuantizedOps.ReleaseQuantizedWeight(allocator, cacheKey);
    }

    [Fact]
    public void CudaQuantizedMatmul_Q8_0WithMultipleBlocksAndScales_MatchesDequantizedReference()
    {
        if (!CudaBackend.IsAvailable())
            return;

        const int inDim = 64;
        const int outDim = 5;
        const int rows = 3;
        byte[] weights = CreateQ8_0Rows(outDim, inDim, (r, c) => (sbyte)(((r + 3) * (c - 17)) % 63), r => 0.25f + r * 0.125f);
        float[,] input = new float[rows, inDim];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < inDim; c++)
                input[r, c] = MathF.Sin((r + 1) * (c + 1) * 0.1f);

        IntPtr host = Marshal.AllocHGlobal(weights.Length);
        try
        {
            Marshal.Copy(weights, 0, host, weights.Length);
            using var allocator = new CudaAllocator();
            using var inputTensor = Tensor.FromArray(allocator, input);
            using var output = new Tensor(allocator, DType.Float32, rows, outDim);

            Assert.True(CudaQuantizedOps.TryAddmmQuantizedToFloat32(output, inputTensor, host, host, 8, inDim, outDim, weights.Length));

            float[] expected = DequantizedMatmulQ80(weights, outDim, inDim, input, rows);
            AssertClose(expected, output.GetElementsAsFloat(rows * outDim), 1e-3f);

            CudaQuantizedOps.ReleaseQuantizedWeight(allocator, host);
        }
        finally
        {
            Marshal.FreeHGlobal(host);
        }
    }

    [Fact]
    public void CudaQuantizedMatmul_Q4_0WithMultipleOutputColumns_MatchesDequantizedReference()
    {
        if (!CudaBackend.IsAvailable())
            return;

        const int inDim = 64;
        const int outDim = 7;
        const int rows = 3;
        byte[] weights = CreateQ4_0Rows(outDim, inDim, (r, c) => (sbyte)(((r * 5 + c * 3) % 16) - 8), r => 0.125f + r * 0.03125f);
        float[,] input = new float[rows, inDim];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < inDim; c++)
                input[r, c] = MathF.Cos((r + 2) * (c + 1) * 0.037f) * 0.75f;

        IntPtr host = Marshal.AllocHGlobal(weights.Length);
        try
        {
            Marshal.Copy(weights, 0, host, weights.Length);
            using var allocator = new CudaAllocator();
            using var inputTensor = Tensor.FromArray(allocator, input);
            using var output = new Tensor(allocator, DType.Float32, rows, outDim);

            Assert.True(CudaQuantizedOps.TryAddmmQuantizedToFloat32(output, inputTensor, host, host, 2, inDim, outDim, weights.Length));

            float[] expected = DequantizedMatmulQ40(weights, outDim, inDim, input, rows);
            AssertClose(expected, output.GetElementsAsFloat(rows * outDim), 1e-3f);

            CudaQuantizedOps.ReleaseQuantizedWeight(allocator, host);
        }
        finally
        {
            Marshal.FreeHGlobal(host);
        }
    }

    [Fact]
    public void CudaScaledDotProductAttention_WithMaskMatchesReference()
    {
        if (!CudaBackend.IsAvailable())
            return;

        const int batch = 2;
        const int seqQ = 3;
        const int seqK = 4;
        const int heads = 2;
        const int keyDim = 5;
        const int valueDim = 6;
        const float scale = 0.37f;

        float[,,,] q = new float[batch, seqQ, heads, keyDim];
        float[,,,] k = new float[batch, seqK, heads, keyDim];
        float[,,,] v = new float[batch, seqK, heads, valueDim];
        float[,,,] mask = new float[batch, heads, seqQ, seqK];

        for (int b = 0; b < batch; b++)
            for (int t = 0; t < seqQ; t++)
                for (int h = 0; h < heads; h++)
                    for (int d = 0; d < keyDim; d++)
                        q[b, t, h, d] = MathF.Sin((b + 1) * (t + 2) * (h + 3) * (d + 1) * 0.031f);

        for (int b = 0; b < batch; b++)
            for (int t = 0; t < seqK; t++)
                for (int h = 0; h < heads; h++)
                    for (int d = 0; d < keyDim; d++)
                        k[b, t, h, d] = MathF.Cos((b + 2) * (t + 1) * (h + 1) * (d + 2) * 0.027f);

        for (int b = 0; b < batch; b++)
            for (int t = 0; t < seqK; t++)
                for (int h = 0; h < heads; h++)
                    for (int d = 0; d < valueDim; d++)
                        v[b, t, h, d] = MathF.Sin((b + 3) * (t + 2) * (h + 2) * (d + 1) * 0.019f) * 0.5f;

        for (int b = 0; b < batch; b++)
            for (int h = 0; h < heads; h++)
                for (int tq = 0; tq < seqQ; tq++)
                    for (int tk = 0; tk < seqK; tk++)
                        mask[b, h, tq, tk] = tk > tq + 1 ? float.NegativeInfinity : 0.01f * (b - h);

        using var allocator = new CudaAllocator();
        using var qTensor = Tensor.FromArray(allocator, q);
        using var kTensor = Tensor.FromArray(allocator, k);
        using var vTensor = Tensor.FromArray(allocator, v);
        using var maskTensor = Tensor.FromArray(allocator, mask);
        using var actualTensor = Ops.ScaledDotProductAttention(null, qTensor, kTensor, vTensor, maskTensor, scale);

        float[] actual = actualTensor.GetElementsAsFloat((int)actualTensor.ElementCount());
        float[] expected = ScaledDotProductAttentionReference(q, k, v, mask, scale);
        AssertClose(expected, actual, 1e-5f);
    }

    [Fact]
    public void CudaGqaPrefillAttention_WithWindowMatchesReference()
    {
        if (!CudaBackend.IsAvailable())
            return;

        const int numQHeads = 4;
        const int numKVHeads = 2;
        const int seqLen = 3;
        const int kvLen = 5;
        const int headDim = 5;
        const int maskStart = kvLen - seqLen;
        const int windowSize = 3;

        float[,,] q = new float[numQHeads, seqLen, headDim];
        float[,,] k = new float[numKVHeads, kvLen, headDim];
        float[,,] v = new float[numKVHeads, kvLen, headDim];

        for (int h = 0; h < numQHeads; h++)
            for (int t = 0; t < seqLen; t++)
                for (int d = 0; d < headDim; d++)
                    q[h, t, d] = MathF.Sin((h + 1) * (t + 2) * (d + 3) * 0.037f);

        for (int h = 0; h < numKVHeads; h++)
        {
            for (int t = 0; t < kvLen; t++)
            {
                for (int d = 0; d < headDim; d++)
                {
                    k[h, t, d] = MathF.Cos((h + 2) * (t + 1) * (d + 1) * 0.041f);
                    v[h, t, d] = MathF.Sin((h + 3) * (t + 2) * (d + 1) * 0.029f) * 0.5f;
                }
            }
        }

        using var allocator = new CudaAllocator();
        using var qTensor = Tensor.FromArray(allocator, q);
        using var kTensor = Tensor.FromArray(allocator, k);
        using var vTensor = Tensor.FromArray(allocator, v);
        using var actualTensor = new Tensor(allocator, DType.Float32, seqLen, numQHeads * headDim);

        Assert.True(CudaFusedOps.TryGqaPrefillAttention(
            actualTensor, qTensor, kTensor, vTensor,
            numQHeads, numKVHeads, headDim,
            seqLen, kvLen,
            maskStart, windowSize, 1.0f));

        float[] expected = GqaPrefillAttentionReference(q, k, v, numQHeads, numKVHeads, seqLen, kvLen, headDim, maskStart, windowSize);
        AssertClose(expected, actualTensor.GetElementsAsFloat(seqLen * numQHeads * headDim), 1e-5f);
    }

    [Fact]
    public void CudaFusedLayoutOps_MatchReference()
    {
        if (!CudaBackend.IsAvailable())
            return;

        using var allocator = new CudaAllocator();
        using var qkv = new Tensor(allocator, DType.Float32, 3, 10);
        FillSequential(qkv, scale: 1f, offset: 0f);

        using var slice = new Tensor(allocator, DType.Float32, 3, 4);
        Assert.True(CudaFusedOps.TrySliceColumns(slice, qkv, 2, 4));
        AssertClose(new[] { 2f, 3f, 4f, 5f, 12f, 13f, 14f, 15f, 22f, 23f, 24f, 25f },
            slice.GetElementsAsFloat(12));

        using var split = new Tensor(allocator, DType.Float32, 2, 3, 2);
        Assert.True(CudaFusedOps.TrySplitQkvToHeadFirst(split, qkv, 1, 2, 3, 2));
        AssertClose(new[] { 1f, 2f, 11f, 12f, 21f, 22f, 3f, 4f, 13f, 14f, 23f, 24f },
            split.GetElementsAsFloat(12));

        using var flat = new Tensor(allocator, DType.Float32, 3, 6);
        FillSequential(flat, scale: 1f, offset: 100f);
        using var heads = new Tensor(allocator, DType.Float32, 2, 3, 3);
        Assert.True(CudaFusedOps.TryFlatToHeadFirst(heads, flat, 2, 3, 3));
        AssertClose(new[] { 100f, 101f, 102f, 106f, 107f, 108f, 112f, 113f, 114f, 103f, 104f, 105f, 109f, 110f, 111f, 115f, 116f, 117f },
            heads.GetElementsAsFloat(18));
    }

    [Fact]
    public void CudaGqaDecodeAttention_CircularMatchesReference()
    {
        if (!CudaBackend.IsAvailable())
            return;

        const int numQHeads = 4;
        const int numKVHeads = 2;
        const int headDim = 3;
        const int cacheSize = 5;
        const int attendStart = 3;
        const int attendLen = 4;

        using var allocator = new CudaAllocator();
        using var q = new Tensor(allocator, DType.Float32, 1, numQHeads * headDim);
        using var k = new Tensor(allocator, DType.Float32, numKVHeads, cacheSize, headDim);
        using var v = new Tensor(allocator, DType.Float32, numKVHeads, cacheSize, headDim);
        FillSinusoidal(q, 0.07f);
        FillSinusoidal(k, 0.11f);
        FillSinusoidal(v, -0.13f);

        using var actual = new Tensor(allocator, DType.Float32, 1, numQHeads * headDim);
        Assert.True(CudaFusedOps.TryGqaDecodeAttention(
            actual, q, k, v, numQHeads, numKVHeads, headDim,
            attendStart, attendLen, cacheSize, circular: true, scale: 1.0f));

        float[] expected = GqaDecodeAttentionReference(
            q.GetElementsAsFloat(numQHeads * headDim),
            k.GetElementsAsFloat(numKVHeads * cacheSize * headDim),
            v.GetElementsAsFloat(numKVHeads * cacheSize * headDim),
            numQHeads, numKVHeads, headDim, attendStart, attendLen, cacheSize, circular: true);
        AssertClose(expected, actual.GetElementsAsFloat(numQHeads * headDim), 1e-5f);
    }

    [Fact]
    public void CudaFusedCacheOps_MatchReference()
    {
        if (!CudaBackend.IsAvailable())
            return;

        using var allocator = new CudaAllocator();
        using var src = new Tensor(allocator, DType.Float32, 2, 4, 3);
        FillSequential(src, scale: 1f, offset: 1f);
        using var cache = new Tensor(allocator, DType.Float32, 2, 5, 3);
        Ops.Fill(cache, -1f);

        Assert.True(CudaFusedOps.TryCopyHeadFirstToCache(cache, src, 3, 4, 5, circular: true));
        using var gathered = new Tensor(allocator, DType.Float32, 2, 4, 3);
        Assert.True(CudaFusedOps.TryGatherCircularHeadFirst(gathered, cache, 3, 4, 5));
        AssertClose(src.GetElementsAsFloat(24), gathered.GetElementsAsFloat(24));

        using var a = new Tensor(allocator, DType.Float32, 2, 2, 3);
        using var b = new Tensor(allocator, DType.Float32, 2, 3, 3);
        FillSequential(a, scale: 1f, offset: 10f);
        FillSequential(b, scale: 1f, offset: 100f);
        using var concat = new Tensor(allocator, DType.Float32, 2, 5, 3);
        Assert.True(CudaFusedOps.TryConcatHeadFirst(concat, a, b));
        AssertClose(new[]
        {
            10f, 11f, 12f, 13f, 14f, 15f, 100f, 101f, 102f, 103f, 104f, 105f, 106f, 107f, 108f,
            16f, 17f, 18f, 19f, 20f, 21f, 109f, 110f, 111f, 112f, 113f, 114f, 115f, 116f, 117f,
        }, concat.GetElementsAsFloat(30));
    }

    [Fact]
    public void CudaFusedGeluAndNeoXRope_MatchReference()
    {
        if (!CudaBackend.IsAvailable())
            return;

        using var allocator = new CudaAllocator();
        using var gateUp = new Tensor(allocator, DType.Float32, 2, 6);
        gateUp.SetElementsAsFloat(new[] { -1f, 0.5f, 2f, 3f, -4f, 0.25f, 1.25f, -0.75f, 0.1f, 2f, 3f, -5f });
        using var activated = new Tensor(allocator, DType.Float32, 2, 3);
        Assert.True(CudaFusedOps.TryGELUMulSplit(activated, gateUp, 3));

        float[] gateUpHost = gateUp.GetElementsAsFloat(12);
        float[] expectedActivated = new float[6];
        for (int row = 0; row < 2; row++)
            for (int col = 0; col < 3; col++)
                expectedActivated[row * 3 + col] = Gelu(gateUpHost[row * 6 + col]) * gateUpHost[row * 6 + 3 + col];
        AssertClose(expectedActivated, activated.GetElementsAsFloat(6), 1e-5f);

        using var data = new Tensor(allocator, DType.Float32, 2, 2, 4);
        data.SetElementsAsFloat(Enumerable.Range(0, 16).Select(i => 0.25f * (i + 1)).ToArray());
        using var cos = Tensor.FromArray(allocator, new[] { 1f, 0.5f, 0.25f, -0.75f });
        using var sin = Tensor.FromArray(allocator, new[] { 0f, 0.8660254f, 0.9682458f, 0.6614378f });
        float[] before = data.GetElementsAsFloat(16);
        Assert.True(CudaFusedOps.TryNeoXRoPEHeadFirst(data, cos, sin, 2, 2, 4, 2));

        float[] expectedRope = (float[])before.Clone();
        for (int h = 0; h < 2; h++)
        {
            for (int s = 0; s < 2; s++)
            {
                int baseIdx = (h * 2 + s) * 4;
                for (int j = 0; j < 2; j++)
                {
                    float c = cos.GetElementAsFloat(s * 2 + j);
                    float sn = sin.GetElementAsFloat(s * 2 + j);
                    float x0 = before[baseIdx + j];
                    float x1 = before[baseIdx + j + 2];
                    expectedRope[baseIdx + j] = x0 * c - x1 * sn;
                    expectedRope[baseIdx + j + 2] = x0 * sn + x1 * c;
                }
            }
        }
        AssertClose(expectedRope, data.GetElementsAsFloat(16), 1e-5f);
    }

    private static float[] SoftmaxRow(params float[] values)
    {
        float max = values.Max();
        float[] exps = values.Select(v => MathF.Exp(v - max)).ToArray();
        float sum = exps.Sum();
        return exps.Select(v => v / sum).ToArray();
    }

    private static float Gelu(float x)
    {
        return 0.5f * x * (1.0f + MathF.Tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
    }

    private static byte[] CreateQ8_0Rows(sbyte[][] rows)
    {
        byte[] data = new byte[rows.Length * 34];
        for (int row = 0; row < rows.Length; row++)
        {
            int offset = row * 34;
            ushort scale = BitConverter.HalfToUInt16Bits((System.Half)1.0f);
            data[offset] = (byte)(scale & 0xFF);
            data[offset + 1] = (byte)(scale >> 8);
            for (int i = 0; i < 32; i++)
                data[offset + 2 + i] = unchecked((byte)rows[row][i]);
        }

        return data;
    }

    private static byte[] CreateQ8_0Rows(int rows, int cols, Func<int, int, sbyte> value, Func<int, float> scale)
    {
        int blocks = cols / 32;
        byte[] data = new byte[rows * blocks * 34];
        for (int row = 0; row < rows; row++)
        {
            for (int block = 0; block < blocks; block++)
            {
                int offset = (row * blocks + block) * 34;
                ushort scaleBits = BitConverter.HalfToUInt16Bits((System.Half)scale(row));
                data[offset] = (byte)(scaleBits & 0xFF);
                data[offset + 1] = (byte)(scaleBits >> 8);
                for (int i = 0; i < 32; i++)
                    data[offset + 2 + i] = unchecked((byte)value(row, block * 32 + i));
            }
        }

        return data;
    }

    private static byte[] CreateQ4_0Rows(int rows, int cols, Func<int, int, sbyte> value, Func<int, float> scale)
    {
        int blocks = cols / 32;
        byte[] data = new byte[rows * blocks * 18];
        for (int row = 0; row < rows; row++)
        {
            for (int block = 0; block < blocks; block++)
            {
                int offset = (row * blocks + block) * 18;
                ushort scaleBits = BitConverter.HalfToUInt16Bits((System.Half)scale(row));
                data[offset] = (byte)(scaleBits & 0xFF);
                data[offset + 1] = (byte)(scaleBits >> 8);
                for (int i = 0; i < 16; i++)
                {
                    int low = value(row, block * 32 + i) + 8;
                    int high = value(row, block * 32 + i + 16) + 8;
                    data[offset + 2 + i] = (byte)(low | (high << 4));
                }
            }
        }

        return data;
    }

    private static float[] DequantizedMatmulQ80(byte[] weights, int outDim, int inDim, float[,] input, int rows)
    {
        int blocks = inDim / 32;
        float[] expected = new float[rows * outDim];
        for (int r = 0; r < rows; r++)
        {
            for (int o = 0; o < outDim; o++)
            {
                float sum = 0;
                for (int block = 0; block < blocks; block++)
                {
                    int offset = (o * blocks + block) * 34;
                    float scale = (float)BitConverter.UInt16BitsToHalf((ushort)(weights[offset] | (weights[offset + 1] << 8)));
                    for (int i = 0; i < 32; i++)
                    {
                        sbyte q = unchecked((sbyte)weights[offset + 2 + i]);
                        sum += scale * q * input[r, block * 32 + i];
                    }
                }
                expected[r * outDim + o] = sum;
            }
        }
        return expected;
    }

    private static float[] DequantizedMatmulQ40(byte[] weights, int outDim, int inDim, float[,] input, int rows)
    {
        int blocks = inDim / 32;
        float[] expected = new float[rows * outDim];
        for (int r = 0; r < rows; r++)
        {
            for (int o = 0; o < outDim; o++)
            {
                float sum = 0;
                for (int block = 0; block < blocks; block++)
                {
                    int offset = (o * blocks + block) * 18;
                    float scale = (float)BitConverter.UInt16BitsToHalf((ushort)(weights[offset] | (weights[offset + 1] << 8)));
                    for (int i = 0; i < 16; i++)
                    {
                        byte packed = weights[offset + 2 + i];
                        int low = (packed & 0x0F) - 8;
                        int high = (packed >> 4) - 8;
                        sum += scale * low * input[r, block * 32 + i];
                        sum += scale * high * input[r, block * 32 + i + 16];
                    }
                }
                expected[r * outDim + o] = sum;
            }
        }
        return expected;
    }

    private static float[] ScaledDotProductAttentionReference(float[,,,] q, float[,,,] k, float[,,,] v, float[,,,] mask, float scale)
    {
        int batch = q.GetLength(0);
        int seqQ = q.GetLength(1);
        int heads = q.GetLength(2);
        int keyDim = q.GetLength(3);
        int seqK = k.GetLength(1);
        int valueDim = v.GetLength(3);
        float[] result = new float[batch * seqQ * heads * valueDim];

        for (int b = 0; b < batch; b++)
        {
            for (int tq = 0; tq < seqQ; tq++)
            {
                for (int h = 0; h < heads; h++)
                {
                    float[] scores = new float[seqK];
                    float max = float.NegativeInfinity;
                    for (int tk = 0; tk < seqK; tk++)
                    {
                        float dot = 0;
                        for (int d = 0; d < keyDim; d++)
                            dot += q[b, tq, h, d] * k[b, tk, h, d];
                        scores[tk] = dot * scale + mask[b, h, tq, tk];
                        max = MathF.Max(max, scores[tk]);
                    }

                    float sum = 0;
                    for (int tk = 0; tk < seqK; tk++)
                    {
                        scores[tk] = MathF.Exp(scores[tk] - max);
                        sum += scores[tk];
                    }

                    for (int d = 0; d < valueDim; d++)
                    {
                        float acc = 0;
                        for (int tk = 0; tk < seqK; tk++)
                            acc += scores[tk] / sum * v[b, tk, h, d];
                        result[((b * seqQ + tq) * heads + h) * valueDim + d] = acc;
                    }
                }
            }
        }

        return result;
    }

    private static float[] GqaPrefillAttentionReference(
        float[,,] q,
        float[,,] k,
        float[,,] v,
        int numQHeads,
        int numKVHeads,
        int seqLen,
        int kvLen,
        int headDim,
        int maskStart,
        int windowSize)
    {
        int groupSize = numQHeads / numKVHeads;
        float[] result = new float[seqLen * numQHeads * headDim];

        for (int h = 0; h < numQHeads; h++)
        {
            int kvHead = h / groupSize;
            for (int tq = 0; tq < seqLen; tq++)
            {
                int visible = maskStart + tq;
                int minVisible = windowSize > 0 ? Math.Max(0, visible - windowSize + 1) : 0;
                float[] scores = new float[kvLen];
                float max = float.NegativeInfinity;
                for (int tk = 0; tk < kvLen; tk++)
                {
                    if (tk > visible || tk < minVisible)
                    {
                        scores[tk] = 0;
                        continue;
                    }

                    float dot = 0;
                    for (int d = 0; d < headDim; d++)
                        dot += q[h, tq, d] * k[kvHead, tk, d];
                    scores[tk] = dot;
                    max = MathF.Max(max, dot);
                }

                float sum = 0;
                for (int tk = minVisible; tk <= visible; tk++)
                {
                    scores[tk] = MathF.Exp(scores[tk] - max);
                    sum += scores[tk];
                }

                for (int d = 0; d < headDim; d++)
                {
                    float acc = 0;
                    for (int tk = minVisible; tk <= visible; tk++)
                        acc += scores[tk] / sum * v[kvHead, tk, d];
                    result[(tq * numQHeads + h) * headDim + d] = acc;
                }
            }
        }

        return result;
    }

    private static float[] GqaDecodeAttentionReference(
        float[] q,
        float[] k,
        float[] v,
        int numQHeads,
        int numKVHeads,
        int headDim,
        int attendStart,
        int attendLen,
        int cacheSize,
        bool circular)
    {
        int groupSize = numQHeads / numKVHeads;
        float[] result = new float[numQHeads * headDim];

        for (int h = 0; h < numQHeads; h++)
        {
            int kvHead = h / groupSize;
            float[] scores = new float[attendLen];
            float max = float.NegativeInfinity;
            for (int t = 0; t < attendLen; t++)
            {
                int logical = attendStart + t;
                int cachePos = circular ? logical % cacheSize : logical;
                float dot = 0;
                for (int d = 0; d < headDim; d++)
                    dot += q[h * headDim + d] * k[(kvHead * cacheSize + cachePos) * headDim + d];
                scores[t] = dot;
                max = MathF.Max(max, dot);
            }

            float sum = 0;
            for (int t = 0; t < attendLen; t++)
            {
                scores[t] = MathF.Exp(scores[t] - max);
                sum += scores[t];
            }

            for (int d = 0; d < headDim; d++)
            {
                float acc = 0;
                for (int t = 0; t < attendLen; t++)
                {
                    int logical = attendStart + t;
                    int cachePos = circular ? logical % cacheSize : logical;
                    acc += scores[t] / sum * v[(kvHead * cacheSize + cachePos) * headDim + d];
                }
                result[h * headDim + d] = acc;
            }
        }

        return result;
    }

    private static float[,] CreateQuantInput()
    {
        var input = new float[2, 32];
        for (int i = 0; i < 32; i++)
        {
            input[0, i] = 1.0f;
            input[1, i] = i + 1;
        }

        return input;
    }

    private static void FillSequential(Tensor tensor, float scale, float offset)
    {
        float[] values = new float[tensor.ElementCount()];
        for (int i = 0; i < values.Length; i++)
            values[i] = offset + i * scale;
        tensor.SetElementsAsFloat(values);
    }

    private static void FillSinusoidal(Tensor tensor, float scale)
    {
        float[] values = new float[tensor.ElementCount()];
        for (int i = 0; i < values.Length; i++)
            values[i] = MathF.Sin(i * scale) * 0.5f + MathF.Cos(i * scale * 0.7f) * 0.25f;
        tensor.SetElementsAsFloat(values);
    }

    private static void AssertClose(float[] expected, float[] actual, float tolerance = 1e-4f)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.InRange(MathF.Abs(actual[i] - expected[i]), 0.0f, tolerance);
    }
}
