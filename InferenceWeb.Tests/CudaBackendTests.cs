using TensorSharp;
using TensorSharp.Cuda;
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

    private static float[] SoftmaxRow(params float[] values)
    {
        float max = values.Max();
        float[] exps = values.Select(v => MathF.Exp(v - max)).ToArray();
        float sum = exps.Sum();
        return exps.Select(v => v / sum).ToArray();
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
