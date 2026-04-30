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

    private static void AssertClose(float[] expected, float[] actual, float tolerance = 1e-4f)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.InRange(MathF.Abs(actual[i] - expected[i]), 0.0f, tolerance);
    }
}
