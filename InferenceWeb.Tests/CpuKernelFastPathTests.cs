using TensorSharp;
using TensorSharp.Cpu;

namespace InferenceWeb.Tests;

public class CpuKernelFastPathTests
{
    private readonly IAllocator _alloc = new CpuAllocator(BlasEnum.DotNet);

    private Tensor TensorFrom(float[] values, params long[] sizes)
    {
        var tensor = new Tensor(_alloc, DType.Float32, sizes);
        tensor.SetElementsAsFloat(values);
        return tensor;
    }

    private static float[] Read(Tensor tensor)
        => tensor.GetElementsAsFloat((int)tensor.ElementCount());

    private static void AssertClose(float[] expected, Tensor actual, float tolerance = 1e-6f)
    {
        float[] observed = Read(actual);
        Assert.Equal(expected.Length, observed.Length);

        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(
                MathF.Abs(expected[i] - observed[i]) <= tolerance,
                $"index {i}: expected {expected[i]}, observed {observed[i]}");
        }
    }

    private static float[] MatMulExpected(float[] a, float[] b, float[] c, int m, int k, int n, float alpha, float beta)
    {
        float[] expected = new float[m * n];
        for (int row = 0; row < m; row++)
        {
            for (int col = 0; col < n; col++)
            {
                float acc = 0.0f;
                for (int kk = 0; kk < k; kk++)
                {
                    acc += a[row * k + kk] * b[kk * n + col];
                }

                expected[row * n + col] = alpha * acc + beta * c[row * n + col];
            }
        }

        return expected;
    }

    [Fact]
    public void SiLU_FastPath_MatchesScalar_WhenOutOfPlaceAndInPlace()
    {
        float[] values = [-4f, -1f, -0.25f, 0f, 0.5f, 1.5f, 4f, 8f];
        float[] expected = values.Select(x => x / (1.0f + MathF.Exp(-x))).ToArray();

        using var src = TensorFrom(values, values.Length);
        using var outOfPlace = new Tensor(_alloc, DType.Float32, values.Length);
        Ops.SiLU(outOfPlace, src);
        AssertClose(expected, outOfPlace);

        using var inPlace = TensorFrom(values, values.Length);
        Ops.SiLU(inPlace, inPlace);
        AssertClose(expected, inPlace);
    }

    [Fact]
    public void SiLUMul_FastPath_MatchesScalar_WithAliasedOutputs()
    {
        float[] gate = [-3f, -1f, 0f, 0.5f, 1f, 2f, 4f, 6f];
        float[] up = [0.25f, -2f, 3f, 4f, -1.5f, 0.75f, 2f, -0.5f];
        float[] expected = gate
            .Zip(up, (g, u) => (g / (1.0f + MathF.Exp(-g))) * u)
            .ToArray();

        using var gateTensor = TensorFrom(gate, gate.Length);
        using var upTensor = TensorFrom(up, up.Length);
        using var outOfPlace = new Tensor(_alloc, DType.Float32, gate.Length);
        Ops.SiLUMul(outOfPlace, gateTensor, upTensor);
        AssertClose(expected, outOfPlace);

        using var gateAliased = TensorFrom(gate, gate.Length);
        using var upForGateAlias = TensorFrom(up, up.Length);
        Ops.SiLUMul(gateAliased, gateAliased, upForGateAlias);
        AssertClose(expected, gateAliased);

        using var gateForUpAlias = TensorFrom(gate, gate.Length);
        using var upAliased = TensorFrom(up, up.Length);
        Ops.SiLUMul(upAliased, gateForUpAlias, upAliased);
        AssertClose(expected, upAliased);
    }

    [Fact]
    public void SigmoidAndExp_FastPaths_SupportInPlace()
    {
        float[] values = [-3f, -1f, 0f, 0.5f, 1f, 2f, 4f, 6f];

        using var sigmoid = TensorFrom(values, values.Length);
        Ops.Sigmoid(sigmoid, sigmoid);
        AssertClose(values.Select(x => 1.0f / (1.0f + MathF.Exp(-x))).ToArray(), sigmoid);

        using var exp = TensorFrom(values, values.Length);
        Ops.Exp(exp, exp);
        AssertClose(values.Select(MathF.Exp).ToArray(), exp, 1e-5f);
    }

    [Fact]
    public void SigmoidMul_FastPath_MatchesScalar_WithAliasedOutputs()
    {
        float[] x = [0.25f, -2f, 3f, 4f, -1.5f, 0.75f, 2f, -0.5f];
        float[] gate = [-3f, -1f, 0f, 0.5f, 1f, 2f, 4f, 6f];
        float[] expected = x
            .Zip(gate, (a, g) => a * (1.0f / (1.0f + MathF.Exp(-g))))
            .ToArray();

        using var xTensor = TensorFrom(x, x.Length);
        using var gateTensor = TensorFrom(gate, gate.Length);
        using var outOfPlace = new Tensor(_alloc, DType.Float32, x.Length);
        Ops.SigmoidMul(outOfPlace, xTensor, gateTensor);
        AssertClose(expected, outOfPlace);

        using var xAliased = TensorFrom(x, x.Length);
        using var gateForXAlias = TensorFrom(gate, gate.Length);
        Ops.SigmoidMul(xAliased, xAliased, gateForXAlias);
        AssertClose(expected, xAliased);

        using var xForGateAlias = TensorFrom(x, x.Length);
        using var gateAliased = TensorFrom(gate, gate.Length);
        Ops.SigmoidMul(gateAliased, xForGateAlias, gateAliased);
        AssertClose(expected, gateAliased);
    }

    [Fact]
    public void LayerNorm_FastPath_AllowsNullBeta()
    {
        float[] values = [1f, 2f, 3f, 4f, -1f, -2f, -3f, -4f];
        using var src = TensorFrom(values, 2, 4);
        using var gamma = TensorFrom([1f, 1f, 1f, 1f], 4);

        using Tensor result = Ops.LayerNorm(null, src, gamma, null, 1e-5f);

        Assert.Equal((long)values.Length, result.ElementCount());
    }

    [Fact]
    public void ManagedGemm_RowMajorAndColumnMajorB_MatchScalar()
    {
        const int m = 5, k = 7, n = 6;
        float[] a = Enumerable.Range(0, m * k).Select(i => MathF.Sin(i * 0.17f)).ToArray();
        float[] b = Enumerable.Range(0, k * n).Select(i => MathF.Cos(i * 0.11f)).ToArray();
        float[] c = Enumerable.Range(0, m * n).Select(i => i * 0.01f - 0.2f).ToArray();
        float[] expected = MatMulExpected(a, b, c, m, k, n, alpha: 0.75f, beta: 0.25f);

        using var aTensor = TensorFrom(a, m, k);
        using var bRow = TensorFrom(b, k, n);
        using var cRow = TensorFrom(c, m, n);
        Ops.Addmm(cRow, 0.25f, cRow, 0.75f, aTensor, bRow);
        AssertClose(expected, cRow, 1e-5f);

        float[] bBaseValues = new float[n * k];
        for (int row = 0; row < k; row++)
        {
            for (int col = 0; col < n; col++)
            {
                bBaseValues[col * k + row] = b[row * n + col];
            }
        }

        using var bBase = TensorFrom(bBaseValues, n, k);
        using var bCol = bBase.Transpose();
        using var cCol = TensorFrom(c, m, n);
        Ops.Addmm(cCol, 0.25f, cCol, 0.75f, aTensor, bCol);
        AssertClose(expected, cCol, 1e-5f);
    }

    [Fact]
    public void ManagedGemmBatch_RowMajorAndColumnMajorB_MatchScalar()
    {
        const int batch = 2, m = 5, k = 7, n = 6;
        float[] a = Enumerable.Range(0, batch * m * k).Select(i => MathF.Sin(i * 0.13f)).ToArray();
        float[] b = Enumerable.Range(0, batch * k * n).Select(i => MathF.Cos(i * 0.07f)).ToArray();
        float[] c = Enumerable.Range(0, batch * m * n).Select(i => i * 0.005f - 0.1f).ToArray();
        float[] expected = new float[batch * m * n];
        for (int batchIndex = 0; batchIndex < batch; batchIndex++)
        {
            float[] expectedBatch = MatMulExpected(
                a.AsSpan(batchIndex * m * k, m * k).ToArray(),
                b.AsSpan(batchIndex * k * n, k * n).ToArray(),
                c.AsSpan(batchIndex * m * n, m * n).ToArray(),
                m, k, n, alpha: 0.75f, beta: 0.25f);
            expectedBatch.CopyTo(expected, batchIndex * m * n);
        }

        using var aTensor = TensorFrom(a, batch, m, k);
        using var bRow = TensorFrom(b, batch, k, n);
        using var cRow = TensorFrom(c, batch, m, n);
        Ops.AddmmBatch(cRow, 0.25f, cRow, 0.75f, aTensor, bRow);
        AssertClose(expected, cRow, 1e-5f);

        float[] bBaseValues = new float[batch * n * k];
        for (int batchIndex = 0; batchIndex < batch; batchIndex++)
        {
            for (int row = 0; row < k; row++)
            {
                for (int col = 0; col < n; col++)
                {
                    bBaseValues[batchIndex * n * k + col * k + row] = b[batchIndex * k * n + row * n + col];
                }
            }
        }

        using var bBase = TensorFrom(bBaseValues, batch, n, k);
        using var bCol = bBase.Transpose(1, 2);
        using var cCol = TensorFrom(c, batch, m, n);
        Ops.AddmmBatch(cCol, 0.25f, cCol, 0.75f, aTensor, bCol);
        AssertClose(expected, cCol, 1e-5f);
    }
}
