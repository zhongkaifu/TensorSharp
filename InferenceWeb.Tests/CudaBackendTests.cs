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
    public void CudaQwen35GatedDeltaNetPacked_MatchesCpuReference()
    {
        if (!CudaBackend.IsAvailable())
            return;

        const int seqLen = 4;
        const int numKHeads = 2;
        const int numVHeads = 4;
        const int headKDim = 4;
        const int headVDim = 3;
        const int convKernel = 3;
        const int convWriteIdx = 1;
        const float eps = 1e-5f;

        int qkDim = numKHeads * headKDim;
        int vDim = numVHeads * headVDim;
        int qkvDim = 2 * qkDim + vDim;
        int packedDim = qkvDim + vDim + 2 * numVHeads;
        int convDim = convKernel - 1;

        float[] packed = MakePattern(seqLen * packedDim, 0.07f, -0.015f);
        float[] qkv = new float[seqLen * qkvDim];
        float[] z = new float[seqLen * vDim];
        float[] beta = new float[seqLen * numVHeads];
        float[] alpha = new float[seqLen * numVHeads];
        for (int s = 0; s < seqLen; s++)
        {
            Array.Copy(packed, s * packedDim, qkv, s * qkvDim, qkvDim);
            Array.Copy(packed, s * packedDim + qkvDim, z, s * vDim, vDim);
            Array.Copy(packed, s * packedDim + qkvDim + vDim, beta, s * numVHeads, numVHeads);
            Array.Copy(packed, s * packedDim + qkvDim + vDim + numVHeads, alpha, s * numVHeads, numVHeads);
        }
        float[] convState = MakePattern(convDim * qkvDim, 0.05f, 0.01f);
        float[] ssmState = MakePattern(numVHeads * headVDim * headKDim, 0.04f, -0.02f);
        float[] convWeight = MakePattern(qkvDim * convKernel, 0.08f, 0.03f);
        float[] dtBias = MakePattern(numVHeads, 0.03f, -0.01f);
        float[] aLog = new float[numVHeads];
        float[] ssmNorm = new float[headVDim];
        for (int i = 0; i < numVHeads; i++)
            aLog[i] = -0.05f - i * 0.01f;
        for (int i = 0; i < headVDim; i++)
            ssmNorm[i] = 0.8f + i * 0.07f;

        float[] expectedConv = (float[])convState.Clone();
        float[] expectedState = (float[])ssmState.Clone();
        float[] expected = Qwen35GdnPackedReference(
            packed,
            expectedConv,
            expectedState,
            convWeight,
            dtBias,
            aLog,
            ssmNorm,
            seqLen,
            packedDim,
            qkvDim,
            qkDim,
            vDim,
            numKHeads,
            numVHeads,
            headKDim,
            headVDim,
            convKernel,
            convWriteIdx,
            eps);

        using var allocator = new CudaAllocator();
        using var packedFromParts = new Tensor(allocator, DType.Float32, seqLen, packedDim);
        using var qkvTensor = new Tensor(allocator, DType.Float32, seqLen, qkvDim);
        using var zTensor = new Tensor(allocator, DType.Float32, seqLen, vDim);
        using var betaTensor = new Tensor(allocator, DType.Float32, seqLen, numVHeads);
        using var alphaTensor = new Tensor(allocator, DType.Float32, seqLen, numVHeads);
        using var convTensor = new Tensor(allocator, DType.Float32, convDim, qkvDim);
        using var stateTensor = new Tensor(allocator, DType.Float32, numVHeads, headVDim, headKDim);
        using var convWeightTensor = new Tensor(allocator, DType.Float32, qkvDim, convKernel);
        using var dtBiasTensor = new Tensor(allocator, DType.Float32, numVHeads);
        using var aLogTensor = new Tensor(allocator, DType.Float32, numVHeads);
        using var ssmNormTensor = new Tensor(allocator, DType.Float32, headVDim);
        using var output = new Tensor(allocator, DType.Float32, seqLen, vDim);

        qkvTensor.SetElementsAsFloat(qkv);
        zTensor.SetElementsAsFloat(z);
        betaTensor.SetElementsAsFloat(beta);
        alphaTensor.SetElementsAsFloat(alpha);
        convTensor.SetElementsAsFloat(convState);
        stateTensor.SetElementsAsFloat(ssmState);
        convWeightTensor.SetElementsAsFloat(convWeight);
        dtBiasTensor.SetElementsAsFloat(dtBias);
        aLogTensor.SetElementsAsFloat(aLog);
        ssmNormTensor.SetElementsAsFloat(ssmNorm);

        Assert.True(CudaFusedOps.TryQwen35GatedDeltaNetPackInputs(
            packedFromParts,
            qkvTensor,
            zTensor,
            betaTensor,
            alphaTensor,
            seqLen,
            qkvDim,
            vDim,
            numVHeads,
            packedDim));
        AssertClose(packed, packedFromParts.GetElementsAsFloat(seqLen * packedDim), 1e-6f);

        Assert.True(CudaFusedOps.TryQwen35GatedDeltaNetPacked(
            output,
            packedFromParts,
            convTensor,
            stateTensor,
            convWeightTensor,
            dtBiasTensor,
            aLogTensor,
            ssmNormTensor,
            seqLen,
            packedDim,
            qkvDim,
            qkDim,
            vDim,
            numKHeads,
            numVHeads,
            headKDim,
            headVDim,
            convKernel,
            convWriteIdx,
            eps));

        AssertClose(expected, output.GetElementsAsFloat(seqLen * vDim), 5e-4f);
        AssertClose(expectedState, stateTensor.GetElementsAsFloat(expectedState.Length), 5e-4f);
        AssertClose(expectedConv, convTensor.GetElementsAsFloat(expectedConv.Length), 1e-6f);
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

    [Theory]
    [InlineData((int)GgmlTensorType.Q4_K)]
    [InlineData((int)GgmlTensorType.Q5_K)]
    [InlineData((int)GgmlTensorType.Q6_K)]
    public void CudaQuantizedMatmul_KQuantsMatchDequantizedReferenceAfterHostRelease(int ggmlType)
    {
        if (!CudaBackend.IsAvailable())
            return;

        const int rows = 3;
        const int inDim = 256;
        const int outDim = 5;
        byte[] weights = CreateKQuantRows(ggmlType, outDim, inDim);
        float[,] input = new float[rows, inDim];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < inDim; c++)
                input[r, c] = MathF.Sin((r + 1) * (c + 1) * 0.013f) + MathF.Cos((r + 2) * (c + 3) * 0.007f) * 0.25f;

        float[] expected = DequantizedMatmulK(weights, outDim, inDim, input, GetKQuantDequantizer(ggmlType));
        IntPtr host = Marshal.AllocHGlobal(weights.Length);
        IntPtr cacheKey = new(0x764000 + ggmlType);
        try
        {
            Marshal.Copy(weights, 0, host, weights.Length);
            using var allocator = new CudaAllocator();
            CudaQuantizedOps.PreloadQuantizedWeight(allocator, cacheKey, host, ggmlType, inDim, outDim, weights.Length);

            try
            {
                using var inputTensor = Tensor.FromArray(allocator, input);
                using var output = new Tensor(allocator, DType.Float32, rows, outDim);

                Assert.True(CudaQuantizedOps.TryAddmmQuantizedToFloat32(output, inputTensor, cacheKey, IntPtr.Zero, ggmlType, inDim, outDim, weights.Length));
                AssertClose(expected, output.GetElementsAsFloat(rows * outDim), 5e-2f);
            }
            finally
            {
                CudaQuantizedOps.ReleaseQuantizedWeight(allocator, cacheKey);
            }
        }
        finally
        {
            Marshal.FreeHGlobal(host);
        }
    }

    [Theory]
    [InlineData((int)GgmlTensorType.Q4_K)]
    [InlineData((int)GgmlTensorType.Q5_K)]
    [InlineData((int)GgmlTensorType.Q6_K)]
    public void CudaQuantizedRows_KQuantsMatchDequantizedReferenceAfterHostRelease(int ggmlType)
    {
        if (!CudaBackend.IsAvailable())
            return;

        const int inDim = 256;
        const int outDim = 5;
        int[] rows = { 4, 1, 3 };
        byte[] weights = CreateKQuantRows(ggmlType, outDim, inDim);
        float[] expected = new float[rows.Length * inDim];
        DequantizeKRow dequantize = GetKQuantDequantizer(ggmlType);
        for (int i = 0; i < rows.Length; i++)
            dequantize(weights, rows[i], inDim, expected, i * inDim);

        IntPtr host = Marshal.AllocHGlobal(weights.Length);
        IntPtr cacheKey = new(0x765000 + ggmlType);
        try
        {
            Marshal.Copy(weights, 0, host, weights.Length);
            using var allocator = new CudaAllocator();
            CudaQuantizedOps.PreloadQuantizedWeight(allocator, cacheKey, host, ggmlType, inDim, outDim, weights.Length);

            try
            {
                using var indices = Tensor.FromArray(allocator, rows);
                using var output = new Tensor(allocator, DType.Float32, rows.Length, inDim);
                Assert.True(CudaQuantizedOps.TryGetRowsQuantizedToFloat32(
                    output,
                    cacheKey,
                    IntPtr.Zero,
                    ggmlType,
                    inDim,
                    outDim,
                    weights.Length,
                    indices));

                AssertClose(expected, output.GetElementsAsFloat(rows.Length * inDim), 5e-2f);
            }
            finally
            {
                CudaQuantizedOps.ReleaseQuantizedWeight(allocator, cacheKey);
            }
        }
        finally
        {
            Marshal.FreeHGlobal(host);
        }
    }

    [Fact]
    public void CudaQuantizedMatmulAndRows_IQ2XXSMatchQ8ActivationReferenceAfterHostRelease()
    {
        if (!CudaBackend.IsAvailable())
            return;

        const int rows = 3;
        const int inDim = 512;
        const int outDim = 5;
        byte[] weights = CreateIq2XxsRows(outDim, inDim);
        float[,] input = new float[rows, inDim];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < inDim; c++)
                input[r, c] = MathF.Cos((r + 3) * (c + 1) * 0.009f);

        IntPtr host = Marshal.AllocHGlobal(weights.Length);
        IntPtr cacheKey = new(0x766000 + (int)GgmlTensorType.IQ2_XXS);
        try
        {
            Marshal.Copy(weights, 0, host, weights.Length);
            using var allocator = new CudaAllocator();
            CudaQuantizedOps.PreloadQuantizedWeight(allocator, cacheKey, host, (int)GgmlTensorType.IQ2_XXS, inDim, outDim, weights.Length);

            try
            {
                using var inputTensor = Tensor.FromArray(allocator, input);
                using var output = new Tensor(allocator, DType.Float32, rows, outDim);
                Assert.True(CudaQuantizedOps.TryAddmmQuantizedToFloat32(
                    output,
                    inputTensor,
                    cacheKey,
                    IntPtr.Zero,
                    (int)GgmlTensorType.IQ2_XXS,
                    inDim,
                    outDim,
                    weights.Length));

                float[] expected = DequantizedMatmulNative(weights, GgmlTensorType.IQ2_XXS, outDim, inDim, QuantizeDequantizeQ8_1(input));
                AssertClose(expected, output.GetElementsAsFloat(rows * outDim), 5e-3f);

                int[] selected = { 4, 1, 3 };
                using var indices = Tensor.FromArray(allocator, selected);
                using var rowOutput = new Tensor(allocator, DType.Float32, selected.Length, inDim);
                Assert.True(CudaQuantizedOps.TryGetRowsQuantizedToFloat32(
                    rowOutput,
                    cacheKey,
                    IntPtr.Zero,
                    (int)GgmlTensorType.IQ2_XXS,
                    inDim,
                    outDim,
                    weights.Length,
                    indices));

                float[] expectedRows = DequantizeNativeRows(weights, GgmlTensorType.IQ2_XXS, inDim, selected);
                AssertClose(expectedRows, rowOutput.GetElementsAsFloat(selected.Length * inDim), 5e-3f);
            }
            finally
            {
                CudaQuantizedOps.ReleaseQuantizedWeight(allocator, cacheKey);
            }
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
    public void CudaGqaPrefillAttention_ReadsFloat16KvCache()
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
        using var kF32 = Tensor.FromArray(allocator, k);
        using var vF32 = Tensor.FromArray(allocator, v);
        using var kF16 = new Tensor(allocator, DType.Float16, numKVHeads, kvLen, headDim);
        using var vF16 = new Tensor(allocator, DType.Float16, numKVHeads, kvLen, headDim);
        Assert.True(CudaFusedOps.TryCopyHeadFirstToCache(kF16, kF32, 0, kvLen, kvLen, circular: false));
        Assert.True(CudaFusedOps.TryCopyHeadFirstToCache(vF16, vF32, 0, kvLen, kvLen, circular: false));

        using var actualTensor = new Tensor(allocator, DType.Float32, seqLen, numQHeads * headDim);
        Assert.True(CudaFusedOps.TryGqaPrefillAttention(
            actualTensor, qTensor, kF16, vF16,
            numQHeads, numKVHeads, headDim,
            seqLen, kvLen,
            maskStart, windowSize, 1.0f));

        float[] expected = GqaPrefillAttentionReference(q, k, v, numQHeads, numKVHeads, seqLen, kvLen, headDim, maskStart, windowSize);
        AssertClose(expected, actualTensor.GetElementsAsFloat(seqLen * numQHeads * headDim), 2e-3f);
    }

    [Fact]
    public void CudaGqaPrefillAttentionWithSinks_ReadsStridedKvCache()
    {
        if (!CudaBackend.IsAvailable())
            return;

        const int numQHeads = 4;
        const int numKVHeads = 2;
        const int seqLen = 3;
        const int kvLen = 5;
        const int cacheSize = 7;
        const int headDim = 5;
        const int maskStart = kvLen - seqLen;
        const int windowSize = 4;
        const float scale = 0.73f;

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

        float[] sinkValues = { 0.15f, -0.2f, 0.35f, -0.05f };

        using var allocator = new CudaAllocator();
        using var qTensor = Tensor.FromArray(allocator, q);
        using var kActive = Tensor.FromArray(allocator, k);
        using var vActive = Tensor.FromArray(allocator, v);
        using var sinks = Tensor.FromArray(allocator, sinkValues);

        using var kCacheF32 = new Tensor(allocator, DType.Float32, numKVHeads, cacheSize, headDim);
        using var vCacheF32 = new Tensor(allocator, DType.Float32, numKVHeads, cacheSize, headDim);
        Ops.Fill(kCacheF32, -9f);
        Ops.Fill(vCacheF32, -9f);
        Assert.True(CudaFusedOps.TryCopyHeadFirstToCache(kCacheF32, kActive, 0, kvLen, cacheSize, circular: false));
        Assert.True(CudaFusedOps.TryCopyHeadFirstToCache(vCacheF32, vActive, 0, kvLen, cacheSize, circular: false));

        float[] expected = GqaPrefillAttentionReference(
            q, k, v, numQHeads, numKVHeads, seqLen, kvLen, headDim, maskStart, windowSize, scale, sinkValues);

        using var actualF32 = new Tensor(allocator, DType.Float32, seqLen, numQHeads * headDim);
        Assert.True(CudaFusedOps.TryGqaPrefillAttentionWithSinks(
            actualF32, qTensor, kCacheF32, vCacheF32, sinks,
            numQHeads, numKVHeads, headDim,
            seqLen, kvLen, cacheSize,
            maskStart, windowSize, scale));
        AssertClose(expected, actualF32.GetElementsAsFloat(seqLen * numQHeads * headDim), 1e-5f);

        using var kCacheF16 = new Tensor(allocator, DType.Float16, numKVHeads, cacheSize, headDim);
        using var vCacheF16 = new Tensor(allocator, DType.Float16, numKVHeads, cacheSize, headDim);
        Assert.True(CudaFusedOps.TryCopyHeadFirstToCache(kCacheF16, kActive, 0, kvLen, cacheSize, circular: false));
        Assert.True(CudaFusedOps.TryCopyHeadFirstToCache(vCacheF16, vActive, 0, kvLen, cacheSize, circular: false));

        using var actualF16 = new Tensor(allocator, DType.Float32, seqLen, numQHeads * headDim);
        Assert.True(CudaFusedOps.TryGqaPrefillAttentionWithSinks(
            actualF16, qTensor, kCacheF16, vCacheF16, sinks,
            numQHeads, numKVHeads, headDim,
            seqLen, kvLen, cacheSize,
            maskStart, windowSize, scale));
        AssertClose(expected, actualF16.GetElementsAsFloat(seqLen * numQHeads * headDim), 2e-3f);
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
    public void CudaGqaDecodeAttention_ReadsFloat16KvCache()
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
        using var kF32 = new Tensor(allocator, DType.Float32, numKVHeads, cacheSize, headDim);
        using var vF32 = new Tensor(allocator, DType.Float32, numKVHeads, cacheSize, headDim);
        FillSinusoidal(q, 0.07f);
        FillSinusoidal(kF32, 0.11f);
        FillSinusoidal(vF32, -0.13f);

        using var kF16 = new Tensor(allocator, DType.Float16, numKVHeads, cacheSize, headDim);
        using var vF16 = new Tensor(allocator, DType.Float16, numKVHeads, cacheSize, headDim);
        Assert.True(CudaFusedOps.TryCopyHeadFirstToCache(kF16, kF32, 0, cacheSize, cacheSize, circular: false));
        Assert.True(CudaFusedOps.TryCopyHeadFirstToCache(vF16, vF32, 0, cacheSize, cacheSize, circular: false));

        using var actual = new Tensor(allocator, DType.Float32, 1, numQHeads * headDim);
        Assert.True(CudaFusedOps.TryGqaDecodeAttention(
            actual, q, kF16, vF16, numQHeads, numKVHeads, headDim,
            attendStart, attendLen, cacheSize, circular: true, scale: 1.0f));

        float[] expected = GqaDecodeAttentionReference(
            q.GetElementsAsFloat(numQHeads * headDim),
            kF32.GetElementsAsFloat(numKVHeads * cacheSize * headDim),
            vF32.GetElementsAsFloat(numKVHeads * cacheSize * headDim),
            numQHeads, numKVHeads, headDim, attendStart, attendLen, cacheSize, circular: true);
        AssertClose(expected, actual.GetElementsAsFloat(numQHeads * headDim), 2e-3f);
    }

    [Fact]
    public void CudaGqaDecodeAttentionWithSinks_ReadsFloat16KvCache()
    {
        if (!CudaBackend.IsAvailable())
            return;

        const int numQHeads = 4;
        const int numKVHeads = 2;
        const int headDim = 3;
        const int cacheSize = 6;
        const int attendStart = 1;
        const int attendLen = 5;
        const float scale = 0.73f;

        using var allocator = new CudaAllocator();
        using var q = new Tensor(allocator, DType.Float32, 1, numQHeads * headDim);
        using var kF32 = new Tensor(allocator, DType.Float32, numKVHeads, cacheSize, headDim);
        using var vF32 = new Tensor(allocator, DType.Float32, numKVHeads, cacheSize, headDim);
        FillSinusoidal(q, 0.07f);
        FillSinusoidal(kF32, 0.11f);
        FillSinusoidal(vF32, -0.13f);

        using var kF16 = new Tensor(allocator, DType.Float16, numKVHeads, cacheSize, headDim);
        using var vF16 = new Tensor(allocator, DType.Float16, numKVHeads, cacheSize, headDim);
        Assert.True(CudaFusedOps.TryCopyHeadFirstToCache(kF16, kF32, 0, cacheSize, cacheSize, circular: false));
        Assert.True(CudaFusedOps.TryCopyHeadFirstToCache(vF16, vF32, 0, cacheSize, cacheSize, circular: false));

        float[] sinkValues = { 0.15f, -0.2f, 0.35f, -0.05f };
        using var sinks = Tensor.FromArray(allocator, sinkValues);
        using var actual = new Tensor(allocator, DType.Float32, 1, numQHeads * headDim);
        Assert.True(CudaFusedOps.TryGqaDecodeAttentionWithSinks(
            actual, q, kF16, vF16, sinks, numQHeads, numKVHeads, headDim,
            attendStart, attendLen, cacheSize, circular: false, scale));

        float[] expected = GqaDecodeAttentionReference(
            q.GetElementsAsFloat(numQHeads * headDim),
            kF32.GetElementsAsFloat(numKVHeads * cacheSize * headDim),
            vF32.GetElementsAsFloat(numKVHeads * cacheSize * headDim),
            numQHeads, numKVHeads, headDim, attendStart, attendLen, cacheSize,
            circular: false, scale, sinkValues);
        AssertClose(expected, actual.GetElementsAsFloat(numQHeads * headDim), 2e-3f);
    }

    [Fact]
    public void CudaGqaDecodeAttentionWithSinks_PartitionedMatchesReference()
    {
        if (!CudaBackend.IsAvailable())
            return;

        const int numQHeads = 4;
        const int numKVHeads = 2;
        const int headDim = 16;
        const int cacheSize = 2350;
        const int attendStart = 29;
        const int attendLen = 2305;
        const float scale = 0.21f;

        using var allocator = new CudaAllocator();
        using var q = new Tensor(allocator, DType.Float32, 1, numQHeads * headDim);
        using var kF32 = new Tensor(allocator, DType.Float32, numKVHeads, cacheSize, headDim);
        using var vF32 = new Tensor(allocator, DType.Float32, numKVHeads, cacheSize, headDim);
        FillSinusoidal(q, 0.07f);
        FillSinusoidal(kF32, 0.011f);
        FillSinusoidal(vF32, -0.013f);

        float[] sinkValues = { 0.08f, -0.17f, 0.29f, -0.04f };
        using var sinks = Tensor.FromArray(allocator, sinkValues);
        float[] expected = GqaDecodeAttentionReference(
            q.GetElementsAsFloat(numQHeads * headDim),
            kF32.GetElementsAsFloat(numKVHeads * cacheSize * headDim),
            vF32.GetElementsAsFloat(numKVHeads * cacheSize * headDim),
            numQHeads, numKVHeads, headDim, attendStart, attendLen, cacheSize,
            circular: false, scale, sinkValues);

        using var actualF32 = new Tensor(allocator, DType.Float32, 1, numQHeads * headDim);
        Assert.True(CudaFusedOps.TryGqaDecodeAttentionWithSinks(
            actualF32, q, kF32, vF32, sinks, numQHeads, numKVHeads, headDim,
            attendStart, attendLen, cacheSize, circular: false, scale));
        AssertClose(expected, actualF32.GetElementsAsFloat(numQHeads * headDim), 1e-5f);

        using var kF16 = new Tensor(allocator, DType.Float16, numKVHeads, cacheSize, headDim);
        using var vF16 = new Tensor(allocator, DType.Float16, numKVHeads, cacheSize, headDim);
        Assert.True(CudaFusedOps.TryCopyHeadFirstToCache(kF16, kF32, 0, cacheSize, cacheSize, circular: false));
        Assert.True(CudaFusedOps.TryCopyHeadFirstToCache(vF16, vF32, 0, cacheSize, cacheSize, circular: false));

        using var actualF16 = new Tensor(allocator, DType.Float32, 1, numQHeads * headDim);
        Assert.True(CudaFusedOps.TryGqaDecodeAttentionWithSinks(
            actualF16, q, kF16, vF16, sinks, numQHeads, numKVHeads, headDim,
            attendStart, attendLen, cacheSize, circular: false, scale));
        AssertClose(expected, actualF16.GetElementsAsFloat(numQHeads * headDim), 2e-3f);
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
    public void CudaFusedFloat16CacheOps_MatchReference()
    {
        if (!CudaBackend.IsAvailable())
            return;

        using var allocator = new CudaAllocator();
        using var filled = new Tensor(allocator, DType.Float16, 1, 4, 2);
        Ops.Fill(filled, 1.5f);
        using var filledGathered = new Tensor(allocator, DType.Float32, 1, 4, 2);
        Assert.True(CudaFusedOps.TryGatherCircularHeadFirst(filledGathered, filled, 0, 4, 4));
        AssertClose(Enumerable.Repeat(1.5f, 8).ToArray(), filledGathered.GetElementsAsFloat(8), 1e-4f);

        using var src = new Tensor(allocator, DType.Float32, 2, 4, 3);
        FillSequential(src, scale: 1f, offset: 1f);
        using var cache = new Tensor(allocator, DType.Float16, 2, 5, 3);
        Ops.Fill(cache, -1f);

        Assert.True(CudaFusedOps.TryCopyHeadFirstToCache(cache, src, 3, 4, 5, circular: true));
        using var gathered = new Tensor(allocator, DType.Float32, 2, 4, 3);
        Assert.True(CudaFusedOps.TryGatherCircularHeadFirst(gathered, cache, 3, 4, 5));
        AssertClose(src.GetElementsAsFloat(24), gathered.GetElementsAsFloat(24), 1e-3f);
    }

    [Fact]
    public void CudaCopy_Float16NarrowedKvCacheLayout_StaysOnDevice()
    {
        if (!CudaBackend.IsAvailable())
            return;

        const int heads = 3;
        const int oldCapacity = 8;
        const int newCapacity = 16;
        const int headDim = 5;
        const int cacheSeqLen = 5;

        using var allocator = new CudaAllocator();
        using var sourceF32 = new Tensor(allocator, DType.Float32, heads, oldCapacity, headDim);
        FillSequential(sourceF32, scale: 0.125f, offset: 1f);

        using var oldCache = new Tensor(allocator, DType.Float16, heads, oldCapacity, headDim);
        using var newCache = new Tensor(allocator, DType.Float16, heads, newCapacity, headDim);
        Assert.True(CudaFusedOps.TryCopyHeadFirstToCache(oldCache, sourceF32, 0, oldCapacity, oldCapacity, circular: false));
        Ops.Fill(newCache, -2f);

        using (var src = oldCache.Narrow(1, 0, cacheSeqLen))
        using (var dst = newCache.Narrow(1, 0, cacheSeqLen))
            Ops.Copy(dst, src);

        using var expected = new Tensor(allocator, DType.Float32, heads, cacheSeqLen, headDim);
        using var actual = new Tensor(allocator, DType.Float32, heads, cacheSeqLen, headDim);
        Assert.True(CudaFusedOps.TryGatherCircularHeadFirst(expected, oldCache, 0, cacheSeqLen, oldCapacity));
        Assert.True(CudaFusedOps.TryGatherCircularHeadFirst(actual, newCache, 0, cacheSeqLen, newCapacity));
        AssertClose(expected.GetElementsAsFloat(heads * cacheSeqLen * headDim),
            actual.GetElementsAsFloat(heads * cacheSeqLen * headDim), 1e-3f);

        using var tail = new Tensor(allocator, DType.Float32, heads, newCapacity - cacheSeqLen, headDim);
        Assert.True(CudaFusedOps.TryGatherCircularHeadFirst(tail, newCache, cacheSeqLen, newCapacity - cacheSeqLen, newCapacity));
        AssertClose(Enumerable.Repeat(-2f, heads * (newCapacity - cacheSeqLen) * headDim).ToArray(),
            tail.GetElementsAsFloat(heads * (newCapacity - cacheSeqLen) * headDim), 1e-3f);
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

    [Fact]
    public void CudaFusedBiasAndOaiSwiGlu_MatchReference()
    {
        if (!CudaBackend.IsAvailable())
            return;

        using var allocator = new CudaAllocator();
        using var matrix = Tensor.FromArray(allocator, new float[,]
        {
            { 1f, -2f, 3f, 4f, -5f },
            { -0.5f, 0.75f, -1.25f, 2.5f, 3.5f },
        });
        using var bias = Tensor.FromArray(allocator, new[] { 0.5f, -1f, 1.5f, -2f, 2.5f });
        Assert.True(CudaFusedOps.TryAddBiasRows(matrix, bias));
        AssertClose(new[]
        {
            1.5f, -3f, 4.5f, 2f, -2.5f,
            0f, -0.25f, 0.25f, 0.5f, 6f,
        }, matrix.GetElementsAsFloat(10), 1e-6f);

        const float alpha = 1.702f;
        const float limit = 2.0f;
        float[] gateUpValues = { -3f, 0.5f, 3f, 1f, -4f, 0.25f, 1.25f, -0.75f, 0.1f, 2f, 3f, -5f };
        using var gateUp = Tensor.FromArray(allocator, new float[,]
        {
            { gateUpValues[0], gateUpValues[1], gateUpValues[2], gateUpValues[3], gateUpValues[4], gateUpValues[5] },
            { gateUpValues[6], gateUpValues[7], gateUpValues[8], gateUpValues[9], gateUpValues[10], gateUpValues[11] },
        });
        using var activated = new Tensor(allocator, DType.Float32, 2, 3);
        Assert.True(CudaFusedOps.TrySwiGluOaiSplit(activated, gateUp, 3, alpha, limit));

        float[] expected = new float[6];
        for (int row = 0; row < 2; row++)
        {
            for (int col = 0; col < 3; col++)
            {
                float x = MathF.Min(gateUpValues[row * 6 + col], limit);
                float y = MathF.Min(MathF.Max(gateUpValues[row * 6 + 3 + col], -limit), limit);
                float sig = 1.0f / (1.0f + MathF.Exp(-alpha * x));
                expected[row * 3 + col] = x * sig * (y + 1.0f);
            }
        }

        AssertClose(expected, activated.GetElementsAsFloat(6), 1e-5f);
    }

    [Fact]
    public void CudaAttentionSoftmaxWithSinksAndWindow_MatchesReference()
    {
        if (!CudaBackend.IsAvailable())
            return;

        const int heads = 2;
        const int seqLen = 3;
        const int kvLen = 6;
        const int maskStart = 3;
        const int windowSize = 3;
        const float scale = 0.5f;

        float[,,] scores = new float[heads, seqLen, kvLen];
        for (int h = 0; h < heads; h++)
            for (int q = 0; q < seqLen; q++)
                for (int k = 0; k < kvLen; k++)
                    scores[h, q, k] = MathF.Sin((h + 1) * (q + 2) * (k + 3) * 0.11f);

        float[] sinksHost = { 0.25f, -0.35f };
        using var allocator = new CudaAllocator();
        using var scoreTensor = Tensor.FromArray(allocator, scores);
        using var sinks = Tensor.FromArray(allocator, sinksHost);
        Assert.True(CudaFusedOps.TryAttentionSoftmaxWithSinks(
            scoreTensor, sinks, heads, seqLen, kvLen, maskStart, windowSize, scale));

        float[] expected = AttentionSoftmaxWithSinksReference(scores, sinksHost, heads, seqLen, kvLen, maskStart, windowSize, scale);
        AssertClose(expected, scoreTensor.GetElementsAsFloat(heads * seqLen * kvLen), 1e-6f);
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

    private static byte[] CreateKQuantRows(int ggmlType, int rows, int cols)
    {
        return ggmlType switch
        {
            (int)GgmlTensorType.Q4_K => CreateQ4KRows(rows, cols),
            (int)GgmlTensorType.Q5_K => CreateQ5KRows(rows, cols),
            (int)GgmlTensorType.Q6_K => CreateQ6KRows(rows, cols),
            _ => throw new ArgumentOutOfRangeException(nameof(ggmlType)),
        };
    }

    private static DequantizeKRow GetKQuantDequantizer(int ggmlType)
    {
        return ggmlType switch
        {
            (int)GgmlTensorType.Q4_K => DequantizeQ4KRow,
            (int)GgmlTensorType.Q5_K => DequantizeQ5KRow,
            (int)GgmlTensorType.Q6_K => DequantizeQ6KRow,
            _ => throw new ArgumentOutOfRangeException(nameof(ggmlType)),
        };
    }

    private static byte[] CreateQ4KRows(int rows, int cols)
    {
        const int blockSize = 256;
        const int blockBytes = 144;
        Assert.Equal(0, cols % blockSize);
        int blocksPerRow = cols / blockSize;
        byte[] raw = new byte[rows * blocksPerRow * blockBytes];
        for (int r = 0; r < rows; r++)
        {
            for (int b = 0; b < blocksPerRow; b++)
            {
                int offset = (r * blocksPerRow + b) * blockBytes;
                WriteHalf(raw, offset, 0.03125f + r * 0.015625f);
                WriteHalf(raw, offset + 2, 0.015625f + r * 0.0078125f);
                for (int i = 0; i < 12; i++)
                    raw[offset + 4 + i] = (byte)((r * 17 + b * 13 + i * 7 + 19) & 0xFF);
                for (int i = 0; i < 128; i++)
                    raw[offset + 16 + i] = (byte)((r * 23 + b * 11 + i * 5 + 3) & 0xFF);
            }
        }

        return raw;
    }

    private static byte[] CreateQ5KRows(int rows, int cols)
    {
        const int blockSize = 256;
        const int blockBytes = 176;
        Assert.Equal(0, cols % blockSize);
        int blocksPerRow = cols / blockSize;
        byte[] raw = new byte[rows * blocksPerRow * blockBytes];
        for (int r = 0; r < rows; r++)
        {
            for (int b = 0; b < blocksPerRow; b++)
            {
                int offset = (r * blocksPerRow + b) * blockBytes;
                WriteHalf(raw, offset, 0.0234375f + r * 0.0078125f);
                WriteHalf(raw, offset + 2, 0.01171875f + r * 0.00390625f);
                for (int i = 0; i < 12; i++)
                    raw[offset + 4 + i] = (byte)((r * 19 + b * 7 + i * 9 + 5) & 0xFF);
                for (int i = 0; i < 32; i++)
                    raw[offset + 16 + i] = (byte)((r * 29 + b * 3 + i * 17 + 1) & 0xFF);
                for (int i = 0; i < 128; i++)
                    raw[offset + 48 + i] = (byte)((r * 31 + b * 5 + i * 3 + 7) & 0xFF);
            }
        }

        return raw;
    }

    private static byte[] CreateQ6KRows(int rows, int cols)
    {
        const int blockSize = 256;
        const int blockBytes = 210;
        Assert.Equal(0, cols % blockSize);
        int blocksPerRow = cols / blockSize;
        byte[] raw = new byte[rows * blocksPerRow * blockBytes];
        for (int r = 0; r < rows; r++)
        {
            for (int b = 0; b < blocksPerRow; b++)
            {
                int offset = (r * blocksPerRow + b) * blockBytes;
                int qlOffsetBase = offset;
                int qhOffsetBase = offset + 128;
                int scalesOffset = offset + 192;
                for (int sub = 0; sub < 16; sub++)
                {
                    int scale = ((r * 3 + b * 5 + sub * 2) % 9) - 4;
                    if (scale == 0)
                        scale = 3;
                    raw[scalesOffset + sub] = unchecked((byte)(sbyte)scale);
                    for (int i = 0; i < 16; i++)
                    {
                        int signed = ((r * 17 + b * 13 + sub * 11 + i * 7) % 63) - 31;
                        WriteQ6Value(raw, qlOffsetBase, qhOffsetBase, sub, i, signed + 32);
                    }
                }

                WriteHalf(raw, offset + 208, 0.015625f + r * 0.00390625f);
            }
        }

        return raw;
    }

    private static byte[] CreateIq2XxsRows(int rows, int cols)
    {
        const int blockSize = 256;
        const int blockBytes = 66;
        Assert.Equal(0, cols % blockSize);
        int blocksPerRow = cols / blockSize;
        byte[] raw = new byte[rows * blocksPerRow * blockBytes];
        for (int r = 0; r < rows; r++)
        {
            for (int b = 0; b < blocksPerRow; b++)
            {
                int offset = (r * blocksPerRow + b) * blockBytes;
                WriteHalf(raw, offset, 0.0078125f + r * 0.001953125f + b * 0.0009765625f);
                for (int i = 0; i < 64; i++)
                    raw[offset + 2 + i] = (byte)((r * 29 + b * 17 + i * 11 + 7) & 0xFF);
            }
        }

        return raw;
    }

    private static void WriteQ6Value(byte[] raw, int qlBase, int qhBase, int sub, int index, int unsignedValue)
    {
        int half = sub / 8;
        int sh = sub % 8;
        int qlOffset = qlBase + half * 64 + (sh % 4) * 16 + index;
        bool isUpper = sh >= 4;
        int qhOffset = qhBase + half * 32 + (sh % 2) * 16 + index;
        int qhShift = (sh / 2) * 2;
        int lo4 = unsignedValue & 0x0F;
        int hi2 = (unsignedValue >> 4) & 0x03;
        if (isUpper)
            raw[qlOffset] = (byte)((raw[qlOffset] & 0x0F) | (lo4 << 4));
        else
            raw[qlOffset] = (byte)((raw[qlOffset] & 0xF0) | lo4);
        raw[qhOffset] = (byte)((raw[qhOffset] & ~(0x03 << qhShift)) | (hi2 << qhShift));
    }

    private static void WriteHalf(byte[] data, int offset, float value)
    {
        ushort bits = BitConverter.HalfToUInt16Bits((System.Half)value);
        data[offset] = (byte)(bits & 0xFF);
        data[offset + 1] = (byte)(bits >> 8);
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

    private delegate void DequantizeKRow(byte[] weights, int row, int inDim, float[] destination, int destinationOffset);

    private static float[] DequantizedMatmulK(byte[] weights, int outDim, int inDim, float[,] input, DequantizeKRow dequantize)
    {
        int rows = input.GetLength(0);
        float[] expected = new float[rows * outDim];
        float[] dequantizedRow = new float[inDim];
        for (int o = 0; o < outDim; o++)
        {
            dequantize(weights, o, inDim, dequantizedRow, 0);
            for (int r = 0; r < rows; r++)
            {
                float sum = 0;
                for (int c = 0; c < inDim; c++)
                    sum += input[r, c] * dequantizedRow[c];
                expected[r * outDim + o] = sum;
            }
        }

        return expected;
    }

    private static float[] DequantizedMatmulNative(byte[] weights, GgmlTensorType type, int outDim, int inDim, float[,] input)
    {
        int rows = input.GetLength(0);
        float[] expected = new float[rows * outDim];
        float[] dequantizedRow = new float[inDim];
        long rowBytes = NativeDequant.RowSize((int)type, inDim);
        for (int o = 0; o < outDim; o++)
        {
            NativeDequant.DequantizeToFloat32((int)type, weights, (int)(o * rowBytes), dequantizedRow, 0, inDim);
            for (int r = 0; r < rows; r++)
            {
                float sum = 0;
                for (int c = 0; c < inDim; c++)
                    sum += input[r, c] * dequantizedRow[c];
                expected[r * outDim + o] = sum;
            }
        }

        return expected;
    }

    private static float[,] QuantizeDequantizeQ8_1(float[,] input)
    {
        int rows = input.GetLength(0);
        int cols = input.GetLength(1);
        Assert.Equal(0, cols % 32);
        float[,] result = new float[rows, cols];
        for (int r = 0; r < rows; r++)
        {
            for (int block = 0; block < cols; block += 32)
            {
                float amax = 0;
                for (int i = 0; i < 32; i++)
                    amax = MathF.Max(amax, MathF.Abs(input[r, block + i]));

                float d = amax > 0 ? amax / 127.0f : 0;
                float id = d > 0 ? 1.0f / d : 0;
                for (int i = 0; i < 32; i++)
                {
                    int q = (int)MathF.Round(input[r, block + i] * id, MidpointRounding.ToEven);
                    q = Math.Max(-127, Math.Min(127, q));
                    result[r, block + i] = d * q;
                }
            }
        }

        return result;
    }

    private static float[] DequantizeNativeRows(byte[] weights, GgmlTensorType type, int inDim, int[] rows)
    {
        float[] expected = new float[rows.Length * inDim];
        long rowBytes = NativeDequant.RowSize((int)type, inDim);
        for (int i = 0; i < rows.Length; i++)
            NativeDequant.DequantizeToFloat32((int)type, weights, (int)(rows[i] * rowBytes), expected, i * inDim, inDim);
        return expected;
    }

    private static void DequantizeQ4KRow(byte[] weights, int row, int inDim, float[] destination, int destinationOffset)
    {
        const int blockSize = 256;
        const int blockBytes = 144;
        int blocksPerRow = inDim / blockSize;
        for (int b = 0; b < blocksPerRow; b++)
        {
            int offset = (row * blocksPerRow + b) * blockBytes;
            float d = (float)BitConverter.UInt16BitsToHalf((ushort)(weights[offset] | (weights[offset + 1] << 8)));
            float min = (float)BitConverter.UInt16BitsToHalf((ushort)(weights[offset + 2] | (weights[offset + 3] << 8)));
            int scalesOffset = offset + 4;
            int qOffset = offset + 16;
            int group = 0;
            for (int j = 0; j < blockSize; j += 64)
            {
                GetScaleMinK4(group, weights, scalesOffset, out byte sc1, out byte m1q);
                GetScaleMinK4(group + 1, weights, scalesOffset, out byte sc2, out byte m2q);
                float d1 = d * sc1;
                float d2 = d * sc2;
                float m1 = min * m1q;
                float m2 = min * m2q;
                for (int l = 0; l < 32; l++)
                    destination[destinationOffset + b * blockSize + j + l] = d1 * (weights[qOffset + l] & 0x0F) - m1;
                for (int l = 0; l < 32; l++)
                    destination[destinationOffset + b * blockSize + j + l + 32] = d2 * (weights[qOffset + l] >> 4) - m2;
                qOffset += 32;
                group += 2;
            }
        }
    }

    private static void DequantizeQ5KRow(byte[] weights, int row, int inDim, float[] destination, int destinationOffset)
    {
        const int blockSize = 256;
        const int blockBytes = 176;
        int blocksPerRow = inDim / blockSize;
        for (int b = 0; b < blocksPerRow; b++)
        {
            int offset = (row * blocksPerRow + b) * blockBytes;
            float d = (float)BitConverter.UInt16BitsToHalf((ushort)(weights[offset] | (weights[offset + 1] << 8)));
            float min = (float)BitConverter.UInt16BitsToHalf((ushort)(weights[offset + 2] | (weights[offset + 3] << 8)));
            int scalesOffset = offset + 4;
            int qhOffset = offset + 16;
            int qlOffset = offset + 48;
            byte u1 = 1;
            byte u2 = 2;
            int group = 0;
            for (int j = 0; j < blockSize; j += 64)
            {
                GetScaleMinK4(group, weights, scalesOffset, out byte sc1, out byte m1q);
                GetScaleMinK4(group + 1, weights, scalesOffset, out byte sc2, out byte m2q);
                float d1 = d * sc1;
                float d2 = d * sc2;
                float m1 = min * m1q;
                float m2 = min * m2q;
                for (int l = 0; l < 32; l++)
                {
                    int lo = (weights[qlOffset + l] & 0x0F) + ((weights[qhOffset + l] & u1) != 0 ? 16 : 0);
                    int hi = (weights[qlOffset + l] >> 4) + ((weights[qhOffset + l] & u2) != 0 ? 16 : 0);
                    destination[destinationOffset + b * blockSize + j + l] = d1 * lo - m1;
                    destination[destinationOffset + b * blockSize + j + l + 32] = d2 * hi - m2;
                }

                qlOffset += 32;
                group += 2;
                u1 <<= 2;
                u2 <<= 2;
            }
        }
    }

    private static void DequantizeQ6KRow(byte[] weights, int row, int inDim, float[] destination, int destinationOffset)
    {
        const int blockSize = 256;
        const int blockBytes = 210;
        int blocksPerRow = inDim / blockSize;
        for (int b = 0; b < blocksPerRow; b++)
        {
            int offset = (row * blocksPerRow + b) * blockBytes;
            int qlBase = offset;
            int qhBase = offset + 128;
            int scalesBase = offset + 192;
            float d = (float)BitConverter.UInt16BitsToHalf((ushort)(weights[offset + 208] | (weights[offset + 209] << 8)));
            for (int sub = 0; sub < 16; sub++)
            {
                int half = sub / 8;
                int sh = sub % 8;
                int qlOffset = qlBase + half * 64 + (sh % 4) * 16;
                bool isUpper = sh >= 4;
                int qhOffset = qhBase + half * 32 + (sh % 2) * 16;
                int qhShift = (sh / 2) * 2;
                float scale = d * unchecked((sbyte)weights[scalesBase + sub]);
                for (int i = 0; i < 16; i++)
                {
                    int lo4 = isUpper ? (weights[qlOffset + i] >> 4) & 0x0F : weights[qlOffset + i] & 0x0F;
                    int hi2 = (weights[qhOffset + i] >> qhShift) & 0x03;
                    int q6 = (lo4 | (hi2 << 4)) - 32;
                    destination[destinationOffset + b * blockSize + sub * 16 + i] = scale * q6;
                }
            }
        }
    }

    private static void GetScaleMinK4(int index, byte[] packed, int offset, out byte scale, out byte min)
    {
        if (index < 4)
        {
            scale = (byte)(packed[offset + index] & 63);
            min = (byte)(packed[offset + index + 4] & 63);
            return;
        }

        scale = (byte)((packed[offset + index + 4] & 0x0F) | ((packed[offset + index - 4] >> 6) << 4));
        min = (byte)((packed[offset + index + 4] >> 4) | ((packed[offset + index] >> 6) << 4));
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
        int windowSize,
        float scale = 1.0f,
        float[] sinks = null)
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
                float max = sinks != null ? sinks[h] : float.NegativeInfinity;
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
                    scores[tk] = dot * scale;
                    max = MathF.Max(max, scores[tk]);
                }

                float sum = sinks != null ? MathF.Exp(sinks[h] - max) : 0;
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
        bool circular,
        float scale = 1.0f,
        float[] sinks = null)
    {
        int groupSize = numQHeads / numKVHeads;
        float[] result = new float[numQHeads * headDim];

        for (int h = 0; h < numQHeads; h++)
        {
            int kvHead = h / groupSize;
            float[] scores = new float[attendLen];
            float max = sinks != null ? sinks[h] : float.NegativeInfinity;
            for (int t = 0; t < attendLen; t++)
            {
                int logical = attendStart + t;
                int cachePos = circular ? logical % cacheSize : logical;
                float dot = 0;
                for (int d = 0; d < headDim; d++)
                    dot += q[h * headDim + d] * k[(kvHead * cacheSize + cachePos) * headDim + d];
                scores[t] = dot * scale;
                max = MathF.Max(max, scores[t]);
            }

            float sum = sinks != null ? MathF.Exp(sinks[h] - max) : 0;
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

    private static float[] AttentionSoftmaxWithSinksReference(
        float[,,] scores,
        float[] sinks,
        int heads,
        int seqLen,
        int kvLen,
        int maskStart,
        int windowSize,
        float scale)
    {
        float[] result = new float[heads * seqLen * kvLen];
        for (int h = 0; h < heads; h++)
        {
            for (int q = 0; q < seqLen; q++)
            {
                int visible = maskStart + q;
                int minVisible = windowSize > 0 ? Math.Max(0, visible - windowSize + 1) : 0;
                float max = sinks != null ? sinks[h] : float.NegativeInfinity;
                for (int k = 0; k < kvLen; k++)
                {
                    if (k <= visible && k >= minVisible)
                        max = MathF.Max(max, scores[h, q, k] * scale);
                }

                float sum = sinks != null ? MathF.Exp(sinks[h] - max) : 0;
                for (int k = 0; k < kvLen; k++)
                {
                    int offset = (h * seqLen + q) * kvLen + k;
                    if (k <= visible && k >= minVisible)
                    {
                        float p = MathF.Exp(scores[h, q, k] * scale - max);
                        result[offset] = p;
                        sum += p;
                    }
                    else
                    {
                        result[offset] = 0;
                    }
                }

                for (int k = 0; k < kvLen; k++)
                    result[(h * seqLen + q) * kvLen + k] /= sum;
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

    private static float[] Qwen35GdnPackedReference(
        float[] packed,
        float[] convState,
        float[] ssmState,
        float[] convWeight,
        float[] dtBias,
        float[] aLog,
        float[] ssmNorm,
        int seqLen,
        int packedDim,
        int qkvDim,
        int qkDim,
        int vDim,
        int numKHeads,
        int numVHeads,
        int headKDim,
        int headVDim,
        int convKernel,
        int convWriteIdx,
        float eps)
    {
        int convDim = convKernel - 1;
        int zDim = vDim;
        float[] output = new float[seqLen * vDim];
        float[] convOut = new float[qkvDim];
        float[] q = new float[numVHeads * headKDim];
        float[] k = new float[numVHeads * headKDim];
        float[] core = new float[headVDim];
        int writeIdx = convWriteIdx;

        for (int s = 0; s < seqLen; s++)
        {
            int packedBase = s * packedDim;
            for (int ch = 0; ch < qkvDim; ch++)
            {
                float acc = 0;
                for (int ki = 0; ki < convKernel; ki++)
                {
                    float x;
                    if (ki < convDim)
                    {
                        int slot = (writeIdx + ki) % convDim;
                        x = convState[slot * qkvDim + ch];
                    }
                    else
                    {
                        x = packed[packedBase + ch];
                    }
                    acc += x * convWeight[ch * convKernel + ki];
                }
                convOut[ch] = Silu(acc);
            }

            if (convDim > 0)
            {
                Array.Copy(packed, packedBase, convState, writeIdx * qkvDim, qkvDim);
                writeIdx = (writeIdx + 1) % convDim;
            }

            for (int h = 0; h < numVHeads; h++)
            {
                int srcHead = h % numKHeads;
                Array.Copy(convOut, srcHead * headKDim, q, h * headKDim, headKDim);
                Array.Copy(convOut, qkDim + srcHead * headKDim, k, h * headKDim, headKDim);
            }

            for (int h = 0; h < numVHeads; h++)
            {
                float qSum = 0;
                float kSum = 0;
                int headBase = h * headKDim;
                for (int d = 0; d < headKDim; d++)
                {
                    qSum += q[headBase + d] * q[headBase + d];
                    kSum += k[headBase + d] * k[headBase + d];
                }

                float qScale = 1.0f / MathF.Sqrt(qSum + eps) / MathF.Sqrt(headVDim);
                float kScale = 1.0f / MathF.Sqrt(kSum + eps);
                for (int d = 0; d < headKDim; d++)
                {
                    q[headBase + d] *= qScale;
                    k[headBase + d] *= kScale;
                }
            }

            for (int h = 0; h < numVHeads; h++)
            {
                int qkBase = h * headKDim;
                int vBase = 2 * qkDim + h * headVDim;
                int zBase = qkvDim + h * headVDim;
                int stateHeadBase = h * headVDim * headKDim;
                float gate = Softplus(packed[packedBase + qkvDim + zDim + numVHeads + h] + dtBias[h]) * aLog[h];
                float stateScale = MathF.Exp(gate);
                float beta = Sigmoid(packed[packedBase + qkvDim + zDim + h]);

                for (int i = 0; i < headVDim * headKDim; i++)
                    ssmState[stateHeadBase + i] *= stateScale;

                for (int row = 0; row < headVDim; row++)
                {
                    int stateRow = stateHeadBase + row * headKDim;
                    float kvMem = 0;
                    for (int d = 0; d < headKDim; d++)
                        kvMem += ssmState[stateRow + d] * k[qkBase + d];
                    float delta = (convOut[vBase + row] - kvMem) * beta;
                    for (int d = 0; d < headKDim; d++)
                        ssmState[stateRow + d] += k[qkBase + d] * delta;

                    float coreValue = 0;
                    for (int d = 0; d < headKDim; d++)
                        coreValue += ssmState[stateRow + d] * q[qkBase + d];
                    core[row] = coreValue;
                }

                float sumSq = 0;
                for (int row = 0; row < headVDim; row++)
                    sumSq += core[row] * core[row];
                float rmsInv = 1.0f / MathF.Sqrt(sumSq / headVDim + eps);

                for (int row = 0; row < headVDim; row++)
                {
                    output[s * vDim + h * headVDim + row] =
                        core[row] * rmsInv * ssmNorm[row] * Silu(packed[packedBase + zBase + row]);
                }
            }
        }

        return output;
    }

    private static float[] MakePattern(int length, float scale, float offset)
    {
        float[] values = new float[length];
        for (int i = 0; i < length; i++)
            values[i] = offset + MathF.Sin(i * 0.37f) * scale + MathF.Cos(i * 0.13f) * (scale * 0.5f);
        return values;
    }

    private static float Silu(float x) => x / (1.0f + MathF.Exp(-x));

    private static float Sigmoid(float x) => 1.0f / (1.0f + MathF.Exp(-x));

    private static float Softplus(float x)
    {
        return x > 0.0f ? x + MathF.Log(1.0f + MathF.Exp(-x)) : MathF.Log(1.0f + MathF.Exp(x));
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
