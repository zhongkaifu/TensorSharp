using System.Runtime.InteropServices;
using TensorSharp;
using TensorSharp.MLX;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

public class MlxBackendTests
{
    [Fact]
    public void MlxAvailabilityProbe_DoesNotThrow()
    {
        _ = MlxBackend.IsAvailable();
    }

    [Fact]
    public void MlxNativeDylibs_LoadOnAppleSiliconWhenCopied()
    {
        if (!OperatingSystem.IsMacOS() || RuntimeInformation.ProcessArchitecture != Architecture.Arm64)
            return;
        if (!File.Exists(Path.Combine(AppContext.BaseDirectory, "libmlxc.dylib")))
            return;

        Assert.True(MlxBackend.IsAvailable());
    }

    [Fact]
    public void MlxAddmm_MatchesCpuForContiguousRhs()
    {
        if (!MlxBackend.IsAvailable())
            return;

        using var allocator = new MlxAllocator();
        using var a = Tensor.FromArray(allocator, new float[,] { { 1, 2, 3 }, { 4, 5, 6 } });
        using var b = Tensor.FromArray(allocator, new float[,] { { 7, 8 }, { 9, 10 }, { 11, 12 } });
        using var c = new Tensor(allocator, DType.Float32, 2, 2);

        Ops.Addmm(c, 0, c, 1, a, b);

        AssertClose(new[] { 58f, 64f, 139f, 154f }, c.GetElementsAsFloat(4));
    }

    [Fact]
    public void MlxSoftmax_MatchesExpectedRows()
    {
        if (!MlxBackend.IsAvailable())
            return;

        using var allocator = new MlxAllocator();
        using var logits = Tensor.FromArray(allocator, new float[,] { { 1, 2, 3 }, { -2, 0, 2 } });
        using var probs = new Tensor(allocator, DType.Float32, 2, 3);

        Ops.Softmax(probs, logits);

        AssertClose(new[]
        {
            0.09003057f, 0.24472848f, 0.66524094f,
            0.01587624f, 0.11731043f, 0.86681333f,
        }, probs.GetElementsAsFloat(6), 1e-5f);
    }

    [Fact]
    public void MlxMul_BroadcastsColumnVector()
    {
        if (!MlxBackend.IsAvailable())
            return;

        using var allocator = new MlxAllocator();
        using var values = Tensor.FromArray(allocator, new float[,]
        {
            { 1, 2, 3 },
            { 4, 5, 6 },
        });
        using var gate = Tensor.FromArray(allocator, new float[,]
        {
            { 2 },
            { -1 },
        });
        using var output = new Tensor(allocator, DType.Float32, 2, 3);

        Ops.Mul(output, values, gate);

        AssertClose(new[] { 2f, 4f, 6f, -4f, -5f, -6f }, output.GetElementsAsFloat(6));
    }

    [Fact]
    public void MlxHostWriteAfterDeviceOp_PreservesTensorData()
    {
        if (!MlxBackend.IsAvailable())
            return;

        using var allocator = new MlxAllocator();
        using var tensor = Tensor.FromArray(allocator, new float[,] { { 1, 2 }, { 3, 4 } });
        Ops.Mul(tensor, tensor, 2f);

        tensor.SetElementAsFloat(100f, 0, 1);

        AssertClose(new[] { 2f, 100f, 6f, 8f }, tensor.GetElementsAsFloat(4));
    }

    [Fact]
    public void MlxCopyIntoNarrowView_UpdatesDeviceStorage()
    {
        if (!MlxBackend.IsAvailable())
            return;

        using var allocator = new MlxAllocator();
        using var tensor = Tensor.FromArray(allocator, new float[,]
        {
            { 0, 1, 2, 3 },
            { 4, 5, 6, 7 },
            { 8, 9, 10, 11 },
        });
        Ops.Mul(tensor, tensor, 2f);

        using var staged = Tensor.FromArray(allocator, new float[,] { { 100, 101, 102, 103 } });
        using (var row = tensor.Narrow(0, 1, 1))
            Ops.Copy(row, staged);

        AssertClose(new[]
        {
            0f, 2f, 4f, 6f,
            100f, 101f, 102f, 103f,
            16f, 18f, 20f, 22f,
        }, tensor.GetElementsAsFloat(12));
    }

    [Fact]
    public void MlxFill_WritesDeviceResidentTensor()
    {
        if (!MlxBackend.IsAvailable())
            return;

        using var allocator = new MlxAllocator();
        using var tensor = new Tensor(allocator, DType.Float32, 2, 3);

        Ops.Fill(tensor, 1.25f);

        AssertClose(new[] { 1.25f, 1.25f, 1.25f, 1.25f, 1.25f, 1.25f }, tensor.GetElementsAsFloat(6));
    }

    [Fact]
    public void MlxRmsNorm_MatchesExpectedRows()
    {
        if (!MlxBackend.IsAvailable())
            return;

        using var allocator = new MlxAllocator();
        using var input = Tensor.FromArray(allocator, new float[,] { { 3, 4 }, { 5, 12 } });
        using var weight = Tensor.FromArray(allocator, new float[] { 1f, 0.5f });
        using var output = new Tensor(allocator, DType.Float32, 2, 2);

        Ops.RMSNorm(output, input, weight, null, 1e-6f);

        AssertClose(new[]
        {
            0.8485281f, 0.5656854f,
            0.54392827f, 0.65271395f,
        }, output.GetElementsAsFloat(4), 1e-4f);
    }

    [Fact]
    public void MlxLayerNorm_MatchesExpectedRows()
    {
        if (!MlxBackend.IsAvailable())
            return;

        using var allocator = new MlxAllocator();
        using var input = Tensor.FromArray(allocator, new float[,] { { 1, 2, 3 }, { -1, 1, 3 } });
        using var weight = Tensor.FromArray(allocator, new float[] { 1f, 2f, 0.5f });
        using var bias = Tensor.FromArray(allocator, new float[] { 0f, 1f, -1f });
        using var output = new Tensor(allocator, DType.Float32, 2, 3);

        Ops.LayerNorm(output, input, weight, bias, 1e-6f);

        AssertClose(new[]
        {
            -1.2247439f, 1f, -0.387628f,
            -1.2247446f, 1f, -0.3876277f,
        }, output.GetElementsAsFloat(6), 1e-4f);
    }

    [Fact]
    public void MlxGeluAndGeluMul_MatchCpuFormula()
    {
        if (!MlxBackend.IsAvailable())
            return;

        float[,] gate =
        {
            { -2.0f, -0.5f, 0.0f },
            { 0.75f, 1.25f, 2.5f },
        };
        float[,] up =
        {
            { 1.5f, -2.0f, 0.25f },
            { -0.75f, 1.0f, 0.5f },
        };

        float[] expectedGelu = new float[gate.Length];
        float[] expectedMul = new float[gate.Length];
        int index = 0;
        for (int r = 0; r < gate.GetLength(0); r++)
            for (int c = 0; c < gate.GetLength(1); c++, index++)
            {
                float gelu = GeluReference(gate[r, c]);
                expectedGelu[index] = gelu;
                expectedMul[index] = gelu * up[r, c];
            }

        using var allocator = new MlxAllocator();
        using var gateTensor = Tensor.FromArray(allocator, gate);
        using var upTensor = Tensor.FromArray(allocator, up);
        using var geluTensor = Ops.GELU(null, gateTensor);
        using var mulTensor = Ops.GELUMul(null, gateTensor, upTensor);

        AssertClose(expectedGelu, geluTensor.GetElementsAsFloat(expectedGelu.Length), 1e-5f);
        AssertClose(expectedMul, mulTensor.GetElementsAsFloat(expectedMul.Length), 1e-5f);
    }

    [Fact]
    public void MlxSiLUMulSplit_MatchesSplitReference()
    {
        if (!MlxBackend.IsAvailable())
            return;

        float[,] gateUp =
        {
            { -2.0f, -0.5f, 0.0f, 1.5f, -2.0f, 0.25f },
            { 0.75f, 1.25f, 2.5f, -0.75f, 1.0f, 0.5f },
        };
        float[] expected = new float[6];
        int index = 0;
        for (int r = 0; r < gateUp.GetLength(0); r++)
            for (int c = 0; c < 3; c++, index++)
                expected[index] = gateUp[r, c] / (1.0f + MathF.Exp(-gateUp[r, c])) * gateUp[r, c + 3];

        using var allocator = new MlxAllocator();
        using var gateUpTensor = Tensor.FromArray(allocator, gateUp);
        using var actualTensor = Ops.SiLUMulSplit(null, gateUpTensor, 3);

        AssertClose(expected, actualTensor.GetElementsAsFloat(expected.Length), 1e-5f);
    }

    [Fact]
    public void MlxScaledDotProductAttention_MatchesReferenceWithoutMask()
    {
        if (!MlxBackend.IsAvailable())
            return;

        const int batch = 1;
        const int seqQ = 3;
        const int seqK = 4;
        const int heads = 2;
        const int keyDim = 3;
        const int valueDim = 2;
        const float scale = 0.57735026f;

        float[,,,] q = BuildAttentionInput(batch, seqQ, heads, keyDim, 0.031f, useCos: false);
        float[,,,] k = BuildAttentionInput(batch, seqK, heads, keyDim, 0.027f, useCos: true);
        float[,,,] v = BuildAttentionInput(batch, seqK, heads, valueDim, 0.019f, useCos: false);

        using var allocator = new MlxAllocator();
        using var qTensor = Tensor.FromArray(allocator, q);
        using var kTensor = Tensor.FromArray(allocator, k);
        using var vTensor = Tensor.FromArray(allocator, v);
        using var actualTensor = Ops.ScaledDotProductAttention(null, qTensor, kTensor, vTensor, null, scale);

        AssertClose(ScaledDotProductAttentionReference(q, k, v, null, scale), actualTensor.GetElementsAsFloat((int)actualTensor.ElementCount()), 1e-4f);
    }

    [Fact]
    public void MlxScaledDotProductAttention_MatchesReferenceWithMask()
    {
        if (!MlxBackend.IsAvailable())
            return;

        const int batch = 1;
        const int seqQ = 3;
        const int seqK = 4;
        const int heads = 2;
        const int keyDim = 3;
        const int valueDim = 2;
        const float scale = 0.57735026f;

        float[,,,] q = BuildAttentionInput(batch, seqQ, heads, keyDim, 0.031f, useCos: false);
        float[,,,] k = BuildAttentionInput(batch, seqK, heads, keyDim, 0.027f, useCos: true);
        float[,,,] v = BuildAttentionInput(batch, seqK, heads, valueDim, 0.019f, useCos: false);
        float[,,,] mask = new float[batch, heads, seqQ, seqK];
        for (int b = 0; b < batch; b++)
            for (int h = 0; h < heads; h++)
                for (int tq = 0; tq < seqQ; tq++)
                    for (int tk = 0; tk < seqK; tk++)
                        mask[b, h, tq, tk] = tk > tq + 1 ? float.NegativeInfinity : 0.01f * (b - h);

        using var allocator = new MlxAllocator();
        using var qTensor = Tensor.FromArray(allocator, q);
        using var kTensor = Tensor.FromArray(allocator, k);
        using var vTensor = Tensor.FromArray(allocator, v);
        using var maskTensor = Tensor.FromArray(allocator, mask);
        using var actualTensor = Ops.ScaledDotProductAttention(null, qTensor, kTensor, vTensor, maskTensor, scale);

        AssertClose(ScaledDotProductAttentionReference(q, k, v, mask, scale), actualTensor.GetElementsAsFloat((int)actualTensor.ElementCount()), 1e-4f);
    }

    [Fact]
    public void MlxCopy_CastsFloat32ToFloat16OnDevice()
    {
        if (!MlxBackend.IsAvailable())
            return;

        using var allocator = new MlxAllocator();
        using var src = Tensor.FromArray(allocator, new float[] { -1.25f, -0.5f, 0f, 0.75f, 1.5f, 3.25f });
        using var dst = new Tensor(allocator, DType.Float16, 6);

        Ops.Copy(dst, src);

        AssertClose(src.GetElementsAsFloat(6), dst.GetElementsAsFloat(6), 1e-3f);
    }

    [Fact]
    public void MlxFusedPrefillAttention_GqaMatchesReference()
    {
        if (!MlxBackend.IsAvailable())
            return;

        const int heads = 4;
        const int kvHeads = 2;
        const int seq = 3;
        const int dim = 256;
        const float scale = 0.125f;

        float[,,] q = BuildHeadFirstInput(heads, seq, dim, 0.037f, useCos: false);
        float[,,] k = BuildHeadFirstInput(kvHeads, seq, dim, 0.029f, useCos: true);
        float[,,] v = BuildHeadFirstInput(kvHeads, seq, dim, 0.021f, useCos: false);

        using var allocator = new MlxAllocator();
        using var qTensor = Tensor.FromArray(allocator, q);
        using var kTensor = Tensor.FromArray(allocator, k);
        using var vTensor = Tensor.FromArray(allocator, v);
        using var actualTensor = new Tensor(allocator, DType.Float32, seq, heads * dim);

        Assert.True(MlxFusedOps.TryPrefillAttention(
            actualTensor,
            qTensor,
            kTensor,
            vTensor,
            heads,
            kvHeads,
            dim,
            seq,
            seq,
            0,
            0,
            scale));

        AssertClose(HeadFirstAttentionReference(q, k, v, scale, causal: true), actualTensor.GetElementsAsFloat((int)actualTensor.ElementCount()), 1e-4f);
    }

    [Fact]
    public void MlxFusedPrefillAttention_HeadDim256UsesChunkedVectorPath()
    {
        if (!MlxBackend.IsAvailable())
            return;

        const int heads = 24;
        const int kvHeads = 4;
        const int seq = 11;
        const int dim = 256;
        const float scale = 0.0625f;

        float[,,] q = BuildHeadFirstInput(heads, seq, dim, 0.0037f, useCos: false);
        float[,,] k = BuildHeadFirstInput(kvHeads, seq, dim, 0.0029f, useCos: true);
        float[,,] v = BuildHeadFirstInput(kvHeads, seq, dim, 0.0021f, useCos: false);

        string previous = Environment.GetEnvironmentVariable("TS_MLX_CHUNKED_VECTOR_PREFILL");
        Environment.SetEnvironmentVariable("TS_MLX_CHUNKED_VECTOR_PREFILL", "1");
        try
        {
            using var allocator = new MlxAllocator();
            using var qTensor = Tensor.FromArray(allocator, q);
            using var kTensor = Tensor.FromArray(allocator, k);
            using var vTensor = Tensor.FromArray(allocator, v);
            using var actualTensor = new Tensor(allocator, DType.Float32, seq, heads * dim);

            Assert.True(MlxFusedOps.TryPrefillAttention(
                actualTensor,
                qTensor,
                kTensor,
                vTensor,
                heads,
                kvHeads,
                dim,
                seq,
                seq,
                0,
                0,
                scale));

            AssertClose(HeadFirstAttentionReference(q, k, v, scale, causal: true), actualTensor.GetElementsAsFloat((int)actualTensor.ElementCount()), 2e-3f);
        }
        finally
        {
            Environment.SetEnvironmentVariable("TS_MLX_CHUNKED_VECTOR_PREFILL", previous);
        }
    }

    [Fact]
    public void MlxFusedDecodeAttention_ReadsFloat16KvCache()
    {
        if (!MlxBackend.IsAvailable())
            return;

        const int heads = 4;
        const int kvHeads = 2;
        const int seq = 5;
        const int dim = 256;
        const float scale = 0.125f;

        float[,,] qHeads = BuildHeadFirstInput(heads, 1, dim, 0.041f, useCos: false);
        float[,,] k = BuildHeadFirstInput(kvHeads, seq, dim, 0.025f, useCos: true);
        float[,,] v = BuildHeadFirstInput(kvHeads, seq, dim, 0.017f, useCos: false);
        float[] qFlat = FlattenHeadFirstSingleToken(qHeads);

        using var allocator = new MlxAllocator();
        using var qTensor = Tensor.FromArray(allocator, qFlat).View(1, heads * dim);
        using var kSrc = Tensor.FromArray(allocator, k);
        using var vSrc = Tensor.FromArray(allocator, v);
        using var kCache = new Tensor(allocator, DType.Float16, kvHeads, seq, dim);
        using var vCache = new Tensor(allocator, DType.Float16, kvHeads, seq, dim);
        using var actualTensor = new Tensor(allocator, DType.Float32, 1, heads * dim);
        Ops.Copy(kCache, kSrc);
        Ops.Copy(vCache, vSrc);

        Assert.True(MlxFusedOps.TryDecodeAttention(
            actualTensor,
            qTensor,
            kCache,
            vCache,
            heads,
            kvHeads,
            dim,
            0,
            seq,
            seq,
            false,
            scale));

        AssertClose(HeadFirstAttentionReference(qHeads, k, v, scale, causal: false), actualTensor.GetElementsAsFloat((int)actualTensor.ElementCount()), 2e-3f);
    }

    [Fact]
    public void MlxFusedCircularDecodeAttention_ReadsWrappedFloat16KvCache()
    {
        if (!MlxBackend.IsAvailable())
            return;

        const int heads = 4;
        const int kvHeads = 2;
        const int cacheLen = 5;
        const int attendLen = 5;
        const int firstSlot = 2;
        const int dim = 256;
        const float scale = 0.125f;

        float[,,] qHeads = BuildHeadFirstInput(heads, 1, dim, 0.041f, useCos: false);
        float[,,] kChronological = BuildHeadFirstInput(kvHeads, attendLen, dim, 0.025f, useCos: true);
        float[,,] vChronological = BuildHeadFirstInput(kvHeads, attendLen, dim, 0.017f, useCos: false);
        float[,,] kCacheData = new float[kvHeads, cacheLen, dim];
        float[,,] vCacheData = new float[kvHeads, cacheLen, dim];

        for (int h = 0; h < kvHeads; h++)
            for (int t = 0; t < attendLen; t++)
            {
                int slot = (firstSlot + t) % cacheLen;
                for (int d = 0; d < dim; d++)
                {
                    kCacheData[h, slot, d] = kChronological[h, t, d];
                    vCacheData[h, slot, d] = vChronological[h, t, d];
                }
            }

        float[] qFlat = FlattenHeadFirstSingleToken(qHeads);

        using var allocator = new MlxAllocator();
        using var qTensor = Tensor.FromArray(allocator, qFlat).View(1, heads * dim);
        using var kSrc = Tensor.FromArray(allocator, kCacheData);
        using var vSrc = Tensor.FromArray(allocator, vCacheData);
        using var kCache = new Tensor(allocator, DType.Float16, kvHeads, cacheLen, dim);
        using var vCache = new Tensor(allocator, DType.Float16, kvHeads, cacheLen, dim);
        using var actualTensor = new Tensor(allocator, DType.Float32, 1, heads * dim);
        Ops.Copy(kCache, kSrc);
        Ops.Copy(vCache, vSrc);

        Assert.True(MlxFusedOps.TryDecodeAttention(
            actualTensor,
            qTensor,
            kCache,
            vCache,
            heads,
            kvHeads,
            dim,
            firstSlot,
            attendLen,
            cacheLen,
            true,
            scale));

        AssertClose(HeadFirstAttentionReference(qHeads, kChronological, vChronological, scale, causal: false),
            actualTensor.GetElementsAsFloat((int)actualTensor.ElementCount()),
            2e-3f);
    }

    [Fact]
    public void MlxIndexSelect_GathersRowsOnDevice()
    {
        if (!MlxBackend.IsAvailable())
            return;

        using var allocator = new MlxAllocator();
        using var src = Tensor.FromArray(allocator, new float[,]
        {
            { 1, 2, 3 },
            { 4, 5, 6 },
            { 7, 8, 9 },
            { 10, 11, 12 },
        });
        using var indices = Tensor.FromArray(allocator, new[] { 2, 0, 3 });
        using var selected = Ops.IndexSelect(null, src, indices);

        AssertClose(new[] { 7f, 8f, 9f, 1f, 2f, 3f, 10f, 11f, 12f }, selected.GetElementsAsFloat(9));
    }

    [Fact]
    public void MlxFusedGatherRows_GathersRowsOnDevice()
    {
        if (!MlxBackend.IsAvailable())
            return;

        using var allocator = new MlxAllocator();
        using var src = Tensor.FromArray(allocator, new float[,]
        {
            { 1, 2, 3 },
            { 4, 5, 6 },
            { 7, 8, 9 },
            { 10, 11, 12 },
        });
        using var indices = Tensor.FromArray(allocator, new[] { 2, 0, 3 });
        using var gathered = new Tensor(allocator, DType.Float32, 3, 3);

        Assert.True(MlxFusedOps.TryGatherRows(gathered, src, indices));

        AssertClose(new[] { 7f, 8f, 9f, 1f, 2f, 3f, 10f, 11f, 12f }, gathered.GetElementsAsFloat(9));
    }

    [Fact]
    public void MlxFusedScatterAddWeightedRows_AccumulatesRowsOnDevice()
    {
        if (!MlxBackend.IsAvailable())
            return;

        using var allocator = new MlxAllocator();
        using var output = Tensor.FromArray(allocator, new float[,]
        {
            { 10, 10, 10 },
            { 20, 20, 20 },
            { 30, 30, 30 },
            { 40, 40, 40 },
        });
        using var rows = Tensor.FromArray(allocator, new float[,]
        {
            { 4, 5, 6 },
            { 1, 2, 3 },
            { 7, 8, 9 },
        });
        using var indices = Tensor.FromArray(allocator, new[] { 0, 2, 3 });
        using var weights = Tensor.FromArray(allocator, new[] { 2.0f, 0.5f, -1.0f });

        Assert.True(MlxFusedOps.TryScatterAddWeightedRows(output, rows, indices, weights));

        AssertClose(new[]
        {
            18f, 20f, 22f,
            20f, 20f, 20f,
            30.5f, 31f, 31.5f,
            33f, 32f, 31f,
        }, output.GetElementsAsFloat(12));
    }

    [Fact]
    public void MlxFusedRmsNormAddInPlace_MatchesCpu()
    {
        if (!MlxBackend.IsAvailable())
            return;

        using var allocator = new MlxAllocator();
        using var residual = Tensor.FromArray(allocator, new float[,]
        {
            { 1, 2, 3, 4 },
            { 10, 20, 30, 40 },
        });
        using var input = Tensor.FromArray(allocator, new float[,]
        {
            { 1, 2, 3, 4 },
            { 2, -2, 1, -1 },
        });
        using var weight = Tensor.FromArray(allocator, new[] { 1.0f, 0.5f, -1.0f, 2.0f });

        Assert.True(MlxFusedOps.TryRmsNormAddInPlace(residual, input, weight, 1e-6f));

        float[] residualHost = { 1, 2, 3, 4, 10, 20, 30, 40 };
        float[] inputHost = { 1, 2, 3, 4, 2, -2, 1, -1 };
        float[] weightHost = { 1.0f, 0.5f, -1.0f, 2.0f };
        float[] expected = new float[8];
        for (int row = 0; row < 2; row++)
        {
            float sum = 0;
            for (int col = 0; col < 4; col++)
            {
                float value = inputHost[row * 4 + col];
                sum += value * value;
            }

            float scale = 1.0f / MathF.Sqrt(sum / 4.0f + 1e-6f);
            for (int col = 0; col < 4; col++)
            {
                int offset = row * 4 + col;
                expected[offset] = residualHost[offset] + inputHost[offset] * scale * weightHost[col];
            }
        }

        AssertClose(expected, residual.GetElementsAsFloat(8), tolerance: 1e-5f);
    }

    [Fact]
    public void MlxFusedGeluMulSplit_MatchesCpu()
    {
        if (!MlxBackend.IsAvailable())
            return;

        using var allocator = new MlxAllocator();
        using var gateUp = Tensor.FromArray(allocator, new float[,]
        {
            { -1.0f, 0.5f, 2.0f, 3.0f, -4.0f, 0.25f },
            { 1.25f, -0.75f, 0.1f, 2.0f, 3.0f, -5.0f },
        });
        using var result = new Tensor(allocator, DType.Float32, 2, 3);

        Assert.True(MlxFusedOps.TryGeluMulSplit(result, gateUp, halfDim: 3));

        float[] src =
        {
            -1.0f, 0.5f, 2.0f, 3.0f, -4.0f, 0.25f,
            1.25f, -0.75f, 0.1f, 2.0f, 3.0f, -5.0f,
        };
        float[] expected = new float[6];
        for (int row = 0; row < 2; row++)
        {
            for (int col = 0; col < 3; col++)
            {
                float gate = src[row * 6 + col];
                float up = src[row * 6 + 3 + col];
                float gelu = 0.5f * gate * (1.0f + MathF.Tanh(0.7978845608f * (gate + 0.044715f * gate * gate * gate)));
                expected[row * 3 + col] = gelu * up;
            }
        }

        AssertClose(expected, result.GetElementsAsFloat(6), tolerance: 1e-5f);
    }

    [Fact]
    public void MlxFusedFlatToHeadFirst_MatchesReference()
    {
        if (!MlxBackend.IsAvailable())
            return;

        using var allocator = new MlxAllocator();
        using var input = Tensor.FromArray(allocator, new float[,]
        {
            { 1, 2, 3, 4, 5, 6, 90 },
            { 7, 8, 9, 10, 11, 12, 91 },
        });
        using var result = new Tensor(allocator, DType.Float32, 3, 2, 2);

        Assert.True(MlxFusedOps.TryFlatToHeadFirst(result, input, numHeads: 3, seqLen: 2, headDim: 2));

        AssertClose(new[]
        {
            1f, 2f, 7f, 8f,
            3f, 4f, 9f, 10f,
            5f, 6f, 11f, 12f,
        }, result.GetElementsAsFloat(12), tolerance: 1e-5f);
    }

    [Fact]
    public void MlxFusedNeoXRoPEFlatAndHeadFirst_MatchReference()
    {
        if (!MlxBackend.IsAvailable())
            return;

        const int heads = 2;
        const int seq = 2;
        const int dim = 4;
        const int rotHalf = 2;
        float[] cos = { 1.0f, 0.0f, 0.5f, -0.25f };
        float[] sin = { 0.0f, 1.0f, 0.8660254f, 0.9682458f };
        float[] flat =
        {
            1, 2, 3, 4,
            10, 20, 30, 40,
            5, 6, 7, 8,
            50, 60, 70, 80,
        };

        float[] expectedFlat = (float[])flat.Clone();
        ApplyNeoXReference(expectedFlat, cos, sin, heads, seq, dim, rotHalf, headFirst: false);

        float[,,] headFirst =
        {
            {
                { 1, 2, 3, 4 },
                { 5, 6, 7, 8 },
            },
            {
                { 10, 20, 30, 40 },
                { 50, 60, 70, 80 },
            },
        };
        float[] expectedHeadFirst = Flatten3D(headFirst);
        ApplyNeoXReference(expectedHeadFirst, cos, sin, heads, seq, dim, rotHalf, headFirst: true);

        using var allocator = new MlxAllocator();
        using var cosTensor = Tensor.FromArray(allocator, cos);
        using var sinTensor = Tensor.FromArray(allocator, sin);
        using var flatTensor = Tensor.FromArray(allocator, flat).View(seq, heads * dim);
        using var headFirstTensor = Tensor.FromArray(allocator, headFirst);

        Assert.True(MlxFusedOps.TryNeoXRoPEFlatInPlace(flatTensor, cosTensor, sinTensor, heads, seq, dim, rotHalf));
        Assert.True(MlxFusedOps.TryNeoXRoPEHeadFirstInPlace(headFirstTensor, cosTensor, sinTensor, heads, seq, dim, rotHalf));

        AssertClose(expectedFlat, flatTensor.GetElementsAsFloat(expectedFlat.Length), tolerance: 1e-5f);
        AssertClose(expectedHeadFirst, headFirstTensor.GetElementsAsFloat(expectedHeadFirst.Length), tolerance: 1e-5f);
    }

    [Fact]
    public void MlxQwen35PackedGdnDecode_MatchesSeparateProjectionPath()
    {
        if (!MlxBackend.IsAvailable())
            return;

        string previousNative = Environment.GetEnvironmentVariable("TS_MLX_GDN_NATIVE");
        Environment.SetEnvironmentVariable("TS_MLX_GDN_NATIVE", null);
        try
        {
            const int seqLen = 1;
            const int numKeyHeads = 1;
            const int numValueHeads = 1;
            const int headKeyDim = 32;
            const int headValueDim = 32;
            const int keyDim = numKeyHeads * headKeyDim;
            const int valueDim = numValueHeads * headValueDim;
            const int qkvDim = keyDim * 2 + valueDim;
            const int packedDim = qkvDim + valueDim + numValueHeads * 2;
            const int convKernel = 2;

            float[,] qkv = new float[seqLen, qkvDim];
            float[,] z = new float[seqLen, valueDim];
            float[,] beta = new float[seqLen, numValueHeads];
            float[,] alpha = new float[seqLen, numValueHeads];
            float[,] packed = new float[seqLen, packedDim];
            for (int i = 0; i < qkvDim; i++)
            {
                float v = MathF.Sin((i + 1) * 0.07f) * 0.25f;
                qkv[0, i] = v;
                packed[0, i] = v;
            }
            for (int i = 0; i < valueDim; i++)
            {
                float v = MathF.Cos((i + 1) * 0.05f) * 0.2f;
                z[0, i] = v;
                packed[0, qkvDim + i] = v;
            }
            beta[0, 0] = 0.35f;
            alpha[0, 0] = -0.15f;
            packed[0, qkvDim + valueDim] = beta[0, 0];
            packed[0, qkvDim + valueDim + numValueHeads] = alpha[0, 0];

            float[,] convWeight = new float[qkvDim, convKernel];
            for (int i = 0; i < qkvDim; i++)
            {
                convWeight[i, 0] = 0.05f;
                convWeight[i, 1] = 0.75f + (i % 7) * 0.01f;
            }

            float[] normWeight = new float[headValueDim];
            Array.Fill(normWeight, 1.0f);

            using var allocator = new MlxAllocator();
            using var packedTensor = Tensor.FromArray(allocator, packed);
            using var qkvTensor = Tensor.FromArray(allocator, qkv);
            using var zTensor = Tensor.FromArray(allocator, z);
            using var betaTensor = Tensor.FromArray(allocator, beta);
            using var alphaTensor = Tensor.FromArray(allocator, alpha);
            using var convTensor = Tensor.FromArray(allocator, convWeight);
            using var dtBiasTensor = Tensor.FromArray(allocator, new[] { 0.1f });
            using var aLogTensor = Tensor.FromArray(allocator, new[] { -0.5f });
            using var normTensor = Tensor.FromArray(allocator, normWeight);
            using var packedResult = new Tensor(allocator, DType.Float32, seqLen, valueDim);
            using var separateResult = new Tensor(allocator, DType.Float32, seqLen, valueDim);
            using var packedCache = new MlxFusedOps.GatedDeltaNetCache();
            using var separateCache = new MlxFusedOps.GatedDeltaNetCache();

            Assert.True(packedCache.TryRunQwen35Packed(
                packedResult, packedTensor, convTensor, dtBiasTensor, aLogTensor, normTensor,
                seqLen, packedDim, qkvDim, keyDim, valueDim,
                numKeyHeads, numValueHeads, headKeyDim, headValueDim, convKernel, 1e-6f));
            Assert.True(separateCache.TryRunQwen35(
                separateResult, qkvTensor, zTensor, betaTensor, alphaTensor,
                convTensor, dtBiasTensor, aLogTensor, normTensor,
                seqLen, qkvDim, keyDim, valueDim,
                numKeyHeads, numValueHeads, headKeyDim, headValueDim, convKernel, 1e-6f));

            AssertClose(separateResult.GetElementsAsFloat(valueDim), packedResult.GetElementsAsFloat(valueDim), 2e-3f);
        }
        finally
        {
            Environment.SetEnvironmentVariable("TS_MLX_GDN_NATIVE", previousNative);
        }
    }

    [Fact]
    public void MlxRepeatInterleave_RepeatsAlongAxis()
    {
        if (!MlxBackend.IsAvailable())
            return;

        using var allocator = new MlxAllocator();
        using var src = Tensor.FromArray(allocator, new float[,]
        {
            { 1, 2 },
            { 3, 4 },
        });
        using var repeated = Ops.RepeatInterleave(null, src, 2, 0);

        AssertClose(new[] { 1f, 2f, 1f, 2f, 3f, 4f, 3f, 4f }, repeated.GetElementsAsFloat(8));
    }

    [Fact]
    public void MlxAddCausalMask_MasksFuturePositionsOnDevice()
    {
        if (!MlxBackend.IsAvailable())
            return;

        using var allocator = new MlxAllocator();
        using var scores = Tensor.FromArray(allocator, new float[,]
        {
            { 0, 1, 2, 3, 4 },
            { 5, 6, 7, 8, 9 },
            { 10, 11, 12, 13, 14 },
            { 15, 16, 17, 18, 19 },
        });

        Ops.AddCausalMask(scores, seqLen: 2, startPos: 1, maskedValue: float.NegativeInfinity);

        float[] actual = scores.GetElementsAsFloat(20);
        AssertClose(new[]
        {
            0f, 1f, float.NegativeInfinity, float.NegativeInfinity, float.NegativeInfinity,
            5f, 6f, 7f, float.NegativeInfinity, float.NegativeInfinity,
            10f, 11f, float.NegativeInfinity, float.NegativeInfinity, float.NegativeInfinity,
            15f, 16f, 17f, float.NegativeInfinity, float.NegativeInfinity,
        }, actual);
    }

    [Fact]
    public void MlxRoPEEx_NeoXDynamicPositions_MatchesReference()
    {
        if (!MlxBackend.IsAvailable())
            return;

        const int batch = 1;
        const int seq = 3;
        const int heads = 2;
        const int dim = 6;
        const int ropeDim = 4;
        const float ropeBase = 10000f;
        const float ropeScale = 0.75f;

        float[,,,] source = BuildAttentionInput(batch, seq, heads, dim, 0.071f, useCos: false);
        int[] positions = { 5, 5, 6, 6, 8, 8 };
        float[] expected = RoPEReference(source, positions, ropeDim, neox: true, ropeBase, ropeScale);

        using var allocator = new MlxAllocator();
        using var input = Tensor.FromArray(allocator, source);
        using var positionTensor = Tensor.FromArray(allocator, positions);
        using var actualTensor = Ops.RoPEEx(null, input, positionTensor, ropeDim, 2, 0, ropeBase, ropeScale);

        AssertClose(expected, actualTensor.GetElementsAsFloat((int)actualTensor.ElementCount()), 1e-4f);
    }

    [Fact]
    public void MlxRoPEEx_InPlaceTraditional_MatchesReference()
    {
        if (!MlxBackend.IsAvailable())
            return;

        const int batch = 1;
        const int seq = 2;
        const int heads = 3;
        const int dim = 6;
        const int ropeDim = 6;
        const float ropeBase = 500000f;
        const float ropeScale = 1.0f;

        float[,,,] source = BuildAttentionInput(batch, seq, heads, dim, 0.047f, useCos: true);
        int[] positions = { 2, 2, 2, 3, 3, 3 };
        float[] expected = RoPEReference(source, positions, ropeDim, neox: false, ropeBase, ropeScale);

        using var allocator = new MlxAllocator();
        using var input = Tensor.FromArray(allocator, source);
        using var positionTensor = Tensor.FromArray(allocator, positions);

        Ops.RoPEEx(input, input, positionTensor, ropeDim, 0, 0, ropeBase, ropeScale);

        AssertClose(expected, input.GetElementsAsFloat((int)input.ElementCount()), 1e-4f);
    }

    [Fact]
    public void MlxQuantizedMatmul_Q80MatchesDequantizedReference()
    {
        if (!MlxBackend.IsAvailable())
            return;

        const int rows = 3;
        const int inDim = 64;
        const int outDim = 4;
        byte[] weights = CreateQ80Rows(outDim, inDim, (r, c) => (sbyte)(((r + 2) * (c - 23)) % 57), r => 0.125f + r * 0.0625f);
        float[,] input = new float[rows, inDim];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < inDim; c++)
                input[r, c] = MathF.Sin((r + 1) * (c + 3) * 0.037f);

        float[] expected = DequantizedMatmulQ80(weights, outDim, inDim, input);
        IntPtr host = Marshal.AllocHGlobal(weights.Length);
        try
        {
            Marshal.Copy(weights, 0, host, weights.Length);
            using var allocator = new MlxAllocator();
            using var inputTensor = Tensor.FromArray(allocator, input);
            using var outputTensor = new Tensor(allocator, DType.Float32, rows, outDim);

            Assert.True(MlxQuantizedOps.TryAddmmQuantizedToFloat32(
                outputTensor,
                inputTensor,
                host,
                host,
                (int)GgmlTensorType.Q8_0,
                inDim,
                outDim,
                weights.Length));

            AssertClose(expected, outputTensor.GetElementsAsFloat(rows * outDim), 2e-3f);
        }
        finally
        {
            Marshal.FreeHGlobal(host);
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void MlxQuantizedMatmul_Q4MatchesDequantizedReference(bool hasExplicitBias)
    {
        if (!MlxBackend.IsAvailable())
            return;

        const int rows = 2;
        const int inDim = 64;
        const int outDim = 3;
        byte[] weights = CreateQ4Rows(
            outDim,
            inDim,
            (r, c) => (byte)(((r + 3) * (c + 5)) & 0x0F),
            r => 0.0625f + r * 0.03125f,
            r => -0.25f + r * 0.125f,
            hasExplicitBias);
        float[,] input = new float[rows, inDim];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < inDim; c++)
                input[r, c] = MathF.Cos((r + 2) * (c + 1) * 0.041f);

        float[] expected = DequantizedMatmulQ4(weights, outDim, inDim, input, hasExplicitBias);
        IntPtr host = Marshal.AllocHGlobal(weights.Length);
        try
        {
            Marshal.Copy(weights, 0, host, weights.Length);
            using var allocator = new MlxAllocator();
            using var inputTensor = Tensor.FromArray(allocator, input);
            using var outputTensor = new Tensor(allocator, DType.Float32, rows, outDim);

            Assert.True(MlxQuantizedOps.TryAddmmQuantizedToFloat32(
                outputTensor,
                inputTensor,
                host,
                host,
                (int)(hasExplicitBias ? GgmlTensorType.Q4_1 : GgmlTensorType.Q4_0),
                inDim,
                outDim,
                weights.Length));

            AssertClose(expected, outputTensor.GetElementsAsFloat(rows * outDim), 2e-3f);
        }
        finally
        {
            Marshal.FreeHGlobal(host);
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void MlxQuantizedMatmul_Q5MatchesDequantizedReference(bool hasExplicitBias)
    {
        if (!MlxBackend.IsAvailable())
            return;

        const int rows = 2;
        const int inDim = 64;
        const int outDim = 3;
        byte[] weights = CreateQ5Rows(
            outDim,
            inDim,
            (r, c) => (byte)(((r + 7) * (c + 3)) & 0x1F),
            r => 0.046875f + r * 0.0234375f,
            r => -0.125f + r * 0.0625f,
            hasExplicitBias);
        float[,] input = new float[rows, inDim];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < inDim; c++)
                input[r, c] = MathF.Sin((r + 4) * (c + 2) * 0.029f);

        float[] expected = DequantizedMatmulQ5(weights, outDim, inDim, input, hasExplicitBias);
        IntPtr host = Marshal.AllocHGlobal(weights.Length);
        try
        {
            Marshal.Copy(weights, 0, host, weights.Length);
            using var allocator = new MlxAllocator();
            using var inputTensor = Tensor.FromArray(allocator, input);
            using var outputTensor = new Tensor(allocator, DType.Float32, rows, outDim);

            Assert.True(MlxQuantizedOps.TryAddmmQuantizedToFloat32(
                outputTensor,
                inputTensor,
                host,
                host,
                (int)(hasExplicitBias ? GgmlTensorType.Q5_1 : GgmlTensorType.Q5_0),
                inDim,
                outDim,
                weights.Length));

            AssertClose(expected, outputTensor.GetElementsAsFloat(rows * outDim), 2e-3f);
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
    public void MlxQuantizedMatmul_KQuantsMatchDequantizedReference(int ggmlType)
    {
        if (!MlxBackend.IsAvailable())
            return;

        const int rows = 2;
        const int inDim = 256;
        const int outDim = 2;
        byte[] weights = ggmlType switch
        {
            (int)GgmlTensorType.Q4_K => CreateQ4KRows(outDim, inDim),
            (int)GgmlTensorType.Q5_K => CreateQ5KRows(outDim, inDim),
            (int)GgmlTensorType.Q6_K => CreateQ6KRows(outDim, inDim),
            _ => throw new ArgumentOutOfRangeException(nameof(ggmlType)),
        };
        float[,] input = new float[rows, inDim];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < inDim; c++)
                input[r, c] = MathF.Sin((r + 1) * (c + 1) * 0.013f);

        float[] expected = ggmlType switch
        {
            (int)GgmlTensorType.Q4_K => DequantizedMatmulK(weights, outDim, inDim, input, DequantizeQ4KRow),
            (int)GgmlTensorType.Q5_K => DequantizedMatmulK(weights, outDim, inDim, input, DequantizeQ5KRow),
            (int)GgmlTensorType.Q6_K => DequantizedMatmulK(weights, outDim, inDim, input, DequantizeQ6KRow),
            _ => throw new ArgumentOutOfRangeException(nameof(ggmlType)),
        };
        IntPtr host = Marshal.AllocHGlobal(weights.Length);
        try
        {
            Marshal.Copy(weights, 0, host, weights.Length);
            using var allocator = new MlxAllocator();
            using var inputTensor = Tensor.FromArray(allocator, input);
            using var outputTensor = new Tensor(allocator, DType.Float32, rows, outDim);

            Assert.True(MlxQuantizedOps.TryAddmmQuantizedToFloat32(
                outputTensor,
                inputTensor,
                host,
                host,
                ggmlType,
                inDim,
                outDim,
                weights.Length));

            AssertClose(expected, outputTensor.GetElementsAsFloat(rows * outDim), 5e-2f);
        }
        finally
        {
            Marshal.FreeHGlobal(host);
        }
    }

    [Fact]
    public void MlxQuantizedMatmul_Q6KSingleRowMatchesDequantizedReference()
    {
        if (!MlxBackend.IsAvailable())
            return;

        const int rows = 1;
        const int inDim = 256;
        const int outDim = 5;
        byte[] weights = CreateQ6KRows(outDim, inDim);
        float[,] input = new float[rows, inDim];
        for (int c = 0; c < inDim; c++)
            input[0, c] = MathF.Cos((c + 1) * 0.017f);

        float[] expected = DequantizedMatmulK(weights, outDim, inDim, input, DequantizeQ6KRow);
        IntPtr host = Marshal.AllocHGlobal(weights.Length);
        string previousOptIn = Environment.GetEnvironmentVariable("TS_MLX_Q6K_MATMUL4");
        try
        {
            Environment.SetEnvironmentVariable("TS_MLX_Q6K_MATMUL4", "1");
            Marshal.Copy(weights, 0, host, weights.Length);
            using var allocator = new MlxAllocator();
            using var inputTensor = Tensor.FromArray(allocator, input);
            using var outputTensor = new Tensor(allocator, DType.Float32, rows, outDim);

            Assert.True(MlxQuantizedOps.TryAddmmQuantizedToFloat32(
                outputTensor,
                inputTensor,
                host,
                host,
                (int)GgmlTensorType.Q6_K,
                inDim,
                outDim,
                weights.Length));

            AssertClose(expected, outputTensor.GetElementsAsFloat(rows * outDim), 5e-2f);
        }
        finally
        {
            Environment.SetEnvironmentVariable("TS_MLX_Q6K_MATMUL4", previousOptIn);
            Marshal.FreeHGlobal(host);
        }
    }

    [Fact]
    public void MlxQuantizedMatmul_Q5KSingleRow4ColumnMatchesDequantizedReference()
    {
        if (!MlxBackend.IsAvailable())
            return;

        const int rows = 1;
        const int inDim = 256;
        const int outDim = 5;
        byte[] weights = CreateQ5KRows(outDim, inDim);
        float[,] input = new float[rows, inDim];
        for (int c = 0; c < inDim; c++)
            input[0, c] = MathF.Sin((c + 3) * 0.015f);

        float[] expected = DequantizedMatmulK(weights, outDim, inDim, input, DequantizeQ5KRow);
        IntPtr host = Marshal.AllocHGlobal(weights.Length);
        string previousOptIn = Environment.GetEnvironmentVariable("TS_MLX_Q5K_MATMUL4");
        try
        {
            Environment.SetEnvironmentVariable("TS_MLX_Q5K_MATMUL4", "1");
            Marshal.Copy(weights, 0, host, weights.Length);
            using var allocator = new MlxAllocator();
            using var inputTensor = Tensor.FromArray(allocator, input);
            using var outputTensor = new Tensor(allocator, DType.Float32, rows, outDim);

            Assert.True(MlxQuantizedOps.TryAddmmQuantizedToFloat32(
                outputTensor,
                inputTensor,
                host,
                host,
                (int)GgmlTensorType.Q5_K,
                inDim,
                outDim,
                weights.Length));

            AssertClose(expected, outputTensor.GetElementsAsFloat(rows * outDim), 5e-2f);
        }
        finally
        {
            Environment.SetEnvironmentVariable("TS_MLX_Q5K_MATMUL4", previousOptIn);
            Marshal.FreeHGlobal(host);
        }
    }

    [Fact]
    public void MlxQuantizedMatmul_RmsNormFusedMatchesReference()
    {
        if (!MlxBackend.IsAvailable())
            return;

        const int rows = 2;
        const int inDim = 256;
        const int outDim = 3;
        const float eps = 1e-6f;
        byte[] weights = CreateQ6KRows(outDim, inDim);
        float[,] input = new float[rows, inDim];
        float[] norm = new float[inDim];
        for (int c = 0; c < inDim; c++)
            norm[c] = 0.75f + 0.002f * c;
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < inDim; c++)
                input[r, c] = MathF.Sin((r + 1) * (c + 3) * 0.011f);

        float[,] normed = new float[rows, inDim];
        for (int r = 0; r < rows; r++)
        {
            float sumSq = 0f;
            for (int c = 0; c < inDim; c++)
                sumSq += input[r, c] * input[r, c];
            float invRms = 1.0f / MathF.Sqrt(sumSq / inDim + eps);
            for (int c = 0; c < inDim; c++)
                normed[r, c] = input[r, c] * invRms * norm[c];
        }
        float[] expected = DequantizedMatmulK(weights, outDim, inDim, normed, DequantizeQ6KRow);

        IntPtr host = Marshal.AllocHGlobal(weights.Length);
        try
        {
            Marshal.Copy(weights, 0, host, weights.Length);
            using var allocator = new MlxAllocator();
            using var inputTensor = Tensor.FromArray(allocator, input);
            using var normTensor = Tensor.FromArray(allocator, norm);
            using var outputTensor = new Tensor(allocator, DType.Float32, rows, outDim);

            Assert.True(MlxQuantizedOps.TryRmsNormAddmmQuantizedToFloat32(
                outputTensor,
                inputTensor,
                normTensor,
                eps,
                host,
                host,
                (int)GgmlTensorType.Q6_K,
                inDim,
                outDim,
                weights.Length));

            AssertClose(expected, outputTensor.GetElementsAsFloat(rows * outDim), 5e-2f);
        }
        finally
        {
            Marshal.FreeHGlobal(host);
        }
    }

    [Fact]
    public void MlxQuantizedMatmul_AddIntoFusedMatchesReference()
    {
        if (!MlxBackend.IsAvailable())
            return;

        const int rows = 2;
        const int inDim = 256;
        const int outDim = 4;
        byte[] weights = CreateQ6KRows(outDim, inDim);
        float[,] input = new float[rows, inDim];
        float[,] residual = new float[rows, outDim];
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < inDim; c++)
                input[r, c] = MathF.Cos((r + 2) * (c + 1) * 0.019f);
            for (int c = 0; c < outDim; c++)
                residual[r, c] = 0.25f * (r + 1) - 0.1f * c;
        }

        float[] expected = DequantizedMatmulK(weights, outDim, inDim, input, DequantizeQ6KRow);
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < outDim; c++)
                expected[r * outDim + c] += residual[r, c];

        IntPtr host = Marshal.AllocHGlobal(weights.Length);
        try
        {
            Marshal.Copy(weights, 0, host, weights.Length);
            using var allocator = new MlxAllocator();
            using var inputTensor = Tensor.FromArray(allocator, input);
            using var residualTensor = Tensor.FromArray(allocator, residual);

            Assert.True(MlxQuantizedOps.TryAddmmQuantizedAddToFloat32(
                residualTensor,
                inputTensor,
                host,
                host,
                (int)GgmlTensorType.Q6_K,
                inDim,
                outDim,
                weights.Length));

            AssertClose(expected, residualTensor.GetElementsAsFloat(rows * outDim), 5e-2f);
        }
        finally
        {
            Marshal.FreeHGlobal(host);
        }
    }

    [Fact]
    public void MlxQuantizedMatmul_MXFP4MatchesDequantizedReference()
    {
        if (!MlxBackend.IsAvailable())
            return;

        const int rows = 3;
        const int inDim = 64;
        const int outDim = 4;
        byte[] weights = CreateMxfp4Rows(
            outDim,
            inDim,
            (r, c) => (byte)(((r + 5) * (c + 7)) & 0x0F),
            (r, b) => (byte)(126 + ((r + b) % 4)));
        float[,] input = new float[rows, inDim];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < inDim; c++)
                input[r, c] = MathF.Cos((r + 1) * (c + 2) * 0.017f);

        float[] expected = DequantizedMatmulMxfp4(weights, outDim, inDim, input);
        IntPtr host = Marshal.AllocHGlobal(weights.Length);
        try
        {
            Marshal.Copy(weights, 0, host, weights.Length);
            using var allocator = new MlxAllocator();
            using var inputTensor = Tensor.FromArray(allocator, input);
            using var outputTensor = new Tensor(allocator, DType.Float32, rows, outDim);

            Assert.True(MlxQuantizedOps.TryAddmmQuantizedToFloat32(
                outputTensor,
                inputTensor,
                host,
                host,
                (int)GgmlTensorType.MXFP4,
                inDim,
                outDim,
                weights.Length));

            AssertClose(expected, outputTensor.GetElementsAsFloat(rows * outDim), 2e-3f);
        }
        finally
        {
            Marshal.FreeHGlobal(host);
        }
    }

    [Fact]
    public void MlxQuantizedMatmul_IQ4XSMatchesDequantizedReferenceAfterHostRelease()
    {
        if (!MlxBackend.IsAvailable())
            return;

        const int rows = 3;
        const int inDim = 512;
        const int outDim = 3;
        byte[] weights = CreateIq4XsRows(outDim, inDim);
        float[,] input = new float[rows, inDim];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < inDim; c++)
                input[r, c] = MathF.Sin((r + 2) * (c + 5) * 0.011f);

        float[] expected = DequantizedMatmulIq4Xs(weights, outDim, inDim, input);
        IntPtr host = Marshal.AllocHGlobal(weights.Length);
        IntPtr cacheKey = new(0x223456);
        string previousBatchedCols = Environment.GetEnvironmentVariable("TS_MLX_IQ4XS_BATCHED_COLS");
        try
        {
            Environment.SetEnvironmentVariable("TS_MLX_IQ4XS_BATCHED_COLS", "1");
            Marshal.Copy(weights, 0, host, weights.Length);
            using var allocator = new MlxAllocator();
            MlxQuantizedOps.PreloadQuantizedWeight(allocator, cacheKey, host, (int)GgmlTensorType.IQ4_XS, inDim, outDim, weights.Length);

            using var inputTensor = Tensor.FromArray(allocator, input);
            using var outputTensor = new Tensor(allocator, DType.Float32, rows, outDim);
            Assert.True(MlxQuantizedOps.TryAddmmQuantizedToFloat32(
                outputTensor,
                inputTensor,
                cacheKey,
                IntPtr.Zero,
                (int)GgmlTensorType.IQ4_XS,
                inDim,
                outDim,
                weights.Length));

            AssertClose(expected, outputTensor.GetElementsAsFloat(rows * outDim), 2e-3f);
        }
        finally
        {
            Environment.SetEnvironmentVariable("TS_MLX_IQ4XS_BATCHED_COLS", previousBatchedCols);
            Marshal.FreeHGlobal(host);
        }
    }

    [Fact]
    public void MlxQuantizedMatmul_IQ2XXSMatchesNativeDequantizedReferenceAfterHostRelease()
    {
        if (!MlxBackend.IsAvailable())
            return;

        const int rows = 3;
        const int inDim = 512;
        const int outDim = 4;
        byte[] weights = CreateIq2XxsRows(outDim, inDim);
        float[,] input = new float[rows, inDim];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < inDim; c++)
                input[r, c] = MathF.Cos((r + 3) * (c + 1) * 0.009f);

        float[] expected = DequantizedMatmulNative(weights, GgmlTensorType.IQ2_XXS, outDim, inDim, input);
        IntPtr host = Marshal.AllocHGlobal(weights.Length);
        IntPtr cacheKey = new(0x223457);
        try
        {
            Marshal.Copy(weights, 0, host, weights.Length);
            using var allocator = new MlxAllocator();
            MlxQuantizedOps.PreloadQuantizedWeight(allocator, cacheKey, host, (int)GgmlTensorType.IQ2_XXS, inDim, outDim, weights.Length);

            using var inputTensor = Tensor.FromArray(allocator, input);
            using var outputTensor = new Tensor(allocator, DType.Float32, rows, outDim);
            Assert.True(MlxQuantizedOps.TryAddmmQuantizedToFloat32(
                outputTensor,
                inputTensor,
                cacheKey,
                IntPtr.Zero,
                (int)GgmlTensorType.IQ2_XXS,
                inDim,
                outDim,
                weights.Length));

            AssertClose(expected, outputTensor.GetElementsAsFloat(rows * outDim), 5e-3f);
        }
        finally
        {
            Marshal.FreeHGlobal(host);
        }
    }

    [Theory]
    [InlineData((int)GgmlTensorType.IQ2_S)]
    [InlineData((int)GgmlTensorType.IQ3_S)]
    public void MlxQuantizedMatmul_IQ2SAndIQ3SMatchNativeDequantizedReferenceAfterHostRelease(int ggmlType)
    {
        if (!MlxBackend.IsAvailable())
            return;

        const int rows = 3;
        const int inDim = 512;
        const int outDim = 4;
        var type = (GgmlTensorType)ggmlType;
        byte[] weights = CreateNativeQuantRows(type, outDim, inDim);
        float[,] input = new float[rows, inDim];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < inDim; c++)
                input[r, c] = MathF.Sin((r + 5) * (c + 3) * 0.007f);

        float[] expected = DequantizedMatmulNative(weights, type, outDim, inDim, input);
        IntPtr host = Marshal.AllocHGlobal(weights.Length);
        IntPtr cacheKey = new(0x423457 + ggmlType);
        try
        {
            Marshal.Copy(weights, 0, host, weights.Length);
            using var allocator = new MlxAllocator();
            MlxQuantizedOps.PreloadQuantizedWeight(allocator, cacheKey, host, ggmlType, inDim, outDim, weights.Length);

            using var inputTensor = Tensor.FromArray(allocator, input);
            using var outputTensor = new Tensor(allocator, DType.Float32, rows, outDim);
            Assert.True(MlxQuantizedOps.TryAddmmQuantizedToFloat32(
                outputTensor,
                inputTensor,
                cacheKey,
                IntPtr.Zero,
                ggmlType,
                inDim,
                outDim,
                weights.Length));

            AssertClose(expected, outputTensor.GetElementsAsFloat(rows * outDim), 5e-2f);
        }
        finally
        {
            Marshal.FreeHGlobal(host);
        }
    }

    [Fact]
    public void MlxQuantizedRows_Q80MatchDequantizedReferenceAfterHostRelease()
    {
        if (!MlxBackend.IsAvailable())
            return;

        const int inDim = 64;
        const int outDim = 5;
        int[] rows = { 4, 1, 3 };
        byte[] weights = CreateQ80Rows(outDim, inDim, (r, c) => (sbyte)(((r + 5) * (c - 11)) % 63), r => 0.09375f + r * 0.03125f);
        float[] expected = new float[rows.Length * inDim];
        for (int i = 0; i < rows.Length; i++)
            DequantizeQ80Row(weights, rows[i], inDim, expected, i * inDim);

        IntPtr host = Marshal.AllocHGlobal(weights.Length);
        IntPtr cacheKey = new(0x123456);
        try
        {
            Marshal.Copy(weights, 0, host, weights.Length);
            using var allocator = new MlxAllocator();
            MlxQuantizedOps.PreloadQuantizedWeight(allocator, cacheKey, host, (int)GgmlTensorType.Q8_0, inDim, outDim, weights.Length);

            using var indices = Tensor.FromArray(allocator, rows);
            using var outputTensor = new Tensor(allocator, DType.Float32, rows.Length, inDim);
            Assert.True(MlxQuantizedOps.TryGetRowsQuantizedToFloat32(
                outputTensor,
                cacheKey,
                IntPtr.Zero,
                (int)GgmlTensorType.Q8_0,
                inDim,
                outDim,
                weights.Length,
                indices));

            AssertClose(expected, outputTensor.GetElementsAsFloat(rows.Length * inDim), 2e-3f);
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
    public void MlxQuantizedRows_KQuantsMatchDequantizedReferenceAfterHostRelease(int ggmlType)
    {
        if (!MlxBackend.IsAvailable())
            return;

        const int inDim = 256;
        const int outDim = 5;
        int[] rows = { 4, 1, 3 };
        byte[] weights = ggmlType switch
        {
            (int)GgmlTensorType.Q4_K => CreateQ4KRows(outDim, inDim),
            (int)GgmlTensorType.Q5_K => CreateQ5KRows(outDim, inDim),
            (int)GgmlTensorType.Q6_K => CreateQ6KRows(outDim, inDim),
            _ => throw new ArgumentOutOfRangeException(nameof(ggmlType)),
        };
        float[] expected = new float[rows.Length * inDim];
        for (int i = 0; i < rows.Length; i++)
        {
            switch (ggmlType)
            {
                case (int)GgmlTensorType.Q4_K:
                    DequantizeQ4KRow(weights, rows[i], inDim, expected, i * inDim);
                    break;
                case (int)GgmlTensorType.Q5_K:
                    DequantizeQ5KRow(weights, rows[i], inDim, expected, i * inDim);
                    break;
                case (int)GgmlTensorType.Q6_K:
                    DequantizeQ6KRow(weights, rows[i], inDim, expected, i * inDim);
                    break;
            }
        }

        IntPtr host = Marshal.AllocHGlobal(weights.Length);
        IntPtr cacheKey = new(0x623457 + ggmlType);
        try
        {
            Marshal.Copy(weights, 0, host, weights.Length);
            using var allocator = new MlxAllocator();
            MlxQuantizedOps.PreloadQuantizedWeight(allocator, cacheKey, host, ggmlType, inDim, outDim, weights.Length);

            using var indices = Tensor.FromArray(allocator, rows);
            using var outputTensor = new Tensor(allocator, DType.Float32, rows.Length, inDim);
            Assert.True(MlxQuantizedOps.TryGetRowsQuantizedToFloat32(
                outputTensor,
                cacheKey,
                IntPtr.Zero,
                ggmlType,
                inDim,
                outDim,
                weights.Length,
                indices));

            AssertClose(expected, outputTensor.GetElementsAsFloat(rows.Length * inDim), 5e-2f);
        }
        finally
        {
            Marshal.FreeHGlobal(host);
        }
    }

    [Fact]
    public void MlxQuantizedMatmul_IQ4XSDecode4ColumnMatchesDequantizedReferenceAfterHostRelease()
    {
        if (!MlxBackend.IsAvailable())
            return;

        const int rows = 1;
        const int inDim = 512;
        const int outDim = 5;
        byte[] weights = CreateIq4XsRows(outDim, inDim);
        float[,] input = new float[rows, inDim];
        for (int c = 0; c < inDim; c++)
            input[0, c] = MathF.Cos((c + 7) * 0.009f);

        float[] expected = DequantizedMatmulIq4Xs(weights, outDim, inDim, input);
        IntPtr host = Marshal.AllocHGlobal(weights.Length);
        IntPtr cacheKey = new(0x223457);
        string previousOptIn = Environment.GetEnvironmentVariable("TS_MLX_IQ4XS_MATMUL4");
        try
        {
            Environment.SetEnvironmentVariable("TS_MLX_IQ4XS_MATMUL4", "1");
            Marshal.Copy(weights, 0, host, weights.Length);
            using var allocator = new MlxAllocator();
            MlxQuantizedOps.PreloadQuantizedWeight(allocator, cacheKey, host, (int)GgmlTensorType.IQ4_XS, inDim, outDim, weights.Length);

            using var inputTensor = Tensor.FromArray(allocator, input);
            using var outputTensor = new Tensor(allocator, DType.Float32, rows, outDim);
            Assert.True(MlxQuantizedOps.TryAddmmQuantizedToFloat32(
                outputTensor,
                inputTensor,
                cacheKey,
                IntPtr.Zero,
                (int)GgmlTensorType.IQ4_XS,
                inDim,
                outDim,
                weights.Length));

            AssertClose(expected, outputTensor.GetElementsAsFloat(rows * outDim), 2e-3f);
            MlxQuantizedOps.ReleaseQuantizedWeight(allocator, cacheKey);
        }
        finally
        {
            Environment.SetEnvironmentVariable("TS_MLX_IQ4XS_MATMUL4", previousOptIn);
            Marshal.FreeHGlobal(host);
        }
    }

    [Fact]
    public void MlxQuantizedRows_IQ4XSMatchDequantizedReferenceAfterHostRelease()
    {
        if (!MlxBackend.IsAvailable())
            return;

        const int inDim = 512;
        const int outDim = 5;
        int[] rows = { 4, 1, 3 };
        byte[] weights = CreateIq4XsRows(outDim, inDim);
        float[] expected = new float[rows.Length * inDim];
        for (int i = 0; i < rows.Length; i++)
            DequantizeIq4XsRow(weights, rows[i], inDim, expected, i * inDim);

        IntPtr host = Marshal.AllocHGlobal(weights.Length);
        IntPtr cacheKey = new(0x223457);
        try
        {
            Marshal.Copy(weights, 0, host, weights.Length);
            using var allocator = new MlxAllocator();
            MlxQuantizedOps.PreloadQuantizedWeight(allocator, cacheKey, host, (int)GgmlTensorType.IQ4_XS, inDim, outDim, weights.Length);

            using var indices = Tensor.FromArray(allocator, rows);
            using var outputTensor = new Tensor(allocator, DType.Float32, rows.Length, inDim);
            Assert.True(MlxQuantizedOps.TryGetRowsQuantizedToFloat32(
                outputTensor,
                cacheKey,
                IntPtr.Zero,
                (int)GgmlTensorType.IQ4_XS,
                inDim,
                outDim,
                weights.Length,
                indices));

            AssertClose(expected, outputTensor.GetElementsAsFloat(rows.Length * inDim), 2e-3f);
        }
        finally
        {
            Marshal.FreeHGlobal(host);
        }
    }

    [Fact]
    public void MlxQuantizedRows_IQ2XXSMatchNativeDequantizedReferenceAfterHostRelease()
    {
        if (!MlxBackend.IsAvailable())
            return;

        const int inDim = 512;
        const int outDim = 5;
        int[] rows = { 4, 1, 3 };
        byte[] weights = CreateIq2XxsRows(outDim, inDim);
        float[] expected = new float[rows.Length * inDim];
        long rowBytes = NativeDequant.RowSize((int)GgmlTensorType.IQ2_XXS, inDim);
        for (int i = 0; i < rows.Length; i++)
            NativeDequant.DequantizeToFloat32((int)GgmlTensorType.IQ2_XXS, weights, (int)(rows[i] * rowBytes), expected, i * inDim, inDim);

        IntPtr host = Marshal.AllocHGlobal(weights.Length);
        IntPtr cacheKey = new(0x323457);
        try
        {
            Marshal.Copy(weights, 0, host, weights.Length);
            using var allocator = new MlxAllocator();
            MlxQuantizedOps.PreloadQuantizedWeight(allocator, cacheKey, host, (int)GgmlTensorType.IQ2_XXS, inDim, outDim, weights.Length);

            using var outputTensor = new Tensor(allocator, DType.Float32, rows.Length, inDim);
            using var indices = Tensor.FromArray(allocator, rows);
            Assert.True(MlxQuantizedOps.TryGetRowsQuantizedToFloat32(
                outputTensor,
                cacheKey,
                IntPtr.Zero,
                (int)GgmlTensorType.IQ2_XXS,
                inDim,
                outDim,
                weights.Length,
                indices));

            AssertClose(expected, outputTensor.GetElementsAsFloat(rows.Length * inDim), 5e-3f);
        }
        finally
        {
            Marshal.FreeHGlobal(host);
        }
    }

    [Theory]
    [InlineData((int)GgmlTensorType.IQ2_S)]
    [InlineData((int)GgmlTensorType.IQ3_S)]
    public void MlxQuantizedRows_IQ2SAndIQ3SMatchNativeDequantizedReferenceAfterHostRelease(int ggmlType)
    {
        if (!MlxBackend.IsAvailable())
            return;

        const int inDim = 512;
        const int outDim = 5;
        int[] rows = { 4, 1, 3 };
        var type = (GgmlTensorType)ggmlType;
        byte[] weights = CreateNativeQuantRows(type, outDim, inDim);
        float[] expected = new float[rows.Length * inDim];
        long rowBytes = NativeDequant.RowSize(ggmlType, inDim);
        for (int i = 0; i < rows.Length; i++)
            NativeDequant.DequantizeToFloat32(ggmlType, weights, (int)(rows[i] * rowBytes), expected, i * inDim, inDim);

        IntPtr host = Marshal.AllocHGlobal(weights.Length);
        IntPtr cacheKey = new(0x523457 + ggmlType);
        try
        {
            Marshal.Copy(weights, 0, host, weights.Length);
            using var allocator = new MlxAllocator();
            MlxQuantizedOps.PreloadQuantizedWeight(allocator, cacheKey, host, ggmlType, inDim, outDim, weights.Length);

            using var outputTensor = new Tensor(allocator, DType.Float32, rows.Length, inDim);
            using var indices = Tensor.FromArray(allocator, rows);
            Assert.True(MlxQuantizedOps.TryGetRowsQuantizedToFloat32(
                outputTensor,
                cacheKey,
                IntPtr.Zero,
                ggmlType,
                inDim,
                outDim,
                weights.Length,
                indices));

            AssertClose(expected, outputTensor.GetElementsAsFloat(rows.Length * inDim), 5e-3f);
        }
        finally
        {
            Marshal.FreeHGlobal(host);
        }
    }

    [Fact]
    public void MlxQuantizedRows_MXFP4MatchDequantizedReferenceAfterHostRelease()
    {
        if (!MlxBackend.IsAvailable())
            return;

        const int inDim = 64;
        const int outDim = 5;
        int[] rows = { 4, 1, 3 };
        byte[] weights = CreateMxfp4Rows(
            outDim,
            inDim,
            (r, c) => (byte)(((r + 9) * (c + 11)) & 0x0F),
            (r, b) => (byte)(125 + ((r * 2 + b) % 5)));
        float[] expected = new float[rows.Length * inDim];
        for (int i = 0; i < rows.Length; i++)
            DequantizeMxfp4Row(weights, rows[i], inDim, expected, i * inDim);

        IntPtr host = Marshal.AllocHGlobal(weights.Length);
        IntPtr cacheKey = new(0x123457);
        try
        {
            Marshal.Copy(weights, 0, host, weights.Length);
            using var allocator = new MlxAllocator();
            MlxQuantizedOps.PreloadQuantizedWeight(allocator, cacheKey, host, (int)GgmlTensorType.MXFP4, inDim, outDim, weights.Length);

            using var indices = Tensor.FromArray(allocator, rows);
            using var outputTensor = new Tensor(allocator, DType.Float32, rows.Length, inDim);
            Assert.True(MlxQuantizedOps.TryGetRowsQuantizedToFloat32(
                outputTensor,
                cacheKey,
                IntPtr.Zero,
                (int)GgmlTensorType.MXFP4,
                inDim,
                outDim,
                weights.Length,
                indices));

            AssertClose(expected, outputTensor.GetElementsAsFloat(rows.Length * inDim), 2e-3f);
        }
        finally
        {
            Marshal.FreeHGlobal(host);
        }
    }

    private static void AssertClose(IReadOnlyList<float> expected, IReadOnlyList<float> actual, float tolerance = 1e-4f)
    {
        Assert.Equal(expected.Count, actual.Count);
        for (int i = 0; i < expected.Count; i++)
        {
            if (float.IsInfinity(expected[i]) || float.IsNaN(expected[i]))
            {
                Assert.Equal(expected[i], actual[i]);
                continue;
            }

            Assert.True(Math.Abs(expected[i] - actual[i]) <= tolerance, $"index {i}: expected {expected[i]}, actual {actual[i]}");
        }
    }

    private static float[,,,] BuildAttentionInput(int batch, int seq, int heads, int dim, float scale, bool useCos)
    {
        float[,,,] result = new float[batch, seq, heads, dim];
        for (int b = 0; b < batch; b++)
            for (int t = 0; t < seq; t++)
                for (int h = 0; h < heads; h++)
                    for (int d = 0; d < dim; d++)
                    {
                        float x = (b + 1) * (t + 2) * (h + 3) * (d + 1) * scale;
                        result[b, t, h, d] = useCos ? MathF.Cos(x) : MathF.Sin(x);
                    }

        return result;
    }

    private static float GeluReference(float x)
    {
        return 0.5f * x * (1.0f + MathF.Tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
    }

    private static byte[] CreateQ80Rows(int rows, int cols, Func<int, int, sbyte> value, Func<int, float> scale)
    {
        return CreateQ8Rows(rows, cols, value, scale, hasBlockSum: false);
    }

    private static byte[] CreateQ8Rows(int rows, int cols, Func<int, int, sbyte> value, Func<int, float> scale, bool hasBlockSum)
    {
        const int blockSize = 32;
        Assert.Equal(0, cols % blockSize);
        int blockBytes = hasBlockSum ? 36 : 34;
        int quantOffset = hasBlockSum ? 4 : 2;
        int blocksPerRow = cols / blockSize;
        byte[] raw = new byte[rows * blocksPerRow * blockBytes];
        for (int r = 0; r < rows; r++)
        {
            for (int b = 0; b < blocksPerRow; b++)
            {
                int offset = (r * blocksPerRow + b) * blockBytes;
                ushort scaleBits = BitConverter.HalfToUInt16Bits((System.Half)scale(r));
                raw[offset] = (byte)scaleBits;
                raw[offset + 1] = (byte)(scaleBits >> 8);
                if (hasBlockSum)
                {
                    raw[offset + 2] = 0;
                    raw[offset + 3] = 0;
                }
                for (int j = 0; j < blockSize; j++)
                    raw[offset + quantOffset + j] = unchecked((byte)value(r, b * blockSize + j));
            }
        }

        return raw;
    }

    private static byte[] CreateQ4Rows(int rows, int cols, Func<int, int, byte> value, Func<int, float> scale, Func<int, float> bias, bool hasExplicitBias)
    {
        const int blockSize = 32;
        Assert.Equal(0, cols % blockSize);
        int blockBytes = hasExplicitBias ? 20 : 18;
        int quantOffset = hasExplicitBias ? 4 : 2;
        int blocksPerRow = cols / blockSize;
        byte[] raw = new byte[rows * blocksPerRow * blockBytes];
        for (int r = 0; r < rows; r++)
        {
            for (int b = 0; b < blocksPerRow; b++)
            {
                int offset = (r * blocksPerRow + b) * blockBytes;
                ushort scaleBits = BitConverter.HalfToUInt16Bits((System.Half)scale(r));
                raw[offset] = (byte)scaleBits;
                raw[offset + 1] = (byte)(scaleBits >> 8);
                if (hasExplicitBias)
                {
                    ushort biasBits = BitConverter.HalfToUInt16Bits((System.Half)bias(r));
                    raw[offset + 2] = (byte)biasBits;
                    raw[offset + 3] = (byte)(biasBits >> 8);
                }

                for (int j = 0; j < blockSize / 2; j++)
                {
                    byte low = (byte)(value(r, b * blockSize + j) & 0x0F);
                    byte high = (byte)(value(r, b * blockSize + blockSize / 2 + j) & 0x0F);
                    raw[offset + quantOffset + j] = (byte)(low | (high << 4));
                }
            }
        }

        return raw;
    }

    private static byte[] CreateQ5Rows(int rows, int cols, Func<int, int, byte> value, Func<int, float> scale, Func<int, float> bias, bool hasExplicitBias)
    {
        const int blockSize = 32;
        Assert.Equal(0, cols % blockSize);
        int blockBytes = hasExplicitBias ? 24 : 22;
        int highBitOffset = hasExplicitBias ? 4 : 2;
        int quantOffset = hasExplicitBias ? 8 : 6;
        int blocksPerRow = cols / blockSize;
        byte[] raw = new byte[rows * blocksPerRow * blockBytes];
        for (int r = 0; r < rows; r++)
        {
            for (int b = 0; b < blocksPerRow; b++)
            {
                int offset = (r * blocksPerRow + b) * blockBytes;
                ushort scaleBits = BitConverter.HalfToUInt16Bits((System.Half)scale(r));
                raw[offset] = (byte)scaleBits;
                raw[offset + 1] = (byte)(scaleBits >> 8);
                if (hasExplicitBias)
                {
                    ushort biasBits = BitConverter.HalfToUInt16Bits((System.Half)bias(r));
                    raw[offset + 2] = (byte)biasBits;
                    raw[offset + 3] = (byte)(biasBits >> 8);
                }

                uint highBits = 0;
                for (int j = 0; j < blockSize / 2; j++)
                {
                    byte low = (byte)(value(r, b * blockSize + j) & 0x1F);
                    byte high = (byte)(value(r, b * blockSize + blockSize / 2 + j) & 0x1F);
                    raw[offset + quantOffset + j] = (byte)((low & 0x0F) | ((high & 0x0F) << 4));
                    highBits |= (uint)((low >> 4) & 1) << j;
                    highBits |= (uint)((high >> 4) & 1) << (j + 16);
                }

                raw[offset + highBitOffset] = (byte)highBits;
                raw[offset + highBitOffset + 1] = (byte)(highBits >> 8);
                raw[offset + highBitOffset + 2] = (byte)(highBits >> 16);
                raw[offset + highBitOffset + 3] = (byte)(highBits >> 24);
            }
        }

        return raw;
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

    private static byte[] CreateIq4XsRows(int rows, int cols)
    {
        const int blockSize = 256;
        const int blockBytes = 136;
        Assert.Equal(0, cols % blockSize);
        int blocksPerRow = cols / blockSize;
        byte[] raw = new byte[rows * blocksPerRow * blockBytes];
        for (int r = 0; r < rows; r++)
        {
            for (int b = 0; b < blocksPerRow; b++)
            {
                int offset = (r * blocksPerRow + b) * blockBytes;
                WriteHalf(raw, offset, 1.0f / 256.0f + r * (1.0f / 2048.0f));
                ushort scalesH = 0;
                for (int group = 0; group < 8; group++)
                {
                    int ls = 33 + ((r * 7 + b * 5 + group * 3) % 17);
                    raw[offset + 4 + (group >> 1)] |= (byte)((ls & 0x0f) << (4 * (group & 1)));
                    scalesH |= (ushort)(((ls >> 4) & 0x03) << (2 * group));

                    for (int j = 0; j < 32; j++)
                    {
                        int q = (r * 11 + b * 13 + group * 7 + j * 5) & 0x0f;
                        int qOffset = offset + 8 + group * 16 + (j & 15);
                        if (j < 16)
                            raw[qOffset] = (byte)((raw[qOffset] & 0xf0) | q);
                        else
                            raw[qOffset] = (byte)((raw[qOffset] & 0x0f) | (q << 4));
                    }
                }

                raw[offset + 2] = (byte)scalesH;
                raw[offset + 3] = (byte)(scalesH >> 8);
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

    private static byte[] CreateNativeQuantRows(GgmlTensorType type, int rows, int cols)
    {
        const int blockSize = 256;
        Assert.Equal(0, cols % blockSize);
        int rowBytes = checked((int)NativeDequant.RowSize((int)type, cols));
        int blocksPerRow = cols / blockSize;
        int blockBytes = rowBytes / blocksPerRow;
        byte[] raw = new byte[rows * rowBytes];

        for (int r = 0; r < rows; r++)
        {
            for (int b = 0; b < blocksPerRow; b++)
            {
                int offset = r * rowBytes + b * blockBytes;
                WriteHalf(raw, offset, 0.001953125f + r * 0.000244140625f + b * 0.0001220703125f);
                for (int i = 2; i < blockBytes; i++)
                    raw[offset + i] = (byte)((r * 37 + b * 19 + i * 13 + (int)type * 7) & 0xFF);
            }
        }

        return raw;
    }

    private static byte[] CreateMxfp4Rows(int rows, int cols, Func<int, int, byte> value, Func<int, int, byte> scaleByte)
    {
        const int blockSize = 32;
        const int blockBytes = 17;
        Assert.Equal(0, cols % blockSize);
        int blocksPerRow = cols / blockSize;
        byte[] raw = new byte[rows * blocksPerRow * blockBytes];
        for (int r = 0; r < rows; r++)
        {
            for (int b = 0; b < blocksPerRow; b++)
            {
                int offset = (r * blocksPerRow + b) * blockBytes;
                raw[offset] = scaleByte(r, b);
                for (int j = 0; j < blockSize / 2; j++)
                {
                    byte low = (byte)(value(r, b * blockSize + j) & 0x0F);
                    byte high = (byte)(value(r, b * blockSize + blockSize / 2 + j) & 0x0F);
                    raw[offset + 1 + j] = (byte)(low | (high << 4));
                }
            }
        }

        return raw;
    }

    private static void WriteHalf(byte[] raw, int offset, float value)
    {
        ushort bits = BitConverter.HalfToUInt16Bits((System.Half)value);
        raw[offset] = (byte)bits;
        raw[offset + 1] = (byte)(bits >> 8);
    }

    private static float[] DequantizedMatmulQ80(byte[] weights, int outDim, int inDim, float[,] input)
    {
        return DequantizedMatmulQ8(weights, outDim, inDim, input, hasBlockSum: false);
    }

    private static float[] DequantizedMatmulQ8(byte[] weights, int outDim, int inDim, float[,] input, bool hasBlockSum)
    {
        int rows = input.GetLength(0);
        float[] expected = new float[rows * outDim];
        float[] dequantizedRow = new float[inDim];
        for (int o = 0; o < outDim; o++)
        {
            DequantizeQ8Row(weights, o, inDim, dequantizedRow, 0, hasBlockSum);
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

    private static float[] DequantizedMatmulMxfp4(byte[] weights, int outDim, int inDim, float[,] input)
    {
        int rows = input.GetLength(0);
        float[] expected = new float[rows * outDim];
        float[] dequantizedRow = new float[inDim];
        for (int o = 0; o < outDim; o++)
        {
            DequantizeMxfp4Row(weights, o, inDim, dequantizedRow, 0);
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

    private static float[] DequantizedMatmulIq4Xs(byte[] weights, int outDim, int inDim, float[,] input)
    {
        int rows = input.GetLength(0);
        float[] expected = new float[rows * outDim];
        float[] dequantizedRow = new float[inDim];
        for (int o = 0; o < outDim; o++)
        {
            DequantizeIq4XsRow(weights, o, inDim, dequantizedRow, 0);
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

    private static float[] DequantizedMatmulQ4(byte[] weights, int outDim, int inDim, float[,] input, bool hasExplicitBias)
    {
        int rows = input.GetLength(0);
        float[] expected = new float[rows * outDim];
        float[] dequantizedRow = new float[inDim];
        for (int o = 0; o < outDim; o++)
        {
            DequantizeQ4Row(weights, o, inDim, dequantizedRow, 0, hasExplicitBias);
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

    private static float[] DequantizedMatmulQ5(byte[] weights, int outDim, int inDim, float[,] input, bool hasExplicitBias)
    {
        int rows = input.GetLength(0);
        float[] expected = new float[rows * outDim];
        float[] dequantizedRow = new float[inDim];
        for (int o = 0; o < outDim; o++)
        {
            DequantizeQ5Row(weights, o, inDim, dequantizedRow, 0, hasExplicitBias);
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

    private static void DequantizeQ80Row(byte[] weights, int row, int inDim, float[] destination, int destinationOffset)
    {
        DequantizeQ8Row(weights, row, inDim, destination, destinationOffset, hasBlockSum: false);
    }

    private static void DequantizeQ8Row(byte[] weights, int row, int inDim, float[] destination, int destinationOffset, bool hasBlockSum)
    {
        const int blockSize = 32;
        int blockBytes = hasBlockSum ? 36 : 34;
        int quantOffset = hasBlockSum ? 4 : 2;
        int blocksPerRow = inDim / blockSize;
        for (int b = 0; b < blocksPerRow; b++)
        {
            int offset = (row * blocksPerRow + b) * blockBytes;
            ushort scaleBits = (ushort)(weights[offset] | (weights[offset + 1] << 8));
            float scale = (float)BitConverter.UInt16BitsToHalf(scaleBits);
            for (int j = 0; j < blockSize; j++)
                destination[destinationOffset + b * blockSize + j] = unchecked((sbyte)weights[offset + quantOffset + j]) * scale;
        }
    }

    private static void DequantizeQ4Row(byte[] weights, int row, int inDim, float[] destination, int destinationOffset, bool hasExplicitBias)
    {
        const int blockSize = 32;
        int blockBytes = hasExplicitBias ? 20 : 18;
        int quantOffset = hasExplicitBias ? 4 : 2;
        int blocksPerRow = inDim / blockSize;
        for (int b = 0; b < blocksPerRow; b++)
        {
            int offset = (row * blocksPerRow + b) * blockBytes;
            float scale = (float)BitConverter.UInt16BitsToHalf((ushort)(weights[offset] | (weights[offset + 1] << 8)));
            float bias = hasExplicitBias
                ? (float)BitConverter.UInt16BitsToHalf((ushort)(weights[offset + 2] | (weights[offset + 3] << 8)))
                : -8.0f * scale;
            for (int j = 0; j < blockSize / 2; j++)
            {
                byte packed = weights[offset + quantOffset + j];
                destination[destinationOffset + b * blockSize + j] = (packed & 0x0F) * scale + bias;
                destination[destinationOffset + b * blockSize + blockSize / 2 + j] = ((packed >> 4) & 0x0F) * scale + bias;
            }
        }
    }

    private static void DequantizeQ5Row(byte[] weights, int row, int inDim, float[] destination, int destinationOffset, bool hasExplicitBias)
    {
        const int blockSize = 32;
        int blockBytes = hasExplicitBias ? 24 : 22;
        int highBitOffset = hasExplicitBias ? 4 : 2;
        int quantOffset = hasExplicitBias ? 8 : 6;
        int blocksPerRow = inDim / blockSize;
        for (int b = 0; b < blocksPerRow; b++)
        {
            int offset = (row * blocksPerRow + b) * blockBytes;
            float scale = (float)BitConverter.UInt16BitsToHalf((ushort)(weights[offset] | (weights[offset + 1] << 8)));
            float bias = hasExplicitBias
                ? (float)BitConverter.UInt16BitsToHalf((ushort)(weights[offset + 2] | (weights[offset + 3] << 8)))
                : -16.0f * scale;
            uint highBits =
                weights[offset + highBitOffset] |
                ((uint)weights[offset + highBitOffset + 1] << 8) |
                ((uint)weights[offset + highBitOffset + 2] << 16) |
                ((uint)weights[offset + highBitOffset + 3] << 24);
            for (int j = 0; j < blockSize / 2; j++)
            {
                byte packed = weights[offset + quantOffset + j];
                int low = (packed & 0x0F) | (int)(((highBits >> j) & 1) << 4);
                int high = ((packed >> 4) & 0x0F) | (int)(((highBits >> (j + 16)) & 1) << 4);
                destination[destinationOffset + b * blockSize + j] = low * scale + bias;
                destination[destinationOffset + b * blockSize + blockSize / 2 + j] = high * scale + bias;
            }
        }
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
                float d1 = (float)(System.Half)(d * sc1);
                float d2 = (float)(System.Half)(d * sc2);
                float m1 = (float)(System.Half)(min * m1q);
                float m2 = (float)(System.Half)(min * m2q);
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
                float d1 = (float)(System.Half)(d * sc1);
                float d2 = (float)(System.Half)(d * sc2);
                float m1 = (float)(System.Half)(min * m1q);
                float m2 = (float)(System.Half)(min * m2q);
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

    private static void DequantizeIq4XsRow(byte[] weights, int row, int inDim, float[] destination, int destinationOffset)
    {
        const int blockSize = 256;
        const int blockBytes = 136;
        ReadOnlySpan<sbyte> values = stackalloc sbyte[]
        {
            -127, -104, -83, -65, -49, -35, -22, -10,
               1,   13,  25,  38,  53,  69,  89, 113,
        };

        int blocksPerRow = inDim / blockSize;
        for (int b = 0; b < blocksPerRow; b++)
        {
            int offset = (row * blocksPerRow + b) * blockBytes;
            float d = (float)BitConverter.UInt16BitsToHalf((ushort)(weights[offset] | (weights[offset + 1] << 8)));
            ushort scalesH = (ushort)(weights[offset + 2] | (weights[offset + 3] << 8));
            int scalesLOffset = offset + 4;
            int qsOffset = offset + 8;
            for (int group = 0; group < 8; group++)
            {
                int ls = ((weights[scalesLOffset + (group >> 1)] >> (4 * (group & 1))) & 0x0f) |
                    (((scalesH >> (2 * group)) & 0x03) << 4);
                float scale = d * (ls - 32);
                for (int j = 0; j < 32; j++)
                {
                    byte packed = weights[qsOffset + group * 16 + (j & 15)];
                    int q = j < 16 ? packed & 0x0f : packed >> 4;
                    destination[destinationOffset + b * blockSize + group * 32 + j] = scale * values[q];
                }
            }
        }
    }

    private static void DequantizeMxfp4Row(byte[] weights, int row, int inDim, float[] destination, int destinationOffset)
    {
        const int blockSize = 32;
        const int blockBytes = 17;
        ReadOnlySpan<sbyte> values = stackalloc sbyte[]
        {
            0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12,
        };

        int blocksPerRow = inDim / blockSize;
        for (int b = 0; b < blocksPerRow; b++)
        {
            int offset = (row * blocksPerRow + b) * blockBytes;
            float scale = E8M0ToFp32Half(weights[offset]);
            for (int j = 0; j < blockSize / 2; j++)
            {
                byte packed = weights[offset + 1 + j];
                destination[destinationOffset + b * blockSize + j] = scale * values[packed & 0x0F];
                destination[destinationOffset + b * blockSize + blockSize / 2 + j] = scale * values[packed >> 4];
            }
        }
    }

    private static float E8M0ToFp32Half(byte value)
    {
        uint bits = value < 2 ? 0x00200000u << value : ((uint)value - 1u) << 23;
        return BitConverter.Int32BitsToSingle((int)bits);
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

    private static float[] RoPEReference(float[,,,] input, int[] positions, int ropeDim, bool neox, float ropeBase, float ropeScale)
    {
        int batch = input.GetLength(0);
        int seq = input.GetLength(1);
        int heads = input.GetLength(2);
        int dim = input.GetLength(3);
        float[] result = new float[input.Length];
        int row = 0;

        for (int b = 0; b < batch; b++)
            for (int s = 0; s < seq; s++)
                for (int h = 0; h < heads; h++, row++)
                {
                    int offset = ((b * seq + s) * heads + h) * dim;
                    for (int d = 0; d < dim; d++)
                        result[offset + d] = input[b, s, h, d];

                    int pairCount = Math.Min(ropeDim, dim) / 2;
                    int position = positions[row];
                    for (int i = 0; i < pairCount; i++)
                    {
                        float invFreq = MathF.Pow(ropeBase, -2.0f * i / Math.Min(ropeDim, dim));
                        float angle = position * invFreq * ropeScale;
                        float cos = MathF.Cos(angle);
                        float sin = MathF.Sin(angle);

                        int leftIndex = neox ? i : 2 * i;
                        int rightIndex = neox ? i + pairCount : 2 * i + 1;
                        float left = input[b, s, h, leftIndex];
                        float right = input[b, s, h, rightIndex];
                        result[offset + leftIndex] = left * cos - right * sin;
                        result[offset + rightIndex] = right * cos + left * sin;
                    }
                }

        return result;
    }

    private static float[,,] BuildHeadFirstInput(int heads, int seq, int dim, float step, bool useCos)
    {
        float[,,] result = new float[heads, seq, dim];
        for (int h = 0; h < heads; h++)
            for (int s = 0; s < seq; s++)
                for (int d = 0; d < dim; d++)
                {
                    float angle = ((h + 1) * 0.7f + (s + 1) * 1.3f + (d + 1) * 0.5f) * step;
                    result[h, s, d] = useCos ? MathF.Cos(angle) : MathF.Sin(angle);
                }

        return result;
    }

    private static float[] FlattenHeadFirstSingleToken(float[,,] input)
    {
        int heads = input.GetLength(0);
        int dim = input.GetLength(2);
        float[] result = new float[heads * dim];
        for (int h = 0; h < heads; h++)
            for (int d = 0; d < dim; d++)
                result[h * dim + d] = input[h, 0, d];

        return result;
    }

    private static float[] HeadFirstAttentionReference(float[,,] q, float[,,] k, float[,,] v, float scale, bool causal)
    {
        int heads = q.GetLength(0);
        int seqQ = q.GetLength(1);
        int keyDim = q.GetLength(2);
        int kvHeads = k.GetLength(0);
        int seqK = k.GetLength(1);
        int valueDim = v.GetLength(2);
        int groupSize = heads / kvHeads;
        float[] result = new float[seqQ * heads * valueDim];
        float[] scores = new float[seqK];

        for (int tq = 0; tq < seqQ; tq++)
            for (int h = 0; h < heads; h++)
            {
                int kvHead = h / groupSize;
                float max = float.NegativeInfinity;
                for (int tk = 0; tk < seqK; tk++)
                {
                    bool masked = causal && tk > tq + (seqK - seqQ);
                    float score = masked ? float.NegativeInfinity : 0f;
                    if (!masked)
                    {
                        for (int d = 0; d < keyDim; d++)
                            score += q[h, tq, d] * k[kvHead, tk, d];
                        score *= scale;
                    }

                    scores[tk] = score;
                    max = MathF.Max(max, score);
                }

                float denom = 0;
                for (int tk = 0; tk < seqK; tk++)
                {
                    scores[tk] = float.IsNegativeInfinity(scores[tk])
                        ? 0f
                        : MathF.Exp(scores[tk] - max);
                    denom += scores[tk];
                }

                for (int d = 0; d < valueDim; d++)
                {
                    float sum = 0;
                    for (int tk = 0; tk < seqK; tk++)
                        sum += scores[tk] / denom * v[kvHead, tk, d];
                    result[(tq * heads + h) * valueDim + d] = sum;
                }
            }

        return result;
    }

    private static float[] Flatten3D(float[,,] input)
    {
        int d0 = input.GetLength(0);
        int d1 = input.GetLength(1);
        int d2 = input.GetLength(2);
        float[] result = new float[d0 * d1 * d2];
        int index = 0;
        for (int i = 0; i < d0; i++)
            for (int j = 0; j < d1; j++)
                for (int k = 0; k < d2; k++)
                    result[index++] = input[i, j, k];
        return result;
    }

    private static void ApplyNeoXReference(float[] values, float[] cos, float[] sin,
        int heads, int seq, int dim, int rotHalf, bool headFirst)
    {
        float[] source = (float[])values.Clone();
        for (int s = 0; s < seq; s++)
            for (int h = 0; h < heads; h++)
            {
                int baseOffset = headFirst
                    ? (h * seq + s) * dim
                    : (s * heads + h) * dim;
                for (int j = 0; j < rotHalf; j++)
                {
                    float c = cos[s * rotHalf + j];
                    float sn = sin[s * rotHalf + j];
                    float x0 = source[baseOffset + j];
                    float x1 = source[baseOffset + rotHalf + j];
                    values[baseOffset + j] = x0 * c - x1 * sn;
                    values[baseOffset + rotHalf + j] = x0 * sn + x1 * c;
                }
            }
    }

    private static float[] ScaledDotProductAttentionReference(float[,,,] q, float[,,,] k, float[,,,] v, float[,,,]? mask, float scale)
    {
        int batch = q.GetLength(0);
        int seqQ = q.GetLength(1);
        int heads = q.GetLength(2);
        int keyDim = q.GetLength(3);
        int seqK = k.GetLength(1);
        int valueDim = v.GetLength(3);
        float[] result = new float[batch * seqQ * heads * valueDim];
        float[] scores = new float[seqK];

        for (int b = 0; b < batch; b++)
            for (int tq = 0; tq < seqQ; tq++)
                for (int h = 0; h < heads; h++)
                {
                    float max = float.NegativeInfinity;
                    for (int tk = 0; tk < seqK; tk++)
                    {
                        float score = 0;
                        for (int d = 0; d < keyDim; d++)
                            score += q[b, tq, h, d] * k[b, tk, h, d];
                        score *= scale;
                        if (mask != null)
                            score += mask[b, h, tq, tk];
                        scores[tk] = score;
                        max = MathF.Max(max, score);
                    }

                    float denom = 0;
                    for (int tk = 0; tk < seqK; tk++)
                    {
                        scores[tk] = MathF.Exp(scores[tk] - max);
                        denom += scores[tk];
                    }

                    for (int d = 0; d < valueDim; d++)
                    {
                        float sum = 0;
                        for (int tk = 0; tk < seqK; tk++)
                            sum += scores[tk] / denom * v[b, tk, h, d];
                        result[((b * seqQ + tq) * heads + h) * valueDim + d] = sum;
                    }
                }

        return result;
    }
}
