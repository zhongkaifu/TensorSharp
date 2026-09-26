using System.Runtime.InteropServices;
using TensorSharp;
using TensorSharp.MLX;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

public class MlxBackendTests
{
    // These two test the availability probe itself, so they are the exemption
    // from the RS0030 ban on calling it directly (see BannedSymbols.txt).
#pragma warning disable RS0030
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
#pragma warning restore RS0030

    [MlxFact]
    public void MlxAddmm_MatchesCpuForContiguousRhs()
    {
        using var allocator = new MlxAllocator();
        using var a = Tensor.FromArray(allocator, new float[,] { { 1, 2, 3 }, { 4, 5, 6 } });
        using var b = Tensor.FromArray(allocator, new float[,] { { 7, 8 }, { 9, 10 }, { 11, 12 } });
        using var c = new Tensor(allocator, DType.Float32, 2, 2);

        Ops.Addmm(c, 0, c, 1, a, b);

        AssertClose(new[] { 58f, 64f, 139f, 154f }, c.GetElementsAsFloat(4));
    }

    [MlxFact]
    public void MlxSoftmax_MatchesExpectedRows()
    {
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

    [MlxFact]
    public void MlxMul_BroadcastsColumnVector()
    {
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

    [MlxFact]
    public void MlxHostWriteAfterDeviceOp_PreservesTensorData()
    {
        using var allocator = new MlxAllocator();
        using var tensor = Tensor.FromArray(allocator, new float[,] { { 1, 2 }, { 3, 4 } });
        Ops.Mul(tensor, tensor, 2f);

        tensor.SetElementAsFloat(100f, 0, 1);

        AssertClose(new[] { 2f, 100f, 6f, 8f }, tensor.GetElementsAsFloat(4));
    }

    [MlxFact]
    public void MlxCopyIntoNarrowView_UpdatesDeviceStorage()
    {
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

    [MlxFact]
    public void MlxFill_WritesDeviceResidentTensor()
    {
        using var allocator = new MlxAllocator();
        using var tensor = new Tensor(allocator, DType.Float32, 2, 3);

        Ops.Fill(tensor, 1.25f);

        AssertClose(new[] { 1.25f, 1.25f, 1.25f, 1.25f, 1.25f, 1.25f }, tensor.GetElementsAsFloat(6));
    }

    [MlxFact]
    public void MlxRmsNorm_MatchesExpectedRows()
    {
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

    [MlxFact]
    public void MlxLayerNorm_MatchesExpectedRows()
    {
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

    [MlxFact]
    public void MlxGeluAndGeluMul_MatchCpuFormula()
    {
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

    [MlxFact]
    public void MlxSiLUMulSplit_MatchesSplitReference()
    {
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

    [MlxFact]
    public void MlxScaledDotProductAttention_MatchesReferenceWithoutMask()
    {
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

        // MLX's fused fast::scaled_dot_product_attention is not an fp32-exact
        // kernel: on Metal-4-class GPUs its QK^T / PV matmuls run on the reduced-
        // precision tensor path, so results land ~6e-4 RELATIVE off an fp32 host
        // reference (here ~1.8e-4 absolute on values near 0.28) no matter how
        // small the problem is. That is fp16/bf16-class error, which is what every
        // production attention kernel delivers; the 1e-4 absolute bound this test
        // used only held on older hardware. TensorSharp's own fused kernels (see
        // MlxFusedPrefillAttention_* below) still assert at 1e-4.
        AssertClose(ScaledDotProductAttentionReference(q, k, v, null, scale), actualTensor.GetElementsAsFloat((int)actualTensor.ElementCount()), 1e-3f);
    }

    [MlxFact]
    public void MlxScaledDotProductAttention_MatchesReferenceWithMask()
    {
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

        // Same reduced-precision fused kernel as the maskless case above.
        AssertClose(ScaledDotProductAttentionReference(q, k, v, mask, scale), actualTensor.GetElementsAsFloat((int)actualTensor.ElementCount()), 1e-3f);
    }

    [MlxFact]
    public void MlxCopy_CastsFloat32ToFloat16OnDevice()
    {
        using var allocator = new MlxAllocator();
        using var src = Tensor.FromArray(allocator, new float[] { -1.25f, -0.5f, 0f, 0.75f, 1.5f, 3.25f });
        using var dst = new Tensor(allocator, DType.Float16, 6);

        Ops.Copy(dst, src);

        AssertClose(src.GetElementsAsFloat(6), dst.GetElementsAsFloat(6), 1e-3f);
    }

    [MlxFact]
    public void MlxFusedPrefillAttention_GqaMatchesReference()
    {
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

    [MlxFact]
    public void MlxFusedPrefillAttention_HeadDim256UsesChunkedVectorPath()
    {
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

    [MlxFact]
    public void MlxFusedDecodeAttention_ReadsFloat16KvCache()
    {
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

    [MlxFact]
    public void MlxFusedCircularDecodeAttention_ReadsWrappedFloat16KvCache()
    {
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

    [MlxFact]
    public void MlxIndexSelect_GathersRowsOnDevice()
    {
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

    [MlxFact]
    public void MlxFusedGatherRows_GathersRowsOnDevice()
    {
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

    [MlxFact]
    public void MlxFusedScatterAddWeightedRows_AccumulatesRowsOnDevice()
    {
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

    [MlxFact]
    public void MlxFusedRmsNormAddInPlace_MatchesCpu()
    {
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

    [MlxFact]
    public void MlxFusedGeluMulSplit_MatchesCpu()
    {
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

    [MlxFact]
    public void MlxFusedFlatToHeadFirst_MatchesReference()
    {
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

    [MlxFact]
    public void MlxFusedNeoXRoPEFlatAndHeadFirst_MatchReference()
    {
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

    [MlxFact]
    public void MlxQwen35PackedGdnDecode_MatchesSeparateProjectionPath()
    {
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

    [MlxFact]
    public void MlxRepeatInterleave_RepeatsAlongAxis()
    {
        using var allocator = new MlxAllocator();
        using var src = Tensor.FromArray(allocator, new float[,]
        {
            { 1, 2 },
            { 3, 4 },
        });
        using var repeated = Ops.RepeatInterleave(null, src, 2, 0);

        AssertClose(new[] { 1f, 2f, 1f, 2f, 3f, 4f, 3f, 4f }, repeated.GetElementsAsFloat(8));
    }

    [MlxFact]
    public void MlxAddCausalMask_MasksFuturePositionsOnDevice()
    {
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

    [MlxTheory]
    [InlineData(2, 64, 64)]   // NeoX over the whole row: gpt-oss attention
    [InlineData(0, 32, 48)]   // traditional, with columns past ropeDim left alone
    public void MlxRoPEEx_YarnMatchesTheCpuImplementation(int mode, int ropeDim, int cols)
    {
        // gpt-oss's YaRN parameters. These used to take the element-by-element CPU fallback;
        // the device path must reproduce the CPU arithmetic, mscale included, and leave the
        // columns past ropeDim exactly as they were.
        const int rows = 7;
        const int nCtxOrig = 4096;
        const float freqBase = 150000f, freqScale = 1f / 32f, extFactor = 1f, attnFactor = 1f, betaFast = 32f, betaSlow = 1f;
        float[,] source = new float[rows, cols];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                source[r, c] = MathF.Sin((r + 1) * 0.37f + (c + 1) * 0.11f);
        int[] positions = { 0, 1, 2, 17, 511, 4095, 20000 };

        var cpu = new TensorSharp.Cpu.CpuAllocator(BlasEnum.DotNet);
        using var cpuInput = Tensor.FromArray(cpu, source);
        using var cpuPositions = Tensor.FromArray(cpu, positions);
        using var expected = Ops.RoPEEx(null, cpuInput, cpuPositions, ropeDim, mode, nCtxOrig, freqBase, freqScale, extFactor, attnFactor, betaFast, betaSlow);

        using var allocator = new MlxAllocator();
        using var input = Tensor.FromArray(allocator, source);
        using var positionTensor = Tensor.FromArray(allocator, positions);
        using var actual = Ops.RoPEEx(null, input, positionTensor, ropeDim, mode, nCtxOrig, freqBase, freqScale, extFactor, attnFactor, betaFast, betaSlow);

        float[] want = expected.GetElementsAsFloat(rows * cols);
        float[] got = actual.GetElementsAsFloat(rows * cols);
        AssertClose(want, got, 2e-3f);
        for (int r = 0; r < rows; r++)
            for (int c = ropeDim; c < cols; c++)
                Assert.Equal(source[r, c], got[r * cols + c]);
    }

    [MlxFact]
    public void MlxRoPEEx_NeoXDynamicPositions_MatchesReference()
    {
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

    [MlxFact]
    public void MlxRoPEEx_InPlaceTraditional_MatchesReference()
    {
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

    [MlxFact]
    public void MlxQuantizedMatmul_Q80MatchesDequantizedReference()
    {
        using var exact = new ExactMlxMatmul();
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

    [MlxTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void MlxQuantizedMatmul_Q4MatchesDequantizedReference(bool hasExplicitBias)
    {
        using var exact = new ExactMlxMatmul();
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

    [MlxTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void MlxQuantizedMatmul_Q5MatchesDequantizedReference(bool hasExplicitBias)
    {
        using var exact = new ExactMlxMatmul();
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

    [MlxTheory]
    [InlineData((int)GgmlTensorType.Q4_K)]
    [InlineData((int)GgmlTensorType.Q5_K)]
    [InlineData((int)GgmlTensorType.Q6_K)]
    public void MlxQuantizedMatmul_KQuantsMatchDequantizedReference(int ggmlType)
    {
        using var exact = new ExactMlxMatmul();
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

    [MlxFact]
    public void MlxQuantizedMatmul_Q6KSingleRowMatchesDequantizedReference()
    {
        using var exact = new ExactMlxMatmul();
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
        int previousMatvecRows = MlxNative.Q6KMatvecMaxRows;
        try
        {
            // The 4-column kernel, not the matrix-vector port that now takes one row.
            MlxNative.Q6KMatvecMaxRows = 0;
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
            MlxNative.Q6KMatvecMaxRows = previousMatvecRows;
            Environment.SetEnvironmentVariable("TS_MLX_Q6K_MATMUL4", previousOptIn);
            Marshal.FreeHGlobal(host);
        }
    }

    [MlxTheory]
    [InlineData(1, 256, 13)]    // one block per row; a last threadgroup with one valid row
    [InlineData(1, 768, 16)]    // odd block count: the two half-simdgroups take 2 and 1 blocks
    [InlineData(3, 512, 13)]
    [InlineData(4, 1024, 6)]
    public void MlxQuantizedMatmul_Q6KMatvecMatchesDequantizedReference(int rows, int inDim, int outDim)
    {
        // Exact Q6_K (no 8-bit regroup) at the row counts the ggml matrix-vector port serves.
        using var exact = new ExactMlxMatmul();
        Assert.True(rows <= MlxNative.Q6KMatvecMaxRows);
        byte[] weights = CreateQ6KRows(outDim, inDim);
        float[,] input = new float[rows, inDim];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < inDim; c++)
                input[r, c] = MathF.Cos((c + 1) * 0.017f + r * 0.61f) * (1f + r);

        float[] expected = DequantizedMatmulK(weights, outDim, inDim, input, DequantizeQ6KRow);
        IntPtr host = Marshal.AllocHGlobal(weights.Length);
        try
        {
            Marshal.Copy(weights, 0, host, weights.Length);
            using var allocator = new MlxAllocator();
            using var inputTensor = Tensor.FromArray(allocator, input);
            using var outputTensor = new Tensor(allocator, DType.Float32, rows, outDim);

            Assert.True(MlxQuantizedOps.TryAddmmQuantizedToFloat32(
                outputTensor, inputTensor, host, host, (int)GgmlTensorType.Q6_K, inDim, outDim, weights.Length));

            AssertRelativelyClose(expected, outputTensor.GetElementsAsFloat(rows * outDim), 1e-5f);
        }
        finally
        {
            Marshal.FreeHGlobal(host);
        }
    }

    [MlxTheory]
    [InlineData(1, 256, 13)]    // one block per row; a last threadgroup with one valid row
    [InlineData(1, 768, 16)]    // odd block count: the two half-simdgroups take 2 and 1 blocks
    [InlineData(3, 512, 13)]
    [InlineData(4, 1024, 6)]
    public void MlxQuantizedMatmul_Iq4XsMatvecMatchesDequantizedReference(int rows, int inDim, int outDim)
    {
        Assert.True(rows <= MlxNative.Iq4XsMatvecMaxRows);
        byte[] weights = CreateIq4XsRows(outDim, inDim);
        float[,] input = new float[rows, inDim];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < inDim; c++)
                input[r, c] = MathF.Cos((c + 1) * 0.017f + r * 0.61f) * (1f + r);

        float[] expected = DequantizedMatmulIq4Xs(weights, outDim, inDim, input);
        IntPtr host = Marshal.AllocHGlobal(weights.Length);
        try
        {
            Marshal.Copy(weights, 0, host, weights.Length);
            using var allocator = new MlxAllocator();
            using var inputTensor = Tensor.FromArray(allocator, input);
            using var outputTensor = new Tensor(allocator, DType.Float32, rows, outDim);

            Assert.True(MlxQuantizedOps.TryAddmmQuantizedToFloat32(
                outputTensor, inputTensor, host, host, (int)GgmlTensorType.IQ4_XS, inDim, outDim, weights.Length));

            AssertRelativelyClose(expected, outputTensor.GetElementsAsFloat(rows * outDim), 1e-5f);
        }
        finally
        {
            Marshal.FreeHGlobal(host);
        }
    }

    [MlxTheory]
    [InlineData(8, 512, 13, false)]
    [InlineData(37, 768, 16, false)]
    [InlineData(8, 512, 13, true)]      // three slices of five output rows
    public void MlxQuantizedMatmul_Iq4XsPrefillThroughF16MatchesDequantizedReference(int rows, int inDim, int outDim, bool slices)
    {
        Assert.True(rows > MlxNative.Iq4XsMatvecMaxRows);
        byte[] weights = CreateIq4XsRows(outDim, inDim);
        float[,] input = new float[rows, inDim];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < inDim; c++)
                input[r, c] = MathF.Sin((c + 3) * 0.013f + r * 0.37f);

        float[] expected = DequantizedMatmulIq4Xs(weights, outDim, inDim, input);
        IntPtr host = Marshal.AllocHGlobal(weights.Length);
        long previousSlice = MlxNative.DequantSliceBytes;
        try
        {
            if (slices)
                MlxNative.DequantSliceBytes = 5L * 2 * inDim;
            Marshal.Copy(weights, 0, host, weights.Length);
            using var allocator = new MlxAllocator();
            using var inputTensor = Tensor.FromArray(allocator, input);
            using var outputTensor = new Tensor(allocator, DType.Float32, rows, outDim);

            Assert.True(MlxQuantizedOps.TryAddmmQuantizedToFloat32(
                outputTensor, inputTensor, host, host, (int)GgmlTensorType.IQ4_XS, inDim, outDim, weights.Length));

            AssertRelativelyClose(expected, outputTensor.GetElementsAsFloat(rows * outDim), 2e-3f);
        }
        finally
        {
            MlxNative.DequantSliceBytes = previousSlice;
            Marshal.FreeHGlobal(host);
        }
    }

    [MlxTheory]
    [InlineData(8, 512, 13, false)]
    [InlineData(37, 768, 16, false)]
    [InlineData(8, 512, 13, true)]      // three slices of five output rows
    public void MlxQuantizedMatmul_Q6KPrefillThroughF16MatchesDequantizedReference(int rows, int inDim, int outDim, bool slices)
    {
        // Past the matrix-vector rows, exact Q6_K is dequantized to F16 and multiplied
        // by MLX's GEMM: the weight and the rows are rounded to F16, the sums are F32.
        using var exact = new ExactMlxMatmul();
        Assert.True(rows > MlxNative.Q6KMatvecMaxRows);
        byte[] weights = CreateQ6KRows(outDim, inDim);
        float[,] input = new float[rows, inDim];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < inDim; c++)
                input[r, c] = MathF.Sin((c + 3) * 0.013f + r * 0.37f);

        float[] expected = DequantizedMatmulK(weights, outDim, inDim, input, DequantizeQ6KRow);
        IntPtr host = Marshal.AllocHGlobal(weights.Length);
        long previousSlice = MlxNative.DequantSliceBytes;
        try
        {
            if (slices)
                MlxNative.DequantSliceBytes = 5L * 2 * inDim;
            Marshal.Copy(weights, 0, host, weights.Length);
            using var allocator = new MlxAllocator();
            using var inputTensor = Tensor.FromArray(allocator, input);
            using var outputTensor = new Tensor(allocator, DType.Float32, rows, outDim);

            Assert.True(MlxQuantizedOps.TryAddmmQuantizedToFloat32(
                outputTensor, inputTensor, host, host, (int)GgmlTensorType.Q6_K, inDim, outDim, weights.Length));

            AssertRelativelyClose(expected, outputTensor.GetElementsAsFloat(rows * outDim), 2e-3f);
        }
        finally
        {
            MlxNative.DequantSliceBytes = previousSlice;
            Marshal.FreeHGlobal(host);
        }
    }

    [MlxFact]
    public void MlxQuantizedMatmul_Q5KSingleRow4ColumnMatchesDequantizedReference()
    {
        using var exact = new ExactMlxMatmul();
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

    [MlxFact]
    public void MlxQuantizedMatmul_RmsNormFusedMatchesReference()
    {
        using var exact = new ExactMlxMatmul();
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

    [MlxFact]
    public void MlxQuantizedMatmul_AddIntoFusedMatchesReference()
    {
        using var exact = new ExactMlxMatmul();
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

    [MlxFact]
    public void MlxQuantizedMatmul_MXFP4MatchesDequantizedReference()
    {
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

    [MlxFact]
    public void MlxQuantizedMatmul_IQ4XSMatchesDequantizedReferenceAfterHostRelease()
    {
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

    [MlxFact]
    public void MlxQuantizedMatmul_IQ2XXSMatchesNativeDequantizedReferenceAfterHostRelease()
    {
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

    [MlxTheory]
    [InlineData((int)GgmlTensorType.IQ2_S)]
    [InlineData((int)GgmlTensorType.IQ3_S)]
    public void MlxQuantizedMatmul_IQ2SAndIQ3SMatchNativeDequantizedReferenceAfterHostRelease(int ggmlType)
    {
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

    [MlxFact]
    public void MlxQuantizedRows_Q80MatchDequantizedReferenceAfterHostRelease()
    {
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

    [MlxTheory]
    [InlineData((int)GgmlTensorType.Q4_K)]
    [InlineData((int)GgmlTensorType.Q5_K)]
    [InlineData((int)GgmlTensorType.Q6_K)]
    public void MlxQuantizedRows_KQuantsMatchDequantizedReferenceAfterHostRelease(int ggmlType)
    {
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

    [MlxFact]
    public void MlxQuantizedMatmul_IQ4XSDecode4ColumnMatchesDequantizedReferenceAfterHostRelease()
    {
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

    [MlxFact]
    public void MlxQuantizedRows_IQ4XSMatchDequantizedReferenceAfterHostRelease()
    {
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

    [MlxFact]
    public void MlxQuantizedRows_IQ2XXSMatchNativeDequantizedReferenceAfterHostRelease()
    {
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

    [MlxTheory]
    [InlineData((int)GgmlTensorType.IQ2_S)]
    [InlineData((int)GgmlTensorType.IQ3_S)]
    public void MlxQuantizedRows_IQ2SAndIQ3SMatchNativeDequantizedReferenceAfterHostRelease(int ggmlType)
    {
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

    [MlxFact]
    public void MlxQuantizedRows_MXFP4MatchDequantizedReferenceAfterHostRelease()
    {
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

    // --- Prefill/decode paths ported from omlx/mlx-lm (see MlxFusedOps.TryCachedAttention,
    // MlxQuantizedOps.AffineQuantizedMatmul / TryRmsNormSwiGluAddHalf, GatedDeltaBlockedSource,
    // MlxBasicOps.TryCopyIntoStridedBox). ---

    [MlxFact]
    public void MlxCachedAttention_DecodeReadsOnlyTheWrittenRowsOfAFloat16Cache()
    {
        const int heads = 4, kvHeads = 2, dim = 256, capacity = 16, kvLen = 11;
        const float scale = 0.0625f;
        float[,,] q = BuildHeadFirstInput(heads, 1, dim, 0.041f, useCos: false);
        // Rows past kvLen hold other values: reading them would change the answer.
        float[,,] kAll = BuildHeadFirstInput(kvHeads, capacity, dim, 0.025f, useCos: true);
        float[,,] vAll = BuildHeadFirstInput(kvHeads, capacity, dim, 0.017f, useCos: false);

        using var allocator = new MlxAllocator();
        using var qTensor = Tensor.FromArray(allocator, FlattenHeadFirstSingleToken(q)).View(1, heads * dim);
        using var kCache = new Tensor(allocator, DType.Float16, kvHeads, capacity, dim);
        using var vCache = new Tensor(allocator, DType.Float16, kvHeads, capacity, dim);
        using (var kSrc = Tensor.FromArray(allocator, kAll)) Ops.Copy(kCache, kSrc);
        using (var vSrc = Tensor.FromArray(allocator, vAll)) Ops.Copy(vCache, vSrc);
        using var actual = new Tensor(allocator, DType.Float32, 1, heads * dim);

        Assert.True(MlxFusedOps.TryCachedAttention(actual, qTensor, kCache, vCache, heads, kvHeads, dim, 1, kvLen, scale));

        float[] expected = HeadFirstAttentionReference(q, SliceSeq(kAll, kvLen), SliceSeq(vAll, kvLen), scale, causal: false);
        AssertClose(expected, actual.GetElementsAsFloat(heads * dim), 2e-3f);
    }

    [MlxFact]
    public void MlxCachedAttention_ChunkAfterAPrefixIsCausalFromItsOwnStart()
    {
        // A second prefill chunk: rows [7, 11) attend to the 7 cached rows and to
        // their own lower triangle, as mlx-lm's "causal" mask aligned to the last key.
        const int heads = 4, kvHeads = 2, dim = 256, capacity = 16, start = 7, chunk = 4;
        const int kvLen = start + chunk;
        const float scale = 0.0625f;
        float[,,] q = BuildHeadFirstInput(heads, chunk, dim, 0.037f, useCos: false);
        float[,,] kAll = BuildHeadFirstInput(kvHeads, capacity, dim, 0.023f, useCos: true);
        float[,,] vAll = BuildHeadFirstInput(kvHeads, capacity, dim, 0.019f, useCos: false);
        float[] qRows = new float[chunk * heads * dim];
        for (int t = 0; t < chunk; t++)
            for (int h = 0; h < heads; h++)
                for (int d = 0; d < dim; d++)
                    qRows[(t * heads + h) * dim + d] = q[h, t, d];

        using var allocator = new MlxAllocator();
        using var qTensor = Tensor.FromArray(allocator, qRows).View(chunk, heads * dim);
        using var kCache = new Tensor(allocator, DType.Float16, kvHeads, capacity, dim);
        using var vCache = new Tensor(allocator, DType.Float16, kvHeads, capacity, dim);
        using (var kSrc = Tensor.FromArray(allocator, kAll)) Ops.Copy(kCache, kSrc);
        using (var vSrc = Tensor.FromArray(allocator, vAll)) Ops.Copy(vCache, vSrc);
        using var actual = new Tensor(allocator, DType.Float32, chunk, heads * dim);

        Assert.True(MlxFusedOps.TryCachedAttention(actual, qTensor, kCache, vCache, heads, kvHeads, dim, chunk, kvLen, scale));

        float[] expected = HeadFirstAttentionReference(q, SliceSeq(kAll, kvLen), SliceSeq(vAll, kvLen), scale, causal: true);
        AssertClose(expected, actual.GetElementsAsFloat(chunk * heads * dim), 2e-3f);
    }

    [MlxTheory]
    [InlineData(0, 9, 5, DType.Float16)]    // prefill longer than the window: explicit mask
    [InlineData(7, 4, 5, DType.Float16)]    // chunk after a prefix: the view skips the first rows
    [InlineData(13, 1, 5, DType.Float16)]   // decode past the window: view only, no mask
    [InlineData(0, 4, 5, DType.Float16)]    // everything inside the window: plain causal
    [InlineData(7, 4, 5, DType.Float32)]
    [InlineData(7, 4, 0, DType.Float16)]    // a full-attention layer: sinks only
    [InlineData(13, 1, 0, DType.Float16)]
    public void MlxCachedAttention_SinksAndSlidingWindowMatchTheReference(int start, int chunk, int window, DType cacheType)
    {
        // gpt-oss attention: a per-head sink logit in the softmax and, on alternate
        // layers, a sliding window of the last `window` positions (the query's own included).
        const int heads = 4, kvHeads = 2, dim = 64, capacity = 24;
        int kvLen = start + chunk;
        const float scale = 0.125f;
        float[] sinks = { 0.7f, -1.3f, 2.1f, 0.05f };
        float[,,] q = BuildHeadFirstInput(heads, chunk, dim, 0.037f, useCos: false);
        float[,,] kAll = BuildHeadFirstInput(kvHeads, capacity, dim, 0.023f, useCos: true);
        float[,,] vAll = BuildHeadFirstInput(kvHeads, capacity, dim, 0.019f, useCos: false);
        float[] qRows = new float[chunk * heads * dim];
        for (int t = 0; t < chunk; t++)
            for (int h = 0; h < heads; h++)
                for (int d = 0; d < dim; d++)
                    qRows[(t * heads + h) * dim + d] = q[h, t, d];

        using var allocator = new MlxAllocator();
        using var qTensor = Tensor.FromArray(allocator, qRows).View(chunk, heads * dim);
        using var kCache = new Tensor(allocator, cacheType, kvHeads, capacity, dim);
        using var vCache = new Tensor(allocator, cacheType, kvHeads, capacity, dim);
        using (var kSrc = Tensor.FromArray(allocator, kAll)) Ops.Copy(kCache, kSrc);
        using (var vSrc = Tensor.FromArray(allocator, vAll)) Ops.Copy(vCache, vSrc);
        using var sinkTensor = Tensor.FromArray(allocator, sinks);
        using var actual = new Tensor(allocator, DType.Float32, chunk, heads * dim);

        Assert.True(MlxFusedOps.TryCachedAttention(actual, qTensor, kCache, vCache, heads, kvHeads, dim,
            chunk, kvLen, scale, sinkTensor, window));

        float[] expected = HeadFirstAttentionReference(q, SliceSeq(kAll, kvLen), SliceSeq(vAll, kvLen), scale,
            causal: true, sinks, window);
        AssertClose(expected, actual.GetElementsAsFloat(chunk * heads * dim), cacheType == DType.Float32 ? 1e-4f : 2e-3f);
    }

    [MlxFact]
    public void MlxCopyIntoAnInnerAxisNarrow_WritesTheBoxOnDevice()
    {
        // The KV-cache growth copy: old rows into Narrow(1, 0, n) of the grown cache.
        using var allocator = new MlxAllocator();
        using var grown = new Tensor(allocator, DType.Float16, 2, 6, 4);
        Ops.Fill(grown, -1f);
        float[,,] old = new float[2, 3, 4];
        for (int h = 0; h < 2; h++)
            for (int t = 0; t < 3; t++)
                for (int d = 0; d < 4; d++)
                    old[h, t, d] = h * 100 + t * 10 + d;
        using var src = Tensor.FromArray(allocator, old);
        using (var box = grown.Narrow(1, 2, 3))
        {
            Assert.True(MlxBasicOps.TryGetStridedBox(box, out int[] parent, out int[] starts, out int[] stops));
            Assert.Equal(new[] { 2, 6, 4 }, parent);
            Assert.Equal(new[] { 0, 2, 0 }, starts);
            Assert.Equal(new[] { 2, 5, 4 }, stops);
            Ops.Copy(box, src);
        }

        float[] actual = grown.GetElementsAsFloat(48);
        for (int h = 0; h < 2; h++)
            for (int t = 0; t < 6; t++)
                for (int d = 0; d < 4; d++)
                {
                    float expected = t >= 2 && t < 5 ? h * 100 + (t - 2) * 10 + d : -1f;
                    Assert.Equal(expected, actual[(h * 6 + t) * 4 + d]);
                }
    }

    [MlxFact]
    public void MlxFill_Float16AndBoxViewsStayOnTheDevice()
    {
        // An F16 KV cache is zeroed on every allocation and reset; that used to go
        // element by element through the host.
        using var allocator = new MlxAllocator();
        long before = MlxCpuFallback.InvocationsOnThisThread;
        using var half = new Tensor(allocator, DType.Float16, 2, 6, 4);
        Ops.Fill(half, 3f);
        using (var box = half.Narrow(1, 2, 3))
            Ops.Fill(box, -1f);
        using var single = new Tensor(allocator, DType.Float32, 3, 5);
        Ops.Fill(single, 0.5f);
        using (var tail = single.Narrow(0, 1, 2))
            Ops.Fill(tail, 2f);
        Assert.Equal(before, MlxCpuFallback.InvocationsOnThisThread);

        float[] h = half.GetElementsAsFloat(48);
        for (int a = 0; a < 2; a++)
            for (int t = 0; t < 6; t++)
                for (int d = 0; d < 4; d++)
                    Assert.Equal(t >= 2 && t < 5 ? -1f : 3f, h[(a * 6 + t) * 4 + d]);
        float[] f = single.GetElementsAsFloat(15);
        for (int i = 0; i < 15; i++)
            Assert.Equal(i < 5 ? 0.5f : 2f, f[i]);
    }

    [MlxFact]
    public void MlxStridedBox_RejectsViewsThatAreNotABox()
    {
        using var allocator = new MlxAllocator();
        using var tensor = new Tensor(allocator, DType.Float32, 3, 4);
        using var transposed = tensor.Transpose();
        Assert.False(MlxBasicOps.TryGetStridedBox(transposed, out _, out _, out _));
        using var expanded = Tensor.FromArray(allocator, new float[] { 1, 2, 3 }).View(1, 3).Expand(4, 3);
        Assert.False(MlxBasicOps.TryGetStridedBox(expanded, out _, out _, out _));
    }

    [MlxFact]
    public void MlxQuantizedMatmul_Q80PrefillRowsMatchTheDequantizedReference()
    {
        // 48 rows takes the half-precision path (AffineQuantizedMatmul): F16 activations,
        // F32 accumulation, like ggml-metal's mul_mm.
        const int rows = 48, inDim = 128, outDim = 24;
        byte[] weights = CreateQ80Rows(outDim, inDim, (r, c) => (sbyte)(((r + 2) * (c - 23)) % 57), r => 0.02f + r * 0.004f);
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
                outputTensor, inputTensor, host, host, (int)GgmlTensorType.Q8_0, inDim, outDim, weights.Length));

            AssertRelativelyClose(expected, outputTensor.GetElementsAsFloat(rows * outDim), 3e-3f);
        }
        finally
        {
            Marshal.FreeHGlobal(host);
        }
    }

    [MlxFact]
    public void MlxHalfSwiGluFfn_MatchesTheF32Reference()
    {
        const int rows = 40, hidden = 64, inter = 96;
        const float eps = 1e-6f;
        byte[] gateUp = CreateQ80Rows(2 * inter, hidden, (r, c) => (sbyte)(((r * 7 + c * 3) % 61) - 30), r => 0.01f + (r % 5) * 0.002f);
        byte[] down = CreateQ80Rows(hidden, inter, (r, c) => (sbyte)(((r * 5 - c * 11) % 53) - 26), r => 0.012f + (r % 3) * 0.003f);
        float[,] residual = new float[rows, hidden];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < hidden; c++)
                residual[r, c] = MathF.Cos((r + 2) * (c + 1) * 0.029f) * 2f;
        float[] norm = new float[hidden];
        for (int c = 0; c < hidden; c++)
            norm[c] = 0.8f + (c % 9) * 0.05f;

        // Reference: rms_norm -> gate|up -> silu(gate) * up -> down -> + residual, all F32.
        float[,] normed = new float[rows, hidden];
        for (int r = 0; r < rows; r++)
        {
            double ss = 0;
            for (int c = 0; c < hidden; c++) ss += residual[r, c] * residual[r, c];
            float inv = 1f / MathF.Sqrt((float)(ss / hidden) + eps);
            for (int c = 0; c < hidden; c++) normed[r, c] = residual[r, c] * inv * norm[c];
        }
        float[] gu = DequantizedMatmulQ80(gateUp, 2 * inter, hidden, normed);
        float[,] act = new float[rows, inter];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < inter; c++)
            {
                float g = gu[r * 2 * inter + c], u = gu[r * 2 * inter + inter + c];
                act[r, c] = g / (1f + MathF.Exp(-g)) * u;
            }
        float[] dn = DequantizedMatmulQ80(down, hidden, inter, act);
        float[] expected = new float[rows * hidden];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < hidden; c++)
                expected[r * hidden + c] = residual[r, c] + dn[r * hidden + c];

        IntPtr gateUpHost = Marshal.AllocHGlobal(gateUp.Length);
        IntPtr downHost = Marshal.AllocHGlobal(down.Length);
        try
        {
            Marshal.Copy(gateUp, 0, gateUpHost, gateUp.Length);
            Marshal.Copy(down, 0, downHost, down.Length);
            using var allocator = new MlxAllocator();
            using var residualTensor = Tensor.FromArray(allocator, residual);
            using var normTensor = Tensor.FromArray(allocator, norm);

            Assert.True(MlxQuantizedOps.TryRmsNormSwiGluAddHalf(
                residualTensor, normTensor, eps, inter,
                gateUpHost, gateUpHost, (int)GgmlTensorType.Q8_0, hidden, 2 * inter, gateUp.Length,
                downHost, downHost, (int)GgmlTensorType.Q8_0, inter, hidden, down.Length));

            AssertRelativelyClose(expected, residualTensor.GetElementsAsFloat(rows * hidden), 5e-3f);
        }
        finally
        {
            Marshal.FreeHGlobal(gateUpHost);
            Marshal.FreeHGlobal(downHost);
        }
    }

    [MlxFact]
    public void MlxHalfSwiGluSplitFfn_MixedFormatsMatchTheF32Reference()
    {
        // A mixed-quant pair kept as two weights (IQ4_XS gate, affine Q8_0 up) and a raw
        // Q6_K down: two of the three go through the F16 dequantize + GEMM path.
        const int rows = 40, hidden = 256, inter = 256;
        const float eps = 1e-6f;
        byte[] gate = CreateIq4XsRows(inter, hidden);
        byte[] up = CreateQ80Rows(inter, hidden, (r, c) => (sbyte)(((r * 7 + c * 3) % 61) - 30), r => 0.01f + (r % 5) * 0.002f);
        byte[] down = CreateQ6KRows(hidden, inter);
        float[,] residual = new float[rows, hidden];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < hidden; c++)
                residual[r, c] = MathF.Cos((r + 2) * (c + 1) * 0.029f) * 2f;
        float[] norm = new float[hidden];
        for (int c = 0; c < hidden; c++)
            norm[c] = (0.8f + (c % 9) * 0.05f) * 0.01f;   // these test weights are large; keep F16 in range

        float[,] normed = new float[rows, hidden];
        for (int r = 0; r < rows; r++)
        {
            double ss = 0;
            for (int c = 0; c < hidden; c++) ss += residual[r, c] * residual[r, c];
            float inv = 1f / MathF.Sqrt((float)(ss / hidden) + eps);
            for (int c = 0; c < hidden; c++) normed[r, c] = residual[r, c] * inv * norm[c];
        }
        float[] g = DequantizedMatmulIq4Xs(gate, inter, hidden, normed);
        float[] u = DequantizedMatmulQ80(up, inter, hidden, normed);
        float[,] act = new float[rows, inter];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < inter; c++)
            {
                float gv = g[r * inter + c];
                act[r, c] = gv / (1f + MathF.Exp(-gv)) * u[r * inter + c];
            }
        float[] dn = DequantizedMatmulK(down, hidden, inter, act, DequantizeQ6KRow);
        // The FFN has to move the residual by far more than the tolerance, or a broken
        // one would pass.
        Assert.True(dn.Max(MathF.Abs) > 2f, $"FFN contribution too small to test: {dn.Max(MathF.Abs)}");
        float[] expected = new float[rows * hidden];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < hidden; c++)
                expected[r * hidden + c] = residual[r, c] + dn[r * hidden + c];

        using var exact = new ExactMlxMatmul();   // raw Q6_K, not the 8-bit regroup
        IntPtr gateHost = Marshal.AllocHGlobal(gate.Length);
        IntPtr upHost = Marshal.AllocHGlobal(up.Length);
        IntPtr downHost = Marshal.AllocHGlobal(down.Length);
        try
        {
            Marshal.Copy(gate, 0, gateHost, gate.Length);
            Marshal.Copy(up, 0, upHost, up.Length);
            Marshal.Copy(down, 0, downHost, down.Length);
            using var allocator = new MlxAllocator();
            using var residualTensor = Tensor.FromArray(allocator, residual);
            using var normTensor = Tensor.FromArray(allocator, norm);

            MlxQuantizedOps.HalfMatmulMinRows = 1;
            Assert.True(MlxQuantizedOps.TryRmsNormSwiGluAddHalfSplit(
                residualTensor, normTensor, eps,
                gateHost, gateHost, (int)GgmlTensorType.IQ4_XS, hidden, inter, gate.Length,
                upHost, upHost, (int)GgmlTensorType.Q8_0, hidden, inter, up.Length,
                downHost, downHost, (int)GgmlTensorType.Q6_K, inter, hidden, down.Length));

            AssertRelativelyClose(expected, residualTensor.GetElementsAsFloat(rows * hidden), 5e-3f);
        }
        finally
        {
            Marshal.FreeHGlobal(gateHost);
            Marshal.FreeHGlobal(upHost);
            Marshal.FreeHGlobal(downHost);
        }
    }

    [MlxFact]
    public unsafe void MlxGatedDeltaNetCache_ExportsImportsAndClonesItsState()
    {
        // The recurrent state a Qwen 3.5 holder carries on MLX: conv tail [convTail, qkvDim]
        // oldest-first, delta [Hv, Dv, Dk]. Checkpoints and swaps move it through these.
        const int convTail = 3, qkvDim = 40, hv = 2, dv = 4, dk = 8;
        float[] conv = new float[convTail * qkvDim];
        float[] delta = new float[hv * dv * dk];
        for (int i = 0; i < conv.Length; i++) conv[i] = i * 0.5f - 7f;
        for (int i = 0; i < delta.Length; i++) delta[i] = MathF.Sin(i * 0.3f);

        using var cache = new MlxFusedOps.GatedDeltaNetCache();
        Assert.False(cache.HasState);
        fixed (float* c = conv)
        fixed (float* d = delta)
            cache.ImportState(c, convTail, qkvDim, d, hv, dv, dk);
        Assert.True(cache.HasState);

        using var clone = cache.CloneState();
        cache.Reset();

        float[] convOut = new float[conv.Length];
        float[] deltaOut = new float[delta.Length];
        fixed (float* c = convOut)
        fixed (float* d = deltaOut)
            clone.ExportState(c, convOut.Length, d, deltaOut.Length);
        Assert.Equal(conv, convOut);
        Assert.Equal(delta, deltaOut);

        // A reset cache exports the implicit zero state.
        Array.Fill(convOut, 1f);
        Array.Fill(deltaOut, 1f);
        fixed (float* c = convOut)
        fixed (float* d = deltaOut)
            cache.ExportState(c, convOut.Length, d, deltaOut.Length);
        Assert.All(convOut, v => Assert.Equal(0f, v));
        Assert.All(deltaOut, v => Assert.Equal(0f, v));
    }

    [MlxFact]
    public unsafe void MlxViewHostCopies_RoundTripACacheBox()
    {
        using var allocator = new MlxAllocator();
        using var cache = new Tensor(allocator, DType.Float16, 2, 8, 4);
        Ops.Fill(cache, 0f);
        float[] rows = new float[2 * 3 * 4];
        for (int i = 0; i < rows.Length; i++) rows[i] = i - 5;
        System.Half[] half = rows.Select(v => (System.Half)v).ToArray();
        using (var box = cache.Narrow(1, 0, 3))
        fixed (System.Half* p = half)
            Assert.True(MlxFusedOps.TryWriteViewFromHost(box, (IntPtr)p, half.Length * sizeof(System.Half)));

        System.Half[] back = new System.Half[half.Length];
        using (var box = cache.Narrow(1, 0, 3))
        fixed (System.Half* p = back)
            Assert.True(MlxFusedOps.TryCopyViewToHost(box, (IntPtr)p, back.Length * sizeof(System.Half)));
        Assert.Equal(half, back);

        float[] all = cache.GetElementsAsFloat(2 * 8 * 4);
        for (int h = 0; h < 2; h++)
            for (int t = 0; t < 8; t++)
                for (int d = 0; d < 4; d++)
                    Assert.Equal(t < 3 ? rows[(h * 3 + t) * 4 + d] : 0f, all[(h * 8 + t) * 4 + d]);
    }

    [MlxFact]
    public void MlxQwen35PackedGdnPrefill_MatchesTokenByTokenDecode()
    {
        // Qwen3.5-shaped heads (128) take the blocked prefill kernel; 37 tokens leave a
        // partial 16-step block, and two key heads shared by four value heads exercise
        // the tiled head mapping (value head hv reads key head hv % Hk). The reference
        // is the same layer fed one token at a time, which runs the T = 1 kernel, and
        // then one more token through each cache to compare the carried state.
        string previousNative = Environment.GetEnvironmentVariable("TS_MLX_GDN_NATIVE");
        Environment.SetEnvironmentVariable("TS_MLX_GDN_NATIVE", null);
        try
        {
            const int seqLen = 37, numKeyHeads = 2, numValueHeads = 4, headDim = 128, convKernel = 4;
            const int keyDim = numKeyHeads * headDim, valueDim = numValueHeads * headDim;
            const int qkvDim = keyDim * 2 + valueDim, packedDim = qkvDim + valueDim + numValueHeads * 2;
            float[,] packed = new float[seqLen + 1, packedDim];
            for (int t = 0; t <= seqLen; t++)
                for (int i = 0; i < packedDim; i++)
                    packed[t, i] = MathF.Sin((t + 1) * 0.31f + (i + 1) * 0.013f) * (i < qkvDim ? 0.6f : 0.4f);
            float[,] convWeight = new float[qkvDim, convKernel];
            for (int i = 0; i < qkvDim; i++)
                for (int k = 0; k < convKernel; k++)
                    convWeight[i, k] = 0.15f + 0.1f * k + (i % 5) * 0.01f;
            float[] dtBias = new float[numValueHeads], aLog = new float[numValueHeads], norm = new float[headDim];
            for (int h = 0; h < numValueHeads; h++) { dtBias[h] = 0.05f * h; aLog[h] = -0.3f - 0.1f * h; }
            Array.Fill(norm, 1.0f);

            using var allocator = new MlxAllocator();
            using var conv = Tensor.FromArray(allocator, convWeight);
            using var dt = Tensor.FromArray(allocator, dtBias);
            using var a = Tensor.FromArray(allocator, aLog);
            using var n = Tensor.FromArray(allocator, norm);
            using var prefillCache = new MlxFusedOps.GatedDeltaNetCache();
            using var stepCache = new MlxFusedOps.GatedDeltaNetCache();

            float[] RunRows(MlxFusedOps.GatedDeltaNetCache cache, int first, int count)
            {
                float[,] rows = new float[count, packedDim];
                for (int t = 0; t < count; t++)
                    for (int i = 0; i < packedDim; i++)
                        rows[t, i] = packed[first + t, i];
                using var input = Tensor.FromArray(allocator, rows);
                using var output = new Tensor(allocator, DType.Float32, count, valueDim);
                Assert.True(cache.TryRunQwen35Packed(output, input, conv, dt, a, n,
                    count, packedDim, qkvDim, keyDim, valueDim, numKeyHeads, numValueHeads, headDim, headDim, convKernel, 1e-6f));
                return output.GetElementsAsFloat(count * valueDim);
            }

            float[] prefill = RunRows(prefillCache, 0, seqLen);
            float[] stepped = new float[seqLen * valueDim];
            for (int t = 0; t < seqLen; t++)
                Array.Copy(RunRows(stepCache, t, 1), 0, stepped, t * valueDim, valueDim);
            AssertClose(stepped, prefill, 2e-4f);

            AssertClose(RunRows(stepCache, seqLen, 1), RunRows(prefillCache, seqLen, 1), 2e-4f);
        }
        finally
        {
            Environment.SetEnvironmentVariable("TS_MLX_GDN_NATIVE", previousNative);
        }
    }

    [MlxFact]
    public void MlxGeluMulSplit_StaysFiniteForLargeGates()
    {
        // From MLX 0.32 on macOS 27 a bare tanh() in a custom kernel is the fast form,
        // which returns NaN past |x| ~ 44; Gemma 4 E4B's layer-0 FFN gates reach ~680
        // and every logit went NaN. gelu(x) -> x for large x, gelu(x) -> 0 for very
        // negative x.
        float[] gates = { -700f, -60f, -8f, -1f, 0f, 0.5f, 3f, 12f, 60f, 700f };
        float[,] gateUp = new float[1, gates.Length * 2];
        float[] expected = new float[gates.Length];
        for (int i = 0; i < gates.Length; i++)
        {
            float g = gates[i];
            float up = 0.25f + i * 0.1f;
            gateUp[0, i] = g;
            gateUp[0, gates.Length + i] = up;
            double inner = 0.7978845608 * (g + 0.044715 * g * g * g);
            expected[i] = (float)(0.5 * g * (1.0 + Math.Tanh(inner)) * up);
        }

        using var allocator = new MlxAllocator();
        using var input = Tensor.FromArray(allocator, gateUp);
        using var output = new Tensor(allocator, DType.Float32, 1, gates.Length);
        Assert.True(MlxFusedOps.TryGeluMulSplit(output, input, gates.Length));

        float[] actual = output.GetElementsAsFloat(gates.Length);
        Assert.All(actual, v => Assert.True(float.IsFinite(v), $"non-finite GELU output {v}"));
        AssertRelativelyClose(expected, actual, 1e-4f);
    }

    [MlxFact]
    public void MlxQuantizedMatmul_Q6KRegroupedToAffine8StaysWithinItsBound()
    {
        // The default regroups Q6_K (16-value scales) to MLX 8-bit affine over 32 values, so
        // each weight may move by up to half an 8-bit step of its group, plus the F16
        // rounding of the group's scale and bias. On top of that the activations and the
        // output are F16 (HalfMatmulMinRows). The bound is computed from the data.
        const int rows = 3, inDim = 512, outDim = 8, group = 32;
        const float half = 1f / 2048f;
        byte[] weights = CreateQ6KRows(outDim, inDim);
        float[,] input = new float[rows, inDim];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < inDim; c++)
                input[r, c] = MathF.Cos((r + 1) * (c + 1) * 0.017f);
        float[] expected = DequantizedMatmulK(weights, outDim, inDim, input, DequantizeQ6KRow);

        float[] dense = new float[outDim * inDim];
        for (int o = 0; o < outDim; o++)
            DequantizeQ6KRow(weights, o, inDim, dense, o * inDim);
        float[] weightError = new float[dense.Length];
        for (int o = 0; o < outDim; o++)
            for (int g = 0; g < inDim; g += group)
            {
                float min = float.PositiveInfinity, max = float.NegativeInfinity;
                for (int i = 0; i < group; i++)
                {
                    min = MathF.Min(min, dense[o * inDim + g + i]);
                    max = MathF.Max(max, dense[o * inDim + g + i]);
                }
                float step = (max - min) / 255f;
                float err = 0.5f * step + (MathF.Abs(min) + 255f * step) * half;
                for (int i = 0; i < group; i++)
                    weightError[o * inDim + g + i] = err;
            }

        IntPtr host = Marshal.AllocHGlobal(weights.Length);
        bool previous = MlxQuantizedOps.PreferAffine8Q6K;
        try
        {
            MlxQuantizedOps.PreferAffine8Q6K = true;
            MlxQuantizedOps.ClearDeviceCache(0);
            Marshal.Copy(weights, 0, host, weights.Length);
            using var allocator = new MlxAllocator();
            using var inputTensor = Tensor.FromArray(allocator, input);
            using var outputTensor = new Tensor(allocator, DType.Float32, rows, outDim);
            Assert.True(MlxQuantizedOps.TryAddmmQuantizedToFloat32(
                outputTensor, inputTensor, host, host, (int)GgmlTensorType.Q6_K, inDim, outDim, weights.Length));
            float[] actual = outputTensor.GetElementsAsFloat(rows * outDim);

            for (int r = 0; r < rows; r++)
                for (int o = 0; o < outDim; o++)
                {
                    double bound = 1e-4;
                    for (int i = 0; i < inDim; i++)
                    {
                        float x = MathF.Abs(input[r, i]);
                        bound += x * (weightError[o * inDim + i] + 2 * MathF.Abs(dense[o * inDim + i]) * half);
                    }
                    float want = expected[r * outDim + o];
                    bound += MathF.Abs(want) * 2 * half;
                    Assert.True(MathF.Abs(actual[r * outDim + o] - want) <= bound,
                        $"[{r},{o}] expected {want}, actual {actual[r * outDim + o]}, bound {bound}");
                }
        }
        finally
        {
            MlxQuantizedOps.PreferAffine8Q6K = previous;
            MlxQuantizedOps.ClearDeviceCache(0);
            Marshal.FreeHGlobal(host);
        }
    }

    [MlxFact]
    public void MlxQuantizedMatmul_HalfPrecisionRowsStayWithinF16OfTheReference()
    {
        // A two-row affine matmul takes the default half-precision path (one row would take
        // the custom Q8_0 kernel): F16 activations and output, F32 accumulation. Each term
        // may move by one F16 rounding of its activation, and the result by one of its own.
        const int rows = 2, inDim = 256, outDim = 16;
        const float half = 1f / 2048f;
        byte[] weights = CreateQ80Rows(outDim, inDim, (r, c) => (sbyte)(((r + 3) * (c - 41)) % 113), r => 0.03f + r * 0.01f);
        float[,] input = new float[rows, inDim];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < inDim; c++)
                input[r, c] = MathF.Sin((r + 2) * (c + 5) * 0.023f) * 3f;
        float[] expected = DequantizedMatmulQ80(weights, outDim, inDim, input);
        float[] dense = new float[outDim * inDim];
        for (int o = 0; o < outDim; o++)
            DequantizeQ80Row(weights, o, inDim, dense, o * inDim);

        IntPtr host = Marshal.AllocHGlobal(weights.Length);
        try
        {
            Marshal.Copy(weights, 0, host, weights.Length);
            using var allocator = new MlxAllocator();
            using var inputTensor = Tensor.FromArray(allocator, input);
            using var outputTensor = new Tensor(allocator, DType.Float32, rows, outDim);
            Assert.True(MlxQuantizedOps.TryAddmmQuantizedToFloat32(
                outputTensor, inputTensor, host, host, (int)GgmlTensorType.Q8_0, inDim, outDim, weights.Length));
            float[] actual = outputTensor.GetElementsAsFloat(rows * outDim);

            for (int r = 0; r < rows; r++)
                for (int o = 0; o < outDim; o++)
                {
                    double bound = 1e-4;
                    for (int i = 0; i < inDim; i++)
                        bound += MathF.Abs(input[r, i] * dense[o * inDim + i]) * 2 * half;
                    float want = expected[r * outDim + o];
                    bound += MathF.Abs(want) * 2 * half;
                    Assert.True(MathF.Abs(actual[r * outDim + o] - want) <= bound,
                        $"[{r},{o}] expected {want}, actual {actual[r * outDim + o]}, bound {bound}");
                }
        }
        finally
        {
            Marshal.FreeHGlobal(host);
        }
    }

    /// <summary>
    /// The kernels' exactness tests compare against an F32 dequantized reference, so they pin
    /// the configuration that reference describes: F32 activations and the lossless Q6_K
    /// kernel. The defaults (half-precision activations, Q6_K regrouped to 8-bit) have their
    /// own tests with their own bounds. Clears the weight cache on both edges so no weight
    /// built under one setting is served under the other.
    /// </summary>
    private sealed class ExactMlxMatmul : IDisposable
    {
        private readonly int _halfMinRows = MlxQuantizedOps.HalfMatmulMinRows;
        private readonly bool _q6kAffine8 = MlxQuantizedOps.PreferAffine8Q6K;

        public ExactMlxMatmul()
        {
            MlxQuantizedOps.HalfMatmulMinRows = int.MaxValue;
            MlxQuantizedOps.PreferAffine8Q6K = false;
            MlxQuantizedOps.ClearDeviceCache(0);
        }

        public void Dispose()
        {
            MlxQuantizedOps.HalfMatmulMinRows = _halfMinRows;
            MlxQuantizedOps.PreferAffine8Q6K = _q6kAffine8;
            MlxQuantizedOps.ClearDeviceCache(0);
        }
    }

    private static float[,,] SliceSeq(float[,,] input, int length)
    {
        float[,,] result = new float[input.GetLength(0), length, input.GetLength(2)];
        for (int h = 0; h < input.GetLength(0); h++)
            for (int t = 0; t < length; t++)
                for (int d = 0; d < input.GetLength(2); d++)
                    result[h, t, d] = input[h, t, d];
        return result;
    }

    private static void AssertRelativelyClose(IReadOnlyList<float> expected, IReadOnlyList<float> actual, float relative)
    {
        Assert.Equal(expected.Count, actual.Count);
        float scaleRef = 0;
        foreach (float e in expected) scaleRef = MathF.Max(scaleRef, MathF.Abs(e));
        for (int i = 0; i < expected.Count; i++)
        {
            float bound = relative * MathF.Max(scaleRef, 1f);
            Assert.True(MathF.Abs(expected[i] - actual[i]) <= bound,
                $"index {i}: expected {expected[i]}, actual {actual[i]} (bound {bound})");
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

    [MlxFact]
    public void MlxGatherQmm_MXFP4StackedExpertsMatchDequantizedReference()
    {
        const int numExperts = 4;
        const int inDim = 64;
        const int outDim = 6;
        const int n = 3;   // tokens
        const int k = 2;   // experts per token

        // Stacked GGUF layout: expert-major [E][outDim][blocksPerRow] blocks.
        byte[][] expertRows = new byte[numExperts][];
        for (int e = 0; e < numExperts; e++)
        {
            int ee = e;
            expertRows[e] = CreateMxfp4Rows(
                outDim,
                inDim,
                (r, c) => (byte)(((r + 3 + ee) * (c + 5)) & 0x0F),
                (r, b) => (byte)(125 + ((r + b + ee) % 5)));
        }
        byte[] stacked = new byte[expertRows[0].Length * numExperts];
        for (int e = 0; e < numExperts; e++)
            Array.Copy(expertRows[e], 0, stacked, e * expertRows[0].Length, expertRows[e].Length);

        float[,] input = new float[n, inDim];
        for (int r = 0; r < n; r++)
            for (int c = 0; c < inDim; c++)
                input[r, c] = MathF.Sin((r + 1) * (c + 3) * 0.019f);

        // (token, expert) pairs sorted by expert — the grouped-GEMM mode's contract.
        int[] tokenSorted = { 0, 1, 0, 2, 1, 2 };
        int[] expertsSorted = { 0, 0, 1, 2, 3, 3 };
        int nk = tokenSorted.Length;
        Assert.Equal(n * k, nk);

        float[] expected = new float[nk * outDim];
        float[] dequantRow = new float[inDim];
        for (int p = 0; p < nk; p++)
        {
            for (int o = 0; o < outDim; o++)
            {
                DequantizeMxfp4Row(expertRows[expertsSorted[p]], o, inDim, dequantRow, 0);
                float sum = 0;
                for (int c = 0; c < inDim; c++)
                    sum += input[tokenSorted[p], c] * dequantRow[c];
                expected[p * outDim + o] = sum;
            }
        }

        IntPtr host = Marshal.AllocHGlobal(stacked.Length);
        try
        {
            Marshal.Copy(stacked, 0, host, stacked.Length);
            using var allocator = new MlxAllocator();
            using var inputTensor = Tensor.FromArray(allocator, input);
            using var input3 = inputTensor.View(n, 1, inDim);
            using var lhs = new Tensor(allocator, DType.Int32, nk);
            lhs.SetElementsAsInt(tokenSorted);
            using var rhs = new Tensor(allocator, DType.Int32, nk);
            rhs.SetElementsAsInt(expertsSorted);
            using var result = new Tensor(allocator, DType.Float32, nk, outDim);

            Assert.True(MlxQuantizedOps.TryGatherQmm(
                result, input3, lhs, rhs,
                host, host, (int)GgmlTensorType.MXFP4, inDim, outDim, numExperts, stacked.Length,
                sortedIndices: true));

            AssertClose(expected, result.GetElementsAsFloat(nk * outDim), 2e-3f);

            // Null-lhs variant (rows pre-gathered into pair order) — the shape
            // the model's fast path uses so MLX's grouped-GEMM kernel engages.
            float[,] gathered = new float[nk, inDim];
            for (int p = 0; p < nk; p++)
                for (int c = 0; c < inDim; c++)
                    gathered[p, c] = input[tokenSorted[p], c];
            using var gatheredTensor = Tensor.FromArray(allocator, gathered);
            using var gathered3 = gatheredTensor.View(nk, 1, inDim);
            using var result2 = new Tensor(allocator, DType.Float32, nk, outDim);

            Assert.True(MlxQuantizedOps.TryGatherQmm(
                result2, gathered3, null, rhs,
                host, host, (int)GgmlTensorType.MXFP4, inDim, outDim, numExperts, stacked.Length,
                sortedIndices: true));

            AssertClose(expected, result2.GetElementsAsFloat(nk * outDim), 2e-3f);
        }
        finally
        {
            Marshal.FreeHGlobal(host);
        }
    }

    [MlxFact]
    public void MlxGatherQmm_SortedRhsProbe_GatesTheFastPath()
    {
        using var allocator = new MlxAllocator();

        // The probe must return a stable verdict for the shipping GPT-OSS
        // expert type. On devices where the sorted-rhs grouped-GEMM kernel is
        // numerically broken (e.g. the MLX NAX variant on M5-class GPUs) it
        // must return false so the model's prefill falls back; when it returns
        // true, the fast path must actually match the per-row reference path
        // at a fast-path-eligible shape (B=64, B/E=16).
        bool usable = MlxQuantizedOps.GatherQmmSortedRhsUsable(allocator, (int)GgmlTensorType.MXFP4);
        Assert.Equal(usable, MlxQuantizedOps.GatherQmmSortedRhsUsable(allocator, (int)GgmlTensorType.MXFP4));

        const int numExperts = 4;
        const int inDim = 64;
        const int outDim = 64;
        const int nk = 64;   // B >= 16 and B/E = 16 >= 4 -> grouped-GEMM eligible

        byte[][] expertRows = new byte[numExperts][];
        for (int e = 0; e < numExperts; e++)
        {
            int ee = e;
            expertRows[e] = CreateMxfp4Rows(
                outDim,
                inDim,
                (r, c) => (byte)(((r + 3 + ee) * (c + 5)) & 0x0F),
                (r, b) => (byte)(125 + ((r + b + ee) % 5)));
        }
        byte[] stacked = new byte[expertRows[0].Length * numExperts];
        for (int e = 0; e < numExperts; e++)
            Array.Copy(expertRows[e], 0, stacked, e * expertRows[0].Length, expertRows[e].Length);

        int[] expertsSorted = new int[nk];
        int[] arange = new int[nk];
        for (int i = 0; i < nk; i++)
        {
            expertsSorted[i] = i / (nk / numExperts);
            arange[i] = i;
        }
        float[,] x = new float[nk, inDim];
        for (int r = 0; r < nk; r++)
            for (int c = 0; c < inDim; c++)
                x[r, c] = MathF.Sin((r + 1) * (c + 3) * 0.017f) * 0.1f;

        IntPtr host = Marshal.AllocHGlobal(stacked.Length);
        try
        {
            Marshal.Copy(stacked, 0, host, stacked.Length);
            using var xT = Tensor.FromArray(allocator, x);
            using var x3 = xT.View(nk, 1, inDim);
            using var rhs = new Tensor(allocator, DType.Int32, nk);
            rhs.SetElementsAsInt(expertsSorted);
            using var lhs = new Tensor(allocator, DType.Int32, nk);
            lhs.SetElementsAsInt(arange);
            using var reference = new Tensor(allocator, DType.Float32, nk, outDim);

            // The per-row reference path (explicit lhs) is correct everywhere.
            Assert.True(MlxQuantizedOps.TryGatherQmm(
                reference, x3, lhs, rhs,
                host, host, (int)GgmlTensorType.MXFP4, inDim, outDim, numExperts, stacked.Length,
                sortedIndices: false));
            float[] expected = new float[nk * outDim];
            float[] dequantRow = new float[inDim];
            for (int p = 0; p < nk; p++)
            {
                for (int o = 0; o < outDim; o++)
                {
                    DequantizeMxfp4Row(expertRows[expertsSorted[p]], o, inDim, dequantRow, 0);
                    float sum = 0;
                    for (int c = 0; c < inDim; c++)
                        sum += x[p, c] * dequantRow[c];
                    expected[p * outDim + o] = sum;
                }
            }
            AssertClose(expected, reference.GetElementsAsFloat(nk * outDim), 2e-3f);

            if (usable)
            {
                // The probe blessed the fast path: it must agree with the reference.
                using var fast = new Tensor(allocator, DType.Float32, nk, outDim);
                Assert.True(MlxQuantizedOps.TryGatherQmm(
                    fast, x3, null, rhs,
                    host, host, (int)GgmlTensorType.MXFP4, inDim, outDim, numExperts, stacked.Length,
                    sortedIndices: true));
                AssertClose(expected, fast.GetElementsAsFloat(nk * outDim), 2e-2f);
            }
        }
        finally
        {
            Marshal.FreeHGlobal(host);
        }
    }

    [MlxFact]
    public void MlxGatherQmm_MXFP4GptOssShapes_NullLhsMatchesProvidedLhs()
    {
        // GPT-OSS-20b decode/short-prefill shapes: E=32 experts, 2880x2880
        // projections, NK=56 pairs (14 tokens x K=4). NK/E < 4 keeps this on
        // the per-row gather_qmv kernel, matching what decode dispatches.
        const int numExperts = 32;
        const int inDim = 2880;
        const int outDim = 2880;
        const int n = 14;
        const int k = 4;
        const int nk = n * k;

        var rng = new Random(1234);
        byte[] stacked = new byte[(long)numExperts * outDim * (inDim / 32) * 17];
        rng.NextBytes(stacked);
        // Clamp scale bytes to sane e8m0 range so dequant values stay finite.
        int blocksPerRow = inDim / 32;
        for (long b = 0; b < (long)numExperts * outDim * blocksPerRow; b++)
            stacked[b * 17] = (byte)(120 + (stacked[b * 17] % 10));

        float[,] input = new float[n, inDim];
        for (int r = 0; r < n; r++)
            for (int c = 0; c < inDim; c++)
                input[r, c] = (float)(rng.NextDouble() * 2 - 1) * 0.05f;

        // Random routing then sort pairs by expert.
        int[] expertsOrig = new int[nk];
        for (int p = 0; p < nk; p++) expertsOrig[p] = rng.Next(numExperts);
        int[] order = Enumerable.Range(0, nk).OrderBy(p => expertsOrig[p]).ToArray();
        int[] expertsSorted = new int[nk];
        int[] tokenSorted = new int[nk];
        for (int i = 0; i < nk; i++)
        {
            expertsSorted[i] = expertsOrig[order[i]];
            tokenSorted[i] = order[i] / k;
        }

        float[,] gathered = new float[nk, inDim];
        for (int p = 0; p < nk; p++)
            for (int c = 0; c < inDim; c++)
                gathered[p, c] = input[tokenSorted[p], c];

        IntPtr host = Marshal.AllocHGlobal(stacked.Length);
        try
        {
            Marshal.Copy(stacked, 0, host, stacked.Length);
            using var allocator = new MlxAllocator();
            using var inputTensor = Tensor.FromArray(allocator, input);
            using var input3 = inputTensor.View(n, 1, inDim);
            using var gatheredTensor = Tensor.FromArray(allocator, gathered);
            using var gathered3 = gatheredTensor.View(nk, 1, inDim);
            using var lhs = new Tensor(allocator, DType.Int32, nk);
            lhs.SetElementsAsInt(tokenSorted);
            using var rhs = new Tensor(allocator, DType.Int32, nk);
            rhs.SetElementsAsInt(expertsSorted);
            using var resultLhs = new Tensor(allocator, DType.Float32, nk, outDim);
            using var resultNull = new Tensor(allocator, DType.Float32, nk, outDim);

            Assert.True(MlxQuantizedOps.TryGatherQmm(
                resultLhs, input3, lhs, rhs,
                host, host, (int)GgmlTensorType.MXFP4, inDim, outDim, numExperts, stacked.Length,
                sortedIndices: true));
            Assert.True(MlxQuantizedOps.TryGatherQmm(
                resultNull, gathered3, null, rhs,
                host, host, (int)GgmlTensorType.MXFP4, inDim, outDim, numExperts, stacked.Length,
                sortedIndices: true));

            float[] a = resultLhs.GetElementsAsFloat(nk * outDim);
            float[] b = resultNull.GetElementsAsFloat(nk * outDim);
            int mismatches = 0;
            float worst = 0f;
            for (int i = 0; i < a.Length; i++)
            {
                float diff = Math.Abs(a[i] - b[i]);
                if (diff > 1e-2f) { mismatches++; worst = Math.Max(worst, diff); }
            }
            Assert.True(mismatches == 0, $"null-lhs deviates from provided-lhs: {mismatches} elements, worst diff {worst}");

            // Spot-check a few rows against the CPU dequantized reference.
            float[] dequantRow = new float[inDim];
            foreach (int p in new[] { 0, 7, nk - 1 })
            {
                int e = expertsSorted[p];
                for (int o = 0; o < outDim; o += 977)
                {
                    DequantizeMxfp4RowStacked(stacked, e, o, inDim, outDim, dequantRow);
                    float sum = 0;
                    for (int c = 0; c < inDim; c++)
                        sum += gathered[p, c] * dequantRow[c];
                    float actual = a[p * outDim + o];
                    Assert.True(Math.Abs(sum - actual) <= 5e-2f, $"pair {p} out {o}: cpu {sum} vs mlx {actual}");
                }
            }
        }
        finally
        {
            Marshal.FreeHGlobal(host);
        }
    }

    private static void DequantizeMxfp4RowStacked(byte[] stacked, int expert, int row, int inDim, int outDim, float[] destination)
    {
        int blocksPerRow = inDim / 32;
        long expertBytes = (long)outDim * blocksPerRow * 17;
        byte[] slice = new byte[outDim * blocksPerRow * 17];
        Array.Copy(stacked, expert * expertBytes, slice, 0, slice.Length);
        DequantizeMxfp4Row(slice, row, inDim, destination, 0);
    }

    [MlxFact]
    public void MlxSwiGluOaiGatherBias_MatchesCpuReference()
    {
        const int rows = 5;
        const int dim = 33;   // deliberately not a threadgroup-width multiple
        const int numExperts = 3;
        const float alpha = 1.702f;
        const float limit = 7.0f;

        float[,] gate = new float[rows, dim];
        float[,] up = new float[rows, dim];
        float[,] gateBias = new float[numExperts, dim];
        float[,] upBias = new float[numExperts, dim];
        int[] experts = { 2, 0, 1, 2, 0 };
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < dim; c++)
            {
                gate[r, c] = MathF.Sin((r + 1) * (c + 2) * 0.13f) * 9f;   // exercises the limit clamp
                up[r, c] = MathF.Cos((r + 2) * (c + 1) * 0.11f) * 9f;
            }
        for (int e = 0; e < numExperts; e++)
            for (int c = 0; c < dim; c++)
            {
                gateBias[e, c] = 0.05f * (e + 1) * MathF.Sin(c * 0.7f);
                upBias[e, c] = 0.04f * (e + 1) * MathF.Cos(c * 0.5f);
            }

        float[] expected = new float[rows * dim];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < dim; c++)
            {
                float g = gate[r, c] + gateBias[experts[r], c];
                float u = up[r, c] + upBias[experts[r], c];
                float x = MathF.Min(g, limit);
                float y = Math.Clamp(u, -limit, limit);
                float glu = x / (1.0f + MathF.Exp(-alpha * x));
                expected[r * dim + c] = glu * (y + 1.0f);
            }

        using var allocator = new MlxAllocator();
        using var gateT = Tensor.FromArray(allocator, gate);
        using var upT = Tensor.FromArray(allocator, up);
        using var gateBiasT = Tensor.FromArray(allocator, gateBias);
        using var upBiasT = Tensor.FromArray(allocator, upBias);
        using var expertsT = new Tensor(allocator, DType.Int32, rows);
        expertsT.SetElementsAsInt(experts);
        using var result = new Tensor(allocator, DType.Float32, rows, dim);

        Assert.True(MlxFusedOps.TrySwiGluOaiGatherBias(result, gateT, upT, gateBiasT, upBiasT, expertsT, alpha, limit));
        AssertClose(expected, result.GetElementsAsFloat(rows * dim), 1e-4f);
    }

    [MlxFact]
    public void MlxMoeBiasWeightedSum_MatchesCpuReference()
    {
        const int n = 3;
        const int k = 2;
        const int dim = 17;
        const int numExperts = 4;
        const int nk = n * k;

        // Pairs in original (token-major) order routed to experts, then sorted by expert.
        int[] expertsOrig = { 1, 3, 0, 3, 2, 1 };
        int[] order = Enumerable.Range(0, nk).OrderBy(p => expertsOrig[p]).ToArray();
        int[] expertsSorted = new int[nk];
        int[] invOrder = new int[nk];
        for (int i = 0; i < nk; i++)
        {
            expertsSorted[i] = expertsOrig[order[i]];
            invOrder[order[i]] = i;
        }

        float[,] downSorted = new float[nk, dim];
        float[,] downBias = new float[numExperts, dim];
        float[] weights = new float[nk];
        for (int i = 0; i < nk; i++)
        {
            weights[i] = 0.1f + 0.13f * i;
            for (int c = 0; c < dim; c++)
                downSorted[i, c] = MathF.Sin((i + 1) * (c + 2) * 0.21f);
        }
        for (int e = 0; e < numExperts; e++)
            for (int c = 0; c < dim; c++)
                downBias[e, c] = 0.02f * (e + 1) * MathF.Cos(c * 0.3f);

        float[] expected = new float[n * dim];
        float[] expectedNoBias = new float[n * dim];
        for (int t = 0; t < n; t++)
            for (int c = 0; c < dim; c++)
            {
                float acc = 0f, accNb = 0f;
                for (int kk = 0; kk < k; kk++)
                {
                    int p = t * k + kk;
                    int srow = invOrder[p];
                    float v = downSorted[srow, c];
                    accNb += weights[p] * v;
                    acc += weights[p] * (v + downBias[expertsSorted[srow], c]);
                }
                expected[t * dim + c] = acc;
                expectedNoBias[t * dim + c] = accNb;
            }

        using var allocator = new MlxAllocator();
        using var downT = Tensor.FromArray(allocator, downSorted);
        using var biasT = Tensor.FromArray(allocator, downBias);
        using var expertsT = new Tensor(allocator, DType.Int32, nk);
        expertsT.SetElementsAsInt(expertsSorted);
        using var invOrderT = new Tensor(allocator, DType.Int32, nk);
        invOrderT.SetElementsAsInt(invOrder);
        using var weightsT = new Tensor(allocator, DType.Float32, nk);
        weightsT.SetElementsAsFloat(weights);
        using var output = new Tensor(allocator, DType.Float32, n, dim);

        Assert.True(MlxFusedOps.TryMoeBiasWeightedSum(output, downT, biasT, expertsT, invOrderT, weightsT, k));
        AssertClose(expected, output.GetElementsAsFloat(n * dim), 1e-5f);

        Assert.True(MlxFusedOps.TryMoeBiasWeightedSum(output, downT, null, expertsT, invOrderT, weightsT, k));
        AssertClose(expectedNoBias, output.GetElementsAsFloat(n * dim), 1e-5f);
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

    private static float[] HeadFirstAttentionReference(float[,,] q, float[,,] k, float[,,] v, float scale, bool causal,
        float[] sinks = null, int window = 0)
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
                    int queryPos = tq + (seqK - seqQ);
                    bool masked = (causal && tk > queryPos) || (window > 0 && queryPos - tk >= window);
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

                if (sinks != null)
                    max = MathF.Max(max, sinks[h]);
                float denom = sinks != null ? MathF.Exp(sinks[h] - max) : 0f;
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
