using System.Buffers.Binary;
using System.Runtime.Intrinsics;

namespace InferenceWeb.Tests;

/// <summary>
/// The ARM64 (AdvSimd + DotProd) bodies of the managed quantized kernels (ManagedQuantizedOps.Arm.cs) against the
/// scalar loops they replace, BIT FOR BIT: they may only change how fast the managed CPU backend runs on ARM64,
/// never what it computes. Skipped on hosts without the ARM64 dot-product extension.
/// </summary>
public class ManagedQuantArmTests
{
    public static IEnumerable<object[]> DotCases()
    {
        foreach (var type in new[] { GgmlTensorType.Q4_K, GgmlTensorType.Q6_K })
            foreach (int k in new[] { 256, 512, 2048, 7680 })
                yield return new object[] { type, k };
        foreach (var type in new[] { GgmlTensorType.Q4_0, GgmlTensorType.Q8_0 })
            foreach (int k in new[] { 32, 64, 96, 128, 160, 224, 2048, 7680 })
                yield return new object[] { type, k };
        foreach (int k in new[] { 64, 128, 192, 2048, 7680 })
            yield return new object[] { GgmlTensorType.NVFP4, k };
    }

    [ArmDotTheory]
    [MemberData(nameof(DotCases))]
    public unsafe void PerRowDot_IsBitIdenticalToTheScalarLoop(GgmlTensorType type, int k)
    {
        var rng = new Random(1000 + (int)type * 31 + k);
        int rows = 24;
        Assert.True(ManagedQuantizedOps.TryGetActivationPlan(type, k, out int actBytes));
        byte[] weights = RandomWeights(rng, type, rows, k);
        int rowBytes = weights.Length / rows;
        float[] x = new float[k];
        byte[] act = new byte[actBytes + 7];
        for (int trial = 0; trial < rows; trial++)
        {
            for (int i = 0; i < k; i++)
                x[i] = (float)(rng.NextDouble() * 2 - 1) * (trial % 5 == 0 ? 1e-3f : 3f) * (i % 97 == 0 ? 40f : 1f);
            int offset = trial % 7;   // unaligned activation rows too
            fixed (float* xp = x)
            fixed (byte* ap = act)
            fixed (byte* wp = weights)
            {
                ManagedQuantizedOps.QuantizeActivationRow(type, xp, ap + offset, k);
                byte* w = wp + (long)trial * rowBytes;
                float expected = ManagedQuantizedOps.DotQuantizedRowScalar(type, w, ap + offset, k);
                float actual = ManagedQuantizedOps.DotQuantizedRow(type, w, ap + offset, k);
                Assert.True(BitConverter.SingleToInt32Bits(expected) == BitConverter.SingleToInt32Bits(actual),
                    $"{type} K={k} row {trial}: scalar {expected:R} vs ARM {actual:R}");
            }
        }
    }

    public static IEnumerable<object[]> GemmCases()
    {
        foreach (var type in new[] { GgmlTensorType.Q4_K, GgmlTensorType.Q6_K, GgmlTensorType.Q4_0, GgmlTensorType.Q8_0, GgmlTensorType.NVFP4 })
        {
            int k = type is GgmlTensorType.Q4_K or GgmlTensorType.Q6_K ? 512 : 192;
            foreach (int rows in new[] { 2, 3, 4, 5, 7, 8, 9, 17, 33 })
                foreach (int n in new[] { 1, 3, 4, 6, 37 })
                    yield return new object[] { type, rows, k, n };
            yield return new object[] { type, 70, 2048, 64 };
        }
    }

    /// <summary>The ARM GEMM (any row count it takes) against the per-row dots, bit for bit, with strided input and
    /// output rows: the row count of a call must never change a result.</summary>
    [ArmDotTheory]
    [MemberData(nameof(GemmCases))]
    public unsafe void ArmGemm_IsBitIdenticalToThePerRowDots(GgmlTensorType type, int rows, int k, int n)
    {
        var rng = new Random(77 + rows * 13 + k + n * 7 + (int)type);
        byte[] weights = RandomWeights(rng, type, n, k);
        int inStride = k + 5, outStride = n + 3;
        float[] x = new float[rows * inStride];
        for (int i = 0; i < x.Length; i++)
            x[i] = (float)(rng.NextDouble() * 2 - 1) * ((i / inStride) % 4 == 3 ? 1e-3f : 2f);
        for (int i = 0; i < Math.Min(k, 256); i++)
            x[inStride + i] = 0f;   // row 1 opens with an all-zero block
        float[] gemm = new float[rows * outStride], perRow = new float[rows * outStride];
        Array.Fill(gemm, float.NaN);
        Array.Fill(perRow, float.NaN);
        fixed (byte* w = weights)
        fixed (float* xp = x)
        fixed (float* g = gemm)
        fixed (float* p = perRow)
        {
            Assert.True(ManagedQuantizedOps.TryQGemmArm((int)type, (IntPtr)w, k, n, xp, inStride, rows, g, outStride, null));
            Assert.True(ManagedQuantizedOps.TryAddmmQuantizedToFloat32((int)type, (IntPtr)w, k, n, xp, inStride, rows,
                p, outStride, null, ManagedQuantizedOps.QGemmIsa.PerRow));
        }
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < outStride; c++)
            {
                float e = perRow[r * outStride + c], a = gemm[r * outStride + c];
                if (c >= n)
                {
                    Assert.True(float.IsNaN(a), $"padding written at row {r} col {c}");
                    continue;
                }
                Assert.True(BitConverter.SingleToInt32Bits(e) == BitConverter.SingleToInt32Bits(a),
                    $"{type} rows={rows} K={k} N={n} [{r},{c}]: per-row {e:R} vs GEMM {a:R}");
            }
    }

    /// <summary>A decode row's product as the pool runs it (ARM64: the shrinking column blocks of
    /// <see cref="ManagedQuantizedOps.DecodeColumnBounds"/>, at the pool's full width and narrowed to two threads) against
    /// the same product on one thread, bit for bit: which thread computes a column must never change it.</summary>
    [ArmDotTheory]
    [InlineData(GgmlTensorType.Q4_K, 2048, 3072)]
    [InlineData(GgmlTensorType.Q6_K, 7680, 2048)]
    [InlineData(GgmlTensorType.Q4_0, 2048, 2050)]
    [InlineData(GgmlTensorType.Q8_0, 2048, 1030)]
    [InlineData(GgmlTensorType.NVFP4, 2048, 4099)]
    public unsafe void DecodeProducts_AreBitIdenticalAtAnyPoolWidth(GgmlTensorType type, int k, int n)
    {
        var rng = new Random(5 + k + n + (int)type);
        byte[] weights = RandomWeights(rng, type, n, k);
        float[] x = new float[k];
        for (int i = 0; i < k; i++) x[i] = (float)(rng.NextDouble() * 2 - 1) * (i % 61 == 0 ? 30f : 1f);
        float[] serial = new float[n], pooled = new float[n], narrow = new float[n];
        CpuWorkerPool pool = CpuWorkerPool.Shared;
        fixed (byte* w = weights)
        fixed (float* xp = x)
        fixed (float* s = serial)
        fixed (float* p = pooled)
        fixed (float* q = narrow)
        {
            Assert.True(ManagedQuantizedOps.TryAddmmQuantizedToFloat32((int)type, (IntPtr)w, k, n, xp, k, 1, s, n,
                new System.Threading.Tasks.ParallelOptions { MaxDegreeOfParallelism = 1 }));
            Assert.True(ManagedQuantizedOps.TryAddmmQuantizedToFloat32((int)type, (IntPtr)w, k, n, xp, k, 1, p, n));
            int width = pool.ActiveThreads;
            try
            {
                pool.SetActiveThreads(2);
                Assert.True(ManagedQuantizedOps.TryAddmmQuantizedToFloat32((int)type, (IntPtr)w, k, n, xp, k, 1, q, n));
            }
            finally
            {
                pool.SetActiveThreads(width);
            }
        }
        for (int c = 0; c < n; c++)
        {
            Assert.True(BitConverter.SingleToInt32Bits(serial[c]) == BitConverter.SingleToInt32Bits(pooled[c]),
                $"{type} K={k} N={n} col {c}: one thread {serial[c]:R} vs the pool {pooled[c]:R}");
            Assert.True(BitConverter.SingleToInt32Bits(serial[c]) == BitConverter.SingleToInt32Bits(narrow[c]),
                $"{type} K={k} N={n} col {c}: one thread {serial[c]:R} vs two threads {narrow[c]:R}");
        }
    }

    /// <summary>Multi-row products as the pool runs them (ARM64: <see cref="ManagedQuantizedOps.GuidedBounds"/>' shrinking
    /// column blocks, at the pool's full width and narrowed to two threads) against the same product on one thread, bit for
    /// bit: few-row batches, prompt chunks, and a chunk past one row block (600 rows of K = 2048).</summary>
    [ArmDotTheory]
    [InlineData(GgmlTensorType.Q4_K, 128, 2048, 1030)]
    [InlineData(GgmlTensorType.Q4_K, 600, 2048, 516)]
    [InlineData(GgmlTensorType.Q6_K, 37, 7680, 260)]
    [InlineData(GgmlTensorType.Q8_0, 3, 2048, 3072)]
    [InlineData(GgmlTensorType.Q4_0, 128, 2048, 2050)]
    [InlineData(GgmlTensorType.NVFP4, 4, 2048, 1030)]
    [InlineData(GgmlTensorType.BF16, 1, 2048, 3074)]
    [InlineData(GgmlTensorType.BF16, 70, 2048, 1030)]
    public unsafe void MultiRowProducts_AreBitIdenticalAtAnyPoolWidth(GgmlTensorType type, int rows, int k, int n)
    {
        var rng = new Random(11 + rows + k + n + (int)type);
        byte[] weights;
        if (type == GgmlTensorType.BF16)
        {
            weights = new byte[2L * n * k];
            for (long i = 0; i < weights.Length; i += 2)
                BinaryPrimitives.WriteUInt16LittleEndian(weights.AsSpan((int)i),
                    (ushort)(BitConverter.SingleToUInt32Bits((float)(rng.NextDouble() - 0.5) * 0.2f) >> 16));
        }
        else
        {
            weights = RandomWeights(rng, type, n, k);
        }
        float[] x = new float[(long)rows * k];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2 - 1) * (i % 61 == 0 ? 30f : 1f);
        float[] serial = new float[(long)rows * n], pooled = new float[(long)rows * n], narrow = new float[(long)rows * n];
        CpuWorkerPool pool = CpuWorkerPool.Shared;
        fixed (byte* w = weights)
        fixed (float* xp = x)
        fixed (float* s = serial)
        fixed (float* p = pooled)
        fixed (float* q = narrow)
        {
            ManagedQuantizedOps.AddmmQuantizedToFloat32((int)type, (IntPtr)w, k, n, xp, k, rows, s, n,
                new System.Threading.Tasks.ParallelOptions { MaxDegreeOfParallelism = 1 });
            ManagedQuantizedOps.AddmmQuantizedToFloat32((int)type, (IntPtr)w, k, n, xp, k, rows, p, n);
            int width = pool.ActiveThreads;
            try
            {
                pool.SetActiveThreads(2);
                ManagedQuantizedOps.AddmmQuantizedToFloat32((int)type, (IntPtr)w, k, n, xp, k, rows, q, n);
            }
            finally
            {
                pool.SetActiveThreads(width);
            }
        }
        for (long i = 0; i < serial.Length; i++)
        {
            Assert.True(BitConverter.SingleToInt32Bits(serial[i]) == BitConverter.SingleToInt32Bits(pooled[i]),
                $"{type} rows={rows} K={k} N={n} [{i / n},{i % n}]: one thread {serial[i]:R} vs the pool {pooled[i]:R}");
            Assert.True(BitConverter.SingleToInt32Bits(serial[i]) == BitConverter.SingleToInt32Bits(narrow[i]),
                $"{type} rows={rows} K={k} N={n} [{i / n},{i % n}]: one thread {serial[i]:R} vs two threads {narrow[i]:R}");
        }
    }

    /// <summary>Guided blocks cover the units once, in order, none empty, none below the floor but the last, never
    /// growing.</summary>
    [Theory]
    [InlineData(3840, 1, 18)]
    [InlineData(3840, 1, 16)]
    [InlineData(258, 4, 18)]
    [InlineData(7, 3, 2)]
    [InlineData(1, 1, 18)]
    [InlineData(100, 200, 4)]
    public void GuidedBounds_CoverEveryUnitOnceLargestFirst(int units, int minUnits, int threads)
    {
        int[] b = ManagedQuantizedOps.GuidedBounds(units, minUnits, threads);
        Assert.Equal(0, b[0]);
        Assert.Equal(units, b[^1]);
        for (int i = 1; i < b.Length; i++)
        {
            Assert.True(b[i] > b[i - 1], $"block {i - 1} is empty");
            if (i < b.Length - 1) Assert.True(b[i] - b[i - 1] >= minUnits, $"block {i - 1}: {b[i] - b[i - 1]} units");
            if (i >= 2) Assert.True(b[i] - b[i - 1] <= b[i - 1] - b[i - 2], $"block {i - 1} is larger than the one before it");
        }
        // the first block is a 1 / (2 x threads) share (or the floor)
        Assert.Equal(Math.Min(units, Math.Max(minUnits, (units + 2 * threads - 1) / (2 * threads))), b[1]);
    }

    /// <summary>The decode blocks cover the columns once, in order, every inner boundary on a four-column group, each
    /// block at least 16 KB of weights (but the last), the large ones first.</summary>
    [Theory]
    [InlineData(2048, 1152, 16)]
    [InlineData(15360, 1152, 16)]
    [InlineData(2048, 6300, 18)]
    [InlineData(128000, 1680, 16)]
    [InlineData(37, 1152, 16)]
    [InlineData(3, 64, 4)]
    [InlineData(4099, 1088, 1)]
    public void DecodeColumnBounds_CoverEveryColumnOnceLargestFirst(int outDim, int rowBytes, int threads)
    {
        int[] b = ManagedQuantizedOps.DecodeColumnBounds(outDim, rowBytes, threads);
        Assert.Equal(0, b[0]);
        Assert.Equal(outDim, b[^1]);
        int minColumns = Math.Max(4, (16 * 1024 + rowBytes - 1) / rowBytes);
        for (int i = 1; i < b.Length; i++)
        {
            Assert.True(b[i] > b[i - 1], $"block {i - 1} is empty");
            if (i < b.Length - 1)
            {
                Assert.Equal(0, b[i] % 4);
                Assert.True(b[i] - b[i - 1] >= minColumns, $"block {i - 1}: {b[i] - b[i - 1]} columns");
            }
            if (i >= 2) Assert.True(b[i] - b[i - 1] <= b[i - 1] - b[i - 2], $"block {i - 1} is larger than the one before it");
        }
        Assert.Same(b, ManagedQuantizedOps.DecodeColumnBounds(outDim, rowBytes, threads));
    }

    /// <summary>BF16 products on ARM64 against DequantMatMulColumns (what they replace), bit for bit: row groups of
    /// four, leftover rows, K with and without 4- and 8-value tails, column counts that leave a partial group.</summary>
    [ArmDotTheory]
    [InlineData(1, 2048, 37)]
    [InlineData(3, 64, 5)]
    [InlineData(4, 70, 4)]
    [InlineData(5, 2048, 130)]
    [InlineData(9, 13, 3)]
    [InlineData(17, 516, 129)]
    [InlineData(31, 2048, 70)]
    [InlineData(64, 2048, 256)]
    [InlineData(70, 7680, 130)]
    public unsafe void Bf16Products_AreBitIdenticalToDequantMatMulColumns(int rows, int k, int n)
    {
        var rng = new Random(rows * 1000 + k + n);
        ushort[] w = new ushort[(long)n * k];
        for (int i = 0; i < w.Length; i++)
            w[i] = (ushort)(BitConverter.SingleToUInt32Bits((float)(rng.NextDouble() - 0.5) * 0.2f) >> 16);
        int inStride = k + 3, outStride = n + 2;
        float[] x = new float[rows * inStride];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);
        float[] expected = new float[rows * outStride], actual = new float[rows * outStride];
        float[] scratch = new float[k];
        fixed (ushort* wp = w)
        fixed (float* xp = x)
        fixed (float* e = expected)
        fixed (float* a = actual)
        fixed (float* s = scratch)
        {
            ManagedQuantizedOps.DequantMatMulColumns((int)GgmlTensorType.BF16, (byte*)wp, 2L * k, k, n, xp, inStride, rows,
                e, outStride, 0, n, s);
            ManagedQuantizedOps.AddmmQuantizedToFloat32((int)GgmlTensorType.BF16, (IntPtr)wp, k, n, xp, inStride, rows,
                a, outStride, null, ManagedQuantizedOps.QGemmIsa.Auto);
        }
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < n; c++)
                Assert.True(BitConverter.SingleToInt32Bits(expected[r * outStride + c]) == BitConverter.SingleToInt32Bits(actual[r * outStride + c]),
                    $"BF16 rows={rows} K={k} N={n} [{r},{c}]: {expected[r * outStride + c]:R} vs {actual[r * outStride + c]:R}");
    }

    /// <summary>The four-row decode matvec (Q4_K, Q6_K, Q4_0, Q8_0, NVFP4) against the per-row dot, lane by lane, bit for bit.</summary>
    [ArmDotTheory]
    [InlineData(GgmlTensorType.Q4_K, 256)]
    [InlineData(GgmlTensorType.Q4_K, 2048)]
    [InlineData(GgmlTensorType.Q4_K, 7680)]
    [InlineData(GgmlTensorType.Q6_K, 256)]
    [InlineData(GgmlTensorType.Q6_K, 2048)]
    [InlineData(GgmlTensorType.Q6_K, 7680)]
    [InlineData(GgmlTensorType.Q4_0, 96)]
    [InlineData(GgmlTensorType.Q4_0, 2048)]
    [InlineData(GgmlTensorType.Q8_0, 160)]
    [InlineData(GgmlTensorType.Q8_0, 2048)]
    [InlineData(GgmlTensorType.NVFP4, 64)]
    [InlineData(GgmlTensorType.NVFP4, 2048)]
    public unsafe void FourRowMatvec_IsBitIdenticalToThePerRowDot(GgmlTensorType type, int k)
    {
        var rng = new Random(500 + (int)type + k);
        Assert.True(ManagedQuantizedOps.HasDot4Rows(type));
        Assert.True(ManagedQuantizedOps.TryGetActivationPlan(type, k, out int actBytes));
        byte[] weights = RandomWeights(rng, type, 12, k);
        int rowBytes = weights.Length / 12;
        float[] x = new float[k];
        byte[] act = new byte[actBytes];
        for (int trial = 0; trial < 3; trial++)
        {
            for (int i = 0; i < k; i++) x[i] = (float)(rng.NextDouble() * 2 - 1) * (trial == 1 ? 1e-3f : 3f);
            fixed (float* xp = x)
            fixed (byte* ap = act)
            fixed (byte* wp = weights)
            {
                ManagedQuantizedOps.QuantizeActivationRow(type, xp, ap, k);
                byte* w = wp + (long)trial * 4 * rowBytes;
                var four = ManagedQuantizedOps.DotQuantized4Rows(type, w, w + rowBytes, w + 2 * rowBytes, w + 3 * rowBytes, ap, k);
                for (int lane = 0; lane < 4; lane++)
                {
                    float expected = ManagedQuantizedOps.DotQuantizedRow(type, w + lane * rowBytes, ap, k);
                    Assert.True(BitConverter.SingleToInt32Bits(expected) == BitConverter.SingleToInt32Bits(four.GetElement(lane)),
                        $"{type} K={k} trial {trial} lane {lane}: per-row {expected:R} vs four-row {four.GetElement(lane):R}");
                }
            }
        }
    }

    [ArmDotFact]
    public unsafe void ActivationQuantizer_IsBitIdenticalToTheScalarLoop()
    {
        var rng = new Random(7);
        const int n = 4096;
        float[] x = new float[n];
        for (int i = 0; i < n; i++)
            x[i] = (float)(rng.NextDouble() * 2 - 1) * 200f;
        // Ties at every half-integer, the clamp edges, signed zeros, NaN and the infinities.
        float[] specials = { 0.5f, 1.5f, 2.5f, -0.5f, -1.5f, -2.5f, 126.5f, 127.5f, -127.5f, 128f, -128f, 1e30f, -1e30f,
            0f, -0f, -0.3f, float.NaN, float.PositiveInfinity, float.NegativeInfinity, 63.5f, -63.5f };
        Array.Copy(specials, x, specials.Length);
        sbyte[] qs = new sbyte[n], qn = new sbyte[n];
        int[] gs = new int[n / 16], gn = new int[n / 16];
        foreach (float inv in new[] { 1f, 0.75f, 1.31f, float.PositiveInfinity })
        {
            fixed (float* xp = x)
            fixed (sbyte* a = qs)
            fixed (sbyte* b = qn)
            fixed (int* ga = gs)
            fixed (int* gb = gn)
            {
                ManagedQuantizedOps.QuantizeInt8Groups16Scalar(xp, inv, a, n, ga);
                ManagedQuantizedOps.QuantizeInt8Groups16Neon(xp, inv, b, n, gb);
            }
            Assert.Equal(qs, qn);
            Assert.Equal(gs, gn);
        }
    }

    /// <summary>Random blocks with every fp16 scale finite but spread over normal and subnormal magnitudes and both
    /// signs (the vector fp16 conversion is checked through them).</summary>
    private static byte[] RandomWeights(Random rng, GgmlTensorType type, int rows, int k)
    {
        int blockBytes = (int)GgufFile.GetTypeSize(type);
        int blockSize = (int)GgufFile.GetBlockSize(type);
        byte[] raw = new byte[(long)rows * (k / blockSize) * blockBytes];
        rng.NextBytes(raw);
        for (long o = 0; o < raw.Length; o += blockBytes)
        {
            switch (type)
            {
                case GgmlTensorType.Q4_0:
                case GgmlTensorType.Q8_0:
                    WriteHalf(raw, o, RandomScale(rng));
                    break;
                case GgmlTensorType.Q4_K:
                    WriteHalf(raw, o, RandomScale(rng));
                    WriteHalf(raw, o + 2, RandomScale(rng));
                    break;
                case GgmlTensorType.Q6_K:
                    WriteHalf(raw, o + blockBytes - 2, RandomScale(rng));
                    break;
                case GgmlTensorType.NVFP4:
                    break;   // UE4M3 sub-scales: every byte value decodes (0x7F to zero)
                default:
                    throw new NotSupportedException(type.ToString());
            }
        }
        return raw;
    }

    private static float RandomScale(Random rng)
    {
        double r = rng.NextDouble();
        float sign = rng.Next(2) == 0 ? 1f : -1f;
        if (r < 0.05) return 0f;
        if (r < 0.15) return sign * (float)(rng.NextDouble() * 6e-5);   // fp16 subnormals
        return sign * (float)(1e-4 + rng.NextDouble() * 0.05);
    }

    private static void WriteHalf(byte[] buffer, long offset, float value)
        => BinaryPrimitives.WriteUInt16LittleEndian(buffer.AsSpan((int)offset), BitConverter.HalfToUInt16Bits((Half)value));
}

/// <summary>Theory that needs the ARM64 dot-product kernels.</summary>
public sealed class ArmDotTheoryAttribute : TheoryAttribute
{
    public ArmDotTheoryAttribute()
    {
        if (!ManagedQuantizedOps.ArmDotSupported)
            Skip = "Requires ARM64 AdvSimd with the DotProd extension.";
    }
}

/// <summary>Fact that needs the ARM64 dot-product kernels.</summary>
public sealed class ArmDotFactAttribute : FactAttribute
{
    public ArmDotFactAttribute()
    {
        if (!ManagedQuantizedOps.ArmDotSupported)
            Skip = "Requires ARM64 AdvSimd with the DotProd extension.";
    }
}
