// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Reflection;
using System.Runtime.InteropServices;
using TensorSharp;
using TensorSharp.Cuda;
using TensorSharp.Cuda.Interop;
using TensorSharp.Models;
using TensorSharp.Models.Direct;
using Xunit.Abstractions;
using Xunit.Sdk;

namespace InferenceWeb.Tests;

/// <summary>[Theory] that needs TF32 tensor cores on GPU 0 (compute capability 8.0+). Before
/// Ampere an explicit TF32 compute type computes F32, so there is no approximation to check.</summary>
[TraitDiscoverer("InferenceWeb.Tests.RequiresTraitDiscoverer", "InferenceWeb.Tests")]
[AttributeUsage(AttributeTargets.Method)]
internal sealed class CudaTf32TheoryAttribute : TheoryAttribute, ITraitAttribute
{
    public string RequiresValue => "Cuda";

    public CudaTf32TheoryAttribute()
    {
        Skip = TestGates.CudaSkip;
        if (Skip != null) return;
        try
        {
            if (CudaDevice.GetDevice(0).ComputeCapabilityMajor < 8)
                Skip = "Requires TF32 tensor cores (compute capability 8.0 or newer).";
        }
        catch (Exception exception)
        {
            Skip = $"Cannot query the CUDA TF32 capability: {exception.Message}";
        }
    }
}

/// <summary>[Theory] that needs half-precision tensor cores on GPU 0 (compute capability 7.0 or
/// newer); without them cuBLAS keeps F32 operands even when a call allows half.</summary>
[TraitDiscoverer("InferenceWeb.Tests.RequiresTraitDiscoverer", "InferenceWeb.Tests")]
[AttributeUsage(AttributeTargets.Method)]
internal sealed class CudaTensorCoreTheoryAttribute : TheoryAttribute, ITraitAttribute
{
    public string RequiresValue => "Cuda";

    public CudaTensorCoreTheoryAttribute()
    {
        Skip = TestGates.CudaSkip;
        if (Skip != null) return;
        try
        {
            if (CudaDevice.GetDevice(0).ComputeCapabilityMajor < 7)
                Skip = "Requires half-precision tensor cores (compute capability 7.0 or newer).";
        }
        catch (Exception exception)
        {
            Skip = $"Cannot query the CUDA compute capability: {exception.Message}";
        }
    }
}

/// <summary>
/// The direct CUDA backend computes every F32-operand GEMM that is not an explicit, documented
/// approximation with exact F32 operands, at every row count, and its explicit approximations (TF32,
/// and half operands named through <see cref="CudaHalfOperandGemm"/>) stay what they say.
/// <para>
/// The shared cuBLAS handle used to be in <c>CUBLAS_TENSOR_OP_MATH</c>, which cuBLAS 13 defines as
/// <c>CUBLAS_COMPUTE_32F_FAST_16F</c>. Sgemm, SgemmStridedBatched and GemmEx with
/// <c>CUBLAS_COMPUTE_32F</c> then rounded both operands to IEEE half on every shape cuBLAS sent to
/// tensor cores, from 2 to 17 rows depending on the shape (see <see cref="CudaCublasHandle.Create"/>).
/// Every exactness check below fails under that mode, and under <c>CUBLAS_TF32_TENSOR_OP_MATH</c>.
/// </para>
/// <para>
/// Method: operands are uniform in [-1, 1) with the low 13 mantissa bits random and nonzero (the
/// bits TF32 and half drop), so any operand rounding shows. Sampled outputs are compared with a
/// float64 product of the same F32 operands, as |c - ref| / sum|a*w|, which is scale-free. Each
/// bound is set against the error the same data has when both operands are rounded to TF32 (round
/// to nearest, ties away from zero, as cuBLAS rounds). Exact F32 sits two to three orders of
/// magnitude below that. "Range" cases scale the weight by 2^17, past half's 65504: a product of
/// half operands is inf there, one of TF32 or F32 operands is finite. The half-operand checks use
/// the same scale with half rounding (round to nearest even) in place of TF32's.
/// </para>
/// </summary>
public sealed class CudaF32GemmPrecisionTests : IDisposable
{
    private const float RangeScale = 131072f; // 2^17: exact, and past half's largest finite value
    private const int CublasDefaultMath = 0;

    private readonly ITestOutputHelper _output;
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-f32-gemm-" + Guid.NewGuid().ToString("N"));
    private readonly EnvScope _env = new EnvScope();

    public CudaF32GemmPrecisionTests(ITestOutputHelper output)
    {
        _output = output;
        Directory.CreateDirectory(_dir);
    }

    public void Dispose()
    {
        _env.Dispose();
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    // ---------------------------------------------------------------- the shared handle

    [CudaFact]
    public void TheSharedHandle_IsInTheDefaultMathMode()
    {
        using var allocator = new CudaAllocator();
        Assert.Equal(CublasDefaultMath, allocator.Blas.GetMathMode());
    }

    // ---------------------------------------------------------------- F32 GEMM entry points

    /// <summary>Ops.Addmm (cublasSgemm): ModelBase's F32 routers, LoRA deltas, vision encoders. The
    /// router layout is the weight's transposed view; the contiguous right operand is the other
    /// layout the entry point takes.</summary>
    [CudaTheory]
    [InlineData(8, 2816, 128, true, false)]    // Gemma 4 router, 8 batched rows
    [InlineData(17, 2048, 256, true, false)]   // Qwen3.5 router
    [InlineData(512, 2880, 32, true, false)]   // gpt-oss router, a prefill chunk
    [InlineData(512, 4096, 256, true, true)]   // range
    [InlineData(64, 1024, 1024, false, false)] // contiguous right operand
    public void Addmm_ComputesF32OperandsExactly(int rows, int inDim, int outDim, bool transposedWeight, bool range)
    {
        using var allocator = new CudaAllocator();
        float[] a = Operands(rows * inDim, seed: 11, scale: 1f);
        float[] w = Operands(outDim * inDim, seed: 12, scale: range ? RangeScale : 1f);
        using var input = Upload(allocator, a, rows, inDim);
        using var weight = transposedWeight
            ? Upload(allocator, w, outDim, inDim)
            : Upload(allocator, Transpose(w, outDim, inDim), inDim, outDim);
        using var right = transposedWeight ? weight.Transpose() : weight.CopyRef();
        using var result = new Tensor(allocator, DType.Float32, rows, outDim);

        Assert.True(CudaBlas.TryAddmm(result, 0, result, 1, input, right), "cuBLAS declined the operands.");
        float[] direct = result.GetElementsAsFloat(rows * outDim);
        AssertExactF32(Measure(direct, a, w, 1, rows, inDim, outDim),
            $"Ops.Addmm {rows} x {inDim} -> {outDim}{(transposedWeight ? "" : ", contiguous right operand")}{(range ? ", range" : "")}");

        // The op the models call takes that GEMM.
        using var viaOps = new Tensor(allocator, DType.Float32, rows, outDim);
        Ops.Addmm(viaOps, 0, viaOps, 1, input, right);
        Assert.Equal(direct, viaOps.GetElementsAsFloat(rows * outDim));
    }

    /// <summary>Ops.AddmmBatch (cublasSgemmStridedBatched): prefill attention scores and values,
    /// vision attention. Under the old mode it rounded to half from 17 rows.</summary>
    [CudaTheory]
    [InlineData(2, 17, 4096, 256, false)]
    [InlineData(32, 64, 128, 512, false)]  // attention scores: 32 heads x 128
    [InlineData(8, 512, 128, 512, true)]   // range
    public void AddmmBatch_ComputesF32OperandsExactly(int batch, int rows, int shared, int cols, bool range)
    {
        using var allocator = new CudaAllocator();
        float[] a = Operands(batch * rows * shared, seed: 21, scale: 1f);
        float[] w = Operands(batch * cols * shared, seed: 22, scale: range ? RangeScale : 1f);
        using var lhs = Upload(allocator, a, batch, rows, shared);
        using var rhs = Upload(allocator, w, batch, cols, shared);
        using var rhsT = rhs.Transpose(1, 2);
        using var result = new Tensor(allocator, DType.Float32, batch, rows, cols);

        Assert.True(CudaBlas.TryAddmmBatch(result, 0, result, 1, lhs, rhsT), "cuBLAS declined the operands.");
        float[] direct = result.GetElementsAsFloat(batch * rows * cols);
        AssertExactF32(Measure(direct, a, w, batch, rows, shared, cols),
            $"Ops.AddmmBatch {batch} x {rows} x {shared} -> {cols}{(range ? ", range" : "")}");

        using var viaOps = new Tensor(allocator, DType.Float32, batch, rows, cols);
        Ops.AddmmBatch(viaOps, 0, viaOps, 1, lhs, rhsT);
        Assert.Equal(direct, viaOps.GetElementsAsFloat(batch * rows * cols));
    }

    /// <summary>CudaQuantizedOps.AddmmResidentToFloat32 with an F32 weight: the route of every direct
    /// engine's MatMulF32 past its 16 GEMV rows (routers, hyper-connection mixers, GDN gates,
    /// indexer projections).</summary>
    [CudaTheory]
    [InlineData(17, 2560, 512, false)]  // Qwen3.8 router
    [InlineData(512, 2560, 48, false)]  // Qwen3.8 GDN alpha/beta, a prefill chunk
    [InlineData(64, 16384, 24, false)]  // DeepSeek V4 hyper-connection mixer
    [InlineData(512, 4096, 256, true)]  // DeepSeek V4 router, a prefill chunk; range
    public void ResidentF32Weight_ComputesExactly(int rows, int inDim, int outDim, bool range)
    {
        using var allocator = new CudaAllocator();
        float[] a = Operands(rows * inDim, seed: 31, scale: 1f);
        float[] w = Operands(outDim * inDim, seed: 32, scale: range ? RangeScale : 1f);
        using var input = Upload(allocator, a, rows, inDim);
        using var weight = Upload(allocator, w, outDim, inDim);
        using var result = new Tensor(allocator, DType.Float32, rows, outDim);
        var storage = (CudaStorage)weight.Storage;
        storage.EnsureDeviceCurrent();

        CudaQuantizedOps.AddmmResidentToFloat32(result, input, storage.DevicePtrAtElement(0), 0, inDim, outDim);
        AssertExactF32(Measure(result.GetElementsAsFloat(rows * outDim), a, w, 1, rows, inDim, outDim),
            $"AddmmResidentToFloat32 F32 {rows} x {inDim} -> {outDim}{(range ? ", range" : "")}");
    }

    /// <summary>The engines' F32 projections name their compute type, so they stay exact even on a
    /// handle in <c>CUBLAS_TENSOR_OP_MATH</c> (1) or <c>CUBLAS_TF32_TENSOR_OP_MATH</c> (3). This
    /// allocator's own handle is switched; every allocator has its own.</summary>
    [CudaTheory]
    [InlineData(1)]
    [InlineData(3)]
    public void ResidentF32Weight_ComputesExactly_WhateverTheHandleMathMode(int mathMode)
    {
        using var allocator = new CudaAllocator();
        CublasApi.cublasSetMathMode(allocator.Blas.Handle, mathMode).ThrowOnCublasError();
        Assert.Equal(mathMode, allocator.Blas.GetMathMode());
        foreach (var (rows, inDim, outDim, range) in new[] { (17, 2560, 512, false), (512, 4096, 256, true) })
        {
            float[] a = Operands(rows * inDim, seed: 33, scale: 1f);
            float[] w = Operands(outDim * inDim, seed: 34, scale: range ? RangeScale : 1f);
            using var input = Upload(allocator, a, rows, inDim);
            using var weight = Upload(allocator, w, outDim, inDim);
            using var result = new Tensor(allocator, DType.Float32, rows, outDim);
            var storage = (CudaStorage)weight.Storage;
            storage.EnsureDeviceCurrent();
            CudaQuantizedOps.AddmmResidentToFloat32(result, input, storage.DevicePtrAtElement(0), 0, inDim, outDim);
            AssertExactF32(Measure(result.GetElementsAsFloat(rows * outDim), a, w, 1, rows, inDim, outDim),
                $"AddmmResidentToFloat32 F32 {rows} x {inDim} -> {outDim}{(range ? ", range" : "")}, handle math mode {mathMode}");
        }
    }

    /// <summary>CudaQuantizedOps.TryAddmmQuantizedToFloat32 with an F32 weight: ModelBase's F32
    /// QuantizedWeight and the direct ops' F32 tensors, uploaded through the weight cache.</summary>
    [CudaTheory]
    [InlineData(64, 2048, 256, false)]
    [InlineData(512, 2816, 128, true)]
    public void CachedF32Weight_ComputesExactly(int rows, int inDim, int outDim, bool range)
    {
        using var allocator = new CudaAllocator();
        float[] a = Operands(rows * inDim, seed: 41, scale: 1f);
        float[] w = Operands(outDim * inDim, seed: 42, scale: range ? RangeScale : 1f);
        using var input = Upload(allocator, a, rows, inDim);
        using var result = new Tensor(allocator, DType.Float32, rows, outDim);
        long bytes = (long)w.Length * sizeof(float);
        IntPtr host = Marshal.AllocHGlobal((IntPtr)bytes);
        try
        {
            Marshal.Copy(w, 0, host, w.Length);
            Assert.True(CudaQuantizedOps.TryAddmmQuantizedToFloat32(result, input, host, host, 0, inDim, outDim, bytes));
            AssertExactF32(Measure(result.GetElementsAsFloat(rows * outDim), a, w, 1, rows, inDim, outDim),
                $"TryAddmmQuantizedToFloat32 F32 {rows} x {inDim} -> {outDim}{(range ? ", range" : "")}");
        }
        finally
        {
            CudaQuantizedOps.ReleaseQuantizedWeight(allocator, host);
            Marshal.FreeHGlobal(host);
        }
    }

    /// <summary>The call sites that name <c>CUBLAS_COMPUTE_32F_PEDANTIC</c> themselves: Gemma 4's audio
    /// projection (weight [in, out]), MiniMax-H3's TryMatmul and LTX 2.5's LoRA linear (weight
    /// [out, in], the latter uploaded through the weight cache).</summary>
    [CudaTheory]
    [InlineData("gemma4-audio", false)]
    [InlineData("gemma4-audio", true)]
    [InlineData("minimax-h3-matmul", false)]
    [InlineData("minimax-h3-matmul", true)]
    [InlineData("ltx25-lora", false)]
    [InlineData("ltx25-lora", true)]
    public void ExplicitPedanticSites_ComputeExactly(string site, bool range)
    {
        const int rows = 512, inDim = 4096, outDim = 256;
        using var allocator = new CudaAllocator();
        float[] a = Operands(rows * inDim, seed: 43, scale: 1f);
        float[] w = Operands(outDim * inDim, seed: 44, scale: range ? RangeScale : 1f);
        using var input = Upload(allocator, a, rows, inDim);
        using var result = new Tensor(allocator, DType.Float32, rows, outDim);
        float[] c;
        switch (site)
        {
            case "gemma4-audio":
            {
                using var weight = Upload(allocator, Transpose(w, outDim, inDim), inDim, outDim);
                Assert.True(CudaGemma4AudioOps.TryProjectionFloat32(result, input, weight));
                c = result.GetElementsAsFloat(rows * outDim);
                break;
            }
            case "minimax-h3-matmul":
            {
                using var weight = Upload(allocator, w, outDim, inDim);
                Assert.True(CudaMiniMaxH3Ops.TryMatmul(result, input, weight));
                c = result.GetElementsAsFloat(rows * outDim);
                break;
            }
            default:
            {
                long bytes = (long)w.Length * sizeof(float);
                IntPtr host = Marshal.AllocHGlobal((IntPtr)bytes);
                try
                {
                    Marshal.Copy(w, 0, host, w.Length);
                    Assert.True(CudaLtx25LoraOps.TryLinear(result, input, host, inDim, outDim, bytes));
                    c = result.GetElementsAsFloat(rows * outDim);
                }
                finally
                {
                    CudaQuantizedOps.ReleaseQuantizedWeight(allocator, host);
                    Marshal.FreeHGlobal(host);
                }
                break;
            }
        }
        AssertExactF32(Measure(c, a, w, 1, rows, inDim, outDim), $"{site} {rows} x {inDim} -> {outDim}{(range ? ", range" : "")}");
    }

    // ---------------------------------------------------------------- explicit half operands

    /// <summary>CudaHalfOperandGemm names <c>CUBLAS_COMPUTE_32F_FAST_16F</c>: on a tensor-core shape both
    /// operands are rounded to half, the old handle's behaviour, kept by name for the bulk attention,
    /// vision-encoder and VAE GEMMs. It leaves the handle in the default mode, so the plain entry point
    /// on the same data right after it is exact.</summary>
    [CudaTensorCoreTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void HalfOperandGemm_RoundsToHalf_AndLeavesTheOtherGemmsExact(bool batched)
    {
        const int rows = 512, inDim = 4096, outDim = 256;
        int batch = batched ? 2 : 1;
        using var allocator = new CudaAllocator();
        float[] a = Operands(batch * rows * inDim, seed: 101, scale: 1f);
        float[] w = Operands(batch * outDim * inDim, seed: 102, scale: 1f);
        using var input = batched ? Upload(allocator, a, batch, rows, inDim) : Upload(allocator, a, rows, inDim);
        using var weight = batched ? Upload(allocator, w, batch, outDim, inDim) : Upload(allocator, w, outDim, inDim);
        using var right = batched ? weight.Transpose(1, 2) : weight.Transpose();
        using var result = batched
            ? new Tensor(allocator, DType.Float32, batch, rows, outDim)
            : new Tensor(allocator, DType.Float32, rows, outDim);
        int count = batch * rows * outDim;
        string shape = $"{batch} x {rows} x {inDim} -> {outDim}";

        Assert.True(batched
            ? CudaHalfOperandGemm.TryAddmmBatch(result, 0, result, 1, input, right)
            : CudaHalfOperandGemm.TryAddmm(result, 0, result, 1, input, right), "cuBLAS declined the operands.");
        AssertHalf(Measure(result.GetElementsAsFloat(count), a, w, batch, rows, inDim, outDim, Half16),
            $"CudaHalfOperandGemm.{(batched ? "TryAddmmBatch" : "TryAddmm")} {shape}");
        Assert.Equal(CublasDefaultMath, allocator.Blas.GetMathMode());

        Assert.True(batched
            ? CudaBlas.TryAddmmBatch(result, 0, result, 1, input, right)
            : CudaBlas.TryAddmm(result, 0, result, 1, input, right), "cuBLAS declined the operands.");
        AssertExactF32(Measure(result.GetElementsAsFloat(count), a, w, batch, rows, inDim, outDim),
            $"CudaBlas.{(batched ? "TryAddmmBatch" : "TryAddmm")} {shape} after the half-operand call");
    }

    /// <summary>DirectOps.Attention (Wan and MiniMax-H3 DiT and VAE attention): past 100 GFLOP it takes
    /// the chunked cuBLAS route, whose two GEMMs name half operands (scores, softmax and output stay
    /// F32, like ggml's F16 flash attention); below that the exact streaming F32 kernel runs. Checked
    /// on sampled (token, head) rows against float64 attention with and without half-rounded
    /// operands (Q and K for the scores, the normalized probabilities and V for the output). The
    /// half-route bound is looser than the GEMM checks' (an eighth of the half gap, not a twentieth):
    /// the F32 softmax and a 4096-term F32 sum sit between the GPU and the float64 reference too.</summary>
    [CudaTensorCoreTheory]
    [InlineData(4096, true)]   // 12 heads x 128: 103 GFLOP, the GEMM route
    [InlineData(1024, false)]  // 6.4 GFLOP: the streaming kernel
    public void DirectAttention_TakesHalfOperandsOnlyOnTheLongSequenceRoute(int tokens, bool halfRoute)
    {
        const int heads = 12, dims = 128, sampled = 24;
        int width = heads * dims;
        using var allocator = new CudaAllocator();
        using var ctx = new DirectContext(allocator);
        float[] q = Operands(tokens * width, seed: 111, scale: 1f);
        float[] k = Operands(tokens * width, seed: 112, scale: 1f);
        float[] v = Operands(tokens * width, seed: 113, scale: 1f);
        using var qt = Upload(allocator, q, tokens, width);
        using var kt = Upload(allocator, k, tokens, width);
        using var vt = Upload(allocator, v, tokens, width);
        float scale = 1f / MathF.Sqrt(dims);
        using var y = DirectOps.Attention(ctx, qt, kt, vt, heads, dims, scale);
        float[] got = y.GetElementsAsFloat(tokens * width);

        double error = 0, fromHalf = 0, halfGap = 0;
        int nonFinite = got.Count(value => !float.IsFinite(value)), counted = 0;
        var exactScores = new double[tokens];
        var halfScores = new double[tokens];
        for (int s = 0; s < sampled; s++)
        {
            int t = (int)((s * 1_000_003L + 7) % tokens), h = s % heads;
            for (int j = 0; j < tokens; j++)
            {
                double exactDot = 0, halfDot = 0;
                for (int d = 0; d < dims; d++)
                {
                    float qv = q[t * width + h * dims + d], kv = k[j * width + h * dims + d];
                    exactDot += (double)qv * kv;
                    halfDot += (double)Half16(qv) * Half16(kv);
                }
                exactScores[j] = exactDot * scale;
                halfScores[j] = halfDot * scale;
            }
            Softmax(exactScores);
            Softmax(halfScores);
            for (int d = 0; d < dims; d++)
            {
                double exact = 0, rounded = 0, magnitude = 0;
                for (int j = 0; j < tokens; j++)
                {
                    float vv = v[j * width + h * dims + d];
                    exact += exactScores[j] * vv;
                    magnitude += Math.Abs(exactScores[j] * vv);
                    rounded += (double)Half16((float)halfScores[j]) * Half16(vv);
                }
                float value = got[t * width + h * dims + d];
                error += Math.Abs(value - exact) / magnitude;
                fromHalf += Math.Abs(value - rounded) / magnitude;
                halfGap += Math.Abs(rounded - exact) / magnitude;
                counted++;
            }
        }
        error /= counted; fromHalf /= counted; halfGap /= counted;
        string what = $"DirectOps.Attention {tokens} tokens x {heads} heads x {dims}: error {error:E2} of sum|p*v|, " +
            $"half-rounded operands would give {halfGap:E2}, distance to that {fromHalf:E2}, {nonFinite} non-finite outputs";
        _output.WriteLine(what);
        Assert.True(nonFinite == 0, what);
        if (halfRoute)
        {
            Assert.True(error >= halfGap / 3, $"The long-sequence route computed exact F32; its named half operands are gone. {what}");
            Assert.True(fromHalf <= halfGap / 8, $"The long-sequence route is not the half-operand attention. {what}");
        }
        else
        {
            Assert.True(error <= halfGap / 20, $"The short route's operands were not kept F32. {what}");
        }
    }

    private static void Softmax(double[] x)
    {
        double max = x.Max(), sum = 0;
        for (int i = 0; i < x.Length; i++)
            sum += x[i] = Math.Exp(x[i] - max);
        for (int i = 0; i < x.Length; i++)
            x[i] /= sum;
    }

    // ---------------------------------------------------------------- explicit TF32 policies

    /// <summary>CudaEmbeddingOps.TryLinearTf32 is the embedding encoders' explicit TF32 projection;
    /// its pedantic twin on the same data is exact.</summary>
    [CudaTf32Theory]
    [InlineData(64, 4096, 256, false)]
    [InlineData(64, 4096, 256, true)] // TF32 keeps F32's exponent range
    public void EmbeddingLinearTf32_StaysTf32(int rows, int inDim, int outDim, bool range)
    {
        Assert.Null(Environment.GetEnvironmentVariable("NVIDIA_TF32_OVERRIDE"));
        using var allocator = new CudaAllocator();
        float[] a = Operands(rows * inDim, seed: 51, scale: 1f);
        float[] w = Operands(outDim * inDim, seed: 52, scale: range ? RangeScale : 1f);
        using var input = Upload(allocator, a, rows, inDim);
        using var weight = Upload(allocator, Transpose(w, outDim, inDim), inDim, outDim); // [in, out]
        using var result = new Tensor(allocator, DType.Float32, rows, outDim);
        string shape = $"{rows} x {inDim} -> {outDim}{(range ? ", range" : "")}";

        Assert.True(CudaEmbeddingOps.TryLinearTf32(result, input, weight));
        AssertTf32(Measure(result.GetElementsAsFloat(rows * outDim), a, w, 1, rows, inDim, outDim),
            $"CudaEmbeddingOps.TryLinearTf32 {shape}");

        Assert.True(CudaEmbeddingOps.TryLinearFloat32(result, input, weight));
        AssertExactF32(Measure(result.GetElementsAsFloat(rows * outDim), a, w, 1, rows, inDim, outDim),
            $"CudaEmbeddingOps.TryLinearFloat32 {shape}");
    }

    /// <summary>MiniMax-H3's mapped F32 dense GEMM is pedantic up to 16 million products and TF32
    /// past them, by design (CudaMiniMaxH3Ops.GemmMapped).</summary>
    [CudaTf32Theory]
    [InlineData(64, 4096, 256, false)]  // 67M products: TF32
    [InlineData(64, 4096, 256, true)]   // TF32 keeps F32's exponent range
    [InlineData(16, 1024, 512, false)]  // 8.4M products: pedantic F32
    [InlineData(16, 1024, 512, true)]
    public void MiniMaxH3MappedF32Dense_KeepsItsTf32Threshold(int rows, int inner, int columns, bool range)
    {
        Assert.Null(Environment.GetEnvironmentVariable("NVIDIA_TF32_OVERRIDE"));
        using var allocator = new CudaAllocator();
        float[] a = Operands(rows * inner, seed: 61, scale: 1f);
        float[] w = Operands(columns * inner, seed: 62, scale: range ? RangeScale : 1f); // [columns, inner]
        using var input = Upload(allocator, a, rows, inner);
        using var output = new Tensor(allocator, DType.Float32, rows, columns);
        long bytes = (long)w.Length * sizeof(float);
        IntPtr host = Marshal.AllocHGlobal((IntPtr)bytes);
        try
        {
            Marshal.Copy(w, 0, host, w.Length);
            Assert.True(CudaMiniMaxH3Ops.TryMappedDense(output, input, host, 0, inner, columns, bytes));
            var fingerprint = Measure(output.GetElementsAsFloat(rows * columns), a, w, 1, rows, inner, columns);
            string what = $"CudaMiniMaxH3Ops.TryMappedDense F32 {rows} x {inner} -> {columns}{(range ? ", range" : "")}";
            if ((long)rows * inner * columns > 16_000_000)
                AssertTf32(fingerprint, what);
            else
                AssertExactF32(fingerprint, what);
        }
        finally
        {
            CudaQuantizedOps.ReleaseQuantizedWeight(allocator, host);
            Marshal.FreeHGlobal(host);
        }
    }

    /// <summary>The embedding attention plan's explicit FastTf32WithF32Storage policy keeps its TF32
    /// scores and values; the default pedantic plan on the same data is exact.</summary>
    [CudaTf32Theory]
    [InlineData(129, 256, 4, 2)]
    public void EmbeddingAttentionTf32Policy_StaysTf32(int tokens, int dims, int heads, int kvHeads)
    {
        Assert.Null(Environment.GetEnvironmentVariable("NVIDIA_TF32_OVERRIDE"));
        using var allocator = new CudaAllocator();
        float[] q = Operands(tokens * heads * dims, seed: 71, scale: 1f);
        float[] k = Operands(tokens * kvHeads * dims, seed: 72, scale: 1f);
        float[] v = Operands(tokens * kvHeads * dims, seed: 73, scale: 1f);
        using var qt = Upload(allocator, q, 1, tokens, heads, dims);
        using var kt = Upload(allocator, k, 1, tokens, kvHeads, dims);
        using var vt = Upload(allocator, v, 1, tokens, kvHeads, dims);
        using var y = new Tensor(allocator, DType.Float32, 1, tokens, heads, dims);
        float scale = 1f / MathF.Sqrt(dims);

        float[] Run(CudaEmbeddingAttentionComputePolicy policy, int expectedCompute)
        {
            using var plan = new CudaEmbeddingAttentionPlan(allocator, computePolicy: policy);
            Assert.True(CudaEmbeddingAttentionOps.TryGlobalAttention(y, qt, kt, vt, plan, [tokens], scale));
            float[] result = y.GetElementsAsFloat(tokens * heads * dims);
            plan.NotifyReadbackCompleted();
            Assert.Equal(expectedCompute, plan.SelectedComputeType);
            return result;
        }

        float[] exact = Run(CudaEmbeddingAttentionComputePolicy.PedanticF32, 69);
        float[] tf32 = Run(CudaEmbeddingAttentionComputePolicy.FastTf32WithF32Storage, 77);
        double[] reference = AttentionReference(q, k, v, tokens, heads, kvHeads, dims, scale);
        double exactError = RelativeRms(exact, reference), tf32Error = RelativeRms(tf32, reference);
        _output.WriteLine($"embedding attention {tokens} tokens x {heads}/{kvHeads} heads x {dims}: " +
            $"pedantic relative RMS error {exactError:E2}, TF32 policy {tf32Error:E2}");
        Assert.True(tf32Error >= 10 * exactError,
            $"The TF32 attention policy computed like the pedantic one ({tf32Error:E2} vs {exactError:E2}): its documented approximation is gone.");
        Assert.True(exactError <= 1e-6, $"The pedantic attention policy strayed {exactError:E2} from float64.");
    }

    // ---------------------------------------------------------------- the direct engines' MatMulF32

    /// <summary>GLM-5.3-Flash: the router and the indexer's head weights, then a 288-expert router
    /// over a 6144-wide hidden state.</summary>
    [CudaFact]
    public void GlmEngine_F32ProjectionsComputeExactly()
    {
        _env.Set("MAX_CONTEXT", "512");
        _env.Set("TS_GLM_UBATCH", "64");
        _env.Set("TS_GLM_NATIVE", null);
        _env.Set("TS_GLM_NGPU", null);
        _env.ClearSpeculationVars();
        string path = GlmDsaSyntheticModelBuilder.WriteGlm5NextCudaFixture(Path.Combine(_dir, "glm5next-cuda.gguf"));
        using var ex = new GlmDsaCudaExecutor(path, 512, 64, 1);
        AssertFiniteLogits(ex.VocabSize, (tokens, logits) => Assert.True(ex.Forward(tokens, logits)));
        int embd = NEmbd(ex.Engine);
        CheckEngineMatMulF32("glm", ex.Engine, new[] { "GateInp", "IdxProj" }, _ => embd,
            new[] { (288, 6144, new[] { 17, 512 }) });
    }

    private const string Dsv41FixtureVar = "TS_TEST_DSV41_CUDA_FIXTURE_DIR";
    private const string Dsv41Fixture = "deepseek41-fixture";

    /// <summary>DeepSeek V4: the router and both hyper-connection mixers of the V4.1 fixture, then
    /// the shipped model's 256 x 4096 router and 24 x 16384 mixer. The S8 finding: these, rounded to
    /// half, picked other experts.</summary>
    [CudaFact(Dsv41FixtureVar, Dsv41Fixture)]
    public void Dsv4Engine_F32ProjectionsComputeExactly()
    {
        using var ex = new DeepSeek4CudaExecutor(
            TestGates.FindGguf(Environment.GetEnvironmentVariable(Dsv41FixtureVar), Dsv41Fixture),
            maxContext: 1024, nUbatch: 32, nGpu: 1);
        AssertFiniteLogits(ex.VocabSize, ex.Forward);
        object engine = Get(ex, "_engine");
        int embd = NEmbd(engine);
        int hc = (int)Const(engine, "HC");
        CheckEngineMatMulF32("dsv4", engine, new[] { "GateInp", "HcAttnFn", "HcFfnFn" },
            name => name == "GateInp" ? embd : hc * embd,
            new[] { (256, 4096, new[] { 17, 512 }), (24, 16384, new[] { 64 }) });
    }

    private const BindingFlags AnyMember = BindingFlags.Instance | BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic;

    private static object Get(object owner, string name) =>
        (owner.GetType().GetField(name, AnyMember)
            ?? throw new MissingFieldException(owner.GetType().FullName, name)).GetValue(owner);

    private static object Const(object owner, string name) =>
        (owner.GetType().GetField(name, AnyMember)
            ?? throw new MissingFieldException(owner.GetType().FullName, name)).GetRawConstantValue();

    private static int NEmbd(object engine) => (int)Get(Get(engine, "_m"), "NEmbd");

    /// <summary>A 40-token prompt through the engine first: its F32 projections take cuBLAS past 16
    /// rows, and the step sizes the buffers (the GEMV scratch among them) the checks then use.</summary>
    private static void AssertFiniteLogits(int vocab, Action<int[], float[]> forward)
    {
        int[] prompt = Enumerable.Range(0, 40).Select(i => 3 + (i * 37 + 11) % 240).ToArray();
        var logits = new float[vocab];
        forward(prompt, logits);
        Assert.All(logits, value => Assert.True(float.IsFinite(value)));
    }

    /// <summary>
    /// Every F32 weight named in <paramref name="weightNames"/> of every layer, through the engine's
    /// own <c>MatMulF32</c> on the layer's device: 16 rows (the engine's row-invariant GEMV), 17 and
    /// 64 (cuBLAS). Then production-width weights on the same route, which the fixture's small
    /// widths cannot stand in for: cuBLAS picks tensor cores (and so, under the old handle mode,
    /// half operands) by shape. Reached by reflection: <c>MatMulF32</c> is the engine's private
    /// route, and the point is to check that route rather than a copy of it. Every projection is
    /// checked before the test fails, so a failure lists all of them.
    /// </summary>
    private void CheckEngineMatMulF32(string engineName, object engine, string[] weightNames, Func<string, int> inDimOf,
        (int Out, int In, int[] Rows)[] productionShapes)
    {
        MethodInfo matMul = engine.GetType().GetMethod("MatMulF32", AnyMember)
            ?? throw new MissingMethodException(engine.GetType().FullName, "MatMulF32");
        bool byPointer = matMul.GetParameters()[1].ParameterType == typeof(IntPtr);
        var devs = (Array)Get(engine, "_devs");
        var layers = (Array)Get(engine, "_layers");
        var failures = new List<string>();

        void Check(object dev, Tensor weight, float[] w, int inDim, int outDim, int rows, string what)
        {
            var allocator = (CudaAllocator)Get(dev, "Alloc");
            float[] a = Operands(rows * inDim, seed: 81 + rows, scale: 1f);
            using var input = Upload(allocator, a, rows, inDim);
            using var output = new Tensor(allocator, DType.Float32, rows, outDim);
            dev.GetType().GetMethod("MakeCurrent")!.Invoke(dev, null);
            ((CudaStorage)weight.Storage).EnsureDeviceCurrent();
            ((CudaStorage)input.Storage).EnsureDeviceCurrent();
            object weightArg = byPointer ? Dsv4CudaEngine.Ptr(weight) : weight;
            matMul.Invoke(matMul.IsStatic ? null : engine, new object[] { dev, weightArg, input, output, inDim, outDim, rows });
            string failure = ExactF32Failure(Measure(output.GetElementsAsFloat(rows * outDim), a, w, 1, rows, inDim, outDim),
                $"{engineName} MatMulF32 {what}, {rows} rows");
            if (failure != null)
                failures.Add(failure);
        }

        int fixtureWeights = 0;
        for (int il = 0; il < layers.Length; il++)
        {
            object layer = layers.GetValue(il);
            object dev = devs.GetValue((int)Get(layer, "Device"));
            foreach (string name in weightNames)
            {
                if (Get(layer, name) is not Tensor weight)
                    continue;
                int inDim = inDimOf(name);
                float[] w = weight.GetElementsAsFloat((int)weight.ElementCount());
                Assert.True(w.Length % inDim == 0, $"layer {il} {name}: {w.Length} values do not split into rows of {inDim}");
                int outDim = w.Length / inDim;
                foreach (int rows in new[] { 16, 17, 64 })
                    Check(dev, weight, w, inDim, outDim, rows, $"layer {il} {name} ({outDim} x {inDim})");
                fixtureWeights++;
            }
        }
        Assert.True(fixtureWeights > 0, $"{engineName}: the fixture has none of {string.Join(", ", weightNames)}.");

        object first = devs.GetValue(0);
        var firstAllocator = (CudaAllocator)Get(first, "Alloc");
        foreach (var (outDim, inDim, rowCounts) in productionShapes)
        {
            float[] w = Operands(outDim * inDim, seed: 91, scale: 1f);
            using var weight = Upload(firstAllocator, w, outDim, inDim);
            foreach (int rows in rowCounts)
                Check(first, weight, w, inDim, outDim, rows, $"{outDim} x {inDim} weight");
        }
        Assert.True(failures.Count == 0, string.Join(Environment.NewLine, failures));
    }

    // ---------------------------------------------------------------- measurement

    /// <summary><see cref="Tf32Error"/> and <see cref="FromTf32"/> are against the product of operands
    /// rounded as <see cref="Rounding"/> names: TF32 unless a check measures half.</summary>
    private readonly record struct Fingerprint(double Error, double MaxError, double Tf32Error, double FromTf32, int NonFinite, int Samples,
        string Rounding = "TF32")
    {
        public override string ToString() =>
            $"error {Error:E2} (max {MaxError:E2}) of sum|a*w| over {Samples} sampled outputs; " +
            $"{Rounding}-rounded operands would give {Tf32Error:E2}; distance to that product {FromTf32:E2}; {NonFinite} non-finite outputs";
    }

    private void AssertExactF32(Fingerprint f, string what)
    {
        string failure = ExactF32Failure(f, what);
        Assert.True(failure == null, failure);
    }

    /// <summary>Null when <paramref name="f"/> is an exact F32 product: finite, with an error at most
    /// a twentieth (mean) and a half (worst output) of what TF32-rounded operands give the same data.</summary>
    private string ExactF32Failure(Fingerprint f, string what)
    {
        _output.WriteLine($"{what}: {f}");
        if (f.NonFinite != 0)
            return $"{what}: {f.NonFinite} non-finite outputs from finite F32 operands (half operands overflow past 65504). {f}";
        if (!(f.Error <= f.Tf32Error / 20 && f.MaxError <= f.Tf32Error / 2))
            return $"{what}: the operands were not kept F32. {f}";
        return null;
    }

    private void AssertTf32(Fingerprint f, string what)
    {
        _output.WriteLine($"{what}: {f}");
        Assert.True(f.NonFinite == 0, $"{what}: {f.NonFinite} non-finite outputs: TF32 keeps F32's exponent range. {f}");
        Assert.True(f.Error >= f.Tf32Error / 3,
            $"{what}: computed as exact F32; the documented TF32 approximation is gone. {f}");
        Assert.True(f.FromTf32 <= f.Tf32Error / 20, $"{what}: not the product of TF32-rounded operands. {f}");
    }

    private void AssertHalf(Fingerprint f, string what)
    {
        _output.WriteLine($"{what}: {f}");
        Assert.True(f.NonFinite == 0, $"{what}: {f.NonFinite} non-finite outputs from operands well inside half's range. {f}");
        Assert.True(f.Error >= f.Tf32Error / 3,
            $"{what}: computed as exact F32; the named half-operand approximation is gone. {f}");
        Assert.True(f.FromTf32 <= f.Tf32Error / 20, $"{what}: not the product of half-rounded operands. {f}");
    }

    /// <summary>
    /// <paramref name="c"/>[batch, rows, outDim] against a[batch, rows, inDim] x w[batch, outDim, inDim]^T
    /// in float64, on up to 1024 sampled outputs (non-finite outputs are counted over all of them), and
    /// against the product of operands rounded by <paramref name="round"/> (TF32 by default).
    /// </summary>
    private static Fingerprint Measure(float[] c, float[] a, float[] w, int batch, int rows, int inDim, int outDim,
        Func<float, float> round = null)
    {
        string rounding = round == null ? "TF32" : "half";
        round ??= Tf32;
        int nonFinite = c.Count(value => !float.IsFinite(value));
        long total = (long)batch * rows * outDim;
        int samples = (int)Math.Min(total, 1024);
        double error = 0, maxError = 0, tf32 = 0, fromTf32 = 0;
        int counted = 0;
        for (int s = 0; s < samples; s++)
        {
            // A large prime stride visits distinct outputs spread over every row and column.
            long index = samples == total ? s : s * 1_000_003L % total;
            float got = c[index];
            if (!float.IsFinite(got))
                continue;
            long b = index / ((long)rows * outDim), r = index / outDim % rows, o = index % outDim;
            long aRow = (b * rows + r) * inDim, wRow = (b * outDim + o) * inDim;
            double exact = 0, rounded = 0, magnitude = 0;
            for (int k = 0; k < inDim; k++)
            {
                double product = (double)a[aRow + k] * w[wRow + k];
                exact += product;
                magnitude += Math.Abs(product);
                rounded += (double)round(a[aRow + k]) * round(w[wRow + k]);
            }
            double e = Math.Abs(got - exact) / magnitude;
            error += e;
            maxError = Math.Max(maxError, e);
            tf32 += Math.Abs(rounded - exact) / magnitude;
            fromTf32 += Math.Abs(got - rounded) / magnitude;
            counted++;
        }
        return counted == 0
            ? new Fingerprint(double.NaN, double.NaN, double.NaN, double.NaN, nonFinite, 0, rounding)
            : new Fingerprint(error / counted, maxError, tf32 / counted, fromTf32 / counted, nonFinite, counted, rounding);
    }

    /// <summary>Uniform in [-1, 1) times <paramref name="scale"/>, every value with its low 13
    /// mantissa bits random and at least one set: the bits TF32 and half drop.</summary>
    private static float[] Operands(int count, int seed, float scale)
    {
        var random = new Random(seed);
        var values = new float[count];
        for (int i = 0; i < count; i++)
        {
            float x = (float)(random.NextDouble() * 2 - 1);
            if (Math.Abs(x) < 1e-3f)
                x = 0.5f;
            uint bits = BitConverter.SingleToUInt32Bits(x) | (uint)(random.Next(0x2000) | 1);
            values[i] = BitConverter.UInt32BitsToSingle(bits) * scale;
        }
        return values;
    }

    /// <summary>F32 to IEEE half and back, round to nearest even (cvt.rn.f16.f32, the tensor-core
    /// conversion of <c>CUBLAS_COMPUTE_32F_FAST_16F</c>).</summary>
    private static float Half16(float x) => (float)(System.Half)x;

    /// <summary>F32 to TF32, round to nearest with ties away from zero (cvt.rna.tf32.f32).</summary>
    private static float Tf32(float x)
    {
        uint bits = BitConverter.SingleToUInt32Bits(x);
        return BitConverter.UInt32BitsToSingle((bits + 0x1000u) & ~0x1fffu);
    }

    private static float[] Transpose(float[] m, int rows, int cols)
    {
        var t = new float[m.Length];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                t[c * rows + r] = m[r * cols + c];
        return t;
    }

    private static Tensor Upload(CudaAllocator allocator, float[] data, params long[] sizes)
    {
        var tensor = new Tensor(allocator, DType.Float32, sizes);
        tensor.SetElementsAsFloat(data);
        return tensor;
    }

    private static double[] AttentionReference(float[] q, float[] k, float[] v, int tokens, int heads, int kvHeads, int dims, float scale)
    {
        var y = new double[tokens * heads * dims];
        var scores = new double[tokens];
        for (int t = 0; t < tokens; t++)
            for (int h = 0; h < heads; h++)
            {
                int kv = h / (heads / kvHeads);
                double max = double.NegativeInfinity;
                for (int j = 0; j < tokens; j++)
                {
                    double dot = 0;
                    for (int d = 0; d < dims; d++)
                        dot += (double)q[(t * heads + h) * dims + d] * k[(j * kvHeads + kv) * dims + d];
                    scores[j] = dot * scale;
                    max = Math.Max(max, scores[j]);
                }
                double sum = 0;
                for (int j = 0; j < tokens; j++)
                    sum += scores[j] = Math.Exp(scores[j] - max);
                for (int d = 0; d < dims; d++)
                {
                    double acc = 0;
                    for (int j = 0; j < tokens; j++)
                        acc += scores[j] * v[(j * kvHeads + kv) * dims + d];
                    y[(t * heads + h) * dims + d] = acc / sum;
                }
            }
        return y;
    }

    private static double RelativeRms(float[] actual, double[] expected)
    {
        double error = 0, norm = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            double d = actual[i] - expected[i];
            error += d * d;
            norm += expected[i] * expected[i];
        }
        return Math.Sqrt(error / norm);
    }
}
