// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Runtime.InteropServices;
using TensorSharp;
using TensorSharp.Cuda;
using TensorSharp.Models;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// NVFP4 products on the direct CUDA backend's resident weights (the executor's QuantizedMatMul): 1..16 rows run the
/// q8_1 dp4a matvec over the ggml blocks. Every row of a multi-row call equals the one-row call on it, to the bit, and
/// the rows are held to the dequantized weights with the activations rounded as the kernel rounds them (q8_1 with an
/// F16 scale), within 1e-4 relative L2 - what remains is accumulation order.
/// </summary>
public sealed class CudaNvfp4MatmulTests
{
    private const int Nvfp4 = 40, BlockBytes = 36, MaxRows = 16;
    private readonly ITestOutputHelper _output;

    public CudaNvfp4MatmulTests(ITestOutputHelper output) => _output = output;

    [CudaTheory]
    [InlineData(5120, 512)]
    [InlineData(5120, 2304)]
    [InlineData(17408, 320)]
    [InlineData(1024, 4100)]
    public void RowsMatchTheOneRowCallAndTheDequantizedWeights(int inDim, int outDim)
    {
        long rowBytes = inDim / 64 * BlockBytes;
        byte[] weights = Nvfp4Rows(outDim, inDim, seed: 7);
        var act = new float[MaxRows, inDim];
        for (int t = 0; t < MaxRows; t++)
            for (int k = 0; k < inDim; k++)
                act[t, k] = MathF.Sin(0.0173f * (k + 1) * (t + 2)) + 0.3f * MathF.Cos(0.061f * k - 3 * t);

        IntPtr host = Marshal.AllocHGlobal(weights.Length);
        try
        {
            Marshal.Copy(weights, 0, host, weights.Length);
            using var allocator = new CudaAllocator();
            using var input = Tensor.FromArray(allocator, act);
            var alone = new float[MaxRows * outDim];
            for (int r = 0; r < MaxRows; r++)
            {
                using Tensor x = input.Narrow(0, r, 1);
                using var y = new Tensor(allocator, DType.Float32, 1, outDim);
                Assert.True(CudaQuantizedOps.TryAddmmQuantizedToFloat32(y, x, host, host, Nvfp4, inDim, outDim, weights.Length));
                Array.Copy(y.GetElementsAsFloat(outDim), 0, alone, r * outDim, outDim);
            }

            // The matvec's rows: up to 16 (a decode step's or a verify window's) where NVFP4 weights are planar for the FP4
            // tensor cores (more run on the GEMMs) and on the ggml blocks elsewhere.
            bool planar = allocator.Fp4Kernels != null;
            float[] got = null;
            foreach (int rows in new[] { 2, 3, 4, 5, 8, 9, 12, 16 })
            {
                using Tensor x = input.Narrow(0, 0, rows);
                using var y = new Tensor(allocator, DType.Float32, rows, outDim);
                Assert.True(CudaQuantizedOps.TryAddmmQuantizedToFloat32(y, x, host, host, Nvfp4, inDim, outDim, weights.Length));
                got = y.GetElementsAsFloat(rows * outDim);
                for (int i = 0; i < rows * outDim; i++)
                    if (BitConverter.SingleToInt32Bits(got[i]) != BitConverter.SingleToInt32Bits(alone[i]))
                        Assert.Fail($"NVFP4 {inDim} -> {outDim}: row {i / outDim} of a {rows}-row call gives {got[i]:R} at column " +
                            $"{i % outDim}, the one-row call {alone[i]:R}");
            }
            CudaQuantizedOps.ReleaseQuantizedWeight(allocator, host);

            // The planar matvec keeps each 32-value scale in F32, the ggml-block one stores it as F16 (q8_1).
            float[,] reference = QuantizeDequantizeQ8_1(act, storeScaleAsHalf: !planar);
            var wRow = new float[inDim];
            double num = 0, den = 0;
            for (int o = 0; o < outDim; o++)
            {
                NativeDequant.DequantizeToFloat32(Nvfp4, weights, checked((int)(o * rowBytes)), wRow, 0, inDim);
                for (int t = 0; t < MaxRows; t++)
                {
                    double want = 0;
                    for (int k = 0; k < inDim; k++)
                        want += (double)wRow[k] * reference[t, k];
                    double d = alone[t * outDim + o] - want;
                    num += d * d;
                    den += want * want;
                }
            }
            double rel = Math.Sqrt(num / den);
            _output.WriteLine($"NVFP4 {inDim} -> {outDim}: rows of multi-row calls bit-equal to one-row calls; relative L2 {rel:E2} " +
                "against the dequantized weights");
            Assert.True(rel <= 1e-4, $"NVFP4 {inDim} -> {outDim}: {rel:E2} from the dequantized weights");
        }
        finally
        {
            Marshal.FreeHGlobal(host);
        }
    }

    /// <summary>
    /// Prompt rows on the FP4 tensor cores (sm_120a / sm_121a): the product equals the dequantized weights times the
    /// activations as the quantizer coded them (its own planes, dequantized on the host), within accumulation order,
    /// for one-wave split-K tiles (17 and 64 rows) and the 128-column tiles (200 rows), and an output edge.
    /// </summary>
    [CudaTheory]
    [InlineData(5120, 512, 17)]
    [InlineData(5120, 2304, 64)]
    [InlineData(17408, 320, 33)]
    [InlineData(1024, 4100, 200)]
    [InlineData(5120, 2304, 300)]
    public void PromptRowsOnTheFp4TensorCoresMatchTheirQuantizedOperands(int inDim, int outDim, int rows)
    {
        using var allocator = new CudaAllocator();
        CudaFp4Kernels fp4 = allocator.Fp4Kernels;
        if (fp4 == null)
        {
            allocator.Context.MakeCurrent();
            CudaFp4Kernels.TryCreate(out CudaFp4Kernels none, out string reason);
            none?.Dispose();
            _output.WriteLine($"no FP4 tensor-core products here ({reason}): the product keeps the dequantize + cuBLAS route");
            return;
        }
        long rowBytes = inDim / 64 * BlockBytes;
        byte[] weights = Nvfp4Rows(outDim, inDim, seed: 5);
        var act = new float[rows, inDim];
        for (int t = 0; t < rows; t++)
            for (int k = 0; k < inDim; k++)
                act[t, k] = MathF.Sin(0.0173f * (k + 1) * (t + 2)) + 0.3f * MathF.Cos(0.061f * k - 3 * t);
        IntPtr host = Marshal.AllocHGlobal(weights.Length);
        try
        {
            Marshal.Copy(weights, 0, host, weights.Length);
            using var input = Tensor.FromArray(allocator, act);
            using var y = new Tensor(allocator, DType.Float32, rows, outDim);
            Assert.True(CudaQuantizedOps.TryAddmmQuantizedToFloat32(y, input, host, host, Nvfp4, inDim, outDim, weights.Length));
            float[] got = y.GetElementsAsFloat(rows * outDim);
            CudaQuantizedOps.ReleaseQuantizedWeight(allocator, host);

            // The activations as the quantizer codes them.
            var (codeBytes, scaleBytes, rowScaleBytes) = CudaFp4Kernels.ActivationBytes(rows, inDim);
            using var codes = new Tensor(allocator, DType.UInt8, codeBytes);
            using var scales = new Tensor(allocator, DType.UInt8, scaleBytes);
            using var rowScale = new Tensor(allocator, DType.Float32, rows);
            allocator.Context.MakeCurrent();
            fp4.Quantize(Ptr(input), Ptr(codes), Ptr(scales), Ptr(rowScale), inDim, rows, allocator.Stream.Handle);
            allocator.Stream.Synchronize();
            byte[] c = Bytes(codes, codeBytes), sc = Bytes(scales, scaleBytes);
            float[] rs = rowScale.GetElementsAsFloat(rows);
            var coded = new float[rows, inDim];
            for (int t = 0; t < rows; t++)
                for (int k = 0; k < inDim; k++)
                {
                    int b = c[(long)t * inDim / 2 + k / 16 * 8 + k % 8];
                    int nibble = k % 16 < 8 ? b & 15 : b >> 4;
                    coded[t, k] = E2M1[nibble] * Ue4m3(sc[(long)t * inDim / 16 + k / 16]) * rs[t];
                }

            var wRow = new float[inDim];
            double num = 0, den = 0;
            for (int o = 0; o < outDim; o++)
            {
                NativeDequant.DequantizeToFloat32(Nvfp4, weights, checked((int)(o * rowBytes)), wRow, 0, inDim);
                for (int t = 0; t < rows; t++)
                {
                    double want = 0;
                    for (int k = 0; k < inDim; k++)
                        want += (double)wRow[k] * coded[t, k];
                    double d = got[t * outDim + o] - want;
                    num += d * d;
                    den += want * want;
                }
            }
            double rel = Math.Sqrt(num / den);
            _output.WriteLine($"NVFP4 FP4 GEMM {inDim} -> {outDim}, {rows} rows: relative L2 {rel:E2} against the quantized operands");
            Assert.True(rel <= 1e-5, $"NVFP4 FP4 GEMM {inDim} -> {outDim}, {rows} rows: {rel:E2} from the quantized operands");
        }
        finally
        {
            Marshal.FreeHGlobal(host);
        }
    }

    /// <summary>
    /// The activation quantizer of the W4A4 prompt products (ts_fp4_quantize_nvfp4_f32) codes every row as ggml-cuda's
    /// quantize_mmq_nvfp4 does (<see cref="Nvfp4ActivationQuantizer"/>, written on the host from its source), to the bit:
    /// row scales, UE4M3 block scales and E2M1 codes. Rows of sines over many magnitudes, a zero row, a row with one
    /// outlier, values whose blocks need subnormal scales and signed zeros.
    /// </summary>
    [CudaTheory(Fp4TensorCores = true)]
    [InlineData(256, 9)]
    [InlineData(2048, 33)]
    [InlineData(7680, 70)]
    public void ActivationQuantizerCodesRowsAsGgmlCudasQuantizer(int k, int rows)
    {
        using var allocator = new CudaAllocator();
        CudaFp4Kernels fp4 = allocator.Fp4Kernels;
        Assert.NotNull(fp4);
        var x = new float[rows * k];
        for (int t = 0; t < rows; t++)
            for (int i = 0; i < k; i++)
            {
                float magnitude = MathF.Pow(10f, (t % 7) - 3);   // 1e-3 .. 1e3 by row
                x[t * k + i] = magnitude * (MathF.Sin(0.0173f * (i + 1) * (t + 2)) + 0.3f * MathF.Cos(0.061f * i - 3 * t));
            }
        Array.Clear(x, 1 * k, k);                                        // a zero row
        x[2 * k + 5] = 5e4f;                                             // one outlier: the rest of the row is tiny next to it
        for (int i = 0; i < 64; i++) x[3 * k + i] = 1e-7f * (i - 32);    // blocks far below the row's largest value
        x[3 * k + 100] = 300f;
        x[4 * k + 7] = -0f;
        var (codeBytes, scaleBytes, _) = CudaFp4Kernels.ActivationBytes(rows, k);
        var x2 = new float[rows, k];
        Buffer.BlockCopy(x, 0, x2, 0, x.Length * sizeof(float));
        using var input = Tensor.FromArray(allocator, x2);
        input.Storage.EnsureDeviceCurrent();   // FromArray fills the host mirror only; the kernel reads the device copy
        using var codes = new Tensor(allocator, DType.UInt8, codeBytes);
        using var scales = new Tensor(allocator, DType.UInt8, scaleBytes);
        using var rowScale = new Tensor(allocator, DType.Float32, rows);
        allocator.Context.MakeCurrent();
        fp4.Quantize(Ptr(input), Ptr(codes), Ptr(scales), Ptr(rowScale), k, rows, allocator.Stream.Handle);
        allocator.Stream.Synchronize();
        foreach (Tensor written in new[] { codes, scales, rowScale }) ((CudaStorage)written.Storage).MarkDeviceModified();
        byte[] gotCodes = Bytes(codes, codeBytes), gotScales = Bytes(scales, scaleBytes);
        float[] gotRows = rowScale.GetElementsAsFloat(rows);

        var wantCodes = new byte[codeBytes];
        var wantScales = new byte[scaleBytes];
        var wantRows = new float[rows];
        Nvfp4ActivationQuantizer.Quantize(x, rows, k, wantCodes, wantScales, wantRows);
        for (int t = 0; t < rows; t++)
            Assert.True(BitConverter.SingleToInt32Bits(gotRows[t]) == BitConverter.SingleToInt32Bits(wantRows[t]),
                $"row {t}: row scale {gotRows[t]:R} against ggml's {wantRows[t]:R}");
        int scaleMismatches = 0, codeMismatches = 0, firstScale = -1, firstCode = -1;
        for (int i = 0; i < gotScales.Length; i++)
            if (gotScales[i] != wantScales[i]) { scaleMismatches++; if (firstScale < 0) firstScale = i; }
        for (int i = 0; i < gotCodes.Length; i++)
            if (gotCodes[i] != wantCodes[i]) { codeMismatches++; if (firstCode < 0) firstCode = i; }
        _output.WriteLine($"{rows} rows of {k}: {scaleMismatches} of {gotScales.Length} block scales and {codeMismatches} of {gotCodes.Length} code bytes differ from ggml's quantizer");
        Assert.True(scaleMismatches == 0, $"block scale {firstScale} (row {firstScale / (k / 16)}): {(firstScale >= 0 ? gotScales[firstScale] : 0)} against ggml's {(firstScale >= 0 ? wantScales[firstScale] : 0)}");
        Assert.True(codeMismatches == 0, $"code byte {firstCode} (row {firstCode / (k / 2)}): {(firstCode >= 0 ? gotCodes[firstCode] : 0):X2} against ggml's {(firstCode >= 0 ? wantCodes[firstCode] : 0):X2}");
    }

    /// <summary>
    /// A SwiGLU feed-forward's down projection on the FP4 tensor cores with the activation quantized straight from the fused
    /// gate|up rows (silu(gate) * up never stored) equals the activation computed first and multiplied, to the bit.
    /// </summary>
    [CudaTheory]
    [InlineData(17408, 5120, 40)]
    [InlineData(512, 256, 300)]
    public void FusedSwiGluDownProjectionMatchesTheTwoStepProduct(int ff, int outDim, int rows)
    {
        using var allocator = new CudaAllocator();
        if (allocator.Fp4Kernels == null)
        {
            _output.WriteLine("no FP4 tensor-core products here: the fused route does not apply");
            return;
        }
        byte[] weights = Nvfp4Rows(outDim, ff, seed: 13);
        var gu = new float[rows, 2 * ff];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < 2 * ff; c++)
                gu[r, c] = 2.5f * MathF.Sin(0.0091f * (c + 1) * (r + 3)) + 0.2f * MathF.Cos(0.033f * c - r);
        IntPtr host = Marshal.AllocHGlobal(weights.Length);
        try
        {
            Marshal.Copy(weights, 0, host, weights.Length);
            using var gateUp = Tensor.FromArray(allocator, gu);
            using var fused = new Tensor(allocator, DType.Float32, rows, outDim);
            Assert.True(CudaQuantizedOps.TryAddmmSwiGluNvfp4(fused, gateUp, host, host, Nvfp4, ff, outDim, weights.Length));
            float[] a = fused.GetElementsAsFloat(rows * outDim);

            using var act = new Tensor(allocator, DType.Float32, rows, ff);
            ((CudaStorage)gateUp.Storage).EnsureDeviceCurrent();
            ((CudaStorage)act.Storage).EnsureDeviceCurrent();
            allocator.Context.MakeCurrent();
            allocator.Kernels.LaunchSiLUMulSplitF32(((CudaStorage)gateUp.Storage).DevicePtrAtElement(0),
                ((CudaStorage)act.Storage).DevicePtrAtElement(0), rows, ff, allocator.Stream.Handle);
            ((CudaStorage)act.Storage).MarkDeviceModified();
            using var twoStep = new Tensor(allocator, DType.Float32, rows, outDim);
            Assert.True(CudaQuantizedOps.TryAddmmQuantizedToFloat32(twoStep, act, host, host, Nvfp4, ff, outDim, weights.Length));
            float[] b = twoStep.GetElementsAsFloat(rows * outDim);
            CudaQuantizedOps.ReleaseQuantizedWeight(allocator, host);
            for (int i = 0; i < a.Length; i++)
                if (BitConverter.SingleToInt32Bits(a[i]) != BitConverter.SingleToInt32Bits(b[i]))
                    Assert.Fail($"row {i / outDim} column {i % outDim}: fused {a[i]:R}, two-step {b[i]:R}");
        }
        finally
        {
            Marshal.FreeHGlobal(host);
        }
    }

    /// <summary>
    /// Rows of a fused gate|up NVFP4 projection with SwiGLU in the GEMM's epilogue equal the [rows, 2 ff] product and a SwiGLU
    /// over it, to the bit: prompt rows on the TMA GEMM (128-row token tiles, a partial one at 65 rows, 256-row ones from 768
    /// rows, a feed-forward width a 128-row weight tile would split across gate and up: ff = 1536 + 64), and a batched verify's
    /// 17-64 rows on the cp.async GEMM's paired tiles (32- and 64-column tiles, K split as the product splits it).
    /// </summary>
    [CudaTheory]
    [InlineData(17408, 5120, 32)]
    [InlineData(17408, 5120, 40)]
    [InlineData(512, 1024, 17)]
    [InlineData(1600, 2048, 64)]
    [InlineData(512, 1024, 65)]
    [InlineData(1600, 2048, 300)]
    [InlineData(17408, 5120, 800)]
    public void GateUpProductWithSwiGluEpilogueMatchesTheProductAndItsSwiGlu(int ff, int inDim, int rows)
    {
        using var allocator = new CudaAllocator();
        if (allocator.Fp4Kernels == null)
        {
            _output.WriteLine("no FP4 tensor-core products here: the epilogue route does not apply");
            return;
        }
        byte[] weights = Nvfp4Rows(2 * ff, inDim, seed: 17);
        var x = new float[rows, inDim];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < inDim; c++)
                x[r, c] = 1.5f * MathF.Sin(0.0137f * (c + 1) * (r + 2)) + 0.25f * MathF.Cos(0.047f * c - 2 * r);
        IntPtr host = Marshal.AllocHGlobal(weights.Length);
        try
        {
            Marshal.Copy(weights, 0, host, weights.Length);
            using var input = Tensor.FromArray(allocator, x);
            using var fused = new Tensor(allocator, DType.Float32, rows, ff);
            Assert.True(CudaQuantizedOps.TryAddmmGateUpSwiGluNvfp4(fused, input, host, host, Nvfp4, inDim, 2 * ff, weights.Length));
            float[] got = fused.GetElementsAsFloat(rows * ff);

            using var gateUp = new Tensor(allocator, DType.Float32, rows, 2 * ff);
            Assert.True(CudaQuantizedOps.TryAddmmQuantizedToFloat32(gateUp, input, host, host, Nvfp4, inDim, 2 * ff, weights.Length));
            using var act = new Tensor(allocator, DType.Float32, rows, ff);
            ((CudaStorage)gateUp.Storage).EnsureDeviceCurrent();
            ((CudaStorage)act.Storage).EnsureDeviceCurrent();
            allocator.Context.MakeCurrent();
            allocator.Kernels.LaunchSiLUMulSplitF32(Ptr(gateUp), Ptr(act), rows, ff, allocator.Stream.Handle);
            ((CudaStorage)act.Storage).MarkDeviceModified();
            float[] want = act.GetElementsAsFloat(rows * ff);
            CudaQuantizedOps.ReleaseQuantizedWeight(allocator, host);
            for (int i = 0; i < want.Length; i++)
                if (BitConverter.SingleToInt32Bits(got[i]) != BitConverter.SingleToInt32Bits(want[i]))
                    Assert.Fail($"ff {ff}, {rows} rows: row {i / ff} column {i % ff}: epilogue {got[i]:R}, product + SwiGLU {want[i]:R}");
        }
        finally
        {
            Marshal.FreeHGlobal(host);
        }
    }

    /// <summary>
    /// The down projection's NVFP4 input is the same from either path: the plain quantizer over the epilogue's activation gives
    /// the codes, scales and row scales the SwiGLU quantizer gives over the [rows, 2 ff] product, byte for byte (a batched
    /// verify's rows and prompt rows).
    /// </summary>
    [CudaTheory]
    [InlineData(17408, 5120, 32)]
    [InlineData(1600, 2048, 64)]
    [InlineData(512, 1024, 65)]
    public void EpilogueAndSwiGluQuantizerGiveTheSameDownInput(int ff, int inDim, int rows)
    {
        using var allocator = new CudaAllocator();
        if (allocator.Fp4Kernels is not { } fp4)
        {
            _output.WriteLine("no FP4 tensor-core products here: the epilogue route does not apply");
            return;
        }
        byte[] weights = Nvfp4Rows(2 * ff, inDim, seed: 29);
        var x = new float[rows, inDim];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < inDim; c++)
                x[r, c] = 1.25f * MathF.Sin(0.0113f * (c + 3) * (r + 1)) + 0.3f * MathF.Cos(0.031f * c + r);
        IntPtr host = Marshal.AllocHGlobal(weights.Length);
        try
        {
            Marshal.Copy(weights, 0, host, weights.Length);
            using var input = Tensor.FromArray(allocator, x);
            using var act = new Tensor(allocator, DType.Float32, rows, ff);
            Assert.True(CudaQuantizedOps.TryAddmmGateUpSwiGluNvfp4(act, input, host, host, Nvfp4, inDim, 2 * ff, weights.Length));
            using var gateUp = new Tensor(allocator, DType.Float32, rows, 2 * ff);
            Assert.True(CudaQuantizedOps.TryAddmmQuantizedToFloat32(gateUp, input, host, host, Nvfp4, inDim, 2 * ff, weights.Length));
            var (codeBytes, scaleBytes, rowScaleBytes) = CudaFp4Kernels.ActivationBytes(rows, ff);
            byte[] Planes(Action<IntPtr, IntPtr, IntPtr> quantize)
            {
                using var codes = new Tensor(allocator, DType.UInt8, codeBytes);
                using var scales = new Tensor(allocator, DType.UInt8, scaleBytes);
                using var rowScale = new Tensor(allocator, DType.Float32, rows);
                allocator.Context.MakeCurrent();
                quantize(Ptr(codes), Ptr(scales), Ptr(rowScale));
                allocator.Stream.Synchronize();
                float[] rs = rowScale.GetElementsAsFloat(rows);
                return Bytes(codes, codeBytes).Concat(Bytes(scales, scaleBytes))
                    .Concat(rs.SelectMany(BitConverter.GetBytes)).ToArray();
            }
            ((CudaStorage)act.Storage).EnsureDeviceCurrent();
            ((CudaStorage)gateUp.Storage).EnsureDeviceCurrent();
            byte[] fromEpilogue = Planes((c, sc, rs) => fp4.Quantize(Ptr(act), c, sc, rs, ff, rows, allocator.Stream.Handle));
            byte[] fromSwiGlu = Planes((c, sc, rs) => fp4.QuantizeSwiGlu(Ptr(gateUp), c, sc, rs, ff, rows, allocator.Stream.Handle));
            CudaQuantizedOps.ReleaseQuantizedWeight(allocator, host);
            Assert.Equal(fromSwiGlu.Length, fromEpilogue.Length);
            for (int i = 0; i < fromSwiGlu.Length; i++)
                if (fromSwiGlu[i] != fromEpilogue[i])
                    Assert.Fail($"ff {ff}, {rows} rows: byte {i} of the down input planes: {fromEpilogue[i]} from the epilogue, {fromSwiGlu[i]} from the SwiGLU quantizer");
        }
        finally
        {
            Marshal.FreeHGlobal(host);
        }
    }

    /// <summary>A gate|up width the epilogue's 64-row tiles do not divide, or rows the q8 matvec takes: the route declines (the
    /// caller runs the product and the SwiGLU).</summary>
    [CudaFact]
    public void GateUpSwiGluEpilogueDeclinesWhatItDoesNotTile()
    {
        using var allocator = new CudaAllocator();
        if (allocator.Fp4Kernels == null)
            return;
        const int inDim = 512;
        foreach ((int ff, int rows) in new[] { (96, 100), (96, 32), (512, 16) })
        {
            byte[] weights = Nvfp4Rows(2 * ff, inDim, seed: 3);
            IntPtr host = Marshal.AllocHGlobal(weights.Length);
            try
            {
                Marshal.Copy(weights, 0, host, weights.Length);
                using var input = new Tensor(allocator, DType.Float32, rows, inDim);
                using var act = new Tensor(allocator, DType.Float32, rows, ff);
                Assert.False(CudaQuantizedOps.TryAddmmGateUpSwiGluNvfp4(act, input, host, host, Nvfp4, inDim, 2 * ff, weights.Length));
                CudaQuantizedOps.ReleaseQuantizedWeight(allocator, host);
            }
            finally
            {
                Marshal.FreeHGlobal(host);
            }
        }
    }

    /// <summary>
    /// <see cref="CudaAllocator.AllNvfp4RowsOnMatvecForTests"/> (the quality probe's q8act arm): a product of prompt rows runs
    /// on the q8 matvec, 16 rows a call, so each row equals its one-row call to the bit where the FP4 GEMM quantizes the same
    /// rows to NVFP4; the fused gate|up epilogue (a GEMM) declines under it. Cleared, the products are the GEMM's again.
    /// Without the FP4 tensor-core products NVFP4 rows never take the GEMMs and the seam changes no route: the test skips there.
    /// </summary>
    [CudaFact(Fp4TensorCores = true)]
    public void PromptRowsOnTheMatvecSeamRoundAsTheirOneRowCalls()
    {
        using var allocator = new CudaAllocator();
        Assert.True(allocator.Fp4Kernels != null, "the gate found the FP4 tensor-core products on GPU 0, but this allocator did not load them");
        const int inDim = 5120, outDim = 512, rows = 70, ff = 512, guIn = 1024;
        byte[] weights = Nvfp4Rows(outDim, inDim, seed: 13), gateUpWeights = Nvfp4Rows(2 * ff, guIn, seed: 19);
        var act = new float[rows, inDim];
        for (int t = 0; t < rows; t++)
            for (int k = 0; k < inDim; k++)
                act[t, k] = MathF.Sin(0.0173f * (k + 1) * (t + 2)) + 0.3f * MathF.Cos(0.061f * k - 3 * t);
        var guAct = new float[rows, guIn];
        for (int t = 0; t < rows; t++)
            for (int k = 0; k < guIn; k++)
                guAct[t, k] = 1.5f * MathF.Sin(0.0137f * (k + 1) * (t + 2)) + 0.25f * MathF.Cos(0.047f * k - 2 * t);
        IntPtr host = Marshal.AllocHGlobal(weights.Length), guHost = Marshal.AllocHGlobal(gateUpWeights.Length);
        try
        {
            Marshal.Copy(weights, 0, host, weights.Length);
            Marshal.Copy(gateUpWeights, 0, guHost, gateUpWeights.Length);
            using var input = Tensor.FromArray(allocator, act);
            using var guInput = Tensor.FromArray(allocator, guAct);
            var alone = new float[rows * outDim];
            for (int r = 0; r < rows; r++)
            {
                using Tensor x = input.Narrow(0, r, 1);
                using var y = new Tensor(allocator, DType.Float32, 1, outDim);
                Assert.True(CudaQuantizedOps.TryAddmmQuantizedToFloat32(y, x, host, host, Nvfp4, inDim, outDim, weights.Length));
                Array.Copy(y.GetElementsAsFloat(outDim), 0, alone, r * outDim, outDim);
            }
            float[] Product()
            {
                using var y = new Tensor(allocator, DType.Float32, rows, outDim);
                Assert.True(CudaQuantizedOps.TryAddmmQuantizedToFloat32(y, input, host, host, Nvfp4, inDim, outDim, weights.Length));
                return y.GetElementsAsFloat(rows * outDim);
            }
            bool GateUpEpilogue()
            {
                using var y = new Tensor(allocator, DType.Float32, rows, ff);
                return CudaQuantizedOps.TryAddmmGateUpSwiGluNvfp4(y, guInput, guHost, guHost, Nvfp4, guIn, 2 * ff, gateUpWeights.Length);
            }

            float[] gemm = Product();
            Assert.True(GateUpEpilogue());
            float[] seam;
            allocator.AllNvfp4RowsOnMatvecForTests = true;
            try
            {
                seam = Product();
                Assert.False(GateUpEpilogue());
            }
            finally
            {
                allocator.AllNvfp4RowsOnMatvecForTests = false;
            }
            float[] again = Product();
            Assert.True(GateUpEpilogue(), "the gate|up epilogue still declines after the seam was cleared");

            int gemmRowsApart = 0;
            for (int r = 0; r < rows; r++)
            {
                bool apart = false;
                for (int o = 0; o < outDim; o++)
                {
                    int i = r * outDim + o;
                    if (BitConverter.SingleToInt32Bits(seam[i]) != BitConverter.SingleToInt32Bits(alone[i]))
                        Assert.Fail($"row {r} of {rows} under the seam gives {seam[i]:R} at column {o}, its one-row call {alone[i]:R}");
                    if (BitConverter.SingleToInt32Bits(again[i]) != BitConverter.SingleToInt32Bits(gemm[i]))
                        Assert.Fail($"row {r}: the GEMM gives {again[i]:R} at column {o} after the seam, {gemm[i]:R} before it");
                    apart |= BitConverter.SingleToInt32Bits(gemm[i]) != BitConverter.SingleToInt32Bits(alone[i]);
                }
                gemmRowsApart += apart ? 1 : 0;
            }
            _output.WriteLine($"{rows} rows under the seam equal their one-row calls; on the FP4 GEMM {gemmRowsApart} of them round apart");
            Assert.True(gemmRowsApart > 0, "the FP4 GEMM's rows equal the matvec's: the seam changed no route");
            CudaQuantizedOps.ReleaseQuantizedWeight(allocator, host);
            CudaQuantizedOps.ReleaseQuantizedWeight(allocator, guHost);
        }
        finally
        {
            Marshal.FreeHGlobal(host);
            Marshal.FreeHGlobal(guHost);
        }
    }

    /// <summary>Rows of an NVFP4 table gathered on the device (the planar layout's lookup where weights are planar) equal
    /// the host's dequantization.</summary>
    /// <summary>
    /// Prompt rows with 8-bit activations (an allocator under <see cref="Nvfp4PromptActivations.Q8"/>, a model's
    /// --nvfp4-prompt-activations q8): the W4A8 GEMM computes the q8 matvec's products at any number of rows, on the F16
    /// tensor cores TMA-fed from 49 rows of weights in 16-row groups (200 x 512, 129 x 1040), on the int8 tensor cores
    /// otherwise (17 and 33 rows; 100 and 4100 weight rows). Held to the dequantized weights times the activations rounded as the matvec rounds them (q8_1, F32
    /// scales) within 1e-4 relative L2, the matvec's own bound, and to the matvec seam's rows (the same quantizer, 16 rows a
    /// call) within 1e-5 of the largest value: only the order of the F32 sums differs. Tiles cut at both edges (token and
    /// weight rows not multiples of the tile) and one or several K steps.
    /// </summary>
    [CudaTheory(Fp4TensorCores = true)]
    [InlineData(256, 64, 17)]
    [InlineData(512, 100, 65)]
    [InlineData(5120, 512, 200)]
    [InlineData(1024, 4100, 33)]
    [InlineData(2048, 1040, 129)]
    public void PromptRowsWithEightBitActivationsGiveTheMatvecsProducts(int inDim, int outDim, int rows)
    {
        long rowBytes = inDim / 64 * BlockBytes;
        byte[] weights = Nvfp4Rows(outDim, inDim, seed: 23);
        var act = new float[rows, inDim];
        for (int t = 0; t < rows; t++)
            for (int k = 0; k < inDim; k++)
                act[t, k] = MathF.Sin(0.0173f * (k + 1) * (t + 2)) + 0.3f * MathF.Cos(0.061f * k - 3 * t);
        IntPtr host = Marshal.AllocHGlobal(weights.Length);
        try
        {
            Marshal.Copy(weights, 0, host, weights.Length);
            using var allocator = new CudaAllocator();
            Assert.True(allocator.Fp4Kernels != null, "the gate found the FP4 tensor-core products on GPU 0, but this allocator did not load them");
            allocator.Nvfp4PromptActivations = Nvfp4PromptActivations.Q8;
            using var input = Tensor.FromArray(allocator, act);
            float[] Run()
            {
                using var y = new Tensor(allocator, DType.Float32, rows, outDim);
                Assert.True(CudaQuantizedOps.TryAddmmQuantizedToFloat32(y, input, host, host, Nvfp4, inDim, outDim, weights.Length));
                return y.GetElementsAsFloat(rows * outDim);
            }
            float[] gemm = Run();
            allocator.AllNvfp4RowsOnMatvecForTests = true;
            float[] matvec = Run();
            allocator.AllNvfp4RowsOnMatvecForTests = false;
            CudaQuantizedOps.ReleaseQuantizedWeight(allocator, host);

            float[,] reference = QuantizeDequantizeQ8_1(act, storeScaleAsHalf: false);
            var wRow = new float[inDim];
            double num = 0, den = 0, apart = 0, largest = 0;
            for (int o = 0; o < outDim; o++)
            {
                NativeDequant.DequantizeToFloat32(Nvfp4, weights, checked((int)(o * rowBytes)), wRow, 0, inDim);
                for (int t = 0; t < rows; t++)
                {
                    double want = 0;
                    for (int k = 0; k < inDim; k++)
                        want += (double)wRow[k] * reference[t, k];
                    double d = gemm[t * outDim + o] - want;
                    num += d * d;
                    den += want * want;
                    apart = Math.Max(apart, Math.Abs((double)gemm[t * outDim + o] - matvec[t * outDim + o]));
                    largest = Math.Max(largest, Math.Abs(want));
                }
            }
            double rel = Math.Sqrt(num / den);
            _output.WriteLine($"W4A8 {rows} x {inDim} -> {outDim}: relative L2 {rel:E2} against the dequantized operands; " +
                $"{apart / largest:E2} of the largest value from the matvec's rows");
            Assert.True(rel <= 1e-4, $"W4A8 {rows} x {inDim} -> {outDim}: {rel:E2} from the dequantized operands");
            Assert.True(apart <= 1e-5 * largest, $"W4A8 {rows} x {inDim} -> {outDim}: {apart / largest:E2} from the matvec's rows");
        }
        finally
        {
            Marshal.FreeHGlobal(host);
        }
    }

    /// <summary>Under <see cref="Nvfp4PromptActivations.Q8"/> the routes that write NVFP4 activation planes (the fused gate|up
    /// epilogue, the SwiGLU quantizer's down projection, a writer's planes) decline, so their products run on the W4A8 GEMM
    /// instead; under the default they take the rows.</summary>
    [CudaFact(Fp4TensorCores = true)]
    public void TheNvfp4ActivationRoutesDeclineUnderEightBitActivations()
    {
        using var allocator = new CudaAllocator();
        Assert.True(allocator.Fp4Kernels != null, "the gate found the FP4 tensor-core products on GPU 0, but this allocator did not load them");
        const int ff = 512, inDim = 1024, rows = 100;
        byte[] gateUp = Nvfp4Rows(2 * ff, inDim, seed: 29), down = Nvfp4Rows(inDim, ff, seed: 31);
        IntPtr hostGateUp = Marshal.AllocHGlobal(gateUp.Length), hostDown = Marshal.AllocHGlobal(down.Length);
        try
        {
            Marshal.Copy(gateUp, 0, hostGateUp, gateUp.Length);
            Marshal.Copy(down, 0, hostDown, down.Length);
            using var x = new Tensor(allocator, DType.Float32, rows, inDim);
            using var gu = new Tensor(allocator, DType.Float32, rows, 2 * ff);
            using var act = new Tensor(allocator, DType.Float32, rows, ff);
            using var y = new Tensor(allocator, DType.Float32, rows, inDim);
            Ops.Fill(x, 0.25f);
            Ops.Fill(gu, 0.5f);
            foreach (Nvfp4PromptActivations activations in new[] { Nvfp4PromptActivations.Q8, Nvfp4PromptActivations.Fp4 })
            {
                allocator.Nvfp4PromptActivations = activations;
                bool fp4 = activations == Nvfp4PromptActivations.Fp4;
                Assert.Equal(fp4, allocator.Nvfp4ActivationRows(rows));
                Assert.Equal(fp4, CudaQuantizedOps.TryAddmmGateUpSwiGluNvfp4(act, x, hostGateUp, hostGateUp, Nvfp4, inDim, 2 * ff, gateUp.Length));
                Assert.Equal(fp4, CudaQuantizedOps.TryAddmmSwiGluNvfp4(y, gu, hostDown, hostDown, Nvfp4, ff, inDim, down.Length));
            }
            Assert.False(allocator.Nvfp4ActivationRows(CudaFp4Kernels.GemvMaxRows));   // decode rows: the matvec either way
            CudaQuantizedOps.ReleaseQuantizedWeight(allocator, hostGateUp);
            CudaQuantizedOps.ReleaseQuantizedWeight(allocator, hostDown);
        }
        finally
        {
            Marshal.FreeHGlobal(hostGateUp);
            Marshal.FreeHGlobal(hostDown);
        }
    }

    [CudaFact]
    public void GatheredRowsMatchTheHostDequantization()
    {
        const int k = 512, tableRows = 300;
        byte[] table = Nvfp4Rows(tableRows, k, seed: 9);
        int[] ids = { 0, 299, 17, 17, 128 };
        IntPtr host = Marshal.AllocHGlobal(table.Length);
        try
        {
            Marshal.Copy(table, 0, host, table.Length);
            using var allocator = new CudaAllocator();
            using var indices = new Tensor(allocator, DType.Int32, ids.Length);
            indices.SetElementsAsInt(ids);
            using var rows = new Tensor(allocator, DType.Float32, ids.Length, k);
            Assert.True(CudaQuantizedOps.TryGetRowsQuantizedToFloat32(rows, host, host, Nvfp4, k, tableRows, table.Length, indices));
            float[] got = rows.GetElementsAsFloat(ids.Length * k);
            CudaQuantizedOps.ReleaseQuantizedWeight(allocator, host);
            var want = new float[k];
            for (int i = 0; i < ids.Length; i++)
            {
                NativeDequant.DequantizeToFloat32(Nvfp4, table, ids[i] * (k / 64) * BlockBytes, want, 0, k);
                for (int c = 0; c < k; c++)
                    Assert.Equal(want[c], got[i * k + c]);
            }
        }
        finally
        {
            Marshal.FreeHGlobal(host);
        }
    }

    private static readonly float[] E2M1 = { 0f, 0.5f, 1f, 1.5f, 2f, 3f, 4f, 6f, -0f, -0.5f, -1f, -1.5f, -2f, -3f, -4f, -6f };

    /// <summary>A UE4M3 scale byte (ggml's ggml_ue4m3_to_fp32 without its halving: E2M1 above is undoubled).</summary>
    private static float Ue4m3(byte v)
    {
        if (v == 0 || v == 0x7F) return 0f;
        int e = (v >> 3) & 15, m = v & 7;
        return e == 0 ? m / 512f : (1f + m / 8f) * MathF.Pow(2, e - 7);
    }

    private static IntPtr Ptr(Tensor t) => ((CudaStorage)t.Storage).DevicePtrAtElement(t.StorageOffset);

    private static byte[] Bytes(Tensor t, long count)
    {
        var bytes = new byte[count];
        t.CopyToArray(bytes);
        return bytes;
    }

    /// <summary>NVFP4 rows of random codes under scales a checkpoint holds (UE4M3 exponents 4..10, never the NaN pattern).</summary>
    internal static byte[] Nvfp4Rows(int rows, int cols, int seed)
    {
        var bytes = new byte[(long)rows * (cols / 64) * BlockBytes];
        uint x = (uint)(0x9E3779B9 * seed);
        for (int i = 0; i < bytes.Length; i++)
        {
            x ^= x << 13; x ^= x >> 17; x ^= x << 5;
            bytes[i] = (byte)x;
        }
        for (long b = 0; b < bytes.Length; b += BlockBytes)
            for (int s = 0; s < 4; s++)
                bytes[b + s] = (byte)(((4 + bytes[b + s] % 7) << 3) | (bytes[b + s] & 7));
        return bytes;
    }

    private static float[,] QuantizeDequantizeQ8_1(float[,] input, bool storeScaleAsHalf)
    {
        int rows = input.GetLength(0), cols = input.GetLength(1);
        var result = new float[rows, cols];
        for (int r = 0; r < rows; r++)
            for (int block = 0; block < cols; block += 32)
            {
                float amax = 0;
                for (int i = 0; i < 32; i++)
                    amax = MathF.Max(amax, MathF.Abs(input[r, block + i]));
                float d = amax > 0 ? amax / 127.0f : 0, id = d > 0 ? 1.0f / d : 0, stored = storeScaleAsHalf ? (float)(System.Half)d : d;
                for (int i = 0; i < 32; i++)
                    result[r, block + i] = stored * Math.Clamp((int)MathF.Round(input[r, block + i] * id, MidpointRounding.ToEven), -127, 127);
            }
        return result;
    }
}
