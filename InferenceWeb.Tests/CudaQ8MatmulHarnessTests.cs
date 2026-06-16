// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Harness for the Q8_0 multi-row GEMM on the pure-C# CUDA backend (the MTP verify
// matmul). It serves two purposes for the tensor-core-MMA effort:
//   1. Go/no-go: measure matmul wall-time vs M (rows). If ~flat for the verify range
//      (M<=8), the matmul is WEIGHT-BANDWIDTH-bound there, so a tensor-core MMA (which
//      accelerates COMPUTE, and is wasteful for M<16) cannot help the verify — before
//      committing to a multi-week kernel.
//   2. Scaffold for the correctness harness: building a random Q8_0 weight + driving
//      CudaQuantizedOps.TryAddmmQuantizedToFloat32, against which a future MMA kernel
//      is unit-tested on random inputs.
// Opt-in: TS_CUDA_Q8_HARNESS=1 (needs CUDA).
using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using TensorSharp;
using TensorSharp.Cuda;
using TensorSharp.Cuda.Interop;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

public class CudaQ8MatmulHarnessTests
{
    private readonly ITestOutputHelper _output;
    public CudaQ8MatmulHarnessTests(ITestOutputHelper output) { _output = output; }

    [Fact]
    public void Cuda_Q8Mma_CorrectnessVsDp4aAndReference()
    {
        if (Environment.GetEnvironmentVariable("TS_CUDA_Q8_HARNESS") != "1")
        { _output.WriteLine("[q8-mma] set TS_CUDA_Q8_HARNESS=1 to run; skipping"); return; }

        var allocator = new CudaAllocator(0);
        // Small shapes for a fast CPU reference; kernel correctness is shape-independent.
        foreach (var (M, K, N) in new[] { (2, 256, 128), (4, 512, 256), (8, 2560, 64), (16, 320, 256), (3, 2048, 80) })
        {
            int blocks = K / 32;
            long rowBytes = (long)blocks * 34;
            long rawBytes = (long)N * rowBytes;
            IntPtr hostData = Marshal.AllocHGlobal((nint)rawBytes);
            try
            {
                var rng = new Random(20260616 + M * 31 + N);
                // Build a real Q8_0 weight + the matching dequantized f32 reference (using
                // the F16-rounded scale the kernel reads, so only act-quant error remains).
                var wRef = new float[(long)N * K];
                unsafe
                {
                    byte* p = (byte*)hostData;
                    for (int n = 0; n < N; n++)
                        for (int bk = 0; bk < blocks; bk++)
                        {
                            float d = (float)(rng.NextDouble() * 0.05 + 0.005);
                            System.Half dh = (System.Half)d;
                            float df = (float)dh;
                            byte* blk = p + ((long)n * blocks + bk) * 34;
                            ushort dbits = BitConverter.HalfToUInt16Bits(dh);
                            blk[0] = (byte)(dbits & 0xFF);
                            blk[1] = (byte)(dbits >> 8);
                            for (int j = 0; j < 32; j++)
                            {
                                int q = rng.Next(-127, 128);
                                ((sbyte*)(blk + 2))[j] = (sbyte)q;
                                wRef[(long)n * K + bk * 32 + j] = df * q;
                            }
                        }
                }

                var actData = new float[(long)M * K];
                for (int i = 0; i < actData.Length; i++) actData[i] = (float)(rng.NextDouble() * 2 - 1);

                using var input = new Tensor(allocator, DType.Float32, M, K);
                input.SetElementsAsFloat(actData);
                using var outMma = new Tensor(allocator, DType.Float32, M, N);
                using var outDp4a = new Tensor(allocator, DType.Float32, M, N);

                Assert.True(CudaQuantizedOps.TryAddmmQuantizedToFloat32(outMma, input, hostData, hostData, 8, K, N, rawBytes, q8Kernel: 2), "MMA path declined");
                Assert.True(CudaQuantizedOps.TryAddmmQuantizedToFloat32(outDp4a, input, hostData, hostData, 8, K, N, rawBytes, q8Kernel: 1), "dp4a path declined");

                float[] mma = outMma.GetElementsAsFloat((int)((long)M * N));
                float[] dp4a = outDp4a.GetElementsAsFloat((int)((long)M * N));

                // CPU f32 reference (raw activations, no q8_1 quant).
                double maxRelRef = 0, maxAbsVsDp4a = 0, refScale = 0;
                for (int m = 0; m < M; m++)
                    for (int n = 0; n < N; n++)
                    {
                        double acc = 0;
                        for (int k = 0; k < K; k++) acc += (double)actData[(long)m * K + k] * wRef[(long)n * K + k];
                        int idx = m * N + n;
                        refScale = Math.Max(refScale, Math.Abs(acc));
                        maxRelRef = Math.Max(maxRelRef, Math.Abs(mma[idx] - acc));
                        maxAbsVsDp4a = Math.Max(maxAbsVsDp4a, Math.Abs(mma[idx] - dp4a[idx]));
                    }
                double relRef = refScale > 0 ? maxRelRef / refScale : 0;
                double relDp4a = refScale > 0 ? maxAbsVsDp4a / refScale : 0;
                _output.WriteLine($"[q8-mma] M={M} K={K} N={N}: max|mma-dp4a|/scale={relDp4a:E2}  max|mma-ref|/scale={relRef:E2}");

                // MMA must match dp4a closely (same q8_1 quant + int dot; only FP order differs)...
                Assert.True(relDp4a < 1e-3, $"MMA vs dp4a diff too large ({relDp4a:E2}) for M={M} K={K} N={N}");
                // ...and match the f32 reference within q8_1 activation-quant error (~<2%).
                Assert.True(relRef < 2e-2, $"MMA vs f32 ref diff too large ({relRef:E2}) for M={M} K={K} N={N}");
            }
            finally
            {
                CudaQuantizedOps.ReleaseQuantizedWeight(allocator, hostData);
                Marshal.FreeHGlobal(hostData);
            }
        }
        _output.WriteLine("[q8-mma] CORRECTNESS PASS — tensor-core MMA matches dp4a + f32 reference.");
    }

    [Fact]
    public void Cuda_Q8Matmul_ScalingProfile_GoNoGo()
    {
        if (Environment.GetEnvironmentVariable("TS_CUDA_Q8_HARNESS") != "1")
        { _output.WriteLine("[q8-harness] set TS_CUDA_Q8_HARNESS=1 to run; skipping"); return; }

        var allocator = new CudaAllocator(0);

        // Representative big FFN matmuls (E4B-ish): gate_up (K=2560,N=8192) + down (K=8192,N=2560).
        RunOne(allocator, K: 2560, N: 8192, label: "gate_up-like");
        RunOne(allocator, K: 8192, N: 2560, label: "down-like");
    }

    private void RunOne(CudaAllocator allocator, int K, int N, string label)
    {
        int blocks = K / 32;
        long rowBytes = (long)blocks * 34;               // Q8_0: 32 elems -> 2B f16 scale + 32 int8
        long rawBytes = (long)N * rowBytes;

        // Host Q8_0 weight; values are irrelevant for a timing profile (the kernel does
        // the same work regardless), so fill with deterministic bytes.
        IntPtr hostData = Marshal.AllocHGlobal((nint)rawBytes);
        try
        {
            unsafe
            {
                byte* p = (byte*)hostData;
                var rng = new Random(1234);
                for (long i = 0; i < rawBytes; i++) p[i] = (byte)rng.Next(256);
            }

            _output.WriteLine($"[q8-harness] {label}: K={K} N={N} weightMB={rawBytes / 1048576.0:F0}");
            _output.WriteLine($"[q8-harness]   M    dp4a ms   mma ms    speedup(dp4a/mma)");
            foreach (int M in new[] { 1, 2, 4, 8, 16, 32, 64 })
            {
                using var input = new Tensor(allocator, DType.Float32, M, K);
                Ops.Fill(input, 0.01f);
                using var result = new Tensor(allocator, DType.Float32, M, N);

                double Time(int q8Kernel)
                {
                    for (int w = 0; w < 5; w++)
                        CudaQuantizedOps.TryAddmmQuantizedToFloat32(result, input, hostData, hostData, 8, K, N, rawBytes, q8Kernel);
                    _ = result.GetElementsAsFloat(1);
                    const int iters = 50;
                    var sw = Stopwatch.StartNew();
                    for (int it = 0; it < iters; it++)
                        CudaQuantizedOps.TryAddmmQuantizedToFloat32(result, input, hostData, hostData, 8, K, N, rawBytes, q8Kernel);
                    _ = result.GetElementsAsFloat(1);
                    sw.Stop();
                    return sw.Elapsed.TotalMilliseconds / iters;
                }

                double dp4aMs = M >= 2 ? Time(1) : Time(3);   // M=1 has no multi-row path; use scalar
                double mmaMs = M >= 2 ? Time(2) : dp4aMs;
                _output.WriteLine($"[q8-harness]  {M,3}   {dp4aMs,7:F3}   {mmaMs,7:F3}    {(mmaMs > 0 ? dp4aMs / mmaMs : 1),5:F2}x");
            }
            _output.WriteLine("[q8-harness]  => if ms/call is ~flat for M<=8, the matmul is BANDWIDTH-bound there");
            _output.WriteLine("[q8-harness]     (tensor-core MMA accelerates compute -> won't help the verify);");
            _output.WriteLine("[q8-harness]     if it grows ~linearly with M, it's COMPUTE-bound -> MMA would help.");
        }
        finally
        {
            CudaQuantizedOps.ReleaseQuantizedWeight(allocator, hostData);
            Marshal.FreeHGlobal(hostData);
        }
    }
}
