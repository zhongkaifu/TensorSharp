// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda;

internal sealed unsafe partial class Q35Kernels
{
    internal const string NinferGateUpSwiGluPolicy = "t8-b1-bn8-gateup-silu-legacy-f32-bf16-v1";
    private IntPtr _ninferGateUpSwiGlu;
    private bool _ninferGateUpSwiGluConfigured;
    internal bool NinferGateUpSwiGluEnabled => _ninferGateUpSwiGlu != IntPtr.Zero;
    internal long NinferGateUpSwiGluHostSubmissions { get; private set; }

    internal bool ConfigureNinferGateUpSwiGlu(bool required)
    {
        ValidateNinferVerify8GemmConfiguration(_ninferGateUpSwiGluConfigured, _gemmWorkSubmitted);
        _ninferGateUpSwiGluConfigured = true;
        if (!NinferVerify8GemmEnabled)
            throw new NotSupportedException("NInfer GateUp/SwiGLU requires the existing Verify8 GEMM policy.");
        IntPtr candidate;
        try { candidate = module.GetFunction("ts_q35_gemm_w4a4_8_swiglu_bf16"); }
        catch (CudaException error) when (error.ErrorCode == 500)
        {
            if (required)
                throw new NotSupportedException("TS_Q35_NINFER_GATEUP_SWIGLU=1 requires the packaged fused GateUp/SwiGLU CUDA entry.", error);
            return false;
        }
        ValidateNinferResidualKernel(candidate,
            new ulong[] { 0, 8, 16, 20, 24, 32, 40, 48, 52, 56, 64, 72 },
            new ulong[] { 8, 8, 4, 4, 8, 8, 8, 4, 4, 8, 8, 8 });
        if (string.IsNullOrWhiteSpace(ImageSha256))
            throw new InvalidOperationException("Loaded GateUp/SwiGLU image identity is required.");
        CudaDriverApi.cuFuncSetAttribute(candidate, CudaDriverApi.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            NinferVerify8GemmSharedBytes).ThrowOnError();
        _ninferGateUpSwiGlu = candidate;
        return true;
    }

    internal void GemmNinferGateUpSwiGlu(IntPtr weights, IntPtr scales, IntPtr codes, IntPtr inputScales,
        IntPtr rowScales, float alpha, IntPtr activation, IntPtr partial, IntPtr counters, IntPtr stream)
    {
        if (!NinferGateUpSwiGluEnabled) throw new InvalidOperationException("GateUp/SwiGLU was not configured.");
        _gemmWorkSubmitted = true;
        int n = 34816, k = 5120, rows = 8;
        void** args = stackalloc void*[] { &weights, &scales, &n, &k, &codes, &inputScales,
            &rowScales, &rows, &alpha, &activation, &partial, &counters };
        // GateUp already has 272 independent tiles: its original split resolver
        // selects one K partition on this qualified 170-SM profile.
        Launch(_ninferGateUpSwiGlu, 272, 1, 1, 256, NinferVerify8GemmSharedBytes, stream, args);
        NinferGateUpSwiGluHostSubmissions++;
    }
}
