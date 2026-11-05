// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda;

internal sealed unsafe partial class Q35Kernels
{
    internal const string NinferResidualNormPolicy = "t8-b1-nv4-residual-bf16-plainnorm-legacy-order-v1";
    private IntPtr _ninferResidualGemm, _ninferPlainNorm;
    private bool _ninferResidualNormConfigured;
    internal bool NinferResidualNormEnabled => _ninferResidualGemm != IntPtr.Zero && _ninferPlainNorm != IntPtr.Zero;
    internal static bool NinferResidualNormSplitKEnabled => SplitK;
    internal long NinferResidualGemmHostSubmissions { get; private set; }
    internal long NinferPlainNormHostSubmissions { get; private set; }

    internal static void ValidateNinferResidualNormConfiguration(bool configured, bool workSubmitted)
    {
        if (configured || workSubmitted)
            throw new InvalidOperationException("NInfer residual/norm must be configured once before any GEMM or graph capture.");
    }

    internal static bool NinferResidualProjectionShape(int rows, int outputRows, int columns)
        => rows == 8 && outputRows == 5120 && columns is 6144 or 17408;

    internal static bool TryResolveNinferResidualNormEntries(bool required, Func<string, IntPtr> resolve,
        out IntPtr fused, out IntPtr plain)
    {
        fused = plain = IntPtr.Zero;
        IntPtr candidateFused, candidatePlain;
        try
        {
            candidateFused = resolve("ts_q35_gemm_w4a4_8_residual_bf16");
            candidatePlain = resolve("ts_q35_ninfer_plain_norm_f32");
        }
        catch (CudaException error) when (error.ErrorCode == 500)
        {
            if (required)
                throw new NotSupportedException("TS_Q35_NINFER_RESIDUAL_NORM=1 requires the packaged main CUDA image containing both residual/norm entries.", error);
            return false;
        }
        fused = candidateFused;
        plain = candidatePlain;
        return true;
    }

    internal bool ConfigureNinferResidualNorm(bool required)
    {
        ValidateNinferResidualNormConfiguration(_ninferResidualNormConfigured, _gemmWorkSubmitted);
        _ninferResidualNormConfigured = true;
        if (!NinferVerify8GemmEnabled || !SplitK)
            throw new NotSupportedException("NInfer residual/norm requires the existing Verify8 GEMM and its split-K policy.");
        if (!TryResolveNinferResidualNormEntries(required, module.GetFunction, out var fused, out var plain)) return false;
        ValidateNinferResidualKernel(fused,
            new ulong[] { 0, 8, 16, 20, 24, 32, 40, 48, 52, 56, 64, 72, 80 },
            new ulong[] { 8, 8, 4, 4, 8, 8, 8, 4, 4, 8, 8, 8, 8 });
        ValidateNinferResidualKernel(plain, new ulong[] { 0, 8, 16, 24, 28 }, new ulong[] { 8, 8, 8, 4, 4 });
        if (string.IsNullOrWhiteSpace(ImageSha256)) throw new InvalidOperationException("Loaded residual/norm image identity is required.");
        CudaDriverApi.cuFuncSetAttribute(fused, CudaDriverApi.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            NinferVerify8GemmSharedBytes).ThrowOnError();
        _ninferResidualGemm = fused;
        _ninferPlainNorm = plain;
        return true;
    }

    private static void ValidateNinferResidualKernel(IntPtr function, ulong[] offsets, ulong[] sizes)
    {
        CudaDriverApi.cuFuncLoad(function).ThrowOnError();
        for (uint i = 0; i < offsets.Length; i++)
        {
            CudaDriverApi.cuFuncGetParamInfo(function, (UIntPtr)i, out var offset, out var size).ThrowOnError();
            if (offset.ToUInt64() != offsets[i] || size.ToUInt64() != sizes[i])
                throw new InvalidOperationException("NInfer residual/norm kernel ABI differs.");
        }
        if (CudaDriverApi.cuFuncGetParamInfo(function, (UIntPtr)(uint)offsets.Length, out _, out _) != 1)
            throw new InvalidOperationException("Unexpected NInfer residual/norm kernel argument.");
    }

    internal void GemmNinferResidual(IntPtr wCodes, IntPtr wScales, int outputRows, int columns,
        IntPtr xCodes, IntPtr xScales, IntPtr rowScale, int rows, float alpha, IntPtr block,
        IntPtr partial, IntPtr counters, IntPtr residual, IntPtr stream)
    {
        if (!NinferResidualNormEnabled || !SplitK || !NinferResidualProjectionShape(rows, outputRows, columns)
            || residual == IntPtr.Zero || residual == block)
            throw new InvalidOperationException("Unqualified NInfer fused residual projection.");
        _gemmWorkSubmitted = true;
        void** args = stackalloc void*[] { &wCodes, &wScales, &outputRows, &columns, &xCodes, &xScales,
            &rowScale, &rows, &alpha, &block, &partial, &counters, &residual };
        // Exactly the admitted legacy planner result for both fixed shapes on170SM.
        Launch(_ninferResidualGemm, 40, 1, 8, 256, NinferVerify8GemmSharedBytes, stream, args);
        NinferResidualGemmHostSubmissions++;
    }

    internal void PlainNinferNorm(IntPtr x, IntPtr weight, IntPtr output, int rows, int width, float eps, IntPtr stream)
    {
        if (!NinferResidualNormEnabled || rows != 8 || width != 5120 || x == IntPtr.Zero || weight == IntPtr.Zero
            || output == IntPtr.Zero || x == output || weight == output || weight == x || !(eps > 0) || !float.IsFinite(eps))
            throw new InvalidOperationException("Unqualified NInfer readonly residual norm.");
        _gemmWorkSubmitted = true;
        void** args = stackalloc void*[] { &x, &weight, &output, &width, &eps };
        Launch(_ninferPlainNorm, 8, 1, 1, 256, 0, stream, args);
        NinferPlainNormHostSubmissions++;
    }
}
