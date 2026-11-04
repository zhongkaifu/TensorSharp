// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    internal sealed unsafe partial class Q35Kernels
    {
        internal const int NinferVerify8GemmSharedBytes = 43520;
        internal const string NinferVerify8GemmPolicy = "bn8-wm8wn1-stages2-k256-legacy-f32-split-v1";
        private IntPtr _ninferVerify8Gemm;
        private bool _ninferVerify8GemmConfigured, _gemmWorkSubmitted;
        internal bool NinferVerify8GemmEnabled => _ninferVerify8Gemm != IntPtr.Zero;
        internal string ImageSha256 => module.ImageSha256;
        /// <summary>Host calls, including capture; GPU graph replays do not increment this count.</summary>
        internal long NinferVerify8GemmHostSubmissions { get; private set; }

        internal static bool UseNinferVerify8Gemm(bool enabled, bool completeVerify8, bool swiglu,
            int rows, int outputRows, int columns)
            => enabled && completeVerify8 && !swiglu && rows == 8 &&
                ((columns == 5120 && outputRows is 16384 or 34816 or 14336)
                 || (outputRows == 5120 && columns is 17408 or 6144));

        internal static void ValidateNinferVerify8GemmConfiguration(bool configured, bool submitted)
        {
            if (configured || submitted)
                throw new InvalidOperationException("NInfer Verify8 GEMM must be configured once before any GEMM or graph capture.");
        }

        internal bool ConfigureNinferVerify8Gemm(bool required)
        {
            ValidateNinferVerify8GemmConfiguration(_ninferVerify8GemmConfigured, _gemmWorkSubmitted);
            _ninferVerify8GemmConfigured = true;
            IntPtr candidate;
            try { candidate = module.GetFunction("ts_q35_gemm_w4a4_8"); }
            catch (CudaException error) when (error.ErrorCode == 500)
            {
                if (required)
                    throw new NotSupportedException("TS_Q35_NINFER_VERIFY8_GEMM=1 requires a packaged main CUDA image containing ts_q35_gemm_w4a4_8.", error);
                return false;
            }
            CudaDriverApi.cuFuncLoad(candidate).ThrowOnError();
            ulong[] offsets = { 0, 8, 16, 20, 24, 32, 40, 48, 52, 56, 64, 72 };
            ulong[] sizes = { 8, 8, 4, 4, 8, 8, 8, 4, 4, 8, 8, 8 };
            for (uint i = 0; i < offsets.Length; i++)
            {
                CudaDriverApi.cuFuncGetParamInfo(candidate, (UIntPtr)i, out var offset, out var size).ThrowOnError();
                if (offset.ToUInt64() != offsets[i] || size.ToUInt64() != sizes[i])
                    throw new InvalidOperationException("NInfer Verify8 GEMM ABI differs.");
            }
            if (CudaDriverApi.cuFuncGetParamInfo(candidate, (UIntPtr)12, out _, out _) != 1)
                throw new InvalidOperationException("Unexpected NInfer Verify8 GEMM argument.");
            if (string.IsNullOrWhiteSpace(module.ImageSha256))
                throw new InvalidOperationException("NInfer Verify8 GEMM requires the actual loaded module identity.");
            CudaDriverApi.cuFuncSetAttribute(candidate, CudaDriverApi.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                NinferVerify8GemmSharedBytes).ThrowOnError();
            _ninferVerify8Gemm = candidate;
            return true;
        }
    }
}
