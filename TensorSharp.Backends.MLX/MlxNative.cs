using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

// Structs with bool fields (MlxOptional*) cross by value; with runtime
// marshalling disabled they pass with their managed layout, which matches
// the mlx-c definitions (1-byte bool).
[assembly: DisableRuntimeMarshalling]

namespace TensorSharp.MLX
{
    internal static partial class MlxNative
    {
        private const string LibraryName = "mlxc";
        private const int MlxGpu = 1;
        private static readonly object initSync = new();
        private static readonly object errorSync = new();
        private static readonly MlxErrorHandler ErrorHandler = CaptureError;
        private static readonly object fastKernelSync = new();
        private static bool resolverInstalled;
        private static bool errorHandlerInstalled;
        private static int initializedDevice = -1;
        private static bool cacheLimitConfigured;
        private static MlxStream cachedDefaultStream;
        private static int cachedDefaultStreamDevice = -1;
        private static string lastError = string.Empty;
        // A/B benchmark toggles. Defaults are the optimized paths; set the
        // env var to "1" to fall back to the pre-optimization behavior so
        // you can measure each change independently with the same binary.
        private static readonly bool DisableStreamCache =
            string.Equals(Environment.GetEnvironmentVariable("TS_MLX_BASELINE_STREAM"), "1", StringComparison.Ordinal);
        private static readonly bool DisableFreeDispatch =
            string.Equals(Environment.GetEnvironmentVariable("TS_MLX_BASELINE_FREE"), "1", StringComparison.Ordinal);
        private static MlxFastMetalKernel iq4XsMatmulKernel;
        private static MlxFastMetalKernel iq4XsMatmulSimdgroupKernel;
        private static MlxFastMetalKernel iq4XsMatmul4Kernel;
        private static MlxFastMetalKernel iq4XsMatmul4SimdKernel;
        private static MlxFastMetalKernel iq4XsMatmulRowsKernel;
        private static MlxFastMetalKernel iq4XsMatmulRows2Kernel;
        private static MlxFastMetalKernel iq4XsGetRowsKernel;
        // IQ4_NL ("4-bit non-linear") matmul. Block layout per ggml-common.h:
        //   struct block_iq4_nl { ggml_half d; uint8_t qs[QK4_NL/2]; }
        //   QK4_NL = 32, sizeof(block) = 2 + 16 = 18 bytes.
        // Dequantisation (see dequantize_row_iq4_nl in ggml-quants.c):
        //   value[j]       = d * kIq4NlValues[qs[j] & 0x0f], j in 0..15
        //   value[j + 16]  = d * kIq4NlValues[qs[j] >> 4]  , j in 0..15
        // The Unsloth UD-IQ2_XXS Nemotron-H pack stores all MoE expert
        // weights (~15.7 GB, 76% of token decode time) as IQ4_NL but MLX
        // had no native kernel for this type — they fell through to the
        // C# `ManagedQuantizedOps` matmul path on the CPU, which is the
        // dominant cost. This kernel mirrors `Iq4XsMatmulSource` but with
        // the 18-byte / 32-element IQ4_NL layout.
        private static MlxFastMetalKernel iq4NlMatmulKernel;
        private static MlxFastMetalKernel iq4NlMatmulRowsKernel;
        private static MlxFastMetalKernel iq4NlMoeMatmulBatchedKernel;
        private static MlxFastMetalKernel iq4NlMoeMatmulBatchedRowedKernel;
        private static MlxFastMetalKernel iq2XxsMatmulKernel;
        private static MlxFastMetalKernel iq2XxsMatmulSimdgroupKernel;
        private static MlxFastMetalKernel iq2XxsMoeMatmulBatchedKernel;
        private static MlxFastMetalKernel iq2XxsMoeMatmulBatchedFusedGateUpSiluKernel;
        private static MlxFastMetalKernel iq2XxsMoeMatmulBatchedRowedKernel;
        private static MlxFastMetalKernel iq2XxsGetRowsKernel;
        private static MlxFastMetalKernel iq2SMatmulKernel;
        private static MlxFastMetalKernel iq2SMatmulSimdgroupKernel;
        private static MlxFastMetalKernel iq2SGetRowsKernel;
        private static MlxFastMetalKernel iq3SMatmulKernel;
        private static MlxFastMetalKernel iq3SMatmulSimdgroupKernel;
        private static MlxFastMetalKernel iq3SGetRowsKernel;
        private static MlxFastMetalKernel iq3XxsMatmulKernel;
        private static MlxFastMetalKernel iq3XxsMatmulSimdgroupKernel;
        private static MlxFastMetalKernel iq3XxsGetRowsKernel;
        private static MlxFastMetalKernel q4KMatmulKernel;
        private static MlxFastMetalKernel q4KMatmulSimdgroupKernel;
        private static MlxFastMetalKernel q4KGetRowsKernel;
        private static MlxFastMetalKernel q5KMatmulKernel;
        private static MlxFastMetalKernel q5KMatmulSimdgroupKernel;
        private static MlxFastMetalKernel q5KMatmul4Kernel;
        private static MlxFastMetalKernel q5KGetRowsKernel;
        private static MlxFastMetalKernel q6KMatmulKernel;
        private static MlxFastMetalKernel q6KMatmulSimdgroupKernel;
        private static MlxFastMetalKernel q6KMatmul4Kernel;
        private static MlxFastMetalKernel q6KMatvecKernel;
        private static MlxFastMetalKernel iq4XsMatvecKernel;
        private static MlxFastMetalKernel q6KDequantF16Kernel;
        private static MlxFastMetalKernel iq4XsDequantF16Kernel;
        private static MlxFastMetalKernel q6KGetRowsKernel;
        private static MlxFastMetalKernel gatedDeltaKernel;
        private static MlxFastMetalKernel gatedDeltaT1Kernel;
        private static MlxFastMetalKernel gatedDeltaBlockedKernel;
        private static readonly bool GatedDeltaBlockedEnabled =
            !string.Equals(Environment.GetEnvironmentVariable("TS_MLX_GDN_BLOCKED"), "0", StringComparison.Ordinal);
        private static MlxFastMetalKernel qwen35GdnPreprocessKernel;
        private static MlxFastMetalKernel qwen35GdnPackedPreprocessKernel;
        private static MlxFastMetalKernel qwen35GdnPostprocessKernel;
        private static MlxFastMetalKernel headDim256AttentionKernel;
        private static MlxFastMetalKernel scatterAddWeightedRowsKernel;
        private static MlxFastMetalKernel rmsNormAddKernel;
        private static MlxFastMetalKernel addRmsNormKernel;
        private static MlxFastMetalKernel geluMulSplitKernel;
        private static MlxFastMetalKernel flatToHeadFirstKernel;
        private static MlxFastMetalKernel neoXRopeKernel;
        private static MlxFastMetalKernel circularDecodeAttentionKernel;
        private static MlxFastMetalKernel decodeAttentionWithSinksKernel;
        private static MlxFastMetalKernel gemma4QkvPreprocessDecodeKernel;
        private static MlxFastMetalKernel q8AddmmAddKernel;
        private static MlxFastMetalKernel q8RmsNormMatmulKernel;
        private static MlxFastMetalKernel q8MatmulKernel;
        private static MlxFastMetalKernel decodeAttentionHeadDim512Kernel;
        private static MlxFastMetalKernel q8MatmulGeluMulKernel;
        private static MlxFastMetalKernel swigluOaiGatherBiasKernel;
        private static MlxFastMetalKernel moeBiasWeightedSumKernel;
        private static bool iq4XsMatmulKernelDisabled;
        private static bool iq4XsMatmulSimdgroupKernelDisabled;
        private static bool iq4XsMatmul4KernelDisabled;
        private static bool iq4XsMatmul4SimdKernelDisabled;
        private static bool iq4XsMatmulRowsKernelDisabled;
        private static bool iq4XsMatmulRows2KernelDisabled;
        private static bool iq4XsGetRowsKernelDisabled;
        private static bool iq4NlMatmulKernelDisabled;
        private static bool iq4NlMatmulRowsKernelDisabled;
        private static bool iq4NlMoeMatmulBatchedKernelDisabled;
        private static bool iq4NlMoeMatmulBatchedRowedKernelDisabled;
        private static bool iq2XxsMatmulKernelDisabled;
        private static bool iq2XxsMatmulSimdgroupKernelDisabled;
        private static bool iq2XxsMoeMatmulBatchedKernelDisabled;
        private static bool iq2XxsMoeMatmulBatchedFusedGateUpSiluKernelDisabled;
        private static bool iq2XxsMoeMatmulBatchedRowedKernelDisabled;
        private static bool iq2XxsGetRowsKernelDisabled;
        private static bool iq2SMatmulKernelDisabled;
        private static bool iq2SMatmulSimdgroupKernelDisabled;
        private static bool iq2SGetRowsKernelDisabled;
        private static bool iq3SMatmulKernelDisabled;
        private static bool iq3SMatmulSimdgroupKernelDisabled;
        private static bool iq3SGetRowsKernelDisabled;
        private static bool iq3XxsMatmulKernelDisabled;
        private static bool iq3XxsMatmulSimdgroupKernelDisabled;
        private static bool iq3XxsGetRowsKernelDisabled;
        private static bool q4KMatmulKernelDisabled;
        private static bool q4KMatmulSimdgroupKernelDisabled;
        private static bool q4KGetRowsKernelDisabled;
        private static bool q5KMatmulKernelDisabled;
        private static bool q5KMatmulSimdgroupKernelDisabled;
        private static bool q5KMatmul4KernelDisabled;
        private static bool q5KGetRowsKernelDisabled;
        private static bool q6KMatmulKernelDisabled;
        private static bool q6KMatmulSimdgroupKernelDisabled;
        private static bool q6KMatmul4KernelDisabled;
        private static bool q6KMatvecKernelDisabled;
        private static bool iq4XsMatvecKernelDisabled;
        private static bool q6KDequantF16KernelDisabled;
        private static bool iq4XsDequantF16KernelDisabled;
        private static bool q6KGetRowsKernelDisabled;
        private static bool gatedDeltaKernelDisabled;
        private static bool gatedDeltaBlockedKernelDisabled;
        private static bool gatedDeltaT1KernelDisabled;
        private static bool qwen35GdnPreprocessKernelDisabled;
        private static bool qwen35GdnPackedPreprocessKernelDisabled;
        private static bool qwen35GdnPostprocessKernelDisabled;
        private static bool headDim256AttentionKernelDisabled;
        private static bool scatterAddWeightedRowsKernelDisabled;
        private static bool rmsNormAddKernelDisabled;
        private static bool addRmsNormKernelDisabled;
        private static bool geluMulSplitKernelDisabled;
        private static bool flatToHeadFirstKernelDisabled;
        private static bool neoXRopeKernelDisabled;
        private static bool circularDecodeAttentionKernelDisabled;
        private static bool decodeAttentionWithSinksKernelDisabled;
        private static bool gemma4QkvPreprocessDecodeKernelDisabled;
        private static bool q8AddmmAddKernelDisabled;
        private static bool q8RmsNormMatmulKernelDisabled;
        private static bool q8MatmulKernelDisabled;
        private static bool decodeAttentionHeadDim512KernelDisabled;
        private static bool q8MatmulGeluMulKernelDisabled;
        private static bool swigluOaiGatherBiasKernelDisabled;
        private static bool moeBiasWeightedSumKernelDisabled;

        private const string Iq4NlLookupHeader = @"
constexpr constant float kIq4NlValues[16] = {
    -127.0f, -104.0f, -83.0f, -65.0f,
    -49.0f,  -35.0f, -22.0f, -10.0f,
       1.0f,   13.0f,  25.0f,  38.0f,
      53.0f,   69.0f,  89.0f, 113.0f,
};
";

        private const string Iq4XsHelpersHeader = Iq4NlLookupHeader + @"
inline float tensorsharp_dequant_iq4xs(const device uchar * block, int within_block) {
    const int ib32 = within_block >> 5;
    const int within_32 = within_block & 31;

    const half d_half = *reinterpret_cast<const device half *>(block);
    const ushort scales_h =
        static_cast<ushort>(block[2]) |
        (static_cast<ushort>(block[3]) << 8);
    const uchar scale_l_byte = block[4 + (ib32 >> 1)];
    const int ls =
        ((scale_l_byte >> (4 * (ib32 & 1))) & 0x0f) |
        (((scales_h >> (2 * ib32)) & 0x03) << 4);

    const uchar packed = block[8 + ib32 * 16 + (within_32 & 15)];
    const uchar q = within_32 < 16 ? (packed & 0x0f) : (packed >> 4);
    return static_cast<float>(d_half) * static_cast<float>(ls - 32) * kIq4NlValues[q];
}
";

        private const string Iq2XxsLookupHeader = @"
constexpr constant ulong kIq2XxsGrid[256] = {
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x08080808082b0808,
    0x08080808082b082b, 0x08080808082b2b08, 0x08080808082b2b2b, 0x0808080819080819,
    0x0808080819081908, 0x0808080819190808, 0x0808080819192b08, 0x08080808192b0819,
    0x08080808192b1908, 0x080808082b080808, 0x080808082b08082b, 0x080808082b082b2b,
    0x080808082b2b082b, 0x0808081908080819, 0x0808081908081908, 0x0808081908190808,
    0x0808081908191919, 0x0808081919080808, 0x080808192b081908, 0x080808192b192b08,
    0x0808082b08080808, 0x0808082b0808082b, 0x0808082b082b082b, 0x0808082b2b08082b,
    0x0808190808080819, 0x0808190808081908, 0x0808190808190808, 0x08081908082b0819,
    0x08081908082b1908, 0x0808190819080808, 0x080819081908082b, 0x0808190819082b08,
    0x08081908192b0808, 0x080819082b080819, 0x080819082b081908, 0x080819082b190808,
    0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b, 0x0808191908082b08,
    0x08081919082b0808, 0x080819191908192b, 0x08081919192b2b19, 0x080819192b080808,
    0x080819192b190819, 0x0808192b08082b19, 0x0808192b08190808, 0x0808192b19080808,
    0x0808192b2b081908, 0x0808192b2b2b1908, 0x08082b0808080808, 0x08082b0808081919,
    0x08082b0808082b08, 0x08082b0808191908, 0x08082b08082b2b08, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b081919082b, 0x08082b082b082b08,
    0x08082b1908081908, 0x08082b1919080808, 0x08082b2b0808082b, 0x08082b2b08191908,
    0x0819080808080819, 0x0819080808081908, 0x0819080808190808, 0x08190808082b0819,
    0x0819080819080808, 0x08190808192b0808, 0x081908082b081908, 0x081908082b190808,
    0x081908082b191919, 0x0819081908080808, 0x0819081908082b08, 0x08190819082b0808,
    0x0819081919190808, 0x0819081919192b2b, 0x081908192b080808, 0x0819082b082b1908,
    0x0819082b19081919, 0x0819190808080808, 0x0819190808082b08, 0x08191908082b0808,
    0x08191908082b1919, 0x0819190819082b19, 0x081919082b080808, 0x0819191908192b08,
    0x08191919192b082b, 0x0819192b08080808, 0x0819192b0819192b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b0808190808, 0x08192b0819080808, 0x08192b082b080819,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b192b2b0808, 0x08192b2b19190819,
    0x082b080808080808, 0x082b08080808082b, 0x082b080808082b2b, 0x082b080819081908,
    0x082b0808192b0819, 0x082b08082b080808, 0x082b08082b08082b, 0x082b0819082b2b19,
    0x082b081919082b08, 0x082b082b08080808, 0x082b082b0808082b, 0x082b190808080819,
    0x082b190808081908, 0x082b190808190808, 0x082b190819080808, 0x082b19081919192b,
    0x082b191908080808, 0x082b191919080819, 0x082b1919192b1908, 0x082b192b2b190808,
    0x082b2b0808082b08, 0x082b2b08082b0808, 0x082b2b082b191908, 0x082b2b2b19081908,
    0x1908080808080819, 0x1908080808081908, 0x1908080808190808, 0x1908080808192b08,
    0x19080808082b0819, 0x19080808082b1908, 0x1908080819080808, 0x1908080819082b08,
    0x190808081919192b, 0x19080808192b0808, 0x190808082b080819, 0x190808082b081908,
    0x190808082b190808, 0x1908081908080808, 0x19080819082b0808, 0x19080819192b0819,
    0x190808192b080808, 0x190808192b081919, 0x1908082b08080819, 0x1908082b08190808,
    0x1908082b19082b08, 0x1908082b1919192b, 0x1908082b192b2b08, 0x1908190808080808,
    0x1908190808082b08, 0x19081908082b0808, 0x190819082b080808, 0x190819082b192b19,
    0x190819190819082b, 0x19081919082b1908, 0x1908192b08080808, 0x19082b0808080819,
    0x19082b0808081908, 0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919,
    0x19082b1908080808, 0x19082b1919192b08, 0x19082b19192b0819, 0x19082b192b08082b,
    0x19082b2b19081919, 0x19082b2b2b190808, 0x1919080808080808, 0x1919080808082b08,
    0x1919080808190819, 0x1919080808192b19, 0x19190808082b0808, 0x191908082b080808,
    0x191908082b082b08, 0x1919081908081908, 0x191908191908082b, 0x191908192b2b1908,
    0x1919082b2b190819, 0x191919082b190808, 0x191919082b19082b, 0x1919191908082b2b,
    0x1919192b08080819, 0x1919192b19191908, 0x19192b0808080808, 0x19192b0808190819,
    0x19192b0808192b19, 0x19192b08192b1908, 0x19192b1919080808, 0x19192b2b08082b08,
    0x192b080808081908, 0x192b080808190808, 0x192b080819080808, 0x192b0808192b2b08,
    0x192b081908080808, 0x192b081919191919, 0x192b082b08192b08, 0x192b082b192b0808,
    0x192b190808080808, 0x192b190808081919, 0x192b191908190808, 0x192b19190819082b,
    0x192b19192b081908, 0x192b2b081908082b, 0x2b08080808080808, 0x2b0808080808082b,
    0x2b08080808082b2b, 0x2b08080819080819, 0x2b0808082b08082b, 0x2b08081908081908,
    0x2b08081908192b08, 0x2b08081919080808, 0x2b08082b08190819, 0x2b08190808080819,
    0x2b08190808081908, 0x2b08190808190808, 0x2b08190808191919, 0x2b08190819080808,
    0x2b081908192b0808, 0x2b08191908080808, 0x2b0819191908192b, 0x2b0819192b191908,
    0x2b08192b08082b19, 0x2b08192b19080808, 0x2b08192b192b0808, 0x2b082b080808082b,
    0x2b082b1908081908, 0x2b082b2b08190819, 0x2b19080808081908, 0x2b19080808190808,
    0x2b190808082b1908, 0x2b19080819080808, 0x2b1908082b2b0819, 0x2b1908190819192b,
    0x2b1908192b080808, 0x2b19082b19081919, 0x2b19190808080808, 0x2b191908082b082b,
    0x2b19190819081908, 0x2b19191919190819, 0x2b192b082b080819, 0x2b192b19082b0808,
    0x2b2b08080808082b, 0x2b2b080819190808, 0x2b2b08082b081919, 0x2b2b081908082b19,
    0x2b2b082b08080808, 0x2b2b190808192b08, 0x2b2b2b0819190808, 0x2b2b2b1908081908,
};

inline uint tensorsharp_iq2_signs(uint index) {
    return index | ((popcount(index) & 1) << 7);
}

inline float tensorsharp_dequant_iq2_xxs(const device uchar * block, int within_block) {
    const half d_half = *reinterpret_cast<const device half *>(block);
    const device ushort * qs = reinterpret_cast<const device ushort *>(block + 2);
    const int ib32 = within_block >> 5;
    const int within_32 = within_block & 31;
    const int lane = within_32 >> 3;
    const int j = within_32 & 7;

    const uint aux_g =
        static_cast<uint>(qs[4 * ib32 + 0]) |
        (static_cast<uint>(qs[4 * ib32 + 1]) << 16);
    const uint aux_s =
        static_cast<uint>(qs[4 * ib32 + 2]) |
        (static_cast<uint>(qs[4 * ib32 + 3]) << 16);
    const uint grid_index = (aux_g >> (8 * lane)) & 255;
    const uint sign_index = (aux_s >> (7 * lane)) & 127;
    const uint signs = tensorsharp_iq2_signs(sign_index);
    const float db = static_cast<float>(d_half) * (0.5f + static_cast<float>(aux_s >> 28)) * 0.25f;
    const ulong packed_grid = kIq2XxsGrid[grid_index];
    const uint grid = static_cast<uint>((packed_grid >> (8 * j)) & 255);
    return db * static_cast<float>(grid) * ((signs & (1u << j)) != 0 ? -1.0f : 1.0f);
}

// Block-amortized dot product: 8 CONSECUTIVE weights starting at
// within_block (a multiple of 8) share one grid codebook entry, one sign
// byte and one scale, so the header loads and the 256-entry table lookup
// happen ONCE for the group instead of once per weight.
//
// The per-element helper above costs ~5 dependent loads + 1 codebook lookup
// for every single weight value, which is what made the decode matmul run at
// ~35 GB/s against a ~205 GB/s roofline (ggml-metal's kernels amortize the
// same way, over a whole 32-value sub-block). Folding the block scale out of
// the inner loop re-associates the sum - the result is not bit-identical to
// the per-element form, but it is the same arithmetic ggml does, and the
// simd_sum reduction that follows already makes bit-exactness order-dependent.
inline float tensorsharp_dot8_iq2_xxs(const device uchar * block, int within_block, thread const float * xv) {
    const half d_half = *reinterpret_cast<const device half *>(block);
    const device ushort * qs = reinterpret_cast<const device ushort *>(block + 2);
    const int ib32 = within_block >> 5;
    const int lane = (within_block >> 3) & 3;

    const uint aux_g =
        static_cast<uint>(qs[4 * ib32 + 0]) |
        (static_cast<uint>(qs[4 * ib32 + 1]) << 16);
    const uint aux_s =
        static_cast<uint>(qs[4 * ib32 + 2]) |
        (static_cast<uint>(qs[4 * ib32 + 3]) << 16);
    const uint grid_index = (aux_g >> (8 * lane)) & 255;
    const uint signs = tensorsharp_iq2_signs((aux_s >> (7 * lane)) & 127);
    const float db = static_cast<float>(d_half) * (0.5f + static_cast<float>(aux_s >> 28)) * 0.25f;
    const ulong packed_grid = kIq2XxsGrid[grid_index];

    float acc = 0.0f;
    for (int j = 0; j < 8; ++j) {
        const float g = static_cast<float>(static_cast<uint>((packed_grid >> (8 * j)) & 255));
        acc += xv[j] * (((signs >> j) & 1u) != 0u ? -g : g);
    }
    return db * acc;
}
";

        // IQ3_XXS (3.0625 bpw). Reuses the IQ2_XXS sign helper
        // (tensorsharp_iq2_signs == ggml's ksigns_iq2xs) and adds the distinct
        // 256-entry iq3xxs_grid codebook, where each entry packs FOUR 8-bit
        // magnitudes (vs IQ2_XXS's eight). Table copied verbatim from
        // ggml-common.h / TensorSharp.Models.IQuantGrids.iq3xxs_grid.
        private const string Iq3XxsHelpersHeader = Iq2XxsLookupHeader + @"
constexpr constant uint kIq3XxsGrid[256] = {
    0x04040404, 0x04040414, 0x04040424, 0x04040c0c, 0x04040c1c, 0x04040c3e, 0x04041404, 0x04041414,
    0x04041c0c, 0x04042414, 0x04043e1c, 0x04043e2c, 0x040c040c, 0x040c041c, 0x040c0c04, 0x040c0c14,
    0x040c140c, 0x040c142c, 0x040c1c04, 0x040c1c14, 0x040c240c, 0x040c2c24, 0x040c3e04, 0x04140404,
    0x04140414, 0x04140424, 0x04140c0c, 0x04141404, 0x04141414, 0x04141c0c, 0x04141c1c, 0x04141c3e,
    0x04142c0c, 0x04142c3e, 0x04143e2c, 0x041c040c, 0x041c043e, 0x041c0c04, 0x041c0c14, 0x041c142c,
    0x041c3e04, 0x04240c1c, 0x04241c3e, 0x04242424, 0x04242c3e, 0x04243e1c, 0x04243e2c, 0x042c040c,
    0x042c043e, 0x042c1c14, 0x042c2c14, 0x04341c2c, 0x04343424, 0x043e0c04, 0x043e0c24, 0x043e0c34,
    0x043e241c, 0x043e340c, 0x0c04040c, 0x0c04041c, 0x0c040c04, 0x0c040c14, 0x0c04140c, 0x0c04141c,
    0x0c041c04, 0x0c041c14, 0x0c041c24, 0x0c04243e, 0x0c042c04, 0x0c0c0404, 0x0c0c0414, 0x0c0c0c0c,
    0x0c0c1404, 0x0c0c1414, 0x0c14040c, 0x0c14041c, 0x0c140c04, 0x0c140c14, 0x0c14140c, 0x0c141c04,
    0x0c143e14, 0x0c1c0404, 0x0c1c0414, 0x0c1c1404, 0x0c1c1c0c, 0x0c1c2434, 0x0c1c3434, 0x0c24040c,
    0x0c24042c, 0x0c242c04, 0x0c2c1404, 0x0c2c1424, 0x0c2c2434, 0x0c2c3e0c, 0x0c34042c, 0x0c3e1414,
    0x0c3e2404, 0x14040404, 0x14040414, 0x14040c0c, 0x14040c1c, 0x14041404, 0x14041414, 0x14041434,
    0x14041c0c, 0x14042414, 0x140c040c, 0x140c041c, 0x140c042c, 0x140c0c04, 0x140c0c14, 0x140c140c,
    0x140c1c04, 0x140c341c, 0x140c343e, 0x140c3e04, 0x14140404, 0x14140414, 0x14140c0c, 0x14140c3e,
    0x14141404, 0x14141414, 0x14141c3e, 0x14142404, 0x14142c2c, 0x141c040c, 0x141c0c04, 0x141c0c24,
    0x141c3e04, 0x141c3e24, 0x14241c2c, 0x14242c1c, 0x142c041c, 0x142c143e, 0x142c240c, 0x142c3e24,
    0x143e040c, 0x143e041c, 0x143e0c34, 0x143e242c, 0x1c04040c, 0x1c040c04, 0x1c040c14, 0x1c04140c,
    0x1c04141c, 0x1c042c04, 0x1c04342c, 0x1c043e14, 0x1c0c0404, 0x1c0c0414, 0x1c0c1404, 0x1c0c1c0c,
    0x1c0c2424, 0x1c0c2434, 0x1c14040c, 0x1c14041c, 0x1c140c04, 0x1c14142c, 0x1c142c14, 0x1c143e14,
    0x1c1c0c0c, 0x1c1c1c1c, 0x1c241c04, 0x1c24243e, 0x1c243e14, 0x1c2c0404, 0x1c2c0434, 0x1c2c1414,
    0x1c2c2c2c, 0x1c340c24, 0x1c341c34, 0x1c34341c, 0x1c3e1c1c, 0x1c3e3404, 0x24040424, 0x24040c3e,
    0x24041c2c, 0x24041c3e, 0x24042c1c, 0x24042c3e, 0x240c3e24, 0x24141404, 0x24141c3e, 0x24142404,
    0x24143404, 0x24143434, 0x241c043e, 0x241c242c, 0x24240424, 0x24242c0c, 0x24243424, 0x242c142c,
    0x242c241c, 0x242c3e04, 0x243e042c, 0x243e0c04, 0x243e0c14, 0x243e1c04, 0x2c040c14, 0x2c04240c,
    0x2c043e04, 0x2c0c0404, 0x2c0c0434, 0x2c0c1434, 0x2c0c2c2c, 0x2c140c24, 0x2c141c14, 0x2c143e14,
    0x2c1c0414, 0x2c1c2c1c, 0x2c240c04, 0x2c24141c, 0x2c24143e, 0x2c243e14, 0x2c2c0414, 0x2c2c1c0c,
    0x2c342c04, 0x2c3e1424, 0x2c3e2414, 0x34041424, 0x34042424, 0x34042434, 0x34043424, 0x340c140c,
    0x340c340c, 0x34140c3e, 0x34143424, 0x341c1c04, 0x341c1c34, 0x34242424, 0x342c042c, 0x342c2c14,
    0x34341c1c, 0x343e041c, 0x343e140c, 0x3e04041c, 0x3e04042c, 0x3e04043e, 0x3e040c04, 0x3e041c14,
    0x3e042c14, 0x3e0c1434, 0x3e0c2404, 0x3e140c14, 0x3e14242c, 0x3e142c14, 0x3e1c0404, 0x3e1c0c2c,
    0x3e1c1c1c, 0x3e1c3404, 0x3e24140c, 0x3e24240c, 0x3e2c0404, 0x3e2c0414, 0x3e2c1424, 0x3e341c04,
};

// block_iq3_xxs: d (half, 2 bytes) | qs[3*QK_K/8] (96 bytes) = 98 bytes per
// 256-element super-block. qs[0..63] are grid indices (8 bytes per 32-element
// group, one index per 4 weights); qs[64..95] are eight little-endian uint32
// words, one per group, packing a 4-bit scale (top nibble) plus four 7-bit
// sign selectors. Ported from ggml dequantize_row_iq3_xxs — matches
// ManagedQuantizedOps.DequantizeIq3Xxs bit for bit.
inline float tensorsharp_dequant_iq3_xxs(const device uchar * block, int within_block) {
    const half d_half = *reinterpret_cast<const device half *>(block);
    const device uchar * qs = block + 2;
    const device uchar * scales_and_signs = qs + 64;
    const int ib32 = within_block >> 5;
    const int within_32 = within_block & 31;
    const int lane = within_32 >> 3;   // 0..3: which (grid pair, sign) tuple
    const int p = within_32 & 7;       // 0..7: position inside the 8-weight tuple

    // Byte-wise load: the super-block stride is 98, so scales_and_signs is only
    // 2-byte aligned and a uint reinterpret_cast would fault / misread.
    const device uchar * aux_bytes = scales_and_signs + 4 * ib32;
    const uint aux32 =
        static_cast<uint>(aux_bytes[0]) |
        (static_cast<uint>(aux_bytes[1]) << 8) |
        (static_cast<uint>(aux_bytes[2]) << 16) |
        (static_cast<uint>(aux_bytes[3]) << 24);

    const float db = static_cast<float>(d_half) * (0.5f + static_cast<float>(aux32 >> 28)) * 0.5f;
    const uint grid_index = static_cast<uint>(qs[8 * ib32 + 2 * lane + (p >> 2)]);
    const uint packed_grid = kIq3XxsGrid[grid_index];
    const uint grid = (packed_grid >> (8 * (p & 3))) & 255u;
    const uint signs = tensorsharp_iq2_signs((aux32 >> (7 * lane)) & 127u);
    return db * static_cast<float>(grid) * ((signs & (1u << p)) != 0 ? -1.0f : 1.0f);
}

// 8 consecutive IQ3_XXS weights: one aux32 (scale + signs) and TWO codebook
// entries (each iq3xxs_grid entry packs four 8-bit magnitudes), against ~4
// dependent loads plus a lookup per weight in the per-element helper.
inline float tensorsharp_dot8_iq3_xxs(const device uchar * block, int within_block, thread const float * xv) {
    const half d_half = *reinterpret_cast<const device half *>(block);
    const device uchar * qs = block + 2;
    const device uchar * scales_and_signs = qs + 64;
    const int ib32 = within_block >> 5;
    const int lane = (within_block >> 3) & 3;

    // Byte-wise load: the super-block stride is 98, so scales_and_signs is only
    // 2-byte aligned and a uint reinterpret_cast would fault / misread.
    const device uchar * aux_bytes = scales_and_signs + 4 * ib32;
    const uint aux32 =
        static_cast<uint>(aux_bytes[0]) |
        (static_cast<uint>(aux_bytes[1]) << 8) |
        (static_cast<uint>(aux_bytes[2]) << 16) |
        (static_cast<uint>(aux_bytes[3]) << 24);

    const float db = static_cast<float>(d_half) * (0.5f + static_cast<float>(aux32 >> 28)) * 0.5f;
    const uint packed_lo = kIq3XxsGrid[static_cast<uint>(qs[8 * ib32 + 2 * lane + 0])];
    const uint packed_hi = kIq3XxsGrid[static_cast<uint>(qs[8 * ib32 + 2 * lane + 1])];
    const uint signs = tensorsharp_iq2_signs((aux32 >> (7 * lane)) & 127u);

    float acc = 0.0f;
    for (int p = 0; p < 8; ++p) {
        const uint packed = p < 4 ? packed_lo : packed_hi;
        const float g = static_cast<float>((packed >> (8 * (p & 3))) & 255u);
        acc += xv[p] * (((signs >> p) & 1u) != 0u ? -g : g);
    }
    return db * acc;
}
";

        private const string Iq2SIq3SLookupHeader = @"
constexpr constant ulong kIq2SGrid[1024] = {
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x080808080819192b,
    0x0808080808192b19, 0x08080808082b0808, 0x08080808082b082b, 0x08080808082b1919,
    0x08080808082b2b08, 0x0808080819080819, 0x0808080819081908, 0x080808081908192b,
    0x0808080819082b19, 0x0808080819190808, 0x080808081919082b, 0x0808080819191919,
    0x0808080819192b08, 0x08080808192b0819, 0x08080808192b1908, 0x08080808192b192b,
    0x08080808192b2b19, 0x080808082b080808, 0x080808082b08082b, 0x080808082b081919,
    0x080808082b082b08, 0x080808082b190819, 0x080808082b191908, 0x080808082b2b0808,
    0x080808082b2b1919, 0x080808082b2b2b2b, 0x0808081908080819, 0x0808081908081908,
    0x080808190808192b, 0x0808081908082b19, 0x0808081908190808, 0x080808190819082b,
    0x0808081908191919, 0x0808081908192b08, 0x08080819082b0819, 0x08080819082b1908,
    0x0808081919080808, 0x080808191908082b, 0x0808081919081919, 0x0808081919082b08,
    0x0808081919190819, 0x0808081919191908, 0x080808191919192b, 0x0808081919192b19,
    0x08080819192b0808, 0x08080819192b1919, 0x08080819192b2b08, 0x080808192b080819,
    0x080808192b081908, 0x080808192b190808, 0x080808192b19082b, 0x080808192b191919,
    0x080808192b2b0819, 0x080808192b2b1908, 0x0808082b08080808, 0x0808082b0808082b,
    0x0808082b08081919, 0x0808082b08082b08, 0x0808082b08190819, 0x0808082b08191908,
    0x0808082b082b0808, 0x0808082b082b2b2b, 0x0808082b19080819, 0x0808082b19081908,
    0x0808082b1908192b, 0x0808082b19082b19, 0x0808082b19190808, 0x0808082b19191919,
    0x0808082b2b080808, 0x0808082b2b081919, 0x0808082b2b082b2b, 0x0808082b2b191908,
    0x0808082b2b2b082b, 0x0808190808080819, 0x0808190808081908, 0x080819080808192b,
    0x0808190808082b19, 0x0808190808190808, 0x080819080819082b, 0x0808190808191919,
    0x0808190808192b08, 0x08081908082b0819, 0x08081908082b1908, 0x08081908082b192b,
    0x08081908082b2b19, 0x0808190819080808, 0x080819081908082b, 0x0808190819081919,
    0x0808190819082b08, 0x0808190819082b2b, 0x0808190819190819, 0x0808190819191908,
    0x080819081919192b, 0x0808190819192b19, 0x08081908192b0808, 0x08081908192b082b,
    0x08081908192b1919, 0x080819082b080819, 0x080819082b081908, 0x080819082b08192b,
    0x080819082b082b19, 0x080819082b190808, 0x080819082b191919, 0x080819082b192b08,
    0x080819082b2b0819, 0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b,
    0x0808191908081919, 0x0808191908082b08, 0x0808191908082b2b, 0x0808191908190819,
    0x0808191908191908, 0x080819190819192b, 0x0808191908192b19, 0x08081919082b0808,
    0x08081919082b1919, 0x08081919082b2b08, 0x0808191919080819, 0x0808191919081908,
    0x080819191908192b, 0x0808191919082b19, 0x0808191919190808, 0x080819191919082b,
    0x0808191919191919, 0x0808191919192b08, 0x08081919192b0819, 0x08081919192b1908,
    0x080819192b080808, 0x080819192b08082b, 0x080819192b081919, 0x080819192b082b08,
    0x080819192b190819, 0x080819192b191908, 0x080819192b2b0808, 0x0808192b08080819,
    0x0808192b08081908, 0x0808192b0808192b, 0x0808192b08082b19, 0x0808192b08190808,
    0x0808192b08191919, 0x0808192b19080808, 0x0808192b19081919, 0x0808192b19082b08,
    0x0808192b19190819, 0x0808192b19191908, 0x0808192b192b0808, 0x0808192b2b080819,
    0x0808192b2b081908, 0x0808192b2b190808, 0x08082b0808080808, 0x08082b080808082b,
    0x08082b0808081919, 0x08082b0808082b08, 0x08082b0808190819, 0x08082b0808191908,
    0x08082b080819192b, 0x08082b0808192b19, 0x08082b08082b0808, 0x08082b08082b1919,
    0x08082b08082b2b2b, 0x08082b0819080819, 0x08082b0819081908, 0x08082b081908192b,
    0x08082b0819082b19, 0x08082b0819190808, 0x08082b081919082b, 0x08082b0819191919,
    0x08082b0819192b08, 0x08082b08192b0819, 0x08082b08192b1908, 0x08082b082b080808,
    0x08082b082b081919, 0x08082b082b191908, 0x08082b082b2b2b2b, 0x08082b1908080819,
    0x08082b1908081908, 0x08082b1908190808, 0x08082b190819082b, 0x08082b1908191919,
    0x08082b1908192b08, 0x08082b19082b0819, 0x08082b1919080808, 0x08082b1919081919,
    0x08082b1919082b08, 0x08082b1919190819, 0x08082b1919191908, 0x08082b19192b0808,
    0x08082b192b080819, 0x08082b192b190808, 0x08082b2b08080808, 0x08082b2b08190819,
    0x08082b2b08191908, 0x08082b2b082b082b, 0x08082b2b082b2b08, 0x08082b2b082b2b2b,
    0x08082b2b19190808, 0x08082b2b2b192b19, 0x0819080808080819, 0x0819080808081908,
    0x081908080808192b, 0x0819080808082b19, 0x0819080808190808, 0x081908080819082b,
    0x0819080808191919, 0x0819080808192b08, 0x08190808082b0819, 0x08190808082b1908,
    0x08190808082b192b, 0x0819080819080808, 0x081908081908082b, 0x0819080819081919,
    0x0819080819082b08, 0x0819080819190819, 0x0819080819191908, 0x081908081919192b,
    0x0819080819192b19, 0x08190808192b0808, 0x08190808192b082b, 0x08190808192b1919,
    0x08190808192b2b08, 0x081908082b080819, 0x081908082b081908, 0x081908082b08192b,
    0x081908082b190808, 0x081908082b191919, 0x081908082b192b08, 0x081908082b2b0819,
    0x081908082b2b1908, 0x0819081908080808, 0x081908190808082b, 0x0819081908081919,
    0x0819081908082b08, 0x0819081908082b2b, 0x0819081908190819, 0x0819081908191908,
    0x081908190819192b, 0x0819081908192b19, 0x08190819082b0808, 0x08190819082b082b,
    0x08190819082b1919, 0x08190819082b2b08, 0x0819081919080819, 0x0819081919081908,
    0x081908191908192b, 0x0819081919082b19, 0x0819081919190808, 0x081908191919082b,
    0x0819081919191919, 0x0819081919192b08, 0x08190819192b0819, 0x08190819192b1908,
    0x081908192b080808, 0x081908192b08082b, 0x081908192b081919, 0x081908192b082b08,
    0x081908192b190819, 0x081908192b191908, 0x0819082b08080819, 0x0819082b08081908,
    0x0819082b08082b19, 0x0819082b08190808, 0x0819082b08191919, 0x0819082b082b0819,
    0x0819082b082b1908, 0x0819082b19080808, 0x0819082b19081919, 0x0819082b19190819,
    0x0819082b19191908, 0x0819082b2b080819, 0x0819082b2b081908, 0x0819082b2b190808,
    0x0819190808080808, 0x081919080808082b, 0x0819190808081919, 0x0819190808082b08,
    0x0819190808190819, 0x0819190808191908, 0x081919080819192b, 0x0819190808192b19,
    0x08191908082b0808, 0x08191908082b1919, 0x08191908082b2b08, 0x0819190819080819,
    0x0819190819081908, 0x081919081908192b, 0x0819190819082b19, 0x0819190819190808,
    0x081919081919082b, 0x0819190819191919, 0x0819190819192b08, 0x08191908192b0819,
    0x08191908192b1908, 0x081919082b080808, 0x081919082b08082b, 0x081919082b081919,
    0x081919082b082b08, 0x081919082b190819, 0x081919082b191908, 0x081919082b2b0808,
    0x0819191908080819, 0x0819191908081908, 0x081919190808192b, 0x0819191908082b19,
    0x0819191908190808, 0x081919190819082b, 0x0819191908191919, 0x0819191908192b08,
    0x08191919082b0819, 0x08191919082b1908, 0x0819191919080808, 0x081919191908082b,
    0x0819191919081919, 0x0819191919082b08, 0x0819191919190819, 0x0819191919191908,
    0x08191919192b0808, 0x081919192b080819, 0x081919192b081908, 0x081919192b190808,
    0x0819192b08080808, 0x0819192b08081919, 0x0819192b08082b08, 0x0819192b08190819,
    0x0819192b08191908, 0x0819192b082b0808, 0x0819192b19080819, 0x0819192b19081908,
    0x0819192b19190808, 0x0819192b2b080808, 0x0819192b2b2b2b2b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b080808192b, 0x08192b0808082b19, 0x08192b0808190808,
    0x08192b0808191919, 0x08192b0808192b08, 0x08192b08082b0819, 0x08192b0819080808,
    0x08192b081908082b, 0x08192b0819081919, 0x08192b0819082b08, 0x08192b0819190819,
    0x08192b0819191908, 0x08192b08192b0808, 0x08192b082b080819, 0x08192b082b081908,
    0x08192b1908080808, 0x08192b190808082b, 0x08192b1908081919, 0x08192b1908082b08,
    0x08192b1908190819, 0x08192b1908191908, 0x08192b19082b0808, 0x08192b1919080819,
    0x08192b1919081908, 0x08192b1919190808, 0x08192b19192b2b19, 0x08192b192b2b082b,
    0x08192b2b08081908, 0x08192b2b08190808, 0x08192b2b19080808, 0x08192b2b1919192b,
    0x082b080808080808, 0x082b08080808082b, 0x082b080808081919, 0x082b080808082b08,
    0x082b080808190819, 0x082b080808191908, 0x082b08080819192b, 0x082b080808192b19,
    0x082b0808082b0808, 0x082b0808082b1919, 0x082b0808082b2b2b, 0x082b080819080819,
    0x082b080819081908, 0x082b080819190808, 0x082b08081919082b, 0x082b080819191919,
    0x082b0808192b1908, 0x082b08082b080808, 0x082b08082b082b2b, 0x082b08082b191908,
    0x082b08082b2b2b2b, 0x082b081908080819, 0x082b081908081908, 0x082b081908190808,
    0x082b08190819082b, 0x082b081908191919, 0x082b0819082b0819, 0x082b081919080808,
    0x082b08191908082b, 0x082b081919081919, 0x082b081919190819, 0x082b081919191908,
    0x082b0819192b0808, 0x082b08192b080819, 0x082b08192b081908, 0x082b08192b190808,
    0x082b082b08080808, 0x082b082b08082b2b, 0x082b082b082b082b, 0x082b082b082b2b08,
    0x082b082b082b2b2b, 0x082b082b19081908, 0x082b082b19190808, 0x082b082b2b082b08,
    0x082b082b2b082b2b, 0x082b082b2b2b2b08, 0x082b190808080819, 0x082b190808081908,
    0x082b19080808192b, 0x082b190808082b19, 0x082b190808190808, 0x082b190808191919,
    0x082b190808192b08, 0x082b1908082b0819, 0x082b1908082b1908, 0x082b190819080808,
    0x082b19081908082b, 0x082b190819081919, 0x082b190819082b08, 0x082b190819190819,
    0x082b190819191908, 0x082b1908192b0808, 0x082b19082b080819, 0x082b19082b081908,
    0x082b19082b190808, 0x082b191908080808, 0x082b191908081919, 0x082b191908082b08,
    0x082b191908190819, 0x082b191908191908, 0x082b1919082b0808, 0x082b191919080819,
    0x082b191919081908, 0x082b191919190808, 0x082b1919192b192b, 0x082b19192b080808,
    0x082b192b08080819, 0x082b192b08081908, 0x082b192b08190808, 0x082b192b19080808,
    0x082b192b19192b19, 0x082b2b0808080808, 0x082b2b0808081919, 0x082b2b0808190819,
    0x082b2b0808191908, 0x082b2b0819080819, 0x082b2b0819081908, 0x082b2b0819190808,
    0x082b2b082b082b2b, 0x082b2b082b2b2b2b, 0x082b2b1908080819, 0x082b2b1908081908,
    0x082b2b1908190808, 0x082b2b192b191919, 0x082b2b2b08082b2b, 0x082b2b2b082b082b,
    0x082b2b2b192b1908, 0x082b2b2b2b082b08, 0x082b2b2b2b082b2b, 0x1908080808080819,
    0x1908080808081908, 0x190808080808192b, 0x1908080808082b19, 0x1908080808190808,
    0x190808080819082b, 0x1908080808191919, 0x1908080808192b08, 0x1908080808192b2b,
    0x19080808082b0819, 0x19080808082b1908, 0x19080808082b192b, 0x1908080819080808,
    0x190808081908082b, 0x1908080819081919, 0x1908080819082b08, 0x1908080819082b2b,
    0x1908080819190819, 0x1908080819191908, 0x190808081919192b, 0x1908080819192b19,
    0x19080808192b0808, 0x19080808192b082b, 0x19080808192b1919, 0x190808082b080819,
    0x190808082b081908, 0x190808082b190808, 0x190808082b191919, 0x190808082b192b08,
    0x190808082b2b0819, 0x190808082b2b1908, 0x1908081908080808, 0x190808190808082b,
    0x1908081908081919, 0x1908081908082b08, 0x1908081908190819, 0x1908081908191908,
    0x190808190819192b, 0x1908081908192b19, 0x19080819082b0808, 0x19080819082b082b,
    0x19080819082b1919, 0x1908081919080819, 0x1908081919081908, 0x190808191908192b,
    0x1908081919082b19, 0x1908081919190808, 0x190808191919082b, 0x1908081919191919,
    0x1908081919192b08, 0x19080819192b0819, 0x19080819192b1908, 0x190808192b080808,
    0x190808192b08082b, 0x190808192b081919, 0x190808192b082b08, 0x190808192b190819,
    0x190808192b191908, 0x190808192b2b0808, 0x1908082b08080819, 0x1908082b08081908,
    0x1908082b08190808, 0x1908082b0819082b, 0x1908082b08191919, 0x1908082b08192b08,
    0x1908082b082b1908, 0x1908082b19080808, 0x1908082b19081919, 0x1908082b19082b08,
    0x1908082b19190819, 0x1908082b19191908, 0x1908082b192b0808, 0x1908082b2b080819,
    0x1908082b2b081908, 0x1908190808080808, 0x190819080808082b, 0x1908190808081919,
    0x1908190808082b08, 0x1908190808082b2b, 0x1908190808190819, 0x1908190808191908,
    0x190819080819192b, 0x1908190808192b19, 0x19081908082b0808, 0x19081908082b082b,
    0x19081908082b1919, 0x19081908082b2b08, 0x1908190819080819, 0x1908190819081908,
    0x190819081908192b, 0x1908190819082b19, 0x1908190819190808, 0x190819081919082b,
    0x1908190819191919, 0x1908190819192b08, 0x19081908192b0819, 0x19081908192b1908,
    0x190819082b080808, 0x190819082b08082b, 0x190819082b081919, 0x190819082b082b08,
    0x190819082b190819, 0x190819082b191908, 0x190819082b2b0808, 0x1908191908080819,
    0x1908191908081908, 0x190819190808192b, 0x1908191908082b19, 0x1908191908190808,
    0x190819190819082b, 0x1908191908191919, 0x1908191908192b08, 0x19081919082b0819,
    0x19081919082b1908, 0x1908191919080808, 0x190819191908082b, 0x1908191919081919,
    0x1908191919082b08, 0x1908191919190819, 0x1908191919191908, 0x19081919192b0808,
    0x19081919192b2b2b, 0x190819192b080819, 0x190819192b081908, 0x190819192b190808,
    0x1908192b08080808, 0x1908192b0808082b, 0x1908192b08081919, 0x1908192b08082b08,
    0x1908192b08190819, 0x1908192b08191908, 0x1908192b082b0808, 0x1908192b19080819,
    0x1908192b19081908, 0x1908192b19190808, 0x1908192b2b080808, 0x1908192b2b2b1919,
    0x19082b0808080819, 0x19082b0808081908, 0x19082b0808082b19, 0x19082b0808190808,
    0x19082b080819082b, 0x19082b0808191919, 0x19082b0808192b08, 0x19082b08082b0819,
    0x19082b08082b1908, 0x19082b0819080808, 0x19082b081908082b, 0x19082b0819081919,
    0x19082b0819082b08, 0x19082b0819190819, 0x19082b0819191908, 0x19082b08192b0808,
    0x19082b082b081908, 0x19082b082b190808, 0x19082b1908080808, 0x19082b190808082b,
    0x19082b1908081919, 0x19082b1908082b08, 0x19082b1908190819, 0x19082b1908191908,
    0x19082b19082b0808, 0x19082b1919080819, 0x19082b1919081908, 0x19082b1919190808,
    0x19082b192b080808, 0x19082b192b19192b, 0x19082b2b08080819, 0x19082b2b08081908,
    0x19082b2b08190808, 0x19082b2b19080808, 0x1919080808080808, 0x191908080808082b,
    0x1919080808081919, 0x1919080808082b08, 0x1919080808190819, 0x1919080808191908,
    0x191908080819192b, 0x1919080808192b19, 0x19190808082b0808, 0x19190808082b082b,
    0x19190808082b1919, 0x19190808082b2b08, 0x1919080819080819, 0x1919080819081908,
    0x191908081908192b, 0x1919080819082b19, 0x1919080819190808, 0x191908081919082b,
    0x1919080819191919, 0x1919080819192b08, 0x19190808192b0819, 0x19190808192b1908,
    0x191908082b080808, 0x191908082b08082b, 0x191908082b081919, 0x191908082b082b08,
    0x191908082b190819, 0x191908082b191908, 0x1919081908080819, 0x1919081908081908,
    0x191908190808192b, 0x1919081908082b19, 0x1919081908190808, 0x191908190819082b,
    0x1919081908191919, 0x1919081908192b08, 0x19190819082b0819, 0x19190819082b1908,
    0x1919081919080808, 0x191908191908082b, 0x1919081919081919, 0x1919081919082b08,
    0x1919081919190819, 0x1919081919191908, 0x19190819192b0808, 0x191908192b080819,
    0x191908192b081908, 0x191908192b190808, 0x1919082b08080808, 0x1919082b08081919,
    0x1919082b08082b08, 0x1919082b08190819, 0x1919082b08191908, 0x1919082b082b0808,
    0x1919082b19080819, 0x1919082b19081908, 0x1919082b19190808, 0x1919082b192b2b19,
    0x1919082b2b080808, 0x1919190808080819, 0x1919190808081908, 0x191919080808192b,
    0x1919190808082b19, 0x1919190808190808, 0x191919080819082b, 0x1919190808191919,
    0x1919190808192b08, 0x19191908082b0819, 0x19191908082b1908, 0x1919190819080808,
    0x191919081908082b, 0x1919190819081919, 0x1919190819082b08, 0x1919190819190819,
    0x1919190819191908, 0x19191908192b0808, 0x191919082b080819, 0x191919082b081908,
    0x191919082b190808, 0x1919191908080808, 0x191919190808082b, 0x1919191908081919,
    0x1919191908082b08, 0x1919191908190819, 0x1919191908191908, 0x19191919082b0808,
    0x1919191919080819, 0x1919191919081908, 0x1919191919190808, 0x191919192b080808,
    0x1919192b08080819, 0x1919192b08081908, 0x1919192b08190808, 0x1919192b082b192b,
    0x1919192b19080808, 0x19192b0808080808, 0x19192b080808082b, 0x19192b0808081919,
    0x19192b0808082b08, 0x19192b0808190819, 0x19192b0808191908, 0x19192b08082b0808,
    0x19192b0819080819, 0x19192b0819081908, 0x19192b0819190808, 0x19192b0819192b2b,
    0x19192b082b080808, 0x19192b1908080819, 0x19192b1908081908, 0x19192b1908190808,
    0x19192b1919080808, 0x19192b2b08080808, 0x19192b2b08192b19, 0x19192b2b2b081919,
    0x19192b2b2b2b2b08, 0x192b080808080819, 0x192b080808081908, 0x192b08080808192b,
    0x192b080808190808, 0x192b08080819082b, 0x192b080808191919, 0x192b080808192b08,
    0x192b0808082b0819, 0x192b0808082b1908, 0x192b080819080808, 0x192b080819081919,
    0x192b080819082b08, 0x192b080819190819, 0x192b080819191908, 0x192b0808192b0808,
    0x192b08082b081908, 0x192b08082b190808, 0x192b081908080808, 0x192b08190808082b,
    0x192b081908081919, 0x192b081908082b08, 0x192b081908190819, 0x192b081908191908,
    0x192b0819082b0808, 0x192b081919080819, 0x192b081919081908, 0x192b081919190808,
    0x192b08192b080808, 0x192b08192b192b19, 0x192b082b08081908, 0x192b082b08190808,
    0x192b082b19080808, 0x192b082b1919192b, 0x192b082b2b2b0819, 0x192b190808080808,
    0x192b190808081919, 0x192b190808082b08, 0x192b190808190819, 0x192b190808191908,
    0x192b1908082b0808, 0x192b190819080819, 0x192b190819081908, 0x192b190819190808,
    0x192b19082b080808, 0x192b191908080819, 0x192b191908081908, 0x192b191908190808,
    0x192b191919080808, 0x192b191919082b2b, 0x192b1919192b2b08, 0x192b19192b19082b,
    0x192b192b08080808, 0x192b192b2b191908, 0x192b2b0808080819, 0x192b2b0808081908,
    0x192b2b0808190808, 0x192b2b08192b1919, 0x192b2b082b192b08, 0x192b2b1908080808,
    0x192b2b19082b2b2b, 0x192b2b2b1908082b, 0x192b2b2b2b2b0819, 0x2b08080808080808,
    0x2b0808080808082b, 0x2b08080808081919, 0x2b08080808082b08, 0x2b08080808190819,
    0x2b08080808191908, 0x2b08080808192b19, 0x2b080808082b0808, 0x2b080808082b1919,
    0x2b08080819080819, 0x2b08080819081908, 0x2b08080819190808, 0x2b0808081919082b,
    0x2b08080819191919, 0x2b08080819192b08, 0x2b080808192b0819, 0x2b0808082b080808,
    0x2b0808082b081919, 0x2b0808082b190819, 0x2b0808082b191908, 0x2b08081908080819,
    0x2b08081908081908, 0x2b08081908082b19, 0x2b08081908190808, 0x2b0808190819082b,
    0x2b08081908191919, 0x2b08081908192b08, 0x2b080819082b0819, 0x2b080819082b1908,
    0x2b08081919080808, 0x2b0808191908082b, 0x2b08081919081919, 0x2b08081919082b08,
    0x2b08081919190819, 0x2b08081919191908, 0x2b0808192b080819, 0x2b0808192b081908,
    0x2b0808192b190808, 0x2b0808192b2b2b19, 0x2b08082b08080808, 0x2b08082b08081919,
    0x2b08082b08082b2b, 0x2b08082b08190819, 0x2b08082b08191908, 0x2b08082b19080819,
    0x2b08082b19081908, 0x2b08082b19190808, 0x2b08190808080819, 0x2b08190808081908,
    0x2b0819080808192b, 0x2b08190808082b19, 0x2b08190808190808, 0x2b0819080819082b,
    0x2b08190808191919, 0x2b08190808192b08, 0x2b081908082b0819, 0x2b08190819080808,
    0x2b0819081908082b, 0x2b08190819081919, 0x2b08190819082b08, 0x2b08190819190819,
    0x2b08190819191908, 0x2b081908192b0808, 0x2b0819082b080819, 0x2b0819082b081908,
    0x2b0819082b190808, 0x2b08191908080808, 0x2b0819190808082b, 0x2b08191908081919,
    0x2b08191908082b08, 0x2b08191908190819, 0x2b08191908191908, 0x2b081919082b0808,
    0x2b08191919080819, 0x2b08191919081908, 0x2b08191919190808, 0x2b0819192b080808,
    0x2b0819192b082b2b, 0x2b08192b08080819, 0x2b08192b08081908, 0x2b08192b08190808,
    0x2b08192b082b2b19, 0x2b08192b19080808, 0x2b082b0808080808, 0x2b082b0808081919,
    0x2b082b0808190819, 0x2b082b0808191908, 0x2b082b0819080819, 0x2b082b0819081908,
    0x2b082b0819190808, 0x2b082b082b2b082b, 0x2b082b1908080819, 0x2b082b1908081908,
    0x2b082b1919080808, 0x2b082b19192b1919, 0x2b082b2b082b082b, 0x2b082b2b19192b08,
    0x2b082b2b19192b2b, 0x2b082b2b2b08082b, 0x2b082b2b2b2b082b, 0x2b19080808080819,
    0x2b19080808081908, 0x2b19080808082b19, 0x2b19080808190808, 0x2b1908080819082b,
    0x2b19080808191919, 0x2b19080808192b08, 0x2b190808082b1908, 0x2b19080819080808,
    0x2b1908081908082b, 0x2b19080819081919, 0x2b19080819082b08, 0x2b19080819190819,
    0x2b19080819191908, 0x2b190808192b0808, 0x2b1908082b080819, 0x2b1908082b081908,
    0x2b1908082b190808, 0x2b19081908080808, 0x2b19081908081919, 0x2b19081908190819,
    0x2b19081908191908, 0x2b19081919080819, 0x2b19081919081908, 0x2b19081919190808,
    0x2b19081919192b2b, 0x2b19082b08080819, 0x2b19082b08081908, 0x2b19082b08190808,
    0x2b19082b19080808, 0x2b19082b2b2b192b, 0x2b19190808080808, 0x2b1919080808082b,
    0x2b19190808081919, 0x2b19190808082b08, 0x2b19190808190819, 0x2b19190808191908,
    0x2b191908082b0808, 0x2b19190819080819, 0x2b19190819081908, 0x2b19190819190808,
    0x2b1919082b080808, 0x2b1919082b19192b, 0x2b19191908080819, 0x2b19191908081908,
    0x2b19191908190808, 0x2b19191919080808, 0x2b1919192b192b08, 0x2b1919192b2b0819,
    0x2b19192b08080808, 0x2b19192b1908192b, 0x2b19192b192b1908, 0x2b192b0808080819,
    0x2b192b0808081908, 0x2b192b0808190808, 0x2b192b08082b192b, 0x2b192b0819080808,
    0x2b192b082b2b2b19, 0x2b192b1908080808, 0x2b192b1919082b19, 0x2b192b191919082b,
    0x2b192b2b2b190808, 0x2b2b080808080808, 0x2b2b080808081919, 0x2b2b080808082b2b,
    0x2b2b080808191908, 0x2b2b0808082b082b, 0x2b2b0808082b2b2b, 0x2b2b080819080819,
    0x2b2b080819081908, 0x2b2b080819190808, 0x2b2b08082b2b082b, 0x2b2b08082b2b2b2b,
    0x2b2b081919080808, 0x2b2b0819192b1919, 0x2b2b082b0808082b, 0x2b2b082b08082b2b,
    0x2b2b082b082b082b, 0x2b2b082b082b2b08, 0x2b2b082b082b2b2b, 0x2b2b082b2b08082b,
    0x2b2b082b2b082b08, 0x2b2b082b2b082b2b, 0x2b2b082b2b2b2b08, 0x2b2b190808080819,
    0x2b2b190808081908, 0x2b2b190808190808, 0x2b2b190819080808, 0x2b2b19082b082b19,
    0x2b2b19082b2b1908, 0x2b2b191908080808, 0x2b2b191908192b19, 0x2b2b192b19190819,
    0x2b2b2b0808082b2b, 0x2b2b2b08082b2b08, 0x2b2b2b082b2b082b, 0x2b2b2b1919191908,
    0x2b2b2b192b08192b, 0x2b2b2b2b08082b08, 0x2b2b2b2b08082b2b, 0x2b2b2b2b082b0808,
    0x2b2b2b2b082b082b, 0x2b2b2b2b082b2b08, 0x2b2b2b2b2b082b08, 0x2b2b2b2b2b2b2b2b,
};

constexpr constant uint kIq3SGrid[512] = {
    0x01010101, 0x01010103, 0x01010105, 0x0101010b, 0x0101010f, 0x01010301, 0x01010303, 0x01010305,
    0x01010309, 0x0101030d, 0x01010501, 0x01010503, 0x0101050b, 0x01010707, 0x01010901, 0x01010905,
    0x0101090b, 0x0101090f, 0x01010b03, 0x01010b07, 0x01010d01, 0x01010d05, 0x01010f03, 0x01010f09,
    0x01010f0f, 0x01030101, 0x01030103, 0x01030105, 0x01030109, 0x01030301, 0x01030303, 0x0103030b,
    0x01030501, 0x01030507, 0x0103050f, 0x01030703, 0x0103070b, 0x01030909, 0x01030d03, 0x01030d0b,
    0x01030f05, 0x01050101, 0x01050103, 0x0105010b, 0x0105010f, 0x01050301, 0x01050307, 0x0105030d,
    0x01050503, 0x0105050b, 0x01050701, 0x01050709, 0x01050905, 0x0105090b, 0x0105090f, 0x01050b03,
    0x01050b07, 0x01050f01, 0x01050f07, 0x01070107, 0x01070303, 0x0107030b, 0x01070501, 0x01070505,
    0x01070703, 0x01070707, 0x0107070d, 0x01070909, 0x01070b01, 0x01070b05, 0x01070d0f, 0x01070f03,
    0x01070f0b, 0x01090101, 0x01090307, 0x0109030f, 0x01090503, 0x01090509, 0x01090705, 0x01090901,
    0x01090907, 0x01090b03, 0x01090f01, 0x010b0105, 0x010b0109, 0x010b0501, 0x010b0505, 0x010b050d,
    0x010b0707, 0x010b0903, 0x010b090b, 0x010b090f, 0x010b0d0d, 0x010b0f07, 0x010d010d, 0x010d0303,
    0x010d0307, 0x010d0703, 0x010d0b05, 0x010d0f03, 0x010f0101, 0x010f0105, 0x010f0109, 0x010f0501,
    0x010f0505, 0x010f050d, 0x010f0707, 0x010f0b01, 0x010f0b09, 0x03010101, 0x03010103, 0x03010105,
    0x03010109, 0x03010301, 0x03010303, 0x03010307, 0x0301030b, 0x0301030f, 0x03010501, 0x03010505,
    0x03010703, 0x03010709, 0x0301070d, 0x03010b09, 0x03010b0d, 0x03010d03, 0x03010f05, 0x03030101,
    0x03030103, 0x03030107, 0x0303010d, 0x03030301, 0x03030309, 0x03030503, 0x03030701, 0x03030707,
    0x03030903, 0x03030b01, 0x03030b05, 0x03030f01, 0x03030f0d, 0x03050101, 0x03050305, 0x0305030b,
    0x0305030f, 0x03050501, 0x03050509, 0x03050705, 0x03050901, 0x03050907, 0x03050b0b, 0x03050d01,
    0x03050f05, 0x03070103, 0x03070109, 0x0307010f, 0x03070301, 0x03070307, 0x03070503, 0x0307050f,
    0x03070701, 0x03070709, 0x03070903, 0x03070d05, 0x03070f01, 0x03090107, 0x0309010b, 0x03090305,
    0x03090309, 0x03090703, 0x03090707, 0x03090905, 0x0309090d, 0x03090b01, 0x03090b09, 0x030b0103,
    0x030b0301, 0x030b0307, 0x030b0503, 0x030b0701, 0x030b0705, 0x030b0b03, 0x030d0501, 0x030d0509,
    0x030d050f, 0x030d0909, 0x030d090d, 0x030f0103, 0x030f0107, 0x030f0301, 0x030f0305, 0x030f0503,
    0x030f070b, 0x030f0903, 0x030f0d05, 0x030f0f01, 0x05010101, 0x05010103, 0x05010107, 0x0501010b,
    0x0501010f, 0x05010301, 0x05010305, 0x05010309, 0x0501030d, 0x05010503, 0x05010507, 0x0501050f,
    0x05010701, 0x05010705, 0x05010903, 0x05010907, 0x0501090b, 0x05010b01, 0x05010b05, 0x05010d0f,
    0x05010f01, 0x05010f07, 0x05010f0b, 0x05030101, 0x05030105, 0x05030301, 0x05030307, 0x0503030f,
    0x05030505, 0x0503050b, 0x05030703, 0x05030709, 0x05030905, 0x05030b03, 0x05050103, 0x05050109,
    0x0505010f, 0x05050503, 0x05050507, 0x05050701, 0x0505070f, 0x05050903, 0x05050b07, 0x05050b0f,
    0x05050f03, 0x05050f09, 0x05070101, 0x05070105, 0x0507010b, 0x05070303, 0x05070505, 0x05070509,
    0x05070703, 0x05070707, 0x05070905, 0x05070b01, 0x05070d0d, 0x05090103, 0x0509010f, 0x05090501,
    0x05090507, 0x05090705, 0x0509070b, 0x05090903, 0x05090f05, 0x05090f0b, 0x050b0109, 0x050b0303,
    0x050b0505, 0x050b070f, 0x050b0901, 0x050b0b07, 0x050b0f01, 0x050d0101, 0x050d0105, 0x050d010f,
    0x050d0503, 0x050d0b0b, 0x050d0d03, 0x050f010b, 0x050f0303, 0x050f050d, 0x050f0701, 0x050f0907,
    0x050f0b01, 0x07010105, 0x07010303, 0x07010307, 0x0701030b, 0x0701030f, 0x07010505, 0x07010703,
    0x07010707, 0x0701070b, 0x07010905, 0x07010909, 0x0701090f, 0x07010b03, 0x07010d07, 0x07010f03,
    0x07030103, 0x07030107, 0x0703010b, 0x07030309, 0x07030503, 0x07030507, 0x07030901, 0x07030d01,
    0x07030f05, 0x07030f0d, 0x07050101, 0x07050305, 0x07050501, 0x07050705, 0x07050709, 0x07050b01,
    0x07070103, 0x07070301, 0x07070309, 0x07070503, 0x07070507, 0x0707050f, 0x07070701, 0x07070903,
    0x07070907, 0x0707090f, 0x07070b0b, 0x07070f07, 0x07090107, 0x07090303, 0x0709030d, 0x07090505,
    0x07090703, 0x07090b05, 0x07090d01, 0x07090d09, 0x070b0103, 0x070b0301, 0x070b0305, 0x070b050b,
    0x070b0705, 0x070b0909, 0x070b0b0d, 0x070b0f07, 0x070d030d, 0x070d0903, 0x070f0103, 0x070f0107,
    0x070f0501, 0x070f0505, 0x070f070b, 0x09010101, 0x09010109, 0x09010305, 0x09010501, 0x09010509,
    0x0901050f, 0x09010705, 0x09010903, 0x09010b01, 0x09010f01, 0x09030105, 0x0903010f, 0x09030303,
    0x09030307, 0x09030505, 0x09030701, 0x0903070b, 0x09030907, 0x09030b03, 0x09030b0b, 0x09050103,
    0x09050107, 0x09050301, 0x0905030b, 0x09050503, 0x09050707, 0x09050901, 0x09050b0f, 0x09050d05,
    0x09050f01, 0x09070109, 0x09070303, 0x09070307, 0x09070501, 0x09070505, 0x09070703, 0x0907070b,
    0x09090101, 0x09090105, 0x09090509, 0x0909070f, 0x09090901, 0x09090f03, 0x090b010b, 0x090b010f,
    0x090b0503, 0x090b0d05, 0x090d0307, 0x090d0709, 0x090d0d01, 0x090f0301, 0x090f030b, 0x090f0701,
    0x090f0907, 0x090f0b03, 0x0b010105, 0x0b010301, 0x0b010309, 0x0b010505, 0x0b010901, 0x0b010909,
    0x0b01090f, 0x0b010b05, 0x0b010d0d, 0x0b010f09, 0x0b030103, 0x0b030107, 0x0b03010b, 0x0b030305,
    0x0b030503, 0x0b030705, 0x0b030f05, 0x0b050101, 0x0b050303, 0x0b050507, 0x0b050701, 0x0b05070d,
    0x0b050b07, 0x0b070105, 0x0b07010f, 0x0b070301, 0x0b07050f, 0x0b070909, 0x0b070b03, 0x0b070d0b,
    0x0b070f07, 0x0b090103, 0x0b090109, 0x0b090501, 0x0b090705, 0x0b09090d, 0x0b0b0305, 0x0b0b050d,
    0x0b0b0b03, 0x0b0b0b07, 0x0b0d0905, 0x0b0f0105, 0x0b0f0109, 0x0b0f0505, 0x0d010303, 0x0d010307,
    0x0d01030b, 0x0d010703, 0x0d010707, 0x0d010d01, 0x0d030101, 0x0d030501, 0x0d03050f, 0x0d030d09,
    0x0d050305, 0x0d050709, 0x0d050905, 0x0d050b0b, 0x0d050d05, 0x0d050f01, 0x0d070101, 0x0d070309,
    0x0d070503, 0x0d070901, 0x0d09050b, 0x0d090907, 0x0d090d05, 0x0d0b0101, 0x0d0b0107, 0x0d0b0709,
    0x0d0b0d01, 0x0d0d010b, 0x0d0d0901, 0x0d0f0303, 0x0d0f0307, 0x0f010101, 0x0f010109, 0x0f01010f,
    0x0f010501, 0x0f010505, 0x0f01070d, 0x0f010901, 0x0f010b09, 0x0f010d05, 0x0f030105, 0x0f030303,
    0x0f030509, 0x0f030907, 0x0f03090b, 0x0f050103, 0x0f050109, 0x0f050301, 0x0f05030d, 0x0f050503,
    0x0f050701, 0x0f050b03, 0x0f070105, 0x0f070705, 0x0f07070b, 0x0f070b07, 0x0f090103, 0x0f09010b,
    0x0f090307, 0x0f090501, 0x0f090b01, 0x0f0b0505, 0x0f0b0905, 0x0f0d0105, 0x0f0d0703, 0x0f0f0101,
};

inline float tensorsharp_dequant_iq2_s(const device uchar * block, int within_block) {
    const half d_half = *reinterpret_cast<const device half *>(block);
    const device uchar * qs = block + 2;
    const device uchar * signs_base = qs + 32;
    const device uchar * qh = qs + 64;
    const device uchar * scales = qh + 8;
    const int ib32 = within_block >> 5;
    const int within_32 = within_block & 31;
    const int lane = within_32 >> 3;
    const int j = within_32 & 7;
    const uchar qh_byte = qh[ib32];
    const uint grid_index = static_cast<uint>(qs[4 * ib32 + lane]) | ((static_cast<uint>(qh_byte) << (8 - 2 * lane)) & 0x300u);
    const uchar scale_byte = scales[ib32];
    const uint scale = lane < 2 ? (scale_byte & 0x0fu) : (scale_byte >> 4);
    const float db = static_cast<float>(d_half) * (0.5f + static_cast<float>(scale)) * 0.25f;
    const ulong packed_grid = kIq2SGrid[grid_index];
    const uint grid = static_cast<uint>((packed_grid >> (8 * j)) & 255);
    const uint signs = signs_base[4 * ib32 + lane];
    return db * static_cast<float>(grid) * ((signs & (1u << j)) != 0 ? -1.0f : 1.0f);
}

inline float tensorsharp_dequant_iq3_s(const device uchar * block, int within_block) {
    const half d_half = *reinterpret_cast<const device half *>(block);
    const device uchar * qs = block + 2;
    const device uchar * qh = qs + 64;
    const device uchar * signs_base = qh + 8;
    const device uchar * scales = signs_base + 32;
    const int ib32 = within_block >> 5;
    const int within_32 = within_block & 31;
    const int half16 = within_32 >> 4;
    const int grid_lane = (within_32 >> 2) & 3;
    const int j = within_32 & 3;
    const uint qh_nibble = static_cast<uint>(qh[ib32] >> (4 * half16));
    const uint grid_index = static_cast<uint>(qs[8 * ib32 + 4 * half16 + grid_lane]) | ((qh_nibble << (8 - grid_lane)) & 256u);
    const uint scale = (scales[ib32 >> 1] >> (4 * (ib32 & 1))) & 0x0fu;
    const float db = static_cast<float>(d_half) * static_cast<float>(1 + 2 * scale);
    const uint packed_grid = kIq3SGrid[grid_index];
    const uint grid = (packed_grid >> (8 * j)) & 255u;
    const uint signs = signs_base[4 * ib32 + 2 * half16 + (grid_lane >> 1)];
    const uint sign_bit = static_cast<uint>(j + ((grid_lane & 1) << 2));
    return db * static_cast<float>(grid) * ((signs & (1u << sign_bit)) != 0 ? -1.0f : 1.0f);
}

// 8 consecutive IQ2_S weights: one 1024-entry codebook lookup (each entry
// packs eight 8-bit magnitudes), one sign byte, one 4-bit scale.
inline float tensorsharp_dot8_iq2_s(const device uchar * block, int within_block, thread const float * xv) {
    const half d_half = *reinterpret_cast<const device half *>(block);
    const device uchar * qs = block + 2;
    const device uchar * signs_base = qs + 32;
    const device uchar * qh = qs + 64;
    const device uchar * scales = qh + 8;
    const int ib32 = within_block >> 5;
    const int lane = (within_block >> 3) & 3;

    const uchar qh_byte = qh[ib32];
    const uint grid_index = static_cast<uint>(qs[4 * ib32 + lane]) | ((static_cast<uint>(qh_byte) << (8 - 2 * lane)) & 0x300u);
    const uchar scale_byte = scales[ib32];
    const uint scale = lane < 2 ? (scale_byte & 0x0fu) : (scale_byte >> 4);
    const float db = static_cast<float>(d_half) * (0.5f + static_cast<float>(scale)) * 0.25f;
    const ulong packed_grid = kIq2SGrid[grid_index];
    const uint signs = signs_base[4 * ib32 + lane];

    float acc = 0.0f;
    for (int j = 0; j < 8; ++j) {
        const float g = static_cast<float>(static_cast<uint>((packed_grid >> (8 * j)) & 255));
        acc += xv[j] * (((signs >> j) & 1u) != 0u ? -g : g);
    }
    return db * acc;
}

// 8 consecutive IQ3_S weights. Unlike IQ2_S an iq3s_grid entry only covers
// FOUR magnitudes, so a run of 8 spans two adjacent grid lanes - but both
// lanes share the same sign byte (index (grid_lane >> 1) is equal for the
// pair) and the same 4-bit scale, so the amortization still holds.
inline float tensorsharp_dot8_iq3_s(const device uchar * block, int within_block, thread const float * xv) {
    const half d_half = *reinterpret_cast<const device half *>(block);
    const device uchar * qs = block + 2;
    const device uchar * qh = qs + 64;
    const device uchar * signs_base = qh + 8;
    const device uchar * scales = signs_base + 32;
    const int ib32 = within_block >> 5;
    const int within_32 = within_block & 31;
    const int half16 = within_32 >> 4;
    const int grid_lane0 = (within_32 >> 2) & 3;   // always even: 0 or 2
    const uint qh_nibble = static_cast<uint>(qh[ib32] >> (4 * half16));
    const uint scale = (scales[ib32 >> 1] >> (4 * (ib32 & 1))) & 0x0fu;
    const float db = static_cast<float>(d_half) * static_cast<float>(1 + 2 * scale);
    const uint signs = signs_base[4 * ib32 + 2 * half16 + (grid_lane0 >> 1)];

    float acc = 0.0f;
    for (int gl = 0; gl < 2; ++gl) {
        const int grid_lane = grid_lane0 + gl;
        const uint grid_index = static_cast<uint>(qs[8 * ib32 + 4 * half16 + grid_lane]) | ((qh_nibble << (8 - grid_lane)) & 256u);
        const uint packed_grid = kIq3SGrid[grid_index];
        for (int j = 0; j < 4; ++j) {
            const float g = static_cast<float>((packed_grid >> (8 * j)) & 255u);
            const uint sign_bit = static_cast<uint>(j + ((grid_lane & 1) << 2));
            acc += xv[4 * gl + j] * (((signs >> sign_bit) & 1u) != 0u ? -g : g);
        }
    }
    return db * acc;
}
";

        private const string KQuantHelpersHeader = @"
inline void tensorsharp_get_scale_min_k4(int index, const device uchar * packed, thread int & scale, thread int & minv) {
    if (index < 4) {
        scale = packed[index] & 63;
        minv = packed[index + 4] & 63;
    } else {
        scale = (packed[index + 4] & 0x0f) | ((packed[index - 4] >> 6) << 4);
        minv = (packed[index + 4] >> 4) | ((packed[index] >> 6) << 4);
    }
}

inline float tensorsharp_dequant_q4k(const device uchar * block, int within_block) {
    int group = within_block >> 5;
    int within_32 = within_block & 31;
    int pair_index = group >> 1;
    bool high_nibble = (group & 1) != 0;

    int scale_byte = 0;
    int min_byte = 0;
    tensorsharp_get_scale_min_k4(group, block + 4, scale_byte, min_byte);

    const half d_half = *reinterpret_cast<const device half *>(block);
    const half min_half = *reinterpret_cast<const device half *>(block + 2);
    const uchar packed = block[16 + pair_index * 32 + within_32];
    const int q = high_nibble ? ((packed >> 4) & 0x0f) : (packed & 0x0f);
    const half scale_h = d_half * static_cast<half>(scale_byte);
    const half min_h = min_half * static_cast<half>(min_byte);
    return static_cast<float>(scale_h) * static_cast<float>(q) - static_cast<float>(min_h);
}

inline float tensorsharp_dequant_q5k(const device uchar * block, int within_block) {
    int group = within_block >> 5;
    int within_32 = within_block & 31;
    int pair_index = group >> 1;
    bool high_nibble = (group & 1) != 0;

    int scale_byte = 0;
    int min_byte = 0;
    tensorsharp_get_scale_min_k4(group, block + 4, scale_byte, min_byte);

    const half d_half = *reinterpret_cast<const device half *>(block);
    const half min_half = *reinterpret_cast<const device half *>(block + 2);
    const uchar packed = block[48 + pair_index * 32 + within_32];
    const int lo4 = high_nibble ? ((packed >> 4) & 0x0f) : (packed & 0x0f);
    const int bit5 = (block[16 + within_32] >> group) & 1;
    const int q = lo4 | (bit5 << 4);
    const half scale_h = d_half * static_cast<half>(scale_byte);
    const half min_h = min_half * static_cast<half>(min_byte);
    return static_cast<float>(scale_h) * static_cast<float>(q) - static_cast<float>(min_h);
}

inline float tensorsharp_dequant_q6k(const device uchar * block, int within_block) {
    const device uchar * ql = block;
    const device uchar * qh = block + 128;
    const device uchar * scales = block + 192;
    const half d_half = *reinterpret_cast<const device half *>(block + 208);

    int sub = within_block >> 4;
    int idx = within_block & 15;
    int half_block = sub >> 3;
    int sh = sub & 7;
    int ql_offset = half_block * 64 + (sh & 3) * 16;
    bool is_upper = sh >= 4;
    int qh_offset = half_block * 32 + (sh & 1) * 16;
    int qh_shift = (sh >> 1) * 2;
    int lo4 = is_upper ? ((ql[ql_offset + idx] >> 4) & 0x0f) : (ql[ql_offset + idx] & 0x0f);
    int hi2 = (qh[qh_offset + idx] >> qh_shift) & 0x03;
    int q = (lo4 | (hi2 << 4)) - 32;
    int scale_i = scales[sub];
    if (scale_i >= 128) {
        scale_i -= 256;
    }

    return static_cast<float>(d_half) * static_cast<float>(scale_i) * static_cast<float>(q);
}
";

        // Phase 8: simdgroup-fast reduction. Replaces the 8-barrier tree
        // reduction with a single barrier + simd_sum, matching the pattern
        // proven on Q8 in Phase 6. Same kernel structure: one threadgroup
        // per (out_col, row_idx), each thread strides through the input.
        private const string Q4KMatmulSource = @"
auto tid = thread_position_in_threadgroup.x;
auto out_col = thread_position_in_grid.y;
auto row_idx = thread_position_in_grid.z;
auto simd_lane = tid & 31u;
auto simd_id = tid >> 5;
threadgroup float simd_partial[8];

float sum = 0.0f;
for (int k = static_cast<int>(tid); k < InDim; k += 256) {
    int block_in_row = k >> 8;
    int within_block = k & 255;
    auto block = w + (out_col * BlocksPerRow + block_in_row) * 144;
    sum += x[row_idx * InDim + k] * tensorsharp_dequant_q4k(block, within_block);
}

float simd_total = simd_sum(sum);
if (simd_lane == 0) simd_partial[simd_id] = simd_total;
threadgroup_barrier(mem_flags::mem_threadgroup);
float lane_v = simd_lane < 8u ? simd_partial[simd_lane] : 0.0f;
float final_sum = simd_sum(lane_v);

if (tid == 0) {
    y[row_idx * OutDim + out_col] = final_sum;
}
";

        // Phase 8: simdgroup-fast reduction (same pattern as Q4KMatmul).
        private const string Q5KMatmulSource = @"
auto tid = thread_position_in_threadgroup.x;
auto out_col = thread_position_in_grid.y;
auto row_idx = thread_position_in_grid.z;
auto simd_lane = tid & 31u;
auto simd_id = tid >> 5;
threadgroup float simd_partial[8];

float sum = 0.0f;
for (int k = static_cast<int>(tid); k < InDim; k += 256) {
    int block_in_row = k >> 8;
    int within_block = k & 255;
    auto block = w + (out_col * BlocksPerRow + block_in_row) * 176;
    sum += x[row_idx * InDim + k] * tensorsharp_dequant_q5k(block, within_block);
}

float simd_total = simd_sum(sum);
if (simd_lane == 0) simd_partial[simd_id] = simd_total;
threadgroup_barrier(mem_flags::mem_threadgroup);
float lane_v = simd_lane < 8u ? simd_partial[simd_lane] : 0.0f;
float final_sum = simd_sum(lane_v);

if (tid == 0) {
    y[row_idx * OutDim + out_col] = final_sum;
}
";

        // Phase 8: simdgroup-fast reduction across 4 output cols per
        // threadgroup. Each col gets its own simd_partial slot.
        private const string Q5KMatmul4Source = @"
auto tid = thread_position_in_threadgroup.x;
auto col_group = thread_position_in_grid.y;
int base_col = static_cast<int>(col_group) * 4;
auto simd_lane = tid & 31u;
auto simd_id = tid >> 5;
threadgroup float simd_partial[4][8];

float sum0 = 0.0f;
float sum1 = 0.0f;
float sum2 = 0.0f;
float sum3 = 0.0f;

for (int k = static_cast<int>(tid); k < InDim; k += 256) {
    const float xv = x[k];
    int block_in_row = k >> 8;
    int within_block = k & 255;

    if (base_col < OutDim) {
        auto block0 = w + (base_col * BlocksPerRow + block_in_row) * 176;
        sum0 += xv * tensorsharp_dequant_q5k(block0, within_block);
    }
    if (base_col + 1 < OutDim) {
        auto block1 = w + ((base_col + 1) * BlocksPerRow + block_in_row) * 176;
        sum1 += xv * tensorsharp_dequant_q5k(block1, within_block);
    }
    if (base_col + 2 < OutDim) {
        auto block2 = w + ((base_col + 2) * BlocksPerRow + block_in_row) * 176;
        sum2 += xv * tensorsharp_dequant_q5k(block2, within_block);
    }
    if (base_col + 3 < OutDim) {
        auto block3 = w + ((base_col + 3) * BlocksPerRow + block_in_row) * 176;
        sum3 += xv * tensorsharp_dequant_q5k(block3, within_block);
    }
}

float s0 = simd_sum(sum0);
float s1 = simd_sum(sum1);
float s2 = simd_sum(sum2);
float s3 = simd_sum(sum3);
if (simd_lane == 0) {
    simd_partial[0][simd_id] = s0;
    simd_partial[1][simd_id] = s1;
    simd_partial[2][simd_id] = s2;
    simd_partial[3][simd_id] = s3;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

float lane_v0 = simd_lane < 8u ? simd_partial[0][simd_lane] : 0.0f;
float lane_v1 = simd_lane < 8u ? simd_partial[1][simd_lane] : 0.0f;
float lane_v2 = simd_lane < 8u ? simd_partial[2][simd_lane] : 0.0f;
float lane_v3 = simd_lane < 8u ? simd_partial[3][simd_lane] : 0.0f;
float final0 = simd_sum(lane_v0);
float final1 = simd_sum(lane_v1);
float final2 = simd_sum(lane_v2);
float final3 = simd_sum(lane_v3);

if (tid == 0) {
    if (base_col < OutDim) {
        y[base_col] = final0;
    }
    if (base_col + 1 < OutDim) {
        y[base_col + 1] = final1;
    }
    if (base_col + 2 < OutDim) {
        y[base_col + 2] = final2;
    }
    if (base_col + 3 < OutDim) {
        y[base_col + 3] = final3;
    }
}
";

        // Phase 8: simdgroup-fast reduction (same pattern as Q4KMatmul).
        private const string Q6KMatmulSource = @"
auto tid = thread_position_in_threadgroup.x;
auto out_col = thread_position_in_grid.y;
auto row_idx = thread_position_in_grid.z;
auto simd_lane = tid & 31u;
auto simd_id = tid >> 5;
threadgroup float simd_partial[8];

float sum = 0.0f;
for (int k = static_cast<int>(tid); k < InDim; k += 256) {
    int block_in_row = k >> 8;
    int within_block = k & 255;
    auto block = w + (out_col * BlocksPerRow + block_in_row) * 210;
    sum += x[row_idx * InDim + k] * tensorsharp_dequant_q6k(block, within_block);
}

float simd_total = simd_sum(sum);
if (simd_lane == 0) simd_partial[simd_id] = simd_total;
threadgroup_barrier(mem_flags::mem_threadgroup);
float lane_v = simd_lane < 8u ? simd_partial[simd_lane] : 0.0f;
float final_sum = simd_sum(lane_v);

if (tid == 0) {
    y[row_idx * OutDim + out_col] = final_sum;
}
";

        // Phase 8: simdgroup-fast reduction across 4 output cols per
        // threadgroup. Each col gets its own simd_partial slot.
        private const string Q6KMatmul4Source = @"
auto tid = thread_position_in_threadgroup.x;
auto col_group = thread_position_in_grid.y;
int base_col = static_cast<int>(col_group) * 4;
auto simd_lane = tid & 31u;
auto simd_id = tid >> 5;
threadgroup float simd_partial[4][8];

float sum0 = 0.0f;
float sum1 = 0.0f;
float sum2 = 0.0f;
float sum3 = 0.0f;

for (int k = static_cast<int>(tid); k < InDim; k += 256) {
    const float xv = x[k];
    int block_in_row = k >> 8;
    int within_block = k & 255;

    if (base_col < OutDim) {
        auto block0 = w + (base_col * BlocksPerRow + block_in_row) * 210;
        sum0 += xv * tensorsharp_dequant_q6k(block0, within_block);
    }
    if (base_col + 1 < OutDim) {
        auto block1 = w + ((base_col + 1) * BlocksPerRow + block_in_row) * 210;
        sum1 += xv * tensorsharp_dequant_q6k(block1, within_block);
    }
    if (base_col + 2 < OutDim) {
        auto block2 = w + ((base_col + 2) * BlocksPerRow + block_in_row) * 210;
        sum2 += xv * tensorsharp_dequant_q6k(block2, within_block);
    }
    if (base_col + 3 < OutDim) {
        auto block3 = w + ((base_col + 3) * BlocksPerRow + block_in_row) * 210;
        sum3 += xv * tensorsharp_dequant_q6k(block3, within_block);
    }
}

float s0 = simd_sum(sum0);
float s1 = simd_sum(sum1);
float s2 = simd_sum(sum2);
float s3 = simd_sum(sum3);
if (simd_lane == 0) {
    simd_partial[0][simd_id] = s0;
    simd_partial[1][simd_id] = s1;
    simd_partial[2][simd_id] = s2;
    simd_partial[3][simd_id] = s3;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

float lane_v0 = simd_lane < 8u ? simd_partial[0][simd_lane] : 0.0f;
float lane_v1 = simd_lane < 8u ? simd_partial[1][simd_lane] : 0.0f;
float lane_v2 = simd_lane < 8u ? simd_partial[2][simd_lane] : 0.0f;
float lane_v3 = simd_lane < 8u ? simd_partial[3][simd_lane] : 0.0f;
float final0 = simd_sum(lane_v0);
float final1 = simd_sum(lane_v1);
float final2 = simd_sum(lane_v2);
float final3 = simd_sum(lane_v3);

if (tid == 0) {
    if (base_col < OutDim) {
        y[base_col] = final0;
    }
    if (base_col + 1 < OutDim) {
        y[base_col + 1] = final1;
    }
    if (base_col + 2 < OutDim) {
        y[base_col + 2] = final2;
    }
    if (base_col + 3 < OutDim) {
        y[base_col + 3] = final3;
    }
}
";

        // Q6_K matrix-vector product, a port of ggml-metal's kernel_mul_mv_q6_K_f32
        // (ggml, MIT License; ExternalProjects/ggml src/ggml-metal/kernels/mul_mv.metal
        // at 353b63b4): two simdgroups of 32 threads per threadgroup, two output rows
        // per simdgroup, each thread reading 16 activations and 4 packed bytes per
        // block half. The kernels above dequantize one element per thread and read
        // its scale and high bits one at a time, which left Qwen3.8-27B decode at
        // 8.2 tok/s on Q6_K against 14.0 on the lossy 8-bit regroup. Only the row
        // guard differs from ggml: MLX buffers are not padded past the last row.
        // One threadgroup per 4 output rows (grid.x) and one per input row (grid.y).
        private const string Q6KMatvecSource = @"
const int nb = in_dim / 256;
const uint row_bytes = uint(nb) * 210u;
const int first_row = (int(threadgroup_position_in_grid.x) * 2 + int(simdgroup_index_in_threadgroup)) * 2;
const uint r1 = threadgroup_position_in_grid.y;
const ushort tiisg = thread_index_in_simdgroup;
if (first_row >= out_dim) {
    return;
}
const int nrows = min(2, out_dim - first_row);

device const uint8_t * rows0 = w + uint64_t(first_row) * row_bytes;
device const float * yy = x + uint64_t(r1) * uint64_t(in_dim);

float sumf[2] = { 0.f, 0.f };
float yl[16];

const short tid = tiisg / 2;
const short ix = tiisg % 2;
const short ip = tid / 8;
const short il = tid % 8;
const short l0 = 4 * il;
const short is = 8 * ip + l0 / 16;

const short y_offset = 128 * ip + l0;
const short q_offset_l = 64 * ip + l0;
const short q_offset_h = 32 * ip + l0;

for (int i = ix; i < nb; i += 2) {
    device const uint8_t * blk = rows0 + uint(i) * 210u;
    device const uint8_t * q1 = blk + q_offset_l;
    device const uint8_t * q2 = q1 + 32;
    device const uint8_t * qh = blk + 128 + q_offset_h;
    device const int8_t * sc = (device const int8_t *)(blk + 192) + is;
    device const half * dh = (device const half *)(blk + 208);

    device const float * yb = yy + i * 256 + y_offset;
    for (short l = 0; l < 4; ++l) {
        yl[4 * l + 0] = yb[l + 0];
        yl[4 * l + 1] = yb[l + 32];
        yl[4 * l + 2] = yb[l + 64];
        yl[4 * l + 3] = yb[l + 96];
    }

    for (short row = 0; row < 2; ++row) {
        if (row < nrows) {
            float4 sums = { 0.f, 0.f, 0.f, 0.f };
            for (short l = 0; l < 4; ++l) {
                sums[0] += yl[4 * l + 0] * ((int8_t)((q1[l] & 0xF) | ((qh[l] & 0x03) << 4)) - 32);
                sums[1] += yl[4 * l + 1] * ((int8_t)((q2[l] & 0xF) | ((qh[l] & 0x0C) << 2)) - 32);
                sums[2] += yl[4 * l + 2] * ((int8_t)((q1[l] >> 4) | ((qh[l] & 0x30) << 0)) - 32);
                sums[3] += yl[4 * l + 3] * ((int8_t)((q2[l] >> 4) | ((qh[l] & 0xC0) >> 2)) - 32);
            }
            sumf[row] += float(dh[0]) * (sums[0] * sc[0] + sums[1] * sc[2] + sums[2] * sc[4] + sums[3] * sc[6]);
        }
        q1 += row_bytes;
        q2 += row_bytes;
        qh += row_bytes;
        sc += row_bytes;
        dh += row_bytes / 2;
    }
}

for (short row = 0; row < nrows; ++row) {
    float total = simd_sum(sumf[row]);
    if (tiisg == 0) {
        y[uint64_t(r1) * uint64_t(out_dim) + uint64_t(first_row + row)] = total;
    }
}
";

        // IQ4_XS matrix-vector product, a port of ggml-metal's kernel_mul_mv_iq4_xs_f32
        // (same revision and license as the Q6_K port above): two simdgroups per
        // threadgroup, two output rows per simdgroup, the 16-entry codebook in
        // threadgroup memory, and each thread reading 16 activations as float4s and two
        // packed words per block. The kernels it replaces for decode reached ~72 GB/s on
        // a 17408x5120 weight (M5 Pro) against ~245 for the other K-quants, which made
        // IQ4_XS ffn_gate the largest cost of a Qwen3.8-27B UD-Q4_K_XL decode step. As in
        // the Q6_K port, rows past the matrix end are guarded instead of read.
        private const string Iq4XsMatvecSource = @"
threadgroup float codebook[32];
const ushort tiisg = thread_index_in_simdgroup;
codebook[tiisg] = kIq4NlValues[tiisg % 16];
threadgroup_barrier(mem_flags::mem_threadgroup);

const int nb = in_dim / 256;
const uint row_bytes = uint(nb) * 136u;
const int first_row = (int(threadgroup_position_in_grid.x) * 2 + int(simdgroup_index_in_threadgroup)) * 2;
const uint r1 = threadgroup_position_in_grid.y;
if (first_row >= out_dim) {
    return;
}
const int nrows = min(2, out_dim - first_row);

const short ix = tiisg / 16;
const short it = tiisg % 16;
const short ib = it / 2;
const short il = it % 2;

device const uint8_t * rows0 = w + uint64_t(first_row) * row_bytes;
device const float * yb = x + uint64_t(r1) * uint64_t(in_dim) + ix * 256 + ib * 32 + il * 8;

float4 yl[4];
float sumf[2] = { 0.f, 0.f };
uint32_t aux32[2];
thread const uint8_t * q8 = (thread const uint8_t *)aux32;
float4 qf1;
float4 qf2;

for (int ibl = ix; ibl < nb; ibl += 2) {
    device const float4 * y4 = (device const float4 *)yb;
    yl[0] = y4[0];
    yl[1] = y4[4];
    yl[2] = y4[1];
    yl[3] = y4[5];

    for (short row = 0; row < 2; ++row) {
        if (row < nrows) {
            device const uint8_t * xb = rows0 + uint(row) * row_bytes + uint(ibl) * 136u;
            device const uint32_t * q4 = (device const uint32_t *)(xb + 8 + 16 * ib + 8 * il);

            float4 acc1 = { 0.f, 0.f, 0.f, 0.f };
            float4 acc2 = { 0.f, 0.f, 0.f, 0.f };

            aux32[0] = (q4[0]) & 0x0f0f0f0f;
            aux32[1] = (q4[0] >> 4) & 0x0f0f0f0f;
            qf1 = float4(codebook[q8[0]], codebook[q8[1]], codebook[q8[2]], codebook[q8[3]]);
            qf2 = float4(codebook[q8[4]], codebook[q8[5]], codebook[q8[6]], codebook[q8[7]]);
            acc1 += yl[0] * qf1;
            acc2 += yl[1] * qf2;

            aux32[0] = (q4[1]) & 0x0f0f0f0f;
            aux32[1] = (q4[1] >> 4) & 0x0f0f0f0f;
            qf1 = float4(codebook[q8[0]], codebook[q8[1]], codebook[q8[2]], codebook[q8[3]]);
            qf2 = float4(codebook[q8[4]], codebook[q8[5]], codebook[q8[6]], codebook[q8[7]]);
            acc1 += yl[2] * qf1;
            acc2 += yl[3] * qf2;

            acc1 += acc2;

            const uint scales_l = xb[4 + ib / 2];
            const uint scales_h = *(device const uint16_t *)(xb + 2);
            const int ls = int(((scales_l >> (4 * (ib % 2))) & 0xf) | (((scales_h >> (2 * ib)) & 3) << 4)) - 32;
            sumf[row] += float(*(device const half *)xb) * float(ls) * (acc1[0] + acc1[1] + acc1[2] + acc1[3]);
        }
    }

    yb += 2 * 256;
}

for (short row = 0; row < nrows; ++row) {
    float total = simd_sum(sumf[row]);
    if (tiisg == 0) {
        y[uint64_t(r1) * uint64_t(out_dim) + uint64_t(first_row + row)] = total;
    }
}
";

        // Q6_K rows [row0, row0 + rows) of a [outDim, inDim] weight to F16, laid out
        // like ggml's dequantize_row_q6_K: each thread produces the four values of one
        // (half-block, l) position, so a warp writes 32 consecutive halves per store.
        // Grid: x over (inDim / 256) * 64 positions per row, y over rows.
        private const string Q6KDequantF16Source = @"
const uint gid = thread_position_in_grid.x;
const uint row = thread_position_in_grid.y;
const uint nb = uint(in_dim) / 256u;
if (gid >= nb * 64u) {
    return;
}
const uint b = gid / 64u;
const uint t = gid % 64u;
const uint n = t / 32u;
const uint l = t % 32u;
device const uint8_t * blk = w + (uint64_t(uint(row0) + row) * uint64_t(nb) + uint64_t(b)) * 210u;
device const uint8_t * ql = blk + n * 64u;
device const uint8_t * qh = blk + 128u + n * 32u;
device const int8_t * sc = (device const int8_t *)(blk + 192u) + n * 8u;
const float d = float(*(device const half *)(blk + 208u));
const uint is = l / 16u;
const int q1 = int((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
const int q2 = int((ql[l + 32u] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
const int q3 = int((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
const int q4 = int((ql[l + 32u] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
device half * o = y + uint64_t(row) * uint64_t(in_dim) + uint64_t(b * 256u + n * 128u + l);
o[0] = half(d * float(sc[is + 0]) * float(q1));
o[32] = half(d * float(sc[is + 2]) * float(q2));
o[64] = half(d * float(sc[is + 4]) * float(q3));
o[96] = half(d * float(sc[is + 6]) * float(q4));
";

        // IQ4_XS rows [row0, row0 + rows) to F16, laid out like ggml's
        // dequantize_row_iq4_xs: each thread decodes one byte of a 32-value sub-block
        // into its low-nibble value j and high-nibble value j + 16. Grid: x over
        // (inDim / 256) * 128 bytes per row, y over rows.
        private const string Iq4XsDequantF16Source = @"
const uint gid = thread_position_in_grid.x;
const uint row = thread_position_in_grid.y;
const uint nb = uint(in_dim) / 256u;
if (gid >= nb * 128u) {
    return;
}
const uint b = gid / 128u;
const uint t = gid % 128u;
const uint ib = t / 16u;
const uint j = t % 16u;
device const uint8_t * blk = w + (uint64_t(uint(row0) + row) * uint64_t(nb) + uint64_t(b)) * 136u;
const float d = float(*(device const half *)blk);
const uint scales_h = *(device const uint16_t *)(blk + 2);
const uint scales_l = blk[4u + ib / 2u];
const int ls = int(((scales_l >> (4u * (ib % 2u))) & 0xfu) | (((scales_h >> (2u * ib)) & 3u) << 4)) - 32;
const float dl = d * float(ls);
const uint q = blk[8u + 16u * ib + j];
device half * o = y + uint64_t(row) * uint64_t(in_dim) + uint64_t(b * 256u + ib * 32u + j);
o[0] = half(dl * kIq4NlValues[q & 0xfu]);
o[16] = half(dl * kIq4NlValues[q >> 4]);
";

        private const string Q4KGetRowsSource = @"
auto col = thread_position_in_grid.x;
auto out_row = thread_position_in_grid.y;
if (col >= InDim) {
    return;
}

int weight_row = indices[out_row];
int block_in_row = col >> 8;
int within_block = col & 255;
auto block = w + (weight_row * BlocksPerRow + block_in_row) * 144;
y[out_row * InDim + col] = tensorsharp_dequant_q4k(block, within_block);
";

        private const string Q5KGetRowsSource = @"
auto col = thread_position_in_grid.x;
auto out_row = thread_position_in_grid.y;
if (col >= InDim) {
    return;
}

int weight_row = indices[out_row];
int block_in_row = col >> 8;
int within_block = col & 255;
auto block = w + (weight_row * BlocksPerRow + block_in_row) * 176;
y[out_row * InDim + col] = tensorsharp_dequant_q5k(block, within_block);
";

        private const string Q6KGetRowsSource = @"
auto col = thread_position_in_grid.x;
auto out_row = thread_position_in_grid.y;
if (col >= InDim) {
    return;
}

int weight_row = indices[out_row];
int block_in_row = col >> 8;
int within_block = col & 255;
auto block = w + (weight_row * BlocksPerRow + block_in_row) * 210;
y[out_row * InDim + col] = tensorsharp_dequant_q6k(block, within_block);
";

        // GatedDelta decode-specialized kernel — assumes T=1 (single
        // token). Drops the outer time loop, hoists g/beta/v scalars
        // outside the dot-product loops (the templated T-loop version
        // re-reads them inside each iteration; for decode this is
        // visible because there's only one iteration to amortize them
        // over), and uses explicit `fma` so the compiler emits fused
        // multiply-adds without relying on ffp-contract guesses.
        //
        // Grid / threadgroup shape is identical to the templated kernel
        // (32, Dv, B*Hv) / (32, min(Dv,4), 1) so the caller code is the
        // same — just a different EnsureKernel + ApplyTemplate path.
        private const string GatedDeltaT1Source = @"
auto n = thread_position_in_grid.z;
auto b_idx = n / Hv;
auto hv_idx = n % Hv;
auto hk_idx = hv_idx % Hk;
constexpr int n_per_t = Dk / 32;

auto dk_idx = thread_position_in_threadgroup.x;
auto dv_idx = thread_position_in_grid.y;

// Per-(n, dv_idx) state slice in device memory.
auto i_state = state_in + (n * Dv + dv_idx) * Dk;
auto o_state = state_out + (n * Dv + dv_idx) * Dk;

// q, k: [B, T=1, Hk, Dk] — single-token offset for this (b, hk).
auto q_ = q + b_idx * Hk * Dk + hk_idx * Dk;
auto k_ = k + b_idx * Hk * Dk + hk_idx * Dk;

// v, y: [B, T=1, Hv, Dv] — single-token offset for this (b, hv).
auto v_row = v + b_idx * Hv * Dv + hv_idx * Dv;
auto y_row = y + b_idx * Hv * Dv + hv_idx * Dv;

// Pre-load the scalars that are constant for this thread (one g
// per (b, hv), one beta per (b, hv), one v per (b, hv, dv_idx)).
const float g_scalar    = g[b_idx * Hv + hv_idx];
const float beta_scalar = beta[b_idx * Hv + hv_idx];
const float v_scalar    = v_row[dv_idx];

// Load per-thread state (n_per_t=4 contiguous elements).
float state[n_per_t];
for (int i = 0; i < n_per_t; ++i) {
  state[i] = i_state[n_per_t * dk_idx + i];
}

// Phase 1: state *= g, then kv_mem = state . k_
float kv_mem = 0.0f;
for (int i = 0; i < n_per_t; ++i) {
  auto s_idx = n_per_t * dk_idx + i;
  state[i] = state[i] * g_scalar;
  kv_mem = fma(state[i], k_[s_idx], kv_mem);
}
kv_mem = simd_sum(kv_mem);

const float delta = (v_scalar - kv_mem) * beta_scalar;

// Phase 2: state += k_ * delta, then out = state . q_
float out_val = 0.0f;
for (int i = 0; i < n_per_t; ++i) {
  auto s_idx = n_per_t * dk_idx + i;
  state[i] = fma(k_[s_idx], delta, state[i]);
  out_val = fma(state[i], q_[s_idx], out_val);
}
out_val = simd_sum(out_val);

// One write per simdgroup (32 threads collaborate on one out element).
if (thread_index_in_simdgroup == 0) {
  y_row[dv_idx] = out_val;
}

// Persist updated state for the next decode step.
for (int i = 0; i < n_per_t; ++i) {
  o_state[n_per_t * dk_idx + i] = state[i];
}
";

        private const string GatedDeltaSource = @"
auto n = thread_position_in_grid.z;
auto b_idx = n / Hv;
auto hv_idx = n % Hv;
auto hk_idx = hv_idx % Hk;
constexpr int n_per_t = Dk / 32;

// q, k: [B, T, Hk, Dk]
auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

// v, y: [B, T, Hv, Dv]
auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
y += b_idx * T * Hv * Dv + hv_idx * Dv;

auto dk_idx = thread_position_in_threadgroup.x;
auto dv_idx = thread_position_in_grid.y;

// state_in, state_out: [B, Hv, Dv, Dk]
auto i_state = state_in + (n * Dv + dv_idx) * Dk;
auto o_state = state_out + (n * Dv + dv_idx) * Dk;

float state[n_per_t];
for (int i = 0; i < n_per_t; ++i) {
  auto s_idx = n_per_t * dk_idx + i;
  state[i] = i_state[s_idx];
}

// g, beta: [B, T, Hv]
auto g_ = g + b_idx * T * Hv;
auto beta_ = beta + b_idx * T * Hv;

for (int t = 0; t < T; ++t) {
  float kv_mem = 0.0f;
  for (int i = 0; i < n_per_t; ++i) {
    auto s_idx = n_per_t * dk_idx + i;
    state[i] = state[i] * g_[hv_idx];
    kv_mem += state[i] * k_[s_idx];
  }
  kv_mem = simd_sum(kv_mem);

  auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

  float out = 0.0f;
  for (int i = 0; i < n_per_t; ++i) {
    auto s_idx = n_per_t * dk_idx + i;
    state[i] = state[i] + k_[s_idx] * delta;
    out += state[i] * q_[s_idx];
  }
  out = simd_sum(out);
  if (thread_index_in_simdgroup == 0) {
    y[dv_idx] = out;
  }

  q_ += Hk * Dk;
  k_ += Hk * Dk;
  v_ += Hv * Dv;
  y += Hv * Dv;
  g_ += Hv;
  beta_ += Hv;
}

for (int i = 0; i < n_per_t; ++i) {
  auto s_idx = n_per_t * dk_idx + i;
  o_state[s_idx] = state[i];
}
";

        // Gated DeltaNet prefill: the same sequential recurrence as GatedDeltaSource,
        // restructured for Apple GPUs. Adapted from omlx's gated_delta_blocked_seq
        // (omlx/custom_kernels/qwen35_prefill/gdn.py, Apache-2.0, jundot/omlx):
        //   * one threadgroup per (32 value rows, value head) instead of one
        //     simdgroup per row, so each q/k row is read from device memory once per
        //     threadgroup rather than once per value row;
        //   * q/k/v/g/beta staged into threadgroup memory TB time steps at a time;
        //   * each thread keeps a 16-wide slice of its state row in registers and
        //     the dot products reduce across the row's 8 threads with simd shuffles.
        // Changes for TensorSharp: F32 inputs (TB = 16 keeps the staging under the
        // 32 KiB threadgroup limit), and value head hv reads key head hv % Hk — the
        // tiled head order Qwen35GdnPreprocessPacked writes, as GatedDeltaSource
        // does. T is a runtime scalar, so a new prompt length reuses the compiled
        // kernel instead of building another. Requires Dk == 128 and Dv % 32 == 0.
        // Measured on an M5 Pro against GatedDeltaSource (Hk 16, Hv 32, D 128):
        // 1.09 vs 2.16 ms at T = 512, 6.13 vs 17.44 ms at T = 4096, outputs equal to
        // 2e-7.
        private const string GatedDeltaBlockedSource = @"
constexpr int TB = 16;
constexpr int DB = 32;
const int tid = thread_position_in_threadgroup.x;
const int blk = threadgroup_position_in_grid.x;
const int hv  = threadgroup_position_in_grid.y;
const int b   = threadgroup_position_in_grid.z;
const int hk  = hv % Hk;
const int dv0 = blk * DB;

const int dv  = tid / 8;
const int seg = tid % 8;
const int d0  = seg * 16;

threadgroup float k_s[TB][Dk + 8];
threadgroup float q_s[TB][Dk + 8];
threadgroup float v_s[TB][DB + 8];
threadgroup float g_s[TB];
threadgroup float b_s[TB];

const device float* k_base = k + ((size_t)b * T * Hk + hk) * Dk;
const device float* q_base = q + ((size_t)b * T * Hk + hk) * Dk;
const device float* v_base = v + ((size_t)b * T * Hv + hv) * Dv + dv0;
const size_t krow = (size_t)Hk * Dk;

float4 st[4];
{
    const device float4* S_in = (const device float4*)(
        state_in + (((size_t)b * Hv + hv) * Dv + dv0 + dv) * Dk + d0);
    for (int i = 0; i < 4; ++i) st[i] = S_in[i];
}

device float* y_base = y + ((size_t)b * T * Hv + hv) * Dv + dv0;

for (int t0 = 0; t0 < T; t0 += TB) {
    const int tt = min(TB, T - t0);
    for (int p = tid; p < tt * Dk; p += 256) {
        const int r = p / Dk, d = p % Dk;
        k_s[r][d] = k_base[(size_t)(t0 + r) * krow + d];
        q_s[r][d] = q_base[(size_t)(t0 + r) * krow + d];
    }
    for (int p = tid; p < tt * DB; p += 256) {
        const int r = p / DB, d = p % DB;
        v_s[r][d] = v_base[(size_t)(t0 + r) * Hv * Dv + d];
    }
    for (int p = tid; p < tt; p += 256) {
        g_s[p] = g[((size_t)b * T + t0 + p) * Hv + hv];
        b_s[p] = beta[((size_t)b * T + t0 + p) * Hv + hv];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int t = 0; t < tt; ++t) {
        const float gt = g_s[t];
        const float bt = b_s[t];
        const threadgroup float4* k4 = (const threadgroup float4*)&k_s[t][d0];
        const threadgroup float4* q4 = (const threadgroup float4*)&q_s[t][d0];
        float4 kf[4];
        for (int i = 0; i < 4; ++i) kf[i] = k4[i];
        float4 p4 = 0.0f;
        for (int i = 0; i < 4; ++i) {
            st[i] *= gt;
            p4 += st[i] * kf[i];
        }
        float part = p4.x + p4.y + p4.z + p4.w;
        part += simd_shuffle_down(part, 4);
        part += simd_shuffle_down(part, 2);
        part += simd_shuffle_down(part, 1);
        const float kv_mem = simd_shuffle(part, (tid % 32) / 8 * 8);
        const float delta = (v_s[t][dv] - kv_mem) * bt;

        float4 o4 = 0.0f;
        for (int i = 0; i < 4; ++i) {
            st[i] += kf[i] * delta;
            o4 += st[i] * q4[i];
        }
        float out = o4.x + o4.y + o4.z + o4.w;
        out += simd_shuffle_down(out, 4);
        out += simd_shuffle_down(out, 2);
        out += simd_shuffle_down(out, 1);
        if (seg == 0) {
            y_base[(size_t)(t0 + t) * Hv * Dv + dv] = out;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

{
    device float4* S_out = (device float4*)(
        state_out + (((size_t)b * Hv + hv) * Dv + dv0 + dv) * Dk + d0);
    for (int i = 0; i < 4; ++i) S_out[i] = st[i];
}
";

        private const string ScatterAddWeightedRowsSource = @"
auto col = thread_position_in_grid.x;
auto out_row = thread_position_in_grid.y;
if (col >= HiddenDim || out_row >= SeqLen) {
    return;
}

float value = in_y[out_row * HiddenDim + col];
int lo = 0;
int hi = BatchSize - 1;
while (lo <= hi) {
    int mid = (lo + hi) >> 1;
    int idx = indices[mid];
    if (idx < out_row) {
        lo = mid + 1;
    } else if (idx > out_row) {
        hi = mid - 1;
    } else {
        value += rows[mid * HiddenDim + col] * weights[mid];
        break;
    }
}

out_y[out_row * HiddenDim + col] = value;
";

        private const string RmsNormAddSource = @"
// Phase 6e: simdgroup-fast reduction. Drops the 8-barrier tree reduction
// to a single barrier across the 8 per-simdgroup sums.
auto tid = thread_position_in_threadgroup.x;
auto row = thread_position_in_grid.y;
auto simd_lane = tid & 31u;
auto simd_id = tid >> 5;
threadgroup float simd_partial[8];

float sum = 0.0f;
for (int col = static_cast<int>(tid); col < HiddenDim; col += 256) {
    float value = input[row * HiddenDim + col];
    sum += value * value;
}

float simd_total = simd_sum(sum);
if (simd_lane == 0) simd_partial[simd_id] = simd_total;
threadgroup_barrier(mem_flags::mem_threadgroup);
float lane_v = simd_lane < 8u ? simd_partial[simd_lane] : 0.0f;
float total = simd_sum(lane_v);

float scale = rsqrt(total / static_cast<float>(HiddenDim) + eps_value);
for (int col = static_cast<int>(tid); col < HiddenDim; col += 256) {
    int offset = row * HiddenDim + col;
    out_y[offset] = residual[offset] + input[offset] * scale * norm_weight[col];
}
";

        // Fused (residual += input; normed = RmsNorm(residual, norm_weight)).
        // Used by pre-norm transformer blocks (Mistral3 / Qwen35 / Nemotron /
        // GptOss): folds the residual-add + post-add rmsnorm into one
        // dispatch, eliminating one MLX kernel launch per residual stage
        // (2 stages per layer × 40 layers = 80 saved dispatches/token on
        // 40-layer models).
        // Outputs:
        //   updated_residual = residual_in + input
        //   normed_out      = RmsNorm(updated_residual, norm_weight)
        private const string AddRmsNormSource = @"
// Phase 6e: simdgroup-fast reduction.
auto tid = thread_position_in_threadgroup.x;
auto row = thread_position_in_grid.y;
auto simd_lane = tid & 31u;
auto simd_id = tid >> 5;
threadgroup float simd_partial[8];

float sum = 0.0f;
for (int col = static_cast<int>(tid); col < HiddenDim; col += 256) {
    int offset = row * HiddenDim + col;
    float val = residual[offset] + input[offset];
    updated_residual[offset] = val;
    sum += val * val;
}

float simd_total = simd_sum(sum);
if (simd_lane == 0) simd_partial[simd_id] = simd_total;
threadgroup_barrier(mem_flags::mem_threadgroup);
float lane_v = simd_lane < 8u ? simd_partial[simd_lane] : 0.0f;
float total = simd_sum(lane_v);

float scale = rsqrt(total / static_cast<float>(HiddenDim) + eps_value);
for (int col = static_cast<int>(tid); col < HiddenDim; col += 256) {
    int offset = row * HiddenDim + col;
    normed_out[offset] = updated_residual[offset] * scale * norm_weight[col];
}
";

        private const string GeluMulSplitSource = @"
auto col = thread_position_in_grid.x;
auto row = thread_position_in_grid.y;
if (col >= HalfDim || row >= Rows) {
    return;
}

int base = row * (HalfDim * 2);
float gate = gate_up[base + col];
float up = gate_up[base + HalfDim + col];
float gate3 = gate * gate * gate;
// Clamped for the same reason as the fused Q8 GELU kernel: from MLX 0.32 on
// macOS 27 (MSL 4.1), a bare tanh() in a custom kernel is the fast
// exp-ratio form and returns NaN once |x| passes ~44. Gemma 4 E4B's layer-0
// FFN reaches |gate| ~ 680 on ordinary prompts. tanh is +/-1 to float
// precision well before 15, so the clamp is exact under either semantics.
float inner = clamp(0.7978845608f * (gate + 0.044715f * gate3), -15.0f, 15.0f);
float gelu = 0.5f * gate * (1.0f + tanh(inner));
out_y[row * HalfDim + col] = gelu * up;
";

        // Clamped SwiGLU (the GPT-OSS / OpenAI-MoE "swiglu_oai" variant) over
        // separate gate / up matrices with a fused per-expert bias gather:
        //   g = gate[row] + gate_bias[experts[row]]
        //   u = up[row]   + up_bias[experts[row]]
        //   x = min(g, limit); y = clamp(u, -limit, limit)
        //   out = (x * sigmoid(alpha * x)) * (y + 1)
        // Mirrors GptOssModel.ApplySwiGluOaiInPlace / ggml's swiglu_oai.
        // alpha_v / limit_v are 0-d scalar inputs (passed by value by MLX).
        private const string SwigluOaiGatherBiasSource = @"
auto col = thread_position_in_grid.x;
auto row = thread_position_in_grid.y;
if (col >= Dim || row >= Rows) {
    return;
}

int e = experts[row];
float g = gate_in[row * Dim + col] + gate_bias[e * Dim + col];
float u = up_in[row * Dim + col] + up_bias[e * Dim + col];
float x = min(g, limit_v);
float y = clamp(u, -limit_v, limit_v);
float glu = x / (1.0f + exp(-alpha_v * x));
out_y[row * Dim + col] = glu * (y + 1.0f);
";

        // Routing-weighted MoE combine with a fused per-expert down-bias
        // gather and unsort. The K down-projection rows of token n live at
        // sorted positions inv_order[n*K+k] of down_rows (the caller sorted
        // the (token, expert) pairs by expert for gather_qmm's grouped-GEMM
        // mode). Computes, per token n and output column d:
        //   out[n, d] = sum_k w[n*K+k] * (down_rows[inv_order[n*K+k], d]
        //                                 + down_bias[experts_sorted[inv_order[n*K+k]], d])
        // HasBias=0 skips the bias term (down_bias may then be any array).
        private const string MoeBiasWeightedSumSource = @"
auto col = thread_position_in_grid.x;
auto n = thread_position_in_grid.y;
if (col >= Dim || n >= Rows) {
    return;
}

float acc = 0.0f;
for (int k = 0; k < K; ++k) {
    int p = n * K + k;
    int srow = inv_order[p];
    float v = down_rows[srow * Dim + col];
    if (HasBias != 0) {
        v += down_bias[experts_sorted[srow] * Dim + col];
    }
    acc += route_weights[p] * v;
}
out_y[n * Dim + col] = acc;
";

        private const string FlatToHeadFirstSource = @"
auto col = thread_position_in_grid.x;
auto seq = thread_position_in_grid.y;
auto head = thread_position_in_grid.z;
if (col >= HeadDim || seq >= SeqLen || head >= NumHeads) {
    return;
}

int src_offset = seq * SourceStride + ColOffset + head * HeadDim + col;
int dst_offset = (head * SeqLen + seq) * HeadDim + col;
out_y[dst_offset] = input[src_offset];
";

        private const string NeoXRopeSource = @"
auto col = thread_position_in_grid.x;
auto head = thread_position_in_grid.y;
auto seq = thread_position_in_grid.z;
if (col >= HeadDim || head >= NumHeads || seq >= SeqLen) {
    return;
}

int base;
int offset;
if (HeadFirst != 0) {
    base = (head * SeqLen + seq) * HeadDim;
    offset = base + col;
} else {
    base = (seq * NumHeads + head) * HeadDim;
    offset = base + col;
}

float value = input[offset];
if (col < RotHalf) {
    float x0 = input[base + col];
    float x1 = input[base + RotHalf + col];
    float c = cos_table[seq * RotHalf + col];
    float s = sin_table[seq * RotHalf + col];
    value = x0 * c - x1 * s;
} else if (col < RotHalf * 2) {
    int j = static_cast<int>(col) - RotHalf;
    float x0 = input[base + j];
    float x1 = input[base + RotHalf + j];
    float c = cos_table[seq * RotHalf + j];
    float s = sin_table[seq * RotHalf + j];
    value = x0 * s + x1 * c;
}

out_y[offset] = value;
";

        private const string Qwen35GdnPreprocessSource = @"
auto idx = thread_position_in_grid.x;
auto t = thread_position_in_grid.y;
auto kind = thread_position_in_grid.z;

if (kind == 0) {
    if (t >= T || idx >= QkvDim) {
        return;
    }

    float acc = 0.0f;
    for (int ki = 0; ki < Kernel; ki++) {
        int src_row = static_cast<int>(t) + ki;
        float xval = 0.0f;
        if (src_row < Tail) {
            xval = conv_state[src_row * QkvDim + idx];
        } else {
            xval = qkv_raw[(src_row - Tail) * QkvDim + idx];
        }

        float wval = 0.0f;
        if (ConvWeightChannelMajor != 0) {
            wval = conv_weight[idx * Kernel + ki];
        } else {
            wval = conv_weight[ki * QkvDim + idx];
        }
        acc += xval * wval;
    }

    float silu = acc / (1.0f + exp(-acc));
    if (idx < KeyDim) {
        int h = static_cast<int>(idx) / HeadKeyDim;
        int d = static_cast<int>(idx) - h * HeadKeyDim;
        q_out[(t * NumKeyHeads + h) * HeadKeyDim + d] = silu;
    } else if (idx < 2 * KeyDim) {
        int local = static_cast<int>(idx) - KeyDim;
        int h = local / HeadKeyDim;
        int d = local - h * HeadKeyDim;
        k_out[(t * NumKeyHeads + h) * HeadKeyDim + d] = silu;
    } else if (idx < 2 * KeyDim + ValueDim) {
        int local = static_cast<int>(idx) - 2 * KeyDim;
        int h = local / HeadValueDim;
        int d = local - h * HeadValueDim;
        v_out[(t * NumValueHeads + h) * HeadValueDim + d] = silu;
    }
    return;
}

if (kind == 1) {
    if (t >= T) {
        return;
    }

    if (idx < ValueDim) {
        float zv = z_raw[t * ValueDim + idx];
        z_silu[(t * NumValueHeads * HeadValueDim) + idx] = zv / (1.0f + exp(-zv));
    }

    if (idx < NumValueHeads) {
        float av = alpha_raw[t * NumValueHeads + idx] + dt_bias[idx];
        float sp = av > 20.0f ? av : log(1.0f + exp(av));
        g_out[t * NumValueHeads + idx] = exp(sp * a_log[idx]);

        float bv = beta_raw[t * NumValueHeads + idx];
        beta_out[t * NumValueHeads + idx] = 1.0f / (1.0f + exp(-bv));
    }
    return;
}

if (kind == 2) {
    if (t >= Tail || idx >= QkvDim) {
        return;
    }

    int row = static_cast<int>(t);
    int col = static_cast<int>(idx);
    int src_row = T + row;
    if (src_row < Tail) {
        next_conv[row * QkvDim + col] = conv_state[src_row * QkvDim + col];
    } else {
        next_conv[row * QkvDim + col] = qkv_raw[(src_row - Tail) * QkvDim + col];
    }
}
";

        private const string Qwen35GdnPackedPreprocessSource = @"
auto idx = thread_position_in_grid.x;
auto t = thread_position_in_grid.y;
auto kind = thread_position_in_grid.z;

if (kind == 0) {
    if (t >= T || idx >= QkvDim) {
        return;
    }

    float acc = 0.0f;
    for (int ki = 0; ki < Kernel; ki++) {
        int src_row = static_cast<int>(t) + ki;
        float xval = 0.0f;
        if (src_row < Tail) {
            xval = conv_state[src_row * QkvDim + idx];
        } else {
            xval = packed_raw[(src_row - Tail) * PackedDim + idx];
        }

        float wval = 0.0f;
        if (ConvWeightChannelMajor != 0) {
            wval = conv_weight[idx * Kernel + ki];
        } else {
            wval = conv_weight[ki * QkvDim + idx];
        }
        acc += xval * wval;
    }

    float silu = acc / (1.0f + exp(-acc));
    if (idx < KeyDim) {
        int h = static_cast<int>(idx) / HeadKeyDim;
        int d = static_cast<int>(idx) - h * HeadKeyDim;
        q_out[(t * NumKeyHeads + h) * HeadKeyDim + d] = silu;
    } else if (idx < 2 * KeyDim) {
        int local = static_cast<int>(idx) - KeyDim;
        int h = local / HeadKeyDim;
        int d = local - h * HeadKeyDim;
        k_out[(t * NumKeyHeads + h) * HeadKeyDim + d] = silu;
    } else if (idx < 2 * KeyDim + ValueDim) {
        int local = static_cast<int>(idx) - 2 * KeyDim;
        int h = local / HeadValueDim;
        int d = local - h * HeadValueDim;
        v_out[(t * NumValueHeads + h) * HeadValueDim + d] = silu;
    }
    return;
}

if (kind == 1) {
    if (t >= T) {
        return;
    }

    auto packed_row = packed_raw + t * PackedDim;
    auto z_raw = packed_row + QkvDim;
    auto beta_raw = z_raw + ValueDim;
    auto alpha_raw = beta_raw + NumValueHeads;

    if (idx < ValueDim) {
        float zv = z_raw[idx];
        z_silu[(t * NumValueHeads * HeadValueDim) + idx] = zv / (1.0f + exp(-zv));
    }

    if (idx < NumValueHeads) {
        float av = alpha_raw[idx] + dt_bias[idx];
        float sp = av > 20.0f ? av : log(1.0f + exp(av));
        g_out[t * NumValueHeads + idx] = exp(sp * a_log[idx]);

        float bv = beta_raw[idx];
        beta_out[t * NumValueHeads + idx] = 1.0f / (1.0f + exp(-bv));
    }
    return;
}

if (kind == 2) {
    if (t >= Tail || idx >= QkvDim) {
        return;
    }

    int row = static_cast<int>(t);
    int col = static_cast<int>(idx);
    int src_row = T + row;
    if (src_row < Tail) {
        next_conv[row * QkvDim + col] = conv_state[src_row * QkvDim + col];
    } else {
        next_conv[row * QkvDim + col] = packed_raw[(src_row - Tail) * PackedDim + col];
    }
}
";

        private const string Qwen35GdnPostprocessSource = @"
auto tid = thread_position_in_threadgroup.x;
auto t = thread_position_in_grid.y;
auto h = thread_position_in_grid.z;
threadgroup float partial[256];
threadgroup float values[256];

float sum = 0.0f;
float val = 0.0f;
if (tid < HeadValueDim) {
    int offset = (t * NumValueHeads + h) * HeadValueDim + tid;
    val = y_in[offset];
    values[tid] = val;
    sum = val * val;
} else {
    values[tid] = 0.0f;
}

partial[tid] = sum;
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint stride = 128; stride > 0; stride >>= 1) {
    if (tid < stride) {
        partial[tid] += partial[tid + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

if (tid < HeadValueDim) {
    float scale = rsqrt(partial[0] / static_cast<float>(HeadValueDim) + 1.0e-6f);
    int offset4 = (t * NumValueHeads + h) * HeadValueDim + tid;
    int offset2 = t * ValueDim + h * HeadValueDim + tid;
    y_out[offset2] = values[tid] * scale * norm_weight[tid] * z_silu[offset4];
}
";

        // Multi-row IQ4_NL matmul kernel. One threadgroup per `out_col`
        // processes ALL `Rows` rows at once — each thread loads the weight
        // value ONCE for its (k, out_col) and reuses it across every row,
        // accumulating per-row partial sums in thread-local arrays. Without
        // this kernel, the basic single-row kernel issues `Rows × OutDim`
        // threadgroups per matmul, each independently re-reading the same
        // weight bytes — which is exactly why MLX prefill on Nemotron-H
        // regressed when we shipped the basic IQ4_NL kernel. With this
        // multi-row variant in place prefill goes back to GPU-bound speed
        // (one weight read per (k, out_col) regardless of row count).
        //
        // `Rows` is a template int (recompiled per row-count value, cached
        // by MLX); capped at 16 in the dispatcher so the threadgroup
        // memory budget (Rows × 256 × 4 = 16 KB at Rows=16) stays safely
        // under Apple GPU limits.
        // Batched IQ4_NL MoE matmul — shared input variant. One Metal
        // dispatch produces K outputs, one per (out_col, k_idx), where
        // expert_indices[k_idx] selects the per-expert weight slice from a
        // stacked weight buffer laid out as [numExperts, OutDim,
        // BlocksPerRow * 18] bytes. Input x is shared across all K
        // experts (decode case: one routed token routed to K experts).
        //
        // For each (k_idx, out_col):
        //     y[k_idx, out_col] = dot(x[0, :], W[expert_indices[k_idx], out_col, :])
        // Mirrors `Iq2XxsMoeMatmulBatchedSource` 's grid/threadgroup shape
        // and partial-sum reduction, but with IQ4_NL's 18-byte / 32-element
        // block layout and nibble extraction.
        private const string Iq4NlMoeMatmulBatchedSource = @"
auto tid = thread_position_in_threadgroup.x;
auto out_col = thread_position_in_grid.y;
auto k_idx = thread_position_in_grid.z;
threadgroup float partial[256];

int expert_idx = expert_indices[k_idx];

float sum = 0.0f;
for (int k = static_cast<int>(tid); k < InDim; k += 256) {
    int block_in_row = k >> 5;       // k / 32
    int within_block = k & 31;       // 0..31
    auto block_offset = (uint)expert_idx * OutDim * BlocksPerRow + out_col * BlocksPerRow + block_in_row;
    auto block = w + block_offset * 18;
    const half d_half = *reinterpret_cast<const device half *>(block);
    const int j = within_block & 15;
    const uchar packed = block[2 + j];
    const uchar q = within_block < 16 ? (packed & 0x0f) : (packed >> 4);
    const float weight_value = static_cast<float>(d_half) * kIq4NlValues[q];
    sum += x[k] * weight_value;
}

partial[tid] = sum;
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint stride = 128; stride > 0; stride >>= 1) {
    if (tid < stride) {
        partial[tid] += partial[tid + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

if (tid == 0) {
    y[k_idx * OutDim + out_col] = partial[0];
}
";

        // Batched IQ4_NL MoE matmul — per-row (rowed) variant. Each row k
        // of input is the k-th expert's post-activation output and gets
        // multiplied by the down-projection weight of expert
        // `expert_indices[k]`. Used for the down matmul where K different
        // input rows pair with K (possibly identical) expert ids.
        private const string Iq4NlMoeMatmulBatchedRowedSource = @"
auto tid = thread_position_in_threadgroup.x;
auto out_col = thread_position_in_grid.y;
auto k_idx = thread_position_in_grid.z;
threadgroup float partial[256];

int expert_idx = expert_indices[k_idx];

float sum = 0.0f;
for (int k = static_cast<int>(tid); k < InDim; k += 256) {
    int block_in_row = k >> 5;
    int within_block = k & 31;
    auto block_offset = (uint)expert_idx * OutDim * BlocksPerRow + out_col * BlocksPerRow + block_in_row;
    auto block = w + block_offset * 18;
    const half d_half = *reinterpret_cast<const device half *>(block);
    const int j = within_block & 15;
    const uchar packed = block[2 + j];
    const uchar q = within_block < 16 ? (packed & 0x0f) : (packed >> 4);
    const float weight_value = static_cast<float>(d_half) * kIq4NlValues[q];
    sum += x[k_idx * InDim + k] * weight_value;
}

partial[tid] = sum;
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint stride = 128; stride > 0; stride >>= 1) {
    if (tid < stride) {
        partial[tid] += partial[tid + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

if (tid == 0) {
    y[k_idx * OutDim + out_col] = partial[0];
}
";

        private const string Iq4NlMatmulRowsSource = @"
auto tid = thread_position_in_threadgroup.x;
auto out_col = thread_position_in_grid.y;
threadgroup float partial[Rows][256];
float sums[Rows];

for (int r = 0; r < Rows; r++) {
    sums[r] = 0.0f;
}

for (int k = static_cast<int>(tid); k < InDim; k += 256) {
    int block_in_row = k >> 5;       // k / 32
    int within_block = k & 31;       // 0..31
    auto block = w + (out_col * BlocksPerRow + block_in_row) * 18;
    const half d_half = *reinterpret_cast<const device half *>(block);
    const int j = within_block & 15;
    const uchar packed = block[2 + j];
    const uchar q = within_block < 16 ? (packed & 0x0f) : (packed >> 4);
    const float weight_value = static_cast<float>(d_half) * kIq4NlValues[q];
    for (int r = 0; r < Rows; r++) {
        sums[r] += x[r * InDim + k] * weight_value;
    }
}

for (int r = 0; r < Rows; r++) {
    partial[r][tid] = sums[r];
}
threadgroup_barrier(mem_flags::mem_threadgroup);

for (uint stride = 128; stride > 0; stride >>= 1) {
    if (tid < stride) {
        for (int r = 0; r < Rows; r++) {
            partial[r][tid] += partial[r][tid + stride];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

if (tid == 0) {
    for (int r = 0; r < Rows; r++) {
        y[r * OutDim + out_col] = partial[r][0];
    }
}
";

        // IQ4_NL matmul kernel — 18-byte block per 32 elements, F16 scale + 16
        // qs bytes. Element i (0..31): nibble = qs[i & 15] & 0x0f when i < 16,
        // else qs[i & 15] >> 4. Then value = d * kIq4NlValues[nibble]. Mirrors
        // the layout/threading of Iq4XsMatmulSource (256 threads per (row,
        // out_col), shared-memory reduction across InDim).
        private const string Iq4NlMatmulSource = @"
auto tid = thread_position_in_threadgroup.x;
auto out_col = thread_position_in_grid.y;
auto row_idx = thread_position_in_grid.z;
threadgroup float partial[256];

float sum = 0.0f;
for (int k = static_cast<int>(tid); k < InDim; k += 256) {
    int block_in_row = k >> 5;       // k / 32  (IQ4_NL block size = 32 elements)
    int within_block = k & 31;       // 0..31
    // Each IQ4_NL block is 18 bytes: 2-byte F16 scale + 16-byte qs.
    auto block = w + (out_col * BlocksPerRow + block_in_row) * 18;
    const half d_half = *reinterpret_cast<const device half *>(block);
    const int j = within_block & 15; // qs byte index 0..15
    const uchar packed = block[2 + j];
    const uchar q = within_block < 16 ? (packed & 0x0f) : (packed >> 4);
    const float weight_value = static_cast<float>(d_half) * kIq4NlValues[q];
    sum += x[row_idx * InDim + k] * weight_value;
}

partial[tid] = sum;
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint stride = 128; stride > 0; stride >>= 1) {
    if (tid < stride) {
        partial[tid] += partial[tid + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

if (tid == 0) {
    y[row_idx * OutDim + out_col] = partial[0];
}
";

        // Phase 8: simdgroup-fast reduction (same pattern as Q4KMatmul).
        private const string Iq4XsMatmulSource = @"
auto tid = thread_position_in_threadgroup.x;
auto out_col = thread_position_in_grid.y;
auto row_idx = thread_position_in_grid.z;
auto simd_lane = tid & 31u;
auto simd_id = tid >> 5;
threadgroup float simd_partial[8];

float sum = 0.0f;
for (int k = static_cast<int>(tid); k < InDim; k += 256) {
    int block_in_row = k >> 8;
    int within_block = k & 255;
    int ib32 = within_block >> 5;
    int within_32 = within_block & 31;

    auto block = w + (out_col * BlocksPerRow + block_in_row) * 136;
    const half d_half = *reinterpret_cast<const device half *>(block);
    const ushort scales_h =
        static_cast<ushort>(block[2]) |
        (static_cast<ushort>(block[3]) << 8);
    const uchar scale_l_byte = block[4 + (ib32 >> 1)];
    int ls =
        ((scale_l_byte >> (4 * (ib32 & 1))) & 0x0f) |
        (((scales_h >> (2 * ib32)) & 0x03) << 4);

    const uchar packed = block[8 + ib32 * 16 + (within_32 & 15)];
    const uchar q = within_32 < 16 ? (packed & 0x0f) : (packed >> 4);
    const float weight_value = static_cast<float>(d_half) * static_cast<float>(ls - 32) * kIq4NlValues[q];
    sum += x[row_idx * InDim + k] * weight_value;
}

float simd_total = simd_sum(sum);
if (simd_lane == 0) simd_partial[simd_id] = simd_total;
threadgroup_barrier(mem_flags::mem_threadgroup);
float lane_v = simd_lane < 8u ? simd_partial[simd_lane] : 0.0f;
float final_sum = simd_sum(lane_v);

if (tid == 0) {
    y[row_idx * OutDim + out_col] = final_sum;
}
";

        // Phase 8: simdgroup-fast reduction (same pattern as Q4KMatmul).
        private const string Iq4XsMatmul4Source = @"
auto tid = thread_position_in_threadgroup.x;
auto col_group = thread_position_in_grid.y;
int base_col = static_cast<int>(col_group) * 4;
auto simd_lane = tid & 31u;
auto simd_id = tid >> 5;
threadgroup float simd_partial[4][8];

float sum0 = 0.0f;
float sum1 = 0.0f;
float sum2 = 0.0f;
float sum3 = 0.0f;

for (int k = static_cast<int>(tid); k < InDim; k += 256) {
    const float xv = x[k];
    int block_in_row = k >> 8;
    int within_block = k & 255;

    if (base_col < OutDim) {
        auto block0 = w + (base_col * BlocksPerRow + block_in_row) * 136;
        sum0 += xv * tensorsharp_dequant_iq4xs(block0, within_block);
    }
    if (base_col + 1 < OutDim) {
        auto block1 = w + ((base_col + 1) * BlocksPerRow + block_in_row) * 136;
        sum1 += xv * tensorsharp_dequant_iq4xs(block1, within_block);
    }
    if (base_col + 2 < OutDim) {
        auto block2 = w + ((base_col + 2) * BlocksPerRow + block_in_row) * 136;
        sum2 += xv * tensorsharp_dequant_iq4xs(block2, within_block);
    }
    if (base_col + 3 < OutDim) {
        auto block3 = w + ((base_col + 3) * BlocksPerRow + block_in_row) * 136;
        sum3 += xv * tensorsharp_dequant_iq4xs(block3, within_block);
    }
}

float s0 = simd_sum(sum0);
float s1 = simd_sum(sum1);
float s2 = simd_sum(sum2);
float s3 = simd_sum(sum3);
if (simd_lane == 0) {
    simd_partial[0][simd_id] = s0;
    simd_partial[1][simd_id] = s1;
    simd_partial[2][simd_id] = s2;
    simd_partial[3][simd_id] = s3;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

float lane_v0 = simd_lane < 8u ? simd_partial[0][simd_lane] : 0.0f;
float lane_v1 = simd_lane < 8u ? simd_partial[1][simd_lane] : 0.0f;
float lane_v2 = simd_lane < 8u ? simd_partial[2][simd_lane] : 0.0f;
float lane_v3 = simd_lane < 8u ? simd_partial[3][simd_lane] : 0.0f;
float final0 = simd_sum(lane_v0);
float final1 = simd_sum(lane_v1);
float final2 = simd_sum(lane_v2);
float final3 = simd_sum(lane_v3);

if (tid == 0) {
    if (base_col < OutDim) {
        y[base_col] = final0;
    }
    if (base_col + 1 < OutDim) {
        y[base_col + 1] = final1;
    }
    if (base_col + 2 < OutDim) {
        y[base_col + 2] = final2;
    }
    if (base_col + 3 < OutDim) {
        y[base_col + 3] = final3;
    }
}
";

        // Iq4XsMatmul4 variant rewritten around simd_sum to eliminate the
        // 8-stage threadgroup reduction (and its 8 barriers) of the legacy
        // kernel. Threadgroup memory shrinks from 4 KiB → 0; the reduction
        // becomes a single 32-lane simd_sum executed in parallel inside
        // each simdgroup. Mirrors the GDN core kernel layout (see
        // `GatedDeltaStepSource` / `GatedDeltaSource` for prior art).
        //
        // Layout:
        //   - Threadgroup = 128 threads = 4 simdgroups.
        //   - Each simdgroup handles 1 output column.
        //   - 4 output columns per threadgroup, matching the legacy kernel
        //     so the dispatch grid in y stays at ceil(OutDim/4).
        //   - Each thread sums InDim/32 weight×activation products, then
        //     simd_sum collapses the 32 lanes into one scalar.
        //
        // Bandwidth note: the original kernel loads x[k] once per thread
        // and reuses it across all 4 partial sums. Here each simdgroup
        // re-reads x for its own column, so x is fetched 4× per
        // threadgroup instead of 1×. The L2 absorbs that — x is tiny
        // (5–10 KB per matmul on this model) and the simdgroups in a
        // threadgroup execute in lockstep, so the first simdgroup warms
        // L1 for the next three.
        private const string Iq4XsMatmul4SimdSource = @"
auto tid_in_sg = thread_index_in_simdgroup;
auto sg_id = simdgroup_index_in_threadgroup;
auto col_group = thread_position_in_grid.y;
int base_col = static_cast<int>(col_group) * 4;
int out_col = base_col + static_cast<int>(sg_id);

if (out_col >= OutDim) {
    return;
}

float sum = 0.0f;
for (int k = static_cast<int>(tid_in_sg); k < InDim; k += 32) {
    int block_in_row = k >> 8;
    int within_block = k & 255;
    auto block = w + (out_col * BlocksPerRow + block_in_row) * 136;
    sum += x[k] * tensorsharp_dequant_iq4xs(block, within_block);
}

sum = simd_sum(sum);

if (tid_in_sg == 0) {
    y[out_col] = sum;
}
";

        private const string Iq4XsMatmulRowsSource = @"
auto tid = thread_position_in_threadgroup.x;
auto out_col = thread_position_in_grid.y;
threadgroup float partial[Rows][256];
float sums[Rows];

for (int r = 0; r < Rows; r++) {
    sums[r] = 0.0f;
}

for (int k = static_cast<int>(tid); k < InDim; k += 256) {
    int block_in_row = k >> 8;
    int within_block = k & 255;
    int ib32 = within_block >> 5;
    int within_32 = within_block & 31;

    auto block = w + (out_col * BlocksPerRow + block_in_row) * 136;
    const half d_half = *reinterpret_cast<const device half *>(block);
    const ushort scales_h =
        static_cast<ushort>(block[2]) |
        (static_cast<ushort>(block[3]) << 8);
    const uchar scale_l_byte = block[4 + (ib32 >> 1)];
    int ls =
        ((scale_l_byte >> (4 * (ib32 & 1))) & 0x0f) |
        (((scales_h >> (2 * ib32)) & 0x03) << 4);

    const uchar packed = block[8 + ib32 * 16 + (within_32 & 15)];
    const uchar q = within_32 < 16 ? (packed & 0x0f) : (packed >> 4);
    const float weight_value = static_cast<float>(d_half) * static_cast<float>(ls - 32) * kIq4NlValues[q];
    for (int r = 0; r < Rows; r++) {
        sums[r] += x[r * InDim + k] * weight_value;
    }
}

for (int r = 0; r < Rows; r++) {
    partial[r][tid] = sums[r];
}
threadgroup_barrier(mem_flags::mem_threadgroup);

for (uint stride = 128; stride > 0; stride >>= 1) {
    if (tid < stride) {
        for (int r = 0; r < Rows; r++) {
            partial[r][tid] += partial[r][tid + stride];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

if (tid == 0) {
    for (int r = 0; r < Rows; r++) {
        y[r * OutDim + out_col] = partial[r][0];
    }
}
";

        private const string Iq4XsMatmulRows2Source = @"
auto tid = thread_position_in_threadgroup.x;
auto col_group = thread_position_in_grid.y;
int base_col = static_cast<int>(col_group) * 2;
threadgroup float partial0[Rows][256];
threadgroup float partial1[Rows][256];
float sums0[Rows];
float sums1[Rows];

for (int r = 0; r < Rows; r++) {
    sums0[r] = 0.0f;
    sums1[r] = 0.0f;
}

for (int k = static_cast<int>(tid); k < InDim; k += 256) {
    int block_in_row = k >> 8;
    int within_block = k & 255;

    float weight0 = 0.0f;
    float weight1 = 0.0f;
    if (base_col < OutDim) {
        auto block0 = w + (base_col * BlocksPerRow + block_in_row) * 136;
        weight0 = tensorsharp_dequant_iq4xs(block0, within_block);
    }
    if (base_col + 1 < OutDim) {
        auto block1 = w + ((base_col + 1) * BlocksPerRow + block_in_row) * 136;
        weight1 = tensorsharp_dequant_iq4xs(block1, within_block);
    }

    for (int r = 0; r < Rows; r++) {
        float xv = x[r * InDim + k];
        sums0[r] += xv * weight0;
        sums1[r] += xv * weight1;
    }
}

for (int r = 0; r < Rows; r++) {
    partial0[r][tid] = sums0[r];
    partial1[r][tid] = sums1[r];
}
threadgroup_barrier(mem_flags::mem_threadgroup);

for (uint stride = 128; stride > 0; stride >>= 1) {
    if (tid < stride) {
        for (int r = 0; r < Rows; r++) {
            partial0[r][tid] += partial0[r][tid + stride];
            partial1[r][tid] += partial1[r][tid + stride];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

if (tid == 0) {
    for (int r = 0; r < Rows; r++) {
        if (base_col < OutDim) {
            y[r * OutDim + base_col] = partial0[r][0];
        }
        if (base_col + 1 < OutDim) {
            y[r * OutDim + base_col + 1] = partial1[r][0];
        }
    }
}
";

        private const string Iq4XsGetRowsSource = @"
auto col = thread_position_in_grid.x;
auto out_row = thread_position_in_grid.y;
if (col >= InDim) {
    return;
}

int weight_row = indices[out_row];
int block_in_row = col >> 8;
int within_block = col & 255;
int ib32 = within_block >> 5;
int within_32 = within_block & 31;

auto block = w + (weight_row * BlocksPerRow + block_in_row) * 136;
const half d_half = *reinterpret_cast<const device half *>(block);
const ushort scales_h =
    static_cast<ushort>(block[2]) |
    (static_cast<ushort>(block[3]) << 8);
const uchar scale_l_byte = block[4 + (ib32 >> 1)];
int ls =
    ((scale_l_byte >> (4 * (ib32 & 1))) & 0x0f) |
    (((scales_h >> (2 * ib32)) & 0x03) << 4);

const uchar packed = block[8 + ib32 * 16 + (within_32 & 15)];
const uchar q = within_32 < 16 ? (packed & 0x0f) : (packed >> 4);
y[out_row * InDim + col] = static_cast<float>(d_half) * static_cast<float>(ls - 32) * kIq4NlValues[q];
";

        // ===================================================================
        // Decode / short-batch IQ matmul body (shared by IQ2_XXS, IQ2_S,
        // IQ3_S and IQ3_XXS - the four only differ in super-block stride and
        // dequant helper)
        // ===================================================================
        //
        // Grid (256, OutDim, InRows) with a 256-thread group: one threadgroup
        // per output column, the 256 threads splitting InDim and reducing
        // through simd_sum. Only rows < Iq2XxsMatmulSimdgroupMinRows reach
        // here; prefill goes to the simdgroup_matrix kernels.
        //
        // DOT8 (default): each thread takes 8 CONSECUTIVE weights per step, so
        // the super-block header loads and the 256/1024-entry codebook lookup
        // are paid once per 8 values instead of once per value - the same
        // amortization ggml-metal's iq kernels do over a 32-value sub-block.
        // The legacy per-element body ran the Muse-Glimmer-30B-UD-IQ2_XXS
        // decode FFN (IQ2_S gate/up + IQ3_XXS down, 74% of the token's weight
        // bytes) at ~34 GB/s against a ~205 GB/s memory roofline; the token was
        // 98% GPU wait with no host thread busy, so the cost was the dequant
        // instruction stream, not bandwidth or scheduling.
        //
        // TS_MLX_IQ_DECODE_DOT8=0 restores the per-element body for an A/B.
        private static readonly bool IQuantDecodeDot8Enabled =
            !string.Equals(Environment.GetEnvironmentVariable("TS_MLX_IQ_DECODE_DOT8"), "0", StringComparison.Ordinal);

        /// <summary>
        /// Emit the decode matmul body for one IQ quant type.
        /// <paramref name="blockBytes"/> is the GGUF super-block stride (256
        /// weights per block for every type here), and the DOT8 body relies on
        /// the dispatch guard that InDim is a multiple of 256 - so every
        /// 8-element group lands inside one super-block and starts on a
        /// multiple of 8, which is what lets one codebook entry cover it.
        /// </summary>
        private static string BuildIQuantMatmulSource(string dot8Helper, string scalarHelper, int blockBytes)
        {
            string reduction = IQuantDecodeDot8Enabled
                ? $@"
float sum = 0.0f;
for (int base_k = static_cast<int>(tid) * 8; base_k < InDim; base_k += 256 * 8) {{
    int block_in_row = base_k >> 8;
    int within_block = base_k & 255;
    auto block = w + (out_col * BlocksPerRow + block_in_row) * {blockBytes};
    int xbase = static_cast<int>(row_idx) * InDim + base_k;
    float xv[8];
    for (int j = 0; j < 8; ++j) {{
        xv[j] = x[xbase + j];
    }}
    sum += {dot8Helper}(block, within_block, xv);
}}"
                : $@"
float sum = 0.0f;
for (int k = static_cast<int>(tid); k < InDim; k += 256) {{
    int block_in_row = k >> 8;
    int within_block = k & 255;
    auto block = w + (out_col * BlocksPerRow + block_in_row) * {blockBytes};
    sum += x[static_cast<int>(row_idx) * InDim + k] * {scalarHelper}(block, within_block);
}}";

            return $@"
auto tid = thread_position_in_threadgroup.x;
auto out_col = thread_position_in_grid.y;
auto row_idx = thread_position_in_grid.z;
auto simd_lane = tid & 31u;
auto simd_id = tid >> 5;
threadgroup float simd_partial[8];
{reduction}

float simd_total = simd_sum(sum);
if (simd_lane == 0) simd_partial[simd_id] = simd_total;
threadgroup_barrier(mem_flags::mem_threadgroup);
float lane_v = simd_lane < 8u ? simd_partial[simd_lane] : 0.0f;
float final_sum = simd_sum(lane_v);

if (tid == 0) {{
    y[row_idx * OutDim + out_col] = final_sum;
}}
";
        }

        private static readonly string Iq2XxsMatmulSource =
            BuildIQuantMatmulSource("tensorsharp_dot8_iq2_xxs", "tensorsharp_dequant_iq2_xxs", 66);

        // IQ2_XXS matmul using Apple's simdgroup_matrix hardware
        // primitives. For batches >= 8 each SIMD group computes an
        // 8×8 output tile and (crucially) DEQUANTS its 8 weight
        // rows ONCE per K-chunk — reused across all 8 input rows in
        // the tile. The legacy per-output kernel dequants W rows
        // independently per input row, so total dequant work is
        // O(M·B·K). The simdgroup variant is O(M·K·B/8), an 8× cut
        // in the dequant pass for prefill (the matmul itself is
        // hardware-accelerated by simdgroup_multiply_accumulate
        // which does an 8×8 fp16-multiply with fp32-accumulate in
        // one cycle).
        //
        // Grid: (1, ceil(OutDim/8), ceil(InRows/8))
        // Threadgroup: 32 threads (1 SIMD group), one output tile each.
        // Inputs:  x [InRows, InDim] fp32, w stacked IQ2_XXS bytes
        // Output:  y [InRows, OutDim] fp32
        // The X load and W dequant both round up to the tile size
        // (8) with zero padding at the boundaries; matching bounds
        // checks at the store guarantee correctness for any InRows.
        private const string Iq2XxsMatmulSimdgroupSource = @"
auto tid = thread_position_in_threadgroup.x;
auto tile_m_idx = thread_position_in_grid.y;
auto tile_b_idx = thread_position_in_grid.z;
constexpr int TileSize = 8;
int tile_m = static_cast<int>(tile_m_idx) * TileSize;
int tile_b = static_cast<int>(tile_b_idx) * TileSize;

threadgroup half X_tile[TileSize * TileSize];
threadgroup half W_tile[TileSize * TileSize];
threadgroup float C_out[TileSize * TileSize];

simdgroup_float8x8 C = simdgroup_float8x8(0.0f);

for (int k0 = 0; k0 < InDim; k0 += TileSize) {
    // 32 threads cooperatively dequant 64 W elements (each thread does 2).
    // W_tile is stored row-major [m_in_tile][k_in_chunk]; we use
    // simdgroup_load(.., transpose=true) below so it appears as B[k][m]
    // for the matrix multiply (C = A · B).
    for (int i = 0; i < 2; ++i) {
        int idx = static_cast<int>(tid) * 2 + i;
        int m_in_tile = idx >> 3;
        int k_in_chunk = idx & 7;
        int m = tile_m + m_in_tile;
        int k = k0 + k_in_chunk;
        half val;
        if (m < OutDim && k < InDim) {
            int block_in_row = k >> 8;
            int within_block = k & 255;
            auto block = w + (m * BlocksPerRow + block_in_row) * 66;
            val = static_cast<half>(tensorsharp_dequant_iq2_xxs(block, within_block));
        } else {
            val = static_cast<half>(0.0f);
        }
        W_tile[m_in_tile * TileSize + k_in_chunk] = val;
    }

    // 32 threads cooperatively load 64 X elements (each thread does 2).
    for (int i = 0; i < 2; ++i) {
        int idx = static_cast<int>(tid) * 2 + i;
        int b_in_tile = idx >> 3;
        int k_in_chunk = idx & 7;
        int row = tile_b + b_in_tile;
        int col = k0 + k_in_chunk;
        half val = (row < InRows && col < InDim)
            ? static_cast<half>(x[row * InDim + col])
            : static_cast<half>(0.0f);
        X_tile[b_in_tile * TileSize + k_in_chunk] = val;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_half8x8 A, B;
    // A[b][k] = X_tile[b][k]  (no transpose)
    simdgroup_load(A, X_tile, TileSize);
    // B[k][m] = W_tile[m][k]  (transposed load)
    simdgroup_load(B, W_tile, TileSize, ulong2(0, 0), true);
    // simdgroup_multiply_accumulate(d, a, b, c): d = a*b + c.
    // We accumulate into C, so pass C as both destination and source.
    simdgroup_multiply_accumulate(C, A, B, C);
}

// Store the 8×8 accumulator. Fast path when the tile is fully within
// bounds; tail path otherwise (handles non-multiple-of-8 batch sizes
// and output dimensions).
if (tile_b + TileSize <= InRows && tile_m + TileSize <= OutDim) {
    simdgroup_store(C, y + tile_b * OutDim + tile_m, OutDim);
} else {
    simdgroup_store(C, C_out, TileSize);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = 0; i < 2; ++i) {
        int idx = static_cast<int>(tid) * 2 + i;
        int b_in_tile = idx >> 3;
        int m_in_tile = idx & 7;
        int row = tile_b + b_in_tile;
        int col = tile_m + m_in_tile;
        if (row < InRows && col < OutDim) {
            y[row * OutDim + col] = C_out[b_in_tile * TileSize + m_in_tile];
        }
    }
}
";

        private const string Iq2XxsGetRowsSource = @"
auto col = thread_position_in_grid.x;
auto out_row = thread_position_in_grid.y;
if (col >= InDim) {
    return;
}

int weight_row = indices[out_row];
int block_in_row = col >> 8;
int within_block = col & 255;
auto block = w + (weight_row * BlocksPerRow + block_in_row) * 66;
y[out_row * InDim + col] = tensorsharp_dequant_iq2_xxs(block, within_block);
";

        // Batched IQ2_XXS MoE matmul (shared input). Replaces K separate
        // per-expert matmul dispatches with one Metal command — for the
        // Qwen3.5 MoE decode path that's K=8 experts × 60 layers worth of
        // launch overhead saved per token.
        //
        // For each (k, out_col): y[k, out_col] = dot(x[0, :], W[expert_indices[k], out_col, :])
        // where W is the stacked uint8 IQ2_XXS bytes for all experts.
        // Input x is shared across experts (decode case: a single token
        // routed to multiple experts). For the down-projection variant
        // each row of x is per-expert; see Iq2XxsMoeMatmulBatchedRowedSource.
        private const string Iq2XxsMoeMatmulBatchedSource = @"
auto tid = thread_position_in_threadgroup.x;
auto out_col = thread_position_in_grid.y;
auto k_idx = thread_position_in_grid.z;
threadgroup float partial[256];

int expert_idx = expert_indices[k_idx];

float sum = 0.0f;
for (int k = static_cast<int>(tid); k < InDim; k += 256) {
    int block_in_row = k >> 8;
    int within_block = k & 255;
    auto block_offset = (uint)expert_idx * OutDim * BlocksPerRow + out_col * BlocksPerRow + block_in_row;
    auto block = w + block_offset * 66;
    sum += x[k] * tensorsharp_dequant_iq2_xxs(block, within_block);
}

partial[tid] = sum;
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint stride = 128; stride > 0; stride >>= 1) {
    if (tid < stride) {
        partial[tid] += partial[tid + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

if (tid == 0) {
    y[k_idx * OutDim + out_col] = partial[0];
}
";

        // Fused IQ2_XXS MoE gate + up + SiLUMul. Each (k, out_col) thread
        // group computes BOTH dot(x, W_gate[expert, out_col, :]) and
        // dot(x, W_up[expert, out_col, :]) over the same hidden dim,
        // then writes y[k, out_col] = silu(gate) * up in one Metal
        // dispatch. Saves: the up matmul kernel launch (~1 dispatch),
        // the SiLUMul kernel launch (~1 dispatch), and the per-(K,
        // intermediate) writes for gate and up plus reads for SiLUMul —
        // about 3× the [K, intermediate] FFN-intermediate buffer of
        // device-memory traffic per MoE layer.
        //
        // Shared input semantics (one row of x reused across all experts).
        // Same w-bytes layout as Iq2XxsMoeMatmulBatched (each per-expert
        // weight is [outDim × inDim/256] × 66 bytes); w_gate and w_up are
        // two distinct stacked weight buffers.
        private const string Iq2XxsMoeMatmulBatchedFusedGateUpSiluSource = @"
auto tid = thread_position_in_threadgroup.x;
auto out_col = thread_position_in_grid.y;
auto k_idx = thread_position_in_grid.z;
threadgroup float partial_gate[256];
threadgroup float partial_up[256];

int expert_idx = expert_indices[k_idx];

float sum_gate = 0.0f;
float sum_up = 0.0f;
for (int k = static_cast<int>(tid); k < InDim; k += 256) {
    int block_in_row = k >> 8;
    int within_block = k & 255;
    auto block_offset = (uint)expert_idx * OutDim * BlocksPerRow + out_col * BlocksPerRow + block_in_row;
    float xv = x[k];
    sum_gate += xv * tensorsharp_dequant_iq2_xxs(w_gate + block_offset * 66, within_block);
    sum_up   += xv * tensorsharp_dequant_iq2_xxs(w_up   + block_offset * 66, within_block);
}

partial_gate[tid] = sum_gate;
partial_up[tid]   = sum_up;
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint stride = 128; stride > 0; stride >>= 1) {
    if (tid < stride) {
        partial_gate[tid] += partial_gate[tid + stride];
        partial_up[tid]   += partial_up[tid + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

if (tid == 0) {
    float g = partial_gate[0];
    float u = partial_up[0];
    float silu_g = g / (1.0f + exp(-g));
    y[k_idx * OutDim + out_col] = silu_g * u;
}
";

        // Batched IQ2_XXS MoE matmul (per-row input). Each row k of the
        // input maps to expert expert_indices[k]. Used for the down
        // projection where the SwiGLU output is already [K, intermediate]
        // and each row needs its expert's down-weight.
        //
        // For each (k, out_col): y[k, out_col] = dot(x[k, :], W[expert_indices[k], out_col, :])
        private const string Iq2XxsMoeMatmulBatchedRowedSource = @"
auto tid = thread_position_in_threadgroup.x;
auto out_col = thread_position_in_grid.y;
auto k_idx = thread_position_in_grid.z;
threadgroup float partial[256];

int expert_idx = expert_indices[k_idx];

float sum = 0.0f;
for (int k = static_cast<int>(tid); k < InDim; k += 256) {
    int block_in_row = k >> 8;
    int within_block = k & 255;
    auto block_offset = (uint)expert_idx * OutDim * BlocksPerRow + out_col * BlocksPerRow + block_in_row;
    auto block = w + block_offset * 66;
    sum += x[k_idx * InDim + k] * tensorsharp_dequant_iq2_xxs(block, within_block);
}

partial[tid] = sum;
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint stride = 128; stride > 0; stride >>= 1) {
    if (tid < stride) {
        partial[tid] += partial[tid + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

if (tid == 0) {
    y[k_idx * OutDim + out_col] = partial[0];
}
";


        private static readonly string Iq2SMatmulSource =
            BuildIQuantMatmulSource("tensorsharp_dot8_iq2_s", "tensorsharp_dequant_iq2_s", 82);

        private const string Iq2SGetRowsSource = @"
auto col = thread_position_in_grid.x;
auto out_row = thread_position_in_grid.y;
if (col >= InDim) {
    return;
}

int weight_row = indices[out_row];
int block_in_row = col >> 8;
int within_block = col & 255;
auto block = w + (weight_row * BlocksPerRow + block_in_row) * 82;
y[out_row * InDim + col] = tensorsharp_dequant_iq2_s(block, within_block);
";

        // Phase 8: simdgroup-fast reduction (same pattern as Q4KMatmul).
        private static readonly string Iq3SMatmulSource =
            BuildIQuantMatmulSource("tensorsharp_dot8_iq3_s", "tensorsharp_dequant_iq3_s", 110);

        private const string Iq3SGetRowsSource = @"
auto col = thread_position_in_grid.x;
auto out_row = thread_position_in_grid.y;
if (col >= InDim) {
    return;
}

int weight_row = indices[out_row];
int block_in_row = col >> 8;
int within_block = col & 255;
auto block = w + (weight_row * BlocksPerRow + block_in_row) * 110;
y[out_row * InDim + col] = tensorsharp_dequant_iq3_s(block, within_block);
";

        // IQ3_XXS decode/short-batch matmul. Same simdgroup reduction shape as
        // the IQ3_S kernel above; only the 98-byte super-block stride and the
        // dequant helper differ.
        private static readonly string Iq3XxsMatmulSource =
            BuildIQuantMatmulSource("tensorsharp_dot8_iq3_xxs", "tensorsharp_dequant_iq3_xxs", 98);

        private const string Iq3XxsGetRowsSource = @"
auto col = thread_position_in_grid.x;
auto out_row = thread_position_in_grid.y;
if (col >= InDim) {
    return;
}

int weight_row = indices[out_row];
int block_in_row = col >> 8;
int within_block = col & 255;
auto block = w + (weight_row * BlocksPerRow + block_in_row) * 98;
y[out_row * InDim + col] = tensorsharp_dequant_iq3_xxs(block, within_block);
";

        private const string HeadDim256AttentionSource = @"
auto tid = thread_position_in_threadgroup.x;
auto head = thread_position_in_grid.y;
auto q_pos = thread_position_in_grid.z;
threadgroup float partial[256];

const int group_size = NumHeads / NumKVHeads;
const int kv_head = head / group_size;
const int causal_limit_raw = Causal == 0 ? (KvLen - 1) : (MaskStart + static_cast<int>(q_pos));
const int causal_limit = causal_limit_raw < KvLen ? causal_limit_raw : (KvLen - 1);

float max_score = -3.4028234663852886e+38f;
float normalizer = 0.0f;
float acc = 0.0f;

for (int t = 0; t < KvLen; t++) {
    float dot_part = 0.0f;
    if (t <= causal_limit) {
        dot_part = q[(head * QLen + q_pos) * 256 + tid] *
            k[(kv_head * KvLen + t) * 256 + tid];
    }

    partial[tid] = dot_part;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (t <= causal_limit) {
        const float score = partial[0] * scale_value;
        const float new_max = score > max_score ? score : max_score;
        const float old_scale = metal::fast::exp(max_score - new_max);
        const float score_scale = metal::fast::exp(score - new_max);
        acc = acc * old_scale + score_scale * v[(kv_head * KvLen + t) * 256 + tid];
        normalizer = normalizer * old_scale + score_scale;
        max_score = new_max;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
}

y[(q_pos * NumHeads + head) * 256 + tid] = acc / normalizer;
";

        // Decode attention with optional attention sinks and sliding-window
        // mask. Used by GPT-OSS-style models where each query head has a
        // learned "sink" logit that participates in the softmax denominator
        // but contributes V=0. Same online-softmax structure as
        // CircularDecodeAttention but:
        //   - max_score initialised to sinks[head] (the sink's logit)
        //   - normalizer initialised to 1.0 (= exp(sink - sink))
        //   - sink contributes only to the denominator (V=0 implicit)
        // Sliding-window: positions before MaskStart are skipped (clamp the
        // effective attention window to the last SlidingWindow tokens). When
        // MaskStart=0 and SlidingWindow=0, this is full causal decode.
        //
        // Inputs:
        //   q [1, NumHeads*HeadDim] f32
        //   k_cache [NumKVHeads, CacheLen, HeadDim] f16 or f32
        //   v_cache [NumKVHeads, CacheLen, HeadDim] f16 or f32
        //   sinks  [NumHeads] f32
        //   scale_value  scalar f32
        // Output:
        //   y [1, NumHeads*HeadDim] f32
        private const string DecodeAttentionWithSinksSource = @"
auto tid = thread_position_in_threadgroup.x;
auto head = thread_position_in_grid.y;
threadgroup float partial[256];

const int group_size = NumHeads / NumKVHeads;
const int kv_head = head / group_size;
const int attend_start = MaskStart;
const int attend_end   = AttendLen;

float max_score = sinks[head];
float normalizer = 1.0f;
float acc = 0.0f;

for (int t = attend_start; t < attend_end; t++) {
    float dot_part = 0.0f;
    if (tid < HeadDim) {
        float qv = static_cast<float>(q[head * HeadDim + tid]);
        float kv = static_cast<float>(k_cache[(kv_head * CacheLen + t) * HeadDim + tid]);
        dot_part = qv * kv;
    }

    partial[tid] = dot_part;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float score = partial[0] * scale_value;
    const float new_max = score > max_score ? score : max_score;
    const float old_scale = metal::fast::exp(max_score - new_max);
    const float score_scale = metal::fast::exp(score - new_max);
    if (tid < HeadDim) {
        acc = acc * old_scale + score_scale *
            static_cast<float>(v_cache[(kv_head * CacheLen + t) * HeadDim + tid]);
    } else {
        acc = acc * old_scale;
    }
    normalizer = normalizer * old_scale + score_scale;
    max_score = new_max;

    threadgroup_barrier(mem_flags::mem_threadgroup);
}

if (tid < HeadDim) {
    y[head * HeadDim + tid] = acc / normalizer;
}
";

        // Phase 6g: decode attention for Gemma 4 global layers
        // (head_dim = 512, non-circular). Mirrors the simdgroup-fast pattern
        // used by CircularDecodeAttention but without the SWA wrap logic
        // and with 16 simdgroups (HeadDim / SIMD_WIDTH = 512/32). Replaces
        // the route through MLX's mlx_fast_scaled_dot_product_attention
        // for Gemma 4's 7 global layers per token. Avoids the
        // `kCache.Narrow + TryRunHeadFirstAttention` chain entirely.
        private const string DecodeAttentionHeadDim512Source = @"
auto tid = thread_position_in_threadgroup.x;
auto head = thread_position_in_grid.y;
auto simd_lane = tid & 31u;
auto simd_id = tid >> 5;
threadgroup float simd_partial[16];

const int group_size = NumHeads / NumKVHeads;
const int kv_head = head / group_size;

float max_score = -3.4028234663852886e+38f;
float normalizer = 0.0f;
float acc = 0.0f;

float qv = q[head * HeadDim + tid];

for (int t = 0; t < AttendLen; t++) {
    int k_off = (kv_head * CacheLen + t) * HeadDim + tid;
    float kv = static_cast<float>(k_cache[k_off]);
    float dot_part = qv * kv;

    float simd_total = simd_sum(dot_part);
    if (simd_lane == 0) simd_partial[simd_id] = simd_total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float lane_v = simd_lane < 16u ? simd_partial[simd_lane] : 0.0f;
    float dot = simd_sum(lane_v);

    const float score = dot * scale_value;
    const float new_max = score > max_score ? score : max_score;
    const float old_scale = metal::fast::exp(max_score - new_max);
    const float score_scale = metal::fast::exp(score - new_max);
    int v_off = (kv_head * CacheLen + t) * HeadDim + tid;
    acc = acc * old_scale + score_scale * static_cast<float>(v_cache[v_off]);
    normalizer = normalizer * old_scale + score_scale;
    max_score = new_max;

    threadgroup_barrier(mem_flags::mem_threadgroup);
}

y[head * HeadDim + tid] = acc / normalizer;
";

        private const string CircularDecodeAttentionSource = @"
auto tid = thread_position_in_threadgroup.x;
auto head = thread_position_in_grid.y;
auto simd_lane = tid & 31u;
auto simd_id = tid >> 5;
// Phase 6d: replaced the 8-barrier threadgroup-tree reduction (which
// fired AttendLen times per kernel = up to 512 × 8 = 4096 barriers /
// dispatch) with a simdgroup-fast reduction (simd_sum + a single
// 8-element broadcast through shared memory). Drops per-position
// barrier count from 8 to 1 and recovers most of the per-attention-
// step GPU time on Apple Silicon.
threadgroup float simd_partial[8];

const int group_size = NumHeads / NumKVHeads;
const int kv_head = head / group_size;

float max_score = -3.4028234663852886e+38f;
float normalizer = 0.0f;
float acc = 0.0f;

for (int t = 0; t < AttendLen; t++) {
    int cache_pos = FirstSlot + t;
    if (cache_pos >= CacheLen) {
        cache_pos -= CacheLen;
    }

    float dot_part = 0.0f;
    if (tid < HeadDim) {
        float qv = static_cast<float>(q[head * HeadDim + tid]);
        float kv = static_cast<float>(k_cache[(kv_head * CacheLen + cache_pos) * HeadDim + tid]);
        dot_part = qv * kv;
    }

    // Two-stage reduction: simd_sum within each 32-thread simdgroup,
    // then one barrier + simd_sum over the 8 per-simdgroup partials.
    float simd_total = simd_sum(dot_part);
    if (simd_lane == 0) simd_partial[simd_id] = simd_total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float lane_v = simd_lane < 8u ? simd_partial[simd_lane] : 0.0f;
    float dot = simd_sum(lane_v);

    const float score = dot * scale_value;
    const float new_max = score > max_score ? score : max_score;
    const float old_scale = metal::fast::exp(max_score - new_max);
    const float score_scale = metal::fast::exp(score - new_max);
    if (tid < HeadDim) {
        acc = acc * old_scale + score_scale *
            static_cast<float>(v_cache[(kv_head * CacheLen + cache_pos) * HeadDim + tid]);
    } else {
        acc = acc * old_scale;
    }
    normalizer = normalizer * old_scale + score_scale;
    max_score = new_max;

    threadgroup_barrier(mem_flags::mem_threadgroup);
}

if (tid < HeadDim) {
    y[head * HeadDim + tid] = acc / normalizer;
}
";

        // Fused Q8 matmul + per-element (gelu_tanh(matmul) * gate). Used by
        // Gemma 4's PLE inp_gate stage which currently runs as two MLX
        // dispatches: a Q8 matmul producing [1, pleDim], followed by
        // mlx_binary(MUL) of gelu(matmul) and a perLayerInput slice. With
        // this kernel both happen in one dispatch — each threadgroup
        // produces one output column, computes gelu(sum), and multiplies
        // by gate[col] before writing. Saves 1 op per layer × 42 layers.
        //
        //   y[col] = gelu_tanh(sum_k(x[k] * dequant(w[col, k]))) * gate[col]
        private const string Q8MatmulGeluMulSource = @"
auto tid = thread_position_in_threadgroup.x;
auto out_col = thread_position_in_grid.y;
auto simd_lane = tid & 31u;
auto simd_id = tid >> 5;
threadgroup float simd_partial[8];

float sum = 0.0f;
for (int k = static_cast<int>(tid) * 4; k < InDim; k += 256 * 4) {
    int block_in_row = k >> 5;
    float s = static_cast<float>(scales[out_col * BlocksPerRow + block_in_row]);
    float b = static_cast<float>(biases[out_col * BlocksPerRow + block_in_row]);
    uint packed = w[out_col * (InDim / 4) + (k >> 2)];

    float dq0 = static_cast<float>((packed >> 0) & 0xFFu) * s + b;
    float dq1 = static_cast<float>((packed >> 8) & 0xFFu) * s + b;
    float dq2 = static_cast<float>((packed >> 16) & 0xFFu) * s + b;
    float dq3 = static_cast<float>((packed >> 24) & 0xFFu) * s + b;

    sum += x[k + 0] * dq0 + x[k + 1] * dq1 + x[k + 2] * dq2 + x[k + 3] * dq3;
}

float simd_total = simd_sum(sum);
if (simd_lane == 0) simd_partial[simd_id] = simd_total;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (simd_id == 0) {
    float v = simd_lane < 8u ? simd_partial[simd_lane] : 0.0f;
    float matmul_v = simd_sum(v);
    if (simd_lane == 0) {
        // GELU (tanh approximation) of matmul_v, then multiply by gate.
        float g = matmul_v;
        float g3 = g * g * g;
        float inner = 0.7978845608f * (g + 0.044715f * g3);
        // Clamp before tanh. metal::fast::tanh is an exp-ratio approximation, so
        // once exp(2*inner) overflows (|inner| > ~44, i.e. |g| > ~11) it computes
        // Inf/Inf and returns NaN instead of saturating to +/-1. That is a real
        // failure, not a rounding artifact: on gemma-4-E4B this produced ONE NaN
        // in the 256-element per-layer-embedding gate at layer 4, the following
        // proj matmul smeared it across all 2560 residual channels, and the model
        // emitted <pad> for every token after the first. tanh is +/-1 to float
        // precision well before |x| = 15, so clamping there is exact.
        inner = metal::clamp(inner, -15.0f, 15.0f);
        float gelu = 0.5f * g * (1.0f + metal::fast::tanh(inner));
        y[out_col] = gelu * gate[out_col];
    }
}
";

        // Plain Q8_0 matmul (no fusion) using simdgroup-fast reductions.
        // Same as the matmul portion of <see cref="Q8AddmmAddSource"/>, but
        // without the residual addend. Drop-in replacement for MLX's
        // built-in <c>mlx_quantized_matmul</c> on Q8_0 weights — typically
        // matches or slightly beats the built-in on decode (rows == 1).
        private const string Q8MatmulSource = @"
auto tid = thread_position_in_threadgroup.x;
auto out_col = thread_position_in_grid.y;
auto simd_lane = tid & 31u;
auto simd_id = tid >> 5;
threadgroup float simd_partial[8];

float sum = 0.0f;
for (int k = static_cast<int>(tid) * 4; k < InDim; k += 256 * 4) {
    int block_in_row = k >> 5;
    float s = static_cast<float>(scales[out_col * BlocksPerRow + block_in_row]);
    float b = static_cast<float>(biases[out_col * BlocksPerRow + block_in_row]);
    uint packed = w[out_col * (InDim / 4) + (k >> 2)];

    float dq0 = static_cast<float>((packed >> 0) & 0xFFu) * s + b;
    float dq1 = static_cast<float>((packed >> 8) & 0xFFu) * s + b;
    float dq2 = static_cast<float>((packed >> 16) & 0xFFu) * s + b;
    float dq3 = static_cast<float>((packed >> 24) & 0xFFu) * s + b;

    sum += x[k + 0] * dq0 + x[k + 1] * dq1 + x[k + 2] * dq2 + x[k + 3] * dq3;
}

float simd_total = simd_sum(sum);
if (simd_lane == 0) simd_partial[simd_id] = simd_total;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (simd_id == 0) {
    float v = simd_lane < 8u ? simd_partial[simd_lane] : 0.0f;
    float final_sum = simd_sum(v);
    if (simd_lane == 0) {
        y[out_col] = final_sum;
    }
}
";

        // Fused RMSNorm(input) + Q8_0 matmul. Each threadgroup independently
        // computes the input row's RMS — there's no good way to share that
        // across threadgroups within one Metal dispatch — but with simdgroup
        // intrinsics the norm phase costs ~50 cycles per threadgroup, which
        // overlaps with the bandwidth-bound matmul kernel reads on Apple
        // Silicon GPUs. End-to-end this is faster than the two-Metal-kernel
        // alternative (mlx_fast_rms_norm + mlx_quantized_matmul) because we
        // save one full kernel launch.
        //
        //   normed = rmsnorm(x, norm_w, eps)
        //   y[col] = sum_k(normed[k] * dequant(w[col, k]))
        //
        // Grid: (256, OutDim, 1). Threadgroup: 256 (= 8 simdgroups).
        private const string Q8RmsNormMatmulSource = @"
auto tid = thread_position_in_threadgroup.x;
auto out_col = thread_position_in_grid.y;
auto simd_lane = tid & 31u;
auto simd_id = tid >> 5;
threadgroup float simd_partial[8];
threadgroup float rms_scale;

// Phase 1: compute sum(x^2) → rms scalar broadcast to all threads.
float sq = 0.0f;
for (int k = static_cast<int>(tid); k < InDim; k += 256) {
    float v = x[k];
    sq += v * v;
}
float simd_sq = simd_sum(sq);
if (simd_lane == 0) simd_partial[simd_id] = simd_sq;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (simd_id == 0) {
    float v = simd_lane < 8u ? simd_partial[simd_lane] : 0.0f;
    float total_sq = simd_sum(v);
    if (simd_lane == 0) {
        rms_scale = metal::fast::rsqrt(total_sq / float(InDim) + eps_value);
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
float rms = rms_scale;

// Phase 2: matmul with normalized + weighted input.
// Each thread strides through input by 4 elements (one packed uint32 = 4
// quants). The norm scale and per-element norm weight are applied as we
// fetch, so the input row isn't materialized in shared memory; only the
// scalar `rms` is shared.
float sum = 0.0f;
for (int k = static_cast<int>(tid) * 4; k < InDim; k += 256 * 4) {
    int block_in_row = k >> 5;
    float s = static_cast<float>(scales[out_col * BlocksPerRow + block_in_row]);
    float b = static_cast<float>(biases[out_col * BlocksPerRow + block_in_row]);
    uint packed = w[out_col * (InDim / 4) + (k >> 2)];

    float dq0 = static_cast<float>((packed >> 0) & 0xFFu) * s + b;
    float dq1 = static_cast<float>((packed >> 8) & 0xFFu) * s + b;
    float dq2 = static_cast<float>((packed >> 16) & 0xFFu) * s + b;
    float dq3 = static_cast<float>((packed >> 24) & 0xFFu) * s + b;

    float nx0 = x[k + 0] * rms * norm_w[k + 0];
    float nx1 = x[k + 1] * rms * norm_w[k + 1];
    float nx2 = x[k + 2] * rms * norm_w[k + 2];
    float nx3 = x[k + 3] * rms * norm_w[k + 3];

    sum += nx0 * dq0 + nx1 * dq1 + nx2 * dq2 + nx3 * dq3;
}

float simd_total = simd_sum(sum);
if (simd_lane == 0) simd_partial[simd_id] = simd_total;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (simd_id == 0) {
    float v = simd_lane < 8u ? simd_partial[simd_lane] : 0.0f;
    float final_sum = simd_sum(v);
    if (simd_lane == 0) {
        y[out_col] = final_sum;
    }
}
";

        // ===== Phase 6 custom Metal kernels =====
        //
        // These kernels collapse the existing two-step
        // {mlx_quantized_matmul + mlx_binary_add} and
        // {mlx_fast_rms_norm + mlx_quantized_matmul} Metal dispatch pairs
        // into a single Metal dispatch each. They read MLX's affine-Q8
        // repacked weight layout (Weight: [outDim, inDim/4] uint32 with each
        // uint32 holding 4 uint8 quants XOR'd by 0x80; Scales / Biases:
        // [outDim, blocksPerRow] f16 with bias = -128 * scale) so the same
        // preloaded DeviceWeight is reused — no extra weight upload.
        //
        // Decode = batch 1 (Rows == 1). The kernels assume that. Prefill /
        // longer rows fall back to MLX's tuned generic path.

        // Fused Q8_0 matmul + residual add. Uses simdgroup-level reductions
        // instead of a threadgroup-barrier tree reduction — Metal's
        // `simd_sum` is a hardware-supported intrinsic (~1 cycle per
        // simdgroup of 32 threads), much faster than the
        // `threadgroup_barrier` ladder. Mirrors the structure ggml_metal
        // uses for its `kernel_mul_mv_q8_0_f32` to close the per-matmul GPU
        // time gap that custom-kernel attempts using naive reductions
        // couldn't close.
        //
        //   y[col] = residual[col] + sum_k(x[k] * dequant(weight[col, k]))
        //
        // where dequant(q) = q * scale + bias (MLX's affine form, bias is
        // pre-baked to -128 * scale so adding it cancels the XOR-0x80
        // offset in `weight`).
        //
        // Grid: (256, OutDim, 1) — one threadgroup per output column.
        // Threadgroup: 256 threads = 8 simdgroups of 32 threads.
        private const string Q8AddmmAddSource = @"
auto tid = thread_position_in_threadgroup.x;
auto out_col = thread_position_in_grid.y;
auto simd_lane = tid & 31u;
auto simd_id = tid >> 5;
threadgroup float simd_partial[8];

float sum = 0.0f;
// Each thread strides by 4 elements (one packed uint32 holds 4 quants).
// Within a 32-element block scale/bias are constant, hoisted out of the
// unpack to save 3 lookups per packed word.
for (int k = static_cast<int>(tid) * 4; k < InDim; k += 256 * 4) {
    int block_in_row = k >> 5;
    float s = static_cast<float>(scales[out_col * BlocksPerRow + block_in_row]);
    float b = static_cast<float>(biases[out_col * BlocksPerRow + block_in_row]);
    uint packed = w[out_col * (InDim / 4) + (k >> 2)];

    float dq0 = static_cast<float>((packed >> 0) & 0xFFu) * s + b;
    float dq1 = static_cast<float>((packed >> 8) & 0xFFu) * s + b;
    float dq2 = static_cast<float>((packed >> 16) & 0xFFu) * s + b;
    float dq3 = static_cast<float>((packed >> 24) & 0xFFu) * s + b;

    sum += x[k + 0] * dq0;
    sum += x[k + 1] * dq1;
    sum += x[k + 2] * dq2;
    sum += x[k + 3] * dq3;
}

// Reduce within simdgroup (hardware-fast).
float simd_total = simd_sum(sum);

// Lane 0 of each simdgroup writes the simdgroup total.
if (simd_lane == 0) {
    simd_partial[simd_id] = simd_total;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

// First simdgroup reduces across the 8 simdgroup partials.
if (simd_id == 0) {
    float v = simd_lane < 8u ? simd_partial[simd_lane] : 0.0f;
    float final_sum = simd_sum(v);
    if (simd_lane == 0) {
        y[out_col] = residual[out_col] + final_sum;
    }
}
";

        // Fused Gemma 4 decode-step QKV preprocessing kernel. Replaces the
        // following 5 separate MLX dispatches per attention layer:
        //   1. RMSNorm on Q  (per-head, weighted by attn_q_norm.weight)
        //   2. RMSNorm on K  (per-head, weighted by attn_k_norm.weight)
        //   3. unweighted RMSNorm on V
        //   4. NeoX RoPE on Q
        //   5. NeoX RoPE on K
        // with a single Metal kernel that does all of them in one dispatch.
        //
        // Input layout (`qkv`, flat, after the fused norm+QKV matmul):
        //   qkv[0 ..              NumHeads*HeadDim)               – Q
        //   qkv[NumHeads*HeadDim ..(NumHeads+NumKVHeads)*HeadDim) – K
        //   qkv[(NumHeads+NumKVHeads)*HeadDim .. end)            – V
        //
        // Outputs (3):
        //   q_out  [1, NumHeads * HeadDim]            – Q post-norm + RoPE, flat
        //   k_out  [NumKVHeads, 1, HeadDim]           – K post-norm + RoPE, head-first
        //   v_out  [NumKVHeads, 1, HeadDim]           – V post-norm (unweighted), head-first
        //
        // Grid: (HeadDim, NumHeads + 2*NumKVHeads, 1) – one threadgroup per output row.
        // Threadgroup: (HeadDim, 1, 1) – collaborative RMS reduction per row.
        //
        // Restrictions:
        //   * HeadDim ≤ 512 and a power of two (256 SWA / 512 global covered).
        //   * Per-token decode only (seqLen == 1).
        //   * cos_table / sin_table indexed by `col` (already pre-built for the
        //     scalar position; no per-position stride).
        private const string Gemma4QkvPreprocessDecodeSource = @"
threadgroup float row_buf[512];
threadgroup float simd_partial[16];

auto tid = thread_position_in_threadgroup.x;
auto row = thread_position_in_grid.y;
auto simd_lane = tid & 31u;
auto simd_id = tid >> 5;

int kind;
int head;
if (row < uint(NumHeads)) {
    kind = 0;
    head = static_cast<int>(row);
} else if (row < uint(NumHeads + NumKVHeads)) {
    kind = 1;
    head = static_cast<int>(row) - NumHeads;
} else {
    kind = 2;
    head = static_cast<int>(row) - NumHeads - NumKVHeads;
}

int input_offset;
if (kind == 0) {
    input_offset = head * HeadDim + static_cast<int>(tid);
} else if (kind == 1) {
    input_offset = NumHeads * HeadDim + head * HeadDim + static_cast<int>(tid);
} else {
    input_offset = (NumHeads + NumKVHeads) * HeadDim + head * HeadDim + static_cast<int>(tid);
}

float x = qkv[input_offset];

// Phase 6f: simdgroup-fast reduction. The tree-reduction (HeadDim/2 → 1
// barriers per iteration, ~9 barriers for HeadDim=512) is now one barrier
// + a cross-simdgroup broadcast through `simd_partial`.
float simd_sq = simd_sum(x * x);
if (simd_lane == 0) simd_partial[simd_id] = simd_sq;
threadgroup_barrier(mem_flags::mem_threadgroup);
uint n_simd = uint(HeadDim) >> 5;  // simdgroup count = HeadDim / 32
float lane_v = simd_lane < n_simd ? simd_partial[simd_lane] : 0.0f;
float total_sq = simd_sum(lane_v);

float rms = metal::fast::rsqrt(total_sq / float(HeadDim) + eps_value);

float w;
if (kind == 0) {
    w = q_norm_w[tid];
} else if (kind == 1) {
    w = k_norm_w[tid];
} else {
    w = 1.0f;
}
float normed = x * rms * w;
row_buf[tid] = normed;
threadgroup_barrier(mem_flags::mem_threadgroup);

float result = normed;
if (kind != 2) {
    if (tid < uint(RotHalf)) {
        float x0 = row_buf[tid];
        float x1 = row_buf[tid + uint(RotHalf)];
        float c = cos_table[tid];
        float s = sin_table[tid];
        result = x0 * c - x1 * s;
    } else if (tid < uint(RotHalf * 2)) {
        int j = static_cast<int>(tid) - RotHalf;
        float x0 = row_buf[j];
        float x1 = row_buf[tid];
        float c = cos_table[j];
        float s = sin_table[j];
        result = x0 * s + x1 * c;
    }
}

if (kind == 0) {
    q_out[head * HeadDim + static_cast<int>(tid)] = result;
} else if (kind == 1) {
    k_out[head * HeadDim + static_cast<int>(tid)] = result;
} else {
    v_out[head * HeadDim + static_cast<int>(tid)] = result;
}
";

        static MlxNative()
        {
            InstallResolver();
        }

        public static bool IsAvailable()
        {
            if (!OperatingSystem.IsMacOS())
                return false;

            try
            {
                InstallResolver();
                return MlxWorker.Shared.Invoke(() =>
                {
                    EnsureErrorHandlerInstalled();
                    if (mlx_metal_is_available(out bool metalAvailable) != 0 || !metalAvailable)
                        return false;

                    MlxDevice device = mlx_device_new_type(MlxGpu, 0);
                    try
                    {
                        return mlx_device_is_available(out bool deviceAvailable, device) == 0 && deviceAvailable;
                    }
                    finally
                    {
                        _ = mlx_device_free(device);
                    }
                });
            }
            catch (DllNotFoundException)
            {
                return false;
            }
            catch (EntryPointNotFoundException)
            {
                return false;
            }
            catch (BadImageFormatException)
            {
                return false;
            }
        }

        public static void EnsureGpuDevice(int deviceId)
        {
            if (deviceId < 0)
                throw new ArgumentOutOfRangeException(nameof(deviceId));

            MlxWorker.Shared.Invoke(() =>
            {
                lock (initSync)
                {
                    if (initializedDevice == deviceId)
                        return;

                    EnsureErrorHandlerInstalled();
                    ThrowIfUnavailable();
                    MlxDevice device = mlx_device_new_type(MlxGpu, deviceId);
                    try
                    {
                        Check(mlx_device_is_available(out bool available, device), "checking MLX GPU availability");
                        if (!available)
                            throw new PlatformNotSupportedException($"MLX GPU device {deviceId} is not available.");

                        Check(mlx_set_default_device(device), "setting default MLX GPU device");
                        ConfigureCacheLimit();
                        ConfigureWiredLimit();
                        ConfigureMemoryLimit();
                        initializedDevice = deviceId;
                        cachedDefaultStreamDevice = -1;
                        _ = mlx_reset_peak_memory();
                    }
                    finally
                    {
                        _ = mlx_device_free(device);
                    }
                }
            });
        }

        public static MlxMemorySnapshot GetMemorySnapshot()
        {
            return MlxWorker.Shared.Invoke(() =>
            {
                ulong active = QueryMemory(mlx_get_active_memory);
                ulong cache = QueryMemory(mlx_get_cache_memory);
                ulong peak = QueryMemory(mlx_get_peak_memory);
                return new MlxMemorySnapshot(active, cache, peak);
            });
        }

        public static void ClearCache()
        {
            try
            {
                MlxWorker.Shared.Invoke(() => _ = mlx_clear_cache());
            }
            catch (DllNotFoundException)
            {
            }
            catch (EntryPointNotFoundException)
            {
            }
        }

        internal static MlxArray NewArrayFromHost(IntPtr data, int[] shape, DType dtype)
        {
            return NewArrayFromHost(data, shape, ToMlxDtype(dtype));
        }

        internal static MlxArray NewArrayFromHostUInt32(IntPtr data, int[] shape)
        {
            return NewArrayFromHost(data, shape, 3);
        }

        private static MlxArray NewArrayFromHost(IntPtr data, int[] shape, int mlxDtype)
        {
            if (data == IntPtr.Zero)
                throw new ArgumentNullException(nameof(data));
            if (shape == null || shape.Length == 0)
                throw new ArgumentException("MLX array shape must be non-empty.", nameof(shape));

            return MlxWorker.Shared.Invoke(() => mlx_array_new_data(data, shape, shape.Length, mlxDtype));
        }

        // Empty deleter for zero-copy MLX arrays whose buffer lifetime is
        // owned by the caller (typically a long-lived GGUF mmap region or
        // a pinned native buffer). Must remain a static field so the GC
        // never collects the delegate (Metal holds the function pointer
        // until the MLX array is freed).
        private static readonly NoCopyDeleter NoOpDeleter = static (IntPtr _) => { };

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate void NoCopyDeleter(IntPtr ptr);

        /// <summary>
        /// Wraps an existing host buffer as an MLX array WITHOUT copying.
        /// Routes through <c>mlx_array_new_data_managed</c>, whose C++
        /// constructor first tries <c>allocator::make_buffer(data, nbytes)</c>;
        /// on Apple Silicon Metal in shared-memory mode that succeeds, so the
        /// returned MLX array references the host pointer directly. Falls
        /// back to a copy only if Metal rejects the wrap (e.g. unaligned
        /// pointer on discrete GPUs).
        ///
        /// The buffer pointed to by <paramref name="data"/> must remain valid
        /// (and unmodified) for as long as any MLX array derived from it is
        /// alive. Callers typically pin the GGUF mmap until model shutdown.
        /// </summary>
        internal static MlxArray NewArrayFromHostNoCopy(IntPtr data, int[] shape, DType dtype)
        {
            if (data == IntPtr.Zero)
                throw new ArgumentNullException(nameof(data));
            if (shape == null || shape.Length == 0)
                throw new ArgumentException("MLX array shape must be non-empty.", nameof(shape));

            int mlxDtype = ToMlxDtype(dtype);
            IntPtr dtorPtr = Marshal.GetFunctionPointerForDelegate<NoCopyDeleter>(NoOpDeleter);
            return MlxWorker.Shared.Invoke(() =>
                mlx_array_new_data_managed(data, shape, shape.Length, mlxDtype, dtorPtr));
        }

        internal static MlxArray NewScalar(float value)
        {
            return MlxWorker.Shared.Invoke(() => mlx_array_new_float32(value));
        }

        internal static MlxArray NewScalar(int value)
        {
            return MlxWorker.Shared.Invoke(() => mlx_array_new_int(value));
        }

        internal static MlxArray Arange(double start, double stop, double step, DType dtype)
        {
            return MlxWorker.Shared.Invoke(() =>
            {
                MlxArray result;
                Check(mlx_arange(out result, start, stop, step, ToMlxDtype(dtype), DefaultStream()), "creating MLX arange");
                return result;
            });
        }

        internal static MlxArray Full(int[] shape, float value, DType dtype)
        {
            if (shape == null || shape.Length == 0)
                throw new ArgumentException("MLX array shape must be non-empty.", nameof(shape));

            return MlxWorker.Shared.Invoke(() =>
            {
                MlxArray scalar = mlx_array_new_float32(value);
                try
                {
                    MlxArray result;
                    Check(mlx_full(out result, shape, (nuint)shape.Length, scalar, ToMlxDtype(dtype), DefaultStream()), "creating MLX full array");
                    return result;
                }
                finally
                {
                    _ = mlx_array_free(scalar);
                }
            });
        }

        internal static void FreeArray(MlxArray array)
        {
            if (!array.IsValid)
                return;

            // mlx_array_free is a side-effect-only ref-decrement; the worker is
            // FIFO so any later Invoke that touches the array still serializes
            // correctly. Dispatch avoids the signal/wait round trip — meaningful
            // since this is called hundreds of times per layer.
            //
            // Hot path: when we're already on the worker thread (typical for
            // ops invoked from inside MlxWorker.Shared.Invoke), call the
            // unmanaged free directly. This skips the closure allocation + the
            // _dispatchCount Interlocked.Increment + the IsOnWorkerThread
            // check that the Dispatch wrapper would do. With ~5000+ FreeArray
            // calls per decode token on Gemma 4, those ~50ns saved per call
            // add up to a meaningful slice of the per-token MLX overhead.
            if (MlxWorker.Shared.IsOnWorkerThread)
            {
                _ = mlx_array_free(array);
                return;
            }
            if (DisableFreeDispatch)
                MlxWorker.Shared.Invoke(() => _ = mlx_array_free(array));
            else
                MlxWorker.Shared.Dispatch(() => _ = mlx_array_free(array));
        }

        internal static MlxArray Astype(MlxArray array, DType dtype)
        {
            if (!array.IsValid)
                throw new ArgumentException("MLX astype input must be a valid array.", nameof(array));

            return MlxWorker.Shared.Invoke(() =>
            {
                MlxArray result;
                Check(mlx_astype(out result, array, ToMlxDtype(dtype), DefaultStream()), "casting MLX array");
                return result;
            });
        }

        // ----- mlx_compile pipeline ----------------------------------------
        //
        // The compile API takes a user-provided closure (a function from a
        // vector of arrays to a vector of arrays) and returns a compiled
        // closure that, when applied, executes a fused/optimized version of
        // the traced graph. MLX caches the compiled artifact per (input
        // shape, dtype) tuple unless `shapeless` is true, in which case it
        // traces once with symbolic shapes.
        //
        // We expose the lifecycle as three static helpers:
        //   - NewClosure(trace, shapeless): allocates the source closure
        //     and the compiled closure, and returns a handle bundling both
        //     plus the GC-rooted trace delegate.
        //   - ApplyClosure(handle, inputs): invokes the compiled closure
        //     and returns the output arrays.
        //   - FreeCompiledClosure(handle): releases everything.
        //
        // The trace callback runs synchronously from within mlx_compile /
        // mlx_closure_apply, which we always invoke from the worker thread.
        // The worker is re-entrant, so the C# `trace` delegate is free to
        // call any MlxNative.* helper.
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private unsafe delegate int MlxClosureCallback(MlxVectorArray* result, MlxVectorArray input, IntPtr payload);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate void MlxClosureDestructor(IntPtr payload);

        private static readonly unsafe MlxClosureCallback ClosureCallbackDelegate = ClosureCallback;
        private static readonly MlxClosureDestructor ClosureDestructorDelegate = ClosureDestructor;
        private static readonly unsafe IntPtr ClosureCallbackPtr = Marshal.GetFunctionPointerForDelegate(ClosureCallbackDelegate);
        private static readonly IntPtr ClosureDestructorPtr = Marshal.GetFunctionPointerForDelegate(ClosureDestructorDelegate);

        internal sealed class CompiledClosure
        {
            internal MlxClosure Compiled;
            internal GCHandle TraceHandle;
            internal bool Disposed;
        }

        // Trace function: takes a vector of input MlxArrays (handles owned by
        // MLX during the trace), returns a vector of output MlxArrays
        // (whose handles are handed to MLX). Inside the trace function,
        // intermediate arrays should be freed via FreeArray as normal.
        internal delegate MlxArray[] TraceFunc(MlxArray[] inputs);

        internal static CompiledClosure NewClosure(TraceFunc trace, bool shapeless)
        {
            if (trace == null) throw new ArgumentNullException(nameof(trace));

            return MlxWorker.Shared.Invoke(() =>
            {
                var holder = new CompiledClosure
                {
                    TraceHandle = GCHandle.Alloc(trace),
                };

                MlxClosure src = mlx_closure_new_func_payload(
                    ClosureCallbackPtr,
                    GCHandle.ToIntPtr(holder.TraceHandle),
                    ClosureDestructorPtr);

                try
                {
                    if (!src.IsValid)
                        throw new InvalidOperationException("mlx_closure_new_func_payload returned null.");

                    Check(mlx_compile(out holder.Compiled, src, shapeless), "compiling MLX closure");
                }
                finally
                {
                    // Per MLX C-API docs, the source closure is copied by
                    // mlx_compile; we always free our reference. The payload's
                    // GCHandle survives via mlx's internal closure copy until
                    // we call free on the compiled closure (which fires our
                    // destructor).
                    if (src.IsValid)
                        _ = mlx_closure_free(src);
                }

                return holder;
            });
        }

        internal static MlxArray ApplyClosure1(CompiledClosure holder, MlxArray input)
        {
            MlxArray[] outputs = ApplyClosure(holder, new[] { input });
            if (outputs.Length == 0) return default;
            // Caller takes ownership of outputs[0]; free any extras.
            for (int i = 1; i < outputs.Length; i++)
                FreeArray(outputs[i]);
            return outputs[0];
        }

        internal static MlxArray ApplyClosure2(CompiledClosure holder, MlxArray a, MlxArray b)
        {
            MlxArray[] outputs = ApplyClosure(holder, new[] { a, b });
            if (outputs.Length == 0) return default;
            for (int i = 1; i < outputs.Length; i++)
                FreeArray(outputs[i]);
            return outputs[0];
        }

        internal static MlxArray[] ApplyClosure(CompiledClosure holder, MlxArray[] inputs)
        {
            if (holder == null) throw new ArgumentNullException(nameof(holder));
            if (holder.Disposed) throw new ObjectDisposedException(nameof(CompiledClosure));

            return MlxWorker.Shared.Invoke(() =>
            {
                MlxVectorArray inVec = mlx_vector_array_new();
                MlxVectorArray outVec = mlx_vector_array_new();
                try
                {
                    for (int i = 0; i < inputs.Length; i++)
                    {
                        if (!inputs[i].IsValid)
                            throw new ArgumentException($"Closure input {i} is invalid.");
                        Check(mlx_vector_array_append_value(inVec, inputs[i]), "appending closure input");
                    }
                    Check(mlx_closure_apply(ref outVec, holder.Compiled, inVec), "applying compiled MLX closure");

                    int count = (int)mlx_vector_array_size(outVec);
                    var results = new MlxArray[count];
                    for (int i = 0; i < count; i++)
                        Check(mlx_vector_array_get(out results[i], outVec, (nuint)i), "extracting closure output");
                    return results;
                }
                finally
                {
                    _ = mlx_vector_array_free(inVec);
                    _ = mlx_vector_array_free(outVec);
                }
            });
        }

        internal static void FreeCompiledClosure(CompiledClosure holder)
        {
            if (holder == null || holder.Disposed) return;
            holder.Disposed = true;

            // Freeing the compiled closure fires our destructor, which frees
            // the payload GCHandle.
            MlxWorker.Shared.Dispatch(() =>
            {
                if (holder.Compiled.IsValid)
                    _ = mlx_closure_free(holder.Compiled);
                // Destructor handles GCHandle.Free(); guard against MLX not
                // firing it (e.g. if compile failed) by checking handle.
                if (holder.TraceHandle.IsAllocated)
                    holder.TraceHandle.Free();
            });
        }

        // Callback invoked by MLX during compile/apply. Runs on the worker
        // thread (since the calling Invoke already pinned it). Marshals the
        // C-side vector of arrays into a managed array, calls the C# trace
        // function, and stuffs the outputs back into the result vector.
        //
        // Memory ownership inside the callback:
        // - `mlx_vector_array_get` bumps a refcount on each input; we must
        //   free those handles before returning, otherwise every closure
        //   apply leaks one ref per input.
        // - The trace function returns output arrays whose refcount belongs
        //   to us. `mlx_vector_array_set_data` bumps that refcount when it
        //   stores them in `result`; we then release our reference so that
        //   when `result` is later freed (by the apply caller) the final
        //   ref drops as expected.
        private static unsafe int ClosureCallback(MlxVectorArray* result, MlxVectorArray input, IntPtr payload)
        {
            MlxArray[] inputs = null;
            MlxArray[] outputs = null;
            try
            {
                if (result == null || payload == IntPtr.Zero) return 1;
                var handle = GCHandle.FromIntPtr(payload);
                var trace = handle.Target as TraceFunc;
                if (trace == null) return 1;

                int n = (int)mlx_vector_array_size(input);
                inputs = new MlxArray[n];
                for (int i = 0; i < n; i++)
                {
                    if (mlx_vector_array_get(out inputs[i], input, (nuint)i) != 0)
                        return 1;
                }

                outputs = trace(inputs);
                if (outputs == null) outputs = Array.Empty<MlxArray>();

                // MlxArray is layout-compatible with the C struct
                // mlx_array { void* ctx; } so an MlxArray[] is a packed array
                // of pointers and we can pass it directly via fixed.
                int rc;
                unsafe
                {
                    if (outputs.Length == 0)
                    {
                        rc = mlx_vector_array_set_data(ref *result, IntPtr.Zero, 0);
                    }
                    else
                    {
                        fixed (MlxArray* p = outputs)
                        {
                            rc = mlx_vector_array_set_data(ref *result, (IntPtr)p, (nuint)outputs.Length);
                        }
                    }
                }
                return rc;
            }
            catch
            {
                // Returning non-zero signals an error to MLX; ClosureCallback
                // is reached during compile so MLX will surface this via the
                // error handler.
                return 1;
            }
            finally
            {
                // Release the per-call refs we acquired via vector_array_get
                // and the per-output refs the trace function handed back
                // (set_data took its own ref).
                if (inputs != null)
                {
                    for (int i = 0; i < inputs.Length; i++)
                        if (inputs[i].IsValid)
                            _ = mlx_array_free(inputs[i]);
                }
                if (outputs != null)
                {
                    for (int i = 0; i < outputs.Length; i++)
                        if (outputs[i].IsValid)
                            _ = mlx_array_free(outputs[i]);
                }
            }
        }

        private static void ClosureDestructor(IntPtr payload)
        {
            // MLX guarantees the destructor runs exactly once when the
            // compiled closure is freed. We free the GCHandle here; the
            // matching CompiledClosure.Disposed=true is set by our
            // FreeCompiledClosure caller (or it will be set after this fires
            // — either way GCHandle is freed exactly once because we guard
            // with IsAllocated on the manual path).
            try
            {
                if (payload == IntPtr.Zero) return;
                var handle = GCHandle.FromIntPtr(payload);
                if (handle.IsAllocated)
                    handle.Free();
            }
            catch
            {
            }
        }

        internal static void Eval(MlxArray array)
        {
            if (!array.IsValid)
                return;

            Eval(new[] { array });
        }

        internal static void Eval(params MlxArray[] arrays)
        {
            if (arrays == null || arrays.Length == 0)
                return;

            MlxWorker.Shared.Invoke(() =>
            {
                MlxVectorArray vector = mlx_vector_array_new();
                try
                {
                    int count = 0;
                    for (int i = 0; i < arrays.Length; i++)
                    {
                        if (arrays[i].IsValid)
                        {
                            Check(mlx_vector_array_append_value(vector, arrays[i]), "building MLX eval vector");
                            count++;
                        }
                    }
                    if (count > 0)
                        Check(mlx_eval(vector), "evaluating MLX graph");
                }
                finally
                {
                    _ = mlx_vector_array_free(vector);
                }
            });
        }

        // AsyncEval schedules graph execution on Metal without waiting for
        // completion. The next host read (CopyArrayToHost) calls mlx_eval which
        // drains the queue. Use this at layer boundaries during prefill/decode
        // so command-buffer issue overlaps with completion of earlier layers.
        internal static void AsyncEval(MlxArray array)
        {
            if (!array.IsValid)
                return;

            AsyncEval(new[] { array });
        }

        internal static void AsyncEval(params MlxArray[] arrays)
        {
            if (arrays == null || arrays.Length == 0)
                return;

            // We Dispatch (fire-and-forget) because AsyncEval itself returns
            // immediately on the MLX side once the graph has been enqueued;
            // there is nothing for the caller to wait on. The next Invoke from
            // this thread (e.g. a host copy) will serialize correctly via the
            // worker's FIFO queue.
            MlxWorker.Shared.Dispatch(() =>
            {
                MlxVectorArray vector = mlx_vector_array_new();
                try
                {
                    int count = 0;
                    for (int i = 0; i < arrays.Length; i++)
                    {
                        if (arrays[i].IsValid)
                        {
                            if (mlx_vector_array_append_value(vector, arrays[i]) != 0)
                                return;
                            count++;
                        }
                    }
                    if (count > 0)
                        _ = mlx_async_eval(vector);
                }
                finally
                {
                    _ = mlx_vector_array_free(vector);
                }
            });
        }

        /// <summary>
        /// A second reference to <paramref name="source"/> (mlx_array_set): the caller
        /// frees it independently. MLX arrays are immutable, so two references are as
        /// good as a copy until one side is replaced.
        /// </summary>
        internal static MlxArray Retain(MlxArray source)
        {
            if (!source.IsValid)
                return default;
            return MlxWorker.Shared.Invoke(() =>
            {
                MlxArray copy = mlx_array_new();
                Check(mlx_array_set(ref copy, source), "referencing MLX array");
                return copy;
            });
        }

        internal static void CopyArrayToHost(MlxArray array, DType dtype, IntPtr destination, long byteCount)
        {
            if (!array.IsValid)
                throw new ArgumentException("MLX array is empty.", nameof(array));
            if (destination == IntPtr.Zero && byteCount > 0)
                throw new ArgumentNullException(nameof(destination));
            if (byteCount < 0)
                throw new ArgumentOutOfRangeException(nameof(byteCount));

            MlxWorker.Shared.Invoke(() =>
            {
                MlxVectorArray vector = mlx_vector_array_new();
                try
                {
                    Check(mlx_vector_array_append_value(vector, array), "building MLX eval vector");
                    Check(mlx_eval(vector), "evaluating MLX array before host copy");

                    IntPtr source = dtype switch
                    {
                        DType.Float32 => mlx_array_data_float32(array),
                        DType.Float64 => mlx_array_data_float64(array),
                        DType.Float16 => mlx_array_data_float16(array),
                        DType.Int32 => mlx_array_data_int32(array),
                        DType.UInt8 => mlx_array_data_uint8(array),
                        _ => throw new NotSupportedException($"MLX host copy does not support {dtype}."),
                    };

                    if (source == IntPtr.Zero && byteCount > 0)
                        throw new InvalidOperationException("MLX returned a null data pointer.");

                    unsafe
                    {
                        Buffer.MemoryCopy(source.ToPointer(), destination.ToPointer(), byteCount, byteCount);
                    }
                }
                finally
                {
                    _ = mlx_vector_array_free(vector);
                }
            });
        }

        internal static MlxArray AsStrided(MlxArray array, int[] shape, long[] strides, long offset)
        {
            if (!array.IsValid)
                throw new ArgumentException("MLX array is empty.", nameof(array));
            if (shape == null || strides == null || shape.Length != strides.Length)
                throw new ArgumentException("Shape and strides must be non-null and have the same length.");
            if (offset < 0)
                throw new ArgumentOutOfRangeException(nameof(offset));

            return MlxWorker.Shared.Invoke(() =>
            {
                MlxArray result;
                Check(mlx_as_strided(out result, array, shape, (nuint)shape.Length, strides, (nuint)strides.Length, (nuint)offset, DefaultStream()), "creating MLX strided view");
                return result;
            });
        }

        internal static MlxArray Reshape(MlxArray array, int[] shape)
        {
            return MlxWorker.Shared.Invoke(() =>
            {
                MlxArray result;
                Check(mlx_reshape(out result, array, shape, (nuint)shape.Length, DefaultStream()), "reshaping MLX array");
                return result;
            });
        }

        internal static MlxArray Contiguous(MlxArray array)
        {
            return MlxWorker.Shared.Invoke(() =>
            {
                MlxArray result;
                Check(mlx_contiguous(out result, array, false, DefaultStream()), "making MLX array contiguous");
                return result;
            });
        }

        internal static MlxArray Transpose(MlxArray array, int[] axes)
        {
            if (!array.IsValid)
                throw new ArgumentException("MLX array is empty.", nameof(array));
            if (axes == null || axes.Length == 0)
                throw new ArgumentException("MLX transpose axes must be non-empty.", nameof(axes));

            return MlxWorker.Shared.Invoke(() =>
            {
                MlxArray result;
                Check(mlx_transpose_axes(out result, array, axes, (nuint)axes.Length, DefaultStream()), "transposing MLX array");
                return result;
            });
        }

        internal static MlxArray Unary(MlxUnaryOp op, MlxArray input)
        {
            return MlxWorker.Shared.Invoke(() =>
            {
                MlxArray result;
                int rc = op switch
                {
                    MlxUnaryOp.Abs => mlx_abs(out result, input, DefaultStream()),
                    MlxUnaryOp.Neg => mlx_negative(out result, input, DefaultStream()),
                    MlxUnaryOp.Sqrt => mlx_sqrt(out result, input, DefaultStream()),
                    MlxUnaryOp.Rsqrt => mlx_rsqrt(out result, input, DefaultStream()),
                    MlxUnaryOp.Exp => mlx_exp(out result, input, DefaultStream()),
                    MlxUnaryOp.Log => mlx_log(out result, input, DefaultStream()),
                    MlxUnaryOp.Log1p => mlx_log1p(out result, input, DefaultStream()),
                    MlxUnaryOp.Floor => mlx_floor(out result, input, DefaultStream()),
                    MlxUnaryOp.Ceil => mlx_ceil(out result, input, DefaultStream()),
                    MlxUnaryOp.Sin => mlx_sin(out result, input, DefaultStream()),
                    MlxUnaryOp.Cos => mlx_cos(out result, input, DefaultStream()),
                    MlxUnaryOp.Tanh => mlx_tanh(out result, input, DefaultStream()),
                    MlxUnaryOp.Sigmoid => mlx_sigmoid(out result, input, DefaultStream()),
                    _ => throw new NotSupportedException($"Unsupported MLX unary op {op}."),
                };
                Check(rc, $"running MLX unary op {op}");
                return result;
            });
        }

        internal static MlxArray Binary(MlxBinaryOp op, MlxArray lhs, MlxArray rhs)
        {
            return MlxWorker.Shared.Invoke(() =>
            {
                MlxArray result;
                int rc = op switch
                {
                    MlxBinaryOp.Add => mlx_add(out result, lhs, rhs, DefaultStream()),
                    MlxBinaryOp.Sub => mlx_subtract(out result, lhs, rhs, DefaultStream()),
                    MlxBinaryOp.Mul => mlx_multiply(out result, lhs, rhs, DefaultStream()),
                    MlxBinaryOp.Div => mlx_divide(out result, lhs, rhs, DefaultStream()),
                    MlxBinaryOp.Maximum => mlx_maximum(out result, lhs, rhs, DefaultStream()),
                    _ => throw new NotSupportedException($"Unsupported MLX binary op {op}."),
                };
                Check(rc, $"running MLX binary op {op}");
                return result;
            });
        }

        internal static MlxArray Remainder(MlxArray lhs, MlxArray rhs)
        {
            return MlxWorker.Shared.Invoke(() =>
            {
                MlxArray result;
                Check(mlx_remainder(out result, lhs, rhs, DefaultStream()), "running MLX remainder");
                return result;
            });
        }

        internal static MlxArray Greater(MlxArray lhs, MlxArray rhs)
        {
            return MlxWorker.Shared.Invoke(() =>
            {
                MlxArray result;
                Check(mlx_greater(out result, lhs, rhs, DefaultStream()), "running MLX greater");
                return result;
            });
        }

        internal static MlxArray Where(MlxArray condition, MlxArray whenTrue, MlxArray whenFalse)
        {
            return MlxWorker.Shared.Invoke(() =>
            {
                MlxArray result;
                Check(mlx_where(out result, condition, whenTrue, whenFalse, DefaultStream()), "running MLX where");
                return result;
            });
        }

        // gather_mm: batched matmul where the rows of `a` are gathered by
        // `lhsIndices` and the (head, expert, ...) rows of `b` are gathered by
        // `rhsIndices` before the matmul. Either indices array may be null.
        // Used for MoE routing / speculative decoding.
        // gather_qmm: fused gather + quantized matmul. Avoids the
        // materialize-and-route overhead for MoE: instead of dequantizing
        // each expert separately or gathering input rows then quantized-
        // matmul, MLX dispatches a single kernel that accesses only the
        // selected expert blocks.
        internal static MlxArray GatherQMM(MlxArray x, MlxArray w, MlxArray scales, MlxArray biases, MlxArray lhsIndices, MlxArray rhsIndices, bool transpose, int groupSize, int bits, string mode, bool sortedIndices)
        {
            if (!x.IsValid || !w.IsValid || !scales.IsValid)
                throw new ArgumentException("MLX gather_qmm inputs must be valid arrays.");

            IntPtr modePtr = mode != null ? Marshal.StringToCoTaskMemAnsi(mode) : IntPtr.Zero;
            try
            {
                return MlxWorker.Shared.Invoke(() =>
                {
                    MlxArray result;
                    Check(mlx_gather_qmm(out result, x, w, scales, biases, lhsIndices, rhsIndices, transpose, MlxOptionalInt.Some(groupSize), MlxOptionalInt.Some(bits), modePtr, sortedIndices, DefaultStream()), "running MLX gather_qmm");
                    return result;
                });
            }
            finally
            {
                if (modePtr != IntPtr.Zero) Marshal.FreeCoTaskMem(modePtr);
            }
        }

        internal static MlxArray Addmm(MlxArray src, MlxArray m1, MlxArray m2, float alpha, float beta)
        {
            return MlxWorker.Shared.Invoke(() =>
            {
                MlxArray result;
                Check(mlx_addmm(out result, src, m1, m2, alpha, beta, DefaultStream()), "running MLX addmm");
                return result;
            });
        }

        internal static MlxArray SoftmaxLastAxis(MlxArray input)
        {
            return MlxWorker.Shared.Invoke(() =>
            {
                MlxArray result;
                Check(mlx_softmax_axis(out result, input, -1, true, DefaultStream()), "running MLX softmax");
                return result;
            });
        }

        internal static MlxArray RepeatAxis(MlxArray input, int repeats, int axis)
        {
            if (!input.IsValid)
                throw new ArgumentException("MLX repeat input must be a valid array.");
            if (repeats < 1)
                throw new ArgumentOutOfRangeException(nameof(repeats));

            return MlxWorker.Shared.Invoke(() =>
            {
                MlxArray result;
                Check(mlx_repeat_axis(out result, input, repeats, axis, DefaultStream()), "running MLX repeat");
                return result;
            });
        }

        internal static MlxArray FastLayerNorm(MlxArray input, MlxArray weight, MlxArray bias, float eps)
        {
            return MlxWorker.Shared.Invoke(() =>
            {
                MlxArray result;
                Check(mlx_fast_layer_norm(out result, input, weight, bias, eps, DefaultStream()), "running MLX layer norm");
                return result;
            });
        }

        internal static MlxArray FastRmsNorm(MlxArray input, MlxArray weight, float eps)
        {
            return MlxWorker.Shared.Invoke(() =>
            {
                MlxArray result;
                Check(mlx_fast_rms_norm(out result, input, weight, eps, DefaultStream()), "running MLX RMS norm");
                return result;
            });
        }

        /// <param name="forceFused">Run MLX's fused attention kernel even for a shape it would
        /// otherwise compute unfused (materialised scores + softmax + matmul).</param>
        /// <param name="sinks">Optional per-query-head attention sinks (gpt-oss): one extra
        /// logit per head that joins the softmax denominator but contributes no value. Its
        /// dtype must promote to the query's.</param>
        internal static MlxArray FastScaledDotProductAttention(MlxArray query, MlxArray key, MlxArray value, float scale, string maskMode, MlxArray mask, bool forceFused = false, MlxArray sinks = default)
        {
            if (!query.IsValid || !key.IsValid || !value.IsValid)
                throw new ArgumentException("MLX attention inputs must be valid arrays.");

            IntPtr maskModePtr = GetModePtr(maskMode ?? string.Empty);
            return MlxWorker.Shared.Invoke(() =>
            {
                MlxArray result;
                Check(mlx_fast_scaled_dot_product_attention(out result, query, key, value, scale, maskModePtr, mask, sinks, forceFused, DefaultStream()), "running MLX scaled dot product attention");
                return result;
            });
        }

        internal static MlxArray ConcatenateAxis(MlxArray first, MlxArray second, int axis)
        {
            if (!first.IsValid || !second.IsValid)
                throw new ArgumentException("MLX concatenate inputs must be valid arrays.");

            return MlxWorker.Shared.Invoke(() =>
            {
                MlxVectorArray inputs = CreateVectorArray(first, second);
                try
                {
                    MlxArray result;
                    Check(mlx_concatenate_axis(out result, inputs, axis, DefaultStream()), "concatenating MLX arrays");
                    return result;
                }
                finally
                {
                    if (inputs.IsValid)
                        _ = mlx_vector_array_free(inputs);
                }
            });
        }

        internal static MlxArray HeadDim256Attention(
            MlxArray qHeads,
            MlxArray kHeads,
            MlxArray vHeads,
            int numHeads,
            int numKVHeads,
            int qLen,
            int kvLen,
            int maskStart,
            bool causal,
            float scale)
        {
            if (!qHeads.IsValid || !kHeads.IsValid || !vHeads.IsValid)
                throw new ArgumentException("MLX attention inputs must be valid arrays.");
            if (numHeads <= 0 || numKVHeads <= 0 || numHeads % numKVHeads != 0 || qLen <= 0 || kvLen <= 0)
                throw new ArgumentOutOfRangeException(nameof(qLen), "Invalid MLX attention dimensions.");

            return MlxWorker.Shared.Invoke(() =>
            {
                MlxFastMetalKernel kernel = EnsureHeadDim256AttentionKernel();
                MlxFastMetalKernelConfig config = mlx_fast_metal_kernel_config_new();
                MlxVectorArray inputs = default;
                MlxVectorArray outputs = default;
                MlxArray scaleArray = default;
                try
                {
                    AddTemplateInt(config, "NumHeads", numHeads);
                    AddTemplateInt(config, "NumKVHeads", numKVHeads);
                    AddTemplateInt(config, "QLen", qLen);
                    AddTemplateInt(config, "KvLen", kvLen);
                    AddTemplateInt(config, "MaskStart", maskStart);
                    AddTemplateInt(config, "Causal", causal ? 1 : 0);
                    int[] shape = { qLen, numHeads * 256 };
                    Check(mlx_fast_metal_kernel_config_add_output_arg(config, shape, (nuint)shape.Length, ToMlxDtype(DType.Float32)), "configuring MLX headDim256 attention output");
                    Check(mlx_fast_metal_kernel_config_set_grid(config, 256, numHeads, qLen), "configuring MLX headDim256 attention grid");
                    Check(mlx_fast_metal_kernel_config_set_thread_group(config, 256, 1, 1), "configuring MLX headDim256 attention threadgroup");

                    scaleArray = mlx_array_new_float32(scale);
                    inputs = CreateVectorArray(qHeads, kHeads, vHeads, scaleArray);
                    outputs = mlx_vector_array_new();
                    Check(mlx_fast_metal_kernel_apply(ref outputs, kernel, inputs, config, DefaultStream()), "running MLX headDim256 attention");
                    if (mlx_vector_array_size(outputs) < 1)
                        throw new InvalidOperationException("MLX headDim256 attention produced no output.");

                    Check(mlx_vector_array_get(out MlxArray result, outputs, 0), "reading MLX headDim256 attention output");
                    return result;
                }
                finally
                {
                    if (scaleArray.IsValid)
                        _ = mlx_array_free(scaleArray);
                    if (inputs.IsValid)
                        _ = mlx_vector_array_free(inputs);
                    if (outputs.IsValid)
                        _ = mlx_vector_array_free(outputs);
                    if (config.IsValid)
                        _ = mlx_fast_metal_kernel_config_free(config);
                }
            });
        }

        // Decode attention with per-head attention sinks (and optional
        // sliding-window mask). Used by GPT-OSS-style models. Reuses the
        // same caller convention as CircularDecodeAttention but adds a
        // sinks input and a MaskStart template parameter.
        //
        // attendStart: first cache position to attend to (max(0, kvLen -
        //              slidingWindow) for SWA layers, else 0).
        // attendEnd:   one past last cache position to attend to (=kvLen).
        internal static MlxArray DecodeAttentionWithSinks(
            MlxArray qFlat,
            MlxArray kCache,
            MlxArray vCache,
            MlxArray sinks,
            int numHeads,
            int numKVHeads,
            int headDim,
            int cacheLen,
            int attendStart,
            int attendEnd,
            float scale)
        {
            if (!qFlat.IsValid || !kCache.IsValid || !vCache.IsValid || !sinks.IsValid)
                throw new ArgumentException("MLX decode attention with sinks inputs must be valid arrays.");
            if (numHeads <= 0 || numKVHeads <= 0 || numHeads % numKVHeads != 0 ||
                headDim <= 0 || headDim > 256 || cacheLen <= 0 || attendEnd <= attendStart || attendEnd > cacheLen)
            {
                throw new ArgumentOutOfRangeException(nameof(headDim), "Invalid MLX decode attention with sinks dimensions.");
            }

            return MlxWorker.Shared.Invoke(() =>
            {
                MlxFastMetalKernel kernel = EnsureDecodeAttentionWithSinksKernel();
                MlxFastMetalKernelConfig config = mlx_fast_metal_kernel_config_new();
                MlxVectorArray inputs = default;
                MlxVectorArray outputs = default;
                MlxArray scaleArray = default; MlxArray maskStartArray = default; MlxArray attendLenArray = default;
                try
                {
                    AddTemplateInt(config, "NumHeads", numHeads);
                    AddTemplateInt(config, "NumKVHeads", numKVHeads);
                    AddTemplateInt(config, "HeadDim", headDim);
                    AddTemplateInt(config, "CacheLen", cacheLen);

                    int[] shape = { 1, numHeads * headDim };
                    Check(mlx_fast_metal_kernel_config_add_output_arg(config, shape, (nuint)shape.Length, ToMlxDtype(DType.Float32)), "configuring decode attention with sinks output");
                    Check(mlx_fast_metal_kernel_config_set_grid(config, 256, numHeads, 1), "configuring decode attention with sinks grid");
                    Check(mlx_fast_metal_kernel_config_set_thread_group(config, 256, 1, 1), "configuring decode attention with sinks threadgroup");

                    scaleArray = mlx_array_new_float32(scale);
                    maskStartArray = mlx_array_new_int(attendStart);
                    attendLenArray = mlx_array_new_int(attendEnd);
                    inputs = CreateVectorArray(qFlat, kCache, vCache, sinks, scaleArray, maskStartArray, attendLenArray);
                    outputs = mlx_vector_array_new();
                    Check(mlx_fast_metal_kernel_apply(ref outputs, kernel, inputs, config, DefaultStream()), "running decode attention with sinks");
                    if (mlx_vector_array_size(outputs) < 1)
                        throw new InvalidOperationException("decode attention with sinks kernel produced no output.");

                    Check(mlx_vector_array_get(out MlxArray result, outputs, 0), "reading decode attention with sinks output");
                    return result;
                }
                finally
                {
                    if (scaleArray.IsValid)
                        _ = mlx_array_free(scaleArray);
                    if (maskStartArray.IsValid)
                        _ = mlx_array_free(maskStartArray);
                    if (attendLenArray.IsValid)
                        _ = mlx_array_free(attendLenArray);
                    if (inputs.IsValid)
                        _ = mlx_vector_array_free(inputs);
                    if (outputs.IsValid)
                        _ = mlx_vector_array_free(outputs);
                    if (config.IsValid)
                        _ = mlx_fast_metal_kernel_config_free(config);
                }
            });
        }

        /// <summary>
        /// Fused Gemma 4 decode-step QKV preprocessing. See
        /// <see cref="Gemma4QkvPreprocessDecodeSource"/> for the kernel
        /// semantics. Replaces ~5 separate MLX dispatches per attention
        /// layer (Q-norm, K-norm, V-norm, Q-RoPE, K-RoPE) with one.
        /// </summary>
        internal static void Gemma4QkvPreprocessDecode(
            MlxArray qkv,
            MlxArray qNormW,
            MlxArray kNormW,
            MlxArray cosTable,
            MlxArray sinTable,
            int numHeads,
            int numKVHeads,
            int headDim,
            int rotHalf,
            float eps,
            out MlxArray qOut,
            out MlxArray kOut,
            out MlxArray vOut)
        {
            qOut = default;
            kOut = default;
            vOut = default;

            if (!qkv.IsValid || !qNormW.IsValid || !kNormW.IsValid || !cosTable.IsValid || !sinTable.IsValid)
                throw new ArgumentException("MLX Gemma4 QKV preprocess decode requires valid input arrays.");
            if (numHeads <= 0 || numKVHeads <= 0 || numHeads % numKVHeads != 0
                || headDim <= 0 || headDim > 512
                || (headDim & (headDim - 1)) != 0
                || rotHalf <= 0 || rotHalf * 2 > headDim)
            {
                throw new ArgumentOutOfRangeException(nameof(headDim),
                    "Gemma4 QKV preprocess decode requires HeadDim ≤ 512 power-of-two and RotHalf*2 ≤ HeadDim.");
            }

            MlxArray qResult = default;
            MlxArray kResult = default;
            MlxArray vResult = default;
            MlxWorker.Shared.Invoke(() =>
            {
                MlxFastMetalKernel kernel = EnsureGemma4QkvPreprocessDecodeKernel();
                MlxFastMetalKernelConfig config = mlx_fast_metal_kernel_config_new();
                MlxVectorArray inputs = default;
                MlxVectorArray outputs = default;
                MlxArray epsArray = default;
                try
                {
                    AddTemplateInt(config, "NumHeads", numHeads);
                    AddTemplateInt(config, "NumKVHeads", numKVHeads);
                    AddTemplateInt(config, "HeadDim", headDim);
                    AddTemplateInt(config, "RotHalf", rotHalf);

                    int[] qShape = { 1, numHeads * headDim };
                    int[] kShape = { numKVHeads, 1, headDim };
                    int[] vShape = { numKVHeads, 1, headDim };
                    Check(mlx_fast_metal_kernel_config_add_output_arg(config, qShape, (nuint)qShape.Length, ToMlxDtype(DType.Float32)), "configuring Gemma4 QKV preprocess q output");
                    Check(mlx_fast_metal_kernel_config_add_output_arg(config, kShape, (nuint)kShape.Length, ToMlxDtype(DType.Float32)), "configuring Gemma4 QKV preprocess k output");
                    Check(mlx_fast_metal_kernel_config_add_output_arg(config, vShape, (nuint)vShape.Length, ToMlxDtype(DType.Float32)), "configuring Gemma4 QKV preprocess v output");

                    int totalRows = numHeads + 2 * numKVHeads;
                    Check(mlx_fast_metal_kernel_config_set_grid(config, headDim, totalRows, 1), "configuring Gemma4 QKV preprocess grid");
                    Check(mlx_fast_metal_kernel_config_set_thread_group(config, headDim, 1, 1), "configuring Gemma4 QKV preprocess threadgroup");

                    epsArray = mlx_array_new_float32(eps);
                    inputs = CreateVectorArray(qkv, qNormW, kNormW, cosTable, sinTable, epsArray);
                    outputs = mlx_vector_array_new();
                    Check(mlx_fast_metal_kernel_apply(ref outputs, kernel, inputs, config, DefaultStream()), "running Gemma4 QKV preprocess decode kernel");
                    if (mlx_vector_array_size(outputs) < 3)
                        throw new InvalidOperationException("Gemma4 QKV preprocess decode kernel produced fewer than 3 outputs.");

                    Check(mlx_vector_array_get(out qResult, outputs, 0), "reading Gemma4 QKV preprocess q output");
                    Check(mlx_vector_array_get(out kResult, outputs, 1), "reading Gemma4 QKV preprocess k output");
                    Check(mlx_vector_array_get(out vResult, outputs, 2), "reading Gemma4 QKV preprocess v output");
                }
                finally
                {
                    if (epsArray.IsValid)
                        _ = mlx_array_free(epsArray);
                    if (inputs.IsValid)
                        _ = mlx_vector_array_free(inputs);
                    if (outputs.IsValid)
                        _ = mlx_vector_array_free(outputs);
                    if (config.IsValid)
                        _ = mlx_fast_metal_kernel_config_free(config);
                }
            });

            qOut = qResult;
            kOut = kResult;
            vOut = vResult;
        }

        internal static MlxArray CircularDecodeAttention(
            MlxArray qFlat,
            MlxArray kCache,
            MlxArray vCache,
            int numHeads,
            int numKVHeads,
            int headDim,
            int cacheLen,
            int firstSlot,
            int attendLen,
            float scale)
        {
            if (!qFlat.IsValid || !kCache.IsValid || !vCache.IsValid)
                throw new ArgumentException("MLX circular decode attention inputs must be valid arrays.");
            if (numHeads <= 0 || numKVHeads <= 0 || numHeads % numKVHeads != 0 ||
                headDim <= 0 || headDim > 256 || cacheLen <= 0 || attendLen <= 0 || attendLen > cacheLen)
            {
                throw new ArgumentOutOfRangeException(nameof(headDim), "Invalid MLX circular decode attention dimensions.");
            }

            firstSlot %= cacheLen;
            if (firstSlot < 0)
                firstSlot += cacheLen;

            return MlxWorker.Shared.Invoke(() =>
            {
                MlxFastMetalKernel kernel = EnsureCircularDecodeAttentionKernel();
                MlxFastMetalKernelConfig config = mlx_fast_metal_kernel_config_new();
                MlxVectorArray inputs = default;
                MlxVectorArray outputs = default;
                MlxArray scaleArray = default; MlxArray firstSlotArray = default; MlxArray attendLenArray = default;
                try
                {
                    AddTemplateInt(config, "NumHeads", numHeads);
                    AddTemplateInt(config, "NumKVHeads", numKVHeads);
                    AddTemplateInt(config, "HeadDim", headDim);
                    AddTemplateInt(config, "CacheLen", cacheLen);

                    int[] shape = { 1, numHeads * headDim };
                    Check(mlx_fast_metal_kernel_config_add_output_arg(config, shape, (nuint)shape.Length, ToMlxDtype(DType.Float32)), "configuring circular decode attention output");
                    Check(mlx_fast_metal_kernel_config_set_grid(config, 256, numHeads, 1), "configuring circular decode attention grid");
                    Check(mlx_fast_metal_kernel_config_set_thread_group(config, 256, 1, 1), "configuring circular decode attention threadgroup");

                    scaleArray = mlx_array_new_float32(scale);
                    firstSlotArray = mlx_array_new_int(firstSlot);
                    attendLenArray = mlx_array_new_int(attendLen);
                    inputs = CreateVectorArray(qFlat, kCache, vCache, scaleArray, firstSlotArray, attendLenArray);
                    outputs = mlx_vector_array_new();
                    Check(mlx_fast_metal_kernel_apply(ref outputs, kernel, inputs, config, DefaultStream()), "running circular decode attention");
                    if (mlx_vector_array_size(outputs) < 1)
                        throw new InvalidOperationException("circular decode attention kernel produced no output.");

                    Check(mlx_vector_array_get(out MlxArray result, outputs, 0), "reading circular decode attention output");
                    return result;
                }
                finally
                {
                    if (scaleArray.IsValid)
                        _ = mlx_array_free(scaleArray);
                    if (firstSlotArray.IsValid)
                        _ = mlx_array_free(firstSlotArray);
                    if (attendLenArray.IsValid)
                        _ = mlx_array_free(attendLenArray);
                    if (inputs.IsValid)
                        _ = mlx_vector_array_free(inputs);
                    if (outputs.IsValid)
                        _ = mlx_vector_array_free(outputs);
                    if (config.IsValid)
                        _ = mlx_fast_metal_kernel_config_free(config);
                }
            });
        }

        internal static MlxArray FastRopeDynamic(MlxArray input, int dims, bool traditional, float baseValue, float scale, MlxArray offsets)
        {
            if (!input.IsValid || !offsets.IsValid)
                throw new ArgumentException("MLX RoPE inputs must be valid arrays.");
            if (dims <= 0 || (dims & 1) != 0)
                throw new ArgumentOutOfRangeException(nameof(dims), "RoPE dimensions must be a positive even number.");

            return MlxWorker.Shared.Invoke(() =>
            {
                MlxArray result;
                Check(mlx_fast_rope_dynamic(
                    out result,
                    input,
                    dims,
                    traditional,
                    MlxOptionalFloat.Some(baseValue),
                    scale,
                    offsets,
                    default,
                    DefaultStream()), "running MLX RoPE");
                return result;
            });
        }

        /// <summary>RoPE with explicit per-pair wavelengths (angle = position * scale /
        /// freqs[i]) instead of a base, the form MLX takes for YaRN and other scaled
        /// variants.</summary>
        internal static MlxArray FastRopeDynamicWithFreqs(MlxArray input, int dims, bool traditional, float scale, MlxArray offsets, MlxArray freqs)
        {
            if (!input.IsValid || !offsets.IsValid || !freqs.IsValid)
                throw new ArgumentException("MLX RoPE inputs must be valid arrays.");
            if (dims <= 0 || (dims & 1) != 0)
                throw new ArgumentOutOfRangeException(nameof(dims), "RoPE dimensions must be a positive even number.");

            return MlxWorker.Shared.Invoke(() =>
            {
                MlxArray result;
                Check(mlx_fast_rope_dynamic(
                    out result,
                    input,
                    dims,
                    traditional,
                    MlxOptionalFloat.None,
                    scale,
                    offsets,
                    freqs,
                    DefaultStream()), "running MLX RoPE with frequencies");
                return result;
            });
        }

        internal static MlxArray TakeAxis(MlxArray input, MlxArray indices, int axis)
        {
            if (!input.IsValid || !indices.IsValid)
                throw new ArgumentException("MLX take inputs must be valid arrays.");

            return MlxWorker.Shared.Invoke(() =>
            {
                MlxArray result;
                Check(mlx_take_axis(out result, input, indices, axis, DefaultStream()), "running MLX take_axis");
                return result;
            });
        }

        // Per-row (take_along_axis) gather: for a 2D [R, C] input and a [R, K] index array,
        // result[r, j] = input[r, indices[r, j]] (axis = 1). Unlike TakeAxis (mx.take, which
        // gathers the SAME columns for every row), this gathers a different set per row — the
        // primitive needed for a batched MoE router's per-token top-K logit gather.
        internal static MlxArray TakeAlongAxis(MlxArray input, MlxArray indices, int axis)
        {
            if (!input.IsValid || !indices.IsValid)
                throw new ArgumentException("MLX take_along_axis inputs must be valid arrays.");

            return MlxWorker.Shared.Invoke(() =>
            {
                MlxArray result;
                Check(mlx_take_along_axis(out result, input, indices, axis, DefaultStream()), "running MLX take_along_axis");
                return result;
            });
        }

        // Argmax along the given axis. With keepDims=false the reduced axis
        // is dropped; on a [1, V] logits tensor with axis=-1 the result is
        // [1] uint32 — the predicted next token for greedy decoding.
        // Caller is expected to cast/reinterpret as needed.
        internal static MlxArray ArgMaxAxis(MlxArray input, int axis, bool keepDims)
        {
            if (!input.IsValid)
                throw new ArgumentException("MLX argmax input must be a valid array.");

            return MlxWorker.Shared.Invoke(() =>
            {
                MlxArray result;
                Check(mlx_argmax_axis(out result, input, axis, keepDims, DefaultStream()), "running MLX argmax_axis");
                return result;
            });
        }

        // Argpartition along the given axis. Returns indices such that:
        //   - indices[..., 0..kth-1] map to values <= indices[..., kth]'s value
        //   - indices[..., kth+1..end] map to values >= indices[..., kth]'s value
        // No ordering within either partition is guaranteed.
        // To get the top-K largest along an axis, negate the input first and
        // take the first K elements of the argpartition with kth=K-1.
        internal static MlxArray ArgPartitionAxis(MlxArray input, int kth, int axis)
        {
            if (!input.IsValid)
                throw new ArgumentException("MLX argpartition input must be a valid array.");

            return MlxWorker.Shared.Invoke(() =>
            {
                MlxArray result;
                Check(mlx_argpartition_axis(out result, input, kth, axis, DefaultStream()), "running MLX argpartition_axis");
                return result;
            });
        }

        internal static MlxArray ScatterAddWeightedRows(
            MlxArray output,
            MlxArray rows,
            MlxArray indices,
            MlxArray weights,
            int seqLen,
            int batchSize,
            int hiddenDim)
        {
            if (!output.IsValid || !rows.IsValid || !indices.IsValid || !weights.IsValid)
                throw new ArgumentException("MLX weighted scatter-add requires valid arrays.");
            if (seqLen <= 0 || batchSize <= 0 || hiddenDim <= 0)
                throw new ArgumentOutOfRangeException(nameof(batchSize));

            return MlxWorker.Shared.Invoke(() =>
            {
                MlxFastMetalKernel kernel = EnsureScatterAddWeightedRowsKernel();
                MlxFastMetalKernelConfig config = mlx_fast_metal_kernel_config_new();
                MlxVectorArray inputs = default;
                MlxVectorArray outputs = default;
                try
                {
                    AddTemplateInt(config, "SeqLen", seqLen);
                    AddTemplateInt(config, "BatchSize", batchSize);
                    AddTemplateInt(config, "HiddenDim", hiddenDim);

                    int[] shape = { seqLen, hiddenDim };
                    Check(mlx_fast_metal_kernel_config_add_output_arg(config, shape, (nuint)shape.Length, ToMlxDtype(DType.Float32)), "configuring weighted scatter-add output");
                    Check(mlx_fast_metal_kernel_config_set_grid(config, hiddenDim, seqLen, 1), "configuring weighted scatter-add grid");
                    Check(mlx_fast_metal_kernel_config_set_thread_group(config, 256, 1, 1), "configuring weighted scatter-add threadgroup");

                    inputs = CreateVectorArray(output, rows, indices, weights);
                    outputs = mlx_vector_array_new();
                    Check(mlx_fast_metal_kernel_apply(ref outputs, kernel, inputs, config, DefaultStream()), "running weighted scatter-add");
                    if (mlx_vector_array_size(outputs) < 1)
                        throw new InvalidOperationException("Weighted scatter-add kernel produced no output.");

                    Check(mlx_vector_array_get(out MlxArray result, outputs, 0), "reading weighted scatter-add output");
                    return result;
                }
                finally
                {
                    if (inputs.IsValid)
                        _ = mlx_vector_array_free(inputs);
                    if (outputs.IsValid)
                        _ = mlx_vector_array_free(outputs);
                    if (config.IsValid)
                        _ = mlx_fast_metal_kernel_config_free(config);
                }
            });
        }

        internal static MlxArray RmsNormAdd(
            MlxArray residual,
            MlxArray input,
            MlxArray normWeight,
            float eps,
            int rows,
            int hiddenDim)
        {
            if (!residual.IsValid || !input.IsValid || !normWeight.IsValid)
                throw new ArgumentException("MLX RMSNorm-add requires valid arrays.");
            if (rows <= 0 || hiddenDim <= 0)
                throw new ArgumentOutOfRangeException(nameof(rows));

            return MlxWorker.Shared.Invoke(() =>
            {
                MlxFastMetalKernel kernel = EnsureRmsNormAddKernel();
                MlxFastMetalKernelConfig config = mlx_fast_metal_kernel_config_new();
                MlxVectorArray inputs = default;
                MlxVectorArray outputs = default;
                MlxArray epsArray = default;
                try
                {
                    AddTemplateInt(config, "HiddenDim", hiddenDim);

                    int[] shape = { rows, hiddenDim };
                    Check(mlx_fast_metal_kernel_config_add_output_arg(config, shape, (nuint)shape.Length, ToMlxDtype(DType.Float32)), "configuring RMSNorm-add output");
                    Check(mlx_fast_metal_kernel_config_set_grid(config, 256, rows, 1), "configuring RMSNorm-add grid");
                    Check(mlx_fast_metal_kernel_config_set_thread_group(config, 256, 1, 1), "configuring RMSNorm-add threadgroup");

                    epsArray = mlx_array_new_float32(eps);
                    inputs = CreateVectorArray(residual, input, normWeight, epsArray);
                    outputs = mlx_vector_array_new();
                    Check(mlx_fast_metal_kernel_apply(ref outputs, kernel, inputs, config, DefaultStream()), "running RMSNorm-add");
                    if (mlx_vector_array_size(outputs) < 1)
                        throw new InvalidOperationException("RMSNorm-add kernel produced no output.");

                    Check(mlx_vector_array_get(out MlxArray result, outputs, 0), "reading RMSNorm-add output");
                    return result;
                }
                finally
                {
                    if (epsArray.IsValid)
                        _ = mlx_array_free(epsArray);
                    if (inputs.IsValid)
                        _ = mlx_vector_array_free(inputs);
                    if (outputs.IsValid)
                        _ = mlx_vector_array_free(outputs);
                    if (config.IsValid)
                        _ = mlx_fast_metal_kernel_config_free(config);
                }
            });
        }

        // Fused (residual += input; normed = RmsNorm(residual, norm_weight)).
        // Returns (updatedResidual, normedOut) - both new MlxArrays. Caller is
        // responsible for replacing residual's storage and consuming normed.
        internal static (MlxArray updatedResidual, MlxArray normed) AddRmsNorm(
            MlxArray residual,
            MlxArray input,
            MlxArray normWeight,
            float eps,
            int rows,
            int hiddenDim)
        {
            if (!residual.IsValid || !input.IsValid || !normWeight.IsValid)
                throw new ArgumentException("MLX add-rmsnorm requires valid arrays.");
            if (rows <= 0 || hiddenDim <= 0)
                throw new ArgumentOutOfRangeException(nameof(rows));

            return MlxWorker.Shared.Invoke(() =>
            {
                MlxFastMetalKernel kernel = EnsureAddRmsNormKernel();
                MlxFastMetalKernelConfig config = mlx_fast_metal_kernel_config_new();
                MlxVectorArray inputs = default;
                MlxVectorArray outputs = default;
                MlxArray epsArray = default;
                try
                {
                    AddTemplateInt(config, "HiddenDim", hiddenDim);

                    int[] shape = { rows, hiddenDim };
                    Check(mlx_fast_metal_kernel_config_add_output_arg(config, shape, (nuint)shape.Length, ToMlxDtype(DType.Float32)), "configuring add-rmsnorm residual output");
                    Check(mlx_fast_metal_kernel_config_add_output_arg(config, shape, (nuint)shape.Length, ToMlxDtype(DType.Float32)), "configuring add-rmsnorm normed output");
                    Check(mlx_fast_metal_kernel_config_set_grid(config, 256, rows, 1), "configuring add-rmsnorm grid");
                    Check(mlx_fast_metal_kernel_config_set_thread_group(config, 256, 1, 1), "configuring add-rmsnorm threadgroup");

                    epsArray = mlx_array_new_float32(eps);
                    inputs = CreateVectorArray(residual, input, normWeight, epsArray);
                    outputs = mlx_vector_array_new();
                    Check(mlx_fast_metal_kernel_apply(ref outputs, kernel, inputs, config, DefaultStream()), "running add-rmsnorm");
                    if (mlx_vector_array_size(outputs) < 2)
                        throw new InvalidOperationException("add-rmsnorm kernel produced fewer than 2 outputs.");

                    Check(mlx_vector_array_get(out MlxArray updated, outputs, 0), "reading add-rmsnorm updated residual");
                    Check(mlx_vector_array_get(out MlxArray normed, outputs, 1), "reading add-rmsnorm normed");
                    return (updated, normed);
                }
                finally
                {
                    if (epsArray.IsValid)
                        _ = mlx_array_free(epsArray);
                    if (inputs.IsValid)
                        _ = mlx_vector_array_free(inputs);
                    if (outputs.IsValid)
                        _ = mlx_vector_array_free(outputs);
                    if (config.IsValid)
                        _ = mlx_fast_metal_kernel_config_free(config);
                }
            });
        }

        internal static MlxArray GeluMulSplit(MlxArray gateUp, int rows, int halfDim)
        {
            if (!gateUp.IsValid)
                throw new ArgumentException("MLX GELU-mul split requires a valid gate_up array.");
            if (rows <= 0 || halfDim <= 0)
                throw new ArgumentOutOfRangeException(nameof(rows));

            return MlxWorker.Shared.Invoke(() =>
            {
                MlxFastMetalKernel kernel = EnsureGeluMulSplitKernel();
                MlxFastMetalKernelConfig config = mlx_fast_metal_kernel_config_new();
                MlxVectorArray inputs = default;
                MlxVectorArray outputs = default;
                try
                {
                    AddTemplateInt(config, "Rows", rows);
                    AddTemplateInt(config, "HalfDim", halfDim);

                    int[] shape = { rows, halfDim };
                    Check(mlx_fast_metal_kernel_config_add_output_arg(config, shape, (nuint)shape.Length, ToMlxDtype(DType.Float32)), "configuring GELU-mul split output");
                    Check(mlx_fast_metal_kernel_config_set_grid(config, halfDim, rows, 1), "configuring GELU-mul split grid");
                    Check(mlx_fast_metal_kernel_config_set_thread_group(config, 256, 1, 1), "configuring GELU-mul split threadgroup");

                    inputs = CreateVectorArray(gateUp);
                    outputs = mlx_vector_array_new();
                    Check(mlx_fast_metal_kernel_apply(ref outputs, kernel, inputs, config, DefaultStream()), "running GELU-mul split");
                    if (mlx_vector_array_size(outputs) < 1)
                        throw new InvalidOperationException("GELU-mul split kernel produced no output.");

                    Check(mlx_vector_array_get(out MlxArray result, outputs, 0), "reading GELU-mul split output");
                    return result;
                }
                finally
                {
                    if (inputs.IsValid)
                        _ = mlx_vector_array_free(inputs);
                    if (outputs.IsValid)
                        _ = mlx_vector_array_free(outputs);
                    if (config.IsValid)
                        _ = mlx_fast_metal_kernel_config_free(config);
                }
            });
        }

        // Fused clamped-SwiGLU (swiglu_oai) + per-expert gate/up bias gather.
        // gate/up: [rows, dim] f32. gateBias/upBias: [E, dim] f32. experts:
        // [rows] int32 (expert id of each row). Returns [rows, dim] f32.
        internal static MlxArray SwigluOaiGatherBias(
            MlxArray gate,
            MlxArray up,
            MlxArray gateBias,
            MlxArray upBias,
            MlxArray experts,
            float alpha,
            float limit,
            int rows,
            int dim)
        {
            if (!gate.IsValid || !up.IsValid || !gateBias.IsValid || !upBias.IsValid || !experts.IsValid)
                throw new ArgumentException("MLX swiglu-oai gather-bias requires valid arrays.");
            if (rows <= 0 || dim <= 0)
                throw new ArgumentOutOfRangeException(nameof(rows));

            return MlxWorker.Shared.Invoke(() =>
            {
                MlxFastMetalKernel kernel = EnsureSwigluOaiGatherBiasKernel();
                MlxFastMetalKernelConfig config = mlx_fast_metal_kernel_config_new();
                MlxVectorArray inputs = default;
                MlxVectorArray outputs = default;
                MlxArray alphaArray = default;
                MlxArray limitArray = default;
                try
                {
                    AddTemplateInt(config, "Rows", rows);
                    AddTemplateInt(config, "Dim", dim);

                    int[] shape = { rows, dim };
                    Check(mlx_fast_metal_kernel_config_add_output_arg(config, shape, (nuint)shape.Length, ToMlxDtype(DType.Float32)), "configuring swiglu-oai output");
                    Check(mlx_fast_metal_kernel_config_set_grid(config, dim, rows, 1), "configuring swiglu-oai grid");
                    Check(mlx_fast_metal_kernel_config_set_thread_group(config, 256, 1, 1), "configuring swiglu-oai threadgroup");

                    alphaArray = mlx_array_new_float32(alpha);
                    limitArray = mlx_array_new_float32(limit);
                    inputs = CreateVectorArray(gate, up, gateBias, upBias, experts, alphaArray, limitArray);
                    outputs = mlx_vector_array_new();
                    Check(mlx_fast_metal_kernel_apply(ref outputs, kernel, inputs, config, DefaultStream()), "running swiglu-oai gather-bias");
                    if (mlx_vector_array_size(outputs) < 1)
                        throw new InvalidOperationException("swiglu-oai gather-bias kernel produced no output.");

                    Check(mlx_vector_array_get(out MlxArray result, outputs, 0), "reading swiglu-oai output");
                    return result;
                }
                finally
                {
                    if (alphaArray.IsValid)
                        _ = mlx_array_free(alphaArray);
                    if (limitArray.IsValid)
                        _ = mlx_array_free(limitArray);
                    if (inputs.IsValid)
                        _ = mlx_vector_array_free(inputs);
                    if (outputs.IsValid)
                        _ = mlx_vector_array_free(outputs);
                    if (config.IsValid)
                        _ = mlx_fast_metal_kernel_config_free(config);
                }
            });
        }

        // Routing-weighted MoE combine + per-expert down-bias gather + unsort.
        // downRows: [n*k, dim] f32 in expert-sorted pair order. downBias:
        // [E, dim] f32 (pass any valid array with hasBias=false to skip).
        // expertsSorted / invOrder: [n*k] int32. routeWeights: [n*k] f32 in
        // ORIGINAL (token-major) pair order. Returns [n, dim] f32.
        internal static MlxArray MoeBiasWeightedSum(
            MlxArray downRows,
            MlxArray downBias,
            bool hasBias,
            MlxArray expertsSorted,
            MlxArray invOrder,
            MlxArray routeWeights,
            int n,
            int k,
            int dim)
        {
            if (!downRows.IsValid || !downBias.IsValid || !expertsSorted.IsValid || !invOrder.IsValid || !routeWeights.IsValid)
                throw new ArgumentException("MLX MoE bias-weighted-sum requires valid arrays.");
            if (n <= 0 || k <= 0 || dim <= 0)
                throw new ArgumentOutOfRangeException(nameof(n));

            return MlxWorker.Shared.Invoke(() =>
            {
                MlxFastMetalKernel kernel = EnsureMoeBiasWeightedSumKernel();
                MlxFastMetalKernelConfig config = mlx_fast_metal_kernel_config_new();
                MlxVectorArray inputs = default;
                MlxVectorArray outputs = default;
                try
                {
                    AddTemplateInt(config, "Rows", n);
                    AddTemplateInt(config, "K", k);
                    AddTemplateInt(config, "Dim", dim);
                    AddTemplateInt(config, "HasBias", hasBias ? 1 : 0);

                    int[] shape = { n, dim };
                    Check(mlx_fast_metal_kernel_config_add_output_arg(config, shape, (nuint)shape.Length, ToMlxDtype(DType.Float32)), "configuring MoE bias-weighted-sum output");
                    Check(mlx_fast_metal_kernel_config_set_grid(config, dim, n, 1), "configuring MoE bias-weighted-sum grid");
                    Check(mlx_fast_metal_kernel_config_set_thread_group(config, 256, 1, 1), "configuring MoE bias-weighted-sum threadgroup");

                    inputs = CreateVectorArray(downRows, downBias, expertsSorted, invOrder, routeWeights);
                    outputs = mlx_vector_array_new();
                    Check(mlx_fast_metal_kernel_apply(ref outputs, kernel, inputs, config, DefaultStream()), "running MoE bias-weighted-sum");
                    if (mlx_vector_array_size(outputs) < 1)
                        throw new InvalidOperationException("MoE bias-weighted-sum kernel produced no output.");

                    Check(mlx_vector_array_get(out MlxArray result, outputs, 0), "reading MoE bias-weighted-sum output");
                    return result;
                }
                finally
                {
                    if (inputs.IsValid)
                        _ = mlx_vector_array_free(inputs);
                    if (outputs.IsValid)
                        _ = mlx_vector_array_free(outputs);
                    if (config.IsValid)
                        _ = mlx_fast_metal_kernel_config_free(config);
                }
            });
        }

        internal static MlxArray FlatToHeadFirst(
            MlxArray input,
            int seqLen,
            int numHeads,
            int headDim,
            int sourceStride,
            int colOffset)
        {
            if (!input.IsValid)
                throw new ArgumentException("MLX flat-to-head-first requires a valid input array.");
            if (seqLen <= 0 || numHeads <= 0 || headDim <= 0 || sourceStride <= 0 || colOffset < 0)
                throw new ArgumentOutOfRangeException(nameof(seqLen));
            if (colOffset + numHeads * headDim > sourceStride)
                throw new ArgumentOutOfRangeException(nameof(colOffset));

            return MlxWorker.Shared.Invoke(() =>
            {
                MlxFastMetalKernel kernel = EnsureFlatToHeadFirstKernel();
                MlxFastMetalKernelConfig config = mlx_fast_metal_kernel_config_new();
                MlxVectorArray inputs = default;
                MlxVectorArray outputs = default;
                try
                {
                    AddTemplateInt(config, "SeqLen", seqLen);
                    AddTemplateInt(config, "NumHeads", numHeads);
                    AddTemplateInt(config, "HeadDim", headDim);
                    AddTemplateInt(config, "SourceStride", sourceStride);
                    AddTemplateInt(config, "ColOffset", colOffset);

                    int[] shape = { numHeads, seqLen, headDim };
                    Check(mlx_fast_metal_kernel_config_add_output_arg(config, shape, (nuint)shape.Length, ToMlxDtype(DType.Float32)), "configuring flat-to-head-first output");
                    Check(mlx_fast_metal_kernel_config_set_grid(config, headDim, seqLen, numHeads), "configuring flat-to-head-first grid");
                    Check(mlx_fast_metal_kernel_config_set_thread_group(config, 256, 1, 1), "configuring flat-to-head-first threadgroup");

                    inputs = CreateVectorArray(input);
                    outputs = mlx_vector_array_new();
                    Check(mlx_fast_metal_kernel_apply(ref outputs, kernel, inputs, config, DefaultStream()), "running flat-to-head-first");
                    if (mlx_vector_array_size(outputs) < 1)
                        throw new InvalidOperationException("flat-to-head-first kernel produced no output.");

                    Check(mlx_vector_array_get(out MlxArray result, outputs, 0), "reading flat-to-head-first output");
                    return result;
                }
                finally
                {
                    if (inputs.IsValid)
                        _ = mlx_vector_array_free(inputs);
                    if (outputs.IsValid)
                        _ = mlx_vector_array_free(outputs);
                    if (config.IsValid)
                        _ = mlx_fast_metal_kernel_config_free(config);
                }
            });
        }

        internal static MlxArray NeoXRoPE(
            MlxArray input,
            MlxArray cosTable,
            MlxArray sinTable,
            int numHeads,
            int seqLen,
            int headDim,
            int rotHalf,
            bool headFirst)
        {
            if (!input.IsValid || !cosTable.IsValid || !sinTable.IsValid)
                throw new ArgumentException("MLX NeoX RoPE requires valid input and table arrays.");
            if (numHeads <= 0 || seqLen <= 0 || headDim <= 0 || rotHalf <= 0 || rotHalf * 2 > headDim)
                throw new ArgumentOutOfRangeException(nameof(headDim));

            return MlxWorker.Shared.Invoke(() =>
            {
                MlxFastMetalKernel kernel = EnsureNeoXRopeKernel();
                MlxFastMetalKernelConfig config = mlx_fast_metal_kernel_config_new();
                MlxVectorArray inputs = default;
                MlxVectorArray outputs = default;
                try
                {
                    AddTemplateInt(config, "NumHeads", numHeads);
                    AddTemplateInt(config, "SeqLen", seqLen);
                    AddTemplateInt(config, "HeadDim", headDim);
                    AddTemplateInt(config, "RotHalf", rotHalf);
                    AddTemplateInt(config, "HeadFirst", headFirst ? 1 : 0);

                    int[] shape = headFirst
                        ? new[] { numHeads, seqLen, headDim }
                        : new[] { seqLen, numHeads * headDim };
                    Check(mlx_fast_metal_kernel_config_add_output_arg(config, shape, (nuint)shape.Length, ToMlxDtype(DType.Float32)), "configuring NeoX RoPE output");
                    Check(mlx_fast_metal_kernel_config_set_grid(config, headDim, numHeads, seqLen), "configuring NeoX RoPE grid");
                    Check(mlx_fast_metal_kernel_config_set_thread_group(config, 256, 1, 1), "configuring NeoX RoPE threadgroup");

                    inputs = CreateVectorArray(input, cosTable, sinTable);
                    outputs = mlx_vector_array_new();
                    Check(mlx_fast_metal_kernel_apply(ref outputs, kernel, inputs, config, DefaultStream()), "running NeoX RoPE");
                    if (mlx_vector_array_size(outputs) < 1)
                        throw new InvalidOperationException("NeoX RoPE kernel produced no output.");

                    Check(mlx_vector_array_get(out MlxArray result, outputs, 0), "reading NeoX RoPE output");
                    return result;
                }
                finally
                {
                    if (inputs.IsValid)
                        _ = mlx_vector_array_free(inputs);
                    if (outputs.IsValid)
                        _ = mlx_vector_array_free(outputs);
                    if (config.IsValid)
                        _ = mlx_fast_metal_kernel_config_free(config);
                }
            });
        }

        // Cache the small set of mode strings ("affine", "mxfp4", "q4_k",
        // "q5_k") used by quantized matmul / dequantize so the hot path
        // doesn't pay Marshal.StringToHGlobalAnsi + FreeHGlobal per call
        // (~1µs × ~5 quantized matmuls per layer × 42 layers = ~0.2ms/token).
        // The native callee only reads the bytes, never frees them, so a
        // process-lifetime pin is safe.
        private static readonly Dictionary<string, IntPtr> s_ModePtrCache = new(StringComparer.Ordinal);
        private static IntPtr GetModePtr(string mode)
        {
            string key = mode ?? "affine";
            lock (s_ModePtrCache)
            {
                if (!s_ModePtrCache.TryGetValue(key, out IntPtr ptr))
                {
                    ptr = Marshal.StringToHGlobalAnsi(key);
                    s_ModePtrCache[key] = ptr;
                }
                return ptr;
            }
        }

        internal static MlxArray QuantizedMatmul(MlxArray input, MlxArray weight, MlxArray scales, MlxArray biases, bool transpose, int groupSize, int bits, string mode)
        {
            if (!input.IsValid || !weight.IsValid || !scales.IsValid)
                throw new ArgumentException("MLX quantized matmul inputs must be valid arrays.");
            if (groupSize <= 0)
                throw new ArgumentOutOfRangeException(nameof(groupSize));
            if (bits <= 0)
                throw new ArgumentOutOfRangeException(nameof(bits));

            IntPtr modePtr = GetModePtr(mode);
            return MlxWorker.Shared.Invoke(() =>
            {
                MlxArray result;
                Check(mlx_quantized_matmul(
                    out result,
                    input,
                    weight,
                    scales,
                    biases,
                    transpose,
                    MlxOptionalInt.Some(groupSize),
                    MlxOptionalInt.Some(bits),
                    modePtr,
                    DefaultStream()), "running MLX quantized matmul");
                return result;
            });
        }

        internal static MlxArray Dequantize(MlxArray weight, MlxArray scales, MlxArray biases, int groupSize, int bits, string mode, DType dtype)
        {
            if (!weight.IsValid || !scales.IsValid)
                throw new ArgumentException("MLX dequantize inputs must be valid arrays.");
            if (groupSize <= 0)
                throw new ArgumentOutOfRangeException(nameof(groupSize));
            if (bits <= 0)
                throw new ArgumentOutOfRangeException(nameof(bits));

            IntPtr modePtr = GetModePtr(mode);
            return MlxWorker.Shared.Invoke(() =>
            {
                MlxArray result;
                Check(mlx_dequantize(
                    out result,
                    weight,
                    scales,
                    biases,
                    MlxOptionalInt.Some(groupSize),
                    MlxOptionalInt.Some(bits),
                    modePtr,
                    default,
                    MlxOptionalDType.Some(ToMlxDtype(dtype)),
                    DefaultStream()), "running MLX dequantize");
                return result;
            });
        }

        // Rows up to which IQ4_XS runs as matrix-vector products (the ggml port).
        // TS_MLX_IQ4XS_MATVEC_MAX_ROWS=0 restores the older kernels. Settable for tests.
        internal static int Iq4XsMatvecMaxRows = ResolveMatvecMaxRows("TS_MLX_IQ4XS_MATVEC_MAX_ROWS");
        // Above the matrix-vector rows, IQ4_XS takes the F16 dequantize + GEMM path of
        // Q6_K (DequantF16Matmul). TS_MLX_IQ4XS_DEQUANT_GEMM=0 keeps the older kernel.
        private static readonly bool Iq4XsDequantGemm =
            !string.Equals(Environment.GetEnvironmentVariable("TS_MLX_IQ4XS_DEQUANT_GEMM"), "0", StringComparison.Ordinal);

        internal static MlxArray Iq4XsMatmul(MlxArray input, MlxArray rawWeight, int rows, int inDim, int outDim)
        {
            if (!input.IsValid || !rawWeight.IsValid)
                throw new ArgumentException("IQ4_XS matmul requires valid input and raw weight arrays.");
            if (rows <= 0 || inDim <= 0 || outDim <= 0 || inDim % 256 != 0)
                throw new ArgumentOutOfRangeException(nameof(inDim), "IQ4_XS matmul requires positive dimensions and input dim aligned to 256.");
            if (rows <= Iq4XsMatvecMaxRows)
            {
                try
                {
                    return Iq4XsMatvec(input, rawWeight, rows, inDim, outDim);
                }
                catch (NotSupportedException)
                {
                }
            }
            else if (Iq4XsDequantGemm)
            {
                try
                {
                    return Iq4XsDequantMatmul(input, rawWeight, rows, inDim, outDim);
                }
                catch (NotSupportedException)
                {
                }
            }
            if (rows == 1 && !string.Equals(Environment.GetEnvironmentVariable("TS_MLX_IQ4XS_MATMUL4"), "0", StringComparison.Ordinal))
            {
                // Two kernels are available for decode (rows==1):
                //
                //   - Iq4XsMatmul4Cols (default): 256 threads / threadgroup,
                //     4 output cols per threadgroup, threadgroup-memory
                //     reduction with 8 barriers. The legacy implementation.
                //
                //   - Iq4XsMatmul4ColsSimd: 128 threads / threadgroup,
                //     4 simdgroups, simd_sum reduction (no barriers, no
                //     shared memory). Cleaner code, eliminates ~8 barriers
                //     per output column. On the M4 Pro / 27B-IQ4_XS
                //     benchmark the two were within 1% of each other —
                //     the matmul is memory-bandwidth bound (we hit ~30%
                //     of M4 Pro peak BW, which both kernels saturate
                //     equally). Kept opt-in via TS_MLX_IQ4XS_MATMUL4_SIMD=1
                //     so future Metal/MLX toolchains that benefit more
                //     from the cleaner reduction can flip it on without
                //     a code change.
                bool useSimd = string.Equals(
                    Environment.GetEnvironmentVariable("TS_MLX_IQ4XS_MATMUL4_SIMD"),
                    "1", StringComparison.Ordinal);
                if (useSimd)
                {
                    try
                    {
                        return Iq4XsMatmul4ColsSimd(input, rawWeight, inDim, outDim);
                    }
                    catch (Exception)
                    {
                        // Fall through to legacy kernel.
                    }
                }

                try
                {
                    return Iq4XsMatmul4Cols(input, rawWeight, inDim, outDim);
                }
                catch (Exception)
                {
                }
            }

            if (rows > 1 && rows <= Iq4XsBatchedRows2Max())
            {
                try
                {
                    return Iq4XsMatmulRows2Cols(input, rawWeight, rows, inDim, outDim);
                }
                catch (Exception)
                {
                }
            }

            if (rows > 1 && rows <= Iq4XsBatchedRowsMax())
            {
                return Iq4XsMatmulRows(input, rawWeight, rows, inDim, outDim);
            }

            // Large-batch prefill: simdgroup_matrix tile-based kernel.
            // Threshold the same as IQ2_XXS (rows >= 8) but in practice
            // we get here only when rows > Iq4XsBatchedRowsMax() = 24,
            // so the simdgroup variant always runs at large batch.
            if (rows >= Iq2XxsMatmulSimdgroupMinRows && !Iq2XxsMatmulSimdgroupDisabled)
            {
                try { return Iq4XsMatmulSimdgroup(input, rawWeight, rows, inDim, outDim); }
                catch (NotSupportedException) { }
            }

            return MlxWorker.Shared.Invoke(() =>
            {
                MlxFastMetalKernel kernel = EnsureIq4XsMatmulKernel();
                MlxFastMetalKernelConfig config = mlx_fast_metal_kernel_config_new();
                MlxVectorArray inputs = default;
                MlxVectorArray outputs = default;
                try
                {
                    AddTemplateInt(config, "InDim", inDim);
                    AddTemplateInt(config, "OutDim", outDim);
                    AddTemplateInt(config, "BlocksPerRow", inDim / 256);
                    int[] shape = { rows, outDim };
                    Check(mlx_fast_metal_kernel_config_add_output_arg(config, shape, (nuint)shape.Length, ToMlxDtype(DType.Float32)), "configuring IQ4_XS matmul output");
                    Check(mlx_fast_metal_kernel_config_set_grid(config, 256, outDim, rows), "configuring IQ4_XS matmul grid");
                    Check(mlx_fast_metal_kernel_config_set_thread_group(config, 256, 1, 1), "configuring IQ4_XS matmul threadgroup");

                    inputs = CreateVectorArray(input, rawWeight);
                    outputs = mlx_vector_array_new();
                    Check(mlx_fast_metal_kernel_apply(ref outputs, kernel, inputs, config, DefaultStream()), "running IQ4_XS matmul");
                    if (mlx_vector_array_size(outputs) < 1)
                        throw new InvalidOperationException("IQ4_XS matmul produced no output.");

                    Check(mlx_vector_array_get(out MlxArray result, outputs, 0), "reading IQ4_XS matmul output");
                    return result;
                }
                finally
                {
                    if (inputs.IsValid)
                        _ = mlx_vector_array_free(inputs);
                    if (outputs.IsValid)
                        _ = mlx_vector_array_free(outputs);
                    if (config.IsValid)
                        _ = mlx_fast_metal_kernel_config_free(config);
                }
            });
        }

        // Per-call IQ4_NL matmul on MLX. Used by MlxQuantizedOps after the
        // preload step (CreateIq4NlRawWeight) registers the GGUF mmap bytes
        // as a single MLX uchar[1, ne0*ne1*18/32] array. Grid: 256-thread
        // simdgroup per (row, out_col), reducing across InDim into shared
        // memory. Requires InDim % 32 == 0 (IQ4_NL block size).
        //
        // For rows > 1 (prefill, multi-token MoE expert batching) the basic
        // kernel issues `rows × outDim` threadgroups — each independently
        // re-reads the weight bytes for its (k, out_col). A multi-row variant
        // (`Iq4NlMatmulRows`) is available that batches across rows in one
        // threadgroup with weight reuse, but on this model's typical
        // workload (each MoE expert sees ~1-2 routed rows during
        // `TryMoEPrefillBatchedByExpert`, plus Mamba2/attention matmuls
        // aren't IQ4_NL) it failed to improve perf in benchmarks and
        // measurably hurt it on some runs — so the dispatch deliberately
        // stays on the basic kernel. The Rows kernel definition is kept in
        // the source for future use.
        internal static MlxArray Iq4NlMatmul(MlxArray input, MlxArray rawWeight, int rows, int inDim, int outDim)
        {
            if (!input.IsValid || !rawWeight.IsValid)
                throw new ArgumentException("IQ4_NL matmul requires valid input and raw weight arrays.");
            if (rows <= 0 || inDim <= 0 || outDim <= 0 || inDim % 32 != 0)
                throw new ArgumentOutOfRangeException(nameof(inDim), "IQ4_NL matmul requires positive dimensions and input dim aligned to 32.");

            return MlxWorker.Shared.Invoke(() =>
            {
                MlxFastMetalKernel kernel = EnsureIq4NlMatmulKernel();
                MlxFastMetalKernelConfig config = mlx_fast_metal_kernel_config_new();
                MlxVectorArray inputs = default;
                MlxVectorArray outputs = default;
                try
                {
                    AddTemplateInt(config, "InDim", inDim);
                    AddTemplateInt(config, "OutDim", outDim);
                    // BlocksPerRow = inDim / 32 (IQ4_NL: 32 elements per block).
                    AddTemplateInt(config, "BlocksPerRow", inDim / 32);
                    int[] shape = { rows, outDim };
                    Check(mlx_fast_metal_kernel_config_add_output_arg(config, shape, (nuint)shape.Length, ToMlxDtype(DType.Float32)), "configuring IQ4_NL matmul output");
                    Check(mlx_fast_metal_kernel_config_set_grid(config, 256, outDim, rows), "configuring IQ4_NL matmul grid");
                    Check(mlx_fast_metal_kernel_config_set_thread_group(config, 256, 1, 1), "configuring IQ4_NL matmul threadgroup");

                    inputs = CreateVectorArray(input, rawWeight);
                    outputs = mlx_vector_array_new();
                    Check(mlx_fast_metal_kernel_apply(ref outputs, kernel, inputs, config, DefaultStream()), "running IQ4_NL matmul");
                    if (mlx_vector_array_size(outputs) < 1)
                        throw new InvalidOperationException("IQ4_NL matmul produced no output.");

                    Check(mlx_vector_array_get(out MlxArray result, outputs, 0), "reading IQ4_NL matmul output");
                    return result;
                }
                finally
                {
                    if (inputs.IsValid)
                        _ = mlx_vector_array_free(inputs);
                    if (outputs.IsValid)
                        _ = mlx_vector_array_free(outputs);
                    if (config.IsValid)
                        _ = mlx_fast_metal_kernel_config_free(config);
                }
            });
        }

        // Multi-row IQ4_NL matmul. One threadgroup per `out_col` handles all
        // `rows` rows, reusing the dequantised weight across rows. Caller
        // must already have validated that 2 <= rows <= Iq4NlBatchedRowsMax.
        private static MlxArray Iq4NlMatmulRows(MlxArray input, MlxArray rawWeight, int rows, int inDim, int outDim)
        {
            return MlxWorker.Shared.Invoke(() =>
            {
                MlxFastMetalKernel kernel = EnsureIq4NlMatmulRowsKernel();
                MlxFastMetalKernelConfig config = mlx_fast_metal_kernel_config_new();
                MlxVectorArray inputs = default;
                MlxVectorArray outputs = default;
                try
                {
                    AddTemplateInt(config, "Rows", rows);
                    AddTemplateInt(config, "InDim", inDim);
                    AddTemplateInt(config, "OutDim", outDim);
                    AddTemplateInt(config, "BlocksPerRow", inDim / 32);
                    int[] shape = { rows, outDim };
                    Check(mlx_fast_metal_kernel_config_add_output_arg(config, shape, (nuint)shape.Length, ToMlxDtype(DType.Float32)), "configuring IQ4_NL multi-row matmul output");
                    Check(mlx_fast_metal_kernel_config_set_grid(config, 256, outDim, 1), "configuring IQ4_NL multi-row matmul grid");
                    Check(mlx_fast_metal_kernel_config_set_thread_group(config, 256, 1, 1), "configuring IQ4_NL multi-row matmul threadgroup");

                    inputs = CreateVectorArray(input, rawWeight);
                    outputs = mlx_vector_array_new();
                    Check(mlx_fast_metal_kernel_apply(ref outputs, kernel, inputs, config, DefaultStream()), "running IQ4_NL multi-row matmul");
                    if (mlx_vector_array_size(outputs) < 1)
                        throw new InvalidOperationException("IQ4_NL multi-row matmul produced no output.");

                    Check(mlx_vector_array_get(out MlxArray result, outputs, 0), "reading IQ4_NL multi-row matmul output");
                    return result;
                }
                finally
                {
                    if (inputs.IsValid)
                        _ = mlx_vector_array_free(inputs);
                    if (outputs.IsValid)
                        _ = mlx_vector_array_free(outputs);
                    if (config.IsValid)
                        _ = mlx_fast_metal_kernel_config_free(config);
                }
            });
        }


        // simd_sum-based 4-col matmul. See Iq4XsMatmul4SimdSource for the
        // kernel; grid is the same as the legacy kernel (ceil(OutDim/4)
        // threadgroups in y) but threadgroup size shrinks from 256 → 128
        // threads (= 4 simdgroups) since each simdgroup now owns one
        // output column and reduces via simd_sum instead of cooperating
        // with the rest of the threadgroup through shared memory.
        private static MlxArray Iq4XsMatmul4ColsSimd(MlxArray input, MlxArray rawWeight, int inDim, int outDim)
        {
            if (!input.IsValid || !rawWeight.IsValid)
                throw new ArgumentException("IQ4_XS simd_sum 4-column matmul requires valid input and raw weight arrays.");
            if (inDim <= 0 || outDim <= 0 || inDim % 256 != 0)
                throw new ArgumentOutOfRangeException(nameof(inDim), "IQ4_XS simd_sum 4-column matmul requires positive dimensions and input dim aligned to 256.");

            return MlxWorker.Shared.Invoke(() =>
            {
                MlxFastMetalKernel kernel = EnsureIq4XsMatmul4SimdKernel();
                MlxFastMetalKernelConfig config = mlx_fast_metal_kernel_config_new();
                MlxVectorArray inputs = default;
                MlxVectorArray outputs = default;
                try
                {
                    AddTemplateInt(config, "InDim", inDim);
                    AddTemplateInt(config, "OutDim", outDim);
                    AddTemplateInt(config, "BlocksPerRow", inDim / 256);
                    int[] shape = { 1, outDim };
                    Check(mlx_fast_metal_kernel_config_add_output_arg(config, shape, (nuint)shape.Length, ToMlxDtype(DType.Float32)), "configuring IQ4_XS simd 4-column matmul output");
                    // 128 threads (= 4 simdgroups) per threadgroup; each
                    // simdgroup owns one of the 4 output cols. Grid in y
                    // is ceil(OutDim/4) so the dispatch shape matches the
                    // legacy kernel exactly.
                    Check(mlx_fast_metal_kernel_config_set_grid(config, 128, (outDim + 3) / 4, 1), "configuring IQ4_XS simd 4-column matmul grid");
                    Check(mlx_fast_metal_kernel_config_set_thread_group(config, 128, 1, 1), "configuring IQ4_XS simd 4-column matmul threadgroup");

                    inputs = CreateVectorArray(input, rawWeight);
                    outputs = mlx_vector_array_new();
                    Check(mlx_fast_metal_kernel_apply(ref outputs, kernel, inputs, config, DefaultStream()), "running IQ4_XS simd 4-column matmul");
                    if (mlx_vector_array_size(outputs) < 1)
                        throw new InvalidOperationException("IQ4_XS simd 4-column matmul produced no output.");

                    Check(mlx_vector_array_get(out MlxArray result, outputs, 0), "reading IQ4_XS simd 4-column matmul output");
                    return result;
                }
                finally
                {
                    if (inputs.IsValid)
                        _ = mlx_vector_array_free(inputs);
                    if (outputs.IsValid)
                        _ = mlx_vector_array_free(outputs);
                    if (config.IsValid)
                        _ = mlx_fast_metal_kernel_config_free(config);
                }
            });
        }

        private static MlxArray Iq4XsMatmul4Cols(MlxArray input, MlxArray rawWeight, int inDim, int outDim)
        {
            if (!input.IsValid || !rawWeight.IsValid)
                throw new ArgumentException("IQ4_XS 4-column matmul requires valid input and raw weight arrays.");
            if (inDim <= 0 || outDim <= 0 || inDim % 256 != 0)
                throw new ArgumentOutOfRangeException(nameof(inDim), "IQ4_XS 4-column matmul requires positive dimensions and input dim aligned to 256.");

            return MlxWorker.Shared.Invoke(() =>
            {
                MlxFastMetalKernel kernel = EnsureIq4XsMatmul4Kernel();
                MlxFastMetalKernelConfig config = mlx_fast_metal_kernel_config_new();
                MlxVectorArray inputs = default;
                MlxVectorArray outputs = default;
                try
                {
                    AddTemplateInt(config, "InDim", inDim);
                    AddTemplateInt(config, "OutDim", outDim);
                    AddTemplateInt(config, "BlocksPerRow", inDim / 256);
                    int[] shape = { 1, outDim };
                    Check(mlx_fast_metal_kernel_config_add_output_arg(config, shape, (nuint)shape.Length, ToMlxDtype(DType.Float32)), "configuring IQ4_XS 4-column matmul output");
                    Check(mlx_fast_metal_kernel_config_set_grid(config, 256, (outDim + 3) / 4, 1), "configuring IQ4_XS 4-column matmul grid");
                    Check(mlx_fast_metal_kernel_config_set_thread_group(config, 256, 1, 1), "configuring IQ4_XS 4-column matmul threadgroup");

                    inputs = CreateVectorArray(input, rawWeight);
                    outputs = mlx_vector_array_new();
                    Check(mlx_fast_metal_kernel_apply(ref outputs, kernel, inputs, config, DefaultStream()), "running IQ4_XS 4-column matmul");
                    if (mlx_vector_array_size(outputs) < 1)
                        throw new InvalidOperationException("IQ4_XS 4-column matmul produced no output.");

                    Check(mlx_vector_array_get(out MlxArray result, outputs, 0), "reading IQ4_XS 4-column matmul output");
                    return result;
                }
                finally
                {
                    if (inputs.IsValid)
                        _ = mlx_vector_array_free(inputs);
                    if (outputs.IsValid)
                        _ = mlx_vector_array_free(outputs);
                    if (config.IsValid)
                        _ = mlx_fast_metal_kernel_config_free(config);
                }
            });
        }

        private static MlxArray Iq4XsMatmulRows(MlxArray input, MlxArray rawWeight, int rows, int inDim, int outDim)
        {
            return MlxWorker.Shared.Invoke(() =>
            {
                MlxFastMetalKernel kernel = EnsureIq4XsMatmulRowsKernel();
                MlxFastMetalKernelConfig config = mlx_fast_metal_kernel_config_new();
                MlxVectorArray inputs = default;
                MlxVectorArray outputs = default;
                try
                {
                    AddTemplateInt(config, "Rows", rows);
                    AddTemplateInt(config, "InDim", inDim);
                    AddTemplateInt(config, "OutDim", outDim);
                    AddTemplateInt(config, "BlocksPerRow", inDim / 256);
                    int[] shape = { rows, outDim };
                    Check(mlx_fast_metal_kernel_config_add_output_arg(config, shape, (nuint)shape.Length, ToMlxDtype(DType.Float32)), "configuring IQ4_XS batched-row matmul output");
                    Check(mlx_fast_metal_kernel_config_set_grid(config, 256, outDim, 1), "configuring IQ4_XS batched-row matmul grid");
                    Check(mlx_fast_metal_kernel_config_set_thread_group(config, 256, 1, 1), "configuring IQ4_XS batched-row matmul threadgroup");

                    inputs = CreateVectorArray(input, rawWeight);
                    outputs = mlx_vector_array_new();
                    Check(mlx_fast_metal_kernel_apply(ref outputs, kernel, inputs, config, DefaultStream()), "running IQ4_XS batched-row matmul");
                    if (mlx_vector_array_size(outputs) < 1)
                        throw new InvalidOperationException("IQ4_XS batched-row matmul produced no output.");

                    Check(mlx_vector_array_get(out MlxArray result, outputs, 0), "reading IQ4_XS batched-row matmul output");
                    return result;
                }
                finally
                {
                    if (inputs.IsValid)
                        _ = mlx_vector_array_free(inputs);
                    if (outputs.IsValid)
                        _ = mlx_vector_array_free(outputs);
                    if (config.IsValid)
                        _ = mlx_fast_metal_kernel_config_free(config);
                }
            });
        }

        private static MlxArray Iq4XsMatmulRows2Cols(MlxArray input, MlxArray rawWeight, int rows, int inDim, int outDim)
        {
            return MlxWorker.Shared.Invoke(() =>
            {
                MlxFastMetalKernel kernel = EnsureIq4XsMatmulRows2Kernel();
                MlxFastMetalKernelConfig config = mlx_fast_metal_kernel_config_new();
                MlxVectorArray inputs = default;
                MlxVectorArray outputs = default;
                try
                {
                    AddTemplateInt(config, "InDim", inDim);
                    AddTemplateInt(config, "OutDim", outDim);
                    AddTemplateInt(config, "BlocksPerRow", inDim / 256);
                    AddTemplateInt(config, "Rows", rows);
                    int[] shape = { rows, outDim };
                    Check(mlx_fast_metal_kernel_config_add_output_arg(config, shape, (nuint)shape.Length, ToMlxDtype(DType.Float32)), "configuring IQ4_XS 2-column batched matmul output");
                    Check(mlx_fast_metal_kernel_config_set_grid(config, 256, (outDim + 1) / 2, 1), "configuring IQ4_XS 2-column batched matmul grid");
                    Check(mlx_fast_metal_kernel_config_set_thread_group(config, 256, 1, 1), "configuring IQ4_XS 2-column batched matmul threadgroup");

                    inputs = CreateVectorArray(input, rawWeight);
                    outputs = mlx_vector_array_new();
                    Check(mlx_fast_metal_kernel_apply(ref outputs, kernel, inputs, config, DefaultStream()), "running IQ4_XS 2-column batched matmul");
                    if (mlx_vector_array_size(outputs) < 1)
                        throw new InvalidOperationException("IQ4_XS 2-column batched matmul produced no output.");

                    Check(mlx_vector_array_get(out MlxArray result, outputs, 0), "reading IQ4_XS 2-column batched matmul output");
                    return result;
                }
                finally
                {
                    if (inputs.IsValid)
                        _ = mlx_vector_array_free(inputs);
                    if (outputs.IsValid)
                        _ = mlx_vector_array_free(outputs);
                    if (config.IsValid)
                        _ = mlx_fast_metal_kernel_config_free(config);
                }
            });
        }

        private static int Iq4XsBatchedRowsMax()
        {
            string env = Environment.GetEnvironmentVariable("TS_MLX_IQ4XS_BATCHED_ROWS_MAX");
            if (!string.IsNullOrWhiteSpace(env) && int.TryParse(env, out int parsed))
                return Math.Clamp(parsed, 0, 24);

            if (string.Equals(Environment.GetEnvironmentVariable("TS_MLX_IQ4XS_BATCHED_ROWS"), "0", StringComparison.Ordinal))
                return 0;

            return 24;
        }

        private static int Iq4XsBatchedRows2Max()
        {
            if (!string.Equals(Environment.GetEnvironmentVariable("TS_MLX_IQ4XS_BATCHED_COLS"), "1", StringComparison.Ordinal))
                return 0;

            return Math.Min(Iq4XsBatchedRowsMax(), 16);
        }

        internal static MlxArray Iq4XsGetRows(MlxArray rawWeight, MlxArray indices, int rows, int inDim)
        {
            if (!rawWeight.IsValid || !indices.IsValid)
                throw new ArgumentException("IQ4_XS get_rows requires valid raw weight and index arrays.");
            if (rows <= 0 || inDim <= 0 || inDim % 256 != 0)
                throw new ArgumentOutOfRangeException(nameof(inDim), "IQ4_XS get_rows requires positive dimensions and input dim aligned to 256.");

            return MlxWorker.Shared.Invoke(() =>
            {
                MlxFastMetalKernel kernel = EnsureIq4XsGetRowsKernel();
                MlxFastMetalKernelConfig config = mlx_fast_metal_kernel_config_new();
                MlxVectorArray inputs = default;
                MlxVectorArray outputs = default;
                try
                {
                    AddTemplateInt(config, "InDim", inDim);
                    AddTemplateInt(config, "BlocksPerRow", inDim / 256);
                    int[] shape = { rows, inDim };
                    Check(mlx_fast_metal_kernel_config_add_output_arg(config, shape, (nuint)shape.Length, ToMlxDtype(DType.Float32)), "configuring IQ4_XS get_rows output");
                    Check(mlx_fast_metal_kernel_config_set_grid(config, inDim, rows, 1), "configuring IQ4_XS get_rows grid");
                    Check(mlx_fast_metal_kernel_config_set_thread_group(config, 256, 1, 1), "configuring IQ4_XS get_rows threadgroup");

                    inputs = CreateVectorArray(rawWeight, indices);
                    outputs = mlx_vector_array_new();
                    Check(mlx_fast_metal_kernel_apply(ref outputs, kernel, inputs, config, DefaultStream()), "running IQ4_XS get_rows");
                    if (mlx_vector_array_size(outputs) < 1)
                        throw new InvalidOperationException("IQ4_XS get_rows produced no output.");

                    Check(mlx_vector_array_get(out MlxArray result, outputs, 0), "reading IQ4_XS get_rows output");
                    return result;
                }
                finally
                {
                    if (inputs.IsValid)
                        _ = mlx_vector_array_free(inputs);
                    if (outputs.IsValid)
                        _ = mlx_vector_array_free(outputs);
                    if (config.IsValid)
                        _ = mlx_fast_metal_kernel_config_free(config);
                }
            });
        }

        // Minimum batch (input row) count for routing to the
        // simdgroup_matrix variant. Below this we keep the per-output
        // kernel (better for batch=1 decode). Tunable via env for A/B.
        private static readonly int Iq2XxsMatmulSimdgroupMinRows =
            int.TryParse(Environment.GetEnvironmentVariable("TS_MLX_IQ2XXS_SG_MIN_ROWS"), out int parsed) && parsed > 0
                ? parsed
                : 8;
        private static readonly bool Iq2XxsMatmulSimdgroupDisabled =
            string.Equals(Environment.GetEnvironmentVariable("TS_MLX_IQ2XXS_DISABLE_SG"), "1", StringComparison.Ordinal);

        internal static MlxArray Iq2XxsMatmul(MlxArray input, MlxArray rawWeight, int rows, int inDim, int outDim)
        {
            if (!input.IsValid || !rawWeight.IsValid)
                throw new ArgumentException("IQ2_XXS matmul requires valid input and raw weight arrays.");
            if (rows <= 0 || inDim <= 0 || outDim <= 0 || inDim % 256 != 0)
                throw new ArgumentOutOfRangeException(nameof(inDim), "IQ2_XXS matmul requires positive dimensions and input dim aligned to 256.");

            // For prefill (rows >= threshold) route to the simdgroup_matrix
            // kernel which is ~8× faster on the dequant pass and uses
            // hardware matrix-multiply for the FMA. Decode (rows < 8)
            // keeps the per-output kernel — the simdgroup kernel would
            // pad to 8 rows and waste 7/8 of the compute.
            if (rows >= Iq2XxsMatmulSimdgroupMinRows && !Iq2XxsMatmulSimdgroupDisabled)
            {
                try
                {
                    return Iq2XxsMatmulSimdgroup(input, rawWeight, rows, inDim, outDim);
                }
                catch (NotSupportedException)
                {
                    // Kernel compilation failed (e.g. older MSL toolchain
                    // without simdgroup_matrix). Fall back to the
                    // per-output kernel — disabled permanently below.
                }
            }

            return MlxWorker.Shared.Invoke(() =>
            {
                MlxFastMetalKernel kernel = EnsureIq2XxsMatmulKernel();
                MlxFastMetalKernelConfig config = mlx_fast_metal_kernel_config_new();
                MlxVectorArray inputs = default;
                MlxVectorArray outputs = default;
                try
                {
                    AddTemplateInt(config, "InDim", inDim);
                    AddTemplateInt(config, "OutDim", outDim);
                    AddTemplateInt(config, "BlocksPerRow", inDim / 256);
                    int[] shape = { rows, outDim };
                    Check(mlx_fast_metal_kernel_config_add_output_arg(config, shape, (nuint)shape.Length, ToMlxDtype(DType.Float32)), "configuring IQ2_XXS matmul output");
                    Check(mlx_fast_metal_kernel_config_set_grid(config, 256, outDim, rows), "configuring IQ2_XXS matmul grid");
                    Check(mlx_fast_metal_kernel_config_set_thread_group(config, 256, 1, 1), "configuring IQ2_XXS matmul threadgroup");

                    inputs = CreateVectorArray(input, rawWeight);
                    outputs = mlx_vector_array_new();
                    Check(mlx_fast_metal_kernel_apply(ref outputs, kernel, inputs, config, DefaultStream()), "running IQ2_XXS matmul");
                    if (mlx_vector_array_size(outputs) < 1)
                        throw new InvalidOperationException("IQ2_XXS matmul produced no output.");

                    Check(mlx_vector_array_get(out MlxArray result, outputs, 0), "reading IQ2_XXS matmul output");
                    return result;
                }
                finally
                {
                    if (inputs.IsValid)
                        _ = mlx_vector_array_free(inputs);
                    if (outputs.IsValid)
                        _ = mlx_vector_array_free(outputs);
                    if (config.IsValid)
                        _ = mlx_fast_metal_kernel_config_free(config);
                }
            });
        }

        // simdgroup_matrix-accelerated IQ2_XXS matmul. Grid is sized in
        // 8-element tiles; the kernel handles non-multiple-of-8 row /
        // column counts via bounds checks at load/store.
        internal static MlxArray Iq2XxsMatmulSimdgroup(MlxArray input, MlxArray rawWeight, int rows, int inDim, int outDim)
        {
            if (!input.IsValid || !rawWeight.IsValid)
                throw new ArgumentException("IQ2_XXS simdgroup matmul requires valid input and raw weight arrays.");
            if (rows <= 0 || inDim <= 0 || outDim <= 0 || inDim % 8 != 0)
                throw new ArgumentOutOfRangeException(nameof(inDim), "IQ2_XXS simdgroup matmul requires positive dimensions and inDim divisible by 8.");

            return MlxWorker.Shared.Invoke(() =>
            {
                MlxFastMetalKernel kernel = EnsureIq2XxsMatmulSimdgroupKernel();
                MlxFastMetalKernelConfig config = mlx_fast_metal_kernel_config_new();
                MlxVectorArray inputs = default;
                MlxVectorArray outputs = default;
                try
                {
                    AddTemplateInt(config, "InDim", inDim);
                    AddTemplateInt(config, "OutDim", outDim);
                    AddTemplateInt(config, "InRows", rows);
                    AddTemplateInt(config, "BlocksPerRow", inDim / 256);
                    int[] shape = { rows, outDim };
                    Check(mlx_fast_metal_kernel_config_add_output_arg(config, shape, (nuint)shape.Length, ToMlxDtype(DType.Float32)), "configuring IQ2_XXS simdgroup matmul output");

                    int tilesM = (outDim + 7) / 8;
                    int tilesB = (rows + 7) / 8;
                    Check(mlx_fast_metal_kernel_config_set_grid(config, 32, tilesM, tilesB), "configuring IQ2_XXS simdgroup matmul grid");
                    Check(mlx_fast_metal_kernel_config_set_thread_group(config, 32, 1, 1), "configuring IQ2_XXS simdgroup matmul threadgroup");

                    inputs = CreateVectorArray(input, rawWeight);
                    outputs = mlx_vector_array_new();
                    Check(mlx_fast_metal_kernel_apply(ref outputs, kernel, inputs, config, DefaultStream()), "running IQ2_XXS simdgroup matmul");
                    if (mlx_vector_array_size(outputs) < 1)
                        throw new InvalidOperationException("IQ2_XXS simdgroup matmul produced no output.");

                    Check(mlx_vector_array_get(out MlxArray result, outputs, 0), "reading IQ2_XXS simdgroup matmul output");
                    return result;
                }
                finally
                {
                    if (inputs.IsValid)
                        _ = mlx_vector_array_free(inputs);
                    if (outputs.IsValid)
                        _ = mlx_vector_array_free(outputs);
                    if (config.IsValid)
                        _ = mlx_fast_metal_kernel_config_free(config);
                }
            });
        }

        // Batched IQ2_XXS MoE matmul (shared single input row).
        // input: [1, inDim] - the decode-token hidden state
        // stackedWeight: stacked uint8 IQ2_XXS bytes for all experts
        // expertIndices: [K] int32 - which K experts to compute
        // Output: [K, outDim] - per-expert dense result
        // Batched IQ4_NL MoE matmul — shared input. Per (k_idx, out_col):
        //     y[k_idx, out_col] = dot(x[0, :], W[expert_indices[k_idx], out_col, :])
        // Replaces K per-expert matmul dispatches with ONE kernel call for
        // the up-projection of the MoE FFN block. Caller invariants match
        // the IQ2_XXS variant: shared single-row input, [K, outDim] output,
        // expert indices on device.
        internal static MlxArray Iq4NlMoeMatmulBatched(
            MlxArray input,
            MlxArray stackedWeight,
            MlxArray expertIndices,
            int K,
            int inDim,
            int outDim)
        {
            if (!input.IsValid || !stackedWeight.IsValid || !expertIndices.IsValid)
                throw new ArgumentException("IQ4_NL MoE batched matmul requires valid input, weight, and expertIndices arrays.");
            if (K <= 0 || inDim <= 0 || outDim <= 0 || inDim % 32 != 0)
                throw new ArgumentOutOfRangeException(nameof(inDim),
                    "IQ4_NL MoE batched matmul requires positive dimensions and input dim aligned to 32.");

            return MlxWorker.Shared.Invoke(() =>
            {
                MlxFastMetalKernel kernel = EnsureIq4NlMoeMatmulBatchedKernel();
                MlxFastMetalKernelConfig config = mlx_fast_metal_kernel_config_new();
                MlxVectorArray inputs = default;
                MlxVectorArray outputs = default;
                try
                {
                    AddTemplateInt(config, "InDim", inDim);
                    AddTemplateInt(config, "OutDim", outDim);
                    AddTemplateInt(config, "BlocksPerRow", inDim / 32);
                    int[] shape = { K, outDim };
                    Check(mlx_fast_metal_kernel_config_add_output_arg(config, shape, (nuint)shape.Length, ToMlxDtype(DType.Float32)),
                        "configuring IQ4_NL MoE batched matmul output");
                    Check(mlx_fast_metal_kernel_config_set_grid(config, 256, outDim, K),
                        "configuring IQ4_NL MoE batched matmul grid");
                    Check(mlx_fast_metal_kernel_config_set_thread_group(config, 256, 1, 1),
                        "configuring IQ4_NL MoE batched matmul threadgroup");

                    inputs = CreateVectorArray(input, stackedWeight, expertIndices);
                    outputs = mlx_vector_array_new();
                    Check(mlx_fast_metal_kernel_apply(ref outputs, kernel, inputs, config, DefaultStream()),
                        "running IQ4_NL MoE batched matmul");
                    if (mlx_vector_array_size(outputs) < 1)
                        throw new InvalidOperationException("IQ4_NL MoE batched matmul produced no output.");

                    Check(mlx_vector_array_get(out MlxArray result, outputs, 0),
                        "reading IQ4_NL MoE batched matmul output");
                    return result;
                }
                finally
                {
                    if (inputs.IsValid) _ = mlx_vector_array_free(inputs);
                    if (outputs.IsValid) _ = mlx_vector_array_free(outputs);
                    if (config.IsValid) _ = mlx_fast_metal_kernel_config_free(config);
                }
            });
        }

        // Batched IQ4_NL MoE matmul — per-row (rowed) variant for the
        // down-projection. Input is [K, inDim] where row k holds expert
        // `expert_indices[k]`'s post-activation output. The expert weight
        // is looked up per row via the same indices tensor.
        internal static MlxArray Iq4NlMoeMatmulBatchedRowed(
            MlxArray input,
            MlxArray stackedWeight,
            MlxArray expertIndices,
            int K,
            int inDim,
            int outDim)
        {
            if (!input.IsValid || !stackedWeight.IsValid || !expertIndices.IsValid)
                throw new ArgumentException("IQ4_NL MoE batched (rowed) matmul requires valid input, weight, and expertIndices arrays.");
            if (K <= 0 || inDim <= 0 || outDim <= 0 || inDim % 32 != 0)
                throw new ArgumentOutOfRangeException(nameof(inDim),
                    "IQ4_NL MoE batched (rowed) matmul requires positive dimensions and input dim aligned to 32.");

            return MlxWorker.Shared.Invoke(() =>
            {
                MlxFastMetalKernel kernel = EnsureIq4NlMoeMatmulBatchedRowedKernel();
                MlxFastMetalKernelConfig config = mlx_fast_metal_kernel_config_new();
                MlxVectorArray inputs = default;
                MlxVectorArray outputs = default;
                try
                {
                    AddTemplateInt(config, "InDim", inDim);
                    AddTemplateInt(config, "OutDim", outDim);
                    AddTemplateInt(config, "BlocksPerRow", inDim / 32);
                    int[] shape = { K, outDim };
                    Check(mlx_fast_metal_kernel_config_add_output_arg(config, shape, (nuint)shape.Length, ToMlxDtype(DType.Float32)),
                        "configuring IQ4_NL MoE batched (rowed) matmul output");
                    Check(mlx_fast_metal_kernel_config_set_grid(config, 256, outDim, K),
                        "configuring IQ4_NL MoE batched (rowed) matmul grid");
                    Check(mlx_fast_metal_kernel_config_set_thread_group(config, 256, 1, 1),
                        "configuring IQ4_NL MoE batched (rowed) matmul threadgroup");

                    inputs = CreateVectorArray(input, stackedWeight, expertIndices);
                    outputs = mlx_vector_array_new();
                    Check(mlx_fast_metal_kernel_apply(ref outputs, kernel, inputs, config, DefaultStream()),
                        "running IQ4_NL MoE batched (rowed) matmul");
                    if (mlx_vector_array_size(outputs) < 1)
                        throw new InvalidOperationException("IQ4_NL MoE batched (rowed) matmul produced no output.");

                    Check(mlx_vector_array_get(out MlxArray result, outputs, 0),
                        "reading IQ4_NL MoE batched (rowed) matmul output");
                    return result;
                }
                finally
                {
                    if (inputs.IsValid) _ = mlx_vector_array_free(inputs);
                    if (outputs.IsValid) _ = mlx_vector_array_free(outputs);
                    if (config.IsValid) _ = mlx_fast_metal_kernel_config_free(config);
                }
            });
        }

        internal static MlxArray Iq2XxsMoeMatmulBatched(
            MlxArray input,
            MlxArray stackedWeight,
            MlxArray expertIndices,
            int K,
            int inDim,
            int outDim)
        {
            if (!input.IsValid || !stackedWeight.IsValid || !expertIndices.IsValid)
                throw new ArgumentException("IQ2_XXS MoE batched matmul requires valid input, weight, and expertIndices arrays.");
            if (K <= 0 || inDim <= 0 || outDim <= 0 || inDim % 256 != 0)
                throw new ArgumentOutOfRangeException(nameof(inDim),
                    "IQ2_XXS MoE batched matmul requires positive dimensions and input dim aligned to 256.");

            return MlxWorker.Shared.Invoke(() =>
            {
                MlxFastMetalKernel kernel = EnsureIq2XxsMoeMatmulBatchedKernel();
                MlxFastMetalKernelConfig config = mlx_fast_metal_kernel_config_new();
                MlxVectorArray inputs = default;
                MlxVectorArray outputs = default;
                try
                {
                    AddTemplateInt(config, "InDim", inDim);
                    AddTemplateInt(config, "OutDim", outDim);
                    AddTemplateInt(config, "BlocksPerRow", inDim / 256);
                    int[] shape = { K, outDim };
                    Check(mlx_fast_metal_kernel_config_add_output_arg(config, shape, (nuint)shape.Length, ToMlxDtype(DType.Float32)),
                        "configuring IQ2_XXS MoE batched matmul output");
                    Check(mlx_fast_metal_kernel_config_set_grid(config, 256, outDim, K),
                        "configuring IQ2_XXS MoE batched matmul grid");
                    Check(mlx_fast_metal_kernel_config_set_thread_group(config, 256, 1, 1),
                        "configuring IQ2_XXS MoE batched matmul threadgroup");

                    inputs = CreateVectorArray(input, stackedWeight, expertIndices);
                    outputs = mlx_vector_array_new();
                    Check(mlx_fast_metal_kernel_apply(ref outputs, kernel, inputs, config, DefaultStream()),
                        "running IQ2_XXS MoE batched matmul");
                    if (mlx_vector_array_size(outputs) < 1)
                        throw new InvalidOperationException("IQ2_XXS MoE batched matmul produced no output.");

                    Check(mlx_vector_array_get(out MlxArray result, outputs, 0),
                        "reading IQ2_XXS MoE batched matmul output");
                    return result;
                }
                finally
                {
                    if (inputs.IsValid) _ = mlx_vector_array_free(inputs);
                    if (outputs.IsValid) _ = mlx_vector_array_free(outputs);
                    if (config.IsValid) _ = mlx_fast_metal_kernel_config_free(config);
                }
            });
        }

        // Fused IQ2_XXS MoE gate + up + SiLUMul (shared single input row).
        // input:        [1, inDim] - the decode-token hidden state
        // stackedGate:  stacked uint8 IQ2_XXS bytes for all experts' gate
        // stackedUp:    stacked uint8 IQ2_XXS bytes for all experts' up
        // expertIndices:[K] int32  - which K experts to compute
        // Output:       [K, outDim] - silu(gate_dot) * up_dot per (k, out_col)
        //
        // This replaces the {gate matmul + up matmul + SiLUMul} 3-dispatch
        // sequence with ONE dispatch. The gate and up matrices share the
        // input row so we read input once per K-chunk and reuse it for
        // both. The per-(k, out_col) thread group reduces partial sums
        // for both gate and up in parallel, then a single thread writes
        // silu(gate) * up to the output.
        internal static MlxArray Iq2XxsMoeMatmulBatchedFusedGateUpSilu(
            MlxArray input,
            MlxArray stackedGate,
            MlxArray stackedUp,
            MlxArray expertIndices,
            int K,
            int inDim,
            int outDim)
        {
            if (!input.IsValid || !stackedGate.IsValid || !stackedUp.IsValid || !expertIndices.IsValid)
                throw new ArgumentException("IQ2_XXS MoE fused gate+up+silu requires valid arrays.");
            if (K <= 0 || inDim <= 0 || outDim <= 0 || inDim % 256 != 0)
                throw new ArgumentOutOfRangeException(nameof(inDim),
                    "IQ2_XXS MoE fused gate+up+silu requires positive dimensions and input dim aligned to 256.");

            return MlxWorker.Shared.Invoke(() =>
            {
                MlxFastMetalKernel kernel = EnsureIq2XxsMoeMatmulBatchedFusedGateUpSiluKernel();
                MlxFastMetalKernelConfig config = mlx_fast_metal_kernel_config_new();
                MlxVectorArray inputs = default;
                MlxVectorArray outputs = default;
                try
                {
                    AddTemplateInt(config, "InDim", inDim);
                    AddTemplateInt(config, "OutDim", outDim);
                    AddTemplateInt(config, "BlocksPerRow", inDim / 256);
                    int[] shape = { K, outDim };
                    Check(mlx_fast_metal_kernel_config_add_output_arg(config, shape, (nuint)shape.Length, ToMlxDtype(DType.Float32)),
                        "configuring IQ2_XXS MoE fused gate+up+silu output");
                    Check(mlx_fast_metal_kernel_config_set_grid(config, 256, outDim, K),
                        "configuring IQ2_XXS MoE fused gate+up+silu grid");
                    Check(mlx_fast_metal_kernel_config_set_thread_group(config, 256, 1, 1),
                        "configuring IQ2_XXS MoE fused gate+up+silu threadgroup");

                    inputs = CreateVectorArray(input, stackedGate, stackedUp, expertIndices);
                    outputs = mlx_vector_array_new();
                    Check(mlx_fast_metal_kernel_apply(ref outputs, kernel, inputs, config, DefaultStream()),
                        "running IQ2_XXS MoE fused gate+up+silu");
                    if (mlx_vector_array_size(outputs) < 1)
                        throw new InvalidOperationException("IQ2_XXS MoE fused gate+up+silu produced no output.");

                    Check(mlx_vector_array_get(out MlxArray result, outputs, 0),
                        "reading IQ2_XXS MoE fused gate+up+silu output");
                    return result;
                }
                finally
                {
                    if (inputs.IsValid) _ = mlx_vector_array_free(inputs);
                    if (outputs.IsValid) _ = mlx_vector_array_free(outputs);
                    if (config.IsValid) _ = mlx_fast_metal_kernel_config_free(config);
                }
            });
        }

        // Batched IQ2_XXS MoE matmul (per-row input).
        // input: [K, inDim] - one row per active expert
        // stackedWeight: stacked uint8 IQ2_XXS bytes for all experts
        // expertIndices: [K] int32 - row k uses expert expertIndices[k]'s weights
        // Output: [K, outDim]
        internal static MlxArray Iq2XxsMoeMatmulBatchedRowed(
            MlxArray input,
            MlxArray stackedWeight,
            MlxArray expertIndices,
            int K,
            int inDim,
            int outDim)
        {
            if (!input.IsValid || !stackedWeight.IsValid || !expertIndices.IsValid)
                throw new ArgumentException("IQ2_XXS MoE batched-rowed matmul requires valid input, weight, and expertIndices arrays.");
            if (K <= 0 || inDim <= 0 || outDim <= 0 || inDim % 256 != 0)
                throw new ArgumentOutOfRangeException(nameof(inDim),
                    "IQ2_XXS MoE batched-rowed matmul requires positive dimensions and input dim aligned to 256.");

            return MlxWorker.Shared.Invoke(() =>
            {
                MlxFastMetalKernel kernel = EnsureIq2XxsMoeMatmulBatchedRowedKernel();
                MlxFastMetalKernelConfig config = mlx_fast_metal_kernel_config_new();
                MlxVectorArray inputs = default;
                MlxVectorArray outputs = default;
                try
                {
                    AddTemplateInt(config, "InDim", inDim);
                    AddTemplateInt(config, "OutDim", outDim);
                    AddTemplateInt(config, "BlocksPerRow", inDim / 256);
                    int[] shape = { K, outDim };
                    Check(mlx_fast_metal_kernel_config_add_output_arg(config, shape, (nuint)shape.Length, ToMlxDtype(DType.Float32)),
                        "configuring IQ2_XXS MoE batched-rowed matmul output");
                    Check(mlx_fast_metal_kernel_config_set_grid(config, 256, outDim, K),
                        "configuring IQ2_XXS MoE batched-rowed matmul grid");
                    Check(mlx_fast_metal_kernel_config_set_thread_group(config, 256, 1, 1),
                        "configuring IQ2_XXS MoE batched-rowed matmul threadgroup");

                    inputs = CreateVectorArray(input, stackedWeight, expertIndices);
                    outputs = mlx_vector_array_new();
                    Check(mlx_fast_metal_kernel_apply(ref outputs, kernel, inputs, config, DefaultStream()),
                        "running IQ2_XXS MoE batched-rowed matmul");
                    if (mlx_vector_array_size(outputs) < 1)
                        throw new InvalidOperationException("IQ2_XXS MoE batched-rowed matmul produced no output.");

                    Check(mlx_vector_array_get(out MlxArray result, outputs, 0),
                        "reading IQ2_XXS MoE batched-rowed matmul output");
                    return result;
                }
                finally
                {
                    if (inputs.IsValid) _ = mlx_vector_array_free(inputs);
                    if (outputs.IsValid) _ = mlx_vector_array_free(outputs);
                    if (config.IsValid) _ = mlx_fast_metal_kernel_config_free(config);
                }
            });
        }

        internal static MlxArray Iq2XxsGetRows(MlxArray rawWeight, MlxArray indices, int rows, int inDim)
        {
            if (!rawWeight.IsValid || !indices.IsValid)
                throw new ArgumentException("IQ2_XXS get_rows requires valid raw weight and index arrays.");
            if (rows <= 0 || inDim <= 0 || inDim % 256 != 0)
                throw new ArgumentOutOfRangeException(nameof(inDim), "IQ2_XXS get_rows requires positive dimensions and input dim aligned to 256.");

            return MlxWorker.Shared.Invoke(() =>
            {
                MlxFastMetalKernel kernel = EnsureIq2XxsGetRowsKernel();
                MlxFastMetalKernelConfig config = mlx_fast_metal_kernel_config_new();
                MlxVectorArray inputs = default;
                MlxVectorArray outputs = default;
                try
                {
                    AddTemplateInt(config, "InDim", inDim);
                    AddTemplateInt(config, "BlocksPerRow", inDim / 256);
                    int[] shape = { rows, inDim };
                    Check(mlx_fast_metal_kernel_config_add_output_arg(config, shape, (nuint)shape.Length, ToMlxDtype(DType.Float32)), "configuring IQ2_XXS get_rows output");
                    Check(mlx_fast_metal_kernel_config_set_grid(config, inDim, rows, 1), "configuring IQ2_XXS get_rows grid");
                    Check(mlx_fast_metal_kernel_config_set_thread_group(config, 256, 1, 1), "configuring IQ2_XXS get_rows threadgroup");

                    inputs = CreateVectorArray(rawWeight, indices);
                    outputs = mlx_vector_array_new();
                    Check(mlx_fast_metal_kernel_apply(ref outputs, kernel, inputs, config, DefaultStream()), "running IQ2_XXS get_rows");
                    if (mlx_vector_array_size(outputs) < 1)
                        throw new InvalidOperationException("IQ2_XXS get_rows produced no output.");

                    Check(mlx_vector_array_get(out MlxArray result, outputs, 0), "reading IQ2_XXS get_rows output");
                    return result;
                }
                finally
                {
                    if (inputs.IsValid)
                        _ = mlx_vector_array_free(inputs);
                    if (outputs.IsValid)
                        _ = mlx_vector_array_free(outputs);
                    if (config.IsValid)
                        _ = mlx_fast_metal_kernel_config_free(config);
                }
            });
        }

        internal static MlxArray Iq2SMatmul(MlxArray input, MlxArray rawWeight, int rows, int inDim, int outDim)
        {
            if (rows >= Iq2XxsMatmulSimdgroupMinRows && !Iq2XxsMatmulSimdgroupDisabled)
            {
                try { return Iq2SMatmulSimdgroup(input, rawWeight, rows, inDim, outDim); }
                catch (NotSupportedException) { }
            }
            return IQuantMatmul(input, rawWeight, rows, inDim, outDim, EnsureIq2SMatmulKernel, "IQ2_S");
        }

        internal static MlxArray Iq2SMatmulSimdgroup(MlxArray input, MlxArray rawWeight, int rows, int inDim, int outDim)
        {
            return IQuantMatmulSimdgroup(input, rawWeight, rows, inDim, outDim, EnsureIq2SMatmulSimdgroupKernel, "IQ2_S");
        }

        internal static MlxArray Iq2SGetRows(MlxArray rawWeight, MlxArray indices, int rows, int inDim)
        {
            return IQuantGetRows(rawWeight, indices, rows, inDim, EnsureIq2SGetRowsKernel, "IQ2_S");
        }

        internal static MlxArray Iq3SMatmul(MlxArray input, MlxArray rawWeight, int rows, int inDim, int outDim)
        {
            if (rows >= Iq2XxsMatmulSimdgroupMinRows && !Iq2XxsMatmulSimdgroupDisabled)
            {
                try { return Iq3SMatmulSimdgroup(input, rawWeight, rows, inDim, outDim); }
                catch (NotSupportedException) { }
            }
            return IQuantMatmul(input, rawWeight, rows, inDim, outDim, EnsureIq3SMatmulKernel, "IQ3_S");
        }

        internal static MlxArray Iq3SMatmulSimdgroup(MlxArray input, MlxArray rawWeight, int rows, int inDim, int outDim)
        {
            return IQuantMatmulSimdgroup(input, rawWeight, rows, inDim, outDim, EnsureIq3SMatmulSimdgroupKernel, "IQ3_S");
        }

        internal static MlxArray Iq3SGetRows(MlxArray rawWeight, MlxArray indices, int rows, int inDim)
        {
            return IQuantGetRows(rawWeight, indices, rows, inDim, EnsureIq3SGetRowsKernel, "IQ3_S");
        }

        internal static MlxArray Iq3XxsMatmul(MlxArray input, MlxArray rawWeight, int rows, int inDim, int outDim)
        {
            if (rows >= Iq2XxsMatmulSimdgroupMinRows && !Iq2XxsMatmulSimdgroupDisabled)
            {
                try { return Iq3XxsMatmulSimdgroup(input, rawWeight, rows, inDim, outDim); }
                catch (NotSupportedException) { }
            }
            return IQuantMatmul(input, rawWeight, rows, inDim, outDim, EnsureIq3XxsMatmulKernel, "IQ3_XXS");
        }

        internal static MlxArray Iq3XxsMatmulSimdgroup(MlxArray input, MlxArray rawWeight, int rows, int inDim, int outDim)
        {
            return IQuantMatmulSimdgroup(input, rawWeight, rows, inDim, outDim, EnsureIq3XxsMatmulSimdgroupKernel, "IQ3_XXS");
        }

        internal static MlxArray Iq3XxsGetRows(MlxArray rawWeight, MlxArray indices, int rows, int inDim)
        {
            return IQuantGetRows(rawWeight, indices, rows, inDim, EnsureIq3XxsGetRowsKernel, "IQ3_XXS");
        }

        private static MlxArray IQuantMatmul(
            MlxArray input,
            MlxArray rawWeight,
            int rows,
            int inDim,
            int outDim,
            Func<MlxFastMetalKernel> ensureKernel,
            string label)
        {
            if (!input.IsValid || !rawWeight.IsValid)
                throw new ArgumentException($"{label} matmul requires valid input and raw weight arrays.");
            if (rows <= 0 || inDim <= 0 || outDim <= 0 || inDim % 256 != 0)
                throw new ArgumentOutOfRangeException(nameof(inDim), $"{label} matmul requires positive dimensions and input dim aligned to 256.");

            return MlxWorker.Shared.Invoke(() =>
            {
                MlxFastMetalKernel kernel = ensureKernel();
                MlxFastMetalKernelConfig config = mlx_fast_metal_kernel_config_new();
                MlxVectorArray inputs = default;
                MlxVectorArray outputs = default;
                try
                {
                    AddTemplateInt(config, "InDim", inDim);
                    AddTemplateInt(config, "OutDim", outDim);
                    AddTemplateInt(config, "BlocksPerRow", inDim / 256);
                    int[] shape = { rows, outDim };
                    Check(mlx_fast_metal_kernel_config_add_output_arg(config, shape, (nuint)shape.Length, ToMlxDtype(DType.Float32)), $"configuring {label} matmul output");
                    Check(mlx_fast_metal_kernel_config_set_grid(config, 256, outDim, rows), $"configuring {label} matmul grid");
                    Check(mlx_fast_metal_kernel_config_set_thread_group(config, 256, 1, 1), $"configuring {label} matmul threadgroup");

                    inputs = CreateVectorArray(input, rawWeight);
                    outputs = mlx_vector_array_new();
                    Check(mlx_fast_metal_kernel_apply(ref outputs, kernel, inputs, config, DefaultStream()), $"running {label} matmul");
                    if (mlx_vector_array_size(outputs) < 1)
                        throw new InvalidOperationException($"{label} matmul produced no output.");

                    Check(mlx_vector_array_get(out MlxArray result, outputs, 0), $"reading {label} matmul output");
                    return result;
                }
                finally
                {
                    if (inputs.IsValid)
                        _ = mlx_vector_array_free(inputs);
                    if (outputs.IsValid)
                        _ = mlx_vector_array_free(outputs);
                    if (config.IsValid)
                        _ = mlx_fast_metal_kernel_config_free(config);
                }
            });
        }

        private static MlxArray IQuantGetRows(
            MlxArray rawWeight,
            MlxArray indices,
            int rows,
            int inDim,
            Func<MlxFastMetalKernel> ensureKernel,
            string label)
        {
            if (!rawWeight.IsValid || !indices.IsValid)
                throw new ArgumentException($"{label} get_rows requires valid raw weight and index arrays.");
            if (rows <= 0 || inDim <= 0 || inDim % 256 != 0)
                throw new ArgumentOutOfRangeException(nameof(inDim), $"{label} get_rows requires positive dimensions and input dim aligned to 256.");

            return MlxWorker.Shared.Invoke(() =>
            {
                MlxFastMetalKernel kernel = ensureKernel();
                MlxFastMetalKernelConfig config = mlx_fast_metal_kernel_config_new();
                MlxVectorArray inputs = default;
                MlxVectorArray outputs = default;
                try
                {
                    AddTemplateInt(config, "InDim", inDim);
                    AddTemplateInt(config, "BlocksPerRow", inDim / 256);
                    int[] shape = { rows, inDim };
                    Check(mlx_fast_metal_kernel_config_add_output_arg(config, shape, (nuint)shape.Length, ToMlxDtype(DType.Float32)), $"configuring {label} get_rows output");
                    Check(mlx_fast_metal_kernel_config_set_grid(config, inDim, rows, 1), $"configuring {label} get_rows grid");
                    Check(mlx_fast_metal_kernel_config_set_thread_group(config, 256, 1, 1), $"configuring {label} get_rows threadgroup");

                    inputs = CreateVectorArray(rawWeight, indices);
                    outputs = mlx_vector_array_new();
                    Check(mlx_fast_metal_kernel_apply(ref outputs, kernel, inputs, config, DefaultStream()), $"running {label} get_rows");
                    if (mlx_vector_array_size(outputs) < 1)
                        throw new InvalidOperationException($"{label} get_rows produced no output.");

                    Check(mlx_vector_array_get(out MlxArray result, outputs, 0), $"reading {label} get_rows output");
                    return result;
                }
                finally
                {
                    if (inputs.IsValid)
                        _ = mlx_vector_array_free(inputs);
                    if (outputs.IsValid)
                        _ = mlx_vector_array_free(outputs);
                    if (config.IsValid)
                        _ = mlx_fast_metal_kernel_config_free(config);
                }
            });
        }

        internal static MlxArray Q4KMatmul(MlxArray input, MlxArray rawWeight, int rows, int inDim, int outDim)
        {
            if (rows >= Iq2XxsMatmulSimdgroupMinRows && !Iq2XxsMatmulSimdgroupDisabled)
            {
                try { return Q4KMatmulSimdgroup(input, rawWeight, rows, inDim, outDim); }
                catch (NotSupportedException) { }
            }
            return KQuantMatmul(input, rawWeight, rows, inDim, outDim, EnsureQ4KMatmulKernel, "Q4_K");
        }

        internal static MlxArray Q4KMatmulSimdgroup(MlxArray input, MlxArray rawWeight, int rows, int inDim, int outDim)
        {
            return IQuantMatmulSimdgroup(input, rawWeight, rows, inDim, outDim, EnsureQ4KMatmulSimdgroupKernel, "Q4_K");
        }

        internal static MlxArray Q5KMatmul(MlxArray input, MlxArray rawWeight, int rows, int inDim, int outDim)
        {
            if (rows == 1 && !string.Equals(Environment.GetEnvironmentVariable("TS_MLX_Q5K_MATMUL4"), "0", StringComparison.Ordinal))
            {
                try
                {
                    return Q5KMatmul4Cols(input, rawWeight, inDim, outDim);
                }
                catch (Exception)
                {
                }
            }

            if (rows >= Iq2XxsMatmulSimdgroupMinRows && !Iq2XxsMatmulSimdgroupDisabled)
            {
                try { return Q5KMatmulSimdgroup(input, rawWeight, rows, inDim, outDim); }
                catch (NotSupportedException) { }
            }
            return KQuantMatmul(input, rawWeight, rows, inDim, outDim, EnsureQ5KMatmulKernel, "Q5_K");
        }

        internal static MlxArray Q5KMatmulSimdgroup(MlxArray input, MlxArray rawWeight, int rows, int inDim, int outDim)
        {
            return IQuantMatmulSimdgroup(input, rawWeight, rows, inDim, outDim, EnsureQ5KMatmulSimdgroupKernel, "Q5_K");
        }

        // Rows up to which Q6_K runs as matrix-vector products (the ggml port), one
        // pass over the weight per row. TS_MLX_Q6K_MATVEC_MAX_ROWS=0 restores the
        // older kernels. Settable so tests can reach those kernels too.
        internal static int Q6KMatvecMaxRows = ResolveMatvecMaxRows("TS_MLX_Q6K_MATVEC_MAX_ROWS");

        private static int ResolveMatvecMaxRows(string variable)
        {
            string value = Environment.GetEnvironmentVariable(variable);
            return int.TryParse(value, out int rows) && rows >= 0 ? rows : 4;
        }

        // Above Q6KMatvecMaxRows (and Iq4XsMatvecMaxRows), the raw weight is dequantized
        // to F16 in slices of at most this many bytes and multiplied by MLX's GEMM.
        // Against an 8-bit affine quantized_matmul (M5 Pro, MLX 0.32.2): +11-26% per
        // matmul at 512 rows, 7% faster at 2048. TS_MLX_Q6K_DEQUANT_GEMM=0 keeps the
        // older Q6_K kernels. Settable so a test can force several slices.
        internal static long DequantSliceBytes = 256L * 1024 * 1024;
        private static readonly bool Q6KDequantGemm =
            !string.Equals(Environment.GetEnvironmentVariable("TS_MLX_Q6K_DEQUANT_GEMM"), "0", StringComparison.Ordinal);

        internal static MlxArray Q6KMatmul(MlxArray input, MlxArray rawWeight, int rows, int inDim, int outDim)
        {
            if (rows >= 1 && rows <= Q6KMatvecMaxRows)
            {
                try
                {
                    return Q6KMatvec(input, rawWeight, rows, inDim, outDim);
                }
                catch (NotSupportedException)
                {
                }
            }
            else if (rows > Q6KMatvecMaxRows && Q6KDequantGemm)
            {
                try
                {
                    return Q6KDequantMatmul(input, rawWeight, rows, inDim, outDim);
                }
                catch (NotSupportedException)
                {
                }
            }

            if (rows == 1 && !string.Equals(Environment.GetEnvironmentVariable("TS_MLX_Q6K_MATMUL4"), "0", StringComparison.Ordinal))
            {
                try
                {
                    return Q6KMatmul4Cols(input, rawWeight, inDim, outDim);
                }
                catch (Exception)
                {
                }
            }

            if (rows >= Iq2XxsMatmulSimdgroupMinRows && !Iq2XxsMatmulSimdgroupDisabled)
            {
                try { return Q6KMatmulSimdgroup(input, rawWeight, rows, inDim, outDim); }
                catch (NotSupportedException) { }
            }
            return KQuantMatmul(input, rawWeight, rows, inDim, outDim, EnsureQ6KMatmulKernel, "Q6_K");
        }

        internal static MlxArray Q6KMatmulSimdgroup(MlxArray input, MlxArray rawWeight, int rows, int inDim, int outDim)
        {
            return IQuantMatmulSimdgroup(input, rawWeight, rows, inDim, outDim, EnsureQ6KMatmulSimdgroupKernel, "Q6_K");
        }

        internal static MlxArray Iq4XsMatmulSimdgroup(MlxArray input, MlxArray rawWeight, int rows, int inDim, int outDim)
        {
            return IQuantMatmulSimdgroup(input, rawWeight, rows, inDim, outDim, EnsureIq4XsMatmulSimdgroupKernel, "IQ4_XS");
        }

        private static MlxArray Q5KMatmul4Cols(MlxArray input, MlxArray rawWeight, int inDim, int outDim)
        {
            if (!input.IsValid || !rawWeight.IsValid)
                throw new ArgumentException("Q5_K 4-column matmul requires valid input and raw weight arrays.");
            if (inDim <= 0 || outDim <= 0 || inDim % 256 != 0)
                throw new ArgumentOutOfRangeException(nameof(inDim), "Q5_K 4-column matmul requires positive dimensions and input dim aligned to 256.");

            return MlxWorker.Shared.Invoke(() =>
            {
                MlxFastMetalKernel kernel = EnsureQ5KMatmul4Kernel();
                MlxFastMetalKernelConfig config = mlx_fast_metal_kernel_config_new();
                MlxVectorArray inputs = default;
                MlxVectorArray outputs = default;
                try
                {
                    AddTemplateInt(config, "InDim", inDim);
                    AddTemplateInt(config, "OutDim", outDim);
                    AddTemplateInt(config, "BlocksPerRow", inDim / 256);
                    int[] shape = { 1, outDim };
                    Check(mlx_fast_metal_kernel_config_add_output_arg(config, shape, (nuint)shape.Length, ToMlxDtype(DType.Float32)), "configuring Q5_K 4-column matmul output");
                    Check(mlx_fast_metal_kernel_config_set_grid(config, 256, (outDim + 3) / 4, 1), "configuring Q5_K 4-column matmul grid");
                    Check(mlx_fast_metal_kernel_config_set_thread_group(config, 256, 1, 1), "configuring Q5_K 4-column matmul threadgroup");

                    inputs = CreateVectorArray(input, rawWeight);
                    outputs = mlx_vector_array_new();
                    Check(mlx_fast_metal_kernel_apply(ref outputs, kernel, inputs, config, DefaultStream()), "running Q5_K 4-column matmul");
                    if (mlx_vector_array_size(outputs) < 1)
                        throw new InvalidOperationException("Q5_K 4-column matmul produced no output.");

                    Check(mlx_vector_array_get(out MlxArray result, outputs, 0), "reading Q5_K 4-column matmul output");
                    return result;
                }
                finally
                {
                    if (inputs.IsValid)
                        _ = mlx_vector_array_free(inputs);
                    if (outputs.IsValid)
                        _ = mlx_vector_array_free(outputs);
                    if (config.IsValid)
                        _ = mlx_fast_metal_kernel_config_free(config);
                }
            });
        }

        /// <summary>
        /// [rows, inDim] x Q6_K [outDim, inDim]^T through an F16 copy of the weight, a
        /// slice of output rows at a time so the copy stays bounded (a 248k-row output
        /// head would otherwise be a 2.5 GB temporary).
        /// </summary>
        /// <param name="halfOutput">Return the F16 product instead of casting it to F32,
        /// for a caller that keeps its activations in half precision.</param>
        internal static MlxArray Q6KDequantMatmul(MlxArray input, MlxArray rawWeight, int rows, int inDim, int outDim, bool halfOutput = false)
            => DequantF16Matmul(input, rawWeight, rows, inDim, outDim, EnsureQ6KDequantF16Kernel, inDim / 256 * 64, "Q6_K", halfOutput);

        internal static MlxArray Iq4XsDequantMatmul(MlxArray input, MlxArray rawWeight, int rows, int inDim, int outDim, bool halfOutput = false)
            => DequantF16Matmul(input, rawWeight, rows, inDim, outDim, EnsureIq4XsDequantF16Kernel, inDim / 256 * 128, "IQ4_XS", halfOutput);

        private static MlxArray DequantF16Matmul(
            MlxArray input, MlxArray rawWeight, int rows, int inDim, int outDim,
            Func<MlxFastMetalKernel> ensureKernel, int threadsPerRow, string label, bool halfOutput)
        {
            if (!input.IsValid || !rawWeight.IsValid)
                throw new ArgumentException($"{label} dequantized matmul requires valid input and raw weight arrays.");
            if (rows <= 0 || inDim <= 0 || outDim <= 0 || inDim % 256 != 0)
                throw new ArgumentOutOfRangeException(nameof(inDim), $"{label} dequantized matmul requires positive dimensions and input dim aligned to 256.");

            int sliceRows = (int)Math.Clamp(DequantSliceBytes / (2L * inDim), 1, outDim);
            return MlxWorker.Shared.Invoke(() =>
            {
                MlxFastMetalKernel kernel = ensureKernel();
                MlxArray inputHalf = default;
                MlxArray inDimArray = default;
                var parts = new List<MlxArray>();
                MlxVectorArray partVector = default;
                MlxArray product = default;
                try
                {
                    Check(mlx_astype(out inputHalf, input, ToMlxDtype(DType.Float16), DefaultStream()), $"casting {label} matmul input");
                    inDimArray = mlx_array_new_int(inDim);
                    for (int row0 = 0; row0 < outDim; row0 += sliceRows)
                    {
                        int count = Math.Min(sliceRows, outDim - row0);
                        MlxArray weightHalf = default;
                        MlxArray weightT = default;
                        MlxArray part = default;
                        MlxFastMetalKernelConfig config = mlx_fast_metal_kernel_config_new();
                        MlxVectorArray inputs = default;
                        MlxVectorArray outputs = default;
                        MlxArray row0Array = default;
                        try
                        {
                            int[] shape = { count, inDim };
                            Check(mlx_fast_metal_kernel_config_add_output_arg(config, shape, (nuint)shape.Length, ToMlxDtype(DType.Float16)), $"configuring {label} dequantization output");
                            Check(mlx_fast_metal_kernel_config_set_grid(config, threadsPerRow, count, 1), $"configuring {label} dequantization grid");
                            Check(mlx_fast_metal_kernel_config_set_thread_group(config, 256, 1, 1), $"configuring {label} dequantization threadgroup");
                            row0Array = mlx_array_new_int(row0);
                            inputs = CreateVectorArray(rawWeight, inDimArray, row0Array);
                            outputs = mlx_vector_array_new();
                            Check(mlx_fast_metal_kernel_apply(ref outputs, kernel, inputs, config, DefaultStream()), $"running {label} dequantization");
                            Check(mlx_vector_array_get(out weightHalf, outputs, 0), $"reading {label} dequantization output");
                            Check(mlx_transpose_axes(out weightT, weightHalf, new[] { 1, 0 }, 2, DefaultStream()), $"transposing {label} weight");
                            Check(mlx_matmul(out part, inputHalf, weightT, DefaultStream()), $"running {label} dequantized matmul");
                            parts.Add(part);
                            part = default;
                        }
                        finally
                        {
                            if (part.IsValid) _ = mlx_array_free(part);
                            if (weightT.IsValid) _ = mlx_array_free(weightT);
                            if (weightHalf.IsValid) _ = mlx_array_free(weightHalf);
                            if (row0Array.IsValid) _ = mlx_array_free(row0Array);
                            if (inputs.IsValid) _ = mlx_vector_array_free(inputs);
                            if (outputs.IsValid) _ = mlx_vector_array_free(outputs);
                            if (config.IsValid) _ = mlx_fast_metal_kernel_config_free(config);
                        }
                    }

                    if (parts.Count == 1)
                    {
                        product = parts[0];
                        parts.Clear();
                    }
                    else
                    {
                        partVector = CreateVectorArray(parts.ToArray());
                        Check(mlx_concatenate_axis(out product, partVector, 1, DefaultStream()), $"joining {label} matmul slices");
                    }

                    if (halfOutput)
                    {
                        MlxArray halfResult = product;
                        product = default;
                        return halfResult;
                    }

                    Check(mlx_astype(out MlxArray result, product, ToMlxDtype(DType.Float32), DefaultStream()), $"casting {label} matmul output");
                    return result;
                }
                finally
                {
                    foreach (MlxArray part in parts)
                        _ = mlx_array_free(part);
                    if (partVector.IsValid) _ = mlx_vector_array_free(partVector);
                    if (product.IsValid) _ = mlx_array_free(product);
                    if (inputHalf.IsValid) _ = mlx_array_free(inputHalf);
                    if (inDimArray.IsValid) _ = mlx_array_free(inDimArray);
                }
            });
        }

        internal static MlxArray Q6KMatvec(MlxArray input, MlxArray rawWeight, int rows, int inDim, int outDim)
            => GgmlPortMatvec(input, rawWeight, rows, inDim, outDim, EnsureQ6KMatvecKernel, "Q6_K");

        internal static MlxArray Iq4XsMatvec(MlxArray input, MlxArray rawWeight, int rows, int inDim, int outDim)
            => GgmlPortMatvec(input, rawWeight, rows, inDim, outDim, EnsureIq4XsMatvecKernel, "IQ4_XS");

        /// <summary>
        /// Launches one of the ggml-metal matrix-vector ports (<see cref="Q6KMatvecSource"/>,
        /// <see cref="Iq4XsMatvecSource"/>): 64-thread threadgroups of two simdgroups, four
        /// output rows per threadgroup (grid.x) and one input row per grid.y, over F32 rows.
        /// </summary>
        private static MlxArray GgmlPortMatvec(
            MlxArray input, MlxArray rawWeight, int rows, int inDim, int outDim,
            Func<MlxFastMetalKernel> ensureKernel, string label)
        {
            if (!input.IsValid || !rawWeight.IsValid)
                throw new ArgumentException($"{label} matrix-vector product requires valid input and raw weight arrays.");
            if (rows <= 0 || inDim <= 0 || outDim <= 0 || inDim % 256 != 0)
                throw new ArgumentOutOfRangeException(nameof(inDim), $"{label} matrix-vector product requires positive dimensions and input dim aligned to 256.");

            return MlxWorker.Shared.Invoke(() =>
            {
                MlxFastMetalKernel kernel = ensureKernel();
                MlxFastMetalKernelConfig config = mlx_fast_metal_kernel_config_new();
                MlxVectorArray inputs = default;
                MlxVectorArray outputs = default;
                MlxArray inputF32 = default;
                MlxArray inDimArray = default;
                MlxArray outDimArray = default;
                try
                {
                    int[] shape = { rows, outDim };
                    Check(mlx_fast_metal_kernel_config_add_output_arg(config, shape, (nuint)shape.Length, ToMlxDtype(DType.Float32)), $"configuring {label} matrix-vector output");
                    Check(mlx_fast_metal_kernel_config_set_grid(config, 64 * ((outDim + 3) / 4), rows, 1), $"configuring {label} matrix-vector grid");
                    Check(mlx_fast_metal_kernel_config_set_thread_group(config, 64, 1, 1), $"configuring {label} matrix-vector threadgroup");

                    // The kernels read F32 rows (as float4s in the IQ4_XS port); anything
                    // else is cast, and a strided view made contiguous, first.
                    Check(mlx_astype(out inputF32, input, ToMlxDtype(DType.Float32), DefaultStream()), $"casting {label} matrix-vector input");
                    MlxArray contiguous = default;
                    Check(mlx_contiguous(out contiguous, inputF32, false, DefaultStream()), $"laying out {label} matrix-vector input");
                    _ = mlx_array_free(inputF32);
                    inputF32 = contiguous;
                    inDimArray = mlx_array_new_int(inDim);
                    outDimArray = mlx_array_new_int(outDim);
                    inputs = CreateVectorArray(inputF32, rawWeight, inDimArray, outDimArray);
                    outputs = mlx_vector_array_new();
                    Check(mlx_fast_metal_kernel_apply(ref outputs, kernel, inputs, config, DefaultStream()), $"running {label} matrix-vector product");
                    if (mlx_vector_array_size(outputs) < 1)
                        throw new InvalidOperationException($"{label} matrix-vector product produced no output.");

                    Check(mlx_vector_array_get(out MlxArray result, outputs, 0), $"reading {label} matrix-vector output");
                    return result;
                }
                finally
                {
                    if (inputF32.IsValid)
                        _ = mlx_array_free(inputF32);
                    if (inDimArray.IsValid)
                        _ = mlx_array_free(inDimArray);
                    if (outDimArray.IsValid)
                        _ = mlx_array_free(outDimArray);
                    if (inputs.IsValid)
                        _ = mlx_vector_array_free(inputs);
                    if (outputs.IsValid)
                        _ = mlx_vector_array_free(outputs);
                    if (config.IsValid)
                        _ = mlx_fast_metal_kernel_config_free(config);
                }
            });
        }

        private static MlxArray Q6KMatmul4Cols(MlxArray input, MlxArray rawWeight, int inDim, int outDim)
        {
            if (!input.IsValid || !rawWeight.IsValid)
                throw new ArgumentException("Q6_K 4-column matmul requires valid input and raw weight arrays.");
            if (inDim <= 0 || outDim <= 0 || inDim % 256 != 0)
                throw new ArgumentOutOfRangeException(nameof(inDim), "Q6_K 4-column matmul requires positive dimensions and input dim aligned to 256.");

            return MlxWorker.Shared.Invoke(() =>
            {
                MlxFastMetalKernel kernel = EnsureQ6KMatmul4Kernel();
                MlxFastMetalKernelConfig config = mlx_fast_metal_kernel_config_new();
                MlxVectorArray inputs = default;
                MlxVectorArray outputs = default;
                try
                {
                    AddTemplateInt(config, "InDim", inDim);
                    AddTemplateInt(config, "OutDim", outDim);
                    AddTemplateInt(config, "BlocksPerRow", inDim / 256);
                    int[] shape = { 1, outDim };
                    Check(mlx_fast_metal_kernel_config_add_output_arg(config, shape, (nuint)shape.Length, ToMlxDtype(DType.Float32)), "configuring Q6_K 4-column matmul output");
                    Check(mlx_fast_metal_kernel_config_set_grid(config, 256, (outDim + 3) / 4, 1), "configuring Q6_K 4-column matmul grid");
                    Check(mlx_fast_metal_kernel_config_set_thread_group(config, 256, 1, 1), "configuring Q6_K 4-column matmul threadgroup");

                    inputs = CreateVectorArray(input, rawWeight);
                    outputs = mlx_vector_array_new();
                    Check(mlx_fast_metal_kernel_apply(ref outputs, kernel, inputs, config, DefaultStream()), "running Q6_K 4-column matmul");
                    if (mlx_vector_array_size(outputs) < 1)
                        throw new InvalidOperationException("Q6_K 4-column matmul produced no output.");

                    Check(mlx_vector_array_get(out MlxArray result, outputs, 0), "reading Q6_K 4-column matmul output");
                    return result;
                }
                finally
                {
                    if (inputs.IsValid)
                        _ = mlx_vector_array_free(inputs);
                    if (outputs.IsValid)
                        _ = mlx_vector_array_free(outputs);
                    if (config.IsValid)
                        _ = mlx_fast_metal_kernel_config_free(config);
                }
            });
        }

        internal static MlxArray Q4KGetRows(MlxArray rawWeight, MlxArray indices, int rows, int inDim)
        {
            return KQuantGetRows(rawWeight, indices, rows, inDim, EnsureQ4KGetRowsKernel, "Q4_K");
        }

        internal static MlxArray Q5KGetRows(MlxArray rawWeight, MlxArray indices, int rows, int inDim)
        {
            return KQuantGetRows(rawWeight, indices, rows, inDim, EnsureQ5KGetRowsKernel, "Q5_K");
        }

        internal static MlxArray Q6KGetRows(MlxArray rawWeight, MlxArray indices, int rows, int inDim)
        {
            return KQuantGetRows(rawWeight, indices, rows, inDim, EnsureQ6KGetRowsKernel, "Q6_K");
        }

        internal static void GatedDelta(
            MlxArray q,
            MlxArray k,
            MlxArray v,
            MlxArray g,
            MlxArray beta,
            MlxArray state,
            int batch,
            int seqLen,
            int numKeyHeads,
            int numValueHeads,
            int keyDim,
            int valueDim,
            out MlxArray y,
            out MlxArray nextState)
        {
            y = default;
            nextState = default;
            if (!q.IsValid || !k.IsValid || !v.IsValid || !g.IsValid || !beta.IsValid || !state.IsValid)
                throw new ArgumentException("GatedDelta requires valid input arrays.");
            if (batch <= 0 || seqLen <= 0 || numKeyHeads <= 0 || numValueHeads <= 0 || keyDim <= 0 || valueDim <= 0)
                throw new ArgumentOutOfRangeException(nameof(seqLen));
            if (keyDim % 32 != 0 || numValueHeads % numKeyHeads != 0)
                throw new ArgumentException("GatedDelta requires keyDim divisible by 32 and value heads divisible by key heads.");

            // For the seqLen==1 decode case, route to the T=1-specialized
            // kernel which drops the outer time loop and hoists the per-
            // head g/beta and per-(head, dv) v scalars outside the dot-
            // product loops. Gated by TS_MLX_DISABLE_GDN_T1=1 for A/B.
            bool useT1Kernel = seqLen == 1
                && !string.Equals(Environment.GetEnvironmentVariable("TS_MLX_DISABLE_GDN_T1"), "1", StringComparison.Ordinal);
            // Prefill takes the blocked kernel wherever its fixed thread layout fits
            // (see GatedDeltaBlockedSource). TS_MLX_GDN_BLOCKED=0 restores the
            // per-row kernel for A/B.
            bool useBlockedKernel = seqLen > 1
                && keyDim == 128
                && valueDim % 32 == 0
                && GatedDeltaBlockedEnabled;

            MlxArray yResult = default;
            MlxArray stateResult = default;
            MlxWorker.Shared.Invoke(() =>
            {
                MlxFastMetalKernel kernel = useT1Kernel
                    ? EnsureGatedDeltaT1Kernel()
                    : useBlockedKernel
                        ? EnsureGatedDeltaBlockedKernel()
                        : EnsureGatedDeltaKernel();
                MlxFastMetalKernelConfig config = mlx_fast_metal_kernel_config_new();
                MlxVectorArray inputs = default;
                MlxVectorArray outputs = default;
                MlxArray length = default;
                try
                {
                    // T template is omitted from the T=1 kernel (specialization
                    // makes it implicit) and from the blocked kernel (a runtime
                    // input). The general kernel still needs T.
                    if (!useT1Kernel && !useBlockedKernel)
                        AddTemplateInt(config, "T", seqLen);
                    AddTemplateInt(config, "Dk", keyDim);
                    AddTemplateInt(config, "Dv", valueDim);
                    AddTemplateInt(config, "Hk", numKeyHeads);
                    AddTemplateInt(config, "Hv", numValueHeads);
                    int[] yShape = { batch, seqLen, numValueHeads, valueDim };
                    int[] stateShape = { batch, numValueHeads, valueDim, keyDim };
                    Check(mlx_fast_metal_kernel_config_add_output_arg(config, yShape, (nuint)yShape.Length, ToMlxDtype(DType.Float32)), "configuring gated-delta output");
                    Check(mlx_fast_metal_kernel_config_add_output_arg(config, stateShape, (nuint)stateShape.Length, ToMlxDtype(DType.Float32)), "configuring gated-delta state output");
                    if (useBlockedKernel)
                    {
                        Check(mlx_fast_metal_kernel_config_set_grid(config, 256 * (valueDim / 32), numValueHeads, batch), "configuring blocked gated-delta grid");
                        Check(mlx_fast_metal_kernel_config_set_thread_group(config, 256, 1, 1), "configuring blocked gated-delta threadgroup");
                        length = mlx_array_new_int(seqLen);
                        inputs = CreateVectorArray(q, k, v, g, beta, state, length);
                    }
                    else
                    {
                        Check(mlx_fast_metal_kernel_config_set_grid(config, 32, valueDim, batch * numValueHeads), "configuring gated-delta grid");
                        Check(mlx_fast_metal_kernel_config_set_thread_group(config, 32, Math.Min(valueDim, 4), 1), "configuring gated-delta threadgroup");
                        inputs = CreateVectorArray(q, k, v, g, beta, state);
                    }
                    outputs = mlx_vector_array_new();
                    Check(mlx_fast_metal_kernel_apply(ref outputs, kernel, inputs, config, DefaultStream()), "running gated-delta kernel");
                    if (mlx_vector_array_size(outputs) < 2)
                        throw new InvalidOperationException("GatedDelta kernel produced fewer than two outputs.");

                    Check(mlx_vector_array_get(out yResult, outputs, 0), "reading gated-delta output");
                    Check(mlx_vector_array_get(out stateResult, outputs, 1), "reading gated-delta next state");
                }
                finally
                {
                    FreeArray(length);
                    if (inputs.IsValid)
                        _ = mlx_vector_array_free(inputs);
                    if (outputs.IsValid)
                        _ = mlx_vector_array_free(outputs);
                    if (config.IsValid)
                        _ = mlx_fast_metal_kernel_config_free(config);
                }
            });
            y = yResult;
            nextState = stateResult;
        }

        internal static void Qwen35GdnPreprocess(
            MlxArray qkvRaw,
            MlxArray zRaw,
            MlxArray betaRaw,
            MlxArray alphaRaw,
            MlxArray convState,
            MlxArray convWeight,
            MlxArray dtBias,
            MlxArray aLog,
            int seqLen,
            int qkvDim,
            int keyDim,
            int valueDim,
            int numKeyHeads,
            int numValueHeads,
            int headKeyDim,
            int headValueDim,
            int convKernel,
            bool convWeightChannelMajor,
            out MlxArray q,
            out MlxArray k,
            out MlxArray v,
            out MlxArray g,
            out MlxArray beta,
            out MlxArray zSilu,
            out MlxArray nextConv)
        {
            q = default;
            k = default;
            v = default;
            g = default;
            beta = default;
            zSilu = default;
            nextConv = default;

            if (!qkvRaw.IsValid || !zRaw.IsValid || !betaRaw.IsValid || !alphaRaw.IsValid
                || !convState.IsValid || !convWeight.IsValid || !dtBias.IsValid || !aLog.IsValid)
                throw new ArgumentException("Qwen35 GDN preprocess requires valid input arrays.");
            if (seqLen <= 0 || qkvDim <= 0 || keyDim <= 0 || valueDim <= 0
                || numKeyHeads <= 0 || numValueHeads <= 0 || headKeyDim <= 0 || headValueDim <= 0
                || convKernel <= 1 || keyDim != numKeyHeads * headKeyDim || valueDim != numValueHeads * headValueDim)
                throw new ArgumentOutOfRangeException(nameof(seqLen));

            MlxArray qResult = default;
            MlxArray kResult = default;
            MlxArray vResult = default;
            MlxArray gResult = default;
            MlxArray betaResult = default;
            MlxArray zSiluResult = default;
            MlxArray nextConvResult = default;
            MlxWorker.Shared.Invoke(() =>
            {
                MlxFastMetalKernel kernel = EnsureQwen35GdnPreprocessKernel();
                MlxFastMetalKernelConfig config = mlx_fast_metal_kernel_config_new();
                MlxVectorArray inputs = default;
                MlxVectorArray outputs = default;
                try
                {
                    int tail = convKernel - 1;
                    AddTemplateInt(config, "T", seqLen);
                    AddTemplateInt(config, "Tail", tail);
                    AddTemplateInt(config, "Kernel", convKernel);
                    AddTemplateInt(config, "QkvDim", qkvDim);
                    AddTemplateInt(config, "KeyDim", keyDim);
                    AddTemplateInt(config, "ValueDim", valueDim);
                    AddTemplateInt(config, "NumKeyHeads", numKeyHeads);
                    AddTemplateInt(config, "NumValueHeads", numValueHeads);
                    AddTemplateInt(config, "HeadKeyDim", headKeyDim);
                    AddTemplateInt(config, "HeadValueDim", headValueDim);
                    AddTemplateInt(config, "ConvWeightChannelMajor", convWeightChannelMajor ? 1 : 0);

                    int[] qShape = { 1, seqLen, numKeyHeads, headKeyDim };
                    int[] kShape = { 1, seqLen, numKeyHeads, headKeyDim };
                    int[] vShape = { 1, seqLen, numValueHeads, headValueDim };
                    int[] gbShape = { 1, seqLen, numValueHeads };
                    int[] zShape = { 1, seqLen, numValueHeads, headValueDim };
                    int[] nextConvShape = { 1, tail, qkvDim };
                    Check(mlx_fast_metal_kernel_config_add_output_arg(config, qShape, (nuint)qShape.Length, ToMlxDtype(DType.Float32)), "configuring Qwen35 GDN q output");
                    Check(mlx_fast_metal_kernel_config_add_output_arg(config, kShape, (nuint)kShape.Length, ToMlxDtype(DType.Float32)), "configuring Qwen35 GDN k output");
                    Check(mlx_fast_metal_kernel_config_add_output_arg(config, vShape, (nuint)vShape.Length, ToMlxDtype(DType.Float32)), "configuring Qwen35 GDN v output");
                    Check(mlx_fast_metal_kernel_config_add_output_arg(config, gbShape, (nuint)gbShape.Length, ToMlxDtype(DType.Float32)), "configuring Qwen35 GDN g output");
                    Check(mlx_fast_metal_kernel_config_add_output_arg(config, gbShape, (nuint)gbShape.Length, ToMlxDtype(DType.Float32)), "configuring Qwen35 GDN beta output");
                    Check(mlx_fast_metal_kernel_config_add_output_arg(config, zShape, (nuint)zShape.Length, ToMlxDtype(DType.Float32)), "configuring Qwen35 GDN z output");
                    Check(mlx_fast_metal_kernel_config_add_output_arg(config, nextConvShape, (nuint)nextConvShape.Length, ToMlxDtype(DType.Float32)), "configuring Qwen35 GDN next conv output");

                    int maxX = Math.Max(qkvDim, valueDim);
                    int maxY = Math.Max(seqLen, tail);
                    Check(mlx_fast_metal_kernel_config_set_grid(config, maxX, maxY, 3), "configuring Qwen35 GDN preprocess grid");
                    Check(mlx_fast_metal_kernel_config_set_thread_group(config, 256, 1, 1), "configuring Qwen35 GDN preprocess threadgroup");

                    inputs = CreateVectorArray(qkvRaw, zRaw, betaRaw, alphaRaw, convState, convWeight, dtBias, aLog);
                    outputs = mlx_vector_array_new();
                    Check(mlx_fast_metal_kernel_apply(ref outputs, kernel, inputs, config, DefaultStream()), "running Qwen35 GDN preprocess kernel");
                    if (mlx_vector_array_size(outputs) < 7)
                        throw new InvalidOperationException("Qwen35 GDN preprocess kernel produced fewer than seven outputs.");

                    Check(mlx_vector_array_get(out qResult, outputs, 0), "reading Qwen35 GDN q output");
                    Check(mlx_vector_array_get(out kResult, outputs, 1), "reading Qwen35 GDN k output");
                    Check(mlx_vector_array_get(out vResult, outputs, 2), "reading Qwen35 GDN v output");
                    Check(mlx_vector_array_get(out gResult, outputs, 3), "reading Qwen35 GDN g output");
                    Check(mlx_vector_array_get(out betaResult, outputs, 4), "reading Qwen35 GDN beta output");
                    Check(mlx_vector_array_get(out zSiluResult, outputs, 5), "reading Qwen35 GDN z output");
                    Check(mlx_vector_array_get(out nextConvResult, outputs, 6), "reading Qwen35 GDN next conv output");
                }
                finally
                {
                    if (inputs.IsValid)
                        _ = mlx_vector_array_free(inputs);
                    if (outputs.IsValid)
                        _ = mlx_vector_array_free(outputs);
                    if (config.IsValid)
                        _ = mlx_fast_metal_kernel_config_free(config);
                }
            });

            q = qResult;
            k = kResult;
            v = vResult;
            g = gResult;
            beta = betaResult;
            zSilu = zSiluResult;
            nextConv = nextConvResult;
        }

        internal static void Qwen35GdnPreprocessPacked(
            MlxArray packedRaw,
            MlxArray convState,
            MlxArray convWeight,
            MlxArray dtBias,
            MlxArray aLog,
            int seqLen,
            int packedDim,
            int qkvDim,
            int keyDim,
            int valueDim,
            int numKeyHeads,
            int numValueHeads,
            int headKeyDim,
            int headValueDim,
            int convKernel,
            bool convWeightChannelMajor,
            out MlxArray q,
            out MlxArray k,
            out MlxArray v,
            out MlxArray g,
            out MlxArray beta,
            out MlxArray zSilu,
            out MlxArray nextConv)
        {
            q = default;
            k = default;
            v = default;
            g = default;
            beta = default;
            zSilu = default;
            nextConv = default;

            if (!packedRaw.IsValid || !convState.IsValid || !convWeight.IsValid || !dtBias.IsValid || !aLog.IsValid)
                throw new ArgumentException("Qwen35 packed GDN preprocess requires valid input arrays.");
            if (seqLen <= 0 || packedDim <= 0 || qkvDim <= 0 || keyDim <= 0 || valueDim <= 0
                || packedDim < qkvDim + valueDim + numValueHeads * 2
                || numKeyHeads <= 0 || numValueHeads <= 0 || headKeyDim <= 0 || headValueDim <= 0
                || convKernel <= 1 || keyDim != numKeyHeads * headKeyDim || valueDim != numValueHeads * headValueDim)
                throw new ArgumentOutOfRangeException(nameof(seqLen));

            MlxArray qResult = default;
            MlxArray kResult = default;
            MlxArray vResult = default;
            MlxArray gResult = default;
            MlxArray betaResult = default;
            MlxArray zSiluResult = default;
            MlxArray nextConvResult = default;
            MlxWorker.Shared.Invoke(() =>
            {
                MlxFastMetalKernel kernel = EnsureQwen35GdnPackedPreprocessKernel();
                MlxFastMetalKernelConfig config = mlx_fast_metal_kernel_config_new();
                MlxVectorArray inputs = default;
                MlxVectorArray outputs = default;
                MlxArray length = default;
                try
                {
                    int tail = convKernel - 1;
                    // The length is a runtime input rather than a template argument:
                    // every distinct template set compiles its own Metal library, so
                    // templating on T built a kernel for each new prompt length.
                    AddTemplateInt(config, "Tail", tail);
                    AddTemplateInt(config, "Kernel", convKernel);
                    AddTemplateInt(config, "PackedDim", packedDim);
                    AddTemplateInt(config, "QkvDim", qkvDim);
                    AddTemplateInt(config, "KeyDim", keyDim);
                    AddTemplateInt(config, "ValueDim", valueDim);
                    AddTemplateInt(config, "NumKeyHeads", numKeyHeads);
                    AddTemplateInt(config, "NumValueHeads", numValueHeads);
                    AddTemplateInt(config, "HeadKeyDim", headKeyDim);
                    AddTemplateInt(config, "HeadValueDim", headValueDim);
                    AddTemplateInt(config, "ConvWeightChannelMajor", convWeightChannelMajor ? 1 : 0);

                    int[] qShape = { 1, seqLen, numKeyHeads, headKeyDim };
                    int[] kShape = { 1, seqLen, numKeyHeads, headKeyDim };
                    int[] vShape = { 1, seqLen, numValueHeads, headValueDim };
                    int[] gbShape = { 1, seqLen, numValueHeads };
                    int[] zShape = { 1, seqLen, numValueHeads, headValueDim };
                    int[] nextConvShape = { 1, tail, qkvDim };
                    Check(mlx_fast_metal_kernel_config_add_output_arg(config, qShape, (nuint)qShape.Length, ToMlxDtype(DType.Float32)), "configuring packed Qwen35 GDN q output");
                    Check(mlx_fast_metal_kernel_config_add_output_arg(config, kShape, (nuint)kShape.Length, ToMlxDtype(DType.Float32)), "configuring packed Qwen35 GDN k output");
                    Check(mlx_fast_metal_kernel_config_add_output_arg(config, vShape, (nuint)vShape.Length, ToMlxDtype(DType.Float32)), "configuring packed Qwen35 GDN v output");
                    Check(mlx_fast_metal_kernel_config_add_output_arg(config, gbShape, (nuint)gbShape.Length, ToMlxDtype(DType.Float32)), "configuring packed Qwen35 GDN g output");
                    Check(mlx_fast_metal_kernel_config_add_output_arg(config, gbShape, (nuint)gbShape.Length, ToMlxDtype(DType.Float32)), "configuring packed Qwen35 GDN beta output");
                    Check(mlx_fast_metal_kernel_config_add_output_arg(config, zShape, (nuint)zShape.Length, ToMlxDtype(DType.Float32)), "configuring packed Qwen35 GDN z output");
                    Check(mlx_fast_metal_kernel_config_add_output_arg(config, nextConvShape, (nuint)nextConvShape.Length, ToMlxDtype(DType.Float32)), "configuring packed Qwen35 GDN next conv output");

                    int maxX = Math.Max(qkvDim, valueDim);
                    int maxY = Math.Max(seqLen, tail);
                    Check(mlx_fast_metal_kernel_config_set_grid(config, maxX, maxY, 3), "configuring packed Qwen35 GDN preprocess grid");
                    Check(mlx_fast_metal_kernel_config_set_thread_group(config, 256, 1, 1), "configuring packed Qwen35 GDN preprocess threadgroup");

                    length = mlx_array_new_int(seqLen);
                    inputs = CreateVectorArray(packedRaw, convState, convWeight, dtBias, aLog, length);
                    outputs = mlx_vector_array_new();
                    Check(mlx_fast_metal_kernel_apply(ref outputs, kernel, inputs, config, DefaultStream()), "running packed Qwen35 GDN preprocess kernel");
                    if (mlx_vector_array_size(outputs) < 7)
                        throw new InvalidOperationException("Packed Qwen35 GDN preprocess kernel produced fewer than seven outputs.");

                    Check(mlx_vector_array_get(out qResult, outputs, 0), "reading packed Qwen35 GDN q output");
                    Check(mlx_vector_array_get(out kResult, outputs, 1), "reading packed Qwen35 GDN k output");
                    Check(mlx_vector_array_get(out vResult, outputs, 2), "reading packed Qwen35 GDN v output");
                    Check(mlx_vector_array_get(out gResult, outputs, 3), "reading packed Qwen35 GDN g output");
                    Check(mlx_vector_array_get(out betaResult, outputs, 4), "reading packed Qwen35 GDN beta output");
                    Check(mlx_vector_array_get(out zSiluResult, outputs, 5), "reading packed Qwen35 GDN z output");
                    Check(mlx_vector_array_get(out nextConvResult, outputs, 6), "reading packed Qwen35 GDN next conv output");
                }
                finally
                {
                    FreeArray(length);
                    if (inputs.IsValid)
                        _ = mlx_vector_array_free(inputs);
                    if (outputs.IsValid)
                        _ = mlx_vector_array_free(outputs);
                    if (config.IsValid)
                        _ = mlx_fast_metal_kernel_config_free(config);
                }
            });

            q = qResult;
            k = kResult;
            v = vResult;
            g = gResult;
            beta = betaResult;
            zSilu = zSiluResult;
            nextConv = nextConvResult;
        }

        internal static MlxArray Qwen35GdnPostprocess(
            MlxArray y,
            MlxArray zSilu,
            MlxArray normWeight,
            int seqLen,
            int valueDim,
            int numValueHeads,
            int headValueDim,
            float eps)
        {
            if (!y.IsValid || !zSilu.IsValid || !normWeight.IsValid)
                throw new ArgumentException("Qwen35 GDN postprocess requires valid input arrays.");
            if (seqLen <= 0 || valueDim <= 0 || numValueHeads <= 0 || headValueDim <= 0
                || valueDim != numValueHeads * headValueDim || headValueDim > 256)
                throw new ArgumentOutOfRangeException(nameof(seqLen));

            return MlxWorker.Shared.Invoke(() =>
            {
                MlxFastMetalKernel kernel = EnsureQwen35GdnPostprocessKernel();
                MlxFastMetalKernelConfig config = mlx_fast_metal_kernel_config_new();
                MlxVectorArray inputs = default;
                MlxVectorArray outputs = default;
                try
                {
                    // No T template: the source never reads it, and each distinct
                    // value compiled another copy of the kernel.
                    AddTemplateInt(config, "ValueDim", valueDim);
                    AddTemplateInt(config, "NumValueHeads", numValueHeads);
                    AddTemplateInt(config, "HeadValueDim", headValueDim);
                    int[] shape = { seqLen, valueDim };
                    Check(mlx_fast_metal_kernel_config_add_output_arg(config, shape, (nuint)shape.Length, ToMlxDtype(DType.Float32)), "configuring Qwen35 GDN postprocess output");
                    Check(mlx_fast_metal_kernel_config_set_grid(config, 256, seqLen, numValueHeads), "configuring Qwen35 GDN postprocess grid");
                    Check(mlx_fast_metal_kernel_config_set_thread_group(config, 256, 1, 1), "configuring Qwen35 GDN postprocess threadgroup");

                    inputs = CreateVectorArray(y, zSilu, normWeight);
                    outputs = mlx_vector_array_new();
                    Check(mlx_fast_metal_kernel_apply(ref outputs, kernel, inputs, config, DefaultStream()), "running Qwen35 GDN postprocess kernel");
                    if (mlx_vector_array_size(outputs) < 1)
                        throw new InvalidOperationException("Qwen35 GDN postprocess kernel produced no output.");

                    Check(mlx_vector_array_get(out MlxArray result, outputs, 0), "reading Qwen35 GDN postprocess output");
                    return result;
                }
                finally
                {
                    if (inputs.IsValid)
                        _ = mlx_vector_array_free(inputs);
                    if (outputs.IsValid)
                        _ = mlx_vector_array_free(outputs);
                    if (config.IsValid)
                        _ = mlx_fast_metal_kernel_config_free(config);
                }
            });
        }

        private static MlxArray KQuantMatmul(
            MlxArray input,
            MlxArray rawWeight,
            int rows,
            int inDim,
            int outDim,
            Func<MlxFastMetalKernel> ensureKernel,
            string label)
        {
            if (!input.IsValid || !rawWeight.IsValid)
                throw new ArgumentException($"{label} matmul requires valid input and raw weight arrays.");
            if (rows <= 0 || inDim <= 0 || outDim <= 0 || inDim % 256 != 0)
                throw new ArgumentOutOfRangeException(nameof(inDim), $"{label} matmul requires positive dimensions and input dim aligned to 256.");

            return MlxWorker.Shared.Invoke(() =>
            {
                MlxFastMetalKernel kernel = ensureKernel();
                MlxFastMetalKernelConfig config = mlx_fast_metal_kernel_config_new();
                MlxVectorArray inputs = default;
                MlxVectorArray outputs = default;
                try
                {
                    AddTemplateInt(config, "InDim", inDim);
                    AddTemplateInt(config, "OutDim", outDim);
                    AddTemplateInt(config, "BlocksPerRow", inDim / 256);
                    int[] shape = { rows, outDim };
                    Check(mlx_fast_metal_kernel_config_add_output_arg(config, shape, (nuint)shape.Length, ToMlxDtype(DType.Float32)), $"configuring {label} matmul output");
                    Check(mlx_fast_metal_kernel_config_set_grid(config, 256, outDim, rows), $"configuring {label} matmul grid");
                    Check(mlx_fast_metal_kernel_config_set_thread_group(config, 256, 1, 1), $"configuring {label} matmul threadgroup");

                    inputs = CreateVectorArray(input, rawWeight);
                    outputs = mlx_vector_array_new();
                    Check(mlx_fast_metal_kernel_apply(ref outputs, kernel, inputs, config, DefaultStream()), $"running {label} matmul");
                    if (mlx_vector_array_size(outputs) < 1)
                        throw new InvalidOperationException($"{label} matmul produced no output.");

                    Check(mlx_vector_array_get(out MlxArray result, outputs, 0), $"reading {label} matmul output");
                    return result;
                }
                finally
                {
                    if (inputs.IsValid)
                        _ = mlx_vector_array_free(inputs);
                    if (outputs.IsValid)
                        _ = mlx_vector_array_free(outputs);
                    if (config.IsValid)
                        _ = mlx_fast_metal_kernel_config_free(config);
                }
            });
        }

        private static MlxArray KQuantGetRows(
            MlxArray rawWeight,
            MlxArray indices,
            int rows,
            int inDim,
            Func<MlxFastMetalKernel> ensureKernel,
            string label)
        {
            if (!rawWeight.IsValid || !indices.IsValid)
                throw new ArgumentException($"{label} get_rows requires valid raw weight and index arrays.");
            if (rows <= 0 || inDim <= 0 || inDim % 256 != 0)
                throw new ArgumentOutOfRangeException(nameof(inDim), $"{label} get_rows requires positive dimensions and input dim aligned to 256.");

            return MlxWorker.Shared.Invoke(() =>
            {
                MlxFastMetalKernel kernel = ensureKernel();
                MlxFastMetalKernelConfig config = mlx_fast_metal_kernel_config_new();
                MlxVectorArray inputs = default;
                MlxVectorArray outputs = default;
                try
                {
                    AddTemplateInt(config, "InDim", inDim);
                    AddTemplateInt(config, "BlocksPerRow", inDim / 256);
                    int[] shape = { rows, inDim };
                    Check(mlx_fast_metal_kernel_config_add_output_arg(config, shape, (nuint)shape.Length, ToMlxDtype(DType.Float32)), $"configuring {label} get_rows output");
                    Check(mlx_fast_metal_kernel_config_set_grid(config, inDim, rows, 1), $"configuring {label} get_rows grid");
                    Check(mlx_fast_metal_kernel_config_set_thread_group(config, 256, 1, 1), $"configuring {label} get_rows threadgroup");

                    inputs = CreateVectorArray(rawWeight, indices);
                    outputs = mlx_vector_array_new();
                    Check(mlx_fast_metal_kernel_apply(ref outputs, kernel, inputs, config, DefaultStream()), $"running {label} get_rows");
                    if (mlx_vector_array_size(outputs) < 1)
                        throw new InvalidOperationException($"{label} get_rows produced no output.");

                    Check(mlx_vector_array_get(out MlxArray result, outputs, 0), $"reading {label} get_rows output");
                    return result;
                }
                finally
                {
                    if (inputs.IsValid)
                        _ = mlx_vector_array_free(inputs);
                    if (outputs.IsValid)
                        _ = mlx_vector_array_free(outputs);
                    if (config.IsValid)
                        _ = mlx_fast_metal_kernel_config_free(config);
                }
            });
        }

        internal static MlxArray SliceUpdate(MlxArray input, MlxArray update, int start, int stop)
        {
            if (!input.IsValid || !update.IsValid)
                throw new ArgumentException("MLX slice update inputs must be valid arrays.");
            if (start < 0 || stop < start)
                throw new ArgumentOutOfRangeException(nameof(start));

            return MlxWorker.Shared.Invoke(() =>
            {
                int[] starts = { start };
                int[] stops = { stop };
                int[] strides = { 1 };
                MlxArray result;
                Check(mlx_slice_update(out result, input, update, starts, 1, stops, 1, strides, 1, DefaultStream()), "running MLX slice_update");
                return result;
            });
        }

        /// <summary>
        /// Multi-dim slice_update. Used by the KV-cache write path to update an
        /// entire <c>[heads, seqLen, headDim]</c> block at one position in a
        /// single MLX op, instead of looping per-head with 1D updates (which
        /// would cost 4+ MLX dispatches per layer for K + V × kvHeads).
        /// </summary>
        internal static MlxArray SliceUpdateMulti(MlxArray input, MlxArray update, int[] starts, int[] stops, int[] strides)
        {
            if (!input.IsValid || !update.IsValid)
                throw new ArgumentException("MLX slice update inputs must be valid arrays.");
            if (starts == null || stops == null || strides == null
                || starts.Length == 0 || starts.Length != stops.Length || starts.Length != strides.Length)
                throw new ArgumentException("MLX slice_update starts/stops/strides must be non-empty arrays of equal length.");

            return MlxWorker.Shared.Invoke(() =>
            {
                MlxArray result;
                Check(mlx_slice_update(out result, input, update,
                    starts, (nuint)starts.Length,
                    stops, (nuint)stops.Length,
                    strides, (nuint)strides.Length,
                    DefaultStream()), "running MLX slice_update (multi-dim)");
                return result;
            });
        }

        internal static MlxArray Slice(MlxArray input, int[] starts, int[] stops, int[] strides)
        {
            if (!input.IsValid)
                throw new ArgumentException("MLX slice input must be a valid array.", nameof(input));
            if (starts == null || stops == null || strides == null || starts.Length == 0 || starts.Length != stops.Length || starts.Length != strides.Length)
                throw new ArgumentException("MLX slice starts, stops, and strides must be non-empty arrays with the same length.");

            return MlxWorker.Shared.Invoke(() =>
            {
                MlxArray result;
                Check(mlx_slice(out result, input, starts, (nuint)starts.Length, stops, (nuint)stops.Length, strides, (nuint)strides.Length, DefaultStream()), "running MLX slice");
                return result;
            });
        }

        private static MlxStream DefaultStream()
        {
            // The default MLX stream for a given device is a stable handle owned
            // by MLX itself; recomputing it on every op (which the previous
            // implementation did) cost a device_new + device_free round trip per
            // call. Cache it once per device so the hot path is a single read.
            int desiredDevice = initializedDevice >= 0 ? initializedDevice : 0;
            if (!DisableStreamCache && cachedDefaultStreamDevice == desiredDevice)
                return cachedDefaultStream;

            MlxDevice device = mlx_device_new_type(MlxGpu, desiredDevice);
            try
            {
                Check(mlx_get_default_stream(out MlxStream stream, device), "getting MLX default stream");
                if (!DisableStreamCache)
                {
                    cachedDefaultStream = stream;
                    cachedDefaultStreamDevice = desiredDevice;
                }
                return stream;
            }
            finally
            {
                _ = mlx_device_free(device);
            }
        }

        private static MlxFastMetalKernel EnsureIq4XsMatmulKernel()
        {
            lock (fastKernelSync)
            {
                if (iq4XsMatmulKernel.IsValid)
                    return iq4XsMatmulKernel;
                if (iq4XsMatmulKernelDisabled)
                    throw new NotSupportedException("MLX IQ4_XS matmul kernel was disabled after initialization failed.");

                iq4XsMatmulKernel = CreateFastMetalKernel(
                    "tensorsharp_iq4xs_matmul",
                    new[] { "x", "w" },
                    new[] { "y" },
                    Iq4XsMatmulSource,
                    Iq4NlLookupHeader);
                if (!iq4XsMatmulKernel.IsValid)
                {
                    iq4XsMatmulKernelDisabled = true;
                    throw new NotSupportedException("Unable to initialize MLX IQ4_XS matmul kernel.");
                }

                return iq4XsMatmulKernel;
            }
        }

        private static MlxFastMetalKernel EnsureIq4NlMatmulKernel()
        {
            lock (fastKernelSync)
            {
                if (iq4NlMatmulKernel.IsValid)
                    return iq4NlMatmulKernel;
                if (iq4NlMatmulKernelDisabled)
                    throw new NotSupportedException("MLX IQ4_NL matmul kernel was disabled after initialization failed.");

                iq4NlMatmulKernel = CreateFastMetalKernel(
                    "tensorsharp_iq4nl_matmul",
                    new[] { "x", "w" },
                    new[] { "y" },
                    Iq4NlMatmulSource,
                    Iq4NlLookupHeader);
                if (!iq4NlMatmulKernel.IsValid)
                {
                    iq4NlMatmulKernelDisabled = true;
                    throw new NotSupportedException("Unable to initialize MLX IQ4_NL matmul kernel.");
                }

                return iq4NlMatmulKernel;
            }
        }

        private static MlxFastMetalKernel EnsureIq4NlMatmulRowsKernel()
        {
            lock (fastKernelSync)
            {
                if (iq4NlMatmulRowsKernel.IsValid)
                    return iq4NlMatmulRowsKernel;
                if (iq4NlMatmulRowsKernelDisabled)
                    throw new NotSupportedException("MLX IQ4_NL multi-row matmul kernel was disabled after initialization failed.");

                iq4NlMatmulRowsKernel = CreateFastMetalKernel(
                    "tensorsharp_iq4nl_matmul_rows",
                    new[] { "x", "w" },
                    new[] { "y" },
                    Iq4NlMatmulRowsSource,
                    Iq4NlLookupHeader);
                if (!iq4NlMatmulRowsKernel.IsValid)
                {
                    iq4NlMatmulRowsKernelDisabled = true;
                    throw new NotSupportedException("Unable to initialize MLX IQ4_NL multi-row matmul kernel.");
                }

                return iq4NlMatmulRowsKernel;
            }
        }

        private static MlxFastMetalKernel EnsureIq4NlMoeMatmulBatchedKernel()
        {
            lock (fastKernelSync)
            {
                if (iq4NlMoeMatmulBatchedKernel.IsValid)
                    return iq4NlMoeMatmulBatchedKernel;
                if (iq4NlMoeMatmulBatchedKernelDisabled)
                    throw new NotSupportedException("MLX IQ4_NL batched-MoE matmul kernel was disabled after initialization failed.");

                iq4NlMoeMatmulBatchedKernel = CreateFastMetalKernel(
                    "tensorsharp_iq4nl_moe_matmul_batched",
                    new[] { "x", "w", "expert_indices" },
                    new[] { "y" },
                    Iq4NlMoeMatmulBatchedSource,
                    Iq4NlLookupHeader);
                if (!iq4NlMoeMatmulBatchedKernel.IsValid)
                {
                    iq4NlMoeMatmulBatchedKernelDisabled = true;
                    throw new NotSupportedException("Unable to initialize MLX IQ4_NL batched-MoE matmul kernel.");
                }

                return iq4NlMoeMatmulBatchedKernel;
            }
        }

        private static MlxFastMetalKernel EnsureIq4NlMoeMatmulBatchedRowedKernel()
        {
            lock (fastKernelSync)
            {
                if (iq4NlMoeMatmulBatchedRowedKernel.IsValid)
                    return iq4NlMoeMatmulBatchedRowedKernel;
                if (iq4NlMoeMatmulBatchedRowedKernelDisabled)
                    throw new NotSupportedException("MLX IQ4_NL batched-MoE (rowed) matmul kernel was disabled after initialization failed.");

                iq4NlMoeMatmulBatchedRowedKernel = CreateFastMetalKernel(
                    "tensorsharp_iq4nl_moe_matmul_batched_rowed",
                    new[] { "x", "w", "expert_indices" },
                    new[] { "y" },
                    Iq4NlMoeMatmulBatchedRowedSource,
                    Iq4NlLookupHeader);
                if (!iq4NlMoeMatmulBatchedRowedKernel.IsValid)
                {
                    iq4NlMoeMatmulBatchedRowedKernelDisabled = true;
                    throw new NotSupportedException("Unable to initialize MLX IQ4_NL batched-MoE (rowed) matmul kernel.");
                }

                return iq4NlMoeMatmulBatchedRowedKernel;
            }
        }

        private static MlxFastMetalKernel EnsureIq4XsMatmul4Kernel()
        {
            lock (fastKernelSync)
            {
                if (iq4XsMatmul4Kernel.IsValid)
                    return iq4XsMatmul4Kernel;
                if (iq4XsMatmul4KernelDisabled)
                    throw new NotSupportedException("MLX IQ4_XS 4-column matmul kernel was disabled after initialization failed.");

                iq4XsMatmul4Kernel = CreateFastMetalKernel(
                    "tensorsharp_iq4xs_matmul4",
                    new[] { "x", "w" },
                    new[] { "y" },
                    Iq4XsMatmul4Source,
                    Iq4XsHelpersHeader);
                if (!iq4XsMatmul4Kernel.IsValid)
                {
                    iq4XsMatmul4KernelDisabled = true;
                    throw new NotSupportedException("Unable to initialize MLX IQ4_XS 4-column matmul kernel.");
                }

                return iq4XsMatmul4Kernel;
            }
        }

        private static MlxFastMetalKernel EnsureIq4XsMatmul4SimdKernel()
        {
            lock (fastKernelSync)
            {
                if (iq4XsMatmul4SimdKernel.IsValid)
                    return iq4XsMatmul4SimdKernel;
                if (iq4XsMatmul4SimdKernelDisabled)
                    throw new NotSupportedException("MLX IQ4_XS simd_sum 4-column matmul kernel was disabled after initialization failed.");

                iq4XsMatmul4SimdKernel = CreateFastMetalKernel(
                    "tensorsharp_iq4xs_matmul4_simd",
                    new[] { "x", "w" },
                    new[] { "y" },
                    Iq4XsMatmul4SimdSource,
                    Iq4XsHelpersHeader);
                if (!iq4XsMatmul4SimdKernel.IsValid)
                {
                    iq4XsMatmul4SimdKernelDisabled = true;
                    throw new NotSupportedException("Unable to initialize MLX IQ4_XS simd_sum 4-column matmul kernel.");
                }

                return iq4XsMatmul4SimdKernel;
            }
        }

        private static MlxFastMetalKernel EnsureIq4XsMatmulRowsKernel()
        {
            lock (fastKernelSync)
            {
                if (iq4XsMatmulRowsKernel.IsValid)
                    return iq4XsMatmulRowsKernel;
                if (iq4XsMatmulRowsKernelDisabled)
                    throw new NotSupportedException("MLX IQ4_XS batched-row matmul kernel was disabled after initialization failed.");

                iq4XsMatmulRowsKernel = CreateFastMetalKernel(
                    "tensorsharp_iq4xs_matmul_rows",
                    new[] { "x", "w" },
                    new[] { "y" },
                    Iq4XsMatmulRowsSource,
                    Iq4NlLookupHeader);
                if (!iq4XsMatmulRowsKernel.IsValid)
                {
                    iq4XsMatmulRowsKernelDisabled = true;
                    throw new NotSupportedException("Unable to initialize MLX IQ4_XS batched-row matmul kernel.");
                }

                return iq4XsMatmulRowsKernel;
            }
        }

        private static MlxFastMetalKernel EnsureIq4XsMatmulRows2Kernel()
        {
            lock (fastKernelSync)
            {
                if (iq4XsMatmulRows2Kernel.IsValid)
                    return iq4XsMatmulRows2Kernel;
                if (iq4XsMatmulRows2KernelDisabled)
                    throw new NotSupportedException("MLX IQ4_XS 2-column batched matmul kernel was disabled after initialization failed.");

                iq4XsMatmulRows2Kernel = CreateFastMetalKernel(
                    "tensorsharp_iq4xs_matmul_rows2",
                    new[] { "x", "w" },
                    new[] { "y" },
                    Iq4XsMatmulRows2Source,
                    Iq4XsHelpersHeader);
                if (!iq4XsMatmulRows2Kernel.IsValid)
                {
                    iq4XsMatmulRows2KernelDisabled = true;
                    throw new NotSupportedException("Unable to initialize MLX IQ4_XS 2-column batched matmul kernel.");
                }

                return iq4XsMatmulRows2Kernel;
            }
        }

        private static MlxFastMetalKernel EnsureIq4XsGetRowsKernel()
        {
            lock (fastKernelSync)
            {
                if (iq4XsGetRowsKernel.IsValid)
                    return iq4XsGetRowsKernel;
                if (iq4XsGetRowsKernelDisabled)
                    throw new NotSupportedException("MLX IQ4_XS get_rows kernel was disabled after initialization failed.");

                iq4XsGetRowsKernel = CreateFastMetalKernel(
                    "tensorsharp_iq4xs_get_rows",
                    new[] { "w", "indices" },
                    new[] { "y" },
                    Iq4XsGetRowsSource,
                    Iq4NlLookupHeader);
                if (!iq4XsGetRowsKernel.IsValid)
                {
                    iq4XsGetRowsKernelDisabled = true;
                    throw new NotSupportedException("Unable to initialize MLX IQ4_XS get_rows kernel.");
                }

                return iq4XsGetRowsKernel;
            }
        }

        private static MlxFastMetalKernel EnsureIq2XxsMatmulKernel()
        {
            lock (fastKernelSync)
            {
                if (iq2XxsMatmulKernel.IsValid)
                    return iq2XxsMatmulKernel;
                if (iq2XxsMatmulKernelDisabled)
                    throw new NotSupportedException("MLX IQ2_XXS matmul kernel was disabled after initialization failed.");

                iq2XxsMatmulKernel = CreateFastMetalKernel(
                    "tensorsharp_iq2xxs_matmul",
                    new[] { "x", "w" },
                    new[] { "y" },
                    Iq2XxsMatmulSource,
                    Iq2XxsLookupHeader);
                if (!iq2XxsMatmulKernel.IsValid)
                {
                    iq2XxsMatmulKernelDisabled = true;
                    throw new NotSupportedException("Unable to initialize MLX IQ2_XXS matmul kernel.");
                }

                return iq2XxsMatmulKernel;
            }
        }

        // Header prelude for kernels that use Apple's simdgroup_matrix
        // hardware primitives. mlx_fast_metal_kernel already pulls in
        // <metal_stdlib> and <metal_simdgroup>, but simdgroup_matrix is
        // in its own header that's not always included automatically.
        // Concatenated with quant-specific lookup headers where needed.
        private const string SimdgroupMatrixHeader = @"
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
";

        // Templated simdgroup_matrix matmul source generator. Same body
        // for every quant type — the only differences are the per-block
        // byte count (encoded in the address arithmetic) and the dequant
        // function name. Building the source via Replace at kernel-init
        // time avoids 6× copy-paste in the file while still producing a
        // const-per-call kernel that the Metal JIT can specialize.
        private static string BuildIQuantMatmulSimdgroupSource(string dequantFunc, int blockBytes)
        {
            const string template = @"
auto tid = thread_position_in_threadgroup.x;
auto tile_m_idx = thread_position_in_grid.y;
auto tile_b_idx = thread_position_in_grid.z;
constexpr int TileSize = 8;
int tile_m = static_cast<int>(tile_m_idx) * TileSize;
int tile_b = static_cast<int>(tile_b_idx) * TileSize;

threadgroup half X_tile[TileSize * TileSize];
threadgroup half W_tile[TileSize * TileSize];
threadgroup float C_out[TileSize * TileSize];

simdgroup_float8x8 C = simdgroup_float8x8(0.0f);

for (int k0 = 0; k0 < InDim; k0 += TileSize) {
    for (int i = 0; i < 2; ++i) {
        int idx = static_cast<int>(tid) * 2 + i;
        int m_in_tile = idx >> 3;
        int k_in_chunk = idx & 7;
        int m = tile_m + m_in_tile;
        int k = k0 + k_in_chunk;
        half val;
        if (m < OutDim && k < InDim) {
            int block_in_row = k >> 8;
            int within_block = k & 255;
            auto block = w + (m * BlocksPerRow + block_in_row) * __BLOCK_BYTES__;
            val = static_cast<half>(__DEQUANT__(block, within_block));
        } else {
            val = static_cast<half>(0.0f);
        }
        W_tile[m_in_tile * TileSize + k_in_chunk] = val;
    }

    for (int i = 0; i < 2; ++i) {
        int idx = static_cast<int>(tid) * 2 + i;
        int b_in_tile = idx >> 3;
        int k_in_chunk = idx & 7;
        int row = tile_b + b_in_tile;
        int col = k0 + k_in_chunk;
        half val = (row < InRows && col < InDim)
            ? static_cast<half>(x[row * InDim + col])
            : static_cast<half>(0.0f);
        X_tile[b_in_tile * TileSize + k_in_chunk] = val;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_half8x8 A, B;
    simdgroup_load(A, X_tile, TileSize);
    simdgroup_load(B, W_tile, TileSize, ulong2(0, 0), true);
    simdgroup_multiply_accumulate(C, A, B, C);
}

if (tile_b + TileSize <= InRows && tile_m + TileSize <= OutDim) {
    simdgroup_store(C, y + tile_b * OutDim + tile_m, OutDim);
} else {
    simdgroup_store(C, C_out, TileSize);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = 0; i < 2; ++i) {
        int idx = static_cast<int>(tid) * 2 + i;
        int b_in_tile = idx >> 3;
        int m_in_tile = idx & 7;
        int row = tile_b + b_in_tile;
        int col = tile_m + m_in_tile;
        if (row < InRows && col < OutDim) {
            y[row * OutDim + col] = C_out[b_in_tile * TileSize + m_in_tile];
        }
    }
}
";
            return template
                .Replace("__DEQUANT__", dequantFunc)
                .Replace("__BLOCK_BYTES__", blockBytes.ToString());
        }

        // Generic IQ/K-quant simdgroup matmul dispatcher. Same grid/TG
        // setup as Iq2XxsMatmulSimdgroup; the kernel itself is closured
        // via ensureKernel.
        private static MlxArray IQuantMatmulSimdgroup(
            MlxArray input, MlxArray rawWeight,
            int rows, int inDim, int outDim,
            Func<MlxFastMetalKernel> ensureKernel,
            string label)
        {
            if (!input.IsValid || !rawWeight.IsValid)
                throw new ArgumentException($"{label} simdgroup matmul requires valid input and raw weight arrays.");
            if (rows <= 0 || inDim <= 0 || outDim <= 0 || inDim % 8 != 0)
                throw new ArgumentOutOfRangeException(nameof(inDim), $"{label} simdgroup matmul requires positive dimensions and inDim divisible by 8.");

            return MlxWorker.Shared.Invoke(() =>
            {
                MlxFastMetalKernel kernel = ensureKernel();
                MlxFastMetalKernelConfig config = mlx_fast_metal_kernel_config_new();
                MlxVectorArray inputs = default;
                MlxVectorArray outputs = default;
                try
                {
                    AddTemplateInt(config, "InDim", inDim);
                    AddTemplateInt(config, "OutDim", outDim);
                    AddTemplateInt(config, "InRows", rows);
                    AddTemplateInt(config, "BlocksPerRow", inDim / 256);
                    int[] shape = { rows, outDim };
                    Check(mlx_fast_metal_kernel_config_add_output_arg(config, shape, (nuint)shape.Length, ToMlxDtype(DType.Float32)), $"configuring {label} simdgroup matmul output");
                    int tilesM = (outDim + 7) / 8;
                    int tilesB = (rows + 7) / 8;
                    Check(mlx_fast_metal_kernel_config_set_grid(config, 32, tilesM, tilesB), $"configuring {label} simdgroup matmul grid");
                    Check(mlx_fast_metal_kernel_config_set_thread_group(config, 32, 1, 1), $"configuring {label} simdgroup matmul threadgroup");

                    inputs = CreateVectorArray(input, rawWeight);
                    outputs = mlx_vector_array_new();
                    Check(mlx_fast_metal_kernel_apply(ref outputs, kernel, inputs, config, DefaultStream()), $"running {label} simdgroup matmul");
                    if (mlx_vector_array_size(outputs) < 1)
                        throw new InvalidOperationException($"{label} simdgroup matmul produced no output.");

                    Check(mlx_vector_array_get(out MlxArray result, outputs, 0), $"reading {label} simdgroup matmul output");
                    return result;
                }
                finally
                {
                    if (inputs.IsValid) _ = mlx_vector_array_free(inputs);
                    if (outputs.IsValid) _ = mlx_vector_array_free(outputs);
                    if (config.IsValid) _ = mlx_fast_metal_kernel_config_free(config);
                }
            });
        }

        // Runs at most once per kernel family: callers only reach this on the
        // disabled-flag transition, which happens once under fastKernelSync.
        private static void WarnSimdgroupKernelUnavailable(string label)
        {
            try
            {
                Console.Error.WriteLine(
                    $"[mlx] the {label} simdgroup_matrix matmul kernel failed to compile; {label} prefill " +
                    "permanently uses the ~8x slower per-output kernel for the rest of this process. " +
                    "This usually means a Metal toolchain without simdgroup_matrix support - " +
                    "update Xcode / the Metal command line tools. Reported once.");
            }
            catch
            {
                // Diagnostics must never break kernel dispatch.
            }
        }

        // Generic per-quant simdgroup kernel ensure: builds the kernel
        // source from the template + a quant-specific dequant function
        // name and block byte count, paired with the per-quant header
        // (lookup tables / dequant helper definitions).
        private static MlxFastMetalKernel EnsureSgKernel(
            ref MlxFastMetalKernel slot,
            ref bool disabledFlag,
            string kernelName,
            string dequantFunc,
            int blockBytes,
            string quantHeader,
            string label)
        {
            lock (fastKernelSync)
            {
                if (slot.IsValid)
                    return slot;
                if (disabledFlag)
                    throw new NotSupportedException($"MLX {label} simdgroup matmul kernel was disabled after initialization failed.");

                slot = CreateFastMetalKernel(
                    kernelName,
                    new[] { "x", "w" },
                    new[] { "y" },
                    BuildIQuantMatmulSimdgroupSource(dequantFunc, blockBytes),
                    SimdgroupMatrixHeader + quantHeader);
                if (!slot.IsValid)
                {
                    disabledFlag = true;
                    WarnSimdgroupKernelUnavailable(label);
                    throw new NotSupportedException($"Unable to initialize MLX {label} simdgroup matmul kernel.");
                }
                return slot;
            }
        }

        private static MlxFastMetalKernel EnsureIq2XxsMatmulSimdgroupKernel()
        {
            lock (fastKernelSync)
            {
                if (iq2XxsMatmulSimdgroupKernel.IsValid)
                    return iq2XxsMatmulSimdgroupKernel;
                if (iq2XxsMatmulSimdgroupKernelDisabled)
                    throw new NotSupportedException("MLX IQ2_XXS simdgroup_matrix matmul kernel was disabled after initialization failed.");

                iq2XxsMatmulSimdgroupKernel = CreateFastMetalKernel(
                    "tensorsharp_iq2xxs_matmul_sg",
                    new[] { "x", "w" },
                    new[] { "y" },
                    Iq2XxsMatmulSimdgroupSource,
                    SimdgroupMatrixHeader + Iq2XxsLookupHeader);
                if (!iq2XxsMatmulSimdgroupKernel.IsValid)
                {
                    iq2XxsMatmulSimdgroupKernelDisabled = true;
                    WarnSimdgroupKernelUnavailable("IQ2_XXS");
                    throw new NotSupportedException("Unable to initialize MLX IQ2_XXS simdgroup_matrix matmul kernel.");
                }
                return iq2XxsMatmulSimdgroupKernel;
            }
        }

        // Per-quant simdgroup_matrix kernel ensures. All share the same
        // kernel body (BuildIQuantMatmulSimdgroupSource) — only the
        // dequant function name + block byte count differ, plus the
        // per-quant lookup table header.
        private static MlxFastMetalKernel EnsureIq2SMatmulSimdgroupKernel() =>
            EnsureSgKernel(ref iq2SMatmulSimdgroupKernel, ref iq2SMatmulSimdgroupKernelDisabled,
                "tensorsharp_iq2s_matmul_sg", "tensorsharp_dequant_iq2_s", 82,
                Iq2SIq3SLookupHeader, "IQ2_S");

        private static MlxFastMetalKernel EnsureIq3SMatmulSimdgroupKernel() =>
            EnsureSgKernel(ref iq3SMatmulSimdgroupKernel, ref iq3SMatmulSimdgroupKernelDisabled,
                "tensorsharp_iq3s_matmul_sg", "tensorsharp_dequant_iq3_s", 110,
                Iq2SIq3SLookupHeader, "IQ3_S");

        // IQ3_XXS shares the 256-element super-block layout of the other
        // i-quants; only the 98-byte block stride and the dequant helper
        // differ. Unsloth's UD mixed quants put IQ3_XXS on ffn_down (the
        // largest matmul in each layer), so it needs the same kernel coverage
        // as IQ3_S or those tensors fall back to the C# row-dequant path.
        private static MlxFastMetalKernel EnsureIq3XxsMatmulSimdgroupKernel() =>
            EnsureSgKernel(ref iq3XxsMatmulSimdgroupKernel, ref iq3XxsMatmulSimdgroupKernelDisabled,
                "tensorsharp_iq3xxs_matmul_sg", "tensorsharp_dequant_iq3_xxs", 98,
                Iq3XxsHelpersHeader, "IQ3_XXS");

        private static MlxFastMetalKernel EnsureIq4XsMatmulSimdgroupKernel() =>
            EnsureSgKernel(ref iq4XsMatmulSimdgroupKernel, ref iq4XsMatmulSimdgroupKernelDisabled,
                "tensorsharp_iq4xs_matmul_sg", "tensorsharp_dequant_iq4xs", 136,
                Iq4XsHelpersHeader, "IQ4_XS");

        private static MlxFastMetalKernel EnsureQ4KMatmulSimdgroupKernel() =>
            EnsureSgKernel(ref q4KMatmulSimdgroupKernel, ref q4KMatmulSimdgroupKernelDisabled,
                "tensorsharp_q4k_matmul_sg", "tensorsharp_dequant_q4k", 144,
                KQuantHelpersHeader, "Q4_K");

        private static MlxFastMetalKernel EnsureQ5KMatmulSimdgroupKernel() =>
            EnsureSgKernel(ref q5KMatmulSimdgroupKernel, ref q5KMatmulSimdgroupKernelDisabled,
                "tensorsharp_q5k_matmul_sg", "tensorsharp_dequant_q5k", 176,
                KQuantHelpersHeader, "Q5_K");

        private static MlxFastMetalKernel EnsureQ6KMatmulSimdgroupKernel() =>
            EnsureSgKernel(ref q6KMatmulSimdgroupKernel, ref q6KMatmulSimdgroupKernelDisabled,
                "tensorsharp_q6k_matmul_sg", "tensorsharp_dequant_q6k", 210,
                KQuantHelpersHeader, "Q6_K");

        private static MlxFastMetalKernel EnsureIq2XxsMoeMatmulBatchedKernel()
        {
            lock (fastKernelSync)
            {
                if (iq2XxsMoeMatmulBatchedKernel.IsValid)
                    return iq2XxsMoeMatmulBatchedKernel;
                if (iq2XxsMoeMatmulBatchedKernelDisabled)
                    throw new NotSupportedException("MLX IQ2_XXS MoE batched matmul kernel was disabled after initialization failed.");

                iq2XxsMoeMatmulBatchedKernel = CreateFastMetalKernel(
                    "tensorsharp_iq2xxs_moe_matmul_batched",
                    new[] { "x", "w", "expert_indices" },
                    new[] { "y" },
                    Iq2XxsMoeMatmulBatchedSource,
                    Iq2XxsLookupHeader);
                if (!iq2XxsMoeMatmulBatchedKernel.IsValid)
                {
                    iq2XxsMoeMatmulBatchedKernelDisabled = true;
                    throw new NotSupportedException("Unable to initialize MLX IQ2_XXS MoE batched matmul kernel.");
                }
                return iq2XxsMoeMatmulBatchedKernel;
            }
        }

        private static MlxFastMetalKernel EnsureIq2XxsMoeMatmulBatchedFusedGateUpSiluKernel()
        {
            lock (fastKernelSync)
            {
                if (iq2XxsMoeMatmulBatchedFusedGateUpSiluKernel.IsValid)
                    return iq2XxsMoeMatmulBatchedFusedGateUpSiluKernel;
                if (iq2XxsMoeMatmulBatchedFusedGateUpSiluKernelDisabled)
                    throw new NotSupportedException("MLX IQ2_XXS MoE fused gate+up+silu kernel was disabled after initialization failed.");

                iq2XxsMoeMatmulBatchedFusedGateUpSiluKernel = CreateFastMetalKernel(
                    "tensorsharp_iq2xxs_moe_matmul_batched_fused_gateup_silu",
                    new[] { "x", "w_gate", "w_up", "expert_indices" },
                    new[] { "y" },
                    Iq2XxsMoeMatmulBatchedFusedGateUpSiluSource,
                    Iq2XxsLookupHeader);
                if (!iq2XxsMoeMatmulBatchedFusedGateUpSiluKernel.IsValid)
                {
                    iq2XxsMoeMatmulBatchedFusedGateUpSiluKernelDisabled = true;
                    throw new NotSupportedException("Unable to initialize MLX IQ2_XXS MoE fused gate+up+silu kernel.");
                }
                return iq2XxsMoeMatmulBatchedFusedGateUpSiluKernel;
            }
        }

        private static MlxFastMetalKernel EnsureIq2XxsMoeMatmulBatchedRowedKernel()
        {
            lock (fastKernelSync)
            {
                if (iq2XxsMoeMatmulBatchedRowedKernel.IsValid)
                    return iq2XxsMoeMatmulBatchedRowedKernel;
                if (iq2XxsMoeMatmulBatchedRowedKernelDisabled)
                    throw new NotSupportedException("MLX IQ2_XXS MoE batched-rowed matmul kernel was disabled after initialization failed.");

                iq2XxsMoeMatmulBatchedRowedKernel = CreateFastMetalKernel(
                    "tensorsharp_iq2xxs_moe_matmul_batched_rowed",
                    new[] { "x", "w", "expert_indices" },
                    new[] { "y" },
                    Iq2XxsMoeMatmulBatchedRowedSource,
                    Iq2XxsLookupHeader);
                if (!iq2XxsMoeMatmulBatchedRowedKernel.IsValid)
                {
                    iq2XxsMoeMatmulBatchedRowedKernelDisabled = true;
                    throw new NotSupportedException("Unable to initialize MLX IQ2_XXS MoE batched-rowed matmul kernel.");
                }
                return iq2XxsMoeMatmulBatchedRowedKernel;
            }
        }

        private static MlxFastMetalKernel EnsureIq2XxsGetRowsKernel()
        {
            lock (fastKernelSync)
            {
                if (iq2XxsGetRowsKernel.IsValid)
                    return iq2XxsGetRowsKernel;
                if (iq2XxsGetRowsKernelDisabled)
                    throw new NotSupportedException("MLX IQ2_XXS get_rows kernel was disabled after initialization failed.");

                iq2XxsGetRowsKernel = CreateFastMetalKernel(
                    "tensorsharp_iq2xxs_get_rows",
                    new[] { "w", "indices" },
                    new[] { "y" },
                    Iq2XxsGetRowsSource,
                    Iq2XxsLookupHeader);
                if (!iq2XxsGetRowsKernel.IsValid)
                {
                    iq2XxsGetRowsKernelDisabled = true;
                    throw new NotSupportedException("Unable to initialize MLX IQ2_XXS get_rows kernel.");
                }

                return iq2XxsGetRowsKernel;
            }
        }



        private static MlxFastMetalKernel EnsureIq2SMatmulKernel()
        {
            lock (fastKernelSync)
            {
                if (iq2SMatmulKernel.IsValid)
                    return iq2SMatmulKernel;
                if (iq2SMatmulKernelDisabled)
                    throw new NotSupportedException("MLX IQ2_S matmul kernel was disabled after initialization failed.");

                iq2SMatmulKernel = CreateFastMetalKernel(
                    "tensorsharp_iq2s_matmul",
                    new[] { "x", "w" },
                    new[] { "y" },
                    Iq2SMatmulSource,
                    Iq2SIq3SLookupHeader);
                if (!iq2SMatmulKernel.IsValid)
                {
                    iq2SMatmulKernelDisabled = true;
                    throw new NotSupportedException("Unable to initialize MLX IQ2_S matmul kernel.");
                }

                return iq2SMatmulKernel;
            }
        }

        private static MlxFastMetalKernel EnsureIq2SGetRowsKernel()
        {
            lock (fastKernelSync)
            {
                if (iq2SGetRowsKernel.IsValid)
                    return iq2SGetRowsKernel;
                if (iq2SGetRowsKernelDisabled)
                    throw new NotSupportedException("MLX IQ2_S get_rows kernel was disabled after initialization failed.");

                iq2SGetRowsKernel = CreateFastMetalKernel(
                    "tensorsharp_iq2s_get_rows",
                    new[] { "w", "indices" },
                    new[] { "y" },
                    Iq2SGetRowsSource,
                    Iq2SIq3SLookupHeader);
                if (!iq2SGetRowsKernel.IsValid)
                {
                    iq2SGetRowsKernelDisabled = true;
                    throw new NotSupportedException("Unable to initialize MLX IQ2_S get_rows kernel.");
                }

                return iq2SGetRowsKernel;
            }
        }

        private static MlxFastMetalKernel EnsureIq3SMatmulKernel()
        {
            lock (fastKernelSync)
            {
                if (iq3SMatmulKernel.IsValid)
                    return iq3SMatmulKernel;
                if (iq3SMatmulKernelDisabled)
                    throw new NotSupportedException("MLX IQ3_S matmul kernel was disabled after initialization failed.");

                iq3SMatmulKernel = CreateFastMetalKernel(
                    "tensorsharp_iq3s_matmul",
                    new[] { "x", "w" },
                    new[] { "y" },
                    Iq3SMatmulSource,
                    Iq2SIq3SLookupHeader);
                if (!iq3SMatmulKernel.IsValid)
                {
                    iq3SMatmulKernelDisabled = true;
                    throw new NotSupportedException("Unable to initialize MLX IQ3_S matmul kernel.");
                }

                return iq3SMatmulKernel;
            }
        }

        private static MlxFastMetalKernel EnsureIq3SGetRowsKernel()
        {
            lock (fastKernelSync)
            {
                if (iq3SGetRowsKernel.IsValid)
                    return iq3SGetRowsKernel;
                if (iq3SGetRowsKernelDisabled)
                    throw new NotSupportedException("MLX IQ3_S get_rows kernel was disabled after initialization failed.");

                iq3SGetRowsKernel = CreateFastMetalKernel(
                    "tensorsharp_iq3s_get_rows",
                    new[] { "w", "indices" },
                    new[] { "y" },
                    Iq3SGetRowsSource,
                    Iq2SIq3SLookupHeader);
                if (!iq3SGetRowsKernel.IsValid)
                {
                    iq3SGetRowsKernelDisabled = true;
                    throw new NotSupportedException("Unable to initialize MLX IQ3_S get_rows kernel.");
                }

                return iq3SGetRowsKernel;
            }
        }

        private static MlxFastMetalKernel EnsureIq3XxsMatmulKernel()
        {
            lock (fastKernelSync)
            {
                if (iq3XxsMatmulKernel.IsValid)
                    return iq3XxsMatmulKernel;
                if (iq3XxsMatmulKernelDisabled)
                    throw new NotSupportedException("MLX IQ3_XXS matmul kernel was disabled after initialization failed.");

                iq3XxsMatmulKernel = CreateFastMetalKernel(
                    "tensorsharp_iq3xxs_matmul",
                    new[] { "x", "w" },
                    new[] { "y" },
                    Iq3XxsMatmulSource,
                    Iq3XxsHelpersHeader);
                if (!iq3XxsMatmulKernel.IsValid)
                {
                    iq3XxsMatmulKernelDisabled = true;
                    throw new NotSupportedException("Unable to initialize MLX IQ3_XXS matmul kernel.");
                }

                return iq3XxsMatmulKernel;
            }
        }

        private static MlxFastMetalKernel EnsureIq3XxsGetRowsKernel()
        {
            lock (fastKernelSync)
            {
                if (iq3XxsGetRowsKernel.IsValid)
                    return iq3XxsGetRowsKernel;
                if (iq3XxsGetRowsKernelDisabled)
                    throw new NotSupportedException("MLX IQ3_XXS get_rows kernel was disabled after initialization failed.");

                iq3XxsGetRowsKernel = CreateFastMetalKernel(
                    "tensorsharp_iq3xxs_get_rows",
                    new[] { "w", "indices" },
                    new[] { "y" },
                    Iq3XxsGetRowsSource,
                    Iq3XxsHelpersHeader);
                if (!iq3XxsGetRowsKernel.IsValid)
                {
                    iq3XxsGetRowsKernelDisabled = true;
                    throw new NotSupportedException("Unable to initialize MLX IQ3_XXS get_rows kernel.");
                }

                return iq3XxsGetRowsKernel;
            }
        }

        private static MlxFastMetalKernel EnsureQ4KMatmulKernel()
        {
            lock (fastKernelSync)
            {
                if (q4KMatmulKernel.IsValid)
                    return q4KMatmulKernel;
                if (q4KMatmulKernelDisabled)
                    throw new NotSupportedException("MLX Q4_K matmul kernel was disabled after initialization failed.");

                q4KMatmulKernel = CreateFastMetalKernel(
                    "tensorsharp_q4k_matmul",
                    new[] { "x", "w" },
                    new[] { "y" },
                    Q4KMatmulSource,
                    KQuantHelpersHeader);
                if (!q4KMatmulKernel.IsValid)
                {
                    q4KMatmulKernelDisabled = true;
                    throw new NotSupportedException("Unable to initialize MLX Q4_K matmul kernel.");
                }

                return q4KMatmulKernel;
            }
        }

        private static MlxFastMetalKernel EnsureQ4KGetRowsKernel()
        {
            lock (fastKernelSync)
            {
                if (q4KGetRowsKernel.IsValid)
                    return q4KGetRowsKernel;
                if (q4KGetRowsKernelDisabled)
                    throw new NotSupportedException("MLX Q4_K get_rows kernel was disabled after initialization failed.");

                q4KGetRowsKernel = CreateFastMetalKernel(
                    "tensorsharp_q4k_get_rows",
                    new[] { "w", "indices" },
                    new[] { "y" },
                    Q4KGetRowsSource,
                    KQuantHelpersHeader);
                if (!q4KGetRowsKernel.IsValid)
                {
                    q4KGetRowsKernelDisabled = true;
                    throw new NotSupportedException("Unable to initialize MLX Q4_K get_rows kernel.");
                }

                return q4KGetRowsKernel;
            }
        }

        private static MlxFastMetalKernel EnsureQ5KMatmulKernel()
        {
            lock (fastKernelSync)
            {
                if (q5KMatmulKernel.IsValid)
                    return q5KMatmulKernel;
                if (q5KMatmulKernelDisabled)
                    throw new NotSupportedException("MLX Q5_K matmul kernel was disabled after initialization failed.");

                q5KMatmulKernel = CreateFastMetalKernel(
                    "tensorsharp_q5k_matmul",
                    new[] { "x", "w" },
                    new[] { "y" },
                    Q5KMatmulSource,
                    KQuantHelpersHeader);
                if (!q5KMatmulKernel.IsValid)
                {
                    q5KMatmulKernelDisabled = true;
                    throw new NotSupportedException("Unable to initialize MLX Q5_K matmul kernel.");
                }

                return q5KMatmulKernel;
            }
        }

        private static MlxFastMetalKernel EnsureQ5KMatmul4Kernel()
        {
            lock (fastKernelSync)
            {
                if (q5KMatmul4Kernel.IsValid)
                    return q5KMatmul4Kernel;
                if (q5KMatmul4KernelDisabled)
                    throw new NotSupportedException("MLX Q5_K 4-column matmul kernel was disabled after initialization failed.");

                q5KMatmul4Kernel = CreateFastMetalKernel(
                    "tensorsharp_q5k_matmul4",
                    new[] { "x", "w" },
                    new[] { "y" },
                    Q5KMatmul4Source,
                    KQuantHelpersHeader);
                if (!q5KMatmul4Kernel.IsValid)
                {
                    q5KMatmul4KernelDisabled = true;
                    throw new NotSupportedException("Unable to initialize MLX Q5_K 4-column matmul kernel.");
                }

                return q5KMatmul4Kernel;
            }
        }

        private static MlxFastMetalKernel EnsureQ5KGetRowsKernel()
        {
            lock (fastKernelSync)
            {
                if (q5KGetRowsKernel.IsValid)
                    return q5KGetRowsKernel;
                if (q5KGetRowsKernelDisabled)
                    throw new NotSupportedException("MLX Q5_K get_rows kernel was disabled after initialization failed.");

                q5KGetRowsKernel = CreateFastMetalKernel(
                    "tensorsharp_q5k_get_rows",
                    new[] { "w", "indices" },
                    new[] { "y" },
                    Q5KGetRowsSource,
                    KQuantHelpersHeader);
                if (!q5KGetRowsKernel.IsValid)
                {
                    q5KGetRowsKernelDisabled = true;
                    throw new NotSupportedException("Unable to initialize MLX Q5_K get_rows kernel.");
                }

                return q5KGetRowsKernel;
            }
        }

        private static MlxFastMetalKernel EnsureQ6KMatmulKernel()
        {
            lock (fastKernelSync)
            {
                if (q6KMatmulKernel.IsValid)
                    return q6KMatmulKernel;
                if (q6KMatmulKernelDisabled)
                    throw new NotSupportedException("MLX Q6_K matmul kernel was disabled after initialization failed.");

                q6KMatmulKernel = CreateFastMetalKernel(
                    "tensorsharp_q6k_matmul",
                    new[] { "x", "w" },
                    new[] { "y" },
                    Q6KMatmulSource,
                    KQuantHelpersHeader);
                if (!q6KMatmulKernel.IsValid)
                {
                    q6KMatmulKernelDisabled = true;
                    throw new NotSupportedException("Unable to initialize MLX Q6_K matmul kernel.");
                }

                return q6KMatmulKernel;
            }
        }

        private static MlxFastMetalKernel EnsureQ6KMatmul4Kernel()
        {
            lock (fastKernelSync)
            {
                if (q6KMatmul4Kernel.IsValid)
                    return q6KMatmul4Kernel;
                if (q6KMatmul4KernelDisabled)
                    throw new NotSupportedException("MLX Q6_K 4-column matmul kernel was disabled after initialization failed.");

                q6KMatmul4Kernel = CreateFastMetalKernel(
                    "tensorsharp_q6k_matmul4",
                    new[] { "x", "w" },
                    new[] { "y" },
                    Q6KMatmul4Source,
                    KQuantHelpersHeader);
                if (!q6KMatmul4Kernel.IsValid)
                {
                    q6KMatmul4KernelDisabled = true;
                    throw new NotSupportedException("Unable to initialize MLX Q6_K 4-column matmul kernel.");
                }

                return q6KMatmul4Kernel;
            }
        }

        private static MlxFastMetalKernel EnsureQ6KMatvecKernel()
        {
            lock (fastKernelSync)
            {
                if (q6KMatvecKernel.IsValid)
                    return q6KMatvecKernel;
                if (q6KMatvecKernelDisabled)
                    throw new NotSupportedException("MLX Q6_K matrix-vector kernel was disabled after initialization failed.");

                q6KMatvecKernel = CreateFastMetalKernel(
                    "tensorsharp_q6k_matvec",
                    new[] { "x", "w", "in_dim", "out_dim" },
                    new[] { "y" },
                    Q6KMatvecSource,
                    string.Empty);
                if (!q6KMatvecKernel.IsValid)
                {
                    q6KMatvecKernelDisabled = true;
                    throw new NotSupportedException("Unable to initialize MLX Q6_K matrix-vector kernel.");
                }

                return q6KMatvecKernel;
            }
        }

        private static MlxFastMetalKernel EnsureIq4XsMatvecKernel()
        {
            lock (fastKernelSync)
            {
                if (iq4XsMatvecKernel.IsValid)
                    return iq4XsMatvecKernel;
                if (iq4XsMatvecKernelDisabled)
                    throw new NotSupportedException("MLX IQ4_XS matrix-vector kernel was disabled after initialization failed.");

                iq4XsMatvecKernel = CreateFastMetalKernel(
                    "tensorsharp_iq4xs_matvec",
                    new[] { "x", "w", "in_dim", "out_dim" },
                    new[] { "y" },
                    Iq4XsMatvecSource,
                    Iq4NlLookupHeader);
                if (!iq4XsMatvecKernel.IsValid)
                {
                    iq4XsMatvecKernelDisabled = true;
                    throw new NotSupportedException("Unable to initialize MLX IQ4_XS matrix-vector kernel.");
                }

                return iq4XsMatvecKernel;
            }
        }

        private static MlxFastMetalKernel EnsureIq4XsDequantF16Kernel()
        {
            lock (fastKernelSync)
            {
                if (iq4XsDequantF16Kernel.IsValid)
                    return iq4XsDequantF16Kernel;
                if (iq4XsDequantF16KernelDisabled)
                    throw new NotSupportedException("MLX IQ4_XS F16 dequantization kernel was disabled after initialization failed.");

                iq4XsDequantF16Kernel = CreateFastMetalKernel(
                    "tensorsharp_iq4xs_dequant_f16",
                    new[] { "w", "in_dim", "row0" },
                    new[] { "y" },
                    Iq4XsDequantF16Source,
                    Iq4NlLookupHeader);
                if (!iq4XsDequantF16Kernel.IsValid)
                {
                    iq4XsDequantF16KernelDisabled = true;
                    throw new NotSupportedException("Unable to initialize MLX IQ4_XS F16 dequantization kernel.");
                }

                return iq4XsDequantF16Kernel;
            }
        }

        private static MlxFastMetalKernel EnsureQ6KDequantF16Kernel()
        {
            lock (fastKernelSync)
            {
                if (q6KDequantF16Kernel.IsValid)
                    return q6KDequantF16Kernel;
                if (q6KDequantF16KernelDisabled)
                    throw new NotSupportedException("MLX Q6_K F16 dequantization kernel was disabled after initialization failed.");

                q6KDequantF16Kernel = CreateFastMetalKernel(
                    "tensorsharp_q6k_dequant_f16",
                    new[] { "w", "in_dim", "row0" },
                    new[] { "y" },
                    Q6KDequantF16Source,
                    string.Empty);
                if (!q6KDequantF16Kernel.IsValid)
                {
                    q6KDequantF16KernelDisabled = true;
                    throw new NotSupportedException("Unable to initialize MLX Q6_K F16 dequantization kernel.");
                }

                return q6KDequantF16Kernel;
            }
        }

        private static MlxFastMetalKernel EnsureQ6KGetRowsKernel()
        {
            lock (fastKernelSync)
            {
                if (q6KGetRowsKernel.IsValid)
                    return q6KGetRowsKernel;
                if (q6KGetRowsKernelDisabled)
                    throw new NotSupportedException("MLX Q6_K get_rows kernel was disabled after initialization failed.");

                q6KGetRowsKernel = CreateFastMetalKernel(
                    "tensorsharp_q6k_get_rows",
                    new[] { "w", "indices" },
                    new[] { "y" },
                    Q6KGetRowsSource,
                    KQuantHelpersHeader);
                if (!q6KGetRowsKernel.IsValid)
                {
                    q6KGetRowsKernelDisabled = true;
                    throw new NotSupportedException("Unable to initialize MLX Q6_K get_rows kernel.");
                }

                return q6KGetRowsKernel;
            }
        }

        private static MlxFastMetalKernel EnsureGatedDeltaKernel()
        {
            lock (fastKernelSync)
            {
                if (gatedDeltaKernel.IsValid)
                    return gatedDeltaKernel;
                if (gatedDeltaKernelDisabled)
                    throw new NotSupportedException("MLX gated-delta kernel was disabled after initialization failed.");

                gatedDeltaKernel = CreateFastMetalKernel(
                    "tensorsharp_gated_delta_step",
                    new[] { "q", "k", "v", "g", "beta", "state_in" },
                    new[] { "y", "state_out" },
                    GatedDeltaSource,
                    string.Empty);
                if (!gatedDeltaKernel.IsValid)
                {
                    gatedDeltaKernelDisabled = true;
                    throw new NotSupportedException("Unable to initialize MLX gated-delta kernel.");
                }

                return gatedDeltaKernel;
            }
        }

        private static MlxFastMetalKernel EnsureGatedDeltaBlockedKernel()
        {
            lock (fastKernelSync)
            {
                if (gatedDeltaBlockedKernel.IsValid)
                    return gatedDeltaBlockedKernel;
                if (gatedDeltaBlockedKernelDisabled)
                    throw new NotSupportedException("MLX blocked gated-delta kernel was disabled after initialization failed.");

                gatedDeltaBlockedKernel = CreateFastMetalKernel(
                    "tensorsharp_gated_delta_blocked",
                    new[] { "q", "k", "v", "g", "beta", "state_in", "T" },
                    new[] { "y", "state_out" },
                    GatedDeltaBlockedSource,
                    string.Empty);
                if (!gatedDeltaBlockedKernel.IsValid)
                {
                    gatedDeltaBlockedKernelDisabled = true;
                    throw new NotSupportedException("Unable to initialize MLX blocked gated-delta kernel.");
                }

                return gatedDeltaBlockedKernel;
            }
        }

        private static MlxFastMetalKernel EnsureGatedDeltaT1Kernel()
        {
            lock (fastKernelSync)
            {
                if (gatedDeltaT1Kernel.IsValid)
                    return gatedDeltaT1Kernel;
                if (gatedDeltaT1KernelDisabled)
                    throw new NotSupportedException("MLX gated-delta T=1 kernel was disabled after initialization failed.");

                gatedDeltaT1Kernel = CreateFastMetalKernel(
                    "tensorsharp_gated_delta_step_t1",
                    new[] { "q", "k", "v", "g", "beta", "state_in" },
                    new[] { "y", "state_out" },
                    GatedDeltaT1Source,
                    string.Empty);
                if (!gatedDeltaT1Kernel.IsValid)
                {
                    gatedDeltaT1KernelDisabled = true;
                    throw new NotSupportedException("Unable to initialize MLX gated-delta T=1 kernel.");
                }

                return gatedDeltaT1Kernel;
            }
        }

        private static MlxFastMetalKernel EnsureQwen35GdnPreprocessKernel()
        {
            lock (fastKernelSync)
            {
                if (qwen35GdnPreprocessKernel.IsValid)
                    return qwen35GdnPreprocessKernel;
                if (qwen35GdnPreprocessKernelDisabled)
                    throw new NotSupportedException("MLX Qwen35 GDN preprocess kernel was disabled after initialization failed.");

                qwen35GdnPreprocessKernel = CreateFastMetalKernel(
                    "tensorsharp_qwen35_gdn_preprocess",
                    new[] { "qkv_raw", "z_raw", "beta_raw", "alpha_raw", "conv_state", "conv_weight", "dt_bias", "a_log" },
                    new[] { "q_out", "k_out", "v_out", "g_out", "beta_out", "z_silu", "next_conv" },
                    Qwen35GdnPreprocessSource,
                    string.Empty);
                if (!qwen35GdnPreprocessKernel.IsValid)
                {
                    qwen35GdnPreprocessKernelDisabled = true;
                    throw new NotSupportedException("Unable to initialize MLX Qwen35 GDN preprocess kernel.");
                }

                return qwen35GdnPreprocessKernel;
            }
        }

        private static MlxFastMetalKernel EnsureQwen35GdnPackedPreprocessKernel()
        {
            lock (fastKernelSync)
            {
                if (qwen35GdnPackedPreprocessKernel.IsValid)
                    return qwen35GdnPackedPreprocessKernel;
                if (qwen35GdnPackedPreprocessKernelDisabled)
                    throw new NotSupportedException("MLX packed Qwen35 GDN preprocess kernel was disabled after initialization failed.");

                qwen35GdnPackedPreprocessKernel = CreateFastMetalKernel(
                    "tensorsharp_qwen35_gdn_packed_preprocess",
                    new[] { "packed_raw", "conv_state", "conv_weight", "dt_bias", "a_log", "T" },
                    new[] { "q_out", "k_out", "v_out", "g_out", "beta_out", "z_silu", "next_conv" },
                    Qwen35GdnPackedPreprocessSource,
                    string.Empty);
                if (!qwen35GdnPackedPreprocessKernel.IsValid)
                {
                    qwen35GdnPackedPreprocessKernelDisabled = true;
                    throw new NotSupportedException("Unable to initialize MLX packed Qwen35 GDN preprocess kernel.");
                }

                return qwen35GdnPackedPreprocessKernel;
            }
        }

        private static MlxFastMetalKernel EnsureQwen35GdnPostprocessKernel()
        {
            lock (fastKernelSync)
            {
                if (qwen35GdnPostprocessKernel.IsValid)
                    return qwen35GdnPostprocessKernel;
                if (qwen35GdnPostprocessKernelDisabled)
                    throw new NotSupportedException("MLX Qwen35 GDN postprocess kernel was disabled after initialization failed.");

                qwen35GdnPostprocessKernel = CreateFastMetalKernel(
                    "tensorsharp_qwen35_gdn_postprocess",
                    new[] { "y_in", "z_silu", "norm_weight" },
                    new[] { "y_out" },
                    Qwen35GdnPostprocessSource,
                    string.Empty);
                if (!qwen35GdnPostprocessKernel.IsValid)
                {
                    qwen35GdnPostprocessKernelDisabled = true;
                    throw new NotSupportedException("Unable to initialize MLX Qwen35 GDN postprocess kernel.");
                }

                return qwen35GdnPostprocessKernel;
            }
        }

        private static MlxFastMetalKernel EnsureHeadDim256AttentionKernel()
        {
            lock (fastKernelSync)
            {
                if (headDim256AttentionKernel.IsValid)
                    return headDim256AttentionKernel;
                if (headDim256AttentionKernelDisabled)
                    throw new NotSupportedException("MLX headDim256 attention kernel was disabled after initialization failed.");

                headDim256AttentionKernel = CreateFastMetalKernel(
                    "tensorsharp_head_dim_256_attention",
                    new[] { "q", "k", "v", "scale_value" },
                    new[] { "y" },
                    HeadDim256AttentionSource,
                    string.Empty);
                if (!headDim256AttentionKernel.IsValid)
                {
                    headDim256AttentionKernelDisabled = true;
                    throw new NotSupportedException("Unable to initialize MLX headDim256 attention kernel.");
                }

                return headDim256AttentionKernel;
            }
        }

        private static MlxFastMetalKernel EnsureCircularDecodeAttentionKernel()
        {
            lock (fastKernelSync)
            {
                if (circularDecodeAttentionKernel.IsValid)
                    return circularDecodeAttentionKernel;
                if (circularDecodeAttentionKernelDisabled)
                    throw new NotSupportedException("MLX circular decode attention kernel was disabled after initialization failed.");

                circularDecodeAttentionKernel = CreateFastMetalKernel(
                    "tensorsharp_circular_decode_attention",
                    new[] { "q", "k_cache", "v_cache", "scale_value", "FirstSlot", "AttendLen" },
                    new[] { "y" },
                    CircularDecodeAttentionSource,
                    string.Empty);
                if (!circularDecodeAttentionKernel.IsValid)
                {
                    circularDecodeAttentionKernelDisabled = true;
                    throw new NotSupportedException("Unable to initialize MLX circular decode attention kernel.");
                }

                return circularDecodeAttentionKernel;
            }
        }

        private static MlxFastMetalKernel EnsureDecodeAttentionWithSinksKernel()
        {
            lock (fastKernelSync)
            {
                if (decodeAttentionWithSinksKernel.IsValid)
                    return decodeAttentionWithSinksKernel;
                if (decodeAttentionWithSinksKernelDisabled)
                    throw new NotSupportedException("MLX decode attention with sinks kernel was disabled after initialization failed.");

                decodeAttentionWithSinksKernel = CreateFastMetalKernel(
                    "tensorsharp_decode_attention_with_sinks",
                    new[] { "q", "k_cache", "v_cache", "sinks", "scale_value", "MaskStart", "AttendLen" },
                    new[] { "y" },
                    DecodeAttentionWithSinksSource,
                    string.Empty);
                if (!decodeAttentionWithSinksKernel.IsValid)
                {
                    decodeAttentionWithSinksKernelDisabled = true;
                    throw new NotSupportedException("Unable to initialize MLX decode attention with sinks kernel.");
                }

                return decodeAttentionWithSinksKernel;
            }
        }

        private static MlxFastMetalKernel EnsureGemma4QkvPreprocessDecodeKernel()
        {
            lock (fastKernelSync)
            {
                if (gemma4QkvPreprocessDecodeKernel.IsValid)
                    return gemma4QkvPreprocessDecodeKernel;
                if (gemma4QkvPreprocessDecodeKernelDisabled)
                    throw new NotSupportedException("MLX Gemma4 QKV preprocess decode kernel was disabled after initialization failed.");

                gemma4QkvPreprocessDecodeKernel = CreateFastMetalKernel(
                    "tensorsharp_gemma4_qkv_preprocess_decode",
                    new[] { "qkv", "q_norm_w", "k_norm_w", "cos_table", "sin_table", "eps_value" },
                    new[] { "q_out", "k_out", "v_out" },
                    Gemma4QkvPreprocessDecodeSource,
                    string.Empty);
                if (!gemma4QkvPreprocessDecodeKernel.IsValid)
                {
                    gemma4QkvPreprocessDecodeKernelDisabled = true;
                    throw new NotSupportedException("Unable to initialize MLX Gemma4 QKV preprocess decode kernel.");
                }

                return gemma4QkvPreprocessDecodeKernel;
            }
        }

        private static MlxFastMetalKernel EnsureQ8AddmmAddKernel()
        {
            lock (fastKernelSync)
            {
                if (q8AddmmAddKernel.IsValid)
                    return q8AddmmAddKernel;
                if (q8AddmmAddKernelDisabled)
                    throw new NotSupportedException("MLX Q8 addmm+add kernel was disabled after initialization failed.");

                q8AddmmAddKernel = CreateFastMetalKernel(
                    "tensorsharp_q8_addmm_add",
                    new[] { "x", "w", "scales", "biases", "residual" },
                    new[] { "y" },
                    Q8AddmmAddSource,
                    string.Empty);
                if (!q8AddmmAddKernel.IsValid)
                {
                    q8AddmmAddKernelDisabled = true;
                    throw new NotSupportedException("Unable to initialize MLX Q8 addmm+add kernel.");
                }

                return q8AddmmAddKernel;
            }
        }

        private static MlxFastMetalKernel EnsureDecodeAttentionHeadDim512Kernel()
        {
            lock (fastKernelSync)
            {
                if (decodeAttentionHeadDim512Kernel.IsValid)
                    return decodeAttentionHeadDim512Kernel;
                if (decodeAttentionHeadDim512KernelDisabled)
                    throw new NotSupportedException("MLX head_dim=512 decode attention kernel was disabled after initialization failed.");

                decodeAttentionHeadDim512Kernel = CreateFastMetalKernel(
                    "tensorsharp_decode_attention_head_dim_512",
                    new[] { "q", "k_cache", "v_cache", "scale_value", "AttendLen" },
                    new[] { "y" },
                    DecodeAttentionHeadDim512Source,
                    string.Empty);
                if (!decodeAttentionHeadDim512Kernel.IsValid)
                {
                    decodeAttentionHeadDim512KernelDisabled = true;
                    throw new NotSupportedException("Unable to initialize MLX head_dim=512 decode attention kernel.");
                }

                return decodeAttentionHeadDim512Kernel;
            }
        }

        /// <summary>
        /// Decode attention for head_dim = 512 (Gemma 4 global layers).
        /// Non-circular. Uses simdgroup-fast online-softmax structure
        /// matching <see cref="CircularDecodeAttentionSource"/>.
        /// </summary>
        internal static MlxArray DecodeAttentionHeadDim512(
            MlxArray qFlat,
            MlxArray kCache,
            MlxArray vCache,
            int numHeads,
            int numKVHeads,
            int headDim,
            int cacheLen,
            int attendLen,
            float scale)
        {
            if (!qFlat.IsValid || !kCache.IsValid || !vCache.IsValid)
                throw new ArgumentException("MLX head_dim=512 decode attention inputs must be valid arrays.");
            if (numHeads <= 0 || numKVHeads <= 0 || numHeads % numKVHeads != 0
                || headDim != 512 || cacheLen <= 0 || attendLen <= 0 || attendLen > cacheLen)
                throw new ArgumentOutOfRangeException(nameof(headDim), "head_dim=512 decode attention requires HeadDim==512 and valid dims.");

            return MlxWorker.Shared.Invoke(() =>
            {
                MlxFastMetalKernel kernel = EnsureDecodeAttentionHeadDim512Kernel();
                MlxFastMetalKernelConfig config = mlx_fast_metal_kernel_config_new();
                MlxVectorArray inputs = default;
                MlxVectorArray outputs = default;
                MlxArray scaleArray = default; MlxArray attendLenArray = default;
                try
                {
                    AddTemplateInt(config, "NumHeads", numHeads);
                    AddTemplateInt(config, "NumKVHeads", numKVHeads);
                    AddTemplateInt(config, "HeadDim", headDim);
                    AddTemplateInt(config, "CacheLen", cacheLen);

                    int[] shape = { 1, numHeads * headDim };
                    Check(mlx_fast_metal_kernel_config_add_output_arg(config, shape, (nuint)shape.Length, ToMlxDtype(DType.Float32)), "configuring head_dim=512 decode attention output");
                    Check(mlx_fast_metal_kernel_config_set_grid(config, headDim, numHeads, 1), "configuring head_dim=512 decode attention grid");
                    Check(mlx_fast_metal_kernel_config_set_thread_group(config, headDim, 1, 1), "configuring head_dim=512 decode attention threadgroup");

                    scaleArray = mlx_array_new_float32(scale);
                    attendLenArray = mlx_array_new_int(attendLen);
                    inputs = CreateVectorArray(qFlat, kCache, vCache, scaleArray, attendLenArray);
                    outputs = mlx_vector_array_new();
                    Check(mlx_fast_metal_kernel_apply(ref outputs, kernel, inputs, config, DefaultStream()), "running head_dim=512 decode attention");
                    if (mlx_vector_array_size(outputs) < 1)
                        throw new InvalidOperationException("head_dim=512 decode attention kernel produced no output.");

                    Check(mlx_vector_array_get(out MlxArray result, outputs, 0), "reading head_dim=512 decode attention output");
                    return result;
                }
                finally
                {
                    if (scaleArray.IsValid) _ = mlx_array_free(scaleArray);
                    if (attendLenArray.IsValid) _ = mlx_array_free(attendLenArray);
                    if (inputs.IsValid) _ = mlx_vector_array_free(inputs);
                    if (outputs.IsValid) _ = mlx_vector_array_free(outputs);
                    if (config.IsValid) _ = mlx_fast_metal_kernel_config_free(config);
                }
            });
        }

        private static MlxFastMetalKernel EnsureQ8MatmulGeluMulKernel()
        {
            lock (fastKernelSync)
            {
                if (q8MatmulGeluMulKernel.IsValid)
                    return q8MatmulGeluMulKernel;
                if (q8MatmulGeluMulKernelDisabled)
                    throw new NotSupportedException("MLX Q8 matmul + GeluMul kernel was disabled after initialization failed.");

                q8MatmulGeluMulKernel = CreateFastMetalKernel(
                    "tensorsharp_q8_matmul_gelumul",
                    new[] { "x", "w", "scales", "biases", "gate" },
                    new[] { "y" },
                    Q8MatmulGeluMulSource,
                    string.Empty);
                if (!q8MatmulGeluMulKernel.IsValid)
                {
                    q8MatmulGeluMulKernelDisabled = true;
                    throw new NotSupportedException("Unable to initialize MLX Q8 matmul + GeluMul kernel.");
                }

                return q8MatmulGeluMulKernel;
            }
        }

        /// <summary>
        /// Fused Q8 matmul + GELU-tanh + per-element multiply (decode,
        /// rows == 1). Replaces (mlx_quantized_matmul + mlx_binary_mul with
        /// gelu activation) for Gemma 4's PLE inp_gate stage.
        /// </summary>
        internal static MlxArray Q8MatmulGeluMul(
            MlxArray input, MlxArray weight, MlxArray scales, MlxArray biases,
            MlxArray gate, int inDim, int outDim, int blocksPerRow)
        {
            if (!input.IsValid || !weight.IsValid || !scales.IsValid || !biases.IsValid || !gate.IsValid)
                throw new ArgumentException("Q8 matmul + GeluMul requires valid input arrays.");
            if (inDim <= 0 || outDim <= 0 || inDim % 32 != 0 || blocksPerRow != inDim / 32)
                throw new ArgumentOutOfRangeException(nameof(inDim), "Q8 matmul + GeluMul requires inDim aligned to 32.");

            return MlxWorker.Shared.Invoke(() =>
            {
                MlxFastMetalKernel kernel = EnsureQ8MatmulGeluMulKernel();
                MlxFastMetalKernelConfig config = mlx_fast_metal_kernel_config_new();
                MlxVectorArray inputs = default;
                MlxVectorArray outputs = default;
                try
                {
                    AddTemplateInt(config, "InDim", inDim);
                    AddTemplateInt(config, "OutDim", outDim);
                    AddTemplateInt(config, "BlocksPerRow", blocksPerRow);

                    int[] shape = { 1, outDim };
                    Check(mlx_fast_metal_kernel_config_add_output_arg(config, shape, (nuint)shape.Length, ToMlxDtype(DType.Float32)), "configuring Q8 matmul+gelumul output");
                    Check(mlx_fast_metal_kernel_config_set_grid(config, 256, outDim, 1), "configuring Q8 matmul+gelumul grid");
                    Check(mlx_fast_metal_kernel_config_set_thread_group(config, 256, 1, 1), "configuring Q8 matmul+gelumul threadgroup");

                    inputs = CreateVectorArray(input, weight, scales, biases, gate);
                    outputs = mlx_vector_array_new();
                    Check(mlx_fast_metal_kernel_apply(ref outputs, kernel, inputs, config, DefaultStream()), "running Q8 matmul+gelumul kernel");
                    if (mlx_vector_array_size(outputs) < 1)
                        throw new InvalidOperationException("Q8 matmul+gelumul kernel produced no output.");

                    Check(mlx_vector_array_get(out MlxArray result, outputs, 0), "reading Q8 matmul+gelumul output");
                    return result;
                }
                finally
                {
                    if (inputs.IsValid) _ = mlx_vector_array_free(inputs);
                    if (outputs.IsValid) _ = mlx_vector_array_free(outputs);
                    if (config.IsValid) _ = mlx_fast_metal_kernel_config_free(config);
                }
            });
        }

        private static MlxFastMetalKernel EnsureQ8MatmulKernel()
        {
            lock (fastKernelSync)
            {
                if (q8MatmulKernel.IsValid)
                    return q8MatmulKernel;
                if (q8MatmulKernelDisabled)
                    throw new NotSupportedException("MLX Q8 matmul kernel was disabled after initialization failed.");

                q8MatmulKernel = CreateFastMetalKernel(
                    "tensorsharp_q8_matmul",
                    new[] { "x", "w", "scales", "biases" },
                    new[] { "y" },
                    Q8MatmulSource,
                    string.Empty);
                if (!q8MatmulKernel.IsValid)
                {
                    q8MatmulKernelDisabled = true;
                    throw new NotSupportedException("Unable to initialize MLX Q8 matmul kernel.");
                }

                return q8MatmulKernel;
            }
        }

        /// <summary>
        /// Plain Q8 matmul for decode (rows == 1). Drop-in replacement for
        /// MLX's built-in <c>mlx_quantized_matmul</c> on Q8_0 weights using
        /// the same affine-Q8 layout (Weight: uint32 packed, Scales/Biases:
        /// f16). Implemented with simdgroup-fast reductions so it matches
        /// or slightly beats the built-in path.
        /// </summary>
        internal static MlxArray Q8Matmul(
            MlxArray input, MlxArray weight, MlxArray scales, MlxArray biases,
            int inDim, int outDim, int blocksPerRow)
        {
            if (!input.IsValid || !weight.IsValid || !scales.IsValid || !biases.IsValid)
                throw new ArgumentException("Q8 matmul requires valid input arrays.");
            if (inDim <= 0 || outDim <= 0 || inDim % 32 != 0 || blocksPerRow != inDim / 32)
                throw new ArgumentOutOfRangeException(nameof(inDim), "Q8 matmul requires inDim aligned to 32.");

            return MlxWorker.Shared.Invoke(() =>
            {
                MlxFastMetalKernel kernel = EnsureQ8MatmulKernel();
                MlxFastMetalKernelConfig config = mlx_fast_metal_kernel_config_new();
                MlxVectorArray inputs = default;
                MlxVectorArray outputs = default;
                try
                {
                    AddTemplateInt(config, "InDim", inDim);
                    AddTemplateInt(config, "OutDim", outDim);
                    AddTemplateInt(config, "BlocksPerRow", blocksPerRow);

                    int[] shape = { 1, outDim };
                    Check(mlx_fast_metal_kernel_config_add_output_arg(config, shape, (nuint)shape.Length, ToMlxDtype(DType.Float32)), "configuring Q8 matmul output");
                    Check(mlx_fast_metal_kernel_config_set_grid(config, 256, outDim, 1), "configuring Q8 matmul grid");
                    Check(mlx_fast_metal_kernel_config_set_thread_group(config, 256, 1, 1), "configuring Q8 matmul threadgroup");

                    inputs = CreateVectorArray(input, weight, scales, biases);
                    outputs = mlx_vector_array_new();
                    Check(mlx_fast_metal_kernel_apply(ref outputs, kernel, inputs, config, DefaultStream()), "running Q8 matmul kernel");
                    if (mlx_vector_array_size(outputs) < 1)
                        throw new InvalidOperationException("Q8 matmul kernel produced no output.");

                    Check(mlx_vector_array_get(out MlxArray result, outputs, 0), "reading Q8 matmul output");
                    return result;
                }
                finally
                {
                    if (inputs.IsValid) _ = mlx_vector_array_free(inputs);
                    if (outputs.IsValid) _ = mlx_vector_array_free(outputs);
                    if (config.IsValid) _ = mlx_fast_metal_kernel_config_free(config);
                }
            });
        }

        private static MlxFastMetalKernel EnsureQ8RmsNormMatmulKernel()
        {
            lock (fastKernelSync)
            {
                if (q8RmsNormMatmulKernel.IsValid)
                    return q8RmsNormMatmulKernel;
                if (q8RmsNormMatmulKernelDisabled)
                    throw new NotSupportedException("MLX Q8 RmsNorm+matmul kernel was disabled after initialization failed.");

                q8RmsNormMatmulKernel = CreateFastMetalKernel(
                    "tensorsharp_q8_rmsnorm_matmul",
                    new[] { "x", "norm_w", "w", "scales", "biases", "eps_value" },
                    new[] { "y" },
                    Q8RmsNormMatmulSource,
                    string.Empty);
                if (!q8RmsNormMatmulKernel.IsValid)
                {
                    q8RmsNormMatmulKernelDisabled = true;
                    throw new NotSupportedException("Unable to initialize MLX Q8 RmsNorm+matmul kernel.");
                }

                return q8RmsNormMatmulKernel;
            }
        }

        /// <summary>
        /// Fused RMSNorm(input) + Q8 matmul for decode (rows == 1). One
        /// Metal dispatch replaces (mlx_fast_rms_norm + mlx_quantized_matmul).
        /// Uses MLX's affine Q8 layout (same inputs as
        /// <see cref="Q8AddmmAdd"/>).
        /// </summary>
        internal static MlxArray Q8RmsNormMatmul(
            MlxArray input, MlxArray normWeight, MlxArray weight, MlxArray scales, MlxArray biases,
            float eps, int inDim, int outDim, int blocksPerRow)
        {
            if (!input.IsValid || !normWeight.IsValid || !weight.IsValid || !scales.IsValid || !biases.IsValid)
                throw new ArgumentException("Q8 RmsNorm+matmul requires valid input arrays.");
            if (inDim <= 0 || outDim <= 0 || inDim % 32 != 0 || blocksPerRow != inDim / 32)
                throw new ArgumentOutOfRangeException(nameof(inDim), "Q8 RmsNorm+matmul requires inDim aligned to 32.");

            return MlxWorker.Shared.Invoke(() =>
            {
                MlxFastMetalKernel kernel = EnsureQ8RmsNormMatmulKernel();
                MlxFastMetalKernelConfig config = mlx_fast_metal_kernel_config_new();
                MlxVectorArray inputs = default;
                MlxVectorArray outputs = default;
                MlxArray epsArray = default;
                try
                {
                    AddTemplateInt(config, "InDim", inDim);
                    AddTemplateInt(config, "OutDim", outDim);
                    AddTemplateInt(config, "BlocksPerRow", blocksPerRow);

                    int[] shape = { 1, outDim };
                    Check(mlx_fast_metal_kernel_config_add_output_arg(config, shape, (nuint)shape.Length, ToMlxDtype(DType.Float32)), "configuring Q8 RmsNorm+matmul output");
                    Check(mlx_fast_metal_kernel_config_set_grid(config, 256, outDim, 1), "configuring Q8 RmsNorm+matmul grid");
                    Check(mlx_fast_metal_kernel_config_set_thread_group(config, 256, 1, 1), "configuring Q8 RmsNorm+matmul threadgroup");

                    epsArray = mlx_array_new_float32(eps);
                    inputs = CreateVectorArray(input, normWeight, weight, scales, biases, epsArray);
                    outputs = mlx_vector_array_new();
                    Check(mlx_fast_metal_kernel_apply(ref outputs, kernel, inputs, config, DefaultStream()), "running Q8 RmsNorm+matmul kernel");
                    if (mlx_vector_array_size(outputs) < 1)
                        throw new InvalidOperationException("Q8 RmsNorm+matmul kernel produced no output.");

                    Check(mlx_vector_array_get(out MlxArray result, outputs, 0), "reading Q8 RmsNorm+matmul output");
                    return result;
                }
                finally
                {
                    if (epsArray.IsValid) _ = mlx_array_free(epsArray);
                    if (inputs.IsValid) _ = mlx_vector_array_free(inputs);
                    if (outputs.IsValid) _ = mlx_vector_array_free(outputs);
                    if (config.IsValid) _ = mlx_fast_metal_kernel_config_free(config);
                }
            });
        }

        /// <summary>
        /// Fused Q8 matmul + residual add for decode (rows == 1). One Metal
        /// dispatch replaces (mlx_quantized_matmul + mlx_binary_add). Uses
        /// MLX's affine Q8 layout: <paramref name="weight"/> is the
        /// <c>[outDim, inDim/4]</c> uint32 packed-quants array,
        /// <paramref name="scales"/> / <paramref name="biases"/> are the
        /// matching <c>[outDim, blocksPerRow]</c> f16 arrays. Output goes
        /// into a fresh array which the caller assigns to the residual
        /// tensor via <c>SetDeviceResult</c>.
        /// </summary>
        internal static MlxArray Q8AddmmAdd(
            MlxArray input, MlxArray weight, MlxArray scales, MlxArray biases,
            MlxArray residual, int inDim, int outDim, int blocksPerRow)
        {
            if (!input.IsValid || !weight.IsValid || !scales.IsValid || !biases.IsValid || !residual.IsValid)
                throw new ArgumentException("Q8 addmm+add requires valid input arrays.");
            if (inDim <= 0 || outDim <= 0 || inDim % 32 != 0 || blocksPerRow != inDim / 32)
                throw new ArgumentOutOfRangeException(nameof(inDim), "Q8 addmm+add requires inDim aligned to 32.");

            return MlxWorker.Shared.Invoke(() =>
            {
                MlxFastMetalKernel kernel = EnsureQ8AddmmAddKernel();
                MlxFastMetalKernelConfig config = mlx_fast_metal_kernel_config_new();
                MlxVectorArray inputs = default;
                MlxVectorArray outputs = default;
                try
                {
                    AddTemplateInt(config, "InDim", inDim);
                    AddTemplateInt(config, "OutDim", outDim);
                    AddTemplateInt(config, "BlocksPerRow", blocksPerRow);

                    int[] shape = { 1, outDim };
                    Check(mlx_fast_metal_kernel_config_add_output_arg(config, shape, (nuint)shape.Length, ToMlxDtype(DType.Float32)), "configuring Q8 addmm+add output");
                    Check(mlx_fast_metal_kernel_config_set_grid(config, 256, outDim, 1), "configuring Q8 addmm+add grid");
                    Check(mlx_fast_metal_kernel_config_set_thread_group(config, 256, 1, 1), "configuring Q8 addmm+add threadgroup");

                    inputs = CreateVectorArray(input, weight, scales, biases, residual);
                    outputs = mlx_vector_array_new();
                    Check(mlx_fast_metal_kernel_apply(ref outputs, kernel, inputs, config, DefaultStream()), "running Q8 addmm+add kernel");
                    if (mlx_vector_array_size(outputs) < 1)
                        throw new InvalidOperationException("Q8 addmm+add kernel produced no output.");

                    Check(mlx_vector_array_get(out MlxArray result, outputs, 0), "reading Q8 addmm+add output");
                    return result;
                }
                finally
                {
                    if (inputs.IsValid) _ = mlx_vector_array_free(inputs);
                    if (outputs.IsValid) _ = mlx_vector_array_free(outputs);
                    if (config.IsValid) _ = mlx_fast_metal_kernel_config_free(config);
                }
            });
        }

        private static MlxFastMetalKernel EnsureScatterAddWeightedRowsKernel()
        {
            lock (fastKernelSync)
            {
                if (scatterAddWeightedRowsKernel.IsValid)
                    return scatterAddWeightedRowsKernel;
                if (scatterAddWeightedRowsKernelDisabled)
                    throw new NotSupportedException("MLX weighted scatter-add rows kernel was disabled after initialization failed.");

                scatterAddWeightedRowsKernel = CreateFastMetalKernel(
                    "tensorsharp_scatter_add_weighted_rows",
                    new[] { "in_y", "rows", "indices", "weights" },
                    new[] { "out_y" },
                    ScatterAddWeightedRowsSource,
                    string.Empty);
                if (!scatterAddWeightedRowsKernel.IsValid)
                {
                    scatterAddWeightedRowsKernelDisabled = true;
                    throw new NotSupportedException("Unable to initialize MLX weighted scatter-add rows kernel.");
                }

                return scatterAddWeightedRowsKernel;
            }
        }

        private static MlxFastMetalKernel EnsureRmsNormAddKernel()
        {
            lock (fastKernelSync)
            {
                if (rmsNormAddKernel.IsValid)
                    return rmsNormAddKernel;
                if (rmsNormAddKernelDisabled)
                    throw new NotSupportedException("MLX RMSNorm-add kernel was disabled after initialization failed.");

                rmsNormAddKernel = CreateFastMetalKernel(
                    "tensorsharp_rmsnorm_add",
                    new[] { "residual", "input", "norm_weight", "eps_value" },
                    new[] { "out_y" },
                    RmsNormAddSource,
                    string.Empty);
                if (!rmsNormAddKernel.IsValid)
                {
                    rmsNormAddKernelDisabled = true;
                    throw new NotSupportedException("Unable to initialize MLX RMSNorm-add kernel.");
                }

                return rmsNormAddKernel;
            }
        }

        private static MlxFastMetalKernel EnsureAddRmsNormKernel()
        {
            lock (fastKernelSync)
            {
                if (addRmsNormKernel.IsValid)
                    return addRmsNormKernel;
                if (addRmsNormKernelDisabled)
                    throw new NotSupportedException("MLX add-rmsnorm kernel was disabled after initialization failed.");

                addRmsNormKernel = CreateFastMetalKernel(
                    "tensorsharp_add_rmsnorm",
                    new[] { "residual", "input", "norm_weight", "eps_value" },
                    new[] { "updated_residual", "normed_out" },
                    AddRmsNormSource,
                    string.Empty);
                if (!addRmsNormKernel.IsValid)
                {
                    addRmsNormKernelDisabled = true;
                    throw new NotSupportedException("Unable to initialize MLX add-rmsnorm kernel.");
                }

                return addRmsNormKernel;
            }
        }

        private static MlxFastMetalKernel EnsureGeluMulSplitKernel()
        {
            lock (fastKernelSync)
            {
                if (geluMulSplitKernel.IsValid)
                    return geluMulSplitKernel;
                if (geluMulSplitKernelDisabled)
                    throw new NotSupportedException("MLX GELU-mul split kernel was disabled after initialization failed.");

                geluMulSplitKernel = CreateFastMetalKernel(
                    "tensorsharp_gelu_mul_split",
                    new[] { "gate_up" },
                    new[] { "out_y" },
                    GeluMulSplitSource,
                    string.Empty);
                if (!geluMulSplitKernel.IsValid)
                {
                    geluMulSplitKernelDisabled = true;
                    throw new NotSupportedException("Unable to initialize MLX GELU-mul split kernel.");
                }

                return geluMulSplitKernel;
            }
        }

        private static MlxFastMetalKernel EnsureSwigluOaiGatherBiasKernel()
        {
            lock (fastKernelSync)
            {
                if (swigluOaiGatherBiasKernel.IsValid)
                    return swigluOaiGatherBiasKernel;
                if (swigluOaiGatherBiasKernelDisabled)
                    throw new NotSupportedException("MLX swiglu-oai gather-bias kernel was disabled after initialization failed.");

                swigluOaiGatherBiasKernel = CreateFastMetalKernel(
                    "tensorsharp_swiglu_oai_gather_bias",
                    new[] { "gate_in", "up_in", "gate_bias", "up_bias", "experts", "alpha_v", "limit_v" },
                    new[] { "out_y" },
                    SwigluOaiGatherBiasSource,
                    string.Empty);
                if (!swigluOaiGatherBiasKernel.IsValid)
                {
                    swigluOaiGatherBiasKernelDisabled = true;
                    throw new NotSupportedException("Unable to initialize MLX swiglu-oai gather-bias kernel.");
                }

                return swigluOaiGatherBiasKernel;
            }
        }

        private static MlxFastMetalKernel EnsureMoeBiasWeightedSumKernel()
        {
            lock (fastKernelSync)
            {
                if (moeBiasWeightedSumKernel.IsValid)
                    return moeBiasWeightedSumKernel;
                if (moeBiasWeightedSumKernelDisabled)
                    throw new NotSupportedException("MLX MoE bias-weighted-sum kernel was disabled after initialization failed.");

                moeBiasWeightedSumKernel = CreateFastMetalKernel(
                    "tensorsharp_moe_bias_weighted_sum",
                    new[] { "down_rows", "down_bias", "experts_sorted", "inv_order", "route_weights" },
                    new[] { "out_y" },
                    MoeBiasWeightedSumSource,
                    string.Empty);
                if (!moeBiasWeightedSumKernel.IsValid)
                {
                    moeBiasWeightedSumKernelDisabled = true;
                    throw new NotSupportedException("Unable to initialize MLX MoE bias-weighted-sum kernel.");
                }

                return moeBiasWeightedSumKernel;
            }
        }

        private static MlxFastMetalKernel EnsureFlatToHeadFirstKernel()
        {
            lock (fastKernelSync)
            {
                if (flatToHeadFirstKernel.IsValid)
                    return flatToHeadFirstKernel;
                if (flatToHeadFirstKernelDisabled)
                    throw new NotSupportedException("MLX flat-to-head-first kernel was disabled after initialization failed.");

                flatToHeadFirstKernel = CreateFastMetalKernel(
                    "tensorsharp_flat_to_head_first",
                    new[] { "input" },
                    new[] { "out_y" },
                    FlatToHeadFirstSource,
                    string.Empty);
                if (!flatToHeadFirstKernel.IsValid)
                {
                    flatToHeadFirstKernelDisabled = true;
                    throw new NotSupportedException("Unable to initialize MLX flat-to-head-first kernel.");
                }

                return flatToHeadFirstKernel;
            }
        }

        private static MlxFastMetalKernel EnsureNeoXRopeKernel()
        {
            lock (fastKernelSync)
            {
                if (neoXRopeKernel.IsValid)
                    return neoXRopeKernel;
                if (neoXRopeKernelDisabled)
                    throw new NotSupportedException("MLX NeoX RoPE kernel was disabled after initialization failed.");

                neoXRopeKernel = CreateFastMetalKernel(
                    "tensorsharp_neox_rope",
                    new[] { "input", "cos_table", "sin_table" },
                    new[] { "out_y" },
                    NeoXRopeSource,
                    string.Empty);
                if (!neoXRopeKernel.IsValid)
                {
                    neoXRopeKernelDisabled = true;
                    throw new NotSupportedException("Unable to initialize MLX NeoX RoPE kernel.");
                }

                return neoXRopeKernel;
            }
        }

        private static MlxFastMetalKernel CreateFastMetalKernel(string name, string[] inputs, string[] outputs, string source, string header)
        {
            IntPtr namePtr = IntPtr.Zero;
            IntPtr sourcePtr = IntPtr.Zero;
            IntPtr headerPtr = IntPtr.Zero;
            MlxVectorString inputVector = default;
            MlxVectorString outputVector = default;
            try
            {
                inputVector = CreateStringVector(inputs);
                outputVector = CreateStringVector(outputs);
                namePtr = Marshal.StringToHGlobalAnsi(name);
                sourcePtr = Marshal.StringToHGlobalAnsi(source);
                headerPtr = Marshal.StringToHGlobalAnsi(header ?? string.Empty);
                return mlx_fast_metal_kernel_new(
                    namePtr,
                    inputVector,
                    outputVector,
                    sourcePtr,
                    headerPtr,
                    true,
                    false);
            }
            finally
            {
                if (inputVector.IsValid)
                    _ = mlx_vector_string_free(inputVector);
                if (outputVector.IsValid)
                    _ = mlx_vector_string_free(outputVector);
                Marshal.FreeHGlobal(namePtr);
                Marshal.FreeHGlobal(sourcePtr);
                Marshal.FreeHGlobal(headerPtr);
            }
        }

        private static MlxVectorString CreateStringVector(string[] values)
        {
            MlxVectorString vector = mlx_vector_string_new();
            try
            {
                foreach (string value in values)
                {
                    IntPtr valuePtr = Marshal.StringToHGlobalAnsi(value);
                    try
                    {
                        Check(mlx_vector_string_append_value(vector, valuePtr), "building MLX string vector");
                    }
                    finally
                    {
                        Marshal.FreeHGlobal(valuePtr);
                    }
                }

                MlxVectorString result = vector;
                vector = default;
                return result;
            }
            finally
            {
                if (vector.IsValid)
                    _ = mlx_vector_string_free(vector);
            }
        }

        private static unsafe MlxVectorArray CreateVectorArray(params MlxArray[] values)
        {
            fixed (MlxArray* ptr = values)
            {
                return mlx_vector_array_new_data((IntPtr)ptr, (nuint)values.Length);
            }
        }

        private static void AddTemplateInt(MlxFastMetalKernelConfig config, string name, int value)
        {
            IntPtr namePtr = Marshal.StringToHGlobalAnsi(name);
            try
            {
                Check(mlx_fast_metal_kernel_config_add_template_arg_int(config, namePtr, value), "adding MLX metal template argument");
            }
            finally
            {
                Marshal.FreeHGlobal(namePtr);
            }
        }

        private static ulong QueryMemory(MemoryGetter getter)
        {
            nuint value = 0;
            return getter(ref value) == 0 ? value : 0;
        }

        private static void ConfigureCacheLimit()
        {
            if (cacheLimitConfigured)
                return;

            cacheLimitConfigured = true;
            const long defaultLimitMb = 2048;
            string value = Environment.GetEnvironmentVariable("TS_MLX_CACHE_LIMIT_MB");
            if (string.Equals(value, "0", StringComparison.Ordinal) ||
                string.Equals(value, "off", StringComparison.OrdinalIgnoreCase) ||
                string.Equals(value, "disabled", StringComparison.OrdinalIgnoreCase))
            {
                return;
            }

            long limitMb = defaultLimitMb;
            if (!string.IsNullOrWhiteSpace(value) &&
                (!long.TryParse(value, out limitMb) || limitMb <= 0))
            {
                limitMb = defaultLimitMb;
            }

            nuint previous = 0;
            nuint limitBytes = checked((nuint)limitMb * 1024u * 1024u);
            Check(mlx_set_cache_limit(ref previous, limitBytes), "setting MLX cache limit");

            Console.WriteLine(
                $"  MLX cache limit: {limitBytes / 1024 / 1024} MB " +
                $"(previous {previous / 1024 / 1024} MB; set TS_MLX_CACHE_LIMIT_MB=0 to use MLX default)");
        }

        private static void ConfigureWiredLimit()
        {
            // mlx_set_wired_limit raises Metal's residency-set ceiling so
            // MLX-allocated MTLBuffers remain pinned in physical RAM. Without
            // it, the kernel can evict cold pages from model weights between
            // forward passes, and each subsequent layer page-faults them
            // back from the GGUF file — observed as ~0.3 tok/s on a 25 GB
            // Mac running a 14 GB IQ4 model.
            //
            // Default: wire up to ~min(physRam-2GB, physRam*0.85). macOS
            // rejects oversized requests; we fall back to 75% then 50% on
            // rejection. Opt-in override via TS_MLX_WIRED_LIMIT_MB=<N>.
            string value = Environment.GetEnvironmentVariable("TS_MLX_WIRED_LIMIT_MB");
            long limitMb;
            if (string.Equals(value, "0", StringComparison.Ordinal) ||
                string.Equals(value, "off", StringComparison.OrdinalIgnoreCase) ||
                string.Equals(value, "disabled", StringComparison.OrdinalIgnoreCase))
            {
                return;
            }
            if (string.IsNullOrWhiteSpace(value))
            {
                long phys = GetPhysicalMemoryBytes();
                if (phys <= 0)
                    return;
                long physMb = phys / 1024 / 1024;
                long reserve = Math.Max(2048, physMb / 8);
                limitMb = Math.Max(physMb - reserve, physMb * 85 / 100);
            }
            else if (!long.TryParse(value, out limitMb) || limitMb <= 0)
            {
                return;
            }

            try
            {
                long attempted = limitMb;
                for (int retry = 0; retry < 3; retry++)
                {
                    nuint previous = 0;
                    nuint limitBytes = checked((nuint)attempted * 1024u * 1024u);
                    int rc = mlx_set_wired_limit(ref previous, limitBytes);
                    if (rc == 0)
                    {
                        Console.WriteLine(
                            $"  MLX wired limit: {attempted} MB " +
                            $"(previous {previous / 1024 / 1024} MB; set TS_MLX_WIRED_LIMIT_MB=0 to disable)");
                        return;
                    }
                    _ = TakeCapturedError();
                    attempted = attempted * 3 / 4;
                    if (attempted < 1024)
                        break;
                }
                Console.WriteLine($"  MLX wired limit could not be set (last attempt {attempted} MB rejected); inference may swap under memory pressure.");
            }
            catch (EntryPointNotFoundException)
            {
                // Older mlxc builds may not export mlx_set_wired_limit yet;
                // silently skip rather than failing init.
            }
        }

        private static long GetPhysicalMemoryBytes()
        {
            try
            {
                if (OperatingSystem.IsMacOS() || OperatingSystem.IsLinux())
                {
                    return (long)GC.GetGCMemoryInfo().TotalAvailableMemoryBytes;
                }
            }
            catch
            {
            }
            return 0;
        }

        private static void ConfigureMemoryLimit()
        {
            // mlx_set_memory_limit bounds total active GPU allocations.
            // Default unset (MLX picks its own). Useful on machines where we
            // need a hard cap so other processes get RAM.
            string value = Environment.GetEnvironmentVariable("TS_MLX_MEMORY_LIMIT_MB");
            if (string.IsNullOrWhiteSpace(value))
                return;
            if (!long.TryParse(value, out long limitMb) || limitMb <= 0)
                return;

            nuint previous = 0;
            nuint limitBytes = checked((nuint)limitMb * 1024u * 1024u);
            try
            {
                int rc = mlx_set_memory_limit(ref previous, limitBytes);
                if (rc != 0)
                {
                    string msg = TakeCapturedError();
                    Console.WriteLine($"  MLX memory limit ({limitMb} MB) rejected: {msg}");
                    return;
                }

                Console.WriteLine(
                    $"  MLX memory limit: {limitBytes / 1024 / 1024} MB " +
                    $"(previous {previous / 1024 / 1024} MB)");
            }
            catch (EntryPointNotFoundException)
            {
            }
        }

        private static void ThrowIfUnavailable()
        {
            if (mlx_metal_is_available(out bool metalAvailable) != 0 || !metalAvailable)
                throw new PlatformNotSupportedException("MLX Metal is not available on this machine.");
        }

        private static void Check(int rc, string action)
        {
            if (rc == 0)
            {
                ClearCapturedError();
                return;
            }

            string message = TakeCapturedError();
            if (string.IsNullOrWhiteSpace(message))
                throw new InvalidOperationException($"MLX-C failed while {action} (error code {rc}).");
            throw new InvalidOperationException($"MLX-C failed while {action} (error code {rc}): {message}");
        }

        private static void EnsureErrorHandlerInstalled()
        {
            lock (initSync)
            {
                if (errorHandlerInstalled)
                    return;

                mlx_set_error_handler(ErrorHandler, IntPtr.Zero, IntPtr.Zero);
                errorHandlerInstalled = true;
            }
        }

        private static void CaptureError(IntPtr message, IntPtr data)
        {
            string text = Marshal.PtrToStringAnsi(message) ?? string.Empty;
            lock (errorSync)
                lastError = text;
        }

        private static void ClearCapturedError()
        {
            lock (errorSync)
                lastError = string.Empty;
        }

        private static string TakeCapturedError()
        {
            lock (errorSync)
            {
                string message = lastError;
                lastError = string.Empty;
                return message;
            }
        }

        private static void InstallResolver()
        {
            lock (initSync)
            {
                if (resolverInstalled)
                    return;

                NativeLibrary.SetDllImportResolver(typeof(MlxNative).Assembly, ResolveLibrary);
                resolverInstalled = true;
            }
        }

        private static IntPtr ResolveLibrary(string libraryName, Assembly assembly, DllImportSearchPath? searchPath)
        {
            if (!string.Equals(libraryName, LibraryName, StringComparison.OrdinalIgnoreCase))
                return IntPtr.Zero;

            foreach (string candidate in GetLibraryCandidates())
            {
                if (NativeLibrary.TryLoad(candidate, assembly, searchPath, out IntPtr handle))
                    return handle;
            }

            if (NativeLibrary.TryLoad("libmlxc.dylib", assembly, searchPath, out IntPtr dylib))
                return dylib;
            if (NativeLibrary.TryLoad("libmlxc.so", assembly, searchPath, out IntPtr so))
                return so;
            if (NativeLibrary.TryLoad("mlxc.dll", assembly, searchPath, out IntPtr dll))
                return dll;

            return IntPtr.Zero;
        }

        private static IEnumerable<string> GetLibraryCandidates()
        {
            foreach (string exact in SplitPathList(Environment.GetEnvironmentVariable("TENSORSHARP_MLX_LIBRARY")))
            {
                if (File.Exists(exact))
                    yield return exact;
            }

            foreach (string dir in GetCandidateDirectories())
            {
                foreach (string candidate in EnumerateLibraryFiles(dir))
                    yield return candidate;
            }
        }

        private static IEnumerable<string> GetCandidateDirectories()
        {
            foreach (string dir in SplitPathList(Environment.GetEnvironmentVariable("TENSORSHARP_MLX_LIBRARY_DIR")))
                yield return dir;

            string baseDir = AppContext.BaseDirectory;
            yield return baseDir;
            yield return Path.Combine(baseDir, "lib", "ollama");
            yield return Path.Combine(baseDir, "mlx_metal_v4");
            yield return Path.Combine(baseDir, "mlx_metal_v3");

            string cwd = Environment.CurrentDirectory;
            foreach (string dir in WalkBuildDirectories(cwd))
                yield return dir;
        }

        private static IEnumerable<string> WalkBuildDirectories(string cwd)
        {
            for (string dir = cwd; !string.IsNullOrEmpty(dir); dir = Directory.GetParent(dir)?.FullName)
            {
                string build = Path.Combine(dir, "build");
                yield return Path.Combine(build, "lib", "ollama");

                if (Directory.Exists(build))
                {
                    foreach (string child in Directory.GetDirectories(build, "*", SearchOption.TopDirectoryOnly))
                        yield return Path.Combine(child, "lib", "ollama");
                }
            }
        }

        private static IEnumerable<string> EnumerateLibraryFiles(string dir)
        {
            if (string.IsNullOrWhiteSpace(dir) || !Directory.Exists(dir))
                yield break;

            string[] patterns = OperatingSystem.IsWindows() ? new[] { "mlxc.dll" } : new[] { "libmlxc.dylib", "libmlxc.so", "libmlxc.*" };
            foreach (string pattern in patterns)
            {
                foreach (string file in Directory.EnumerateFiles(dir, pattern, SearchOption.TopDirectoryOnly))
                    yield return file;
            }

            foreach (string subdir in Directory.EnumerateDirectories(dir, "mlx_*", SearchOption.TopDirectoryOnly))
            {
                foreach (string file in EnumerateLibraryFiles(subdir))
                    yield return file;
            }
        }

        private static IEnumerable<string> SplitPathList(string value)
        {
            if (string.IsNullOrWhiteSpace(value))
                yield break;

            foreach (string part in value.Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
                yield return part;
        }

        private delegate int MemoryGetter(ref nuint value);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate void MlxErrorHandler(IntPtr message, IntPtr data);

        private static int ToMlxDtype(DType dtype)
        {
            return dtype switch
            {
                DType.UInt8 => 1,
                DType.Int32 => 7,
                DType.Float16 => 9,
                DType.Float32 => 10,
                DType.Float64 => 11,
                _ => throw new NotSupportedException($"MLX dtype mapping does not support {dtype}."),
            };
        }

        internal readonly struct MlxArray
        {
            public readonly IntPtr Ctx;
            public bool IsValid => Ctx != IntPtr.Zero;
        }

        internal readonly struct MlxStream
        {
            public readonly IntPtr Ctx;
        }

        internal readonly struct MlxClosure
        {
            public readonly IntPtr Ctx;
            public bool IsValid => Ctx != IntPtr.Zero;
        }

        internal readonly struct MlxVectorArray
        {
            public readonly IntPtr Ctx;
            public bool IsValid => Ctx != IntPtr.Zero;
        }

        internal readonly struct MlxVectorString
        {
#pragma warning disable CS0649 // populated by native interop
            public readonly IntPtr Ctx;
#pragma warning restore CS0649
            public bool IsValid => Ctx != IntPtr.Zero;
        }

        internal readonly struct MlxFastMetalKernelConfig
        {
#pragma warning disable CS0649 // populated by native interop
            public readonly IntPtr Ctx;
#pragma warning restore CS0649
            public bool IsValid => Ctx != IntPtr.Zero;
        }

        internal readonly struct MlxFastMetalKernel
        {
#pragma warning disable CS0649 // populated by native interop
            public readonly IntPtr Ctx;
#pragma warning restore CS0649
            public bool IsValid => Ctx != IntPtr.Zero;
        }

        [StructLayout(LayoutKind.Sequential)]
        private readonly struct MlxOptionalFloat
        {
            private readonly float value;
            [MarshalAs(UnmanagedType.I1)]
            private readonly bool has_value;

            private MlxOptionalFloat(float value, bool hasValue)
            {
                this.value = value;
                has_value = hasValue;
            }

            public static MlxOptionalFloat Some(float value) => new(value, true);
            public static MlxOptionalFloat None => new(0f, false);
        }

        [StructLayout(LayoutKind.Sequential)]
        private readonly struct MlxOptionalInt
        {
            private readonly int value;
            [MarshalAs(UnmanagedType.I1)]
            private readonly bool has_value;

            private MlxOptionalInt(int value, bool hasValue)
            {
                this.value = value;
                has_value = hasValue;
            }

            public static MlxOptionalInt Some(int value) => new(value, true);
        }

        [StructLayout(LayoutKind.Sequential)]
        private readonly struct MlxOptionalDType
        {
            private readonly int value;
            [MarshalAs(UnmanagedType.I1)]
            private readonly bool has_value;

            private MlxOptionalDType(int value, bool hasValue)
            {
                this.value = value;
                has_value = hasValue;
            }

            public static MlxOptionalDType Some(int value) => new(value, true);
        }

        [StructLayout(LayoutKind.Sequential)]
        private readonly struct MlxDevice
        {
            public readonly IntPtr Ctx;
        }

        internal enum MlxUnaryOp
        {
            Abs,
            Neg,
            Sqrt,
            Rsqrt,
            Exp,
            Log,
            Log1p,
            Floor,
            Ceil,
            Sin,
            Cos,
            Tanh,
            Sigmoid,
        }

        internal enum MlxBinaryOp
        {
            Add,
            Sub,
            Mul,
            Div,
            Maximum,
        }

        [LibraryImport(LibraryName, EntryPoint = "mlx_metal_is_available")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int mlx_metal_is_available([MarshalAs(UnmanagedType.I1)] out bool res);

        [LibraryImport(LibraryName, EntryPoint = "mlx_set_error_handler")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial void mlx_set_error_handler(MlxErrorHandler handler, IntPtr data, IntPtr destructor);

        [LibraryImport(LibraryName, EntryPoint = "mlx_device_new_type")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial MlxDevice mlx_device_new_type(int type, int index);

        [LibraryImport(LibraryName, EntryPoint = "mlx_device_free")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int mlx_device_free(MlxDevice dev);

        [LibraryImport(LibraryName, EntryPoint = "mlx_device_is_available")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int mlx_device_is_available([MarshalAs(UnmanagedType.I1)] out bool avail, MlxDevice dev);

        [LibraryImport(LibraryName, EntryPoint = "mlx_set_default_device")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int mlx_set_default_device(MlxDevice dev);

        [LibraryImport(LibraryName, EntryPoint = "mlx_clear_cache")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int mlx_clear_cache();

        [LibraryImport(LibraryName, EntryPoint = "mlx_get_active_memory")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int mlx_get_active_memory(ref nuint res);

        [LibraryImport(LibraryName, EntryPoint = "mlx_get_cache_memory")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int mlx_get_cache_memory(ref nuint res);

        [LibraryImport(LibraryName, EntryPoint = "mlx_get_peak_memory")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int mlx_get_peak_memory(ref nuint res);

        [LibraryImport(LibraryName, EntryPoint = "mlx_reset_peak_memory")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int mlx_reset_peak_memory();

        [LibraryImport(LibraryName, EntryPoint = "mlx_set_cache_limit")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int mlx_set_cache_limit(ref nuint previous, nuint limit);

        [LibraryImport(LibraryName, EntryPoint = "mlx_set_wired_limit")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int mlx_set_wired_limit(ref nuint previous, nuint limit);

        [LibraryImport(LibraryName, EntryPoint = "mlx_set_memory_limit")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int mlx_set_memory_limit(ref nuint previous, nuint limit);

        [LibraryImport(LibraryName, EntryPoint = "mlx_get_default_stream")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int mlx_get_default_stream(out MlxStream stream, MlxDevice dev);

        [LibraryImport(LibraryName, EntryPoint = "mlx_stream_free")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int mlx_stream_free(MlxStream stream);

        [LibraryImport(LibraryName, EntryPoint = "mlx_array_new_data")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial MlxArray mlx_array_new_data(IntPtr data, int[] shape, int dim, int dtype);

        [LibraryImport(LibraryName, EntryPoint = "mlx_array_new_data_managed")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial MlxArray mlx_array_new_data_managed(IntPtr data, int[] shape, int dim, int dtype, IntPtr dtor);

        [LibraryImport(LibraryName, EntryPoint = "mlx_array_new_float32")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial MlxArray mlx_array_new_float32(float value);

        [LibraryImport(LibraryName, EntryPoint = "mlx_array_new_int")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial MlxArray mlx_array_new_int(int value);

        [LibraryImport(LibraryName, EntryPoint = "mlx_array_new")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial MlxArray mlx_array_new();

        [LibraryImport(LibraryName, EntryPoint = "mlx_array_set")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int mlx_array_set(ref MlxArray array, MlxArray source);

        [LibraryImport(LibraryName, EntryPoint = "mlx_array_free")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int mlx_array_free(MlxArray array);

        [LibraryImport(LibraryName, EntryPoint = "mlx_astype")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int mlx_astype(out MlxArray result, MlxArray array, int dtype, MlxStream stream);

        [LibraryImport(LibraryName, EntryPoint = "mlx_array_data_float32")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial IntPtr mlx_array_data_float32(MlxArray array);

        [LibraryImport(LibraryName, EntryPoint = "mlx_array_data_float64")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial IntPtr mlx_array_data_float64(MlxArray array);

        [LibraryImport(LibraryName, EntryPoint = "mlx_array_data_float16")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial IntPtr mlx_array_data_float16(MlxArray array);

        [LibraryImport(LibraryName, EntryPoint = "mlx_array_data_int32")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial IntPtr mlx_array_data_int32(MlxArray array);

        [LibraryImport(LibraryName, EntryPoint = "mlx_array_data_uint8")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial IntPtr mlx_array_data_uint8(MlxArray array);

        [LibraryImport(LibraryName, EntryPoint = "mlx_vector_array_new")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial MlxVectorArray mlx_vector_array_new();

        [LibraryImport(LibraryName, EntryPoint = "mlx_vector_array_new_data")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial MlxVectorArray mlx_vector_array_new_data(IntPtr values, nuint size);

        [LibraryImport(LibraryName, EntryPoint = "mlx_vector_array_free")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int mlx_vector_array_free(MlxVectorArray vector);

        [LibraryImport(LibraryName, EntryPoint = "mlx_vector_array_append_value")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int mlx_vector_array_append_value(MlxVectorArray vector, MlxArray value);

        [LibraryImport(LibraryName, EntryPoint = "mlx_vector_array_size")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial nuint mlx_vector_array_size(MlxVectorArray vector);

        [LibraryImport(LibraryName, EntryPoint = "mlx_vector_array_get")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int mlx_vector_array_get(out MlxArray result, MlxVectorArray vector, nuint index);

        [LibraryImport(LibraryName, EntryPoint = "mlx_vector_string_new")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial MlxVectorString mlx_vector_string_new();

        [LibraryImport(LibraryName, EntryPoint = "mlx_vector_string_free")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int mlx_vector_string_free(MlxVectorString vector);

        [LibraryImport(LibraryName, EntryPoint = "mlx_vector_string_append_value")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int mlx_vector_string_append_value(MlxVectorString vector, IntPtr value);

        [LibraryImport(LibraryName, EntryPoint = "mlx_eval")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int mlx_eval(MlxVectorArray outputs);

        [LibraryImport(LibraryName, EntryPoint = "mlx_async_eval")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int mlx_async_eval(MlxVectorArray outputs);

        [LibraryImport(LibraryName, EntryPoint = "mlx_as_strided")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int mlx_as_strided(out MlxArray result, MlxArray array, int[] shape, nuint shapeCount, long[] strides, nuint stridesCount, nuint offset, MlxStream stream);

        [LibraryImport(LibraryName, EntryPoint = "mlx_reshape")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int mlx_reshape(out MlxArray result, MlxArray array, int[] shape, nuint shapeCount, MlxStream stream);

        [LibraryImport(LibraryName, EntryPoint = "mlx_contiguous")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int mlx_contiguous(out MlxArray result, MlxArray array, [MarshalAs(UnmanagedType.I1)] bool allowColMajor, MlxStream stream);

        [LibraryImport(LibraryName, EntryPoint = "mlx_concatenate_axis")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int mlx_concatenate_axis(out MlxArray result, MlxVectorArray arrays, int axis, MlxStream stream);

        [LibraryImport(LibraryName, EntryPoint = "mlx_full")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int mlx_full(out MlxArray result, int[] shape, nuint shapeCount, MlxArray values, int dtype, MlxStream stream);

        [LibraryImport(LibraryName, EntryPoint = "mlx_arange")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int mlx_arange(out MlxArray result, double start, double stop, double step, int dtype, MlxStream stream);

        [LibraryImport(LibraryName, EntryPoint = "mlx_transpose_axes")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int mlx_transpose_axes(out MlxArray result, MlxArray array, int[] axes, nuint axesCount, MlxStream stream);

        [LibraryImport(LibraryName, EntryPoint = "mlx_abs")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int mlx_abs(out MlxArray result, MlxArray input, MlxStream stream);

        [LibraryImport(LibraryName, EntryPoint = "mlx_negative")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int mlx_negative(out MlxArray result, MlxArray input, MlxStream stream);

        [LibraryImport(LibraryName, EntryPoint = "mlx_sqrt")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int mlx_sqrt(out MlxArray result, MlxArray input, MlxStream stream);

        [LibraryImport(LibraryName, EntryPoint = "mlx_rsqrt")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int mlx_rsqrt(out MlxArray result, MlxArray input, MlxStream stream);

        [LibraryImport(LibraryName, EntryPoint = "mlx_exp")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int mlx_exp(out MlxArray result, MlxArray input, MlxStream stream);

        [LibraryImport(LibraryName, EntryPoint = "mlx_log")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int mlx_log(out MlxArray result, MlxArray input, MlxStream stream);

        [LibraryImport(LibraryName, EntryPoint = "mlx_log1p")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int mlx_log1p(out MlxArray result, MlxArray input, MlxStream stream);

        [LibraryImport(LibraryName, EntryPoint = "mlx_floor")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int mlx_floor(out MlxArray result, MlxArray input, MlxStream stream);

        [LibraryImport(LibraryName, EntryPoint = "mlx_ceil")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int mlx_ceil(out MlxArray result, MlxArray input, MlxStream stream);

        [LibraryImport(LibraryName, EntryPoint = "mlx_sin")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int mlx_sin(out MlxArray result, MlxArray input, MlxStream stream);

        [LibraryImport(LibraryName, EntryPoint = "mlx_cos")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int mlx_cos(out MlxArray result, MlxArray input, MlxStream stream);

        [LibraryImport(LibraryName, EntryPoint = "mlx_tanh")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int mlx_tanh(out MlxArray result, MlxArray input, MlxStream stream);

        [LibraryImport(LibraryName, EntryPoint = "mlx_sigmoid")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int mlx_sigmoid(out MlxArray result, MlxArray input, MlxStream stream);

        [LibraryImport(LibraryName, EntryPoint = "mlx_add")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int mlx_add(out MlxArray result, MlxArray lhs, MlxArray rhs, MlxStream stream);

        [LibraryImport(LibraryName, EntryPoint = "mlx_subtract")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int mlx_subtract(out MlxArray result, MlxArray lhs, MlxArray rhs, MlxStream stream);

        [LibraryImport(LibraryName, EntryPoint = "mlx_multiply")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int mlx_multiply(out MlxArray result, MlxArray lhs, MlxArray rhs, MlxStream stream);

        [LibraryImport(LibraryName, EntryPoint = "mlx_divide")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int mlx_divide(out MlxArray result, MlxArray lhs, MlxArray rhs, MlxStream stream);

        [LibraryImport(LibraryName, EntryPoint = "mlx_maximum")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int mlx_maximum(out MlxArray result, MlxArray lhs, MlxArray rhs, MlxStream stream);

        [LibraryImport(LibraryName, EntryPoint = "mlx_remainder")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int mlx_remainder(out MlxArray result, MlxArray lhs, MlxArray rhs, MlxStream stream);

        [LibraryImport(LibraryName, EntryPoint = "mlx_greater")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int mlx_greater(out MlxArray result, MlxArray lhs, MlxArray rhs, MlxStream stream);

        [LibraryImport(LibraryName, EntryPoint = "mlx_where")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int mlx_where(out MlxArray result, MlxArray condition, MlxArray whenTrue, MlxArray whenFalse, MlxStream stream);

        [LibraryImport(LibraryName, EntryPoint = "mlx_matmul")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int mlx_matmul(out MlxArray result, MlxArray a, MlxArray b, MlxStream stream);

        [LibraryImport(LibraryName, EntryPoint = "mlx_addmm")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int mlx_addmm(out MlxArray result, MlxArray src, MlxArray m1, MlxArray m2, float alpha, float beta, MlxStream stream);

        [LibraryImport(LibraryName, EntryPoint = "mlx_softmax_axis")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int mlx_softmax_axis(out MlxArray result, MlxArray input, int axis, [MarshalAs(UnmanagedType.I1)] bool precise, MlxStream stream);

        [LibraryImport(LibraryName, EntryPoint = "mlx_repeat_axis")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int mlx_repeat_axis(out MlxArray result, MlxArray input, int repeats, int axis, MlxStream stream);

        [LibraryImport(LibraryName, EntryPoint = "mlx_fast_layer_norm")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int mlx_fast_layer_norm(out MlxArray result, MlxArray input, MlxArray weight, MlxArray bias, float eps, MlxStream stream);

        [LibraryImport(LibraryName, EntryPoint = "mlx_fast_rms_norm")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int mlx_fast_rms_norm(out MlxArray result, MlxArray input, MlxArray weight, float eps, MlxStream stream);

        [LibraryImport(LibraryName, EntryPoint = "mlx_fast_scaled_dot_product_attention")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int mlx_fast_scaled_dot_product_attention(out MlxArray result, MlxArray query, MlxArray key, MlxArray value, float scale, IntPtr maskMode, MlxArray mask, MlxArray sinks, [MarshalAs(UnmanagedType.I1)] bool forceFused, MlxStream stream);

        [LibraryImport(LibraryName, EntryPoint = "mlx_fast_rope_dynamic")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int mlx_fast_rope_dynamic(out MlxArray result, MlxArray input, int dims, [MarshalAs(UnmanagedType.I1)] bool traditional, MlxOptionalFloat baseValue, float scale, MlxArray offsets, MlxArray freqs, MlxStream stream);

        [LibraryImport(LibraryName, EntryPoint = "mlx_take_axis")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int mlx_take_axis(out MlxArray result, MlxArray input, MlxArray indices, int axis, MlxStream stream);

        [LibraryImport(LibraryName, EntryPoint = "mlx_take_along_axis")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int mlx_take_along_axis(out MlxArray result, MlxArray input, MlxArray indices, int axis, MlxStream stream);

        [LibraryImport(LibraryName, EntryPoint = "mlx_argmax_axis")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int mlx_argmax_axis(out MlxArray result, MlxArray input, int axis, [MarshalAs(UnmanagedType.I1)] bool keepdims, MlxStream stream);

        [LibraryImport(LibraryName, EntryPoint = "mlx_argpartition_axis")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int mlx_argpartition_axis(out MlxArray result, MlxArray input, int kth, int axis, MlxStream stream);

        [LibraryImport(LibraryName, EntryPoint = "mlx_slice_update")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int mlx_slice_update(out MlxArray result, MlxArray input, MlxArray update, int[] starts, nuint startCount, int[] stops, nuint stopCount, int[] strides, nuint strideCount, MlxStream stream);

        [LibraryImport(LibraryName, EntryPoint = "mlx_slice")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int mlx_slice(out MlxArray result, MlxArray input, int[] starts, nuint startCount, int[] stops, nuint stopCount, int[] strides, nuint strideCount, MlxStream stream);

        [LibraryImport(LibraryName, EntryPoint = "mlx_quantized_matmul")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int mlx_quantized_matmul(out MlxArray result, MlxArray input, MlxArray weight, MlxArray scales, MlxArray biases, [MarshalAs(UnmanagedType.I1)] bool transpose, MlxOptionalInt groupSize, MlxOptionalInt bits, IntPtr mode, MlxStream stream);

        [LibraryImport(LibraryName, EntryPoint = "mlx_dequantize")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int mlx_dequantize(out MlxArray result, MlxArray weight, MlxArray scales, MlxArray biases, MlxOptionalInt groupSize, MlxOptionalInt bits, IntPtr mode, MlxArray globalScale, MlxOptionalDType dtype, MlxStream stream);

        [LibraryImport(LibraryName, EntryPoint = "mlx_fast_metal_kernel_config_new")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial MlxFastMetalKernelConfig mlx_fast_metal_kernel_config_new();

        [LibraryImport(LibraryName, EntryPoint = "mlx_fast_metal_kernel_config_free")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int mlx_fast_metal_kernel_config_free(MlxFastMetalKernelConfig config);

        [LibraryImport(LibraryName, EntryPoint = "mlx_fast_metal_kernel_config_add_output_arg")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int mlx_fast_metal_kernel_config_add_output_arg(MlxFastMetalKernelConfig config, int[] shape, nuint size, int dtype);

        [LibraryImport(LibraryName, EntryPoint = "mlx_fast_metal_kernel_config_set_grid")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int mlx_fast_metal_kernel_config_set_grid(MlxFastMetalKernelConfig config, int grid1, int grid2, int grid3);

        [LibraryImport(LibraryName, EntryPoint = "mlx_fast_metal_kernel_config_set_thread_group")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int mlx_fast_metal_kernel_config_set_thread_group(MlxFastMetalKernelConfig config, int thread1, int thread2, int thread3);

        [LibraryImport(LibraryName, EntryPoint = "mlx_fast_metal_kernel_config_add_template_arg_int")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int mlx_fast_metal_kernel_config_add_template_arg_int(MlxFastMetalKernelConfig config, IntPtr name, int value);

        [LibraryImport(LibraryName, EntryPoint = "mlx_fast_metal_kernel_new")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial MlxFastMetalKernel mlx_fast_metal_kernel_new(
            IntPtr name,
            MlxVectorString inputNames,
            MlxVectorString outputNames,
            IntPtr source,
            IntPtr header,
            [MarshalAs(UnmanagedType.I1)] bool ensureRowContiguous,
            [MarshalAs(UnmanagedType.I1)] bool atomicOutputs);

        [LibraryImport(LibraryName, EntryPoint = "mlx_fast_metal_kernel_apply")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int mlx_fast_metal_kernel_apply(ref MlxVectorArray outputs, MlxFastMetalKernel kernel, MlxVectorArray inputs, MlxFastMetalKernelConfig config, MlxStream stream);

        // ----- mlx_compile / mlx_closure_* (graph compilation) -----

        [LibraryImport(LibraryName, EntryPoint = "mlx_compile")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int mlx_compile(out MlxClosure result, MlxClosure fun, [MarshalAs(UnmanagedType.I1)] bool shapeless);

        [LibraryImport(LibraryName, EntryPoint = "mlx_enable_compile")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int mlx_enable_compile();

        [LibraryImport(LibraryName, EntryPoint = "mlx_disable_compile")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int mlx_disable_compile();

        [LibraryImport(LibraryName, EntryPoint = "mlx_set_compile_mode")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int mlx_set_compile_mode(int mode);

        [LibraryImport(LibraryName, EntryPoint = "mlx_closure_new")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial MlxClosure mlx_closure_new();

        [LibraryImport(LibraryName, EntryPoint = "mlx_closure_free")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int mlx_closure_free(MlxClosure closure);

        [LibraryImport(LibraryName, EntryPoint = "mlx_closure_new_func_payload")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial MlxClosure mlx_closure_new_func_payload(IntPtr fun, IntPtr payload, IntPtr destructor);

        [LibraryImport(LibraryName, EntryPoint = "mlx_closure_apply")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int mlx_closure_apply(ref MlxVectorArray result, MlxClosure closure, MlxVectorArray input);

        [LibraryImport(LibraryName, EntryPoint = "mlx_vector_array_set_data")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int mlx_vector_array_set_data(ref MlxVectorArray vec, IntPtr data, nuint size);

        [LibraryImport(LibraryName, EntryPoint = "mlx_array_dtype")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int mlx_array_dtype(out int dtype, MlxArray array);

        // ----- mlx_gather_mm / mlx_gather_qmm (MoE-friendly batched matmul) -----

        [LibraryImport(LibraryName, EntryPoint = "mlx_gather_mm")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int mlx_gather_mm(out MlxArray result, MlxArray a, MlxArray b, MlxArray lhsIndices, MlxArray rhsIndices, [MarshalAs(UnmanagedType.I1)] bool sortedIndices, MlxStream stream);

        [LibraryImport(LibraryName, EntryPoint = "mlx_gather_qmm")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int mlx_gather_qmm(out MlxArray result, MlxArray x, MlxArray w, MlxArray scales, MlxArray biases, MlxArray lhsIndices, MlxArray rhsIndices, [MarshalAs(UnmanagedType.I1)] bool transpose, MlxOptionalInt groupSize, MlxOptionalInt bits, IntPtr mode, [MarshalAs(UnmanagedType.I1)] bool sortedIndices, MlxStream stream);
    }
}
