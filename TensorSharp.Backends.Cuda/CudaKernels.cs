using System;
using System.Collections.Generic;
using System.IO;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    internal sealed unsafe class CudaKernels : IDisposable
    {
        private const int BlockSize = 256;

        // The GatedDeltaNet kernel assigns one warp per delta-net row, so a wider
        // block processes more rows concurrently (head_v_dim / num_warps passes).
        // 512 threads = 16 warps keeps occupancy high without exhausting shared mem.
        private const int GdnBlockSize = 512;

        private readonly CudaModule module;
        private readonly IntPtr copy2DBytes;
        private readonly IntPtr fillF32;
        private readonly IntPtr fillF16;
        private readonly IntPtr unaryF32;
        private readonly IntPtr binaryF32;
        private readonly IntPtr scalarF32;
        private readonly IntPtr ternaryF32;
        private readonly IntPtr addMulScalarF32;
        private readonly IntPtr mulMulAddF32;
        private readonly IntPtr binaryActivationF32;
        private readonly IntPtr binaryActivationStridedF32;
        private readonly IntPtr deinterleaveQGateF32;
        private readonly IntPtr addBiasRowsF32;
        private readonly IntPtr siluMulSplitF32;
        private readonly IntPtr geluMulSplitF32;
        private readonly IntPtr swigluOaiSplitF32;
        private readonly IntPtr qwen35GdnPackedF32;
        private readonly IntPtr qwen35GdnPrefillConvF32;
        private readonly IntPtr qwen35GdnPrefillScanF32;
        private readonly IntPtr qwen35GdnPrefillOutF32;
        private readonly IntPtr qwen35GdnUpdateConvStateF32;
        private readonly IntPtr qwen35GdnPackInputsF32;
        private readonly IntPtr rmsNormF32;
        private readonly IntPtr rmsNormResidualAddF32;
        private readonly IntPtr softmaxF32;
        private readonly IntPtr attentionSoftmaxSinksF32;
        private readonly IntPtr scaledDotProductAttentionF32;
        private readonly IntPtr gqaPrefillAttentionF32;
        private readonly IntPtr gqaPrefillAttentionF16;
        private readonly IntPtr gqaPrefillAttentionGroup4D256F32;
        private readonly IntPtr gqaPrefillAttentionGroup4D256F16;
        private readonly IntPtr gqaPrefillAttentionGroup4D512F32;
        private readonly IntPtr gqaPrefillAttentionGroup4D512F16;
        private readonly IntPtr gqaPrefillAttentionGroup4OnlineD512F32;
        private readonly IntPtr gqaPrefillAttentionGroup4OnlineD512F16;
        private readonly IntPtr gqaPrefillFlashGroup4D256F16;
        private readonly IntPtr gqaPrefillFlashGroup4D512F16;
        private readonly IntPtr gqaPrefillFlashGroup4D256F32;
        private readonly IntPtr gqaPrefillFlashGroup4D512F32;
        private readonly IntPtr gqaPrefillFlash2Group4D256F16;
        private readonly IntPtr gqaPrefillFlash2Group4D512F16;
        private readonly IntPtr gqaPrefillFlash2Group4D256F32;
        private readonly IntPtr gqaPrefillFlash2Group4D512F32;
        private uint flash2SharedCapacityD256 = 48 * 1024;
        private uint flash2SharedCapacityD512 = 48 * 1024;
        private readonly IntPtr gqaPrefillAttentionSinksF32;
        private readonly IntPtr gqaPrefillAttentionSinksF16;
        private readonly IntPtr gqaDecodeAttentionF32;
        private readonly IntPtr gqaDecodeAttentionF16;
        private readonly IntPtr gqaDecodeAttentionGroup4D256F16;
        private readonly IntPtr gqaDecodeAttentionGroup4D512F16;
        private readonly IntPtr gqaDecodeAttentionPartitionGroup4D256F16;
        private readonly IntPtr gqaDecodeAttentionSinksF32;
        private readonly IntPtr gqaDecodeAttentionSinksF16;
        private readonly IntPtr gqaDecodeAttentionPartitionF32;
        private readonly IntPtr gqaDecodeAttentionPartitionF16;
        private readonly IntPtr gqaDecodeAttentionPartitionGroup4D512F16;
        private readonly IntPtr gqaDecodeAttentionPartitionReduceF32;
        private readonly IntPtr sliceColumnsF32;
        private readonly IntPtr flatToHeadFirstF32;
        private readonly IntPtr splitQkvHeadFirstF32;
        private readonly IntPtr copyHeadFirstToCacheF32;
        private readonly IntPtr fillRopePositionsI32;
        private readonly IntPtr copyHeadFirstToCacheF16;
        private readonly IntPtr gatherCircularHeadFirstF32;
        private readonly IntPtr gatherCircularHeadFirstF16;
        private readonly IntPtr expandKvHeadsF32;
        private readonly IntPtr expandKvHeadsF16;
        private readonly IntPtr repeatInterleaveF32;
        private readonly IntPtr repeatInterleaveF16;
        private readonly IntPtr concatHeadFirstF32;
        private readonly IntPtr neoxRopeHeadFirstF32;
        private readonly IntPtr neoxRopeFlatF32;
        private readonly IntPtr fillNeoXRopeTablesDynamicF32;
        private readonly IntPtr indexSelectF32;
        private readonly IntPtr addCausalMaskF32;
        private readonly IntPtr ropeF32;
        private readonly IntPtr ropeExF32;
        private readonly IntPtr quantMatmulF32;
        private readonly IntPtr quantMatmulBatchedF32;
        private readonly IntPtr quantMatmulVecF32;
        private readonly IntPtr moeRouterF32;
        private readonly IntPtr moeExpertGateUpVecF32;
        private readonly IntPtr moeExpertDownAccumF32;
        private readonly IntPtr moeExpertGateUpDp4aF32;
        private readonly IntPtr moeExpertDownDp4aF32;
        private readonly IntPtr siluMulF32;
        private readonly IntPtr moeSharedGatedAddF32;
        private readonly IntPtr moeRouterBatchedF32;
        private readonly IntPtr moeExpertGateUpBatchedDp4aF32;
        private readonly IntPtr moeExpertGateUpBatchedVecF32;
        private readonly IntPtr moeExpertDownBatchedDp4aF32;
        private readonly IntPtr moeExpertDownBatchedAccumF32;
        private readonly IntPtr moeSharedGatedAddBatchedF32;
        private readonly IntPtr moeScatterAddWeightedRowsF32;
        private readonly IntPtr quantMatmulIq2XxsQ81F32;
        private readonly IntPtr quantMatmulIq2VecQ81F32;
        private readonly IntPtr quantMatmulQ40F32;
        private readonly IntPtr quantMatmulQ40BatchedF32;
        private readonly IntPtr quantMatmulQ40Dp4aF32;
        private readonly IntPtr quantMatmulQ80SingleF32;
        private readonly IntPtr quantMatmulQ80VecF32;
        private readonly IntPtr quantMatmulQ4KDp4aF32;
        private readonly IntPtr quantMatmulQ5KDp4aF32;
        private readonly IntPtr quantMatmulQ6KDp4aF32;
        private readonly IntPtr quantMatmulQ80MmqF32;
        private readonly IntPtr quantMatmulQ80Mmq2F32;
        private readonly IntPtr quantizeQ81SplitRowsF32;
        private readonly IntPtr quantMatmulQ80F32;
        private readonly IntPtr quantMatmulQ80Dp4aF32;
        private readonly IntPtr quantMatmulQ80MmaF32;
        private readonly IntPtr quantizeQ81RowsF32;
        private readonly IntPtr quantizeQ81RowsWarpF32;
        private readonly IntPtr dequantWeightF16;
        private readonly IntPtr dequantWeightQ80F16;
        private readonly IntPtr convertF32F16;
        private readonly IntPtr qkNormRopeNeoxF32;
        private readonly IntPtr qwen35GdnFusedF32;

        // q8_1 block = half d + half s + 32 int8 = 36 bytes (matches ts_block_q8_1).
        public const int Q81BlockBytes = 36;
        public const int Q8Dp4aTileRows = 4;   // matches TS_Q8_DP4A_ROWS in the kernel
        public const int Q40Dp4aTileRows = 4;  // matches TS_Q40_DP4A_ROWS in the kernel
        // Multi-row Q8_0 uses the block-tile dp4a kernel by default;
        // TS_CUDA_Q8_DP4A=0 reverts to the scalar 4-row block-reduce kernel (A/B).
        public static readonly bool Q8Dp4aEnabled =
            !string.Equals(Environment.GetEnvironmentVariable("TS_CUDA_Q8_DP4A"), "0", StringComparison.Ordinal);
        // Opt-in tensor-core (wmma int8 MMA) multi-row Q8_0 path: TS_CUDA_Q8_MMA=1.
        public static readonly bool Q8MmaEnabled =
            string.Equals(Environment.GetEnvironmentVariable("TS_CUDA_Q8_MMA"), "1", StringComparison.Ordinal);
        // The GQA prefill QK phase maps one warp to each key for coalesced Q/K loads.
        // TS_CUDA_GQA_PREFILL_WARP=0 keeps the legacy one-thread-per-key path for A/B.
        public static readonly bool GqaPrefillWarpCooperativeEnabled =
            !string.Equals(Environment.GetEnvironmentVariable("TS_CUDA_GQA_PREFILL_WARP"), "0", StringComparison.Ordinal);
        // Gemma 4 local attention has four Q heads per KV head, d=256, and a
        // bounded sliding window.  Share each K/V load across the four heads.
        // TS_CUDA_GQA_PREFILL_GROUP4=0 keeps the generic warp kernel for A/B.
        public static readonly bool GqaPrefillGroup4Enabled =
            !string.Equals(Environment.GetEnvironmentVariable("TS_CUDA_GQA_PREFILL_GROUP4"), "0", StringComparison.Ordinal);
        // Flash-style tiled prefill attention (f16 K/V, group4): stages the Q
        // tile in shared memory once and walks K/V with an online softmax, so
        // K/V traffic is amortized over 4x-8x more score rows than the two-pass
        // group4 kernels. TS_CUDA_FLASH_PREFILL=0 keeps the two-pass kernels
        // for A/B.
        public static readonly bool GqaPrefillFlashEnabled =
            !string.Equals(Environment.GetEnvironmentVariable("TS_CUDA_FLASH_PREFILL"), "0", StringComparison.Ordinal);
        // flash2: thread-per-score QK (no warp-shuffle reduction, transposed-K
        // shared staging). Default on; TS_CUDA_FLASH2=0 reverts to flash1 for A/B.
        public static readonly bool GqaPrefillFlash2Enabled =
            !string.Equals(Environment.GetEnvironmentVariable("TS_CUDA_FLASH2"), "0", StringComparison.Ordinal);
        // Gemma 4 local/global decode has four d=256/d=512 Q heads sharing each
        // F16 KV head. Share coalesced K/V reads across the group (ggml-style GQA).
        public static readonly bool GqaDecodeGroup4Enabled =
            !string.Equals(Environment.GetEnvironmentVariable("TS_CUDA_GQA_DECODE_GROUP4"), "0", StringComparison.Ordinal);
        private readonly IntPtr quantGetRowsF32;

        private CudaKernels(CudaModule module)
        {
            this.module = module;
            copy2DBytes = module.GetFunction("ts_copy2d_bytes");
            fillF32 = module.GetFunction("ts_fill_f32");
            fillF16 = module.GetFunction("ts_fill_f16");
            unaryF32 = module.GetFunction("ts_unary_f32");
            binaryF32 = module.GetFunction("ts_binary_f32");
            scalarF32 = module.GetFunction("ts_scalar_f32");
            ternaryF32 = module.GetFunction("ts_ternary_f32");
            addMulScalarF32 = module.GetFunction("ts_addmul_scalar_f32");
            mulMulAddF32 = module.GetFunction("ts_mulmuladd_f32");
            binaryActivationF32 = module.GetFunction("ts_binary_activation_f32");
            binaryActivationStridedF32 = module.GetFunction("ts_binary_activation_strided_f32");
            deinterleaveQGateF32 = module.GetFunction("ts_deinterleave_qgate_f32");
            addBiasRowsF32 = module.GetFunction("ts_add_bias_rows_f32");
            siluMulSplitF32 = module.GetFunction("ts_silu_mul_split_f32");
            geluMulSplitF32 = module.GetFunction("ts_gelu_mul_split_f32");
            swigluOaiSplitF32 = module.GetFunction("ts_swiglu_oai_split_f32");
            qwen35GdnPackedF32 = module.GetFunction("ts_qwen35_gdn_packed_f32");
            qwen35GdnPrefillConvF32 = module.GetFunction("ts_qwen35_gdn_prefill_conv_f32");
            qwen35GdnPrefillScanF32 = module.GetFunction("ts_qwen35_gdn_prefill_scan_f32");
            qwen35GdnPrefillOutF32 = module.GetFunction("ts_qwen35_gdn_prefill_out_f32");
            qwen35GdnUpdateConvStateF32 = module.GetFunction("ts_qwen35_gdn_update_conv_state_f32");
            qwen35GdnPackInputsF32 = module.GetFunction("ts_qwen35_gdn_pack_inputs_f32");
            rmsNormF32 = module.GetFunction("ts_rmsnorm_f32");
            rmsNormResidualAddF32 = module.GetFunction("ts_rmsnorm_residual_add_f32");
            softmaxF32 = module.GetFunction("ts_softmax_f32");
            attentionSoftmaxSinksF32 = module.GetFunction("ts_attention_softmax_sinks_f32");
            scaledDotProductAttentionF32 = module.GetFunction("ts_scaled_dot_product_attention_f32");
            gqaPrefillAttentionF32 = module.GetFunction("ts_gqa_prefill_attention_f32");
            gqaPrefillAttentionF16 = module.GetFunction("ts_gqa_prefill_attention_f16");
            gqaPrefillAttentionGroup4D256F32 = module.GetFunction("ts_gqa_prefill_attention_group4_d256_f32");
            gqaPrefillAttentionGroup4D256F16 = module.GetFunction("ts_gqa_prefill_attention_group4_d256_f16");
            gqaPrefillAttentionGroup4D512F32 = module.GetFunction("ts_gqa_prefill_attention_group4_d512_f32");
            gqaPrefillAttentionGroup4D512F16 = module.GetFunction("ts_gqa_prefill_attention_group4_d512_f16");
            gqaPrefillAttentionGroup4OnlineD512F32 =
                module.GetFunction("ts_gqa_prefill_attention_group4_online_d512_f32");
            gqaPrefillAttentionGroup4OnlineD512F16 =
                module.GetFunction("ts_gqa_prefill_attention_group4_online_d512_f16");
            gqaPrefillFlashGroup4D256F16 =
                module.GetFunction("ts_gqa_prefill_flash_group4_d256_f16");
            gqaPrefillFlashGroup4D512F16 =
                module.GetFunction("ts_gqa_prefill_flash_group4_d512_f16");
            gqaPrefillFlashGroup4D256F32 =
                module.GetFunction("ts_gqa_prefill_flash_group4_d256_f32");
            gqaPrefillFlashGroup4D512F32 =
                module.GetFunction("ts_gqa_prefill_flash_group4_d512_f32");
            gqaPrefillFlash2Group4D256F16 =
                module.GetFunction("ts_gqa_prefill_flash2_group4_d256_f16");
            gqaPrefillFlash2Group4D512F16 =
                module.GetFunction("ts_gqa_prefill_flash2_group4_d512_f16");
            gqaPrefillFlash2Group4D256F32 =
                module.GetFunction("ts_gqa_prefill_flash2_group4_d256_f32");
            gqaPrefillFlash2Group4D512F32 =
                module.GetFunction("ts_gqa_prefill_flash2_group4_d512_f32");
            gqaPrefillAttentionSinksF32 = module.GetFunction("ts_gqa_prefill_attention_sinks_f32");
            gqaPrefillAttentionSinksF16 = module.GetFunction("ts_gqa_prefill_attention_sinks_f16");
            gqaDecodeAttentionF32 = module.GetFunction("ts_gqa_decode_attention_f32");
            gqaDecodeAttentionF16 = module.GetFunction("ts_gqa_decode_attention_f16");
            gqaDecodeAttentionGroup4D256F16 = module.GetFunction("ts_gqa_decode_attention_group4_d256_f16");
            gqaDecodeAttentionGroup4D512F16 = module.GetFunction("ts_gqa_decode_attention_group4_d512_f16");
            gqaDecodeAttentionPartitionGroup4D256F16 = module.GetFunction("ts_gqa_decode_attention_partition_group4_d256_f16");
            gqaDecodeAttentionSinksF32 = module.GetFunction("ts_gqa_decode_attention_sinks_f32");
            gqaDecodeAttentionSinksF16 = module.GetFunction("ts_gqa_decode_attention_sinks_f16");
            gqaDecodeAttentionPartitionF32 = module.GetFunction("ts_gqa_decode_attention_partition_f32");
            gqaDecodeAttentionPartitionF16 = module.GetFunction("ts_gqa_decode_attention_partition_f16");
            gqaDecodeAttentionPartitionGroup4D512F16 =
                module.GetFunction("ts_gqa_decode_attention_partition_group4_d512_f16");
            gqaDecodeAttentionPartitionReduceF32 = module.GetFunction("ts_gqa_decode_attention_partition_reduce_f32");
            sliceColumnsF32 = module.GetFunction("ts_slice_columns_f32");
            flatToHeadFirstF32 = module.GetFunction("ts_flat_to_head_first_f32");
            splitQkvHeadFirstF32 = module.GetFunction("ts_split_qkv_head_first_f32");
            copyHeadFirstToCacheF32 = module.GetFunction("ts_copy_head_first_to_cache_f32");
            copyHeadFirstToCacheF16 = module.GetFunction("ts_copy_head_first_to_cache_f16");
            fillRopePositionsI32 = module.GetFunction("ts_fill_rope_positions_i32");
            gatherCircularHeadFirstF32 = module.GetFunction("ts_gather_circular_head_first_f32");
            gatherCircularHeadFirstF16 = module.GetFunction("ts_gather_circular_head_first_f16");
            expandKvHeadsF32 = module.GetFunction("ts_expand_kv_heads_f32");
            expandKvHeadsF16 = module.GetFunction("ts_expand_kv_heads_f16");
            repeatInterleaveF32 = module.GetFunction("ts_repeat_interleave_f32");
            repeatInterleaveF16 = module.GetFunction("ts_repeat_interleave_f16");
            concatHeadFirstF32 = module.GetFunction("ts_concat_head_first_f32");
            neoxRopeHeadFirstF32 = module.GetFunction("ts_neox_rope_head_first_f32");
            neoxRopeFlatF32 = module.GetFunction("ts_neox_rope_flat_f32");
            fillNeoXRopeTablesDynamicF32 = module.GetFunction("ts_fill_neox_rope_tables_dyn_f32");
            indexSelectF32 = module.GetFunction("ts_index_select_f32");
            addCausalMaskF32 = module.GetFunction("ts_add_causal_mask_f32");
            ropeF32 = module.GetFunction("ts_rope_f32");
            ropeExF32 = module.GetFunction("ts_rope_ex_f32");
            quantMatmulF32 = module.GetFunction("ts_quant_matmul_f32");
            quantMatmulBatchedF32 = module.GetFunction("ts_quant_matmul_batched_f32");
            quantMatmulVecF32 = module.GetFunction("ts_quant_matmul_vec_f32");
            moeRouterF32 = module.GetFunction("ts_moe_router_f32");
            moeExpertGateUpVecF32 = module.GetFunction("ts_moe_expert_gate_up_vec_f32");
            moeExpertDownAccumF32 = module.GetFunction("ts_moe_expert_down_accum_f32");
            moeExpertGateUpDp4aF32 = module.GetFunction("ts_moe_expert_gate_up_dp4a_f32");
            moeExpertDownDp4aF32 = module.GetFunction("ts_moe_expert_down_dp4a_f32");
            siluMulF32 = module.GetFunction("ts_silu_mul_f32");
            moeSharedGatedAddF32 = module.GetFunction("ts_moe_shared_gated_add");
            moeRouterBatchedF32 = module.GetFunction("ts_moe_router_batched_f32");
            moeExpertGateUpBatchedDp4aF32 = module.GetFunction("ts_moe_expert_gate_up_batched_dp4a_f32");
            moeExpertGateUpBatchedVecF32 = module.GetFunction("ts_moe_expert_gate_up_batched_vec_f32");
            moeExpertDownBatchedDp4aF32 = module.GetFunction("ts_moe_expert_down_batched_dp4a_f32");
            moeExpertDownBatchedAccumF32 = module.GetFunction("ts_moe_expert_down_batched_accum_f32");
            moeSharedGatedAddBatchedF32 = module.GetFunction("ts_moe_shared_gated_add_batched");
            moeScatterAddWeightedRowsF32 = module.GetFunction("ts_moe_scatter_add_weighted_rows_f32");
            quantMatmulIq2XxsQ81F32 = module.GetFunction("ts_quant_matmul_iq2_xxs_q8_1_f32");
            quantMatmulIq2VecQ81F32 = module.GetFunction("ts_quant_matmul_iq2_vec_q8_1_f32");
            quantMatmulQ40F32 = module.GetFunction("ts_quant_matmul_q4_0_f32");
            quantMatmulQ40BatchedF32 = module.GetFunction("ts_quant_matmul_q4_0_batched_f32");
            quantMatmulQ40Dp4aF32 = module.GetFunction("ts_quant_matmul_q4_0_dp4a_f32");
            quantMatmulQ80SingleF32 = module.GetFunction("ts_quant_matmul_q8_0_single_f32");
            quantMatmulQ80VecF32 = module.GetFunction("ts_quant_matmul_q8_0_vec_f32");
            quantMatmulQ4KDp4aF32 = module.GetFunction("ts_quant_matmul_q4k_dp4a_f32");
            quantMatmulQ5KDp4aF32 = module.GetFunction("ts_quant_matmul_q5k_dp4a_f32");
            quantMatmulQ6KDp4aF32 = module.GetFunction("ts_quant_matmul_q6k_dp4a_f32");
            quantMatmulQ80MmqF32 = module.GetFunction("ts_quant_matmul_q8_0_mmq_f32");
            quantMatmulQ80Mmq2F32 = module.GetFunction("ts_quant_matmul_q8_0_mmq2_f32");
            quantMatmulQ80F32 = module.GetFunction("ts_quant_matmul_q8_0_f32");
            quantMatmulQ80Dp4aF32 = module.GetFunction("ts_quant_matmul_q8_0_dp4a_f32");
            quantMatmulQ80MmaF32 = module.GetFunction("ts_quant_matmul_q8_0_mma_f32");
            quantizeQ81RowsF32 = module.GetFunction("ts_quantize_q8_1_rows_f32");
            quantizeQ81RowsWarpF32 = module.GetFunction("ts_quantize_q8_1_rows_warp_f32");
            quantizeQ81SplitRowsF32 = module.GetFunction("ts_quantize_q8_1_split_rows_f32");
            dequantWeightF16 = module.GetFunction("ts_dequant_weight_f16");
            dequantWeightQ80F16 = module.GetFunction("ts_dequant_weight_q8_0_f16");
            convertF32F16 = module.GetFunction("ts_convert_f32_f16");
            quantGetRowsF32 = module.GetFunction("ts_quant_get_rows_f32");
            qkNormRopeNeoxF32 = module.GetFunction("ts_qk_norm_rope_neox_f32");
            qwen35GdnFusedF32 = module.GetFunction("ts_qwen35_gdn_fused_f32");
        }

        public static CudaKernels TryCreate()
        {
            string path = LocatePtxPath();
            if (path == null)
            {
                WarnKernelsUnavailable(
                    "the compiled PTX (tensorsharp_kernels.ptx) could not be located next to the application " +
                    "or under a 'cuda_kernels' folder in its output directory");
                return null;
            }

            try
            {
                return new CudaKernels(CudaModule.LoadFromFile(path));
            }
            catch (Exception ex)
            {
                WarnKernelsUnavailable($"loading the PTX module from '{path}' failed: {ex.Message}");
                return null;
            }
        }

        private static int kernelWarningEmitted;

        private static void WarnKernelsUnavailable(string reason)
        {
            // Without the PTX kernels every PTX-backed op (fill, attention, RoPE, ...)
            // silently falls back to the slow CPU path, and Float16 caches in
            // particular used to surface this only as a deep NotSupportedException.
            // Emit a single, actionable warning so the degraded mode is visible.
            if (System.Threading.Interlocked.Exchange(ref kernelWarningEmitted, 1) != 0)
                return;

            try
            {
                Console.Error.WriteLine(
                    "WARNING: TensorSharp CUDA kernels are unavailable because " + reason + ". " +
                    "The Cuda backend will run PTX-backed ops on a slow CPU fallback. " +
                    "Build TensorSharp.Backends.Cuda with nvcc on the PATH and ensure " +
                    "cuda_kernels/tensorsharp_kernels.ptx is copied next to your application.");
            }
            catch
            {
                // Diagnostics must never break model initialization.
            }
        }

        public void LaunchCopy2DBytes(
            IntPtr src, IntPtr dst, long rows, long innerBytes, long srcPitch, long dstPitch, IntPtr stream)
        {
            IntPtr srcArg = src;
            IntPtr dstArg = dst;
            long rowsArg = rows;
            long innerArg = innerBytes;
            long srcPitchArg = srcPitch;
            long dstPitchArg = dstPitch;
            void** args = stackalloc void*[] { &srcArg, &dstArg, &rowsArg, &innerArg, &srcPitchArg, &dstPitchArg };
            // Rows spread over grid.y (kernel row-loops past 65535); grid.x
            // covers the row bytes with a strided per-thread loop.
            uint gx = (uint)Math.Min(Math.Max(1, (innerBytes / 16 + BlockSize - 1) / BlockSize), 8);
            uint gy = (uint)Math.Min(rows, 65535);
            Launch(copy2DBytes, gx, gy, 1, BlockSize, 1, 1, 0, stream, args);
        }

        public void LaunchFillF32(IntPtr output, int count, float value, IntPtr stream)
        {
            IntPtr outArg = output;
            int countArg = count;
            float valueArg = value;
            void** args = stackalloc void*[] { &outArg, &countArg, &valueArg };
            Launch(fillF32, Grid(count), 1, 1, BlockSize, 1, 1, 0, stream, args);
        }

        public void LaunchFillF16(IntPtr output, int count, float value, IntPtr stream)
        {
            IntPtr outArg = output;
            int countArg = count;
            float valueArg = value;
            void** args = stackalloc void*[] { &outArg, &countArg, &valueArg };
            Launch(fillF16, Grid(count), 1, 1, BlockSize, 1, 1, 0, stream, args);
        }

        public void LaunchUnaryF32(IntPtr input, IntPtr output, int count, int op, IntPtr stream)
        {
            IntPtr inputArg = input;
            IntPtr outputArg = output;
            int countArg = count;
            int opArg = op;
            void** args = stackalloc void*[] { &inputArg, &outputArg, &countArg, &opArg };
            Launch(unaryF32, Grid(count), 1, 1, BlockSize, 1, 1, 0, stream, args);
        }

        public void LaunchBinaryF32(IntPtr lhs, IntPtr rhs, IntPtr output, int count, int op, IntPtr stream)
        {
            IntPtr lhsArg = lhs;
            IntPtr rhsArg = rhs;
            IntPtr outputArg = output;
            int countArg = count;
            int opArg = op;
            void** args = stackalloc void*[] { &lhsArg, &rhsArg, &outputArg, &countArg, &opArg };
            Launch(binaryF32, Grid(count), 1, 1, BlockSize, 1, 1, 0, stream, args);
        }

        public void LaunchScalarF32(IntPtr input, IntPtr output, int count, float value, int op, IntPtr stream)
        {
            IntPtr inputArg = input;
            IntPtr outputArg = output;
            int countArg = count;
            float valueArg = value;
            int opArg = op;
            void** args = stackalloc void*[] { &inputArg, &outputArg, &countArg, &valueArg, &opArg };
            Launch(scalarF32, Grid(count), 1, 1, BlockSize, 1, 1, 0, stream, args);
        }

        public void LaunchTernaryF32(IntPtr x, IntPtr y, IntPtr z, IntPtr output, int count, int op, IntPtr stream)
        {
            IntPtr xArg = x;
            IntPtr yArg = y;
            IntPtr zArg = z;
            IntPtr outputArg = output;
            int countArg = count;
            int opArg = op;
            void** args = stackalloc void*[] { &xArg, &yArg, &zArg, &outputArg, &countArg, &opArg };
            Launch(ternaryF32, Grid(count), 1, 1, BlockSize, 1, 1, 0, stream, args);
        }

        public void LaunchAddMulScalarF32(IntPtr x, IntPtr y, IntPtr output, int count, float value, IntPtr stream)
        {
            IntPtr xArg = x;
            IntPtr yArg = y;
            IntPtr outputArg = output;
            int countArg = count;
            float valueArg = value;
            void** args = stackalloc void*[] { &xArg, &yArg, &outputArg, &countArg, &valueArg };
            Launch(addMulScalarF32, Grid(count), 1, 1, BlockSize, 1, 1, 0, stream, args);
        }

        public void LaunchMulMulAddF32(IntPtr x, IntPtr y, IntPtr z, IntPtr w, IntPtr output, int count, IntPtr stream)
        {
            IntPtr xArg = x;
            IntPtr yArg = y;
            IntPtr zArg = z;
            IntPtr wArg = w;
            IntPtr outputArg = output;
            int countArg = count;
            void** args = stackalloc void*[] { &xArg, &yArg, &zArg, &wArg, &outputArg, &countArg };
            Launch(mulMulAddF32, Grid(count), 1, 1, BlockSize, 1, 1, 0, stream, args);
        }

        public void LaunchBinaryActivationF32(IntPtr a, IntPtr b, IntPtr output, int count, int op, IntPtr stream)
        {
            IntPtr aArg = a;
            IntPtr bArg = b;
            IntPtr outputArg = output;
            int countArg = count;
            int opArg = op;
            void** args = stackalloc void*[] { &aArg, &bArg, &outputArg, &countArg, &opArg };
            Launch(binaryActivationF32, Grid(count), 1, 1, BlockSize, 1, 1, 0, stream, args);
        }

        public void LaunchBinaryActivationStridedF32(
            IntPtr a, IntPtr b, IntPtr output,
            int rows, int cols,
            long aRowStride, long bRowStride, long outRowStride,
            int op, IntPtr stream)
        {
            IntPtr aArg = a;
            IntPtr bArg = b;
            IntPtr outputArg = output;
            int rowsArg = rows;
            int colsArg = cols;
            long aStrideArg = aRowStride;
            long bStrideArg = bRowStride;
            long outStrideArg = outRowStride;
            int opArg = op;
            int count = checked(rows * cols);
            void** args = stackalloc void*[] { &aArg, &bArg, &outputArg, &rowsArg, &colsArg, &aStrideArg, &bStrideArg, &outStrideArg, &opArg };
            Launch(binaryActivationStridedF32, Grid(count), 1, 1, BlockSize, 1, 1, 0, stream, args);
        }

        public void LaunchDeinterleaveQGateF32(
            IntPtr src, IntPtr q, IntPtr gate,
            int rows, int numHeads, int headDim, long srcRowStride, IntPtr stream)
        {
            IntPtr srcArg = src;
            IntPtr qArg = q;
            IntPtr gateArg = gate;
            int rowsArg = rows;
            int numHeadsArg = numHeads;
            int headDimArg = headDim;
            long srcStrideArg = srcRowStride;
            int count = checked(rows * numHeads * headDim);
            void** args = stackalloc void*[] { &srcArg, &qArg, &gateArg, &rowsArg, &numHeadsArg, &headDimArg, &srcStrideArg };
            Launch(deinterleaveQGateF32, Grid(count), 1, 1, BlockSize, 1, 1, 0, stream, args);
        }

        public void LaunchAddBiasRowsF32(IntPtr tensor, IntPtr bias, int rows, int cols, int biasCols, IntPtr stream)
        {
            IntPtr tensorArg = tensor;
            IntPtr biasArg = bias;
            int rowsArg = rows;
            int colsArg = cols;
            int biasColsArg = biasCols;
            int count = checked(rows * cols);
            void** args = stackalloc void*[] { &tensorArg, &biasArg, &rowsArg, &colsArg, &biasColsArg };
            Launch(addBiasRowsF32, Grid(count), 1, 1, BlockSize, 1, 1, 0, stream, args);
        }

        public void LaunchSiLUMulSplitF32(IntPtr gateUp, IntPtr output, int rows, int halfDim, IntPtr stream)
        {
            IntPtr gateUpArg = gateUp;
            IntPtr outputArg = output;
            int rowsArg = rows;
            int halfDimArg = halfDim;
            int count = checked(rows * halfDim);
            void** args = stackalloc void*[] { &gateUpArg, &outputArg, &rowsArg, &halfDimArg };
            Launch(siluMulSplitF32, Grid(count), 1, 1, BlockSize, 1, 1, 0, stream, args);
        }

        public void LaunchGELUMulSplitF32(IntPtr gateUp, IntPtr output, int rows, int halfDim, IntPtr stream)
        {
            IntPtr gateUpArg = gateUp;
            IntPtr outputArg = output;
            int rowsArg = rows;
            int halfDimArg = halfDim;
            int count = checked(rows * halfDim);
            void** args = stackalloc void*[] { &gateUpArg, &outputArg, &rowsArg, &halfDimArg };
            Launch(geluMulSplitF32, Grid(count), 1, 1, BlockSize, 1, 1, 0, stream, args);
        }

        public void LaunchSwiGluOaiSplitF32(IntPtr gateUp, IntPtr output, int rows, int halfDim, float alpha, float limit, IntPtr stream)
        {
            IntPtr gateUpArg = gateUp;
            IntPtr outputArg = output;
            int rowsArg = rows;
            int halfDimArg = halfDim;
            float alphaArg = alpha;
            float limitArg = limit;
            int count = checked(rows * halfDim);
            void** args = stackalloc void*[] { &gateUpArg, &outputArg, &rowsArg, &halfDimArg, &alphaArg, &limitArg };
            Launch(swigluOaiSplitF32, Grid(count), 1, 1, BlockSize, 1, 1, 0, stream, args);
        }

        public void LaunchQwen35GatedDeltaNetPackedF32(
            IntPtr packed,
            IntPtr convState,
            IntPtr ssmState,
            IntPtr convWeight,
            IntPtr dtBias,
            IntPtr aLog,
            IntPtr ssmNorm,
            IntPtr output,
            int seqLen,
            int packedDim,
            int qkvDim,
            int qkDim,
            int vDim,
            int numKHeads,
            int numVHeads,
            int headKDim,
            int headVDim,
            int convKernel,
            int convWriteIdx,
            float eps,
            IntPtr stream,
            IntPtr dynParams = default)
        {
            IntPtr packedArg = packed;
            IntPtr convStateArg = convState;
            IntPtr ssmStateArg = ssmState;
            IntPtr convWeightArg = convWeight;
            IntPtr dtBiasArg = dtBias;
            IntPtr aLogArg = aLog;
            IntPtr ssmNormArg = ssmNorm;
            IntPtr outputArg = output;
            int seqLenArg = seqLen;
            int packedDimArg = packedDim;
            int qkvDimArg = qkvDim;
            int qkDimArg = qkDim;
            int vDimArg = vDim;
            int numKHeadsArg = numKHeads;
            int numVHeadsArg = numVHeads;
            int headKDimArg = headKDim;
            int headVDimArg = headVDim;
            int convKernelArg = convKernel;
            int convWriteIdxArg = convWriteIdx;
            float epsArg = eps;
            IntPtr dynArg = dynParams;
            void** args = stackalloc void*[]
            {
                &packedArg, &convStateArg, &ssmStateArg, &convWeightArg, &dtBiasArg,
                &aLogArg, &ssmNormArg, &outputArg, &seqLenArg, &packedDimArg,
                &qkvDimArg, &qkDimArg, &vDimArg, &numKHeadsArg, &numVHeadsArg,
                &headKDimArg, &headVDimArg, &convKernelArg, &convWriteIdxArg, &epsArg,
                &dynArg
            };
            // Multi-token prefill on supported geometry: the 3-phase split path
            // (parallel conv/norm -> sync-free register-resident row scan ->
            // parallel RMS+gate). The single-kernel path below walks the sequence
            // with block-wide barriers per token and only numVHeads blocks in
            // flight, which is latency-bound for long windows.
            if (GdnPrefillSplitEnabled && seqLen >= 8 && headKDim == 128 && (headVDim & 31) == 0)
            {
                LaunchQwen35GdnPrefillSplit(
                    packed, convState, ssmState, convWeight, dtBias, aLog, ssmNorm, output,
                    seqLen, packedDim, qkvDim, qkDim, vDim,
                    numKHeads, numVHeads, headKDim, headVDim,
                    convKernel, convWriteIdx, eps, stream);
            }
            else
            {
                // One block per head; the head's recurrent state ([headVDim, headKDim])
                // is staged in dynamic shared memory for the whole sequence. Above the
                // default 48 KB dynamic-shared limit the function attribute must be
                // raised once (sm_86 allows ~99 KB/block).
                uint sharedBytes = checked((uint)((2L * headKDim + headVDim + (long)headVDim * headKDim) * sizeof(float)));
                EnsureGdnPackedSharedCapacity(sharedBytes);
                Launch(qwen35GdnPackedF32, (uint)numVHeads, 1, 1, GdnBlockSize, 1, 1, sharedBytes, stream, args);
            }

            int convDim = convKernel - 1;
            if (convDim <= 0)
                return;

            int tail = Math.Min(seqLen, convDim);
            int updateCount = checked(tail * qkvDim);
            int convDimArg = convDim;
            void** updateArgs = stackalloc void*[]
            {
                &packedArg, &convStateArg, &seqLenArg, &packedDimArg, &qkvDimArg,
                &convDimArg, &convWriteIdxArg, &dynArg
            };
            Launch(qwen35GdnUpdateConvStateF32, Grid(updateCount), 1, 1, BlockSize, 1, 1, 0, stream, updateArgs);
        }

        // Split (3-phase) GDN prefill path; TS_CUDA_GDN_PREFILL_SPLIT=0 pins the
        // legacy single-kernel sequential walk for A/B comparison.
        internal static readonly bool GdnPrefillSplitEnabled =
            !string.Equals(Environment.GetEnvironmentVariable("TS_CUDA_GDN_PREFILL_SPLIT"), "0", StringComparison.Ordinal);

        // Token window per split-GDN pass; bounds the conv/core scratch size.
        private const int GdnSplitMaxWindow = 512;

        private IntPtr gdnSplitScratch;
        private long gdnSplitScratchBytes;

        private IntPtr EnsureGdnSplitScratch(long bytes)
        {
            if (gdnSplitScratch != IntPtr.Zero && gdnSplitScratchBytes >= bytes)
                return gdnSplitScratch;
            if (gdnSplitScratch != IntPtr.Zero)
                CudaDriverApi.cuMemFree(gdnSplitScratch);
            CudaDriverApi.cuMemAlloc(out gdnSplitScratch, new UIntPtr((ulong)bytes)).ThrowOnError();
            gdnSplitScratchBytes = bytes;
            return gdnSplitScratch;
        }

        private void LaunchQwen35GdnPrefillSplit(
            IntPtr packed, IntPtr convState, IntPtr ssmState,
            IntPtr convWeight, IntPtr dtBias, IntPtr aLog, IntPtr ssmNorm, IntPtr output,
            int seqLen, int packedDim, int qkvDim, int qkDim, int vDim,
            int numKHeads, int numVHeads, int headKDim, int headVDim,
            int convKernel, int convWriteIdx, float eps, IntPtr stream)
        {
            int stride = 2 * headKDim + headVDim + 2;
            int win = Math.Min(seqLen, GdnSplitMaxWindow);
            long convBytes = (long)numVHeads * win * stride * sizeof(float);
            long coreBytes = (long)numVHeads * win * headVDim * sizeof(float);
            IntPtr scr = EnsureGdnSplitScratch(convBytes + coreBytes);
            IntPtr core = new IntPtr(scr.ToInt64() + convBytes);

            for (int winStart = 0; winStart < seqLen; winStart += win)
            {
                int winLen = Math.Min(win, seqLen - winStart);

                IntPtr packedArg = packed; IntPtr convStateArg = convState; IntPtr convWArg = convWeight;
                IntPtr dtBiasArg = dtBias; IntPtr aLogArg = aLog; IntPtr scrArg = scr;
                int winStartArg = winStart; int winLenArg = winLen; int seqLenArg = seqLen;
                int packedDimArg = packedDim; int qkvDimArg = qkvDim; int qkDimArg = qkDim; int vDimArg = vDim;
                int numKHeadsArg = numKHeads; int numVHeadsArg = numVHeads;
                int headKDimArg = headKDim; int headVDimArg = headVDim;
                int convKernelArg = convKernel; int convWriteIdxArg = convWriteIdx;
                float epsArg = eps;

                void** convArgs = stackalloc void*[]
                {
                    &packedArg, &convStateArg, &convWArg, &dtBiasArg, &aLogArg, &scrArg,
                    &winStartArg, &winLenArg, &seqLenArg,
                    &packedDimArg, &qkvDimArg, &qkDimArg, &vDimArg,
                    &numKHeadsArg, &numVHeadsArg, &headKDimArg, &headVDimArg,
                    &convKernelArg, &convWriteIdxArg, &epsArg
                };
                uint convWarps = (uint)((long)winLen * numVHeads);
                uint convGrid = (uint)((convWarps + 7) / 8);   // 8 warps per 256-thread block
                Launch(qwen35GdnPrefillConvF32, convGrid, 1, 1, BlockSize, 1, 1, 0, stream, convArgs);

                IntPtr ssmStateArg = ssmState; IntPtr coreArg = core;
                void** scanArgs = stackalloc void*[]
                {
                    &scrArg, &ssmStateArg, &coreArg,
                    &winLenArg, &numVHeadsArg, &headKDimArg, &headVDimArg
                };
                uint scanGridY = (uint)((headVDim + 31) / 32);  // 8 warps x 4 rows per block
                Launch(qwen35GdnPrefillScanF32, (uint)numVHeads, scanGridY, 1, BlockSize, 1, 1, 0, stream, scanArgs);

                IntPtr ssmNormArg = ssmNorm; IntPtr outputArg = output;
                void** outArgs = stackalloc void*[]
                {
                    &coreArg, &packedArg, &ssmNormArg, &outputArg,
                    &winStartArg, &winLenArg, &packedDimArg, &qkvDimArg, &vDimArg,
                    &numVHeadsArg, &headVDimArg, &epsArg
                };
                Launch(qwen35GdnPrefillOutF32, convGrid, 1, 1, BlockSize, 1, 1, 0, stream, outArgs);
            }
        }

        // Maximum dynamic shared memory the GDN packed kernel can request; checked by
        // the caller so unsupported geometries fall back before a launch failure.
        public const uint GdnPackedMaxSharedBytes = 96 * 1024;

        private uint gdnPackedSharedCapacity = 48 * 1024;

        private void EnsureGdnPackedSharedCapacity(uint sharedBytes)
        {
            if (sharedBytes <= gdnPackedSharedCapacity)
                return;
            CudaDriverApi.cuFuncSetAttribute(
                qwen35GdnPackedF32,
                CudaDriverApi.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                checked((int)sharedBytes)).ThrowOnError();
            gdnPackedSharedCapacity = sharedBytes;
        }

        public void LaunchQwen35GatedDeltaNetPackInputsF32(
            IntPtr qkv,
            IntPtr z,
            IntPtr beta,
            IntPtr alpha,
            IntPtr packed,
            int seqLen,
            int qkvDim,
            int zDim,
            int numVHeads,
            int packedDim,
            IntPtr stream)
        {
            IntPtr qkvArg = qkv;
            IntPtr zArg = z;
            IntPtr betaArg = beta;
            IntPtr alphaArg = alpha;
            IntPtr packedArg = packed;
            int seqLenArg = seqLen;
            int qkvDimArg = qkvDim;
            int zDimArg = zDim;
            int numVHeadsArg = numVHeads;
            int packedDimArg = packedDim;
            int count = checked(seqLen * packedDim);
            void** args = stackalloc void*[]
            {
                &qkvArg, &zArg, &betaArg, &alphaArg, &packedArg,
                &seqLenArg, &qkvDimArg, &zDimArg, &numVHeadsArg, &packedDimArg
            };
            Launch(qwen35GdnPackInputsF32, Grid(count), 1, 1, BlockSize, 1, 1, 0, stream, args);
        }

        public void LaunchQwen35GdnFusedF32(
            IntPtr qkv, IntPtr z, IntPtr beta, IntPtr alpha,
            IntPtr convState, IntPtr ssmState, IntPtr convWeight,
            IntPtr dtBias, IntPtr aLog, IntPtr ssmNorm, IntPtr output,
            int seqLen, int qkvDim, int zDim, int qkDim, int vDim,
            int numKHeads, int numVHeads, int headKDim, int headVDim,
            int convKernel, int convWriteIdx, float eps, IntPtr stream)
        {
            IntPtr qkvArg = qkv; IntPtr zArg = z; IntPtr betaArg = beta; IntPtr alphaArg = alpha;
            IntPtr convStateArg = convState; IntPtr ssmStateArg = ssmState; IntPtr convWeightArg = convWeight;
            IntPtr dtBiasArg = dtBias; IntPtr aLogArg = aLog; IntPtr ssmNormArg = ssmNorm; IntPtr outputArg = output;
            int seqLenArg = seqLen; int qkvDimArg = qkvDim; int zDimArg = zDim;
            int qkDimArg = qkDim; int vDimArg = vDim;
            int numKHeadsArg = numKHeads; int numVHeadsArg = numVHeads;
            int headKDimArg = headKDim; int headVDimArg = headVDim;
            int convKernelArg = convKernel; int convWriteIdxArg = convWriteIdx;
            float epsArg = eps;

            void** args = stackalloc void*[]
            {
                &qkvArg, &zArg, &betaArg, &alphaArg,
                &convStateArg, &ssmStateArg, &convWeightArg,
                &dtBiasArg, &aLogArg, &ssmNormArg, &outputArg,
                &seqLenArg, &qkvDimArg, &zDimArg, &qkDimArg, &vDimArg,
                &numKHeadsArg, &numVHeadsArg, &headKDimArg, &headVDimArg,
                &convKernelArg, &convWriteIdxArg, &epsArg
            };
            // One block per head; warps stride the head's rows (multi-block-per-head
            // raced the shared-state decay and core[] reduction).
            uint sharedBytes = checked((uint)((2 * headKDim + headVDim) * sizeof(float)));
            Launch(qwen35GdnFusedF32, (uint)numVHeads, 1, 1, GdnBlockSize, 1, 1, sharedBytes, stream, args);
        }

        public void LaunchRMSNormF32(IntPtr input, IntPtr alpha, IntPtr beta, IntPtr output, int rows, int cols, float eps, IntPtr stream)
        {
            IntPtr inputArg = input;
            IntPtr alphaArg = alpha;
            IntPtr betaArg = beta;
            IntPtr outputArg = output;
            int rowsArg = rows;
            int colsArg = cols;
            float epsArg = eps;
            void** args = stackalloc void*[] { &inputArg, &alphaArg, &betaArg, &outputArg, &rowsArg, &colsArg, &epsArg };
            Launch(rmsNormF32, (uint)rows, 1, 1, BlockSize, 1, 1, 0, stream, args);
        }

        // residual[row,i] += rms_norm(input[row], alpha)[i] ÔÇö fused norm + residual add.
        public void LaunchRMSNormResidualAddF32(IntPtr input, IntPtr alpha, IntPtr residual, int rows, int cols, float eps, IntPtr stream)
        {
            IntPtr inputArg = input;
            IntPtr alphaArg = alpha;
            IntPtr residualArg = residual;
            int rowsArg = rows;
            int colsArg = cols;
            float epsArg = eps;
            void** args = stackalloc void*[] { &inputArg, &alphaArg, &residualArg, &rowsArg, &colsArg, &epsArg };
            Launch(rmsNormResidualAddF32, (uint)rows, 1, 1, BlockSize, 1, 1, 0, stream, args);
        }

        public void LaunchSoftmaxF32(IntPtr input, IntPtr output, int rows, int cols, IntPtr stream)
        {
            IntPtr inputArg = input;
            IntPtr outputArg = output;
            int rowsArg = rows;
            int colsArg = cols;
            void** args = stackalloc void*[] { &inputArg, &outputArg, &rowsArg, &colsArg };
            Launch(softmaxF32, (uint)rows, 1, 1, BlockSize, 1, 1, 0, stream, args);
        }

        public void LaunchAttentionSoftmaxSinksF32(
            IntPtr scores,
            IntPtr sinks,
            int numHeads,
            int seqLen,
            int kvLen,
            int maskStart,
            int windowSize,
            float scale,
            int hasSinks,
            IntPtr stream)
        {
            IntPtr scoresArg = scores;
            IntPtr sinksArg = sinks;
            int numHeadsArg = numHeads;
            int seqLenArg = seqLen;
            int kvLenArg = kvLen;
            int maskStartArg = maskStart;
            int windowSizeArg = windowSize;
            float scaleArg = scale;
            int hasSinksArg = hasSinks;
            void** args = stackalloc void*[]
            {
                &scoresArg, &sinksArg, &numHeadsArg, &seqLenArg, &kvLenArg,
                &maskStartArg, &windowSizeArg, &scaleArg, &hasSinksArg
            };
            Launch(attentionSoftmaxSinksF32, (uint)(numHeads * seqLen), 1, 1, BlockSize, 1, 1, 0, stream, args);
        }

        public void LaunchScaledDotProductAttentionF32(
            IntPtr query,
            IntPtr key,
            IntPtr value,
            IntPtr mask,
            IntPtr output,
            int batch,
            int seqQ,
            int seqK,
            int heads,
            int keyDim,
            int valueDim,
            float scale,
            int hasMask,
            IntPtr stream)
        {
            IntPtr queryArg = query;
            IntPtr keyArg = key;
            IntPtr valueArg = value;
            IntPtr maskArg = mask;
            IntPtr outputArg = output;
            int batchArg = batch;
            int seqQArg = seqQ;
            int seqKArg = seqK;
            int headsArg = heads;
            int keyDimArg = keyDim;
            int valueDimArg = valueDim;
            float scaleArg = scale;
            int hasMaskArg = hasMask;
            void** args = stackalloc void*[]
            {
                &queryArg, &keyArg, &valueArg, &maskArg, &outputArg, &batchArg, &seqQArg,
                &seqKArg, &headsArg, &keyDimArg, &valueDimArg, &scaleArg, &hasMaskArg
            };
            Launch(scaledDotProductAttentionF32, (uint)(batch * heads), (uint)seqQ, 1, BlockSize, 1, 1, (uint)(seqK * sizeof(float)), stream, args);
        }

        public void LaunchGqaPrefillAttentionF32(
            IntPtr query,
            IntPtr key,
            IntPtr value,
            IntPtr output,
            int numQHeads,
            int numKVHeads,
            int seqLen,
            int kvLen,
            int headDim,
            int maskStart,
            int windowSize,
            float scale,
            int kvStride,
            IntPtr stream)
        {
            IntPtr queryArg = query;
            IntPtr keyArg = key;
            IntPtr valueArg = value;
            IntPtr outputArg = output;
            int numQHeadsArg = numQHeads;
            int numKVHeadsArg = numKVHeads;
            int seqLenArg = seqLen;
            int kvLenArg = kvLen;
            int headDimArg = headDim;
            int maskStartArg = maskStart;
            int windowSizeArg = windowSize;
            float scaleArg = scale;
            int kvStrideArg = kvStride;
            int warpCooperativeArg = GqaPrefillWarpCooperativeEnabled ? 1 : 0;
            void** args = stackalloc void*[]
            {
                &queryArg, &keyArg, &valueArg, &outputArg, &numQHeadsArg, &numKVHeadsArg,
                &seqLenArg, &kvLenArg, &headDimArg, &maskStartArg, &windowSizeArg, &scaleArg, &kvStrideArg,
                &warpCooperativeArg
            };
            Launch(gqaPrefillAttentionF32, (uint)numQHeads, (uint)seqLen, 1, BlockSize, 1, 1, (uint)(kvLen * sizeof(float)), stream, args);
        }

        public void LaunchGqaPrefillAttentionF16(
            IntPtr query,
            IntPtr key,
            IntPtr value,
            IntPtr output,
            int numQHeads,
            int numKVHeads,
            int seqLen,
            int kvLen,
            int headDim,
            int maskStart,
            int windowSize,
            float scale,
            int kvStride,
            IntPtr stream)
        {
            IntPtr queryArg = query;
            IntPtr keyArg = key;
            IntPtr valueArg = value;
            IntPtr outputArg = output;
            int numQHeadsArg = numQHeads;
            int numKVHeadsArg = numKVHeads;
            int seqLenArg = seqLen;
            int kvLenArg = kvLen;
            int headDimArg = headDim;
            int maskStartArg = maskStart;
            int windowSizeArg = windowSize;
            float scaleArg = scale;
            int kvStrideArg = kvStride;
            int warpCooperativeArg = GqaPrefillWarpCooperativeEnabled ? 1 : 0;
            void** args = stackalloc void*[]
            {
                &queryArg, &keyArg, &valueArg, &outputArg, &numQHeadsArg, &numKVHeadsArg,
                &seqLenArg, &kvLenArg, &headDimArg, &maskStartArg, &windowSizeArg, &scaleArg, &kvStrideArg,
                &warpCooperativeArg
            };
            Launch(gqaPrefillAttentionF16, (uint)numQHeads, (uint)seqLen, 1, BlockSize, 1, 1, (uint)(kvLen * sizeof(float)), stream, args);
        }

        public void LaunchGqaPrefillAttentionGroup4D256F32(
            IntPtr query,
            IntPtr key,
            IntPtr value,
            IntPtr output,
            int numQHeads,
            int numKVHeads,
            int seqLen,
            int kvLen,
            int headDim,
            int maskStart,
            int windowSize,
            float scale,
            int kvStride,
            IntPtr stream)
        {
            LaunchGqaPrefillAttentionGroup4D256(
                gqaPrefillAttentionGroup4D256F32,
                query, key, value, output,
                numQHeads, numKVHeads, seqLen, kvLen, headDim,
                maskStart, windowSize, scale, kvStride, stream);
        }

        public void LaunchGqaPrefillAttentionGroup4D256F16(
            IntPtr query,
            IntPtr key,
            IntPtr value,
            IntPtr output,
            int numQHeads,
            int numKVHeads,
            int seqLen,
            int kvLen,
            int headDim,
            int maskStart,
            int windowSize,
            float scale,
            int kvStride,
            IntPtr stream)
        {
            LaunchGqaPrefillAttentionGroup4D256(
                gqaPrefillAttentionGroup4D256F16,
                query, key, value, output,
                numQHeads, numKVHeads, seqLen, kvLen, headDim,
                maskStart, windowSize, scale, kvStride, stream);
        }

        public void LaunchGqaPrefillAttentionGroup4D512F32(
            IntPtr query,
            IntPtr key,
            IntPtr value,
            IntPtr output,
            int numQHeads,
            int numKVHeads,
            int seqLen,
            int kvLen,
            int headDim,
            int maskStart,
            int windowSize,
            float scale,
            int kvStride,
            IntPtr stream)
        {
            LaunchGqaPrefillAttentionGroup4D512(
                gqaPrefillAttentionGroup4D512F32,
                query, key, value, output,
                numQHeads, numKVHeads, seqLen, kvLen, headDim,
                maskStart, windowSize, scale, kvStride, stream);
        }

        public void LaunchGqaPrefillAttentionGroup4D512F16(
            IntPtr query,
            IntPtr key,
            IntPtr value,
            IntPtr output,
            int numQHeads,
            int numKVHeads,
            int seqLen,
            int kvLen,
            int headDim,
            int maskStart,
            int windowSize,
            float scale,
            int kvStride,
            IntPtr stream)
        {
            LaunchGqaPrefillAttentionGroup4D512(
                gqaPrefillAttentionGroup4D512F16,
                query, key, value, output,
                numQHeads, numKVHeads, seqLen, kvLen, headDim,
                maskStart, windowSize, scale, kvStride, stream);
        }

        public void LaunchGqaPrefillAttentionGroup4OnlineD512F32(
            IntPtr query,
            IntPtr key,
            IntPtr value,
            IntPtr output,
            int numQHeads,
            int numKVHeads,
            int seqLen,
            int kvLen,
            int headDim,
            int maskStart,
            int windowSize,
            float scale,
            int kvStride,
            IntPtr stream)
        {
            LaunchGqaPrefillAttentionGroup4OnlineD512(
                gqaPrefillAttentionGroup4OnlineD512F32,
                query, key, value, output,
                numQHeads, numKVHeads, seqLen, kvLen, headDim,
                maskStart, windowSize, scale, kvStride, stream);
        }

        public void LaunchGqaPrefillAttentionGroup4OnlineD512F16(
            IntPtr query,
            IntPtr key,
            IntPtr value,
            IntPtr output,
            int numQHeads,
            int numKVHeads,
            int seqLen,
            int kvLen,
            int headDim,
            int maskStart,
            int windowSize,
            float scale,
            int kvStride,
            IntPtr stream)
        {
            LaunchGqaPrefillAttentionGroup4OnlineD512(
                gqaPrefillAttentionGroup4OnlineD512F16,
                query, key, value, output,
                numQHeads, numKVHeads, seqLen, kvLen, headDim,
                maskStart, windowSize, scale, kvStride, stream);
        }

        private void LaunchGqaPrefillAttentionGroup4OnlineD512(
            IntPtr function,
            IntPtr query,
            IntPtr key,
            IntPtr value,
            IntPtr output,
            int numQHeads,
            int numKVHeads,
            int seqLen,
            int kvLen,
            int headDim,
            int maskStart,
            int windowSize,
            float scale,
            int kvStride,
            IntPtr stream)
        {
            IntPtr queryArg = query;
            IntPtr keyArg = key;
            IntPtr valueArg = value;
            IntPtr outputArg = output;
            int numQHeadsArg = numQHeads;
            int numKVHeadsArg = numKVHeads;
            int seqLenArg = seqLen;
            int kvLenArg = kvLen;
            int headDimArg = headDim;
            int maskStartArg = maskStart;
            int windowSizeArg = windowSize;
            float scaleArg = scale;
            int kvStrideArg = kvStride;
            void** args = stackalloc void*[]
            {
                &queryArg, &keyArg, &valueArg, &outputArg,
                &numQHeadsArg, &numKVHeadsArg, &seqLenArg, &kvLenArg,
                &headDimArg, &maskStartArg, &windowSizeArg, &scaleArg,
                &kvStrideArg
            };
            const int onlineBlockSize = 128;
            Launch(
                function,
                (uint)numKVHeads,
                (uint)seqLen,
                1,
                onlineBlockSize,
                1,
                1,
                0,
                stream,
                args);
        }

        /// <summary>Queries per CTA of the flash prefill kernels; must match the
        /// QROWS template arguments of ts_gqa_prefill_flash_group4_d256/d512_f16.</summary>
        internal const int FlashPrefillQRowsD256 = 8;
        internal const int FlashPrefillQRowsD512 = 4;
        private const int FlashPrefillKChunk = 32;

        public void LaunchGqaPrefillFlashGroup4(
            IntPtr query,
            IntPtr key,
            IntPtr value,
            IntPtr sinksPtr,
            IntPtr output,
            int numQHeads,
            int numKVHeads,
            int seqLen,
            int kvLen,
            int headDim,
            int maskStart,
            int windowSize,
            float scale,
            int kvStride,
            int hasSinks,
            bool cacheIsHalf,
            IntPtr stream)
        {
            IntPtr function = headDim == 256
                ? (cacheIsHalf ? gqaPrefillFlashGroup4D256F16 : gqaPrefillFlashGroup4D256F32)
                : (cacheIsHalf ? gqaPrefillFlashGroup4D512F16 : gqaPrefillFlashGroup4D512F32);
            int qRows = headDim == 256 ? FlashPrefillQRowsD256 : FlashPrefillQRowsD512;
            int rows = qRows * 4;
            IntPtr queryArg = query;
            IntPtr keyArg = key;
            IntPtr valueArg = value;
            IntPtr sinksArg = sinksPtr;
            IntPtr outputArg = output;
            int numQHeadsArg = numQHeads;
            int numKVHeadsArg = numKVHeads;
            int seqLenArg = seqLen;
            int kvLenArg = kvLen;
            int headDimArg = headDim;
            int maskStartArg = maskStart;
            int windowSizeArg = windowSize;
            float scaleArg = scale;
            int kvStrideArg = kvStride;
            int hasSinksArg = hasSinks;
            void** args = stackalloc void*[]
            {
                &queryArg, &keyArg, &valueArg, &sinksArg, &outputArg, &numQHeadsArg, &numKVHeadsArg,
                &seqLenArg, &kvLenArg, &headDimArg, &maskStartArg, &windowSizeArg, &scaleArg,
                &kvStrideArg, &hasSinksArg
            };
            // Workspace: Q tile (rows*headDim) + score tile (rows*KCHUNK) + the
            // per-row m/l/alpha state (3*rows), all f32.
            uint sharedBytes = checked((uint)(
                ((long)rows * headDim + (long)rows * FlashPrefillKChunk + 3L * rows) * sizeof(float)));
            uint qTiles = (uint)((seqLen + qRows - 1) / qRows);
            Launch(
                function, (uint)numKVHeads, qTiles, 1,
                BlockSize, 1, 1, sharedBytes, stream, args);
        }

        // flash2 KCHUNK (must match the KC template args of
        // ts_gqa_prefill_flash2_group4_d256/d512_*).
        private const int Flash2KChunk = 64;

        /// <summary>Whether the flash2 kernel supports this head dim (and its
        /// shared footprint fits the per-block cap on this GPU).</summary>
        public bool Flash2Supports(int headDim)
        {
            if (headDim != 256 && headDim != 512)
                return false;
            return Flash2SharedBytes(headDim) <= 100 * 1024;
        }

        private static uint Flash2SharedBytes(int headDim)
        {
            int qRows = headDim == 256 ? FlashPrefillQRowsD256 : FlashPrefillQRowsD512;
            int rows = qRows * 4;
            int kc = Flash2KChunk;
            // q_s (rows*headDim) + scores (rows*kc) + m/l/alpha (3*rows), all f32.
            long floats = (long)rows * headDim + (long)rows * kc + 3L * rows;
            return checked((uint)(floats * sizeof(float)));
        }

        public void LaunchGqaPrefillFlash2Group4(
            IntPtr query,
            IntPtr key,
            IntPtr value,
            IntPtr sinksPtr,
            IntPtr output,
            int numQHeads,
            int numKVHeads,
            int seqLen,
            int kvLen,
            int headDim,
            int maskStart,
            int windowSize,
            float scale,
            int kvStride,
            int hasSinks,
            bool cacheIsHalf,
            IntPtr stream)
        {
            IntPtr function = headDim == 256
                ? (cacheIsHalf ? gqaPrefillFlash2Group4D256F16 : gqaPrefillFlash2Group4D256F32)
                : (cacheIsHalf ? gqaPrefillFlash2Group4D512F16 : gqaPrefillFlash2Group4D512F32);
            int qRows = headDim == 256 ? FlashPrefillQRowsD256 : FlashPrefillQRowsD512;
            uint sharedBytes = Flash2SharedBytes(headDim);
            EnsureFlash2SharedCapacity(headDim, sharedBytes);

            IntPtr queryArg = query, keyArg = key, valueArg = value, sinksArg = sinksPtr, outputArg = output;
            int numQHeadsArg = numQHeads, numKVHeadsArg = numKVHeads, seqLenArg = seqLen, kvLenArg = kvLen;
            int headDimArg = headDim, maskStartArg = maskStart, windowSizeArg = windowSize;
            float scaleArg = scale;
            int kvStrideArg = kvStride, hasSinksArg = hasSinks;
            void** args = stackalloc void*[]
            {
                &queryArg, &keyArg, &valueArg, &sinksArg, &outputArg, &numQHeadsArg, &numKVHeadsArg,
                &seqLenArg, &kvLenArg, &headDimArg, &maskStartArg, &windowSizeArg, &scaleArg,
                &kvStrideArg, &hasSinksArg
            };
            uint qTiles = (uint)((seqLen + qRows - 1) / qRows);
            Launch(function, (uint)numKVHeads, qTiles, 1, BlockSize, 1, 1, sharedBytes, stream, args);
        }

        private void EnsureFlash2SharedCapacity(int headDim, uint sharedBytes)
        {
            if (headDim == 256)
            {
                if (sharedBytes <= flash2SharedCapacityD256) return;
                CudaDriverApi.cuFuncSetAttribute(gqaPrefillFlash2Group4D256F16,
                    CudaDriverApi.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, checked((int)sharedBytes)).ThrowOnError();
                CudaDriverApi.cuFuncSetAttribute(gqaPrefillFlash2Group4D256F32,
                    CudaDriverApi.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, checked((int)sharedBytes)).ThrowOnError();
                flash2SharedCapacityD256 = sharedBytes;
            }
            else
            {
                if (sharedBytes <= flash2SharedCapacityD512) return;
                CudaDriverApi.cuFuncSetAttribute(gqaPrefillFlash2Group4D512F16,
                    CudaDriverApi.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, checked((int)sharedBytes)).ThrowOnError();
                CudaDriverApi.cuFuncSetAttribute(gqaPrefillFlash2Group4D512F32,
                    CudaDriverApi.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, checked((int)sharedBytes)).ThrowOnError();
                flash2SharedCapacityD512 = sharedBytes;
            }
        }

        private void LaunchGqaPrefillAttentionGroup4D512(
            IntPtr function,
            IntPtr query,
            IntPtr key,
            IntPtr value,
            IntPtr output,
            int numQHeads,
            int numKVHeads,
            int seqLen,
            int kvLen,
            int headDim,
            int maskStart,
            int windowSize,
            float scale,
            int kvStride,
            IntPtr stream)
        {
            IntPtr queryArg = query;
            IntPtr keyArg = key;
            IntPtr valueArg = value;
            IntPtr outputArg = output;
            int numQHeadsArg = numQHeads;
            int numKVHeadsArg = numKVHeads;
            int seqLenArg = seqLen;
            int kvLenArg = kvLen;
            int headDimArg = headDim;
            int maskStartArg = maskStart;
            int windowSizeArg = windowSize;
            float scaleArg = scale;
            int kvStrideArg = kvStride;
            void** args = stackalloc void*[]
            {
                &queryArg, &keyArg, &valueArg, &outputArg, &numQHeadsArg, &numKVHeadsArg,
                &seqLenArg, &kvLenArg, &headDimArg, &maskStartArg, &windowSizeArg, &scaleArg,
                &kvStrideArg
            };
            uint sharedBytes = (uint)(4 * kvLen * sizeof(float));
            Launch(
                function, (uint)numKVHeads, (uint)seqLen, 1,
                BlockSize, 1, 1, sharedBytes, stream, args);
        }

        private void LaunchGqaPrefillAttentionGroup4D256(
            IntPtr function,
            IntPtr query,
            IntPtr key,
            IntPtr value,
            IntPtr output,
            int numQHeads,
            int numKVHeads,
            int seqLen,
            int kvLen,
            int headDim,
            int maskStart,
            int windowSize,
            float scale,
            int kvStride,
            IntPtr stream)
        {
            IntPtr queryArg = query;
            IntPtr keyArg = key;
            IntPtr valueArg = value;
            IntPtr outputArg = output;
            int numQHeadsArg = numQHeads;
            int numKVHeadsArg = numKVHeads;
            int seqLenArg = seqLen;
            int kvLenArg = kvLen;
            int headDimArg = headDim;
            int maskStartArg = maskStart;
            int windowSizeArg = windowSize;
            float scaleArg = scale;
            int kvStrideArg = kvStride;
            void** args = stackalloc void*[]
            {
                &queryArg, &keyArg, &valueArg, &outputArg, &numQHeadsArg, &numKVHeadsArg,
                &seqLenArg, &kvLenArg, &headDimArg, &maskStartArg, &windowSizeArg, &scaleArg,
                &kvStrideArg
            };
            const int queryTile = 2;
            int scoreStride = Math.Min(kvLen, windowSize + queryTile - 1);
            uint sharedBytes = (uint)(4 * queryTile * scoreStride * sizeof(float));
            Launch(
                function, (uint)numKVHeads, (uint)((seqLen + queryTile - 1) / queryTile), 1,
                BlockSize, 1, 1, sharedBytes, stream, args);
        }

        public void LaunchGqaPrefillAttentionSinksF32(
            IntPtr query,
            IntPtr keyCache,
            IntPtr valueCache,
            IntPtr sinks,
            IntPtr output,
            int numQHeads,
            int numKVHeads,
            int seqLen,
            int kvLen,
            int cacheSize,
            int headDim,
            int maskStart,
            int windowSize,
            float scale,
            int hasSinks,
            IntPtr stream)
        {
            IntPtr queryArg = query;
            IntPtr keyCacheArg = keyCache;
            IntPtr valueCacheArg = valueCache;
            IntPtr sinksArg = sinks;
            IntPtr outputArg = output;
            int numQHeadsArg = numQHeads;
            int numKVHeadsArg = numKVHeads;
            int seqLenArg = seqLen;
            int kvLenArg = kvLen;
            int cacheSizeArg = cacheSize;
            int headDimArg = headDim;
            int maskStartArg = maskStart;
            int windowSizeArg = windowSize;
            float scaleArg = scale;
            int hasSinksArg = hasSinks;
            int warpCooperativeArg = GqaPrefillWarpCooperativeEnabled ? 1 : 0;
            void** args = stackalloc void*[]
            {
                &queryArg, &keyCacheArg, &valueCacheArg, &sinksArg, &outputArg,
                &numQHeadsArg, &numKVHeadsArg, &seqLenArg, &kvLenArg, &cacheSizeArg,
                &headDimArg, &maskStartArg, &windowSizeArg, &scaleArg, &hasSinksArg,
                &warpCooperativeArg
            };
            Launch(gqaPrefillAttentionSinksF32, (uint)numQHeads, (uint)seqLen, 1, BlockSize, 1, 1, (uint)(kvLen * sizeof(float)), stream, args);
        }

        public void LaunchGqaPrefillAttentionSinksF16(
            IntPtr query,
            IntPtr keyCache,
            IntPtr valueCache,
            IntPtr sinks,
            IntPtr output,
            int numQHeads,
            int numKVHeads,
            int seqLen,
            int kvLen,
            int cacheSize,
            int headDim,
            int maskStart,
            int windowSize,
            float scale,
            int hasSinks,
            IntPtr stream)
        {
            IntPtr queryArg = query;
            IntPtr keyCacheArg = keyCache;
            IntPtr valueCacheArg = valueCache;
            IntPtr sinksArg = sinks;
            IntPtr outputArg = output;
            int numQHeadsArg = numQHeads;
            int numKVHeadsArg = numKVHeads;
            int seqLenArg = seqLen;
            int kvLenArg = kvLen;
            int cacheSizeArg = cacheSize;
            int headDimArg = headDim;
            int maskStartArg = maskStart;
            int windowSizeArg = windowSize;
            float scaleArg = scale;
            int hasSinksArg = hasSinks;
            int warpCooperativeArg = GqaPrefillWarpCooperativeEnabled ? 1 : 0;
            void** args = stackalloc void*[]
            {
                &queryArg, &keyCacheArg, &valueCacheArg, &sinksArg, &outputArg,
                &numQHeadsArg, &numKVHeadsArg, &seqLenArg, &kvLenArg, &cacheSizeArg,
                &headDimArg, &maskStartArg, &windowSizeArg, &scaleArg, &hasSinksArg,
                &warpCooperativeArg
            };
            Launch(gqaPrefillAttentionSinksF16, (uint)numQHeads, (uint)seqLen, 1, BlockSize, 1, 1, (uint)(kvLen * sizeof(float)), stream, args);
        }

        public void LaunchGqaDecodeAttentionF32(
            IntPtr query,
            IntPtr keyCache,
            IntPtr valueCache,
            IntPtr output,
            int numQHeads,
            int numKVHeads,
            int headDim,
            int attendStart,
            int attendLen,
            int cacheSize,
            int circular,
            float scale,
            IntPtr stream,
            IntPtr dynParams = default,
            int smemTokens = 0)
        {
            IntPtr queryArg = query;
            IntPtr keyCacheArg = keyCache;
            IntPtr valueCacheArg = valueCache;
            IntPtr outputArg = output;
            int numQHeadsArg = numQHeads;
            int numKVHeadsArg = numKVHeads;
            int headDimArg = headDim;
            int attendStartArg = attendStart;
            int attendLenArg = attendLen;
            int cacheSizeArg = cacheSize;
            int circularArg = circular;
            float scaleArg = scale;
            IntPtr dynArg = dynParams;
            void** args = stackalloc void*[]
            {
                &queryArg, &keyCacheArg, &valueCacheArg, &outputArg, &numQHeadsArg, &numKVHeadsArg,
                &headDimArg, &attendStartArg, &attendLenArg, &cacheSizeArg, &circularArg, &scaleArg,
                &dynArg
            };
            // The scores[] shared buffer must cover the largest attend_len the launch
            // can see: a decode-graph replay reads attend_len from dynParams, so the
            // caller passes the entry's validity ceiling via smemTokens.
            int smem = smemTokens > 0 ? smemTokens : attendLen;
            Launch(gqaDecodeAttentionF32, (uint)numQHeads, 1, 1, BlockSize, 1, 1, (uint)(smem * sizeof(float)), stream, args);
        }

        public void LaunchGqaDecodeAttentionF16(
            IntPtr query,
            IntPtr keyCache,
            IntPtr valueCache,
            IntPtr output,
            int numQHeads,
            int numKVHeads,
            int headDim,
            int attendStart,
            int attendLen,
            int cacheSize,
            int circular,
            float scale,
            IntPtr stream,
            IntPtr dynParams = default,
            int smemTokens = 0)
        {
            IntPtr queryArg = query;
            IntPtr keyCacheArg = keyCache;
            IntPtr valueCacheArg = valueCache;
            IntPtr outputArg = output;
            int numQHeadsArg = numQHeads;
            int numKVHeadsArg = numKVHeads;
            int headDimArg = headDim;
            int attendStartArg = attendStart;
            int attendLenArg = attendLen;
            int cacheSizeArg = cacheSize;
            int circularArg = circular;
            float scaleArg = scale;
            IntPtr dynArg = dynParams;
            void** args = stackalloc void*[]
            {
                &queryArg, &keyCacheArg, &valueCacheArg, &outputArg, &numQHeadsArg, &numKVHeadsArg,
                &headDimArg, &attendStartArg, &attendLenArg, &cacheSizeArg, &circularArg, &scaleArg,
                &dynArg
            };
            int smem = smemTokens > 0 ? smemTokens : attendLen;
            Launch(gqaDecodeAttentionF16, (uint)numQHeads, 1, 1, BlockSize, 1, 1, (uint)(smem * sizeof(float)), stream, args);
        }

        public void LaunchGqaDecodeAttentionGroup4D256F16(
            IntPtr query,
            IntPtr keyCache,
            IntPtr valueCache,
            IntPtr output,
            int numKVHeads,
            int attendLen,
            int cacheSize,
            float scale,
            IntPtr stream,
            IntPtr dynParams = default,
            int smemTokens = 0)
        {
            IntPtr queryArg = query;
            IntPtr keyCacheArg = keyCache;
            IntPtr valueCacheArg = valueCache;
            IntPtr outputArg = output;
            int numKVHeadsArg = numKVHeads;
            int attendLenArg = attendLen;
            int cacheSizeArg = cacheSize;
            float scaleArg = scale;
            int scoreCapacityArg = smemTokens > 0 ? smemTokens : Math.Min(attendLen, cacheSize);
            IntPtr dynArg = dynParams;
            void** args = stackalloc void*[]
            {
                &queryArg, &keyCacheArg, &valueCacheArg, &outputArg,
                &numKVHeadsArg, &attendLenArg, &cacheSizeArg,
                &scaleArg, &scoreCapacityArg, &dynArg
            };
            uint sharedBytes = checked((uint)(
                (4L * scoreCapacityArg + 4L * 256) * sizeof(float)));
            Launch(
                gqaDecodeAttentionGroup4D256F16,
                (uint)numKVHeads,
                1,
                1,
                BlockSize,
                1,
                1,
                sharedBytes,
                stream,
                args);
        }

        public void LaunchGqaDecodeAttentionGroup4D512F16(
            IntPtr query,
            IntPtr keyCache,
            IntPtr valueCache,
            IntPtr output,
            int numKVHeads,
            int attendStart,
            int attendLen,
            int cacheSize,
            float scale,
            IntPtr stream,
            IntPtr dynParams = default,
            int smemTokens = 0)
        {
            IntPtr queryArg = query;
            IntPtr keyCacheArg = keyCache;
            IntPtr valueCacheArg = valueCache;
            IntPtr outputArg = output;
            int numKVHeadsArg = numKVHeads;
            int attendStartArg = attendStart;
            int attendLenArg = attendLen;
            int cacheSizeArg = cacheSize;
            float scaleArg = scale;
            int scoreCapacityArg = smemTokens > 0 ? smemTokens : attendLen;
            IntPtr dynArg = dynParams;
            void** args = stackalloc void*[]
            {
                &queryArg, &keyCacheArg, &valueCacheArg, &outputArg,
                &numKVHeadsArg, &attendStartArg, &attendLenArg, &cacheSizeArg,
                &scaleArg, &scoreCapacityArg, &dynArg
            };
            uint sharedBytes = checked((uint)(
                (4L * scoreCapacityArg + 4L * 512) * sizeof(float)));
            Launch(
                gqaDecodeAttentionGroup4D512F16,
                (uint)numKVHeads,
                1,
                1,
                BlockSize,
                1,
                1,
                sharedBytes,
                stream,
                args);
        }

        public void LaunchGqaDecodeAttentionSinksF32(
            IntPtr query,
            IntPtr keyCache,
            IntPtr valueCache,
            IntPtr sinks,
            IntPtr output,
            int numQHeads,
            int numKVHeads,
            int headDim,
            int attendStart,
            int attendLen,
            int cacheSize,
            int circular,
            float scale,
            int hasSinks,
            IntPtr stream)
        {
            IntPtr queryArg = query;
            IntPtr keyCacheArg = keyCache;
            IntPtr valueCacheArg = valueCache;
            IntPtr sinksArg = sinks;
            IntPtr outputArg = output;
            int numQHeadsArg = numQHeads;
            int numKVHeadsArg = numKVHeads;
            int headDimArg = headDim;
            int attendStartArg = attendStart;
            int attendLenArg = attendLen;
            int cacheSizeArg = cacheSize;
            int circularArg = circular;
            float scaleArg = scale;
            int hasSinksArg = hasSinks;
            void** args = stackalloc void*[]
            {
                &queryArg, &keyCacheArg, &valueCacheArg, &sinksArg, &outputArg,
                &numQHeadsArg, &numKVHeadsArg, &headDimArg, &attendStartArg,
                &attendLenArg, &cacheSizeArg, &circularArg, &scaleArg, &hasSinksArg
            };
            Launch(gqaDecodeAttentionSinksF32, (uint)numQHeads, 1, 1, BlockSize, 1, 1, (uint)(attendLen * sizeof(float)), stream, args);
        }

        public void LaunchGqaDecodeAttentionSinksF16(
            IntPtr query,
            IntPtr keyCache,
            IntPtr valueCache,
            IntPtr sinks,
            IntPtr output,
            int numQHeads,
            int numKVHeads,
            int headDim,
            int attendStart,
            int attendLen,
            int cacheSize,
            int circular,
            float scale,
            int hasSinks,
            IntPtr stream)
        {
            IntPtr queryArg = query;
            IntPtr keyCacheArg = keyCache;
            IntPtr valueCacheArg = valueCache;
            IntPtr sinksArg = sinks;
            IntPtr outputArg = output;
            int numQHeadsArg = numQHeads;
            int numKVHeadsArg = numKVHeads;
            int headDimArg = headDim;
            int attendStartArg = attendStart;
            int attendLenArg = attendLen;
            int cacheSizeArg = cacheSize;
            int circularArg = circular;
            float scaleArg = scale;
            int hasSinksArg = hasSinks;
            void** args = stackalloc void*[]
            {
                &queryArg, &keyCacheArg, &valueCacheArg, &sinksArg, &outputArg,
                &numQHeadsArg, &numKVHeadsArg, &headDimArg, &attendStartArg,
                &attendLenArg, &cacheSizeArg, &circularArg, &scaleArg, &hasSinksArg
            };
            Launch(gqaDecodeAttentionSinksF16, (uint)numQHeads, 1, 1, BlockSize, 1, 1, (uint)(attendLen * sizeof(float)), stream, args);
        }

        public void LaunchGqaDecodeAttentionPartitionF32(
            IntPtr query,
            IntPtr keyCache,
            IntPtr valueCache,
            IntPtr sinks,
            IntPtr partial,
            int numQHeads,
            int numKVHeads,
            int headDim,
            int attendStart,
            int attendLen,
            int cacheSize,
            int circular,
            float scale,
            int hasSinks,
            int numPartitions,
            int partitionSize,
            IntPtr stream,
            IntPtr dynParams = default)
        {
            IntPtr queryArg = query;
            IntPtr keyCacheArg = keyCache;
            IntPtr valueCacheArg = valueCache;
            IntPtr sinksArg = sinks;
            IntPtr partialArg = partial;
            int numQHeadsArg = numQHeads;
            int numKVHeadsArg = numKVHeads;
            int headDimArg = headDim;
            int attendStartArg = attendStart;
            int attendLenArg = attendLen;
            int cacheSizeArg = cacheSize;
            int circularArg = circular;
            float scaleArg = scale;
            int hasSinksArg = hasSinks;
            int numPartitionsArg = numPartitions;
            int partitionSizeArg = partitionSize;
            IntPtr dynArg = dynParams;
            void** args = stackalloc void*[]
            {
                &queryArg, &keyCacheArg, &valueCacheArg, &sinksArg, &partialArg,
                &numQHeadsArg, &numKVHeadsArg, &headDimArg, &attendStartArg,
                &attendLenArg, &cacheSizeArg, &circularArg, &scaleArg, &hasSinksArg,
                &numPartitionsArg, &partitionSizeArg, &dynArg
            };
            Launch(gqaDecodeAttentionPartitionF32, (uint)numQHeads, (uint)numPartitions, 1, BlockSize, 1, 1, (uint)(partitionSize * sizeof(float)), stream, args);
        }

        public void LaunchGqaDecodeAttentionPartitionF16(
            IntPtr query,
            IntPtr keyCache,
            IntPtr valueCache,
            IntPtr sinks,
            IntPtr partial,
            int numQHeads,
            int numKVHeads,
            int headDim,
            int attendStart,
            int attendLen,
            int cacheSize,
            int circular,
            float scale,
            int hasSinks,
            int numPartitions,
            int partitionSize,
            IntPtr stream,
            IntPtr dynParams = default)
        {
            IntPtr queryArg = query;
            IntPtr keyCacheArg = keyCache;
            IntPtr valueCacheArg = valueCache;
            IntPtr sinksArg = sinks;
            IntPtr partialArg = partial;
            int numQHeadsArg = numQHeads;
            int numKVHeadsArg = numKVHeads;
            int headDimArg = headDim;
            int attendStartArg = attendStart;
            int attendLenArg = attendLen;
            int cacheSizeArg = cacheSize;
            int circularArg = circular;
            float scaleArg = scale;
            int hasSinksArg = hasSinks;
            int numPartitionsArg = numPartitions;
            int partitionSizeArg = partitionSize;
            IntPtr dynArg = dynParams;
            void** args = stackalloc void*[]
            {
                &queryArg, &keyCacheArg, &valueCacheArg, &sinksArg, &partialArg,
                &numQHeadsArg, &numKVHeadsArg, &headDimArg, &attendStartArg,
                &attendLenArg, &cacheSizeArg, &circularArg, &scaleArg, &hasSinksArg,
                &numPartitionsArg, &partitionSizeArg, &dynArg
            };
            Launch(gqaDecodeAttentionPartitionF16, (uint)numQHeads, (uint)numPartitions, 1, BlockSize, 1, 1, (uint)(partitionSize * sizeof(float)), stream, args);
        }

        public void LaunchGqaDecodeAttentionPartitionGroup4D512F16(
            IntPtr query,
            IntPtr keyCache,
            IntPtr valueCache,
            IntPtr partial,
            int numKVHeads,
            int attendStart,
            int attendLen,
            int cacheSize,
            float scale,
            int numPartitions,
            int partitionSize,
            IntPtr stream,
            IntPtr dynParams = default)
        {
            IntPtr queryArg = query;
            IntPtr keyCacheArg = keyCache;
            IntPtr valueCacheArg = valueCache;
            IntPtr partialArg = partial;
            int numKVHeadsArg = numKVHeads;
            int attendStartArg = attendStart;
            int attendLenArg = attendLen;
            int cacheSizeArg = cacheSize;
            float scaleArg = scale;
            int numPartitionsArg = numPartitions;
            int partitionSizeArg = partitionSize;
            IntPtr dynArg = dynParams;
            void** args = stackalloc void*[]
            {
                &queryArg, &keyCacheArg, &valueCacheArg, &partialArg,
                &numKVHeadsArg, &attendStartArg, &attendLenArg, &cacheSizeArg,
                &scaleArg, &numPartitionsArg, &partitionSizeArg, &dynArg
            };
            uint sharedBytes = checked((uint)(
                (4L * partitionSize + 4L * 512) * sizeof(float)));
            Launch(
                gqaDecodeAttentionPartitionGroup4D512F16,
                (uint)numKVHeads,
                (uint)numPartitions,
                1,
                BlockSize,
                1,
                1,
                sharedBytes,
                stream,
                args);
        }

        /// <summary>Partitioned SWA-ring counterpart of
        /// <see cref="LaunchGqaDecodeAttentionGroup4D256F16"/>: splits the physical
        /// ring across gridDim.y partitions so a 2-KV-head model no longer runs its
        /// whole decode window on two CTAs. Pair with
        /// <see cref="LaunchGqaDecodeAttentionPartitionReduceF32"/>.</summary>
        public void LaunchGqaDecodeAttentionPartitionGroup4D256F16(
            IntPtr query,
            IntPtr keyCache,
            IntPtr valueCache,
            IntPtr sinksPtr,
            IntPtr partial,
            int numKVHeads,
            int attendLen,
            int cacheSize,
            float scale,
            int hasSinks,
            int numPartitions,
            int partitionSize,
            IntPtr stream,
            IntPtr dynParams = default)
        {
            IntPtr queryArg = query;
            IntPtr keyCacheArg = keyCache;
            IntPtr valueCacheArg = valueCache;
            IntPtr sinksArg = sinksPtr;
            IntPtr partialArg = partial;
            int numKVHeadsArg = numKVHeads;
            int attendLenArg = attendLen;
            int cacheSizeArg = cacheSize;
            float scaleArg = scale;
            int hasSinksArg = hasSinks;
            int numPartitionsArg = numPartitions;
            int partitionSizeArg = partitionSize;
            IntPtr dynArg = dynParams;
            void** args = stackalloc void*[]
            {
                &queryArg, &keyCacheArg, &valueCacheArg, &sinksArg, &partialArg,
                &numKVHeadsArg, &attendLenArg, &cacheSizeArg,
                &scaleArg, &hasSinksArg, &numPartitionsArg, &partitionSizeArg, &dynArg
            };
            uint sharedBytes = checked((uint)(
                (4L * partitionSize + 4L * 256) * sizeof(float)));
            Launch(
                gqaDecodeAttentionPartitionGroup4D256F16,
                (uint)numKVHeads,
                (uint)numPartitions,
                1,
                BlockSize,
                1,
                1,
                sharedBytes,
                stream,
                args);
        }

        public void LaunchGqaDecodeAttentionPartitionReduceF32(
            IntPtr partial,
            IntPtr output,
            int numQHeads,
            int headDim,
            int numPartitions,
            IntPtr stream)
        {
            IntPtr partialArg = partial;
            IntPtr outputArg = output;
            int numQHeadsArg = numQHeads;
            int headDimArg = headDim;
            int numPartitionsArg = numPartitions;
            void** args = stackalloc void*[] { &partialArg, &outputArg, &numQHeadsArg, &headDimArg, &numPartitionsArg };
            Launch(gqaDecodeAttentionPartitionReduceF32, (uint)numQHeads, 1, 1, BlockSize, 1, 1, 0, stream, args);
        }

        public void LaunchSliceColumnsF32(
            IntPtr source,
            IntPtr output,
            int rows,
            int sourceCols,
            int colOffset,
            int width,
            IntPtr stream)
        {
            IntPtr sourceArg = source;
            IntPtr outputArg = output;
            int rowsArg = rows;
            int sourceColsArg = sourceCols;
            int colOffsetArg = colOffset;
            int widthArg = width;
            int count = checked(rows * width);
            void** args = stackalloc void*[] { &sourceArg, &outputArg, &rowsArg, &sourceColsArg, &colOffsetArg, &widthArg };
            Launch(sliceColumnsF32, Grid(count), 1, 1, BlockSize, 1, 1, 0, stream, args);
        }

        public void LaunchFlatToHeadFirstF32(
            IntPtr source,
            IntPtr output,
            int seqLen,
            int numHeads,
            int headDim,
            IntPtr stream)
        {
            IntPtr sourceArg = source;
            IntPtr outputArg = output;
            int seqLenArg = seqLen;
            int numHeadsArg = numHeads;
            int headDimArg = headDim;
            int count = checked(seqLen * numHeads * headDim);
            void** args = stackalloc void*[] { &sourceArg, &outputArg, &seqLenArg, &numHeadsArg, &headDimArg };
            Launch(flatToHeadFirstF32, Grid(count), 1, 1, BlockSize, 1, 1, 0, stream, args);
        }

        public void LaunchSplitQkvHeadFirstF32(
            IntPtr source,
            IntPtr output,
            int seqLen,
            int sourceCols,
            int colOffset,
            int numHeads,
            int headDim,
            IntPtr stream)
        {
            IntPtr sourceArg = source;
            IntPtr outputArg = output;
            int seqLenArg = seqLen;
            int sourceColsArg = sourceCols;
            int colOffsetArg = colOffset;
            int numHeadsArg = numHeads;
            int headDimArg = headDim;
            int count = checked(seqLen * numHeads * headDim);
            void** args = stackalloc void*[]
            {
                &sourceArg, &outputArg, &seqLenArg, &sourceColsArg, &colOffsetArg, &numHeadsArg, &headDimArg
            };
            Launch(splitQkvHeadFirstF32, Grid(count), 1, 1, BlockSize, 1, 1, 0, stream, args);
        }

        public void LaunchCopyHeadFirstToCacheF32(
            IntPtr source,
            IntPtr cache,
            int numHeads,
            int seqLen,
            int headDim,
            int startPos,
            int cacheSize,
            int circular,
            IntPtr stream,
            IntPtr dynParams = default)
        {
            IntPtr sourceArg = source;
            IntPtr cacheArg = cache;
            int numHeadsArg = numHeads;
            int seqLenArg = seqLen;
            int headDimArg = headDim;
            int startPosArg = startPos;
            int cacheSizeArg = cacheSize;
            int circularArg = circular;
            IntPtr dynArg = dynParams;
            int count = checked(numHeads * seqLen * headDim);
            void** args = stackalloc void*[]
            {
                &sourceArg, &cacheArg, &numHeadsArg, &seqLenArg, &headDimArg, &startPosArg, &cacheSizeArg, &circularArg,
                &dynArg
            };
            Launch(copyHeadFirstToCacheF32, Grid(count), 1, 1, BlockSize, 1, 1, 0, stream, args);
        }

        public void LaunchCopyHeadFirstToCacheF16(
            IntPtr source,
            IntPtr cache,
            int numHeads,
            int seqLen,
            int headDim,
            int startPos,
            int cacheSize,
            int circular,
            IntPtr stream,
            IntPtr dynParams = default)
        {
            IntPtr sourceArg = source;
            IntPtr cacheArg = cache;
            int numHeadsArg = numHeads;
            int seqLenArg = seqLen;
            int headDimArg = headDim;
            int startPosArg = startPos;
            int cacheSizeArg = cacheSize;
            int circularArg = circular;
            IntPtr dynArg = dynParams;
            int count = checked(numHeads * seqLen * headDim);
            void** args = stackalloc void*[]
            {
                &sourceArg, &cacheArg, &numHeadsArg, &seqLenArg, &headDimArg, &startPosArg, &cacheSizeArg, &circularArg,
                &dynArg
            };
            Launch(copyHeadFirstToCacheF16, Grid(count), 1, 1, BlockSize, 1, 1, 0, stream, args);
        }

        /// <summary>
        /// Refreshes the two cached RoPE position tensors (Q rows / K rows, one row
        /// per head for single-token decode) from the decode-graph dynamic parameter
        /// block. Captured at the top of a decode graph.
        /// </summary>
        public void LaunchFillRopePositionsI32(
            IntPtr posQ,
            int qRows,
            IntPtr posK,
            int kRows,
            IntPtr dynParams,
            IntPtr stream)
        {
            IntPtr posQArg = posQ;
            int qRowsArg = qRows;
            IntPtr posKArg = posK;
            int kRowsArg = kRows;
            IntPtr dynArg = dynParams;
            int count = Math.Max(qRows, kRows);
            void** args = stackalloc void*[] { &posQArg, &qRowsArg, &posKArg, &kRowsArg, &dynArg };
            Launch(fillRopePositionsI32, Grid(count), 1, 1, BlockSize, 1, 1, 0, stream, args);
        }

        public void LaunchGatherCircularHeadFirstF32(
            IntPtr cache,
            IntPtr output,
            int numHeads,
            int seqLen,
            int headDim,
            int startPos,
            int cacheSize,
            IntPtr stream)
        {
            IntPtr cacheArg = cache;
            IntPtr outputArg = output;
            int numHeadsArg = numHeads;
            int seqLenArg = seqLen;
            int headDimArg = headDim;
            int startPosArg = startPos;
            int cacheSizeArg = cacheSize;
            int count = checked(numHeads * seqLen * headDim);
            void** args = stackalloc void*[]
            {
                &cacheArg, &outputArg, &numHeadsArg, &seqLenArg, &headDimArg, &startPosArg, &cacheSizeArg
            };
            Launch(gatherCircularHeadFirstF32, Grid(count), 1, 1, BlockSize, 1, 1, 0, stream, args);
        }

        public void LaunchGatherCircularHeadFirstF16(
            IntPtr cache,
            IntPtr output,
            int numHeads,
            int seqLen,
            int headDim,
            int startPos,
            int cacheSize,
            IntPtr stream)
        {
            IntPtr cacheArg = cache;
            IntPtr outputArg = output;
            int numHeadsArg = numHeads;
            int seqLenArg = seqLen;
            int headDimArg = headDim;
            int startPosArg = startPos;
            int cacheSizeArg = cacheSize;
            int count = checked(numHeads * seqLen * headDim);
            void** args = stackalloc void*[]
            {
                &cacheArg, &outputArg, &numHeadsArg, &seqLenArg, &headDimArg, &startPosArg, &cacheSizeArg
            };
            Launch(gatherCircularHeadFirstF16, Grid(count), 1, 1, BlockSize, 1, 1, 0, stream, args);
        }

        public void LaunchExpandKvHeads(
            IntPtr cache,
            IntPtr output,
            int numKvHeads,
            int seqLen,
            int cacheSize,
            int headDim,
            int groupSize,
            bool cacheIsHalf,
            IntPtr stream)
        {
            IntPtr cacheArg = cache;
            IntPtr outputArg = output;
            int numKvHeadsArg = numKvHeads;
            int seqLenArg = seqLen;
            int cacheSizeArg = cacheSize;
            int headDimArg = headDim;
            int groupSizeArg = groupSize;
            int count = checked(numKvHeads * groupSize * seqLen * headDim);
            void** args = stackalloc void*[]
            {
                &cacheArg, &outputArg, &numKvHeadsArg, &seqLenArg,
                &cacheSizeArg, &headDimArg, &groupSizeArg
            };
            Launch(
                cacheIsHalf ? expandKvHeadsF16 : expandKvHeadsF32,
                Grid(count), 1, 1, BlockSize, 1, 1, 0, stream, args);
        }

        public void LaunchRepeatInterleave(
            IntPtr source,
            IntPtr output,
            int outerSize,
            int dimSize,
            int repeats,
            int innerSize,
            bool isHalf,
            IntPtr stream)
        {
            IntPtr sourceArg = source;
            IntPtr outputArg = output;
            int outerSizeArg = outerSize;
            int dimSizeArg = dimSize;
            int repeatsArg = repeats;
            int innerSizeArg = innerSize;
            int count = checked(outerSize * dimSize * repeats * innerSize);
            void** args = stackalloc void*[]
            {
                &sourceArg, &outputArg, &outerSizeArg, &dimSizeArg,
                &repeatsArg, &innerSizeArg
            };
            Launch(
                isHalf ? repeatInterleaveF16 : repeatInterleaveF32,
                Grid(count), 1, 1, BlockSize, 1, 1, 0, stream, args);
        }

        public void LaunchConcatHeadFirstF32(
            IntPtr a,
            IntPtr b,
            IntPtr output,
            int numHeads,
            int lenA,
            int lenB,
            int headDim,
            IntPtr stream)
        {
            IntPtr aArg = a;
            IntPtr bArg = b;
            IntPtr outputArg = output;
            int numHeadsArg = numHeads;
            int lenAArg = lenA;
            int lenBArg = lenB;
            int headDimArg = headDim;
            int count = checked(numHeads * (lenA + lenB) * headDim);
            void** args = stackalloc void*[] { &aArg, &bArg, &outputArg, &numHeadsArg, &lenAArg, &lenBArg, &headDimArg };
            Launch(concatHeadFirstF32, Grid(count), 1, 1, BlockSize, 1, 1, 0, stream, args);
        }

        public void LaunchNeoXRoPEHeadFirstF32(
            IntPtr data,
            IntPtr cosTable,
            IntPtr sinTable,
            int numHeads,
            int seqLen,
            int headDim,
            int ropeHalf,
            IntPtr stream)
        {
            IntPtr dataArg = data;
            IntPtr cosTableArg = cosTable;
            IntPtr sinTableArg = sinTable;
            int numHeadsArg = numHeads;
            int seqLenArg = seqLen;
            int headDimArg = headDim;
            int ropeHalfArg = ropeHalf;
            int count = checked(numHeads * seqLen * ropeHalf);
            void** args = stackalloc void*[]
            {
                &dataArg, &cosTableArg, &sinTableArg, &numHeadsArg, &seqLenArg, &headDimArg, &ropeHalfArg
            };
            Launch(neoxRopeHeadFirstF32, Grid(count), 1, 1, BlockSize, 1, 1, 0, stream, args);
        }

        public void LaunchNeoXRoPEFlatF32(
            IntPtr data,
            IntPtr cosTable,
            IntPtr sinTable,
            int numHeads,
            int seqLen,
            int headDim,
            int ropeHalf,
            IntPtr stream)
        {
            IntPtr dataArg = data;
            IntPtr cosTableArg = cosTable;
            IntPtr sinTableArg = sinTable;
            int numHeadsArg = numHeads;
            int seqLenArg = seqLen;
            int headDimArg = headDim;
            int ropeHalfArg = ropeHalf;
            int count = checked(numHeads * seqLen * ropeHalf);
            void** args = stackalloc void*[]
            { &dataArg, &cosTableArg, &sinTableArg, &numHeadsArg, &seqLenArg, &headDimArg, &ropeHalfArg };
            Launch(neoxRopeFlatF32, Grid(count), 1, 1, BlockSize, 1, 1, 0, stream, args);
        }

        public void LaunchFillNeoXRopeTablesDynamicF32(
            IntPtr localCos,
            IntPtr localSin,
            IntPtr localFrequencies,
            int localHalf,
            IntPtr globalCos,
            IntPtr globalSin,
            IntPtr globalFrequencies,
            int globalHalf,
            IntPtr dynParams,
            IntPtr stream)
        {
            IntPtr localCosArg = localCos;
            IntPtr localSinArg = localSin;
            IntPtr localFrequenciesArg = localFrequencies;
            int localHalfArg = localHalf;
            IntPtr globalCosArg = globalCos;
            IntPtr globalSinArg = globalSin;
            IntPtr globalFrequenciesArg = globalFrequencies;
            int globalHalfArg = globalHalf;
            IntPtr dynParamsArg = dynParams;
            int count = Math.Max(localHalf, globalHalf);
            void** args = stackalloc void*[]
            {
                &localCosArg, &localSinArg, &localFrequenciesArg, &localHalfArg,
                &globalCosArg, &globalSinArg, &globalFrequenciesArg, &globalHalfArg,
                &dynParamsArg
            };
            Launch(fillNeoXRopeTablesDynamicF32, Grid(count), 1, 1, BlockSize, 1, 1, 0, stream, args);
        }

        /// <summary>
        /// Fused QK-RMSNorm + NeoX RoPE kernel.  Normalizes each row via RMSNorm,
        /// then applies NeoX rotary position embeddings in-place ÔÇö all in a single
        /// kernel launch with one shared-memory pass.
        /// </summary>
        /// <param name="data">Q or K tensor [rows, cols], modified in-place.</param>
        /// <param name="alpha">RMSNorm weight [cols].</param>
        /// <param name="positions">Token positions [rows] as int32.</param>
        /// <param name="rows">seqLen * numHeads (or seqLen * kvHeads).</param>
        /// <param name="cols">headDim.</param>
        /// <param name="ropeHalf">rope_dims / 2 (number of rotary pairs).</param>
        /// <param name="eps">RMSNorm epsilon.</param>
        /// <param name="ropeBase">RoPE base frequency (e.g. 1000000.0).</param>
        /// <param name="ropeFreqScale">1.0 / rope_scale.</param>
        public void LaunchQKNormRopeNeoxF32(
            IntPtr data,
            IntPtr alpha,
            IntPtr positions,
            int rows,
            int cols,
            int ropeHalf,
            float eps,
            float ropeBase,
            float ropeFreqScale,
            IntPtr stream)
        {
            IntPtr dataArg = data;
            IntPtr alphaArg = alpha;
            IntPtr positionsArg = positions;
            int rowsArg = rows;
            int colsArg = cols;
            int ropeHalfArg = ropeHalf;
            float epsArg = eps;
            float ropeBaseArg = ropeBase;
            float ropeFreqScaleArg = ropeFreqScale;

            void** args = stackalloc void*[]
            {
                &dataArg, &alphaArg, &positionsArg, &rowsArg, &colsArg,
                &ropeHalfArg, &epsArg, &ropeBaseArg, &ropeFreqScaleArg
            };

            // ONE BLOCK PER ROW: the kernel block-reduces each row's RMS and processes
            // that row's rotation (row = blockIdx.x). Grid(rows) — ceil(rows/BlockSize)
            // — collapsed every decode-sized launch to a single block, so only the
            // first attention head was ever normalized/rotated; the rest passed
            // through raw and corrupted generations progressively with position.
            uint sharedBytes = checked((uint)(2 * cols * sizeof(float)));
            Launch(qkNormRopeNeoxF32, (uint)rows, 1, 1, BlockSize, 1, 1, sharedBytes, stream, args);
        }

        public void LaunchIndexSelectF32(IntPtr source, IntPtr indices, IntPtr output, int rows, int cols, int sourceRows, int indicesAreInt32, int isAdd, IntPtr stream)
        {
            IntPtr sourceArg = source;
            IntPtr indicesArg = indices;
            IntPtr outputArg = output;
            int rowsArg = rows;
            int colsArg = cols;
            int sourceRowsArg = sourceRows;
            int indicesAreInt32Arg = indicesAreInt32;
            int isAddArg = isAdd;
            void** args = stackalloc void*[] { &sourceArg, &indicesArg, &outputArg, &rowsArg, &colsArg, &sourceRowsArg, &indicesAreInt32Arg, &isAddArg };
            Launch(indexSelectF32, (uint)rows, 1, 1, BlockSize, 1, 1, 0, stream, args);
        }

        public void LaunchAddCausalMaskF32(IntPtr tensor, int totalRows, int cols, int seqLen, int startPos, float maskedValue, IntPtr stream)
        {
            IntPtr tensorArg = tensor;
            int totalRowsArg = totalRows;
            int colsArg = cols;
            int seqLenArg = seqLen;
            int startPosArg = startPos;
            float maskedValueArg = maskedValue;
            void** args = stackalloc void*[] { &tensorArg, &totalRowsArg, &colsArg, &seqLenArg, &startPosArg, &maskedValueArg };
            Launch(addCausalMaskF32, (uint)totalRows, 1, 1, BlockSize, 1, 1, 0, stream, args);
        }

        public void LaunchRoPEF32(IntPtr input, IntPtr output, int rows, int cols, int seqLen, int rowOffset, IntPtr stream)
        {
            IntPtr inputArg = input;
            IntPtr outputArg = output;
            int rowsArg = rows;
            int colsArg = cols;
            int seqLenArg = seqLen;
            int rowOffsetArg = rowOffset;
            int pairCount = cols / 2;
            void** args = stackalloc void*[] { &inputArg, &outputArg, &rowsArg, &colsArg, &seqLenArg, &rowOffsetArg };
            Launch(ropeF32, Grid(rows * pairCount), 1, 1, BlockSize, 1, 1, 0, stream, args);
        }

        public void LaunchRoPEExF32(
            IntPtr input,
            IntPtr positions,
            IntPtr output,
            int rows,
            int cols,
            int ropeDim,
            int mode,
            int positionsAreInt32,
            int originalContextLength,
            float freqBase,
            float freqScale,
            float extFactor,
            float attnFactor,
            float betaFast,
            float betaSlow,
            int addToResult,
            IntPtr stream)
        {
            IntPtr inputArg = input;
            IntPtr positionsArg = positions;
            IntPtr outputArg = output;
            int rowsArg = rows;
            int colsArg = cols;
            int ropeDimArg = ropeDim;
            int modeArg = mode;
            int positionsAreInt32Arg = positionsAreInt32;
            int originalContextLengthArg = originalContextLength;
            float freqBaseArg = freqBase;
            float freqScaleArg = freqScale;
            float extFactorArg = extFactor;
            float attnFactorArg = attnFactor;
            float betaFastArg = betaFast;
            float betaSlowArg = betaSlow;
            int addToResultArg = addToResult;
            int pairCount = Math.Min(ropeDim, cols) / 2;
            int copyWork = addToResult == 0 ? rows * cols : rows;
            void** args = stackalloc void*[]
            {
                &inputArg, &positionsArg, &outputArg, &rowsArg, &colsArg, &ropeDimArg, &modeArg,
                &positionsAreInt32Arg, &originalContextLengthArg, &freqBaseArg, &freqScaleArg,
                &extFactorArg, &attnFactorArg, &betaFastArg, &betaSlowArg, &addToResultArg
            };
            Launch(ropeExF32, Grid(Math.Max(rows * pairCount, copyWork)), 1, 1, BlockSize, 1, 1, 0, stream, args);
        }

        public void LaunchQuantMatmulF32(IntPtr weights, IntPtr input, IntPtr output, int type, int inDim, int outDim, int rows, IntPtr stream)
        {
            IntPtr weightsArg = weights;
            IntPtr inputArg = input;
            IntPtr outputArg = output;
            int typeArg = type;
            int inDimArg = inDim;
            int outDimArg = outDim;
            int rowsArg = rows;
            void** args = stackalloc void*[] { &weightsArg, &inputArg, &outputArg, &typeArg, &inDimArg, &outDimArg, &rowsArg };
            uint gridX = (uint)((outDim + 3) / 4);
            Launch(quantMatmulF32, gridX, (uint)rows, 1, BlockSize, 1, 1, 0, stream, args);
        }

        /// <summary>Row tile height of the batched quantized matmul kernels; must
        /// match <c>TS_QMM_ROW_TILE</c> in the kernel source.</summary>
        private const int QuantMatmulRowTile = 4;

        /// <summary>Maximum row count routed to the row-batched quantized matmul
        /// kernels. The kernels themselves tile over rows (grid.y), so this is a
        /// policy cap that keeps the batched path on the small speculative-verify
        /// / short-prefill windows it is tuned for.</summary>
        public const int QuantMatmulBatchMaxRows = 32;

        /// <summary>Row-batched quantized matmul: streams each weight row once per
        /// <see cref="QuantMatmulRowTile"/>-row tile and reuses every dequantized
        /// weight across the tile's rows (weight traffic ~ceil(rows/tile) instead
        /// of rows).</summary>
        public void LaunchQuantMatmulBatchedF32(IntPtr weights, IntPtr input, IntPtr output, int type, int inDim, int outDim, int rows, IntPtr stream)
        {
            IntPtr weightsArg = weights;
            IntPtr inputArg = input;
            IntPtr outputArg = output;
            int typeArg = type;
            int inDimArg = inDim;
            int outDimArg = outDim;
            int rowsArg = rows;
            void** args = stackalloc void*[] { &weightsArg, &inputArg, &outputArg, &typeArg, &inDimArg, &outDimArg, &rowsArg };
            // One warp per output column; rows tiled by QuantMatmulRowTile down grid.y.
            uint warpsPerBlock = (uint)(BlockSize >> 5);
            uint gridX = ((uint)outDim + warpsPerBlock - 1) / warpsPerBlock;
            uint gridY = (uint)((rows + QuantMatmulRowTile - 1) / QuantMatmulRowTile);
            Launch(quantMatmulBatchedF32, gridX, gridY, 1, BlockSize, 1, 1, 0, stream, args);
        }

        /// <summary>Single-token decode (rows==1) generic quant matmul: one block
        /// per output column at the same blockDim.x thread budget as
        /// <see cref="LaunchQuantMatmulF32"/>'s per-column split, but without its
        /// 4-columns-per-block serialization or the row-batched kernel's
        /// under-populated one-warp-per-column split (which has nothing to
        /// amortize at a single row).</summary>
        public void LaunchQuantMatmulVecF32(IntPtr weights, IntPtr input, IntPtr output, int type, int inDim, int outDim, IntPtr stream)
        {
            IntPtr weightsArg = weights;
            IntPtr inputArg = input;
            IntPtr outputArg = output;
            int typeArg = type;
            int inDimArg = inDim;
            int outDimArg = outDim;
            void** args = stackalloc void*[] { &weightsArg, &inputArg, &outputArg, &typeArg, &inDimArg, &outDimArg };
            Launch(quantMatmulVecF32, (uint)outDim, 1, 1, BlockSize, 1, 1, 0, stream, args);
        }

        // On-device MoE decode (see the ts_moe_* kernels). All three run without
        // any host readback, so a Gemma 4 MoE decode layer loop that uses them is
        // fully CUDA-graph capturable.

        public void LaunchMoERouterF32(
            IntPtr logits, IntPtr perExpertScale, IntPtr selectedExperts, IntPtr routingWeights,
            int numExperts, int nUsed, IntPtr stream)
        {
            IntPtr logitsArg = logits;
            IntPtr scaleArg = perExpertScale;
            IntPtr selArg = selectedExperts;
            IntPtr weightsArg = routingWeights;
            int numExpertsArg = numExperts;
            int nUsedArg = nUsed;
            void** args = stackalloc void*[] { &logitsArg, &scaleArg, &selArg, &weightsArg, &numExpertsArg, &nUsedArg };
            // Single block; the router's serial top-k runs in thread 0.
            Launch(moeRouterF32, 1, 1, 1, 32, 1, 1, 0, stream, args);
        }

        public void LaunchMoEExpertGateUpVecF32(
            IntPtr expertWeightPtrs, IntPtr selectedExperts, IntPtr input, IntPtr gateUpOut,
            int type, int inDim, int outDim, int nUsed, IntPtr stream)
        {
            IntPtr ptrsArg = expertWeightPtrs;
            IntPtr selArg = selectedExperts;
            IntPtr inputArg = input;
            IntPtr outArg = gateUpOut;
            int typeArg = type;
            int inDimArg = inDim;
            int outDimArg = outDim;
            void** args = stackalloc void*[] { &ptrsArg, &selArg, &inputArg, &outArg, &typeArg, &inDimArg, &outDimArg };
            const int threads = 128;
            const uint warps = threads / 32;
            uint blocks = ((uint)outDim + warps - 1) / warps;
            Launch(moeExpertGateUpVecF32, blocks, (uint)nUsed, 1, threads, 1, 1, 0, stream, args);
        }

        public void LaunchMoEExpertDownAccumF32(
            IntPtr expertWeightPtrs, IntPtr selectedExperts, IntPtr routingWeights, IntPtr hAll, IntPtr output,
            int type, int inDim, int outDim, int nUsed, IntPtr stream)
        {
            IntPtr ptrsArg = expertWeightPtrs;
            IntPtr selArg = selectedExperts;
            IntPtr weightsArg = routingWeights;
            IntPtr hArg = hAll;
            IntPtr outArg = output;
            int typeArg = type;
            int inDimArg = inDim;
            int outDimArg = outDim;
            int nUsedArg = nUsed;
            void** args = stackalloc void*[] { &ptrsArg, &selArg, &weightsArg, &hArg, &outArg, &typeArg, &inDimArg, &outDimArg, &nUsedArg };
            const int threads = 128;
            const uint warps = threads / 32;
            uint blocks = ((uint)outDim + warps - 1) / warps;
            Launch(moeExpertDownAccumF32, blocks, 1, 1, threads, 1, 1, 0, stream, args);
        }

        public void LaunchMoEExpertGateUpDp4a(
            IntPtr expertWeightPtrs, IntPtr selectedExperts, IntPtr xqInput, IntPtr gateUpOut,
            int type, int inDim, int outDim, int nUsed, IntPtr stream)
        {
            IntPtr ptrsArg = expertWeightPtrs;
            IntPtr selArg = selectedExperts;
            IntPtr xqArg = xqInput;
            IntPtr outArg = gateUpOut;
            int typeArg = type;
            int inDimArg = inDim;
            int outDimArg = outDim;
            void** args = stackalloc void*[] { &ptrsArg, &selArg, &xqArg, &outArg, &typeArg, &inDimArg, &outDimArg };
            // Four warp-owned output rows per CTA. Each warp independently walks
            // in_dim/32 quant blocks, avoiding idle lanes and cutting CTA count 4x.
            const int threads = 128;
            const uint warps = threads / 32;
            uint blocks = ((uint)outDim + warps - 1) / warps;
            Launch(moeExpertGateUpDp4aF32, blocks, (uint)nUsed, 1, threads, 1, 1, 0, stream, args);
        }

        public void LaunchMoEExpertDownDp4a(
            IntPtr expertWeightPtrs, IntPtr selectedExperts, IntPtr routingWeights, IntPtr xqH, IntPtr output,
            int type, int inDim, int outDim, int nUsed, IntPtr stream)
        {
            IntPtr ptrsArg = expertWeightPtrs;
            IntPtr selArg = selectedExperts;
            IntPtr weightsArg = routingWeights;
            IntPtr xqArg = xqH;
            IntPtr outArg = output;
            int typeArg = type;
            int inDimArg = inDim;
            int outDimArg = outDim;
            int nUsedArg = nUsed;
            void** args = stackalloc void*[] { &ptrsArg, &selArg, &weightsArg, &xqArg, &outArg, &typeArg, &inDimArg, &outDimArg, &nUsedArg };
            const int threads = 128;
            const uint warps = threads / 32;
            uint blocks = ((uint)outDim + warps - 1) / warps;
            Launch(moeExpertDownDp4aF32, blocks, 1, 1, threads, 1, 1, 0, stream, args);
        }

        // dst[i] = silu(a[i]) * b[i] over n elements (SwiGLU combine of two
        // separate gate/up buffers). In-place safe when dst == a.
        public void LaunchSiluMulF32(IntPtr dst, IntPtr a, IntPtr b, long n, IntPtr stream)
        {
            IntPtr dstArg = dst;
            IntPtr aArg = a;
            IntPtr bArg = b;
            long nArg = n;
            void** args = stackalloc void*[] { &dstArg, &aArg, &bArg, &nArg };
            uint blocks = (uint)((n + BlockSize - 1) / BlockSize);
            if (blocks == 0) blocks = 1;
            Launch(siluMulF32, blocks, 1, 1, BlockSize, 1, 1, 0, stream, args);
        }

        // output[j] += sigmoid(input . gate_vec) * shared_down[j], gate computed
        // on device (one block). gateVec == IntPtr.Zero => ungated (gate = 1).
        public void LaunchMoESharedGatedAdd(
            IntPtr output, IntPtr sharedDown, IntPtr input, IntPtr gateVec,
            int hidden, int gateDim, IntPtr stream)
        {
            IntPtr outArg = output;
            IntPtr sdArg = sharedDown;
            IntPtr inArg = input;
            IntPtr gvArg = gateVec;
            int hiddenArg = hidden;
            int gateDimArg = gateDim;
            void** args = stackalloc void*[] { &outArg, &sdArg, &inArg, &gvArg, &hiddenArg, &gateDimArg };
            Launch(moeSharedGatedAddF32, 1, 1, 1, BlockSize, 1, 1, 0, stream, args);
        }

        // ---- Batched (multi-token) MoE prefill launchers ----

        public void LaunchMoERouterBatched(
            IntPtr logits, IntPtr perExpertScale, IntPtr selectedExperts, IntPtr routingWeights,
            int numExperts, int nUsed, int numTokens, IntPtr stream)
        {
            IntPtr logitsArg = logits;
            IntPtr scaleArg = perExpertScale;
            IntPtr selArg = selectedExperts;
            IntPtr weightsArg = routingWeights;
            int numExpertsArg = numExperts;
            int nUsedArg = nUsed;
            int numTokensArg = numTokens;
            void** args = stackalloc void*[] { &logitsArg, &scaleArg, &selArg, &weightsArg, &numExpertsArg, &nUsedArg, &numTokensArg };
            // One block per token; each token's serial top-k runs in its thread 0.
            Launch(moeRouterBatchedF32, (uint)numTokens, 1, 1, 32, 1, 1, 0, stream, args);
        }

        public void LaunchMoEExpertGateUpBatchedDp4a(
            IntPtr expertWeightPtrs, IntPtr selectedExperts, IntPtr xqInput, IntPtr gateUpOut,
            int type, int inDim, int outDim, int nUsed, int numTokens, IntPtr stream)
        {
            IntPtr ptrsArg = expertWeightPtrs;
            IntPtr selArg = selectedExperts;
            IntPtr xqArg = xqInput;
            IntPtr outArg = gateUpOut;
            int typeArg = type;
            int inDimArg = inDim;
            int outDimArg = outDim;
            int nUsedArg = nUsed;
            void** args = stackalloc void*[] { &ptrsArg, &selArg, &xqArg, &outArg, &typeArg, &inDimArg, &outDimArg, &nUsedArg };
            Launch(moeExpertGateUpBatchedDp4aF32, (uint)outDim, (uint)nUsed, (uint)numTokens, 128, 1, 1, 0, stream, args);
        }

        public void LaunchMoEExpertGateUpBatchedVec(
            IntPtr expertWeightPtrs, IntPtr selectedExperts, IntPtr input, IntPtr gateUpOut,
            int type, int inDim, int outDim, int nUsed, int numTokens, IntPtr stream)
        {
            IntPtr ptrsArg = expertWeightPtrs;
            IntPtr selArg = selectedExperts;
            IntPtr inputArg = input;
            IntPtr outArg = gateUpOut;
            int typeArg = type;
            int inDimArg = inDim;
            int outDimArg = outDim;
            int nUsedArg = nUsed;
            void** args = stackalloc void*[] { &ptrsArg, &selArg, &inputArg, &outArg, &typeArg, &inDimArg, &outDimArg, &nUsedArg };
            Launch(moeExpertGateUpBatchedVecF32, (uint)outDim, (uint)nUsed, (uint)numTokens, BlockSize, 1, 1, 0, stream, args);
        }

        public void LaunchMoEExpertDownBatchedDp4a(
            IntPtr expertWeightPtrs, IntPtr selectedExperts, IntPtr routingWeights, IntPtr xqH, IntPtr output,
            int type, int inDim, int outDim, int nUsed, int numTokens, IntPtr stream)
        {
            IntPtr ptrsArg = expertWeightPtrs;
            IntPtr selArg = selectedExperts;
            IntPtr weightsArg = routingWeights;
            IntPtr xqArg = xqH;
            IntPtr outArg = output;
            int typeArg = type;
            int inDimArg = inDim;
            int outDimArg = outDim;
            int nUsedArg = nUsed;
            void** args = stackalloc void*[] { &ptrsArg, &selArg, &weightsArg, &xqArg, &outArg, &typeArg, &inDimArg, &outDimArg, &nUsedArg };
            Launch(moeExpertDownBatchedDp4aF32, (uint)outDim, (uint)numTokens, 1, 128, 1, 1, 0, stream, args);
        }

        public void LaunchMoEExpertDownBatchedAccum(
            IntPtr expertWeightPtrs, IntPtr selectedExperts, IntPtr routingWeights, IntPtr hAll, IntPtr output,
            int type, int inDim, int outDim, int nUsed, int numTokens, IntPtr stream)
        {
            IntPtr ptrsArg = expertWeightPtrs;
            IntPtr selArg = selectedExperts;
            IntPtr weightsArg = routingWeights;
            IntPtr hArg = hAll;
            IntPtr outArg = output;
            int typeArg = type;
            int inDimArg = inDim;
            int outDimArg = outDim;
            int nUsedArg = nUsed;
            void** args = stackalloc void*[] { &ptrsArg, &selArg, &weightsArg, &hArg, &outArg, &typeArg, &inDimArg, &outDimArg, &nUsedArg };
            Launch(moeExpertDownBatchedAccumF32, (uint)outDim, (uint)numTokens, 1, BlockSize, 1, 1, 0, stream, args);
        }

        public void LaunchMoESharedGatedAddBatched(
            IntPtr output, IntPtr sharedDown, IntPtr input, IntPtr gateVec,
            int hidden, int gateDim, int numTokens, IntPtr stream)
        {
            IntPtr outArg = output;
            IntPtr sdArg = sharedDown;
            IntPtr inArg = input;
            IntPtr gvArg = gateVec;
            int hiddenArg = hidden;
            int gateDimArg = gateDim;
            int numTokensArg = numTokens;
            void** args = stackalloc void*[] { &outArg, &sdArg, &inArg, &gvArg, &hiddenArg, &gateDimArg, &numTokensArg };
            Launch(moeSharedGatedAddBatchedF32, (uint)numTokens, 1, 1, BlockSize, 1, 1, 0, stream, args);
        }

        public void LaunchMoEScatterAddWeightedRows(
            IntPtr output, IntPtr expertOutput, IntPtr rowIndices, IntPtr routingWeights,
            int batchSize, int numTokens, int hidden, IntPtr stream)
        {
            IntPtr outArg = output;
            IntPtr expertArg = expertOutput;
            IntPtr rowsArg = rowIndices;
            IntPtr weightsArg = routingWeights;
            int batchArg = batchSize;
            int tokensArg = numTokens;
            int hiddenArg = hidden;
            void** args = stackalloc void*[] {
                &outArg, &expertArg, &rowsArg, &weightsArg,
                &batchArg, &tokensArg, &hiddenArg
            };
            Launch(moeScatterAddWeightedRowsF32, (uint)batchSize, 1, 1, BlockSize, 1, 1, 0, stream, args);
        }

        public void LaunchQuantMatmulIq2XxsQ81F32(IntPtr weights, IntPtr input, IntPtr output, int inDim, int outDim, int rows, IntPtr stream)
        {
            IntPtr weightsArg = weights;
            IntPtr inputArg = input;
            IntPtr outputArg = output;
            int inDimArg = inDim;
            int outDimArg = outDim;
            int rowsArg = rows;
            void** args = stackalloc void*[] { &weightsArg, &inputArg, &outputArg, &inDimArg, &outDimArg, &rowsArg };
            // One warp per output column; BlockSize/32 outputs per block.
            uint warpsPerBlock = (uint)(BlockSize >> 5);
            uint gridX = ((uint)outDim + warpsPerBlock - 1) / warpsPerBlock;
            uint sharedBytes = checked((uint)((inDim / 32) * 36));
            Launch(quantMatmulIq2XxsQ81F32, gridX, (uint)rows, 1, BlockSize, 1, 1, sharedBytes, stream, args);
        }

        public void LaunchQuantMatmulIq2VecQ81F32(
            IntPtr weights, IntPtr xqInput, IntPtr output,
            int type, int inDim, int outDim, IntPtr stream)
        {
            IntPtr weightsArg = weights;
            IntPtr xqArg = xqInput;
            IntPtr outputArg = output;
            int typeArg = type;
            int inDimArg = inDim;
            int outDimArg = outDim;
            void** args = stackalloc void*[] {
                &weightsArg, &xqArg, &outputArg, &typeArg, &inDimArg, &outDimArg
            };
            const int threads = 128;
            const uint warps = threads / 32;
            uint gridX = ((uint)outDim + warps - 1) / warps;
            Launch(quantMatmulIq2VecQ81F32, gridX, 1, 1, threads, 1, 1, 0, stream, args);
        }

        public void LaunchQuantMatmulQ40F32(IntPtr weights, IntPtr input, IntPtr output, int inDim, int outDim, int rows, IntPtr stream)
        {
            IntPtr weightsArg = weights;
            IntPtr inputArg = input;
            IntPtr outputArg = output;
            int inDimArg = inDim;
            int outDimArg = outDim;
            int rowsArg = rows;
            void** args = stackalloc void*[] { &weightsArg, &inputArg, &outputArg, &inDimArg, &outDimArg, &rowsArg };
            uint gridX = (uint)((outDim + 3) / 4);
            Launch(quantMatmulQ40F32, gridX, (uint)rows, 1, BlockSize, 1, 1, 0, stream, args);
        }

        /// <summary>Row tile height of the batched Q4_0 matmul kernel; must match
        /// <c>TS_Q40_ROW_TILE</c> in the kernel source.</summary>
        private const int QuantMatmulQ40RowTile = 12;

        /// <summary>Output columns per block of the batched Q4_0 matmul kernel; must
        /// match <c>TS_Q40_COLS</c> in the kernel source.</summary>
        private const int QuantMatmulQ40Cols = 2;

        /// <summary>Row-tiled Q4_0 matmul: <see cref="QuantMatmulQ40Cols"/> output
        /// columns per block, all threads cooperating on the dot product, with each
        /// block computing a tile of <see cref="QuantMatmulQ40RowTile"/> rows so
        /// every unpacked weight nibble is reused across the tile (weight traffic
        /// ~ceil(rows/tile) instead of rows). Used for the small speculative-verify
        /// / short-prefill windows.</summary>
        public void LaunchQuantMatmulQ40BatchedF32(IntPtr weights, IntPtr input, IntPtr output, int inDim, int outDim, int rows, IntPtr stream)
        {
            IntPtr weightsArg = weights;
            IntPtr inputArg = input;
            IntPtr outputArg = output;
            int inDimArg = inDim;
            int outDimArg = outDim;
            int rowsArg = rows;
            void** args = stackalloc void*[] { &weightsArg, &inputArg, &outputArg, &inDimArg, &outDimArg, &rowsArg };
            uint gridX = (uint)((outDim + QuantMatmulQ40Cols - 1) / QuantMatmulQ40Cols);
            uint gridY = (uint)((rows + QuantMatmulQ40RowTile - 1) / QuantMatmulQ40RowTile);
            Launch(quantMatmulQ40BatchedF32, gridX, gridY, 1, BlockSize, 1, 1, 0, stream, args);
        }

        // Block-tile int8 dp4a Q4_0 GEMM: weights (q4_0) x pre-quantized q8_1
        // activations (xqScratch) -> output [rows, outDim]. 256 threads / 4x4 tile.
        // The fast path for both single-token decode and the verify window.
        public void LaunchQuantMatmulQ40Dp4a(IntPtr weights, IntPtr xqScratch, IntPtr output, int inDim, int outDim, int rows, IntPtr stream)
        {
            IntPtr weightsArg = weights;
            IntPtr xqArg = xqScratch;
            IntPtr outputArg = output;
            int inDimArg = inDim;
            int outDimArg = outDim;
            int rowsArg = rows;
            void** args = stackalloc void*[] { &weightsArg, &xqArg, &outputArg, &inDimArg, &outDimArg, &rowsArg };
            uint gridX = (uint)((outDim + 3) / 4);
            uint gridY = (uint)((rows + Q40Dp4aTileRows - 1) / Q40Dp4aTileRows);
            Launch(quantMatmulQ40Dp4aF32, gridX, gridY, 1, BlockSize, 1, 1, 0, stream, args);
        }

        public void LaunchQuantMatmulQ80F32(IntPtr weights, IntPtr input, IntPtr output, int inDim, int outDim, int rows, IntPtr stream)
        {
            IntPtr weightsArg = weights;
            IntPtr inputArg = input;
            IntPtr outputArg = output;
            int inDimArg = inDim;
            int outDimArg = outDim;
            int rowsArg = rows;
            void** args = stackalloc void*[] { &weightsArg, &inputArg, &outputArg, &inDimArg, &outDimArg, &rowsArg };
            uint gridX = (uint)((outDim + 3) / 4);
            // rows == 1 (decode): single-row kernel. rows >= 2: the scalar 4-row-tile
            // kernel. The dp4a fast path (rows >= 2) is dispatched one level up in
            // CudaQuantizedOps (it needs a quantized-activation scratch buffer).
            if (rows < 2)
                Launch(quantMatmulQ80SingleF32, gridX, (uint)rows, 1, BlockSize, 1, 1, 0, stream, args);
            else
                Launch(quantMatmulQ80F32, gridX, (uint)((rows + 3) / 4), 1, BlockSize, 1, 1, 0, stream, args);
        }

        // Quantize `rows` activation rows ([rows, inDim] f32) to q8_1 into
        // `outScratch` (rows * inDim/32 ts_block_q8_1). The default warp path
        // assigns one 32-lane warp to each 32-value block; the legacy path keeps
        // one thread per block for controlled A/B comparisons.
        public void LaunchQuantizeQ81Rows(
            IntPtr input, IntPtr outScratch, int inDim, int rows, IntPtr stream, bool warpCooperative)
        {
            IntPtr inputArg = input;
            IntPtr outArg = outScratch;
            int inDimArg = inDim;
            int rowsArg = rows;
            void** args = stackalloc void*[] { &inputArg, &outArg, &inDimArg, &rowsArg };
            long totalBlocks = (long)rows * (inDim / 32);
            long workItems = warpCooperative ? totalBlocks * 32 : totalBlocks;
            uint grid = (uint)((workItems + BlockSize - 1) / BlockSize);
            Launch(
                warpCooperative ? quantizeQ81RowsWarpF32 : quantizeQ81RowsF32,
                grid, 1, 1, BlockSize, 1, 1, 0, stream, args);
        }

        // Split-layout q8_1 quantization (dense qs rows + separate float scales)
        // feeding the cp.async MMQ kernel; bit-identical values to the
        // interleaved variant above.
        public void LaunchQuantizeQ81SplitRows(IntPtr input, IntPtr qsScratch, IntPtr dScratch, int inDim, int rows, IntPtr stream)
        {
            IntPtr inputArg = input;
            IntPtr qsArg = qsScratch;
            IntPtr dArg = dScratch;
            int inDimArg = inDim;
            int rowsArg = rows;
            void** args = stackalloc void*[] { &inputArg, &qsArg, &dArg, &inDimArg, &rowsArg };
            long total = (long)rows * (inDim / 32);
            uint grid = (uint)((total + BlockSize - 1) / BlockSize);
            Launch(quantizeQ81SplitRowsF32, grid, 1, 1, BlockSize, 1, 1, 0, stream, args);
        }

        // Dequantize a whole quantized weight [outDim, inDim] to f16 row-major.
        public void LaunchDequantWeightF16(IntPtr weights, IntPtr outputF16, int ggmlType, int inDim, long totalElements, IntPtr stream)
        {
            IntPtr weightsArg = weights;
            IntPtr outArg = outputF16;
            int typeArg = ggmlType;
            int inDimArg = inDim;
            long totalArg = totalElements;
            void** args = stackalloc void*[] { &weightsArg, &outArg, &typeArg, &inDimArg, &totalArg };
            uint grid = (uint)((totalElements + BlockSize - 1) / BlockSize);
            Launch(dequantWeightF16, grid, 1, 1, BlockSize, 1, 1, 0, stream, args);
        }

        // Q8_0-specialized whole-weight dequantizer: one 32-thread block handles
        // 2048 values using coalesced raw staging and packed half2 stores.
        public void LaunchDequantWeightQ80F16(
            IntPtr weights, IntPtr outputF16, long totalElements, IntPtr stream)
        {
            IntPtr weightsArg = weights;
            IntPtr outArg = outputF16;
            long totalArg = totalElements;
            void** args = stackalloc void*[] { &weightsArg, &outArg, &totalArg };
            uint grid = (uint)((totalElements + 2047) / 2048);
            Launch(dequantWeightQ80F16, grid, 1, 1, 32, 1, 1, 0, stream, args);
        }

        public void LaunchConvertF32F16(IntPtr src, IntPtr dstF16, long count, IntPtr stream)
        {
            IntPtr srcArg = src;
            IntPtr dstArg = dstF16;
            long countArg = count;
            void** args = stackalloc void*[] { &srcArg, &dstArg, &countArg };
            uint grid = (uint)((count + BlockSize - 1) / BlockSize);
            Launch(convertF32F16, grid, 1, 1, BlockSize, 1, 1, 0, stream, args);
        }

        // Single-row Q8_0 dp4a matvec: four warps cooperate on each output
        // column over the pre-quantized q8_1 activation row.
        public void LaunchQuantMatmulQ80Vec(IntPtr weights, IntPtr xqScratch, IntPtr output, int inDim, int outDim, IntPtr stream)
        {
            IntPtr weightsArg = weights;
            IntPtr xqArg = xqScratch;
            IntPtr outputArg = output;
            int inDimArg = inDim;
            int outDimArg = outDim;
            void** args = stackalloc void*[] { &weightsArg, &xqArg, &outputArg, &inDimArg, &outDimArg };
            const int vecBlockSize = 128;
            Launch(quantMatmulQ80VecF32, (uint)outDim, 1, 1, vecBlockSize, 1, 1, 0, stream, args);
        }

        public void LaunchQuantMatmulQ4KDp4a(IntPtr weights, IntPtr xqScratch, IntPtr output, int inDim, int outDim, IntPtr stream)
        {
            IntPtr weightsArg = weights;
            IntPtr xqArg = xqScratch;
            IntPtr outputArg = output;
            int inDimArg = inDim;
            int outDimArg = outDim;
            void** args = stackalloc void*[] { &weightsArg, &xqArg, &outputArg, &inDimArg, &outDimArg };
            const int vecBlockSize = 128;
            Launch(quantMatmulQ4KDp4aF32, (uint)outDim, 1, 1, vecBlockSize, 1, 1, 0, stream, args);
        }

        public void LaunchQuantMatmulQ5KDp4a(IntPtr weights, IntPtr xqScratch, IntPtr output, int inDim, int outDim, IntPtr stream)
        {
            IntPtr weightsArg = weights;
            IntPtr xqArg = xqScratch;
            IntPtr outputArg = output;
            int inDimArg = inDim;
            int outDimArg = outDim;
            void** args = stackalloc void*[] { &weightsArg, &xqArg, &outputArg, &inDimArg, &outDimArg };
            const int vecBlockSize = 128;
            Launch(quantMatmulQ5KDp4aF32, (uint)outDim, 1, 1, vecBlockSize, 1, 1, 0, stream, args);
        }

        public void LaunchQuantMatmulQ6KDp4a(IntPtr weights, IntPtr xqScratch, IntPtr output, int inDim, int outDim, IntPtr stream)
        {
            IntPtr weightsArg = weights;
            IntPtr xqArg = xqScratch;
            IntPtr outputArg = output;
            int inDimArg = inDim;
            int outDimArg = outDim;
            void** args = stackalloc void*[] { &weightsArg, &xqArg, &outputArg, &inDimArg, &outDimArg };
            const int vecBlockSize = 128;
            Launch(quantMatmulQ6KDp4aF32, (uint)outDim, 1, 1, vecBlockSize, 1, 1, 0, stream, args);
        }

        // Block-tile dp4a Q8_0 GEMM: weights (q8_0) x pre-quantized q8_1 activations
        // (xqScratch) -> output [rows, outDim]. 256 threads / 4x4 output tile.
        public void LaunchQuantMatmulQ80Dp4a(IntPtr weights, IntPtr xqScratch, IntPtr output, int inDim, int outDim, int rows, IntPtr stream)
        {
            IntPtr weightsArg = weights;
            IntPtr xqArg = xqScratch;
            IntPtr outputArg = output;
            int inDimArg = inDim;
            int outDimArg = outDim;
            int rowsArg = rows;
            void** args = stackalloc void*[] { &weightsArg, &xqArg, &outputArg, &inDimArg, &outDimArg, &rowsArg };
            uint gridX = (uint)((outDim + 3) / 4);
            uint gridY = (uint)((rows + Q8Dp4aTileRows - 1) / Q8Dp4aTileRows);
            Launch(quantMatmulQ80Dp4aF32, gridX, gridY, 1, BlockSize, 1, 1, 0, stream, args);
        }

        // MMQ-style Q8_0 GEMM (mma.m16n8k32): 8-warp CTA per 128x8 output tile,
        // weights read directly from Q8_0 blocks (sm_80+; PTX is built for the
        // resolved local arch, same contract as the wmma kernel).
        public void LaunchQuantMatmulQ80Mmq(IntPtr weights, IntPtr xqScratch, IntPtr output, int inDim, int outDim, int rows, IntPtr stream)
        {
            IntPtr weightsArg = weights;
            IntPtr xqArg = xqScratch;
            IntPtr outputArg = output;
            int inDimArg = inDim;
            int outDimArg = outDim;
            int rowsArg = rows;
            void** args = stackalloc void*[] { &weightsArg, &xqArg, &outputArg, &inDimArg, &outDimArg, &rowsArg };
            uint gridX = (uint)((outDim + 63) / 64);    // TS_MMQ_N
            uint gridY = (uint)((rows + 127) / 128);    // TS_MMQ_M
            Launch(quantMatmulQ80MmqF32, gridX, gridY, 1, BlockSize, 1, 1, 0, stream, args);
        }

        // cp.async MMQ variant over the split q8_1 scratch (requires inDim % 256 == 0;
        // bit-identical results to LaunchQuantMatmulQ80Mmq).
        public void LaunchQuantMatmulQ80Mmq2(IntPtr weights, IntPtr xqQs, IntPtr xqD, IntPtr output, int inDim, int outDim, int rows, IntPtr stream)
        {
            IntPtr weightsArg = weights;
            IntPtr qsArg = xqQs;
            IntPtr dArg = xqD;
            IntPtr outputArg = output;
            int inDimArg = inDim;
            int outDimArg = outDim;
            int rowsArg = rows;
            void** args = stackalloc void*[] { &weightsArg, &qsArg, &dArg, &outputArg, &inDimArg, &outDimArg, &rowsArg };
            uint gridX = (uint)((outDim + 63) / 64);    // TS_MMQ_N
            uint gridY = (uint)((rows + 127) / 128);    // TS_MMQ_M
            Launch(quantMatmulQ80Mmq2F32, gridX, gridY, 1, BlockSize, 1, 1, 0, stream, args);
        }

        // Tensor-core (wmma int8 MMA) Q8_0 GEMM: one warp per 16x16 output tile.
        public void LaunchQuantMatmulQ80Mma(IntPtr weights, IntPtr xqScratch, IntPtr output, int inDim, int outDim, int rows, IntPtr stream)
        {
            IntPtr weightsArg = weights;
            IntPtr xqArg = xqScratch;
            IntPtr outputArg = output;
            int inDimArg = inDim;
            int outDimArg = outDim;
            int rowsArg = rows;
            void** args = stackalloc void*[] { &weightsArg, &xqArg, &outputArg, &inDimArg, &outDimArg, &rowsArg };
            uint gridX = (uint)((outDim + 15) / 16);
            uint gridY = (uint)((rows + 15) / 16);
            Launch(quantMatmulQ80MmaF32, gridX, gridY, 1, 32, 1, 1, 0, stream, args);   // one warp / tile
        }

        public void LaunchQuantGetRowsF32(IntPtr weights, IntPtr indices, IntPtr output, int type, int cols, int rows, int indicesAreInt32, IntPtr stream)
        {
            IntPtr weightsArg = weights;
            IntPtr indicesArg = indices;
            IntPtr outputArg = output;
            int typeArg = type;
            int colsArg = cols;
            int rowsArg = rows;
            int indicesAreInt32Arg = indicesAreInt32;
            void** args = stackalloc void*[] { &weightsArg, &indicesArg, &outputArg, &typeArg, &colsArg, &rowsArg, &indicesAreInt32Arg };
            Launch(quantGetRowsF32, (uint)rows, 1, 1, BlockSize, 1, 1, 0, stream, args);
        }

        public void Dispose()
        {
            if (gdnSplitScratch != IntPtr.Zero)
            {
                CudaDriverApi.cuMemFree(gdnSplitScratch);
                gdnSplitScratch = IntPtr.Zero;
                gdnSplitScratchBytes = 0;
            }
            module.Dispose();
        }

        private static void Launch(IntPtr function, uint gx, uint gy, uint gz, int bx, int by, int bz, uint sharedBytes, IntPtr stream, void** args)
        {
            CudaDriverApi.cuLaunchKernel(
                function,
                gx,
                gy,
                gz,
                (uint)bx,
                (uint)by,
                (uint)bz,
                sharedBytes,
                stream,
                (IntPtr)args,
                IntPtr.Zero).ThrowOnError();
        }

        private static uint Grid(int count)
        {
            return (uint)Math.Max(1, (count + BlockSize - 1) / BlockSize);
        }

        private static string LocatePtxPath()
        {
            // Probe the application base directory first, then the directory the
            // CUDA backend assembly itself was loaded from. When TensorSharp is
            // consumed from another solution these are usually the same folder,
            // but they can differ for plugin hosts, single-file publishes, or
            // shadow-copied assemblies, so checking both avoids the silent CPU
            // fallback reported when the consuming app lives outside this repo.
            foreach (string baseDirectory in EnumeratePtxBaseDirectories())
            {
                string[] candidates =
                {
                    Path.Combine(baseDirectory, "cuda_kernels", "tensorsharp_kernels.ptx"),
                    Path.Combine(baseDirectory, "tensorsharp_kernels.ptx"),
                    Path.Combine(baseDirectory, "..", "..", "..", "native", "ptx", "tensorsharp_kernels.ptx"),
                };

                foreach (string candidate in candidates)
                {
                    string fullPath = Path.GetFullPath(candidate);
                    if (File.Exists(fullPath))
                        return fullPath;
                }
            }

            DirectoryInfo dir = new DirectoryInfo(AppContext.BaseDirectory);
            while (dir != null)
            {
                string sourceTreePath = Path.Combine(dir.FullName, "TensorSharp.Backends.Cuda", "native", "ptx", "tensorsharp_kernels.ptx");
                if (File.Exists(sourceTreePath))
                    return sourceTreePath;

                string localProjectPath = Path.Combine(dir.FullName, "native", "ptx", "tensorsharp_kernels.ptx");
                if (File.Exists(localProjectPath))
                    return localProjectPath;

                dir = dir.Parent;
            }

            return null;
        }

        private static IEnumerable<string> EnumeratePtxBaseDirectories()
        {
            yield return AppContext.BaseDirectory;

            string assemblyLocation = null;
            try
            {
                assemblyLocation = typeof(CudaKernels).Assembly.Location;
            }
            catch
            {
                assemblyLocation = null;
            }

            if (!string.IsNullOrEmpty(assemblyLocation))
            {
                string assemblyDir = Path.GetDirectoryName(assemblyLocation);
                if (!string.IsNullOrEmpty(assemblyDir) &&
                    !string.Equals(
                        Path.GetFullPath(assemblyDir),
                        Path.GetFullPath(AppContext.BaseDirectory),
                        StringComparison.OrdinalIgnoreCase))
                {
                    yield return assemblyDir;
                }
            }
        }
    }
}
