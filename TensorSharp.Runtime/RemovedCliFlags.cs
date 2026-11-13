// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System;
using System.Collections.Generic;

namespace TensorSharp.Runtime
{
    /// <summary>
    /// Host options that were removed, shared by <c>TensorSharp.Cli</c> and
    /// <c>TensorSharp.Server</c>: features that went outright, a spelling only one host
    /// accepted, and second spellings of a surviving option, retired so every option has
    /// one name on both. The speculative and
    /// code-execution families keep their own tables; entries here carry a whole
    /// sentence of advice instead of just a flag name, because most have no survivor
    /// and the one that does needs saying why it changed.
    /// </summary>
    /// <remarks>
    /// <para>
    /// A hard error, never a silent ignore. The CLI's argument switch drops flags it has
    /// no case for, so without this a retired flag in a script would simply stop doing
    /// anything; the server's unknown-option trap would refuse it, but with a bare
    /// "Unknown option" and no word about why or what to do instead.
    /// </para>
    /// <para>
    /// Both hosts call <see cref="RejectRemoved"/> before any other option is applied,
    /// and <see cref="ConfigFileArgs"/> refuses the same names as <c>--config</c> keys
    /// before it resolves (or downloads) anything the file names.
    /// </para>
    /// </remarks>
    public static class RemovedCliFlags
    {
        /// <summary>
        /// The removed flags, each with advice for operators who still pass them.
        /// </summary>
        public static readonly IReadOnlyList<(string Flag, string Advice)> RemovedFlags = new[]
        {
            ("--qwen-image-lora",
                "it only applied to the retired Qwen-Image-Edit-2511 pipeline. "
                + "Qwen-Image-2.1 takes LoRA plug-ins with --lora <file> (plus --lora-scale and --lora-config)."),
            ("--offload-cpu",
                "it only applied to the retired Qwen-Image-Edit-2511 DiT. "
                + "Qwen-Image-2.1 keeps its weights resident; use smaller --width/--height if memory is short."),
            // The CLI alone spelled the penalty window this way; the server, the request
            // field (repeat_last_n) and TENSORSHARP_REPEAT_LAST_N all say repeat-last-n,
            // so a shared config's "repeat-last-n" key was silently dropped by the CLI.
            ("--penalty-last-n",
                "it was the CLI-only name of the repeat-penalty window. Use --repeat-last-n <N> instead, "
                + "the spelling both hosts and the repeat_last_n request field use."),
            // The standalone paged KV cache (RAM/SSD/Redis block tiers, TurboQuant codec) that
            // only --paged-bench built; the server accepted these and ignored them.
            ("--paged-kv", StandalonePagedKvRemoved),
            ("--no-paged-kv", StandalonePagedKvRemoved),
            ("--paged-kv-cache", StandalonePagedKvRemoved),
            ("--no-paged-kv-cache", StandalonePagedKvRemoved),
            ("--paged-kv-block-size", StandalonePagedKvRemoved),
            ("--paged-kv-ram-mb", StandalonePagedKvRemoved),
            ("--paged-kv-ssd-dir", StandalonePagedKvRemoved),
            ("--paged-kv-ssd-mb", StandalonePagedKvRemoved),
            ("--paged-kv-quant-bits", StandalonePagedKvRemoved),
            ("--paged-kv-redis-url", StandalonePagedKvRemoved),
            ("--paged-kv-redis-ttl", StandalonePagedKvRemoved),
            ("--paged-bench", StandalonePagedKvRemoved),
            ("--paged-bench-prompt", StandalonePagedKvRemoved),
            ("--paged-bench-trials", StandalonePagedKvRemoved),
            // Second spellings of the continuous-batching switch: one name per option.
            ("--paged-batching",
                "it was a second spelling of --continuous-batching. Use --continuous-batching instead."),
            ("--no-paged-batching",
                "it was a second spelling of --no-continuous-batching. Use --no-continuous-batching instead."),
            // Spellings from before the second video model: one name per option.
            ("--wan-vae", "it was the Wan-era spelling of --video-vae. Use --video-vae instead."),
            ("--wan-te", "it was the Wan-era spelling of --video-text-encoder. Use --video-text-encoder instead."),
            ("--wan-dit2", "it was the Wan-era spelling of --video-dit2. Use --video-dit2 instead."),
            ("--video-te", "it was a second spelling of --video-text-encoder. Use --video-text-encoder instead."),
        };

        private const string StandalonePagedKvRemoved =
            "the standalone paged KV cache it configured was removed. Prompt reuse across requests is the "
            + "engine's radix prefix cache, on by default (--no-prefix-cache or TS_SCHED_PREFIX_CACHE=0 turns it off).";

        /// <summary>
        /// Environment variables that were removed with a feature, and what to do instead. Nothing
        /// reads them any more, so a deployment still exporting one would silently run without what
        /// it asked for; both hosts refuse to start instead (<see cref="RejectRemovedEnvironment"/>).
        /// </summary>
        public static readonly IReadOnlyList<(string Name, string Advice)> RemovedEnvironmentVariables = new[]
        {
            ("TS_KV_PAGED_CACHE", StandalonePagedKvRemoved),
            ("TS_KV_BLOCK_SIZE", StandalonePagedKvRemoved),
            ("TS_KV_CACHE_MAX_RAM_MB", StandalonePagedKvRemoved),
            ("TS_KV_CACHE_SSD_DIR", StandalonePagedKvRemoved),
            ("TS_KV_CACHE_MAX_SSD_MB", StandalonePagedKvRemoved),
            ("TS_KV_PAGED_QUANT_BITS", StandalonePagedKvRemoved),
            ("TS_KV_CACHE_REDIS_URL", StandalonePagedKvRemoved),
            ("TS_KV_CACHE_REDIS_TTL_MINUTES", StandalonePagedKvRemoved),
            // Switches whose other setting was removed with its code path.
            ("TS_DSV41_RETAINED_CACHE",
                "DeepSeek V4.1 always retains a finished conversation's native slot for its next turn "
                + "(TS_DSV41_RETAINED_CACHE_MB sets the budget)."),
            ("TS_MTP_FOLD_CATCHUP",
                "the Qwen 3.5 / 3.6 draft head always folds its catch-up into the first draft step."),
            ("TS_MTP_FUSED_DRAFT",
                "the Qwen 3.5 / 3.6 draft head always uses its fused draft block where the backend supports it."),
            // Per-model switches for the per-sequence path: one switch covers every model.
            ("TS_QWEN35_BATCHED", PerModelBatchedRemoved),
            ("TS_GEMMA4_BATCHED", PerModelBatchedRemoved),
            ("TS_GPTOSS_BATCHED", PerModelBatchedRemoved),
            ("TS_NEMOTRON_BATCHED", PerModelBatchedRemoved),
            ("TS_HUNYUAN_BATCHED", PerModelBatchedRemoved),
            ("TS_WAN_VAE", "every video model reads TS_VIDEO_VAE (--video-vae)."),
            ("TS_WAN_TE", "every video model reads TS_VIDEO_TEXT_ENCODER (--video-text-encoder)."),
            ("TS_WAN_DIT2", "every video model reads TS_VIDEO_DIT2 (--video-dit2)."),
            ("DIFFUSION_CPU_LEGACY", DiffusionCpuLegacyRemoved),
            ("DIFFUSION_CPU_LEGACY_MOE", DiffusionCpuLegacyRemoved),
            ("DIFFUSION_CPU_LEGACY_PROJ", DiffusionCpuLegacyRemoved),
            ("DIFFUSION_CPU_LEGACY_ATTN", DiffusionCpuLegacyRemoved),
            ("DIFFUSION_CPU_LEGACY_ROUTER", DiffusionCpuLegacyRemoved),
            // Switches that restored the pure-C# CPU backend's previous kernels.
            ("TS_CPU_QGEMM", CpuPreviousKernelsRemoved),
            ("TS_CPU_FGEMM", CpuPreviousKernelsRemoved),
            ("TS_CPU_SGEMM", CpuPreviousKernelsRemoved),
            ("TS_CPU_SIMD_ELEMENTWISE", CpuPreviousKernelsRemoved),
            ("TS_QWEN_VAE_CPU", CpuPreviousKernelsRemoved),
            ("TS_QWEN_TE_CPU_GEMM", CpuPreviousKernelsRemoved),
            ("TS_QWEN_TE_CPU_ATTN", CpuPreviousKernelsRemoved),
            ("TS_QWEN35_VENC_CPU_GEMM", CpuPreviousKernelsRemoved),
            ("TS_QWEN35_VENC_CPU_ATTN", CpuPreviousKernelsRemoved),
            ("TS_DIRECT_QUANT_WEIGHTS",
                "the direct video networks always multiply quantized weights in their GGUF storage type."),
            ("TS_QWEN35_MIGRATE",
                "a lone Qwen 3.5 request always moves onto the batched path when a second one arrives; "
                + "--no-continuous-batching (TS_SCHED_DISABLE_BATCHED=1) keeps every request on the per-sequence path."),
            // Switches that restored an earlier kernel or engaged a measured-slower experiment.
            ("TS_MLX_BASELINE_STREAM", MlxKernelSwitchRemoved),
            ("TS_MLX_BASELINE_FREE", MlxKernelSwitchRemoved),
            ("TS_MLX_BASELINE_ASYNC_LAYER", MlxKernelSwitchRemoved),
            ("TS_MLX_CACHED_ATTENTION", MlxKernelSwitchRemoved),
            ("TS_MLX_DEVICE_ROUTER", MlxKernelSwitchRemoved),
            ("TS_MLX_DEVICE_MOE_ROUTING", MlxKernelSwitchRemoved),
            ("TS_MLX_DEVICE_KV_COPY", MlxKernelSwitchRemoved),
            ("TS_MLX_DISABLE_COMPILE", MlxKernelSwitchRemoved),
            ("TS_MLX_DISABLE_GDN_T1", MlxKernelSwitchRemoved),
            ("TS_MLX_GDN_BLOCKED", MlxKernelSwitchRemoved),
            ("TS_MLX_GDN_NATIVE", MlxKernelSwitchRemoved),
            ("TS_MLX_QWEN35_GDN_PACKED_KERNELS", MlxKernelSwitchRemoved),
            ("TS_MLX_QWEN35_GDN_PACKED_MIN_SEQ_LEN", MlxKernelSwitchRemoved),
            ("TS_MLX_FUSED_KV_WRITE", MlxKernelSwitchRemoved),
            ("TS_MLX_FUSED_PLE_GATE", MlxKernelSwitchRemoved),
            ("TS_MLX_FUSED_Q8_MATMUL", MlxKernelSwitchRemoved),
            ("TS_MLX_FUSED_Q8_RMSNORM_MATMUL", MlxKernelSwitchRemoved),
            ("TS_MLX_FUSED_Q8_ADDMM_ADD", MlxKernelSwitchRemoved),
            ("TS_MLX_FUSED_FFN", MlxKernelSwitchRemoved),
            ("TS_MLX_FUSED_QKV_PREP", MlxKernelSwitchRemoved),
            ("TS_MLX_FUSED_HEADDIM512_SDPA", MlxKernelSwitchRemoved),
            ("TS_MLX_HEAD_DIM256_ATTENTION", MlxKernelSwitchRemoved),
            ("TS_MLX_CHUNKED_VECTOR_PREFILL", MlxKernelSwitchRemoved),
            ("TS_MLX_FUSED_DEVICE_MOE", MlxKernelSwitchRemoved),
            ("TS_MLX_MOE_GATHER_QMM", MlxKernelSwitchRemoved),
            ("TS_MLX_MOE_FUSED_GATE_UP_SILU", MlxKernelSwitchRemoved),
            ("TS_MLX_QUANT_MATMUL_CONTIG", MlxKernelSwitchRemoved),
            ("TS_MLX_STRIDED_BOX_COPY", MlxKernelSwitchRemoved),
            ("TS_MLX_MIXED_GATE_UP_SPLIT", MlxKernelSwitchRemoved),
            ("TS_MLX_KQUANT_AFFINE", MlxKernelSwitchRemoved),
            ("TS_MLX_Q5K_RAW", MlxKernelSwitchRemoved),
            ("TS_MLX_Q5K_MATMUL4", MlxKernelSwitchRemoved),
            ("TS_MLX_Q6K_MATMUL4", MlxKernelSwitchRemoved),
            ("TS_MLX_Q6K_DEQUANT_GEMM", MlxKernelSwitchRemoved),
            ("TS_MLX_Q6K_AFFINE8", MlxKernelSwitchRemoved),
            ("TS_MLX_IQ_DECODE_DOT8", MlxKernelSwitchRemoved),
            ("TS_MLX_IQ2XXS_DISABLE_SG", MlxKernelSwitchRemoved),
            ("TS_MLX_IQ4XS_DEQUANT_GEMM", MlxKernelSwitchRemoved),
            ("TS_MLX_IQ4XS_MATMUL4", MlxKernelSwitchRemoved),
            ("TS_MLX_IQ4XS_MATMUL4_SIMD", MlxKernelSwitchRemoved),
            ("TS_MLX_IQ4XS_BATCHED_ROWS", MlxKernelSwitchRemoved),
            ("TS_MLX_IQ4XS_BATCHED_COLS", MlxKernelSwitchRemoved),
            ("TS_MLX_EVAL_DECODE_LAYER_BOUNDARIES", MlxKernelSwitchRemoved),
            ("TS_MLX_GEMMA4_EVAL_DECODE_LAYER_BOUNDARIES", MlxKernelSwitchRemoved),
            ("TS_MLX_EVAL_FINAL_LOGITS", MlxKernelSwitchRemoved),
            ("TS_MLX_QWEN_GPU_DEINTERLEAVE", MlxKernelSwitchRemoved),
            ("TS_MLX_FLASH_ATTN_DECODE_MIN_SEQ_LEN", MlxKernelSwitchRemoved),
            ("TS_MLX_SINKS_ATTN_MIN_KV_LEN", MlxKernelSwitchRemoved),
            ("FUSED_ATTN_LAYER_MIN_SEQ_LEN", MlxKernelSwitchRemoved),
            ("TS_MLX_GEMMA4_EVAL_EVERY_N_LAYERS", "every MLX model reads TS_MLX_EVAL_EVERY_N_LAYERS."),
            ("TS_MLX_MUSE_GLIMMER_EVAL_EVERY_N_LAYERS", "every MLX model reads TS_MLX_EVAL_EVERY_N_LAYERS."),
            ("TS_MLX_GEMMA4_LOCAL_KV_MATERIALIZE_INTERVAL", "Gemma 4 reads TS_MLX_LOCAL_KV_MATERIALIZE_INTERVAL."),
            ("TS_MLX_MUSE_GLIMMER_KV_MATERIALIZE", "Muse-Glimmer reads TS_MLX_KV_MATERIALIZE_INTERVAL."),
            ("TS_MLX_PIPELINED_DECODE",
                "the CLI benchmark always uses the pipelined greedy decode where the model supports it."),
            // Switches that turned off a retained (replayed) whole-model graph or its pieces.
            ("TS_GPTOSS_FD_PERSIST", FusedGraphSwitchRemoved),
            ("TS_GPTOSS_BATCHED_ARENA", FusedGraphSwitchRemoved),
            ("TS_GEMMA4_FD_PERSIST", FusedGraphSwitchRemoved),
            ("TS_GEMMA4_METAL_PERSIST", FusedGraphSwitchRemoved),
            ("TS_QWEN35_FD_PERSIST", FusedGraphSwitchRemoved),
            ("TS_QWEN35_BFD_PERSIST", FusedGraphSwitchRemoved),
            ("TS_QWEN35_BATCHED_ARENA", FusedGraphSwitchRemoved),
            ("TS_QWEN35_METAL_ASYNC_SUBMIT", FusedGraphSwitchRemoved),
            ("TS_QWEN35_METAL_GDN_INPLACE_STATE", FusedGraphSwitchRemoved),
            ("TS_QWEN35_METAL_KV_CPY", FusedGraphSwitchRemoved),
            ("TS_QWEN35_VULKAN_FLASH", FusedGraphSwitchRemoved),
            ("TS_QWEN4EXP_BATCHED_ARENA", FusedGraphSwitchRemoved),
            ("TS_MUSE_GLIMMER_PERSIST", FusedGraphSwitchRemoved),
            ("TS_DFLASH_PERSIST", FusedGraphSwitchRemoved),
            ("TS_Q35_VERIFY_PERSIST", FusedGraphSwitchRemoved),
            ("TS_Q35_VERIFY_DEFER_STATE", FusedGraphSwitchRemoved),
            ("TS_Q35_VERIFY_STRIDED_VIEWS", FusedGraphSwitchRemoved),
            ("TS_Q35_MTP_DRAFT_PERSIST", FusedGraphSwitchRemoved),
            // Qwen 3.5 / Gemma 4 switches that turned a default path off.
            ("TS_QWEN35_FULL_DECODE", DefaultPathSwitchRemoved),
            ("TS_QWEN35_FUSED_REC_PREFILL", DefaultPathSwitchRemoved),
            ("TS_QWEN35_METAL_TOKEN_INPUT", DefaultPathSwitchRemoved),
            ("GDN_DISABLE_CHUNKED_PREFILL", DefaultPathSwitchRemoved),
            ("TS_QWEN35_FUSED_VERIFY", DefaultPathSwitchRemoved),
            ("TS_QWEN35_BATCHED_FUSED", DefaultPathSwitchRemoved),
            ("TS_QWEN35_STACKED_MOE", DefaultPathSwitchRemoved),
            ("TS_QWEN35_MROPE_NATIVE", DefaultPathSwitchRemoved),
            ("TS_QWEN35_MROPE_VERIFY", DefaultPathSwitchRemoved),
            ("TS_QWEN35_PREFILL_VERIFY", DefaultPathSwitchRemoved),
            ("TS_QWEN35_SPEC_DEVICE_STATE", DefaultPathSwitchRemoved),
            ("TS_QWEN35_TP_FUSED", DefaultPathSwitchRemoved),
            ("TS_QWEN35_TP_FUSED_DECODE", DefaultPathSwitchRemoved),
            ("TS_QWEN35_TP_FUSED_PREFILL", DefaultPathSwitchRemoved),
            ("TS_QWEN35_VENC_FUSED", DefaultPathSwitchRemoved),
            ("TS_QWEN35_VENC_FUSED_ATTN", DefaultPathSwitchRemoved),
            ("QWEN35_DISABLE_FUSED_FFN", DefaultPathSwitchRemoved),
            ("TS_GEMMA4_MOE_MODEL_DECODE", DefaultPathSwitchRemoved),
            ("TS_GEMMA4_FLASH_GLOBAL", DefaultPathSwitchRemoved),
            ("TS_GEMMA4_FLASH_F16KV", DefaultPathSwitchRemoved),
            ("TS_GEMMA4_FD_FOLD_LMHEAD", DefaultPathSwitchRemoved),
            ("TS_GEMMA4_TP_FUSED_DECODE", DefaultPathSwitchRemoved),
            ("TS_G4_WHOLE_PREFILL", DefaultPathSwitchRemoved),
            ("TS_G4_MM_PREFILL", DefaultPathSwitchRemoved),
            ("TS_G4_VERIFY_SWAPREV", DefaultPathSwitchRemoved),
            ("TS_G4_PLE_IN_KERNEL", DefaultPathSwitchRemoved),
            ("TS_GMTP_NO_FUSED", DefaultPathSwitchRemoved),
            ("TS_GMTP_PLE_IN_KERNEL", DefaultPathSwitchRemoved),
            ("TS_GMTP_NO_FOLD_HEAD", DefaultPathSwitchRemoved),
            ("TS_GMTP_NO_FAST_ROLLBACK", DefaultPathSwitchRemoved),
            ("TS_G4_VERIFY_NPAD", DefaultPathSwitchRemoved),
            ("TS_G4_GPU_MASK", DefaultPathSwitchRemoved),
            ("TS_G4_SWA_TILED", DefaultPathSwitchRemoved),
            ("TS_G4_FLASH_KV_PAD", DefaultPathSwitchRemoved),
            ("TS_G4_MOE_ATTN_TILED", DefaultPathSwitchRemoved),
            ("TS_QWEN35_VERIFY_RESIDENT", ExperimentRemoved),
            ("TS_QWEN35_BFD_NOMIRROR", ExperimentRemoved),
            ("TS_QWEN35_MLX_TENSOR_PAGED_ATTN", ExperimentRemoved),
            ("TS_GMTP_BATCHED_TRUNK", ExperimentRemoved),
            ("TS_QWEN35_BATCHED_GDN_NATIVE", ExperimentRemoved),
            ("TS_QWEN35_HOST_MOE_VERIFY", "every model reads TS_HOST_MOE_VERIFY."),
            ("TS_FUSED_LAYER_PREFILL", DefaultPathSwitchRemoved),
            ("TS_GEMMA4_FORCE_UNFUSED", DefaultPathSwitchRemoved),
            ("TS_PAGED_ATTN_KERNEL",
                "Mistral 3 and Hunyuan Dense run the native paged attention on the GGML backends and the managed one elsewhere."),
            ("TS_STRUCTURED_STREAM_BUFFER",
                "json_object streams incrementally and only json_schema buffers the whole response."),
            ("TS_GEMMA4_BATCHED_CAPS",
                "Gemma 4's token-batched decode always covers per-layer embeddings, shared-KV layers and a wrapped "
                + "sliding window; TS_BATCHED_FUSED_DECODE=0 decodes concurrent requests round-robin instead."),
            ("TS_BATCHED_N1_FAST_PATH",
                "a lone request always takes the model's fused single-sequence decode where the model supports it; "
                + "the switch was removed."),
            // Second spellings of a surviving budget: one name per option.
            ("TS_RETAINED_FUSED_CACHE",
                "it was a second spelling of TS_RETAINED_FUSED_CACHE_MAX=0, which turns end-state retention off."),
            ("TS_PREFIX_CHECKPOINTS",
                "it was a second spelling of TS_PREFIX_CHECKPOINTS_MAX=0, which turns shared-prefix checkpoints off."),
            ("TS_GPTOSS_FUSED_DECODE", GptOssSwitchRemoved),
            ("TS_GPTOSS_MODEL_DECODE", GptOssSwitchRemoved),
            ("TS_GPTOSS_MODEL_PREFILL", GptOssSwitchRemoved),
            ("TS_GPTOSS_TP_FUSED_DECODE", GptOssSwitchRemoved),
            ("TS_GPTOSS_TP_EXPERT_PARALLEL", GptOssSwitchRemoved),
            ("TS_GPTOSS_MLX_MOE_GQMM", GptOssSwitchRemoved),
            ("TS_GPTOSS_MLX_MOE_SELFCHECK", GptOssSwitchRemoved),
            ("TS_GPTOSS_PAGED_ATTN_MANAGED", GptOssSwitchRemoved),
            // Direct-cuda switches back to an earlier kernel or off a CUDA graph.
            ("TS_CUDA_QMM_BATCHED", CudaKernelSwitchRemoved),
            ("TS_CUDA_QMM_VEC", CudaKernelSwitchRemoved),
            ("TS_CUDA_QMM_F16GEMM", CudaKernelSwitchRemoved),
            ("TS_CUDA_Q80_F16_DEQUANT", CudaKernelSwitchRemoved),
            ("TS_CUDA_BF16_MATVEC", CudaKernelSwitchRemoved),
            ("TS_DSV4_BF16_MATVEC", CudaKernelSwitchRemoved),
            ("TS_CUDA_Q40_DP4A", CudaKernelSwitchRemoved),
            ("TS_CUDA_Q80_VEC", CudaKernelSwitchRemoved),
            ("TS_CUDA_Q4K_DP4A", CudaKernelSwitchRemoved),
            ("TS_CUDA_Q5K_DP4A", CudaKernelSwitchRemoved),
            ("TS_CUDA_Q6K_DP4A", CudaKernelSwitchRemoved),
            ("TS_CUDA_IQ2XXS_DP4A", CudaKernelSwitchRemoved),
            ("TS_CUDA_IQ2S_DP4A", CudaKernelSwitchRemoved),
            ("TS_CUDA_IQ2_VEC", CudaKernelSwitchRemoved),
            ("TS_CUDA_Q81_WARP", CudaKernelSwitchRemoved),
            ("TS_CUDA_Q80_MMQ", CudaKernelSwitchRemoved),
            ("TS_CUDA_Q80_MMQ2", CudaKernelSwitchRemoved),
            ("TS_CUDA_Q8_DP4A", CudaKernelSwitchRemoved),
            ("TS_CUDA_GQA_PREFILL_WARP", CudaKernelSwitchRemoved),
            ("TS_CUDA_GQA_PREFILL_GROUP4", CudaKernelSwitchRemoved),
            ("TS_CUDA_FLASH_PREFILL", CudaKernelSwitchRemoved),
            ("TS_CUDA_FLASH2", CudaKernelSwitchRemoved),
            ("TS_CUDA_GQA_DECODE_GROUP4", CudaKernelSwitchRemoved),
            ("TS_CUDA_GDN_PREFILL_SPLIT", CudaKernelSwitchRemoved),
            ("TS_CUDA_PREFILL_GRAPH", CudaKernelSwitchRemoved),
            ("TS_CUDA_DECODE_GRAPH", CudaKernelSwitchRemoved),
            ("TS_CUDA_MOE_ONDEVICE", CudaKernelSwitchRemoved),
            ("TS_CUDA_MOE_PREFILL_GROUPED", CudaKernelSwitchRemoved),
            ("TS_CUDA_QWEN35_GDN_NATIVE", CudaKernelSwitchRemoved),
            ("TS_TP_MOE_PREFILL_ONDEVICE", CudaKernelSwitchRemoved),
            ("TS_DSV4_MMA_EXPERTS", CudaKernelSwitchRemoved),
            ("TS_DSV4_STAGED_EXPERTS", CudaKernelSwitchRemoved),
            ("TS_DSV4_DSPARK_CAPTURE",
                "the DeepSeek V4.1 DSpark drafter always captures its target features; the switch was removed."),
            ("TS_DSV4_HC_NATIVE",
                "DeepSeek V4 picks the fused hyper-connection kernels or their decomposition from what the backend "
                + "supports; the switch was removed."),
            ("TS_GLM_HC_NATIVE",
                "GLM 5.3-Flash picks the fused hyper-connection kernels or their decomposition from what the backend "
                + "supports; the switch was removed."),
            ("TS_GLM_BATCHED_DECODE",
                "it was a second spelling of TS_BATCHED_FUSED_DECODE=0, which turns the batched decode off for every model."),
            ("TS_GLM_MTP",
                "the GLM native loader pages the NextN block in exactly when speculation (TS_SPEC / --spec) will use it."),
            // Model switches that turned a default path off, or engaged a never-default experiment.
            ("TS_Q4E_FUSED_ATTN", DefaultPathSwitchRemoved),
            ("TS_Q4E_FUSED_FFN", DefaultPathSwitchRemoved),
            ("TS_Q4E_FUSED_GDN", DefaultPathSwitchRemoved),
            ("TS_Q4E_SPAN_ATTN",
                "Qwen 3.8 attention layers always run inside the token-span graph; the per-layer attention "
                + "split was removed."),
            // Qwen 3.8 Flash Next's own GGML implementation (Qwen4ExpModel) was removed: the shared decoder runtime serves
            // every backend but --backend cuda (the direct engine, whose TS_Q4E_* switches stay).
            ("TS_Q4E_ARENA_BATCHED_ATTN", Qwen4ExpModelRemoved),
            ("TS_Q4E_ARENA_BATCHED_GDN", Qwen4ExpModelRemoved),
            ("TS_Q4E_DEBUG_SPAN_INPUT", Qwen4ExpModelRemoved),
            ("TS_Q4E_DEBUG_SPAN_INPUT_LAYER", Qwen4ExpModelRemoved),
            ("TS_Q4E_DEBUG_SPAN_LAYERS", Qwen4ExpModelRemoved),
            ("TS_Q4E_DISABLE_ARENA_DECODE", Qwen4ExpModelRemoved),
            ("TS_Q4E_DRIVER_DUMP", Qwen4ExpModelRemoved),
            ("TS_Q4E_DRIVER_TRACE", Qwen4ExpModelRemoved),
            ("TS_Q4E_EXPERT_CACHE_CAPTURE_DIR", Qwen4ExpModelRemoved),
            ("TS_Q4E_EXPERT_CACHE_REFERENCE_DIR", Qwen4ExpModelRemoved),
            ("TS_Q4E_GDN_VERIFY", Qwen4ExpModelRemoved),
            ("TS_Q4E_LAYER_SPLIT", Qwen4ExpModelRemoved),
            ("TS_Q4E_LOG", Qwen4ExpModelRemoved),
            ("TS_Q4E_METAL_FUSE", Qwen4ExpModelRemoved),
            ("TS_Q4E_NODE_DUMP", Qwen4ExpModelRemoved),
            ("TS_Q4E_NODE_DUMP_LAYER", Qwen4ExpModelRemoved),
            ("TS_Q4E_PLE_PREFETCH", Qwen4ExpModelRemoved),
            ("TS_Q4E_PREFILL_COMBINE", Qwen4ExpModelRemoved),
            ("TS_Q4E_PROFILE", Qwen4ExpModelRemoved),
            ("TS_Q4E_PROMPT_SPAN_CAP", Qwen4ExpModelRemoved),
            ("TS_Q4E_QSA_POOL_CACHE", Qwen4ExpModelRemoved),
            ("TS_Q4E_QSA_SPARSE_FA", Qwen4ExpModelRemoved),
            ("TS_Q4E_SNAPSHOT_GPU", Qwen4ExpModelRemoved),
            ("TS_Q4E_SPAN_TRACE", Qwen4ExpModelRemoved),
            ("TS_Q4E_TEST_BATCHED_VERIFY", Qwen4ExpModelRemoved),
            ("TS_Q4E_TEST_MMVQ_CHANNELS", Qwen4ExpModelRemoved),
            ("TS_Q4E_VERIFY_ROW_COMMIT", Qwen4ExpModelRemoved),
            ("TS_Q4E_VERIFY_ROW_KERNELS", Qwen4ExpModelRemoved),
            ("TS_Q4E_VISION_MERGER_ERF",
                "Qwen 3.8 Flash Next's vision merger keeps its tanh GELU (the shipped projector's validated default); the "
                + "erf opt-in of the removed Qwen4ExpModel was removed."),
            ("TS_GGML_TP_FUSED_MATMUL",
                "the single-thread multi-rank linear it engaged (never the default) allocated a device buffer per rank "
                + "per call and measured 2.3x slower on Qwen 3.5 35B (ggml_cuda --tp 2); local TP always runs the per-rank path."),
            ("TS_DSV41_TP",
                "--tp N enables DeepSeek V4.1 routed-MoE tensor parallelism (2..8 GPUs); the environment entry "
                + "point was removed."),
            ("TS_DSV4_NGPU",
                "it was a second spelling of the GPU count: --layer-split N places whole layers on N GPUs and "
                + "--tp N selects tensor parallelism (there is no all-visible-GPUs shorthand)."),
            ("TS_GLM_NGPU",
                "it was a second spelling of the GPU count: --layer-split N places whole layers on N GPUs and "
                + "--tp N selects tensor parallelism (there is no all-visible-GPUs shorthand)."),
            ("TS_DISABLE_FUSED_DENSE_FFN", DefaultPathSwitchRemoved),
            ("TS_ENCODER_YIELD", DefaultPathSwitchRemoved),
            ("TS_GEMMA4V_FUSED", DefaultPathSwitchRemoved),
            ("TS_GGML_FUSED_NORM_ADD", DefaultPathSwitchRemoved),
            ("TS_GGML_MOE_FUSED_DECODE", DefaultPathSwitchRemoved),
            ("TS_QWEN_VAE_POOL", DefaultPathSwitchRemoved),
            ("TS_QWEN_VAE_POOL_TRIM", DefaultPathSwitchRemoved),
            ("TENSORSHARP_CUDA_POOL",
                "the direct CUDA backend always pools device memory; TENSORSHARP_CUDA_POOL_MAX_MB / "
                + "_LARGE_MB size the cache (0 caches nothing)."),
            ("TS_Q4E_TOKEN_GRAPH", DefaultPathSwitchRemoved),
            ("TS_Q4E_FLASH_ATTN", DefaultPathSwitchRemoved),
            ("TS_Q4E_GRAPH_UID", DefaultPathSwitchRemoved),
            ("TS_Q4E_SPAN_STATE", DefaultPathSwitchRemoved),
            ("TS_Q4E_SPAN_REBUILD", ExperimentRemoved),
            ("TS_Q4E_SPAN_FA_MAX", ExperimentRemoved),
            ("TS_Q4E_GDN_MAX_LAYERS", ExperimentRemoved),
            ("TS_Q4E_RES_RESIDENT", ExperimentRemoved),
            ("TS_Q4E_RETAINED_CACHE",
                "Qwen 3.8 always retains finished conversations and shared-prefix checkpoints; "
                + "TS_Q4E_RETAINED_CACHE_MB sizes that memory (0 retains nothing)."),
            ("TS_NEMOTRON_FLASH_DECODE", DefaultPathSwitchRemoved),
            ("TS_NEMOTRON_LINEAR_RESIDUAL_FUSED", DefaultPathSwitchRemoved),
            ("TS_NEMOTRON_MAMBA2_NATIVE_DECODE", DefaultPathSwitchRemoved),
            ("TS_NEMOTRON_MAMBA2_NATIVE_PREFILL", DefaultPathSwitchRemoved),
            ("TS_NEMOTRON_MOE_PREFILL_BATCHED", DefaultPathSwitchRemoved),
            ("TS_MUSE_GLIMMER_FUSED", DefaultPathSwitchRemoved),
            ("TS_MUSE_GLIMMER_FUSED_CPU", DefaultPathSwitchRemoved),
            ("TS_MUSE_GLIMMER_TP_FUSED", DefaultPathSwitchRemoved),
            ("TS_MUSE_GLIMMER_VENC_FUSED", DefaultPathSwitchRemoved),
            ("TS_MUSE_GLIMMER_INGRAPH_EMBED",
                "Muse-Glimmer binds the embedding table in-graph exactly where it costs no second copy "
                + "(a tied LM head, Metal, CPU); the override was removed."),
            ("TS_MUSE_GLIMMER_VENC_F32",
                "the Muse-Glimmer vision tower keeps its quantized weights wherever the backend can multiply them; "
                + "the F32 override was removed."),
            ("TS_MUSE_GLIMMER_GELU_TANH",
                "the Muse-Glimmer vision tower always uses the reference erf GELU; the tanh approximation was removed."),
            ("TS_DFLASH_FUSED", DefaultPathSwitchRemoved),
            ("TS_NEMOTRON_LINEAR_RESIDUAL_FUSED_PREFILL", ExperimentRemoved),
            ("TS_NEMOTRON_MOE_PREFILL_FUSED", ExperimentRemoved),
            ("TS_CUDA_Q8_MMA", ExperimentRemoved),
            ("TS_CUDA_MOE_PREFILL_ONDEVICE", ExperimentRemoved),
            ("TS_CUDA_GEMMA4_GLOBAL_GEMM_ATTN", ExperimentRemoved),
            ("TS_GGML_REUSE_COMPUTE_BUF", DefaultPathSwitchRemoved),
            ("TSG_USE_FLASH_ATTN_PREFILL", DefaultPathSwitchRemoved),
            ("TS_EMBEDDING_Q8_F32", ExperimentRemoved),
            ("TS_GLM_FA",
                "GLM 5.x uses flash attention wherever the backend supports it (the backend probe decides); the switch was removed."),
            ("TS_GLM_FUSED_LID",
                "GLM 5.x uses the fused lightning indexer wherever the backend supports it (the backend probe decides); the switch was removed."),
            ("TS_GLM_TOPK",
                "GLM 5.x always attends through the indexer's sparse top-k selection past top_k; the dense A/B was removed."),
            ("TS_GLM_TP_FUSED", DefaultPathSwitchRemoved),
            ("TS_GLM_VENC_FUSED", DefaultPathSwitchRemoved),
            ("TS_JSON_GRAMMAR",
                "json_object / json_schema always decode under the JSON grammar; the prompt-and-repair fallback "
                + "was removed (a schema the grammar cannot express still falls back to the first-token constraint)."),
            ("TS_JSON_FORCE_OPEN",
                "the first-token constraint always applies where no grammar can be built for the requested format; "
                + "the switch was removed."),
        };

        private const string CpuPreviousKernelsRemoved =
            "the pure-C# cpu backend always runs its current kernels; the switch back to the previous ones was removed.";

        private const string DiffusionCpuLegacyRemoved =
            "DiffusionGemma on the cpu backend always runs its batched CPU stages "
            + "(DIFFUSION_NO_PKV=1 still turns the prompt-KV cache off).";

        private const string MlxKernelSwitchRemoved =
            "the MLX backend always runs its current kernels; the switch to an earlier or experimental one was removed.";

        private const string CudaKernelSwitchRemoved =
            "the direct cuda backend always runs its current kernels and CUDA graphs where the shape allows; "
            + "the switch was removed.";

        private const string FusedGraphSwitchRemoved =
            "the GGML backends always build, keep and replay their whole-model graphs where the backend supports it; "
            + "the switch was removed.";

        private const string DefaultPathSwitchRemoved =
            "the path it switched off always runs where the backend supports it (its fallback still serves the "
            + "shapes it declines); the switch was removed.";

        private const string Qwen4ExpModelRemoved =
            "Qwen 3.8 Flash Next runs on the shared decoder runtime on the GGML backends and cpu; the switch of the "
            + "removed Qwen4ExpModel (its token span, arenas and holders) was removed with it.";

        private const string ExperimentRemoved =
            "the experimental path it selected (never the default) was removed.";

        private const string GptOssSwitchRemoved =
            "GPT-OSS always takes its fused and batched paths where the backend supports them; the switch was removed.";

        private const string PerModelBatchedRemoved =
            "--no-continuous-batching (TS_SCHED_DISABLE_BATCHED=1) forces the per-sequence path for every model.";

        /// <summary>Throw for the first removed environment variable that is set, naming what
        /// to do instead.</summary>
        /// <exception cref="ArgumentException">A removed variable is set.</exception>
        public static void RejectRemovedEnvironment()
        {
            foreach ((string name, string advice) in RemovedEnvironmentVariables)
            {
                if (!string.IsNullOrEmpty(Environment.GetEnvironmentVariable(name)))
                    throw new ArgumentException($"{name} was removed: {advice} Unset it.");
            }
        }

        /// <summary>
        /// The error for <paramref name="arg"/> when it names a removed flag, in either the
        /// spaced (<c>--flag value</c>) or the joined (<c>--flag=value</c>) spelling,
        /// case-insensitively; null otherwise. A bare config-file key (no leading dashes)
        /// is matched too, so the file and the command line share one message.
        /// </summary>
        public static string? Describe(string? arg)
        {
            if (string.IsNullOrWhiteSpace(arg))
                return null;

            string name = arg.Trim();
            int equals = name.IndexOf('=');
            if (equals >= 0)
                name = name.Substring(0, equals);
            if (!name.StartsWith("--", StringComparison.Ordinal))
                name = "--" + name;

            foreach ((string flag, string advice) in RemovedFlags)
            {
                if (name.Equals(flag, StringComparison.OrdinalIgnoreCase))
                    return $"{flag} was removed: {advice}";
            }
            return null;
        }

        /// <summary>
        /// Throw for the first removed flag in <paramref name="args"/>. Only tokens that
        /// start with <c>--</c> are checked, so a plain value such as a prompt reading
        /// "offload-cpu" is left alone.
        /// </summary>
        /// <exception cref="ArgumentException">A removed flag was present.</exception>
        public static void RejectRemoved(IReadOnlyList<string>? args)
        {
            if (args == null)
                return;
            foreach (string arg in args)
            {
                if (arg == null || !arg.StartsWith("--", StringComparison.Ordinal))
                    continue;
                if (Describe(arg) is { } message)
                    throw new ArgumentException(message);
            }
        }
    }
}
