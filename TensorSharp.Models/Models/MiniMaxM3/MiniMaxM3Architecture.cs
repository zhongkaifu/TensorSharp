// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using TensorSharp.Models.Architecture;

namespace TensorSharp.Models
{
    /// <summary>
    /// MiniMax-M3.1 (GGUF architecture <c>minimax-m3</c>): GQA attention with MiniMax Sparse
    /// Attention (a block-sparse lightning indexer on every layer), three leading dense layers and
    /// a 128-expert sigmoid-routed MoE with a shared expert. It runs only on the TensorSharp native
    /// whole-model executor, which either places whole layers across local GPUs (--layer-split) or
    /// splits every layer over them (--tp, ggml's meta backend).
    /// </summary>
    internal static class MiniMaxM3Architecture
    {
        /// <summary>What <c>--tp</c> runs. MiniMax-M3 has four KV heads and its sparse attention
        /// cannot be split by head on ggml's meta backend, so attention is mirrored.</summary>
        internal const string TensorParallelPlacement =
            "minimax-m3 (MiniMax-M3.1) splits every layer over the GPUs of one process with --tp N on ggml_cuda " +
            "(ggml's meta backend): attention, its sparse-attention indexer and the K, V and index-key caches are " +
            "mirrored on every GPU, the dense, routed and shared-expert FFNs are split along n_ff and the output head " +
            "by vocabulary rows. Whole-layer placement (--layer-split N) remains available separately.";

        /// <summary>Why a distributed group is refused: the executor's tensor parallelism is one process.</summary>
        internal const string DistributedRefusal =
            "minimax-m3 (MiniMax-M3.1) runs tensor parallelism in one process only (--tp N on ggml_cuda); distributed " +
            "--tp-node-id/--tp-peers groups are not implemented for it.";

        /// <summary>Tensor parallelism needs the meta backend over CUDA GPUs.</summary>
        internal const string TensorParallelBackendRefusal =
            "MiniMax-M3.1 tensor parallelism (--tp) runs on --backend ggml_cuda only (ggml's meta backend over CUDA GPUs).";

        /// <summary>A drafter under --tp is refused rather than run untested through the meta backend.</summary>
        internal const string TensorParallelDraftRefusal =
            "MiniMax-M3.1's DSpark drafter (--draft-model) has not been validated under --tp (ggml's meta backend); " +
            "use --layer-split N with --draft-model, or --tp N without it.";

        internal const string MlxRefusal =
            "MiniMax-M3.1 (minimax-m3) has no MLX implementation: its sparse-attention indexer, block-sparse " +
            "attention and NVFP4 experts run only in the TensorSharp ggml executor. Use --backend ggml_cuda.";

        internal const string MetalRefusal =
            "MiniMax-M3.1 (minimax-m3) has no Metal implementation: its native executor and MiniMax Sparse " +
            "Attention kernels exist for CUDA (and the CPU backend as a correctness path) only. " +
            "Use --backend ggml_cuda.";

        internal const string VulkanRefusal =
            "MiniMax-M3.1 (minimax-m3) has no Vulkan implementation: its native executor's MiniMax Sparse " +
            "Attention kernels exist for CUDA (and the CPU backend as a correctness path) only. " +
            "Use --backend ggml_cuda.";

        internal const string DirectCudaRefusal =
            "MiniMax-M3.1 (minimax-m3) has no direct-CUDA engine (--backend cuda); it runs on the TensorSharp " +
            "ggml executor. Use --backend ggml_cuda.";

        internal const string ManagedCpuRefusal =
            "MiniMax-M3.1 (minimax-m3) has no pure C# implementation (--backend cpu); it runs only on the " +
            "TensorSharp native executor. Use --backend ggml_cuda, or --backend ggml_cpu for the native " +
            "executor on the CPU (a correctness path, not a serving path).";

        /// <summary>Why a drafter attached after the load is refused: the native executor loads the
        /// DSpark drafter with the target (beside the output head it reads, with its context
        /// committed inside the trunk graphs), and the generic managed DFlash host cannot run it.</summary>
        internal const string DraftModelRefusal =
            "MiniMax-M3.1 (minimax-m3) attaches its DSpark drafter only while the model loads (--draft-model / " +
            "TS_SPEC_DRAFT_MODEL), on the native executor; the generic DFlash drafter host does not run it (it " +
            "assumes SwiGLU, full-width RoPE and no Markov head). n-gram speculation (--spec) needs no drafter.";

        /// <summary>The native option that selects the attention path (see <see cref="MsaModes"/>).</summary>
        internal const string MsaModeVariable = "TS_MINIMAX_M3_MSA";

        /// <summary>
        /// Values <see cref="MsaModeVariable"/> accepts: <c>auto</c> (the default: dense attention
        /// while every query sits below top_k x block_size, where it is exactly the sparse result,
        /// block-sparse otherwise: through TensorSharp's MSA kernels on CUDA GPUs, through upstream
        /// ggml ops on any other device), <c>dense</c> (dense everywhere: debugging only, it changes the
        /// model's results past that point), <c>ggml</c> (block-sparse through upstream ggml ops) and
        /// <c>kernel</c> (block-sparse through TensorSharp's CUDA kernels).
        /// </summary>
        internal static readonly string[] MsaModes = { "auto", "dense", "ggml", "kernel" };

        public static ModelArchitectureDescriptor Descriptor { get; } = new()
        {
            Id = MiniMaxM3Model.ArchitectureId,
            DisplayName = "MiniMax-M3.1",
            Aliases = new[] { MiniMaxM3Model.ArchitectureId, "minimax_m3" },
            Factory = c => new MiniMaxM3Model(c.GgufPath, c.Backend, c.TpDegree, c.TpGroup,
                c.LayerSplitDegree, c.DraftModelPath),
            Serving = new ServingContract
            {
                Implementation = ServingImplementation.Bespoke,
                Backends = new[] { BackendType.GgmlCuda, BackendType.GgmlCpu },
                HasRoutedExperts = true,
                // The native executor batches decode and prefill over per-sequence slots in every placement.
                ContinuousBatching = MechanismSupport.On(BackendType.GgmlCuda, BackendType.GgmlCpu),
                TensorParallel = MechanismSupport.On(BackendType.GgmlCuda).KeepsBatching(),
                LayerSplit = MechanismSupport.On(BackendType.GgmlCuda).KeepsBatching(),
                HybridParallel = MechanismSupport.None(
                    "MiniMax-M3.1's native executor runs --tp and --layer-split as separate placement modes."),
                ExpertOffload = MechanismSupport.On(BackendType.GgmlCuda, BackendType.GgmlCpu),
            },
            SupportsDistributedTensorParallel = false,
            DescribeMultiGpuPlacement = n =>
                $"  Multi-GPU: MiniMax-M3.1 uses {n} GPUs by LAYER SPLIT (whole-layer placement in its native " +
                "executor, one graph per ubatch spanning the GPUs); --tp N splits every layer instead.",
            ApplyNativeTunables = c => ValidateLoad(c.Backend, c.TpGroup, c.DraftModelPath),
            ProjectorFileHints = new[] { "*mmproj*MiniMax*M3*.gguf", "*mmproj*.gguf" },
        };

        /// <summary>
        /// Every refusal that needs no checkpoint, said before a 232 GiB model is mapped: the
        /// backend, a distributed tensor-parallel group, a block-quantized K/V cache, a draft model
        /// that is not a DSpark drafter (<see cref="MiniMaxM3Dspark.ValidateFile"/>) and an unknown
        /// <see cref="MsaModeVariable"/>.
        /// </summary>
        internal static void ValidateLoad(BackendType backend, ITensorParallelGroup tpGroup, string draftModelPath)
        {
            string backendRefusal = backend switch
            {
                BackendType.GgmlCuda or BackendType.GgmlCpu => null,
                BackendType.Mlx => MlxRefusal,
                BackendType.GgmlMetal => MetalRefusal,
                BackendType.GgmlVulkan => VulkanRefusal,
                BackendType.Cuda => DirectCudaRefusal,
                BackendType.Cpu => ManagedCpuRefusal,
                _ => $"MiniMax-M3.1 (minimax-m3) does not run on --backend {backend}. Use --backend ggml_cuda.",
            };
            if (backendRefusal != null)
                throw new NotSupportedException(backendRefusal);
            if (tpGroup != null)
                throw new NotSupportedException(DistributedRefusal + " Start without --tp-node-id/--tp-peers.");
            KvCacheDtype requested = KvCacheDtypeConfig.Current;
            if (requested.IsBlockQuantized())
                throw new NotSupportedException(BlockQuantizedKvCacheError(requested));
            if (!string.IsNullOrWhiteSpace(draftModelPath))
                MiniMaxM3Dspark.ValidateFile(draftModelPath.Trim());
            string msa = Environment.GetEnvironmentVariable(MsaModeVariable);
            if (!string.IsNullOrWhiteSpace(msa) && Array.IndexOf(MsaModes, msa.Trim()) < 0)
                throw new NotSupportedException(
                    $"{MsaModeVariable}={msa} is not one of {string.Join(", ", MsaModes)} (unset selects auto).");
        }

        internal static string BlockQuantizedKvCacheError(KvCacheDtype requested)
            => $"KV_CACHE_DTYPE={requested.ToShortString()} is not supported by MiniMax-M3.1 (minimax-m3). Its native " +
               "executor keeps every sequence's K, V and MiniMax Sparse Attention index-key rows as F16 tensors " +
               "that the indexer and the block-sparse attention read directly, with no block-dequantize step. " +
               "Unset KV_CACHE_DTYPE or set it to f16.";
    }
}
