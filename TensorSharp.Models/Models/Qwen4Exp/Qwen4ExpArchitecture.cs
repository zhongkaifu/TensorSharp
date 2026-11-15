// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using TensorSharp.Models.Architecture;
using TensorSharp.Runtime;

namespace TensorSharp.Models
{
    /// <summary>Qwen3.8-Flash-Next architecture plug-in: hyper-connections, PLE n-gram
    /// embeddings, Qwen Sparse Attention and Gated DeltaNet over a 512-expert MoE.</summary>
    internal static class Qwen4ExpArchitecture
    {
        /// <summary>
        /// Why --backend mlx is refused, said before anything is mapped. The model runs only as the
        /// GGML token span: Qwen Sparse Attention's indexer, the hyper-connections, the n-gram PLE
        /// block and the IQ2_XS / IQ3_XXS experts of the shipped quants have no MLX kernels, and the
        /// MLX loader would preload the 28.8 GB n-gram table (past its 2 GB wrap limit) and mlock
        /// the experts that this model is meant to read from the SSD. It used to fail exactly there,
        /// with an Int32 overflow, deep inside weight loading.
        /// </summary>
        internal const string MlxRefusal =
            "Qwen3.8-Flash-Next (qwen4exp) has no MLX implementation: its sparse-attention indexer, " +
            "hyper-connections, n-gram embedding table and IQ2_XS/IQ3_XXS experts have no MLX kernels. " +
            "On Apple silicon use --backend ggml_metal, which also reads the n-gram table and the " +
            "offloaded experts from the SSD on demand.";

        public static ModelArchitectureDescriptor Descriptor { get; } = new()
        {
            Id = Qwen4ExpFormat.ArchitectureId,
            DisplayName = "Qwen3.8-Flash-Next",
            Aliases = new[] { Qwen4ExpFormat.ArchitectureId },
            // The shared decoder runtime (Qwen4ExpRuntimeModel) serves every backend, --backend cuda included (its graphs
            // run on the CUDA graph executor there). MLX has no kernels for this architecture and is refused up front.
            Factory = c => c.Backend == BackendType.Mlx
                ? throw new NotSupportedException(MlxRefusal)
                : new Qwen4ExpRuntimeModel(c.GgufPath, c.Backend, c.DraftModelPath, c.LayerSplitDegree, c.TpDegree, c.TpGroup),
            ProjectorFileHints = new[] { "*mmproj*.gguf" },

            Serving = new ServingContract
            {
                // The shared runtime on every backend but MLX; not ServingContract.SharedRuntime because the runtime runs
                // --tp and --layer-split only as separate placement modes for this family (no hybrid).
                Implementation = ServingImplementation.Bespoke,
                Backends = new[] { BackendType.Cpu, BackendType.Cuda, BackendType.GgmlCpu, BackendType.GgmlMetal,
                    BackendType.GgmlCuda, BackendType.GgmlVulkan },
                HasRoutedExperts = true,
                ContinuousBatching = MechanismSupport.On(BackendType.Cpu, BackendType.Cuda, BackendType.GgmlCpu, BackendType.GgmlMetal,
                    BackendType.GgmlCuda, BackendType.GgmlVulkan),
                // Per-rank graphs: ggml's multi-device backends, and the CUDA graph executor's ranks on --backend cuda.
                TensorParallel = MechanismSupport.On(BackendType.GgmlCuda, BackendType.GgmlVulkan, BackendType.Cuda).KeepsBatching(),
                LayerSplit = MechanismSupport.On(BackendType.GgmlCuda, BackendType.GgmlVulkan, BackendType.Cuda).KeepsBatching(),
                // The runtime's staged graphs do not run per-rank graphs yet, and qwen4exp runs only as graphs.
                HybridParallel = MechanismSupport.None("qwen4exp runs --tp and --layer-split as separate placement modes."),
                ExpertOffload = MechanismSupport.On(BackendType.Cpu, BackendType.Cuda, BackendType.GgmlCpu, BackendType.GgmlMetal,
                    BackendType.GgmlCuda, BackendType.GgmlVulkan),
            },
            SupportsDistributedTensorParallel = true,
            DistributedTensorParallelBackends = new[] { BackendType.GgmlCuda },
            SupportsDistributedLayerSplit = true,
            DistributedLayerSplitBackends = new[] { BackendType.GgmlCuda },

        };
    }
}
