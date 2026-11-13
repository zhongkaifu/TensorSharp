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
            // --backend cuda loads the direct-CUDA whole-model engine (Qwen4ExpCudaModel, the Q4e kernels); every other
            // backend the shared decoder runtime (Qwen4ExpRuntimeModel). MLX has no kernels for this architecture and is
            // refused up front.
            Factory = c => c.Backend switch
            {
                BackendType.Cuda => new Qwen4ExpCudaModel(c.GgufPath, c.TpDegree, c.TpGroup, c.LayerSplitDegree, c.DraftModelPath),
                BackendType.Mlx => throw new NotSupportedException(MlxRefusal),
                _ => new Qwen4ExpRuntimeModel(c.GgufPath, c.Backend, c.DraftModelPath, c.LayerSplitDegree, c.TpDegree, c.TpGroup),
            },
            ProjectorFileHints = new[] { "*mmproj*.gguf" },

            Serving = new ServingContract
            {
                // The shared runtime serves every backend but cuda, where the direct engine's placement rules apply; the
                // contract states both (a shared-runtime contract would claim the runtime's --tp on cuda).
                Implementation = ServingImplementation.Bespoke,
                Backends = new[] { BackendType.Cpu, BackendType.Cuda, BackendType.GgmlCpu, BackendType.GgmlMetal,
                    BackendType.GgmlCuda, BackendType.GgmlVulkan },
                HasRoutedExperts = true,
                // The runtime's slots on every backend; the direct-CUDA engine's slots.
                ContinuousBatching = MechanismSupport.On(BackendType.Cpu, BackendType.Cuda, BackendType.GgmlCpu, BackendType.GgmlMetal,
                    BackendType.GgmlCuda, BackendType.GgmlVulkan),
                TensorParallel = MechanismSupport.On(ServingBackends.GgmlMultiDevice).KeepsBatching()
                    .ElsewhereGap("the direct-CUDA qwen4exp engine places whole layers; use --layer-split N instead of --tp."),
                LayerSplit = MechanismSupport.On(BackendType.GgmlCuda, BackendType.GgmlVulkan, BackendType.Cuda).KeepsBatching(),
                // The runtime's staged graphs do not run per-rank graphs yet, and qwen4exp runs only as graphs.
                HybridParallel = MechanismSupport.None("qwen4exp runs --tp and --layer-split as separate placement modes."),
                ExpertOffload = MechanismSupport.On(BackendType.Cpu, BackendType.GgmlCpu, BackendType.GgmlMetal, BackendType.GgmlCuda,
                        BackendType.GgmlVulkan)
                    .ElsewhereGap("the direct-CUDA qwen4exp engine keeps every expert resident; use a GGML backend for host experts."),
            },
            SupportsDistributedTensorParallel = true,
            DistributedTensorParallelBackends = new[] { BackendType.GgmlCuda },
            SupportsDistributedLayerSplit = true,
            DistributedLayerSplitBackends = new[] { BackendType.GgmlCuda },

        };
    }
}
