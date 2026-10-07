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
            Id = Qwen4ExpModel.ArchitectureId,
            DisplayName = "Qwen3.8-Flash-Next",
            Aliases = new[] { Qwen4ExpModel.ArchitectureId },
            // --backend cuda loads the direct-CUDA whole-model engine; the GGML backends the native
            // token spans. MLX has no kernels for this architecture and is refused up front.
            Factory = c => c.Backend switch
            {
                BackendType.Cuda => new Qwen4ExpCudaModel(c.GgufPath, c.TpDegree, c.TpGroup, c.LayerSplitDegree, c.DraftModelPath),
                BackendType.Mlx => throw new NotSupportedException(MlxRefusal),
                _ => new Qwen4ExpModel(c.GgufPath, c.Backend, c.TpDegree, c.TpGroup, c.LayerSplitDegree, c.DraftModelPath,
                    c.ProjectorPath),
            },
            ProjectorFileHints = new[] { "*mmproj*.gguf" },

            MultiGpu = MultiGpuMode.TensorParallel,
            SupportsLayerSplit = true,
            // The direct-CUDA engine places whole layers too.
            LayerSplitBackends = new[] { BackendType.GgmlCuda, BackendType.GgmlVulkan, BackendType.Cuda },
            SupportsDistributedTensorParallel = false,

        };
    }
}
