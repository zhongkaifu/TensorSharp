// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using TensorSharp.Models.Architecture;

namespace TensorSharp.Models
{
    /// <summary>Qwen 3.5 / Qwen 3.6 (Gated DeltaNet + MoE) architecture plug-in.</summary>
    internal static class Qwen35Architecture
    {
        public static ModelArchitectureDescriptor Descriptor { get; } = new()
        {
            Id = "qwen35",
            DisplayName = "Qwen 3.5 / Qwen 3.6 (Gated DeltaNet + MoE)",
            Aliases = new[] { "qwen35", "qwen35moe", "qwen3next" },
            // The shared decoder runtime (Qwen35RuntimeModel) serves every placement on every backend: one device,
            // --tp, whole-layer stages (local or multi-node), their hybrid, and routed experts on the host. --backend
            // cuda on one GPU with a checkpoint the direct-CUDA engine serves (dense NVFP4; TS_Q35_NATIVE=0 declines it)
            // runs that engine instead.
            Factory = c => Qwen35CudaModel.ServesLoad(c)
                ? new Qwen35CudaModel(c.GgufPath, c.DraftModelPath)
                : new Qwen35RuntimeModel(c.GgufPath, c.Backend, c.TpDegree, c.TpGroup, c.LayerSplitDegree, c.DraftModelPath),
            ProjectorFileHints = new[]
            {
                "Qwen3.5-mmproj-F16.gguf",
                "Ternary-Bonsai-2-27B-mmproj-BF16.gguf",
                "Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf",
            },
            // A multi-node --layer-split runs on the shared decoder runtime: each node loads only its layers, and its
            // stages may be --tp groups of their own size (nodes of different shapes).
            SupportsDistributedLayerSplit = true,
            DistributedLayerSplitBackends = new[] { BackendType.GgmlCuda, BackendType.GgmlVulkan, BackendType.Cuda },
            Serving = ServingContract.SharedRuntime(hasRoutedExperts: true),
        };
    }
}
