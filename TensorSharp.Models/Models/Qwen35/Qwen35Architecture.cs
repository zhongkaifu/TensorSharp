// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Linq;
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
            // --backend cuda on a dense NVFP4 checkpoint runs the direct-CUDA engine
            // (TS_Q35_NATIVE=0 keeps the managed path).
            // The shared decoder runtime (Qwen35RuntimeModel) serves whole-layer stages (local or
            // multi-node), their hybrid with --tp, routed experts on the host for cuda and mlx, and
            // one device on ggml_cuda / ggml_metal / ggml_cpu / mlx for checkpoints without routed experts, Bonsai2's
            // PRISM rotations included, and routed-expert ones on ggml_cpu (see Qwen35RuntimeModel.ServesOneDevice).
            // Qwen35Model keeps routed-expert checkpoints on the other devices, its own --tp, and the other backends.
            Factory = c =>
            {
                if (Qwen35RuntimeModel.Serves(c))
                {
                    return new Qwen35RuntimeModel(c.GgufPath, c.Backend, c.TpDegree, c.TpGroup, c.LayerSplitDegree, c.DraftModelPath);
                }
                bool native = c.Backend == BackendType.Cuda && c.TpDegree <= 1 && c.TpGroup == null && Qwen35CudaModel.CanServe(c.Probe);
                return native
                    ? new Qwen35CudaModel(c.GgufPath, c.DraftModelPath)
                    : new Qwen35Model(c.GgufPath, c.Backend, c.TpDegree, c.TpGroup, c.DraftModelPath);
            },
            ProjectorFileHints = new[]
            {
                "Qwen3.5-mmproj-F16.gguf",
                "Ternary-Bonsai-2-27B-mmproj-BF16.gguf",
                "Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf",
            },
            // A multi-node --layer-split runs on the shared decoder runtime (Qwen35RuntimeModel): each node loads
            // only its layers, and its stages may be --tp groups of their own size (nodes of different shapes).
            SupportsDistributedLayerSplit = true,
            DistributedLayerSplitBackends = new[] { BackendType.GgmlCuda, BackendType.GgmlVulkan, BackendType.Cuda },
            Serving = new ServingContract
            {
                Implementation = ServingImplementation.Bespoke,
                HasRoutedExperts = true,
                ContinuousBatching = MechanismSupport.On(ServingBackends.All),
                // Qwen35Model's own tensor parallelism: per-rank fused graphs, no batched forward.
                TensorParallel = MechanismSupport.On(ServingBackends.PerOpTensorParallel),
                // The shared decoder runtime: batching survives both.
                LayerSplit = MechanismSupport.On(ServingContract.MultiDeviceBackends.ToArray()).KeepsBatching(),
                HybridParallel = MechanismSupport.On(ServingContract.MultiDeviceBackends.ToArray()).KeepsBatching(),
                // GGML backends: Qwen35Model's host-MoE seam; cuda and mlx: the shared runtime.
                ExpertOffload = MechanismSupport.On(ServingBackends.All),
            },
        };
    }
}
