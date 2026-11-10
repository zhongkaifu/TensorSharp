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
            // --backend cuda on a dense NVFP4 checkpoint runs the direct-CUDA engine
            // (TS_Q35_NATIVE=0 keeps the managed path).
            Factory = c =>
            {
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
            Serving = new ServingContract
            {
                Implementation = ServingImplementation.Bespoke,
                HasRoutedExperts = true,
                ContinuousBatching = MechanismSupport.On(ServingBackends.All),
                TensorParallel = MechanismSupport.On(ServingBackends.PerOpTensorParallel),
                LayerSplit = MechanismSupport.None("Qwen 3.5 / 3.6 does not implement whole-layer placement; use --tp N."),
                HybridParallel = MechanismSupport.None("Qwen 3.5 / 3.6 does not implement whole-layer placement, so it cannot combine it with --tp."),
                ExpertOffload = MechanismSupport.On(ServingBackends.Ggml).ElsewhereGap(
                    "Qwen 3.5 MoE offloads routed experts only on the GGML backends; cpu, cuda and mlx keep them in their stacked-expert buffers."),
            },
        };
    }
}
