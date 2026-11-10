// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using TensorSharp.Models.Architecture;

namespace TensorSharp.Models.MiniMaxH3
{
    /// <summary>MiniMax-H3 architecture plug-in (video/image generation).</summary>
    internal static class MiniMaxH3Architecture
    {
        public static ModelArchitectureDescriptor Descriptor { get; } = new()
        {
            Id = MiniMaxH3Model.ArchitectureId,
            DisplayName = "MiniMax-H3",
            Aliases = new[] { MiniMaxH3Model.ArchitectureId, "minimax_h3" },
            Factory = c => new MiniMaxH3Model(c.GgufPath, c.Backend),

            // MiniMax-H3's published GGUFs carry ZERO metadata - no architecture string at
            // all - so they have to be recognised by their tensors. Without this detector,
            // architecture resolution fails closed.
            DetectFromTensors = MiniMaxH3Model.LooksLikeMiniMaxH3,

            // The factory never passes the degree or group on, so without this
            // declaration --tp N was dropped in silence and a distributed group
            // was left waiting on collectives this rank never issues.
            Serving = new ServingContract
            {
                Implementation = ServingImplementation.Bespoke,
                Backends = new[] { BackendType.Cpu, BackendType.Cuda, BackendType.Mlx, BackendType.GgmlCpu, BackendType.GgmlMetal, BackendType.GgmlCuda },
                Autoregressive = false,
                HasRoutedExperts = false,
                ContinuousBatching = MechanismSupport.NotApplicable("generates media, not decode steps; requests are serialized by its pipeline"),
                TensorParallel = MechanismSupport.None("minimax-h3 has no tensor-parallel path; it runs on one GPU and extra GPUs stay idle."),
                LayerSplit = MechanismSupport.None("minimax-h3 has no layer-split path; it runs on one GPU and extra GPUs stay idle."),
                HybridParallel = MechanismSupport.None("minimax-h3 has no multi-GPU placement to combine."),
                ExpertOffload = MechanismSupport.NotApplicable("no routed experts"),
            },

            // Step-distillation adapters (the LightX2V Turbo plug-ins) as minimax-h3-lora
            // configs, which declare the checkpoint partition and the sampling contract.
            LoraPluginType = LoraCliFlags.MiniMaxH3PluginType,
        };
    }
}
