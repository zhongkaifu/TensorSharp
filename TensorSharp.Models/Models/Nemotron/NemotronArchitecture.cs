// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using TensorSharp.Models.Architecture;

namespace TensorSharp.Models
{
    /// <summary>Nemotron-H architecture plug-in.</summary>
    internal static class NemotronArchitecture
    {
        public static ModelArchitectureDescriptor Descriptor { get; } = new()
        {
            Id = "nemotron_h",
            DisplayName = "Nemotron-H",
            Aliases = new[] { "nemotron_h", "nemotron_h_moe", "nemotron_h_omni" },
            Factory = c => new NemotronModel(c.GgufPath, c.Backend, c.TpDegree, c.TpGroup, c.DraftModelPath),
            ProjectorFileHints = new[] { "*Nemotron*mmproj*.gguf", "*mmproj*Nemotron*.gguf" },
            Serving = new ServingContract
            {
                Implementation = ServingImplementation.Bespoke,
                HasRoutedExperts = true,
                ContinuousBatching = MechanismSupport.On(ServingBackends.All),
                TensorParallel = MechanismSupport.On(ServingBackends.PerOpTensorParallel),
                LayerSplit = MechanismSupport.None("Nemotron-H does not implement whole-layer placement; use --tp N."),
                HybridParallel = MechanismSupport.None("Nemotron-H does not implement whole-layer placement, so it cannot combine it with --tp."),
                // Only the GGML tensor-parallel MoE consults the host-expert placement.
                ExpertOffload = MechanismSupport.On(ServingBackends.GgmlMultiDevice)
                    .When((tp, _) => tp > 1 ? null :
                        "Nemotron-H honours --n-cpu-moe / --cpu-moe only in its GGML tensor-parallel MoE; add --tp N or remove the option.")
                    .ElsewhereGap("Nemotron-H keeps routed experts on the device outside its GGML tensor-parallel MoE."),
            },
        };
    }
}
