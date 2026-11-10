// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using TensorSharp.Models.Architecture;

namespace TensorSharp.Models
{
    /// <summary>Gemma 4 architecture plug-in.</summary>
    internal static class Gemma4Architecture
    {
        public static ModelArchitectureDescriptor Descriptor { get; } = new()
        {
            Id = "gemma4",
            DisplayName = "Gemma 4",
            Aliases = new[] { "gemma4" },
            Factory = c => new Gemma4Model(c.GgufPath, c.Backend, c.TpDegree, c.TpGroup),
            ProjectorFileHints = new[] { "gemma-4-mmproj-F16.gguf" },
            Serving = new ServingContract
            {
                Implementation = ServingImplementation.Bespoke,
                HasRoutedExperts = true,
                ContinuousBatching = MechanismSupport.On(ServingBackends.All),
                // The per-op TP forward has no batched variant: concurrent requests take turns.
                TensorParallel = MechanismSupport.On(ServingBackends.PerOpTensorParallel),
                LayerSplit = MechanismSupport.None("Gemma 4 does not implement whole-layer placement; use --tp N."),
                HybridParallel = MechanismSupport.None("Gemma 4 does not implement whole-layer placement, so it cannot combine it with --tp."),
                ExpertOffload = MechanismSupport.On(ServingBackends.Ggml).ElsewhereGap(
                    "Gemma 4 offloads routed experts only on the GGML backends; cpu, cuda and mlx serve every expert from their stacked-expert device buffers."),
            },
        };
    }
}
