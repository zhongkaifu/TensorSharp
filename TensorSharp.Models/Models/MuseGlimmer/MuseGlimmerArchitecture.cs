// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using TensorSharp.Models.Architecture;

namespace TensorSharp.Models
{
    /// <summary>Muse-Glimmer architecture plug-in.</summary>
    internal static class MuseGlimmerArchitecture
    {
        public static ModelArchitectureDescriptor Descriptor { get; } = new()
        {
            Id = "muse-glimmer",
            DisplayName = "Muse-Glimmer",
            Aliases = new[] { "muse-glimmer", "muse_glimmer" },
            Factory = c => new MuseGlimmerModel(c.GgufPath, c.Backend, c.TpDegree, c.TpGroup, c.DraftModelPath),
            ProjectorFileHints = new[] { "*mmproj*Muse*Glimmer*.gguf" },
            Serving = new ServingContract
            {
                Implementation = ServingImplementation.Bespoke,
                HasRoutedExperts = false,
                ContinuousBatching = MechanismSupport.None(
                    "Muse-Glimmer has no batched forward; concurrent requests take turns through the per-sequence KV swap."),
                TensorParallel = MechanismSupport.On(ServingBackends.GgmlMultiDevice)
                    .When((tp, _) => tp <= 2 ? null : "Muse-Glimmer has 2 KV heads, so --tp is at most 2.")
                    .ElsewhereGap("Muse-Glimmer tensor parallelism runs on ggml_cuda or ggml_vulkan only."),
                LayerSplit = MechanismSupport.None("Muse-Glimmer does not implement whole-layer placement; use --tp 2."),
                HybridParallel = MechanismSupport.None("Muse-Glimmer does not implement whole-layer placement, so it cannot combine it with --tp."),
                ExpertOffload = MechanismSupport.NotApplicable("dense: no routed experts"),
            },
        };
    }
}
