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
        /// <summary>
        /// The test seam of the unified-runtime family (<see cref="MuseGlimmerRuntimeModel"/>) before the switch: tests, the
        /// KV-grow probe and benchmarks build it here from a GGUF, while <see cref="Descriptor"/>'s factory builds the legacy
        /// <see cref="MuseGlimmerModel"/> until stage S8 of docs/design/muse-glimmer-unified.md, which points the factory here.
        /// No option, switch or environment variable reaches it.
        /// </summary>
        internal static ModelBase CreateRuntimeModel(string ggufPath, BackendType backend, int tpDegree = 1, ITensorParallelGroup tpGroup = null,
            int layerSplitDegree = 1, string draftModelPath = null)
            => new MuseGlimmerRuntimeModel(ggufPath, backend, tpDegree, tpGroup, layerSplitDegree, draftModelPath);

        /// <summary>
        /// The serving seam of the unified-runtime family before the switch: set by the eng tool that runs the real server and CLI
        /// on it (eng/validation/MuseGlimmerRuntimeHost) before anything loads a model, it makes <see cref="Descriptor"/>'s factory
        /// build <see cref="MuseGlimmerRuntimeModel"/> instead of the legacy <see cref="MuseGlimmerModel"/>, so the serving gates
        /// of stage S3 (prefix reuse, concurrency, checkpoint files) run through every surface unchanged. No option, switch or
        /// environment variable sets it, and tests do not (they build the family through <see cref="CreateRuntimeModel"/>).
        /// Stage S8 makes the runtime family the factory's and deletes this.
        /// </summary>
        internal static volatile bool DescriptorBuildsTheRuntimeFamily;

        public static ModelArchitectureDescriptor Descriptor { get; } = new()
        {
            Id = "muse-glimmer",
            DisplayName = "Muse-Glimmer",
            Aliases = new[] { "muse-glimmer", "muse_glimmer" },
            Factory = c => DescriptorBuildsTheRuntimeFamily
                ? CreateRuntimeModel(c.GgufPath, c.Backend, c.TpDegree, c.TpGroup, c.LayerSplitDegree, c.DraftModelPath)
                : new MuseGlimmerModel(c.GgufPath, c.Backend, c.TpDegree, c.TpGroup, c.DraftModelPath),
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
