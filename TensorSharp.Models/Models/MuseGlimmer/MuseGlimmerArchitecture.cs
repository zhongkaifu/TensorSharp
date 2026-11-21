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
        /// The family on the shared decoder runtime (<see cref="MuseGlimmerRuntimeModel"/>), which <see cref="Descriptor"/>'s factory
        /// builds: every placement (one device, --tp, whole-layer stages, their hybrid), continuous batching, the prefix cache and
        /// checkpoints, DFlash and n-gram speculation, images. Tests and the KV-grow probe build it here from a GGUF.
        /// </summary>
        internal static ModelBase CreateRuntimeModel(string ggufPath, BackendType backend, int tpDegree = 1, ITensorParallelGroup tpGroup = null,
            int layerSplitDegree = 1, string draftModelPath = null)
            => new MuseGlimmerRuntimeModel(ggufPath, backend, tpDegree, tpGroup, layerSplitDegree, draftModelPath);

        /// <summary>
        /// The oracle seam: the legacy <see cref="MuseGlimmerModel"/>, which the runtime family replaced (stage S8 of
        /// docs/design/muse-glimmer-unified.md). Only the oracle tests and the KV-grow probe's <c>--family legacy</c> build it, as
        /// the reference the runtime is held to until stage S9 deletes it; no option, switch or environment variable reaches it.
        /// </summary>
        internal static ModelBase CreateLegacyModel(string ggufPath, BackendType backend, int tpDegree = 1, ITensorParallelGroup tpGroup = null,
            string draftModelPath = null)
            => new MuseGlimmerModel(ggufPath, backend, tpDegree, tpGroup, draftModelPath);

        public static ModelArchitectureDescriptor Descriptor { get; } = new()
        {
            Id = "muse-glimmer",
            DisplayName = "Muse-Glimmer",
            Aliases = new[] { "muse-glimmer", "muse_glimmer" },
            // The shared decoder runtime serves every placement on every backend: one device, --tp (2 KV heads, so at most 2),
            // whole-layer stages, their hybrid; continuous batching, the prefix cache, DFlash and n-gram speculation, images.
            Factory = c => CreateRuntimeModel(c.GgufPath, c.Backend, c.TpDegree, c.TpGroup, c.LayerSplitDegree, c.DraftModelPath),
            ProjectorFileHints = new[] { "*mmproj*Muse*Glimmer*.gguf" },
            Serving = ServingContract.SharedRuntime(hasRoutedExperts: false),
        };
    }
}
