// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System.Linq;

using TensorSharp.Models.Architecture;

namespace TensorSharp.Models
{
    /// <summary>Gemma 4 architecture plug-in.</summary>
    internal static class Gemma4Architecture
    {
        /// <summary>The family on the shared decoder runtime (tests and eng/validation/Gemma4RuntimeAB build it directly).</summary>
        internal static ModelBase CreateRuntimeModel(string ggufPath, BackendType backend, int tpDegree = 1, ITensorParallelGroup tpGroup = null,
            int layerSplitDegree = 1, string draftModelPath = null)
            => new Gemma4RuntimeModel(ggufPath, backend, tpDegree, tpGroup, layerSplitDegree, draftModelPath);

        public static ModelArchitectureDescriptor Descriptor { get; } = new()
        {
            Id = "gemma4",
            DisplayName = "Gemma 4",
            Aliases = new[] { "gemma4" },
            // On the shared decoder runtime (Gemma4RuntimeModel): one device, --tp, whole-layer stages and their hybrid,
            // continuous batching, the prefix cache, checkpoints, speculation (n-gram, the gemma4-assistant head, DFlash).
            Factory = c => new Gemma4RuntimeModel(c.GgufPath, c.Backend, c.TpDegree, c.TpGroup, c.LayerSplitDegree, c.DraftModelPath),
            ProjectorFileHints = new[] { "gemma-4-mmproj-F16.gguf" },
            // Gemma4RuntimeTests proves the mechanisms on the family's synthetic fixtures. Routed experts run from host memory on
            // every backend (GGML's host MoE segment and the tensor executors' expert host take the fused GeGLU stack).
            Serving = new ServingContract
            {
                Implementation = ServingImplementation.SharedRuntime,
                HasRoutedExperts = true,
                ContinuousBatching = MechanismSupport.On(ServingContract.AllBackends.ToArray()),
                TensorParallel = MechanismSupport.On(ServingContract.MultiDeviceBackends.ToArray()).KeepsBatching(),
                LayerSplit = MechanismSupport.On(ServingContract.MultiDeviceBackends.ToArray()).KeepsBatching(),
                // The runtime's hybrid: every whole-layer stage runs its layers as per-rank graphs over its own group, the per-layer
                // embeddings carried from stage to stage.
                HybridParallel = MechanismSupport.On(ServingContract.MultiDeviceBackends.ToArray()).KeepsBatching(),
                ExpertOffload = MechanismSupport.On(ServingContract.AllBackends.ToArray()),
            },
        };
    }
}
