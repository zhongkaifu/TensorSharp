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
    /// <summary>Nemotron-H architecture plug-in.</summary>
    internal static class NemotronArchitecture
    {
        /// <summary>The family on the shared decoder runtime (tests and the validation tools build it directly).</summary>
        internal static ModelBase CreateRuntimeModel(string ggufPath, BackendType backend, int tpDegree = 1, ITensorParallelGroup tpGroup = null,
            int layerSplitDegree = 1, string draftModelPath = null)
            => new NemotronRuntimeModel(ggufPath, backend, tpDegree, tpGroup, layerSplitDegree, draftModelPath);

        public static ModelArchitectureDescriptor Descriptor { get; } = new()
        {
            Id = "nemotron_h",
            DisplayName = "Nemotron-H",
            Aliases = new[] { "nemotron_h", "nemotron_h_moe", "nemotron_h_omni" },
            // On the shared decoder runtime (NemotronRuntimeModel): one device, --tp, whole-layer stages and their hybrid, continuous batching,
            // the prefix cache, checkpoints, host-served experts, speculation (n-gram, the Nemotron 3.5 MTP head, DFlash / DSpark
            // drafters).
            Factory = c => new NemotronRuntimeModel(c.GgufPath, c.Backend, c.TpDegree, c.TpGroup, c.LayerSplitDegree, c.DraftModelPath),
            ProjectorFileHints = new[] { "*Nemotron*mmproj*.gguf", "*mmproj*Nemotron*.gguf" },
            Serving = new ServingContract
            {
                Implementation = ServingImplementation.SharedRuntime,
                HasRoutedExperts = true,
                ContinuousBatching = MechanismSupport.On(ServingContract.AllBackends.ToArray()),
                TensorParallel = MechanismSupport.On(ServingContract.MultiDeviceBackends.ToArray()).KeepsBatching(),
                LayerSplit = MechanismSupport.On(ServingContract.MultiDeviceBackends.ToArray()).KeepsBatching(),
                // The runtime's hybrid: every whole-layer stage runs its layers as per-rank graphs over its own group.
                HybridParallel = MechanismSupport.On(ServingContract.MultiDeviceBackends.ToArray()).KeepsBatching(),
                // --n-cpu-moe / --cpu-moe: an offloaded layer's routed experts stay in host memory and run at the graph's host
                // seam (the tensor executors' expert host; GGML's host MoE segment): one projection, relu(s up)^2, each route's
                // NVFP4 up scale s applied before the square. Per-rank graphs keep their experts on the devices.
                ExpertOffload = MechanismSupport.On(ServingContract.AllBackends.ToArray())
                    .When((tp, _) => tp > 1
                        ? "Nemotron-H serves --n-cpu-moe / --cpu-moe on one device or over --layer-split stages; its tensor-parallel " +
                          "graphs keep the routed experts on the devices. Remove --tp N, or the option."
                        : null),
            },
        };
    }
}
