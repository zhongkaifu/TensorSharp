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
    /// <summary>GPT-OSS architecture plug-in.</summary>
    internal static class GptOssArchitecture
    {
        /// <summary>The family on the shared decoder runtime (tests and the validation tools build it directly).</summary>
        internal static ModelBase CreateRuntimeModel(string ggufPath, BackendType backend, int tpDegree = 1, ITensorParallelGroup tpGroup = null,
            int layerSplitDegree = 1, string draftModelPath = null)
            => new GptOssRuntimeModel(ggufPath, backend, tpDegree, tpGroup, layerSplitDegree, draftModelPath);

        public static ModelArchitectureDescriptor Descriptor { get; } = new()
        {
            Id = "gptoss",
            DisplayName = "GPT-OSS",
            Aliases = new[] { "gptoss", "gpt-oss" },
            // On the shared decoder runtime (GptOssRuntimeModel): one device, --tp, whole-layer stages, continuous batching, the
            // prefix cache, checkpoints, speculation (n-gram drafts, DFlash drafters).
            Factory = c => new GptOssRuntimeModel(c.GgufPath, c.Backend, c.TpDegree, c.TpGroup, c.LayerSplitDegree, c.DraftModelPath),
            Serving = new ServingContract
            {
                Implementation = ServingImplementation.SharedRuntime,
                HasRoutedExperts = true,
                ContinuousBatching = MechanismSupport.On(ServingContract.AllBackends.ToArray()),
                TensorParallel = MechanismSupport.On(ServingContract.MultiDeviceBackends.ToArray()).KeepsBatching(),
                LayerSplit = MechanismSupport.On(ServingContract.MultiDeviceBackends.ToArray()).KeepsBatching(),
                // The graph-only family has no graphs for tensor-parallel stages yet.
                HybridParallel = MechanismSupport.None("GPT-OSS runs whole-forward graphs only, which do not cover tensor-parallel stages yet; use --tp N or --layer-split N."),
                // --n-cpu-moe / --cpu-moe: an offloaded layer's routed experts stay in host memory and run at the graph's host
                // seam (GGML's host MoE segment; the tensor executors' expert host): the biased gate and up products, the clamped
                // SwiGLU, the biased down product. Per-rank graphs keep their experts on the devices.
                ExpertOffload = MechanismSupport.On(ServingContract.AllBackends.ToArray())
                    .When((tp, _) => tp > 1
                        ? "GPT-OSS serves --n-cpu-moe / --cpu-moe on one device or over --layer-split stages; its tensor-parallel " +
                          "graphs keep the routed experts on the devices. Remove --tp N, or the option."
                        : null),
            },
        };
    }
}
