// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using TensorSharp.Models.Architecture;
using TensorSharp.Runtime;

namespace TensorSharp.Models
{
    /// <summary>Thinking Machines Inkling 2 (GGUF <c>inkling2</c>; Inkling 2 Mini's looped stack is <c>inkling2_loop</c>)
    /// architecture plug-in.</summary>
    internal static class Inkling2Architecture
    {
        /// <summary>The backends milestone 1 serves: whole-forward graphs on the managed executor and on GGML.</summary>
        private static readonly BackendType[] GraphBackends =
            { BackendType.Cpu, BackendType.GgmlCpu, BackendType.GgmlMetal, BackendType.GgmlCuda, BackendType.GgmlVulkan };

        internal const string BackendRefusal =
            "Inkling 2 runs as whole-forward graphs on the managed and GGML backends (--backend cpu, ggml_cpu, ggml_metal, " +
            "ggml_cuda, ggml_vulkan); --backend cuda and mlx have not run it yet.";

        private const string OneDevice =
            "Inkling 2 runs on one device so far: its loop, depth-mixed residual stream and per-layer convolution state " +
            "have no tensor-parallel or whole-layer placement yet.";

        public static ModelArchitectureDescriptor Descriptor { get; } = new()
        {
            Id = "inkling2",
            DisplayName = "Thinking Machines Inkling 2",
            Aliases = new[] { "inkling2", "inkling2_loop" },
            // Milestone 1 (docs/design/inkling2-on-shared-runtime.md): whole-forward graphs on one device.
            Factory = c => Array.IndexOf(GraphBackends, c.Backend) < 0
                ? throw new NotSupportedException(BackendRefusal)
                : new Inkling2RuntimeModel(c.GgufPath, c.Backend, c.TpDegree, c.TpGroup, c.LayerSplitDegree, c.DraftModelPath),
            // On the shared runtime, with the mechanisms it does not run yet declared as gaps (ServingContractConformanceTests
            // .KnownGaps); Inkling2RuntimeTests proves the rest.
            Serving = new ServingContract
            {
                Implementation = ServingImplementation.SharedRuntime,
                Backends = GraphBackends,
                HasRoutedExperts = true,
                ContinuousBatching = MechanismSupport.On(GraphBackends),
                TensorParallel = MechanismSupport.None(OneDevice),
                LayerSplit = MechanismSupport.None(OneDevice),
                HybridParallel = MechanismSupport.None(OneDevice),
                ExpertOffload = MechanismSupport.None(
                    "Inkling 2's routed experts (sigmoid routing, fused gate|up, SITU) have no host-expert path yet."),
            },
        };
    }
}
