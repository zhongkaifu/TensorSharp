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
    /// <summary>KRAFTON Raon-1 nano (GGUF <c>raon1nano</c>) architecture plug-in.</summary>
    internal static class RaonArchitecture
    {
        /// <summary>The backends Raon runs on: whole-forward graphs on the managed executor and on GGML.</summary>
        private static readonly BackendType[] GraphBackends =
            { BackendType.Cpu, BackendType.GgmlCpu, BackendType.GgmlMetal, BackendType.GgmlCuda, BackendType.GgmlVulkan };

        internal const string BackendRefusal =
            "Raon-1 nano runs as whole-forward graphs on the managed and GGML backends (--backend cpu, ggml_cpu, ggml_metal, " +
            "ggml_cuda, ggml_vulkan); --backend cuda and mlx have not run it yet.";

        private const string OneDevice =
            "Raon-1 nano runs on one device so far: its hyper-connection streams, per-layer embeddings and RDA state " +
            "have no tensor-parallel or whole-layer placement yet.";

        public static ModelArchitectureDescriptor Descriptor { get; } = new()
        {
            Id = "raon1nano",
            DisplayName = "KRAFTON Raon-1 nano",
            Aliases = new[] { "raon1nano" },
            Factory = c => Array.IndexOf(GraphBackends, c.Backend) < 0
                ? throw new NotSupportedException(BackendRefusal)
                : new RaonRuntimeModel(c.GgufPath, c.Backend, c.TpDegree, c.TpGroup, c.LayerSplitDegree),
            // On the shared runtime, with the placements it does not run yet declared as gaps (ServingContractConformanceTests
            // .KnownGaps); RaonRuntimeTests proves the rest.
            Serving = new ServingContract
            {
                Implementation = ServingImplementation.SharedRuntime,
                Backends = GraphBackends,
                HasRoutedExperts = false,
                ContinuousBatching = MechanismSupport.On(GraphBackends),
                TensorParallel = MechanismSupport.None(OneDevice),
                LayerSplit = MechanismSupport.None(OneDevice),
                HybridParallel = MechanismSupport.None(OneDevice),
                ExpertOffload = MechanismSupport.NotApplicable("dense: no routed experts"),
            },
        };
    }
}
