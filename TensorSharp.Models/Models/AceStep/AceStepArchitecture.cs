// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using TensorSharp.Models.Architecture;
using TensorSharp.Runtime;

namespace TensorSharp.Models.AceStep
{
    /// <summary>ACE-Step 1.5 text to music (GGUFs from eng/audio-models/acestep_to_gguf.py).</summary>
    internal static class AceStepArchitecture
    {
        public static ModelArchitectureDescriptor Descriptor { get; } = new()
        {
            Id = AceStepModel.ArchitectureId,
            DisplayName = "ACE-Step 1.5",
            Aliases = new[] { AceStepModel.ArchitectureId, "ace_step_1.5", "ace-step" },
            Factory = c => new AceStepModel(c.GgufPath, c.Backend),
            Serving = new ServingContract
            {
                Implementation = ServingImplementation.Bespoke,
                Backends = new[] { BackendType.Cuda, BackendType.Cpu },
                Autoregressive = false,
                HasRoutedExperts = false,
                ContinuousBatching = MechanismSupport.NotApplicable("generates one song at a time; requests are serialized by the model"),
                TensorParallel = MechanismSupport.None("ace-step has no tensor-parallel path; it runs on one GPU and extra GPUs stay idle."),
                LayerSplit = MechanismSupport.None("ace-step has no layer-split path; it runs on one GPU and extra GPUs stay idle."),
                HybridParallel = MechanismSupport.None("ace-step has no multi-GPU placement to combine."),
                ExpertOffload = MechanismSupport.NotApplicable("no routed experts"),
            },
            SupportsDistributedTensorParallel = false,
        };
    }
}
