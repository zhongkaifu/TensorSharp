// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using TensorSharp.Models.Architecture;
using TensorSharp.Runtime;

namespace TensorSharp.Models.Kokoro
{
    /// <summary>Kokoro-82M text to speech (GGUFs from eng/audio-models/kokoro_to_gguf.py).</summary>
    internal static class KokoroArchitecture
    {
        public static ModelArchitectureDescriptor Descriptor { get; } = new()
        {
            Id = KokoroModel.ArchitectureId,
            DisplayName = "Kokoro",
            Aliases = new[] { KokoroModel.ArchitectureId },
            Factory = c => new KokoroModel(c.GgufPath, c.Backend),
            Serving = new ServingContract
            {
                Implementation = ServingImplementation.Bespoke,
                Backends = new[] { BackendType.Cpu },
                Autoregressive = false,
                HasRoutedExperts = false,
                ContinuousBatching = MechanismSupport.NotApplicable("synthesizes one request at a time; requests are serialized by the model"),
                TensorParallel = MechanismSupport.None("kokoro runs on the CPU; it has no multi-GPU path."),
                LayerSplit = MechanismSupport.None("kokoro runs on the CPU; it has no multi-GPU path."),
                HybridParallel = MechanismSupport.None("kokoro has no multi-GPU placement to combine."),
                ExpertOffload = MechanismSupport.NotApplicable("no routed experts"),
            },
            SupportsDistributedTensorParallel = false,
        };
    }
}
