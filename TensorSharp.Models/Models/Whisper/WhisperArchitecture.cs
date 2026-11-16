// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using TensorSharp.Models.Architecture;
using TensorSharp.Runtime;

namespace TensorSharp.Models.Whisper
{
    /// <summary>OpenAI Whisper speech recognition (GGUFs from eng/audio-models/whisper_to_gguf.py).</summary>
    internal static class WhisperArchitecture
    {
        public static ModelArchitectureDescriptor Descriptor { get; } = new()
        {
            Id = WhisperModel.ArchitectureId,
            DisplayName = "Whisper",
            Aliases = new[] { WhisperModel.ArchitectureId },
            Factory = c => new WhisperModel(c.GgufPath, c.Backend),
            Serving = new ServingContract
            {
                Implementation = ServingImplementation.Bespoke,
                Backends = new[] { BackendType.Cpu, BackendType.Cuda },
                Autoregressive = false,
                HasRoutedExperts = false,
                ContinuousBatching = MechanismSupport.NotApplicable("transcribes one clip at a time; requests are serialized by the model"),
                TensorParallel = MechanismSupport.None("whisper has no tensor-parallel path; it runs on one GPU and extra GPUs stay idle."),
                LayerSplit = MechanismSupport.None("whisper has no layer-split path; it runs on one GPU and extra GPUs stay idle."),
                HybridParallel = MechanismSupport.None("whisper has no multi-GPU placement to combine."),
                ExpertOffload = MechanismSupport.NotApplicable("no routed experts"),
            },
            SupportsDistributedTensorParallel = false,
        };
    }
}
