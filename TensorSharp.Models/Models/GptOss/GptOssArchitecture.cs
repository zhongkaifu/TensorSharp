// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using TensorSharp.Models.Architecture;

namespace TensorSharp.Models
{
    /// <summary>GPT-OSS architecture plug-in.</summary>
    internal static class GptOssArchitecture
    {
        public static ModelArchitectureDescriptor Descriptor { get; } = new()
        {
            Id = "gptoss",
            DisplayName = "GPT-OSS",
            Aliases = new[] { "gptoss", "gpt-oss" },
            Factory = c => new GptOssModel(c.GgufPath, c.Backend, c.TpDegree, c.TpGroup),
            Serving = new ServingContract
            {
                Implementation = ServingImplementation.Bespoke,
                HasRoutedExperts = true,
                ContinuousBatching = MechanismSupport.On(ServingBackends.All),
                TensorParallel = MechanismSupport.On(ServingBackends.PerOpTensorParallel),
                LayerSplit = MechanismSupport.None("GPT-OSS does not implement whole-layer placement; use --tp N."),
                HybridParallel = MechanismSupport.None("GPT-OSS does not implement whole-layer placement, so it cannot combine it with --tp."),
                ExpertOffload = MechanismSupport.On(ServingBackends.Ggml).ElsewhereGap(
                    "GPT-OSS runs host experts through the GGML CPU backend; cpu, cuda and mlx keep every expert on their own device."),
            },
        };
    }
}
