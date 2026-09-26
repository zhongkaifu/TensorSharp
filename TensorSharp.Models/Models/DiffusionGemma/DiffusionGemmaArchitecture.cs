// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using TensorSharp.Models.Architecture;

namespace TensorSharp.Models
{
    /// <summary>Diffusion Gemma (diffusion LM) architecture plug-in.</summary>
    internal static class DiffusionGemmaArchitecture
    {
        public static ModelArchitectureDescriptor Descriptor { get; } = new()
        {
            Id = "diffusion-gemma",
            DisplayName = "Diffusion Gemma (diffusion LM)",
            Aliases = new[] { "diffusion-gemma", "diffusion_gemma" },
            Factory = c => new DiffusionGemmaModel(c.GgufPath, c.Backend),
            // The factory never passes the degree or group on, so without this
            // declaration --tp N was dropped in silence and a distributed group
            // was left waiting on collectives this rank never issues.
            MultiGpu = MultiGpuMode.SingleDevice,
            MultiGpuLimitation =
                "diffusion-gemma has no tensor-parallel or layer-split path; it runs on one GPU and extra GPUs stay idle.",
        };
    }
}
