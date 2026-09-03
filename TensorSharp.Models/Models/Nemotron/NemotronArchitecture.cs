// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using TensorSharp.Models.Architecture;

namespace TensorSharp.Models
{
    /// <summary>Nemotron-H architecture plug-in.</summary>
    internal static class NemotronArchitecture
    {
        public static ModelArchitectureDescriptor Descriptor { get; } = new()
        {
            Id = "nemotron_h",
            DisplayName = "Nemotron-H",
            Aliases = new[] { "nemotron_h", "nemotron_h_moe", "nemotron_h_omni" },
            Factory = c => new NemotronModel(c.GgufPath, c.Backend, c.TpDegree, c.TpGroup, c.DraftModelPath),
            ProjectorFileHints = new[] { "*Nemotron*mmproj*.gguf", "*mmproj*Nemotron*.gguf" },
        };
    }
}
