// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using TensorSharp.Models.Architecture;

namespace TensorSharp.Models
{
    /// <summary>Gemma 4 architecture plug-in.</summary>
    internal static class Gemma4Architecture
    {
        public static ModelArchitectureDescriptor Descriptor { get; } = new()
        {
            Id = "gemma4",
            DisplayName = "Gemma 4",
            Aliases = new[] { "gemma4" },
            SupportsWeightStreaming = true,
            Factory = c => new Gemma4Model(c.GgufPath, c.Backend, c.TpDegree, c.TpGroup, c.WeightStreaming, c.DraftModelPath, c.MemoryPolicy),
            ProjectorFileHints = new[] { "gemma-4-mmproj-F16.gguf" },
        };
    }
}
