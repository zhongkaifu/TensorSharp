// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using TensorSharp.Models.Architecture;

namespace TensorSharp.Models
{
    /// <summary>Qwen 3.5 / Qwen 3.6 (Gated DeltaNet + MoE) architecture plug-in.</summary>
    internal static class Qwen35Architecture
    {
        public static ModelArchitectureDescriptor Descriptor { get; } = new()
        {
            Id = "qwen35",
            DisplayName = "Qwen 3.5 / Qwen 3.6 (Gated DeltaNet + MoE)",
            Aliases = new[] { "qwen35", "qwen35moe", "qwen3next" },
            SupportsWeightStreaming = true,
            Factory = c => new Qwen35Model(c.GgufPath, c.Backend, c.TpDegree, c.TpGroup, c.DraftModelPath, c.WeightStreaming),
            ProjectorFileHints = new[]
            {
                "Qwen3.5-mmproj-F16.gguf",
                "Ternary-Bonsai-2-27B-mmproj-BF16.gguf",
                "Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf",
            },
        };
    }
}
