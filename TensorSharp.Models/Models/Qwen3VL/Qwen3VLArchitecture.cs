// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using TensorSharp.Models.Architecture;

namespace TensorSharp.Models
{
    /// <summary>Qwen3-VL (dense and MoE) architecture plug-in.</summary>
    internal static class Qwen3VLArchitecture
    {
        public static ModelArchitectureDescriptor Descriptor { get; } = new()
        {
            Id = "qwen3vl",
            DisplayName = "Qwen3-VL (dense and MoE)",
            // llama.cpp's converter labels the dense checkpoints qwen3vl and the mixture-of-experts ones qwen3vlmoe.
            Aliases = new[] { "qwen3vl", "qwen3vlmoe" },
            // The shared decoder runtime (Qwen3VLModel) serves every placement; DeepStack adds a
            // per-step input to the first layers' graphs, so a media prompt runs as graphs.
            Factory = c => new Qwen3VLModel(c.GgufPath, c.Backend, c.TpDegree, c.TpGroup, c.LayerSplitDegree),
            ProjectorFileHints = new[]
            {
                "mmproj-Qwen3VL-30B-A3B-Instruct-F16.gguf",
                "mmproj-Qwen3VL-30B-A3B-Instruct-Q8_0.gguf",
                "mmproj-Qwen3VL-30B-A3B-Thinking-F16.gguf",
                "mmproj-Qwen3VL-30B-A3B-Thinking-Q8_0.gguf",
                "mmproj-F16.gguf",
                "mmproj-BF16.gguf",
            },
            Serving = ServingContract.SharedRuntime(hasRoutedExperts: true),
        };
    }
}
