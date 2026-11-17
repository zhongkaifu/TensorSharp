// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using TensorSharp.Models.Architecture;

namespace TensorSharp.Models
{
    /// <summary>LiquidAI LFM3 (LFM3-1.7B, LFM3-Omni-2B's text decoder) architecture plug-in.</summary>
    internal static class Lfm3Architecture
    {
        public static ModelArchitectureDescriptor Descriptor { get; } = new()
        {
            Id = "lfm3",
            DisplayName = "LiquidAI LFM3",
            Aliases = new[] { "lfm3" },
            // The shared decoder runtime serves every placement on every backend: one device, --tp (channel-sharded short
            // convolutions), whole-layer stages, their hybrid; continuous batching and speculation.
            Factory = c => new Lfm3RuntimeModel(c.GgufPath, c.Backend, c.TpDegree, c.TpGroup, c.LayerSplitDegree, c.DraftModelPath),
            // LiquidAI's published Omni projector (vision + audio towers) beside the decoder.
            ProjectorFileHints = new[]
            {
                "mmproj-LFM3-Omni-2B-F16.gguf",
                "mmproj-LFM3-Omni-2B-Q8_0.gguf",
                "*mmproj*LFM3*.gguf",
            },
            Serving = ServingContract.SharedRuntime(hasRoutedExperts: false),
        };
    }
}
