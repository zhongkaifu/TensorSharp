// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using TensorSharp.Models.Architecture;

namespace TensorSharp.Models
{
    /// <summary>Qwen3.8-Flash-Next architecture plug-in: hyper-connections, PLE n-gram
    /// embeddings, Qwen Sparse Attention and Gated DeltaNet over a 512-expert MoE.</summary>
    internal static class Qwen4ExpArchitecture
    {
        public static ModelArchitectureDescriptor Descriptor { get; } = new()
        {
            Id = Qwen4ExpModel.ArchitectureId,
            DisplayName = "Qwen3.8-Flash-Next",
            Aliases = new[] { Qwen4ExpModel.ArchitectureId },
            Factory = c => new Qwen4ExpModel(c.GgufPath, c.Backend, c.TpDegree, c.TpGroup, c.LayerSplitDegree, c.DraftModelPath),
            ProjectorFileHints = new[] { "*mmproj*.gguf" },

            // Not tensor-parallel: none of its weights are sharded, and its decode is one
            // persisted single-device GGML graph per token whose GDN/PLE recurrent state
            // lives in device buffers owned by a single backend. It CAN spread whole
            // layers, which is the same and only multi-GPU mode llama.cpp offers for it.
            MultiGpu = MultiGpuMode.LayerSplit,
            MultiGpuLimitation =
                "qwen4exp (Qwen3.8-Flash-Next) has no tensor-parallel path: none of its weights are " +
                "sharded, and its decode is one persisted single-device GGML graph per token whose " +
                "GDN/PLE recurrent state lives in device buffers owned by a single backend.",
        };
    }
}
