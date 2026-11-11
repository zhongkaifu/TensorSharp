// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using TensorSharp.Models.Architecture;

namespace TensorSharp.Models.MiniMaxH3
{
    /// <summary>MiniMax-H3-Image architecture plug-in (text-to-image). Its denoiser is
    /// usually the safetensors release, which <see cref="ModelBase.Create"/> routes here
    /// before any GGUF probe; a GGUF with the same tensors is recognised by them.</summary>
    internal static class MiniMaxH3ImageArchitecture
    {
        public static ModelArchitectureDescriptor Descriptor { get; } = new()
        {
            Id = MiniMaxH3ImageModel.ArchitectureId,
            DisplayName = "MiniMax-H3-Image",
            Aliases = new[] { MiniMaxH3ImageModel.ArchitectureId, "minimax_h3_image" },
            Factory = c => new MiniMaxH3ImageModel(c.GgufPath, c.Backend),
            DetectFromTensors = MiniMaxH3ImageModel.LooksLikeMiniMaxH3Image,
            MultiGpu = MultiGpuMode.SingleDevice,
            MultiGpuLimitation =
                "minimax-h3-image has no tensor-parallel or layer-split path; it runs on one GPU and extra GPUs stay idle.",
        };
    }
}
