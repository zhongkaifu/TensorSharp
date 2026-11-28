// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using TensorSharp.Models.Architecture;

namespace TensorSharp.Models.MiniMaxH3
{
    /// <summary>MiniMax-H3 architecture plug-in (video/image generation).</summary>
    internal static class MiniMaxH3Architecture
    {
        public static ModelArchitectureDescriptor Descriptor { get; } = new()
        {
            Id = MiniMaxH3Model.ArchitectureId,
            DisplayName = "MiniMax-H3",
            Aliases = new[] { MiniMaxH3Model.ArchitectureId, "minimax_h3" },
            Factory = c => new MiniMaxH3Model(c.GgufPath, c.Backend, c.LayerSplitDegree),

            // MiniMax-H3's published GGUFs carry ZERO metadata - no architecture string at
            // all - so they have to be recognised by their tensors. Without this detector,
            // architecture resolution fails closed.
            DetectFromTensors = MiniMaxH3Model.LooksLikeMiniMaxH3,

            // A clip runs on the shared diffusion runtime (DiffusionVideoPipeline with MiniMaxH3Family; MiniMaxH3Denoiser is
            // its IDenoiser). --layer-split N runs the denoiser's blocks as whole-block stages over N GPUs on cuda and
            // ggml_cuda (DenoiserStages; MiniMaxH3DenoiserTests.WholeBlockStages*); the factory passes the degree on, and the
            // text encoder, vision tower and VAEs stay on the first GPU. Tensor parallelism and the hybrid are gaps: --tp N
            // is refused by the gate rather than dropped in silence, and a distributed group never reaches the model.
            Serving = ServingContract.SharedDiffusionRuntime(
                new[] { BackendType.Cpu, BackendType.Cuda, BackendType.Mlx, BackendType.GgmlCpu, BackendType.GgmlMetal, BackendType.GgmlCuda },
                tensorParallelGap: "minimax-h3 has no tensor-parallel path (its denoiser declares no shardable blocks); " +
                    "--layer-split N places its denoiser's blocks over N GPUs instead.",
                layerSplitBackends: new[] { BackendType.Cuda, BackendType.GgmlCuda }),

            // Step-distillation adapters (the LightX2V Turbo plug-ins) as minimax-h3-lora
            // configs, which declare the checkpoint partition and the sampling contract.
            LoraPluginType = LoraCliFlags.MiniMaxH3PluginType,
        };
    }
}
