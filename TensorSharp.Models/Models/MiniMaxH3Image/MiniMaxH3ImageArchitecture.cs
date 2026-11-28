// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using TensorSharp.Models.Architecture;

namespace TensorSharp.Models.MiniMaxH3Image
{
    /// <summary>MiniMax-H3-Image architecture plug-in (text to image), built on the shared diffusion runtime. Its denoiser is
    /// usually the safetensors release, which the registry offers to <see cref="ModelArchitectureDescriptor.DetectSafetensors"/>
    /// before any GGUF probe; a GGUF with the same tensors is recognised by them.</summary>
    internal static class MiniMaxH3ImageArchitecture
    {
        private const string HyVidLabel = "hyvid";

        public static ModelArchitectureDescriptor Descriptor { get; } = new()
        {
            Id = MiniMaxH3ImageFamily.ArchitectureId,
            DisplayName = "MiniMax-H3-Image",
            Aliases = new[] { MiniMaxH3ImageFamily.ArchitectureId, "minimax_h3_image" },
            Factory = c => new MiniMaxH3ImageDiffusionModel(c.GgufPath, c.Backend, c.LayerSplitDegree),
            // Style adapters (any number, bare weights too) and one speed adapter (config/lora/minimax-h3-image-*.json).
            LoraPluginType = Runtime.LoraCliFlags.MiniMaxH3ImagePluginType,
            DetectFromTensors = LooksLikeDenoiser,
            // The Comfy-Org release: the denoiser in its original tensor names, read from the header alone.
            DetectSafetensors = file => file != null && MiniMaxH3ImageFamily.IsDenoiser(file.Tensors.Keys),
            // ComfyUI-GGUF's converter labels the quantized denoisers it writes 'hyvid'
            // (unsloth/MiniMax-H3-Image-GGUF); the label never names this model, so the
            // tensors decide, as for an unlabelled file.
            RecognizeRelabelledFile = (label, gguf) =>
                string.Equals(label, HyVidLabel, System.StringComparison.OrdinalIgnoreCase) && LooksLikeDenoiser(gguf),
            RelabelledFileDescription =
                "a '" + HyVidLabel + "'-labelled GGUF from ComfyUI-GGUF's converter whose tensors are the " +
                "MiniMax-H3-Image denoiser (the H3 DiT with no audio stream) - e.g. unsloth/MiniMax-H3-Image-GGUF",
            Serving = ServingContract.SharedDiffusionRuntime(
                new[] { BackendType.GgmlCpu, BackendType.GgmlMetal, BackendType.GgmlCuda },
                tensorParallelGap: "minimax-h3-image has no tensor-parallel path; it runs on one GPU and extra GPUs stay idle.",
                layerSplitBackends: new[] { BackendType.GgmlCuda }),
        };

        /// <summary>The architecture detector for a GGUF that holds the image denoiser.</summary>
        internal static bool LooksLikeDenoiser(Runtime.GgufFile gguf) =>
            gguf != null && MiniMaxH3ImageFamily.IsDenoiser(gguf.Tensors.Keys);
    }
}
