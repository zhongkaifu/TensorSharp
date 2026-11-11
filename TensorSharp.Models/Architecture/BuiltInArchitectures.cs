// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
namespace TensorSharp.Models.Architecture
{
    /// <summary>
    /// The manifest of architectures shipped with TensorSharp.
    ///
    /// THIS LIST AND THE ARCHITECTURE'S OWN DIRECTORY ARE THE ONLY THINGS A NEW MODEL
    /// FAMILY HAS TO TOUCH. Each entry points at a <c>&lt;Name&gt;Architecture.cs</c>
    /// living beside its model, which declares the family's aliases, factory, multi-GPU
    /// mode and any native tuning. Nothing else in the loader, planner, CLI or server
    /// switches on an architecture name.
    ///
    /// The order matters only for metadata-free detection (see
    /// <see cref="ModelArchitectureDescriptor.DetectFromTensors"/>), which runs in this
    /// order for a GGUF that declares no architecture at all.
    /// </summary>
    internal static class BuiltInArchitectures
    {
        public static void RegisterAll()
        {
            // Metadata-free detectors first so unlabelled files can be recognized
            // from their tensor layout before resolution fails closed.
            ModelArchitectureRegistry.Register(MiniMaxH3.MiniMaxH3Architecture.Descriptor);
            ModelArchitectureRegistry.Register(MiniMaxH3.MiniMaxH3ImageArchitecture.Descriptor);
            ModelArchitectureRegistry.Register(Ltx25.Ltx25Architecture.Descriptor);

            // Text / multimodal language models.
            ModelArchitectureRegistry.Register(Qwen35Architecture.Descriptor);
            ModelArchitectureRegistry.Register(Qwen4ExpArchitecture.Descriptor);
            ModelArchitectureRegistry.Register(Gemma4Architecture.Descriptor);
            ModelArchitectureRegistry.Register(GptOssArchitecture.Descriptor);
            ModelArchitectureRegistry.Register(NemotronArchitecture.Descriptor);
            ModelArchitectureRegistry.Register(Mistral3Architecture.Descriptor);
            ModelArchitectureRegistry.Register(HunyuanDenseArchitecture.Descriptor);
            ModelArchitectureRegistry.Register(MuseGlimmerArchitecture.Descriptor);
            ModelArchitectureRegistry.Register(MuseSparkArchitecture.Descriptor);
            ModelArchitectureRegistry.Register(DeepSeek4Architecture.Descriptor);
            ModelArchitectureRegistry.Register(DeepSeek41Architecture.Descriptor);
            ModelArchitectureRegistry.Register(GlmDsaArchitecture.Descriptor);
            ModelArchitectureRegistry.Register(GlmDsaArchitecture.FlashDescriptor);
            ModelArchitectureRegistry.Register(MiniMaxM3Architecture.Descriptor);

            // Generative media.
            ModelArchitectureRegistry.Register(DiffusionGemmaArchitecture.Descriptor);
            ModelArchitectureRegistry.Register(QwenImage.QwenImageArchitecture.Descriptor);
            ModelArchitectureRegistry.Register(WanVideo.WanVideoArchitecture.Descriptor);
        }
    }
}
