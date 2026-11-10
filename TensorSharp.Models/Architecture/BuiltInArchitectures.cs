// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System.Collections.Generic;

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
        /// <summary>
        /// Every built-in architecture, in registration (detection) order. Tests judge these,
        /// not the live registry, which tests also extend with throwaway descriptors.
        /// </summary>
        public static IReadOnlyList<ModelArchitectureDescriptor> Descriptors { get; } = new[]
        {
            // Metadata-free detectors first so unlabelled files can be recognized
            // from their tensor layout before resolution fails closed.
            MiniMaxH3.MiniMaxH3Architecture.Descriptor,
            Ltx25.Ltx25Architecture.Descriptor,

            // Text / multimodal language models.
            Qwen35Architecture.Descriptor,
            Qwen4ExpArchitecture.Descriptor,
            Gemma4Architecture.Descriptor,
            GptOssArchitecture.Descriptor,
            NemotronArchitecture.Descriptor,
            Mistral3Architecture.Descriptor,
            HunyuanDenseArchitecture.Descriptor,
            MuseGlimmerArchitecture.Descriptor,
            MuseSparkArchitecture.Descriptor,
            DeepSeek4Architecture.Descriptor,
            DeepSeek41Architecture.Descriptor,
            GlmDsaArchitecture.Descriptor,
            GlmDsaArchitecture.FlashDescriptor,
            MiniMaxM3Architecture.Descriptor,

            // Generative media.
            DiffusionGemmaArchitecture.Descriptor,
            QwenImage.QwenImageArchitecture.Descriptor,
            WanVideo.WanVideoArchitecture.Descriptor,
        };

        public static void RegisterAll()
        {
            foreach (ModelArchitectureDescriptor descriptor in Descriptors)
                ModelArchitectureRegistry.Register(descriptor);
        }
    }
}
