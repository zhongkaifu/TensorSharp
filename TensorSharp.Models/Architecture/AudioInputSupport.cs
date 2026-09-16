// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;

namespace TensorSharp.Models.Architecture
{
    /// <summary>
    /// Which families refuse audio input, and with what words. One table so the
    /// OpenAI chat and Responses parsers, the Web UI, the shared chat pipeline,
    /// the CLI's <c>--audio</c> / <c>/audio</c> and the multimodal injector all
    /// give the same answer before an upload is written or a prompt rendered.
    /// Keyed on the GGUF's <c>general.architecture</c> as the model reports it
    /// and resolved through <see cref="ModelArchitectureRegistry"/>, so every
    /// alias of a family (Nemotron's <c>nemotron_h</c> / <c>nemotron_h_moe</c> /
    /// <c>nemotron_h_omni</c>) is covered without a second list.
    /// </summary>
    public static class AudioInputSupport
    {
        /// <summary>DeepSeek V4.1 Flash has no audio tower at all.</summary>
        public const string DeepSeek41Message =
            "DeepSeek V4.1 Flash does not support audio input. Remove audio attachments or use a model with an audio encoder.";

        /// <summary>
        /// The refusal <paramref name="architecture"/> gives audio input when no
        /// optional audio tower is known to be loaded, or null when the family can
        /// consume audio or is not one this table knows (an unknown family is left
        /// to its own multimodal contract). Callers that hold the loaded model use
        /// <see cref="UnsupportedReasonFor(ModelBase)"/> instead.
        /// </summary>
        public static string UnsupportedReasonFor(string architecture)
            => UnsupportedReasonFor(architecture, audioEncoderLoaded: false);

        /// <summary>
        /// The refusal <paramref name="architecture"/> gives audio input, given
        /// whether the loaded model actually carries a supported audio tower.
        /// DeepSeek V4.1 has no audio path whatever is loaded. Nemotron-H serves
        /// audio only when its companion GGUF contained the Parakeet/FastConformer
        /// tower and sound projector (<see cref="NemotronModel.IsAudioEncoderLoaded"/>):
        /// the public Omni mmproj carries only the vision tower, so the default is
        /// an explicit refusal, never a decoded clip that is silently dropped.
        /// </summary>
        public static string UnsupportedReasonFor(string architecture, bool audioEncoderLoaded)
        {
            if (string.IsNullOrEmpty(architecture))
                return null;
            if (string.Equals(architecture, "deepseek41", StringComparison.OrdinalIgnoreCase))
                return DeepSeek41Message;
            if (!audioEncoderLoaded &&
                ModelArchitectureRegistry.TryGet(architecture, out ModelArchitectureDescriptor descriptor) &&
                string.Equals(descriptor.Id, "nemotron_h", StringComparison.OrdinalIgnoreCase))
                return NemotronModel.AudioInputUnsupportedMessage;
            return null;
        }

        /// <summary>The refusal the loaded <paramref name="model"/> gives audio input, or null.</summary>
        public static string UnsupportedReasonFor(ModelBase model)
            => model == null ? null : UnsupportedReasonFor(model.Config?.Architecture, IsAudioEncoderLoaded(model));

        /// <summary>
        /// Whether <paramref name="model"/> has an optional audio tower loaded that
        /// this table's refusals depend on. Only Nemotron-H's tower is optional
        /// here; other families answer false, which the table ignores for them.
        /// </summary>
        public static bool IsAudioEncoderLoaded(ModelBase model)
            => model is NemotronModel { IsAudioEncoderLoaded: true };
    }
}
