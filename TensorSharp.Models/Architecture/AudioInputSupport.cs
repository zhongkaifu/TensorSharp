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
        /// DiffusionGemma inherits Gemma 4's vocabulary, <c>&lt;|audio&gt;</c> ids
        /// included, but the published checkpoint has no <c>audio_config</c> and ships
        /// no audio weights: there is nothing behind those ids and no projector file
        /// could supply one. This is therefore NOT a "load the projector" message —
        /// the refusal is permanent for this checkpoint, unlike its image input, which
        /// a gemma4v tower (mmproj GGUF or HF safetensors shard) does enable.
        /// </summary>
        public const string DiffusionGemmaMessage =
            "DiffusionGemma does not support audio input: this checkpoint has no audio encoder " +
            "(no audio tower is published upstream, so no projector can add one). " +
            "Remove the audio attachment. Images are supported once a vision projector is loaded.";

        /// <summary>
        /// MiniMax-M3.1's vocabulary carries speech tokens, but the release ships a
        /// vision projector only: there is no audio tower to put behind them.
        /// </summary>
        public const string MiniMaxM3Message =
            "MiniMax-M3.1 does not support audio input: the release has no audio encoder (its projector is a " +
            "vision tower only). Remove the audio attachment; images are supported once the vision projector is loaded.";

        /// <summary>
        /// LiquidAI LFM3: one architecture for LFM3-1.7B (text only) and LFM3-Omni-2B, whose
        /// audio tower is in the Omni mmproj beside the vision tower. Without that tower loaded
        /// an audio clip has nothing to encode it.
        /// </summary>
        public const string Lfm3Message =
            "This LFM3 model has no audio encoder loaded: audio input needs LFM3-Omni-2B with its projector " +
            "(--mmproj mmproj-LFM3-Omni-2B-Q8_0.gguf, which carries the audio tower beside the vision one). " +
            "LFM3-1.7B is text only. Remove the audio attachment or load LFM3-Omni with its mmproj.";

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
        /// DeepSeek V4.1 has no audio path whatever is loaded. LFM3 serves audio
        /// only with LFM3-Omni's audio tower loaded from its mmproj. Nemotron-H serves
        /// audio only when its companion GGUF contained the Parakeet/FastConformer
        /// tower and sound projector (<see cref="NemotronRuntimeModel.IsAudioEncoderLoaded"/>):
        /// the public Omni mmproj carries only the vision tower, so the default is
        /// an explicit refusal, never a decoded clip that is silently dropped.
        /// </summary>
        public static string UnsupportedReasonFor(string architecture, bool audioEncoderLoaded)
        {
            if (string.IsNullOrEmpty(architecture))
                return null;
            if (string.Equals(architecture, "deepseek41", StringComparison.OrdinalIgnoreCase))
                return DeepSeek41Message;
            if (ModelArchitectureRegistry.TryGet(architecture, out ModelArchitectureDescriptor diffusion)
                && string.Equals(diffusion.Id, "diffusion-gemma", StringComparison.OrdinalIgnoreCase))
                return DiffusionGemmaMessage;
            if (ModelArchitectureRegistry.TryGet(architecture, out ModelArchitectureDescriptor minimax)
                && string.Equals(minimax.Id, MiniMaxM3Model.ArchitectureId, StringComparison.OrdinalIgnoreCase))
                return MiniMaxM3Message;
            if (!audioEncoderLoaded &&
                ModelArchitectureRegistry.TryGet(architecture, out ModelArchitectureDescriptor descriptor) &&
                string.Equals(descriptor.Id, "nemotron_h", StringComparison.OrdinalIgnoreCase))
                return NemotronRuntimeModel.AudioInputUnsupportedMessage;
            if (!audioEncoderLoaded &&
                ModelArchitectureRegistry.TryGet(architecture, out ModelArchitectureDescriptor lfm3) &&
                string.Equals(lfm3.Id, "lfm3", StringComparison.OrdinalIgnoreCase))
                return Lfm3Message;
            return null;
        }

        /// <summary>The refusal the loaded <paramref name="model"/> gives audio input, or null.</summary>
        public static string UnsupportedReasonFor(ModelBase model)
            => model == null ? null : UnsupportedReasonFor(model.Config?.Architecture, IsAudioEncoderLoaded(model));

        /// <summary>
        /// Whether <paramref name="model"/> has an optional audio tower loaded that
        /// this table's refusals depend on: Nemotron-H's and LFM3-Omni's (from the
        /// mmproj); other families answer false, which the table ignores for them.
        /// </summary>
        public static bool IsAudioEncoderLoaded(ModelBase model)
            => model is NemotronRuntimeModel { AudioEncoder: not null } or Lfm3RuntimeModel { IsAudioEncoderLoaded: true };
    }
}
