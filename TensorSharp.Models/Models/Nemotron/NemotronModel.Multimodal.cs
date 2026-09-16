// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Nemotron-H's multimodal contract. Audio weights are an optional companion.
using System.Collections.Generic;

using TensorSharp.Models.Architecture;
using TensorSharp.Runtime;

namespace TensorSharp.Models
{
    public partial class NemotronModel : IVisionCapableModel, IAudioCapableModel, IAudioEncoderLoader, IMultimodalPromptExpander
    {
        /// <summary>
        /// The one refusal every entry point gives an audio attachment on this
        /// family when no audio tower is loaded (OpenAI chat/Responses parsers,
        /// the Web UI, the CLI and the multimodal injector). The public GGUF
        /// distribution of Nemotron 3 Nano Omni ships no audio encoder: its
        /// <c>mmproj</c> carries the RADIO vision tower only (<c>v.blk.*</c> +
        /// <c>mm.model.mlp.*</c>, <c>clip.has_vision_encoder</c> and nothing for
        /// audio), and the language GGUF has only the <c>&lt;so_embedding&gt;</c>
        /// placeholder token. Without the Parakeet/FastConformer tower and sound
        /// projector a clip cannot be encoded, and the model would see an unfilled
        /// placeholder and answer as if no audio had been sent, so the request is
        /// refused instead. <see cref="NemotronAudioEncoder"/> runs that tower when
        /// a companion GGUF carrying NVIDIA's <c>sound_encoder.*</c> /
        /// <c>sound_projection.*</c> tensors is loaded (<see cref="LoadAudioEncoder"/>);
        /// then <see cref="IsAudioEncoderLoaded"/> is true and audio is served.
        /// </summary>
        public const string AudioInputUnsupportedMessage =
            "This Nemotron-H model does not support audio input: no audio tower is loaded. The public GGUF " +
            "distribution of Nemotron 3 Nano Omni ships no audio encoder - its mmproj carries only the RADIO vision " +
            "tower, not the Parakeet/FastConformer audio tower or its sound projector - so an audio clip cannot be " +
            "encoded. Load an audio companion GGUF that contains the sound_encoder/sound_projection tensors " +
            "(--mmproj, or TS_NEMOTRON_AUDIO_MMPROJ beside a vision mmproj), remove the audio attachments, or use " +
            "a model with an audio encoder (Gemma 4).";

        /// <summary>True when a supported Parakeet audio tower and sound projector were loaded from a companion GGUF.</summary>
        public bool IsAudioEncoderLoaded => _audioEncoder != null;

        bool IVisionCapableModel.IsVisionEncoderLoaded => VisionEncoder != null;

        List<int> IMultimodalPromptExpander.ExpandMultimodalPrompt(
            ModelMultimodalInjector injector, List<ChatMessage> history, List<int> inputTokens)
            => injector.ProcessNemotronHistory(this, history, inputTokens);
    }
}
