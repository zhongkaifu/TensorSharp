// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Nemotron-H's multimodal contract. Its audio path is built during the main model load, so it is an audio SINK without an IAudioEncoderLoader.
using System.Collections.Generic;

using TensorSharp.Models.Architecture;
using TensorSharp.Runtime;

namespace TensorSharp.Models
{
    public partial class NemotronModel : IVisionCapableModel, IAudioCapableModel, IMultimodalPromptExpander
    {
        /// <summary>
        /// The one refusal every entry point gives an audio attachment on this
        /// family (OpenAI chat/Responses parsers, the Web UI, the CLI and the
        /// multimodal injector). The public GGUF distribution of Nemotron 3 Nano
        /// Omni ships no audio encoder: its <c>mmproj</c> carries the RADIO vision
        /// tower only (<c>v.blk.*</c> + <c>mm.model.mlp.*</c>, <c>clip.has_vision_encoder</c>
        /// and nothing for audio), and the language GGUF has only the
        /// <c>&lt;so_embedding&gt;</c> placeholder token. The Parakeet/FastConformer
        /// tower that turns a log-mel spectrogram into those embeddings exists
        /// only as weights TensorSharp does not have, so an audio clip cannot be
        /// encoded; the model would see an unfilled placeholder and answer as if
        /// no audio had been sent. Stock llama.cpp refuses the same request.
        /// </summary>
        public const string AudioInputUnsupportedMessage =
            "Nemotron-H (including Nemotron 3 Nano Omni) does not support audio input: the public GGUF " +
            "distribution ships no audio encoder - the Omni mmproj carries only the RADIO vision tower, not the " +
            "Parakeet/FastConformer audio tower or its projector - so an audio clip cannot be encoded. Remove " +
            "audio attachments or use a model with an audio encoder (Gemma 4).";

        bool IVisionCapableModel.IsVisionEncoderLoaded => VisionEncoder != null;

        List<int> IMultimodalPromptExpander.ExpandMultimodalPrompt(
            ModelMultimodalInjector injector, List<ChatMessage> history, List<int> inputTokens)
            => injector.ProcessNemotronHistory(this, history, inputTokens);
    }
}
