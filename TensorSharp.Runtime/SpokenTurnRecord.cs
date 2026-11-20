// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System.Collections.Generic;

namespace TensorSharp.Runtime
{
    /// <summary>
    /// A spoken assistant reply (LFM3-Omni) as generation decoded it, for a later turn that carries it back (OpenAI's
    /// <c>{"role":"assistant","audio":{"id":...}}</c>): the reply's tokens after the prompt's framing - the token that opened a
    /// forced mode, the text tokens and one frame placeholder per frame - and the codes of those frames, the end-of-speech
    /// frame included. Spliced like a tracked text turn (<see cref="ChatMessage.RawOutputTokens"/>) with the frames'
    /// rows put back by the multimodal expander (<see cref="ChatMessage.SpeechFrames"/>), the turn re-renders exactly as the
    /// sequence held it.
    /// </summary>
    public sealed record SpokenTurnRecord(
        IReadOnlyList<int> Tokens,
        IReadOnlyList<int[]> Frames,
        string? PromptTrailingWhitespace,
        string? GenerationSuffix)
    {
        /// <summary>Make <paramref name="message"/> (the assistant turn) splice this reply.</summary>
        public void ApplyTo(ChatMessage message)
        {
            message.RawOutputTokens = new List<int>(Tokens);
            message.SpeechFrames = Frames;
            message.RawPromptTrailingWhitespace = PromptTrailingWhitespace;
            message.RawGenerationSuffix = GenerationSuffix;
        }
    }
}
