// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using TensorSharp.Models.Audio;

namespace TensorSharp.Chat
{
    /// <summary>
    /// A spoken web chat reply: <c>/api/chat</c>'s <c>speech</c>, <c>true</c> or <c>{"voice": ..., "language": ...}</c>
    /// (both optional). The reply interleaves text and speech (the text is the speech's transcript) with thinking off; the
    /// voice defaults to the model's first one (<see cref="ISpeechOutputModel.SpeechVoices"/>), which the system line then
    /// names, as <c>/api/models</c>' <c>speechOutput.voices</c> lists them. The language defaults to the one the user's
    /// message is written in; one the model's speech modes do not speak is spoken from the text reply by its TTS
    /// (<see cref="SpeechReplyPath.Synthesized"/>).
    /// </summary>
    internal sealed class WebUiSpeechRequest
    {
        /// <summary>The voice asked for, or null for the model's default.</summary>
        public string Voice { get; init; }

        /// <summary>The speech's language code (e.g. <c>en</c>, <c>fr</c>), or null.</summary>
        public string Language { get; init; }

        /// <summary>
        /// Read <c>speech</c> from a <c>/api/chat</c> body: null (and no error) for a text reply (absent, <c>null</c> or
        /// <c>false</c>), else the spoken reply; <paramref name="error"/> says what is wrong with it. Members other than
        /// <c>voice</c> and <c>language</c> are refused rather than ignored.
        /// </summary>
        public static WebUiSpeechRequest Parse(JsonElement body, out string error)
        {
            error = null;
            if (body.ValueKind != JsonValueKind.Object || !body.TryGetProperty("speech", out JsonElement speech)
                || speech.ValueKind is JsonValueKind.Null or JsonValueKind.False)
                return null;
            if (speech.ValueKind == JsonValueKind.True)
                return new WebUiSpeechRequest();
            if (speech.ValueKind != JsonValueKind.Object)
            {
                error = "speech must be true or an object {\"voice\": ..., \"language\": ...}.";
                return null;
            }
            string voice = null, language = null;
            foreach (JsonProperty member in speech.EnumerateObject())
            {
                switch (member.Name)
                {
                    case "voice":
                        if (!TryText(member, out voice, out error)) return null;
                        break;
                    case "language":
                        if (!TryText(member, out language, out error)) return null;
                        break;
                    default:
                        error = $"speech.{member.Name} is not an option of a spoken reply: only voice and language.";
                        return null;
                }
            }
            return new WebUiSpeechRequest { Voice = voice, Language = language };
        }

        private static bool TryText(JsonProperty member, out string value, out string error)
        {
            value = null;
            error = null;
            if (member.Value.ValueKind == JsonValueKind.Null)
                return true;
            if (member.Value.ValueKind != JsonValueKind.String || string.IsNullOrWhiteSpace(member.Value.GetString()))
            {
                error = $"speech.{member.Name} must be a nonempty string (leave it out for the default).";
                return false;
            }
            value = member.Value.GetString().Trim();
            return true;
        }

        /// <summary>Why the loaded model cannot give this reply (null: it can): it does not speak, its speech decoder is
        /// not loaded, or the voice is not one of its.</summary>
        public string Refusal(ISpeechOutputModel speaker, string architecture)
        {
            string unavailable = UnavailableReason(speaker, architecture);
            if (unavailable != null) return unavailable;
            if (Voice != null && !speaker.IsSpeechVoice(Voice))
                return $"speech.voice '{Voice}' is not one of the loaded model's voices: {string.Join(", ", speaker.SpeechVoices)} "
                       + "(case does not matter).";
            return null;
        }

        /// <summary>Why the loaded model gives no spoken replies, or null when it does: what <c>/api/models</c> reports as
        /// <c>speechOutput.reason</c> and what a request with <c>speech</c> is refused with.</summary>
        public static string UnavailableReason(ISpeechOutputModel speaker, string architecture)
        {
            if (speaker == null)
                return $"The loaded model ({architecture ?? "none"}) has no speech output: send the message without speech.";
            if (!speaker.HasSpeechOutput)
                return speaker.SpeechOutputUnavailableReason
                       ?? $"The loaded model ({architecture ?? "none"}) has no speech decoder loaded.";
            return null;
        }

        /// <summary>The engine's options for this reply: interleaved text and speech, in the asked voice or the model's
        /// first one (so the system line always names a voice).</summary>
        public SpeechReplyOptions OptionsFor(ISpeechOutputModel speaker) => new()
        {
            Mode = SpeechReplyMode.Interleaved,
            Voice = Voice ?? speaker?.SpeechVoices?.FirstOrDefault(),
            Language = Language,
            Synthesis = false,
        };

        /// <summary>
        /// What a spoken turn does not run that this request would otherwise have had: <c>reasoning</c> (<c>think</c>),
        /// <c>tools</c> (the client's, or the host's code execution), <c>skills</c> (named, or the catalog the host
        /// discovers from) and <c>knowledge</c> (collections). A spoken reply is one driven generation, never a tool loop;
        /// the turn says so in a <c>notice</c> frame instead of dropping them silently.
        /// </summary>
        public static IReadOnlyList<string> Without(bool think, bool tools, bool skills, bool knowledge)
        {
            var without = new List<string>(4);
            if (think) without.Add("reasoning");
            if (tools) without.Add("tools");
            if (skills) without.Add("skills");
            if (knowledge) without.Add("knowledge");
            return without;
        }

        /// <summary>The notice's sentence for <paramref name="without"/> ("This reply is spoken, so it runs without
        /// reasoning, tools or skills.").</summary>
        public static string NoticeText(IReadOnlyList<string> without)
        {
            string[] names = without.Select(w => w == "knowledge" ? "knowledge lookups" : w).ToArray();
            string list = names.Length switch
            {
                0 => string.Empty,
                1 => names[0],
                _ => string.Join(", ", names.Take(names.Length - 1)) + " or " + names[^1],
            };
            return "This reply is spoken, so it runs without " + list + ".";
        }
    }
}
