// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;

namespace TensorSharp.Models.Lfm3
{
    /// <summary>
    /// LFM3-Omni's speech prompts. The model has no speaker embedding: the voice and language are named in the system prompt.
    /// <list type="bullet">
    /// <item><see cref="Tts"/> renders the LM GGUF's <c>tokenizer.chat_template.tts</c> (written by llama.cpp's converter;
    /// the HF model card's TTS recipe): "Perform TTS in &lt;language&gt; using a &lt;voice&gt; voice." then the text, then
    /// the assistant turn opened with &lt;think&gt;&lt;/think&gt;&lt;|audio_start|&gt;.</item>
    /// <item><see cref="SpeechSystemLine"/> is llama.cpp's realtime system line for spoken replies in a conversation
    /// (common_chat_get_speech_system_prompt: "Speak in &lt;language&gt; using a &lt;voice&gt; voice."; its language names
    /// differ from the TTS template's: "en" is "English").</item>
    /// </list>
    /// </summary>
    public static class Lfm3SpeechPrompt
    {
        /// <summary>The voices (the default first); matched without regard to case.</summary>
        public static IReadOnlyList<string> Voices { get; } = new[] { "female", "male" };

        /// <summary>The TTS template's language names by code (lower case).</summary>
        public static IReadOnlyDictionary<string, string> TtsLanguages { get; } = new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["ar"] = "Arabic", ["da"] = "Danish", ["de"] = "German", ["el"] = "Greek", ["en"] = "American English",
            ["en-us"] = "American English", ["en-gb"] = "British English", ["fi"] = "Finnish", ["fr"] = "French", ["he"] = "Hebrew",
            ["it"] = "Italian", ["ja"] = "Japanese", ["ko"] = "Korean", ["nl"] = "Dutch", ["pl"] = "Polish", ["pt"] = "Portuguese",
            ["es"] = "Spanish", ["sv"] = "Swedish", ["tr"] = "Turkish", ["zh"] = "Chinese",
        };

        /// <summary>llama.cpp's realtime language names by code (lower case).</summary>
        public static IReadOnlyDictionary<string, string> SpeechLanguages { get; } = new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["ar"] = "Arabic", ["da"] = "Danish", ["de"] = "German", ["el"] = "Greek", ["en"] = "English", ["es"] = "Spanish",
            ["fi"] = "Finnish", ["fr"] = "French", ["he"] = "Hebrew", ["it"] = "Italian", ["ja"] = "Japanese", ["ko"] = "Korean",
            ["nl"] = "Dutch", ["pl"] = "Polish", ["pt"] = "Portuguese", ["sv"] = "Swedish", ["tr"] = "Turkish", ["zh"] = "Chinese",
        };

        /// <summary>The voice the template uses for <paramref name="voice"/>: "female" or "male" (case-insensitive), the default
        /// "female" otherwise.</summary>
        public static string NormalizeVoice(string voice)
        {
            string v = voice?.ToLowerInvariant();
            return v == "female" || v == "male" ? v : Voices[0];
        }

        /// <summary>Whether <paramref name="voice"/> names one of <see cref="Voices"/>.</summary>
        public static bool IsVoice(string voice) => voice != null && (string.Equals(voice, "female", StringComparison.OrdinalIgnoreCase)
            || string.Equals(voice, "male", StringComparison.OrdinalIgnoreCase));

        /// <summary>The TTS template's language name of <paramref name="language"/> (a code; "en" when empty; an unknown value
        /// is used as written, as the template does).</summary>
        public static string TtsLanguageName(string language)
        {
            if (string.IsNullOrEmpty(language)) return TtsLanguages["en"];
            return TtsLanguages.TryGetValue(language.ToLowerInvariant(), out string name) ? name : language;
        }

        /// <summary>
        /// The TTS prompt of <paramref name="text"/>: what the "tts" chat template renders (llama.cpp
        /// common_chat_get_tts_prompt), tokenized with special tokens parsed and nothing added. <paramref name="think"/>: the
        /// Omni template's empty &lt;think&gt;&lt;/think&gt; before &lt;|audio_start|&gt; (LFM3-Audio's has none).
        /// </summary>
        public static string Tts(string text, string language = null, string voice = null, string bosToken = "<|startoftext|>", bool think = true)
        {
            ArgumentNullException.ThrowIfNull(text);
            return bosToken + "<|im_start|>system\nPerform TTS in " + TtsLanguageName(language) + " using a " + NormalizeVoice(voice) +
                   " voice.<|im_end|>\n<|im_start|>user\n" + text + "<|im_end|>\n<|im_start|>assistant\n" + (think ? "<think></think>" : "") +
                   "<|audio_start|>";
        }

        /// <summary>llama.cpp's realtime system line for spoken replies ("Speak in English using a female voice."), or "" when
        /// neither a language nor a known voice is given.</summary>
        public static string SpeechSystemLine(string language, string voice)
        {
            string v = voice?.ToLowerInvariant();
            bool hasVoice = v == "female" || v == "male";
            if (string.IsNullOrEmpty(language) && !hasVoice) return "";
            string line = "Speak";
            if (!string.IsNullOrEmpty(language))
                line += " in " + (SpeechLanguages.TryGetValue(language.ToLowerInvariant(), out string name) ? name : language);
            if (hasVoice) line += " using a " + v + " voice";
            return line + ".";
        }
    }
}
