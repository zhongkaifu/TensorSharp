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

        /// <summary>The TTS template's language name of <paramref name="language"/> (a code; "en" when empty; a regional code
        /// the template does not list, such as zh-cn, by its language; an unknown value is used as written, as the template
        /// does).</summary>
        public static string TtsLanguageName(string language)
        {
            if (string.IsNullOrEmpty(language)) return TtsLanguages["en"];
            string code = language.Trim().ToLowerInvariant();
            if (TtsLanguages.TryGetValue(code, out string name)) return name;
            int dash = code.IndexOf('-');
            return dash > 0 && TtsLanguages.TryGetValue(code.Substring(0, dash), out name) ? name : language;
        }

        /// <summary>The speech languages written in a script of their own (the rest use Latin letters).</summary>
        private static readonly HashSet<string> OwnScript = new(StringComparer.Ordinal) { "zh", "ja", "ko", "ar", "he", "el" };

        /// <summary>
        /// The speech language <paramref name="text"/>'s script names, or null for Latin letters (English, French, ... look
        /// alike) and text without letters: the script with the most of the text, a run of Latin letters counting as one
        /// (a Chinese character is about a word), so "用Python写一个程序" is Chinese and "What does 你好 mean?" is not.
        /// Chinese characters with kana (at least one for every four) are Japanese.
        /// </summary>
        public static string LanguageOfText(string text)
        {
            if (string.IsNullOrEmpty(text)) return null;
            int han = 0, kana = 0, hangul = 0, arabic = 0, hebrew = 0, greek = 0, latin = 0;
            bool inLatin = false;
            foreach (System.Text.Rune r in text.EnumerateRunes())
            {
                int c = r.Value;
                bool latinLetter = c < 0x250 && System.Text.Rune.IsLetter(r);
                if (latinLetter)
                {
                    if (!inLatin) latin++;
                    inLatin = true;
                    continue;
                }
                inLatin = false;
                if ((c >= 0x4E00 && c <= 0x9FFF) || (c >= 0x3400 && c <= 0x4DBF) || (c >= 0xF900 && c <= 0xFAFF) || (c >= 0x20000 && c <= 0x2FA1F)) han++;
                else if ((c >= 0x3040 && c <= 0x30FF) || (c >= 0x31F0 && c <= 0x31FF) || (c >= 0xFF66 && c <= 0xFF9F)) kana++;
                else if ((c >= 0xAC00 && c <= 0xD7AF) || (c >= 0x1100 && c <= 0x11FF) || (c >= 0x3130 && c <= 0x318F)) hangul++;
                else if ((c >= 0x0600 && c <= 0x06FF) || (c >= 0x0750 && c <= 0x077F) || (c >= 0xFB50 && c <= 0xFDFF) || (c >= 0xFE70 && c <= 0xFEFF)) arabic++;
                else if (c >= 0x0590 && c <= 0x05FF) hebrew++;
                else if ((c >= 0x0370 && c <= 0x03FF) || (c >= 0x1F00 && c <= 0x1FFF)) greek++;
            }
            int cjk = han + kana;
            string best = null;
            int most = latin;
            void Take(string language, int count)
            {
                if (count > most) { most = count; best = language; }
            }
            Take(kana > 0 && kana * 4 >= han ? "ja" : "zh", cjk);
            Take("ko", hangul);
            Take("ar", arabic);
            Take("he", hebrew);
            Take("el", greek);
            return best;
        }

        /// <summary>
        /// The language to speak <paramref name="text"/> in (<see cref="TensorSharp.Models.Audio.ISpeechOutputModel.SpeechLanguageOf"/>):
        /// <paramref name="preferred"/> where the text's script suits it; the script's language where it names another
        /// (Chinese in a French reply), or English for Latin letters in a reply whose language has a script of its own;
        /// without <paramref name="preferred"/>, the script's language (null for Latin letters: the default, English). Kanji
        /// alone in a Japanese reply stays Japanese.
        /// </summary>
        public static string SpeechLanguageOf(string text, string preferred)
        {
            string script = LanguageOfText(text);
            if (string.IsNullOrWhiteSpace(preferred)) return script;
            string code = preferred.Trim().ToLowerInvariant();
            int dash = code.IndexOf('-');
            string baseCode = dash > 0 ? code.Substring(0, dash) : code;
            if (script == null)
            {
                // Latin letters (or none): English in a reply whose language has a script of its own, unless nothing is said.
                bool letters = false;
                foreach (System.Text.Rune r in (text ?? string.Empty).EnumerateRunes())
                    if (System.Text.Rune.IsLetter(r)) { letters = true; break; }
                return letters && OwnScript.Contains(baseCode) ? "en" : preferred;
            }
            if (script == baseCode || (script == "zh" && baseCode == "ja")) return preferred;
            return script;
        }

        /// <summary>
        /// Whether LFM3-Omni's own spoken-reply modes (interleaved and audio) speak <paramref name="language"/> (null: the
        /// default, English): English only. Their speech is English whatever the text: Chinese replies in interleaved mode
        /// are read with English sounds under every system prompt tried (llama.cpp and TensorSharp alike, Whisper's language
        /// ID "en"; LFM3-Audio's card: "voice chat currently supports English only"), while its TTS prompt speaks all of
        /// <see cref="TtsLanguages"/>.
        /// </summary>
        public static bool SpeaksRepliesIn(string language)
        {
            if (string.IsNullOrWhiteSpace(language)) return true;
            string code = language.Trim().ToLowerInvariant();
            return code == "en" || code.StartsWith("en-", StringComparison.Ordinal) || code == "english" || code == "american english"
                   || code == "british english";
        }

        /// <summary>
        /// What the TTS prompt is given to say for <paramref name="text"/> in <paramref name="language"/> (null: the language
        /// the text's script names): Chinese text with its numerals read out (<see cref="TensorSharp.Models.Audio.ChineseNumberReading"/>),
        /// which LFM3-Omni says as noise when it sees the digits; any other text as it is.
        /// </summary>
        public static string SpokenForm(string text, string language)
        {
            if (string.IsNullOrEmpty(text)) return text ?? string.Empty;
            string code = string.IsNullOrWhiteSpace(language) ? LanguageOfText(text) : language.Trim().ToLowerInvariant();
            return code == "zh" || (code != null && code.StartsWith("zh-", StringComparison.Ordinal))
                ? TensorSharp.Models.Audio.ChineseNumberReading.Read(text) : text;
        }

        /// <summary>
        /// The TTS prompt of <paramref name="text"/>: what the "tts" chat template renders (llama.cpp
        /// common_chat_get_tts_prompt), tokenized with special tokens parsed and nothing added. Without
        /// <paramref name="language"/> the language is the one the text's script names (<see cref="LanguageOfText"/>), else
        /// American English (the template's default). <paramref name="think"/>: the Omni template's empty
        /// &lt;think&gt;&lt;/think&gt; before &lt;|audio_start|&gt; (LFM3-Audio's has none).
        /// </summary>
        public static string Tts(string text, string language = null, string voice = null, string bosToken = "<|startoftext|>", bool think = true)
        {
            ArgumentNullException.ThrowIfNull(text);
            if (string.IsNullOrWhiteSpace(language)) language = LanguageOfText(text);
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
