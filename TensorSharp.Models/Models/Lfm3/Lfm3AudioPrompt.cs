// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;

namespace TensorSharp.Models
{
    /// <summary>
    /// Where LFM3-Omni's audio rows go in a prompt, and the prompts its audio tasks use (llama.cpp mtmd + common/chat.cpp,
    /// HF processing_lfm3_omni.py):
    /// <list type="bullet">
    /// <item>An audio clip is <c>&lt;|vision_start|&gt;</c> + one row per 80 ms (<see cref="Lfm3AudioPreprocessor.RowCountForSamples"/>)
    /// + <c>&lt;|vision_end|&gt;</c>: Omni wraps audio in the image span markers (LFM3-Audio, without a vision tower, places
    /// the rows bare). Each row is one ordinary 1-D position, attended causally, injected unscaled.</item>
    /// <item>ASR: system <c>Perform ASR in &lt;Language&gt;.</c> (<see cref="AsrSystemPrompt"/>), the user turn the audio alone,
    /// the assistant turn opened with <c>&lt;think&gt;&lt;/think&gt;</c> (thinking off).</item>
    /// <item>A video's soundtrack: one clip of the whole track, its rows split after each group of two frames
    /// (<see cref="SplitSoundtrack"/>), each piece placed bare inside the video's span.</item>
    /// </list>
    /// </summary>
    public static class Lfm3AudioPrompt
    {
        /// <summary>The token before an Omni audio clip's rows (id 125009 in LFM3-Omni-2B).</summary>
        public const string SpanStart = "<|vision_start|>";

        /// <summary>The token after them (id 125010).</summary>
        public const string SpanEnd = "<|vision_end|>";

        /// <summary>The chat template's own placeholder of an audio part (<c>&lt;|vision_start|&gt;&lt;|audio|&gt;&lt;|vision_end|&gt;</c>):
        /// an expander replaces this inner token with the rows.</summary>
        public const string Placeholder = "<|audio|>";

        /// <summary>The opening of an assistant turn without thinking (the template's <c>enable_thinking = false</c>).</summary>
        public const string AssistantOpening = "<think></think>";

        /// <summary>The token that ends a text segment in LFM3's interleaved and live-ASR outputs (id 125014): a stop token
        /// next to <c>&lt;|im_end|&gt;</c> (HF generation_config eos_token_id [124900, 125014]).</summary>
        public const string TextEnd = "<|text_end|>";

        /// <summary>The LFM3 ASR languages by ISO 639-1 code (llama.cpp <c>lfm3_language_name</c>).</summary>
        public static IReadOnlyDictionary<string, string> Languages { get; } = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
        {
            ["ar"] = "Arabic", ["da"] = "Danish", ["de"] = "German", ["el"] = "Greek", ["en"] = "English", ["es"] = "Spanish",
            ["fi"] = "Finnish", ["fr"] = "French", ["he"] = "Hebrew", ["it"] = "Italian", ["ja"] = "Japanese", ["ko"] = "Korean",
            ["nl"] = "Dutch", ["pl"] = "Polish", ["pt"] = "Portuguese", ["sv"] = "Swedish", ["tr"] = "Turkish", ["zh"] = "Chinese",
        };

        /// <summary>
        /// The ASR system prompt: <c>Perform ASR.</c> without a language (the model detects it), else <c>Perform ASR in
        /// &lt;Name&gt;.</c> with the English name of a known code; any other value is used as given (a name already, as
        /// llama.cpp passes unknown codes through).
        /// </summary>
        public static string AsrSystemPrompt(string language = null)
        {
            if (string.IsNullOrWhiteSpace(language)) return "Perform ASR.";
            string code = language.Trim();
            return $"Perform ASR in {(Languages.TryGetValue(code, out string name) ? name : code)}.";
        }

        /// <summary>Prompt positions an Omni audio clip of <paramref name="samples"/> 16 kHz samples takes: its rows and the
        /// two span markers.</summary>
        public static int PositionsForSamples(long samples) => Lfm3AudioPreprocessor.RowCountForSamples(samples) + 2;

        /// <summary>One soundtrack piece: the group of frames it follows, its samples [SampleStart, SampleEnd) of the trimmed
        /// track, and its rows [RowStart, RowEnd) of the track's encoding.</summary>
        public readonly record struct SoundtrackPiece(int Group, long SampleStart, long SampleEnd, int RowStart, int RowEnd)
        {
            public int Rows => RowEnd - RowStart;
        }

        /// <summary>
        /// A video's soundtrack split as llama.cpp's mtmd helper splits it (mtmd-helper.cpp <c>set_soundtrack</c> /
        /// <c>soundtrack_slice</c>, mtmd.cpp <c>add_soundtrack_piece</c>): the track is cut to the video stream's span
        /// (round(<paramref name="sourceFrames"/> / <paramref name="sourceFps"/> * rate) samples); the sampled frames
        /// (<paramref name="frameIndices"/>, source frame numbers) form groups of <paramref name="framesPerGroup"/>, group g's
        /// audio runs from round(t_g * rate) (0 for the first group) to the next group's start (the track's end for the
        /// last), t_g the time of the group's first frame; a group whose audio is empty gets no piece. The track is encoded
        /// once (<see cref="Lfm3AudioPreprocessor.RowCountForSamples"/> of the trimmed length); a piece takes rows up to
        /// round(samples so far / 1280), the last piece the rest. Rounding is to nearest, ties to even (C's nearbyint).
        /// </summary>
        /// <param name="trimmedSamples">The trimmed track's length (the caller cuts the decoded track to it).</param>
        public static IReadOnlyList<SoundtrackPiece> SplitSoundtrack(IReadOnlyList<int> frameIndices, double sourceFps, long sourceFrames,
            long soundtrackSamples, out long trimmedSamples, int framesPerGroup = 2, int sampleRate = Lfm3AudioPreprocessor.SampleRate)
        {
            ArgumentNullException.ThrowIfNull(frameIndices);
            if (sourceFps <= 0) throw new ArgumentOutOfRangeException(nameof(sourceFps));
            if (framesPerGroup < 1) throw new ArgumentOutOfRangeException(nameof(framesPerGroup));
            var pieces = new List<SoundtrackPiece>();
            long span = (long)Math.Round(sourceFrames / sourceFps * sampleRate, MidpointRounding.ToEven);
            trimmedSamples = Math.Max(0, Math.Min(soundtrackSamples, span));
            if (trimmedSamples == 0 || frameIndices.Count == 0) return pieces;
            int groups = (frameIndices.Count + framesPerGroup - 1) / framesPerGroup;
            long total = trimmedSamples;
            long GroupStart(int g) => Math.Min(total, (long)Math.Round(frameIndices[g * framesPerGroup] / sourceFps * sampleRate, MidpointRounding.ToEven));
            int rows = Lfm3AudioPreprocessor.RowCountForSamples(trimmedSamples);
            var kept = new List<(int Group, long Begin, long End)>();
            for (int g = 0; g < groups; g++)
            {
                long begin = g == 0 ? 0 : GroupStart(g);
                long end = g == groups - 1 ? total : GroupStart(g + 1);
                if (begin < end) kept.Add((g, begin, end));
            }
            long seen = 0;
            int assigned = 0;
            for (int i = 0; i < kept.Count; i++)
            {
                seen += kept[i].End - kept[i].Begin;
                int end = i == kept.Count - 1
                    ? rows
                    : (int)Math.Clamp((long)Math.Round(seen / (double)Lfm3AudioPreprocessor.SamplesPerRow, MidpointRounding.ToEven), assigned, rows);
                pieces.Add(new SoundtrackPiece(kept[i].Group, kept[i].Begin, kept[i].End, assigned, end));
                assigned = end;
            }
            return pieces;
        }
    }
}
