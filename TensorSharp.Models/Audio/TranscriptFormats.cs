// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;

namespace TensorSharp.Models.Audio
{
    /// <summary>The response formats of OpenAI's /v1/audio/transcriptions: json, text, srt, vtt, verbose_json.</summary>
    public static class TranscriptFormats
    {
        public static readonly string[] Names = { "json", "text", "srt", "vtt", "verbose_json" };

        /// <summary>The format a file name asks for (.srt, .vtt, .json, anything else plain text).</summary>
        public static string FromExtension(string path) =>
            Path.GetExtension(path ?? "").ToLowerInvariant() switch
            {
                ".srt" => "srt",
                ".vtt" => "vtt",
                ".json" => "verbose_json",
                _ => "text",
            };

        public static string ContentType(string format) => format switch
        {
            "json" or "verbose_json" => "application/json",
            "vtt" => "text/vtt",
            _ => "text/plain",
        };

        public static string Render(TranscriptionResult r, string format, bool words = false) => format switch
        {
            "json" => JsonSerializer.Serialize(new { text = r.Text.Trim() }, Json),
            "text" => r.Text.Trim() + "\n",
            "srt" => ToSrt(r),
            "vtt" => ToVtt(r),
            "verbose_json" => ToVerboseJson(r, words),
            _ => throw new ArgumentException($"unknown response_format '{format}'; use one of {string.Join(", ", Names)}."),
        };

        public static string ToSrt(TranscriptionResult r)
        {
            var sb = new StringBuilder();
            int n = 1;
            foreach (var s in r.Segments)
            {
                sb.Append(n++).Append('\n');
                sb.Append(Stamp(s.Start, ',')).Append(" --> ").Append(Stamp(s.End, ',')).Append('\n');
                sb.Append(s.Text.Trim().Replace("-->", "->")).Append("\n\n");
            }
            return sb.ToString();
        }

        public static string ToVtt(TranscriptionResult r)
        {
            var sb = new StringBuilder("WEBVTT\n\n");
            foreach (var s in r.Segments)
            {
                sb.Append(Stamp(s.Start, '.')).Append(" --> ").Append(Stamp(s.End, '.')).Append('\n');
                sb.Append(s.Text.Trim().Replace("-->", "->")).Append("\n\n");
            }
            return sb.ToString();
        }

        public static string ToVerboseJson(TranscriptionResult r, bool words)
        {
            var payload = new
            {
                task = r.Task,
                language = r.Language,
                duration = Math.Round(r.Duration, 3),
                text = r.Text.Trim(),
                segments = r.Segments.Select(s => new
                {
                    id = s.Id,
                    seek = 0,
                    start = s.Start,
                    end = s.End,
                    text = s.Text,
                    tokens = s.Tokens,
                    temperature = s.Temperature,
                    avg_logprob = Math.Round(s.AvgLogprob, 6),
                    compression_ratio = Math.Round(s.CompressionRatio, 6),
                    no_speech_prob = Math.Round(s.NoSpeechProb, 6),
                }).ToArray(),
                words = words
                    ? r.Words.Select(w => new { word = w.Word.Trim(), start = w.Start, end = w.End, probability = w.Probability }).ToArray()
                    : null,
            };
            return JsonSerializer.Serialize(payload, Json);
        }

        /// <summary>Text as UTF-8 rather than escapes (transcripts are mostly not ASCII), and no null members.</summary>
        private static readonly JsonSerializerOptions Json = new()
        {
            DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull,
            Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
        };

        private static string Stamp(double seconds, char fraction)
        {
            if (seconds < 0) seconds = 0;
            long ms = (long)Math.Round(seconds * 1000);
            long h = ms / 3_600_000, m = ms / 60_000 % 60, s = ms / 1000 % 60, f = ms % 1000;
            return string.Create(CultureInfo.InvariantCulture, $"{h:00}:{m:00}:{s:00}{fraction}{f:000}");
        }
    }
}
