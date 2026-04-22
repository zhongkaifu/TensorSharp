// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Collections.Generic;
using System.Text.Json;
using TensorSharp.Runtime;

namespace TensorSharp.Server.RequestParsers
{
    /// <summary>
    /// Pure parsers that translate the various flavours of sampling parameters
    /// (Web UI snake/camel, Ollama under <c>options</c>, OpenAI flat) into a
    /// single <see cref="SamplingConfig"/>. Kept as static methods because each
    /// call is cheap, stateless, and easy to unit test.
    ///
    /// Every overload accepts an optional <c>defaults</c> argument. When supplied,
    /// the parser starts from a clone of those defaults and only overwrites
    /// fields that are explicitly present in the request body. This is what lets
    /// the server expose a global <c>--temperature</c>/<c>--top-k</c>/&hellip;
    /// CLI surface without forcing every client to repeat the same parameters.
    /// Passing <c>null</c> (or omitting the argument) keeps the historical
    /// behaviour where missing fields fall back to the SamplingConfig defaults
    /// baked into the type (Ollama-compatible: temp=0.8, topK=40, topP=0.9).
    /// </summary>
    internal static class SamplingConfigParser
    {
        /// <summary>
        /// Parse the Web UI body which accepts both snake_case (kept for API
        /// parity with Ollama/OpenAI) and camelCase (preferred by the JS UI).
        /// </summary>
        public static SamplingConfig ParseWebUi(JsonElement body, SamplingConfig defaults = null)
        {
            var cfg = SeedFromDefaults(defaults);
            if (body.TryGetProperty("temperature", out var temp))
                cfg.Temperature = temp.GetSingle();
            if (body.TryGetProperty("top_k", out var tk))
                cfg.TopK = tk.GetInt32();
            if (body.TryGetProperty("topK", out var tk2))
                cfg.TopK = tk2.GetInt32();
            if (body.TryGetProperty("top_p", out var tp))
                cfg.TopP = tp.GetSingle();
            if (body.TryGetProperty("topP", out var tp2))
                cfg.TopP = tp2.GetSingle();
            if (body.TryGetProperty("min_p", out var mp))
                cfg.MinP = mp.GetSingle();
            if (body.TryGetProperty("minP", out var mp2))
                cfg.MinP = mp2.GetSingle();
            if (body.TryGetProperty("repetition_penalty", out var rp))
                cfg.RepetitionPenalty = rp.GetSingle();
            if (body.TryGetProperty("repetitionPenalty", out var rp2))
                cfg.RepetitionPenalty = rp2.GetSingle();
            if (body.TryGetProperty("presence_penalty", out var pp))
                cfg.PresencePenalty = pp.GetSingle();
            if (body.TryGetProperty("presencePenalty", out var pp2))
                cfg.PresencePenalty = pp2.GetSingle();
            if (body.TryGetProperty("frequency_penalty", out var fp))
                cfg.FrequencyPenalty = fp.GetSingle();
            if (body.TryGetProperty("frequencyPenalty", out var fp2))
                cfg.FrequencyPenalty = fp2.GetSingle();
            if (body.TryGetProperty("seed", out var sd))
                cfg.Seed = sd.GetInt32();
            if (body.TryGetProperty("stop", out var stopEl) && stopEl.ValueKind == JsonValueKind.Array)
            {
                // The "stop" key being present in the request is treated as a
                // full replacement of any defaults so callers can intentionally
                // disable a global default by sending an empty array.
                cfg.StopSequences = new List<string>();
                foreach (var s in stopEl.EnumerateArray())
                    if (s.GetString() is string sv)
                        cfg.StopSequences.Add(sv);
            }
            return cfg;
        }

        /// <summary>
        /// Parse Ollama's body where every sampler value lives inside a nested
        /// <c>options</c> object (e.g. <c>{ "options": { "temperature": 0.7 } }</c>).
        /// </summary>
        public static SamplingConfig ParseOllama(JsonElement body, SamplingConfig defaults = null)
        {
            var cfg = SeedFromDefaults(defaults);
            if (body.TryGetProperty("options", out var opts))
            {
                if (opts.TryGetProperty("temperature", out var temp)) cfg.Temperature = temp.GetSingle();
                if (opts.TryGetProperty("top_k", out var tk)) cfg.TopK = tk.GetInt32();
                if (opts.TryGetProperty("top_p", out var tp)) cfg.TopP = tp.GetSingle();
                if (opts.TryGetProperty("min_p", out var mp)) cfg.MinP = mp.GetSingle();
                if (opts.TryGetProperty("repeat_penalty", out var rp)) cfg.RepetitionPenalty = rp.GetSingle();
                if (opts.TryGetProperty("presence_penalty", out var pp)) cfg.PresencePenalty = pp.GetSingle();
                if (opts.TryGetProperty("frequency_penalty", out var fp)) cfg.FrequencyPenalty = fp.GetSingle();
                if (opts.TryGetProperty("seed", out var sd)) cfg.Seed = sd.GetInt32();
                if (opts.TryGetProperty("stop", out var stopEl) && stopEl.ValueKind == JsonValueKind.Array)
                {
                    cfg.StopSequences = new List<string>();
                    foreach (var s in stopEl.EnumerateArray())
                        if (s.GetString() is string sv) cfg.StopSequences.Add(sv);
                }
            }
            return cfg;
        }

        /// <summary>
        /// Parse OpenAI's flat body. The OpenAI spec also allows <c>stop</c> to
        /// be a single string instead of an array.
        /// </summary>
        public static SamplingConfig ParseOpenAI(JsonElement body, SamplingConfig defaults = null)
        {
            var cfg = SeedFromDefaults(defaults);
            if (body.TryGetProperty("temperature", out var temp)) cfg.Temperature = temp.GetSingle();
            if (body.TryGetProperty("top_p", out var tp)) cfg.TopP = tp.GetSingle();
            if (body.TryGetProperty("presence_penalty", out var pp)) cfg.PresencePenalty = pp.GetSingle();
            if (body.TryGetProperty("frequency_penalty", out var fp)) cfg.FrequencyPenalty = fp.GetSingle();
            if (body.TryGetProperty("seed", out var sd)) cfg.Seed = sd.GetInt32();
            if (body.TryGetProperty("stop", out var stopEl))
            {
                cfg.StopSequences = new List<string>();
                if (stopEl.ValueKind == JsonValueKind.Array)
                {
                    foreach (var s in stopEl.EnumerateArray())
                        if (s.GetString() is string stopVal) cfg.StopSequences.Add(stopVal);
                }
                else if (stopEl.ValueKind == JsonValueKind.String && stopEl.GetString() is string singleStop)
                {
                    cfg.StopSequences.Add(singleStop);
                }
            }
            return cfg;
        }

        /// <summary>
        /// Returns a writable starting point for a parse: a clone of
        /// <paramref name="defaults"/> when supplied, or a fresh
        /// <see cref="SamplingConfig"/> with the type's built-in defaults
        /// otherwise. We always clone (rather than mutate the supplied
        /// instance) so per-request overrides cannot bleed into the shared
        /// server-wide defaults singleton.
        /// </summary>
        private static SamplingConfig SeedFromDefaults(SamplingConfig defaults)
        {
            return defaults != null ? defaults.Clone() : new SamplingConfig();
        }
    }
}
