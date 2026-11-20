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
using TensorSharp.Server.Hosting;

namespace TensorSharp.Server.RequestParsers
{
    /// <summary>
    /// Pure parsers that translate the various flavours of sampling parameters
    /// (Web UI snake/camel, Ollama under <c>options</c>, OpenAI flat) into a
    /// single <see cref="SamplingConfig"/>. Kept as static methods because each
    /// call is cheap, stateless, and easy to unit test.
    ///
    /// Every method accepts an optional <c>defaults</c> argument. When supplied,
    /// the parser starts from a clone of those defaults and only overwrites
    /// fields that are explicitly present in the request body. This is what lets
    /// the server expose a global <c>--temperature</c>/<c>--top-k</c>/&hellip;
    /// CLI surface without forcing every client to repeat the same parameters.
    /// Passing <c>null</c> (or omitting the argument) keeps the historical
    /// behaviour where missing fields fall back to the SamplingConfig defaults
    /// baked into the type (Ollama-compatible: temp=0.8, topK=40, topP=0.9).
    ///
    /// The merged result is then handed to
    /// <see cref="SamplingDefaults.ApplyPinned"/>, which re-asserts the
    /// parameters the operator pinned on the command line / config file when the
    /// server runs with <see cref="SamplingPrecedence.Config"/>. Clients that
    /// hardcode sampling values (VS Code Copilot Chat sends
    /// <c>temperature</c>/<c>top_p</c> on every request) therefore stop silently
    /// overriding the server's configuration. A plain <see cref="SamplingConfig"/>
    /// converts implicitly to defaults with nothing pinned, i.e. request-wins.
    /// </summary>
    internal static class SamplingConfigParser
    {
        /// <summary>
        /// Parse the Web UI body which accepts both snake_case (kept for API
        /// parity with Ollama/OpenAI) and camelCase (preferred by the JS UI).
        /// </summary>
        public static SamplingConfig ParseWebUi(JsonElement body, SamplingDefaults defaults = null)
        {
            var cfg = SeedFromDefaults(defaults?.Values);
            if (TryGetSingle(body, "temperature", out float temp))
                cfg.Temperature = temp;
            if (TryGetInt32(body, "top_k", out int tk) || TryGetInt32(body, "topK", out tk))
                cfg.TopK = tk;
            if (TryGetSingle(body, "top_p", out float tp) || TryGetSingle(body, "topP", out tp))
                cfg.TopP = tp;
            if (TryGetSingle(body, "min_p", out float mp) || TryGetSingle(body, "minP", out mp))
                cfg.MinP = mp;
            if (TryGetSingle(body, "repetition_penalty", out float rp)
                || TryGetSingle(body, "repetitionPenalty", out rp)
                || TryGetSingle(body, "repeat_penalty", out rp)
                || TryGetSingle(body, "repeatPenalty", out rp))
                cfg.RepetitionPenalty = rp;
            if (TryGetInt32(body, "repeat_last_n", out int rln) || TryGetInt32(body, "repeatLastN", out rln))
                cfg.PenaltyLastN = rln;
            if (TryGetSingle(body, "presence_penalty", out float pp) || TryGetSingle(body, "presencePenalty", out pp))
                cfg.PresencePenalty = pp;
            if (TryGetSingle(body, "frequency_penalty", out float fp) || TryGetSingle(body, "frequencyPenalty", out fp))
                cfg.FrequencyPenalty = fp;
            if (TryGetInt32(body, "seed", out int sd))
                cfg.Seed = sd;
            if (body.TryGetProperty("stop", out var stopEl) && stopEl.ValueKind == JsonValueKind.Array)
            {
                // The "stop" key being present in the request is treated as a
                // full replacement of any defaults so callers can intentionally
                // disable a global default by sending an empty array.
                cfg.StopSequences = new List<string>();
                foreach (var s in stopEl.EnumerateArray())
                    if (s.ValueKind == JsonValueKind.String && s.GetString() is string sv)
                        cfg.StopSequences.Add(sv);
            }
            return ApplyPinned(cfg, defaults);
        }

        /// <summary>
        /// Parse Ollama's body where every sampler value lives inside a nested
        /// <c>options</c> object (e.g. <c>{ "options": { "temperature": 0.7 } }</c>).
        /// </summary>
        public static SamplingConfig ParseOllama(JsonElement body, SamplingDefaults defaults = null)
        {
            var cfg = SeedFromDefaults(defaults?.Values);
            if (body.TryGetProperty("options", out var opts) && opts.ValueKind == JsonValueKind.Object)
            {
                if (TryGetSingle(opts, "temperature", out float temp)) cfg.Temperature = temp;
                if (TryGetInt32(opts, "top_k", out int tk)) cfg.TopK = tk;
                if (TryGetSingle(opts, "top_p", out float tp)) cfg.TopP = tp;
                if (TryGetSingle(opts, "min_p", out float mp)) cfg.MinP = mp;
                if (TryGetSingle(opts, "repeat_penalty", out float rp)) cfg.RepetitionPenalty = rp;
                if (TryGetInt32(opts, "repeat_last_n", out int rln)) cfg.PenaltyLastN = rln;
                if (TryGetSingle(opts, "presence_penalty", out float pp)) cfg.PresencePenalty = pp;
                if (TryGetSingle(opts, "frequency_penalty", out float fp)) cfg.FrequencyPenalty = fp;
                if (TryGetInt32(opts, "seed", out int sd)) cfg.Seed = sd;
                if (opts.TryGetProperty("stop", out var stopEl) && stopEl.ValueKind == JsonValueKind.Array)
                {
                    cfg.StopSequences = new List<string>();
                    foreach (var s in stopEl.EnumerateArray())
                        if (s.ValueKind == JsonValueKind.String && s.GetString() is string sv)
                            cfg.StopSequences.Add(sv);
                }
            }
            return ApplyPinned(cfg, defaults);
        }

        /// <summary>
        /// Parse OpenAI's flat body. The OpenAI spec also allows <c>stop</c> to
        /// be a single string instead of an array. Beyond the spec we accept the
        /// samplers llama.cpp / vLLM expose on the same surface (<c>top_k</c>,
        /// <c>min_p</c>, <c>repeat_penalty</c>) so an OpenAI-shaped client can
        /// still reach them.
        /// </summary>
        public static SamplingConfig ParseOpenAI(JsonElement body, SamplingDefaults defaults = null)
        {
            var cfg = SeedFromDefaults(defaults?.Values);
            if (TryGetSingle(body, "temperature", out float temp)) cfg.Temperature = temp;
            if (TryGetSingle(body, "top_p", out float tp)) cfg.TopP = tp;
            if (TryGetInt32(body, "top_k", out int tk)) cfg.TopK = tk;
            if (TryGetSingle(body, "min_p", out float mp)) cfg.MinP = mp;
            if (TryGetSingle(body, "repeat_penalty", out float rp) || TryGetSingle(body, "repetition_penalty", out rp))
                cfg.RepetitionPenalty = rp;
            if (TryGetSingle(body, "presence_penalty", out float pp)) cfg.PresencePenalty = pp;
            if (TryGetSingle(body, "frequency_penalty", out float fp)) cfg.FrequencyPenalty = fp;
            if (TryGetInt32(body, "repeat_last_n", out int rln)) cfg.PenaltyLastN = rln;
            if (TryGetInt32(body, "seed", out int sd)) cfg.Seed = sd;
            if (body.ValueKind == JsonValueKind.Object
                && body.TryGetProperty(IgnoreEosField, out var ignoreEos) && ignoreEos.ValueKind == JsonValueKind.True)
            {
                cfg.IgnoreEos = true;
                // Only the token limit (or the caller's stop strings) ends such a request:
                // the repetition guard would cut a fixed-length run short as surely as EOS.
                cfg.StopRepetition = false;
            }
            if (body.TryGetProperty("stop", out var stopEl))
            {
                if (stopEl.ValueKind == JsonValueKind.Array)
                {
                    cfg.StopSequences = new List<string>();
                    foreach (var s in stopEl.EnumerateArray())
                        if (s.ValueKind == JsonValueKind.String && s.GetString() is string stopVal)
                            cfg.StopSequences.Add(stopVal);
                }
                else if (stopEl.ValueKind == JsonValueKind.String && stopEl.GetString() is string singleStop)
                {
                    cfg.StopSequences = new List<string> { singleStop };
                }
            }
            return ApplyPinned(cfg, defaults);
        }

        /// <summary>The OpenAI-compatible request field (vLLM, SGLang and llama.cpp spell it
        /// the same) that keeps generating past end-of-sequence tokens; see
        /// <see cref="SamplingConfig.IgnoreEos"/>.</summary>
        public const string IgnoreEosField = "ignore_eos";

        /// <summary>
        /// Check <c>ignore_eos</c> before the request is admitted: a boolean or null, or the
        /// caller answers HTTP 400 with <paramref name="error"/>. A benchmark that spelled it
        /// <c>"true"</c> would otherwise run with ordinary EOS stops and report lengths it
        /// never asked for.
        /// </summary>
        public static bool TryValidateIgnoreEos(JsonElement body, out string error)
        {
            error = null;
            if (body.ValueKind == JsonValueKind.Object && body.TryGetProperty(IgnoreEosField, out var value)
                && value.ValueKind is not (JsonValueKind.True or JsonValueKind.False or JsonValueKind.Null))
            {
                error = $"{IgnoreEosField} must be a boolean.";
                return false;
            }
            return true;
        }

        /// <summary>
        /// Read the request's generation budget from whichever field this
        /// protocol flavour uses. Returns null when the client did not ask for
        /// one, so the caller can fall back to the server default. OpenAI
        /// renamed <c>max_tokens</c> to <c>max_completion_tokens</c>; clients are
        /// split across both, and some send an explicit <c>null</c> for the one
        /// they don't use.
        /// </summary>
        public static int? ReadRequestedMaxTokens(JsonElement body, params string[] fieldNames)
        {
            if (fieldNames == null)
                return null;
            foreach (string name in fieldNames)
            {
                if (TryGetInt32(body, name, out int value))
                    return value;
            }
            return null;
        }

        /// <summary>
        /// Read a numeric property, tolerating an explicit JSON <c>null</c> or a
        /// wrong-typed value instead of throwing. Several OpenAI clients
        /// serialise unset options as <c>"temperature": null</c>, and a 500 on
        /// the whole request is a poor answer to that.
        /// </summary>
        private static bool TryGetSingle(JsonElement obj, string name, out float value)
        {
            value = 0f;
            if (obj.ValueKind != JsonValueKind.Object)
                return false;
            if (!obj.TryGetProperty(name, out var el) || el.ValueKind != JsonValueKind.Number)
                return false;
            if (!el.TryGetSingle(out value))
                return false;
            return true;
        }

        private static bool TryGetInt32(JsonElement obj, string name, out int value)
        {
            value = 0;
            if (obj.ValueKind != JsonValueKind.Object)
                return false;
            if (!obj.TryGetProperty(name, out var el) || el.ValueKind != JsonValueKind.Number)
                return false;
            if (el.TryGetInt32(out value))
                return true;
            // Tolerate 40.0-style integers (JS clients frequently emit those).
            if (el.TryGetDouble(out double d) && d >= int.MinValue && d <= int.MaxValue)
            {
                value = (int)d;
                return true;
            }
            return false;
        }

        private static SamplingConfig ApplyPinned(SamplingConfig parsed, SamplingDefaults defaults)
            => defaults != null ? defaults.ApplyPinned(parsed) : parsed;

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
