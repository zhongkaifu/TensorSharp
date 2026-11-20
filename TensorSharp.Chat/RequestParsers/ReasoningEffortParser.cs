// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

#nullable enable
using System;
using System.Collections.Generic;
using System.Text.Json;
using TensorSharp.Runtime;

namespace TensorSharp.Server.RequestParsers
{
    /// <summary>
    /// Reads the request's reasoning-effort level: OpenAI chat's flat
    /// <c>reasoning_effort</c> (<c>low</c> | <c>medium</c> | <c>high</c>), or the
    /// Responses API's <c>reasoning: { "effort": ... }</c> object. The level is a
    /// prompt fact - Harmony's <c>Reasoning:</c> system line - and rides on the
    /// request's <see cref="SamplingConfig.ReasoningEffort"/>.
    ///
    /// <para>
    /// A request that explicitly sends <c>"think": false</c> and names no effort is
    /// mapped to <c>low</c>: GPT-OSS cannot switch its analysis channel off, so the
    /// shortest reasoning it was trained for is the closest thing to "off". An absent
    /// <c>think</c> leaves the family's default (<c>medium</c>). Every surface applies
    /// the same rule so a chat warmed through one surface is reused by the others.
    /// </para>
    /// <para>
    /// A named level also says whether to think (<see cref="AsksToReason"/>,
    /// <see cref="AsksNotToReason"/>): <see cref="ThinkingSwitchParser"/> and the
    /// Responses route read it when the request sends no thinking switch of its own.
    /// </para>
    /// </summary>
    internal static class ReasoningEffortParser
    {
        /// <summary>
        /// OpenAI's other levels, read as the nearest one a template renders. GPT-5
        /// added <c>minimal</c> and <c>none</c> below <c>low</c> and <c>xhigh</c> above
        /// <c>high</c>, and clients pass them through: pi sends <c>minimal</c> for its
        /// lowest thinking level, which was a 400 here. The trained lines stop at
        /// low/high (GPT-OSS cannot stop reasoning, so <c>none</c> is its shortest,
        /// <c>low</c>, exactly as <c>"think": false</c> is).
        /// </summary>
        private static readonly Dictionary<string, string> OpenAILevelAliases = new(StringComparer.OrdinalIgnoreCase)
        {
            ["none"] = ReasoningEffort.Low,
            ["minimal"] = ReasoningEffort.Low,
            ["xhigh"] = ReasoningEffort.High,
        };

        /// <summary>
        /// The effective level for the request, or null for the family's default.
        /// Returns false with <paramref name="error"/> set when the field is present
        /// but not one of the accepted levels, which the caller answers with HTTP 400.
        /// </summary>
        public static bool TryParse(JsonElement body, out string? effort, out string? error)
        {
            effort = null;
            error = null;
            string? raw = null;
            if (body.ValueKind == JsonValueKind.Object && body.TryGetProperty("reasoning_effort", out var flat))
            {
                if (flat.ValueKind == JsonValueKind.String)
                    raw = flat.GetString();
                else if (flat.ValueKind != JsonValueKind.Null)
                {
                    error = "reasoning_effort must be a string: one of \"low\", \"medium\" or \"high\".";
                    return false;
                }
            }
            else if (body.ValueKind == JsonValueKind.Object
                     && body.TryGetProperty("reasoning", out var reasoning)
                     && reasoning.ValueKind == JsonValueKind.Object
                     && reasoning.TryGetProperty("effort", out var nested))
            {
                if (nested.ValueKind == JsonValueKind.String)
                    raw = nested.GetString();
                else if (nested.ValueKind != JsonValueKind.Null)
                {
                    error = "reasoning.effort must be a string: one of \"low\", \"medium\" or \"high\".";
                    return false;
                }
            }

            if (raw != null && OpenAILevelAliases.TryGetValue(raw.Trim(), out string? nearest))
                raw = nearest;
            if (!ReasoningEffort.TryNormalize(raw, out string? level))
            {
                error = $"reasoning_effort must be one of \"low\", \"medium\" or \"high\" " +
                        $"(\"none\" and \"minimal\" read as low, \"xhigh\" as high; got \"{raw}\").";
                return false;
            }

            effort = ReasoningEffort.ForRequest(level, ThinkingSwitchParser.ExplicitlyOff(body));
            return true;
        }

        /// <summary>
        /// The level the request names, trimmed and as sent (before the aliases above):
        /// <c>reasoning_effort</c>, else <c>reasoning.effort</c>, read in the same order as
        /// <see cref="TryParse"/>. Null when it names none. A value of the wrong type is
        /// null here; <see cref="TryParse"/> answers it with HTTP 400.
        /// </summary>
        public static string? RequestedLevel(JsonElement body)
        {
            if (body.ValueKind != JsonValueKind.Object)
                return null;
            JsonElement value;
            if (!body.TryGetProperty("reasoning_effort", out value)
                && !(body.TryGetProperty("reasoning", out JsonElement reasoning)
                     && reasoning.ValueKind == JsonValueKind.Object
                     && reasoning.TryGetProperty("effort", out value)))
                return null;
            string? raw = value.ValueKind == JsonValueKind.String ? value.GetString()?.Trim() : null;
            return string.IsNullOrEmpty(raw) ? null : raw;
        }

        /// <summary>
        /// True when the request names a reasoning level other than <c>none</c>: it asks
        /// the model to reason, as OpenAI's reasoning models read the field (where
        /// <c>none</c> is "no reasoning"). Microsoft.Extensions.AI sends
        /// <c>ChatOptions.Reasoning.Effort</c> this way, with no thinking switch beside it.
        /// </summary>
        public static bool AsksToReason(JsonElement body)
            => RequestedLevel(body) is string level && !IsNone(level);

        /// <summary>True when the request names OpenAI's <c>none</c> level: do not reason.</summary>
        public static bool AsksNotToReason(JsonElement body)
            => RequestedLevel(body) is string level && IsNone(level);

        private static bool IsNone(string level) => string.Equals(level, "none", StringComparison.OrdinalIgnoreCase);
    }
}
