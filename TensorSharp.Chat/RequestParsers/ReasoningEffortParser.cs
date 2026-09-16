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
    /// </summary>
    internal static class ReasoningEffortParser
    {
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

            if (!ReasoningEffort.TryNormalize(raw, out string? level))
            {
                error = $"reasoning_effort must be one of \"low\", \"medium\" or \"high\" (got \"{raw}\").";
                return false;
            }

            bool thinkExplicitlyFalse = body.ValueKind == JsonValueKind.Object
                && body.TryGetProperty("think", out var think) && think.ValueKind == JsonValueKind.False;
            effort = ReasoningEffort.ForRequest(level, thinkExplicitlyFalse);
            return true;
        }
    }
}
