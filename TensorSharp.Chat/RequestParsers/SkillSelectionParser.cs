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

namespace TensorSharp.Server.RequestParsers
{
    /// <summary>
    /// Reads the <c>skills</c> selection off a chat request body.
    ///
    /// <para>
    /// The field is spelled identically on every surface — <c>/v1/chat/completions</c>,
    /// <c>/v1/responses</c>, the Ollama-compatible <c>/api/chat/ollama</c> and the Web
    /// UI's <c>/api/chat</c> — so unlike
    /// <see cref="ToolFunctionParser"/> there is nothing per-protocol to vary and one
    /// reader serves all four. It is a top-level array of strings, sitting alongside
    /// <c>tools</c>:
    /// </para>
    /// <code>
    /// { "model": "...", "messages": [...], "skills": ["pdf", "xlsx"] }
    /// </code>
    /// <para>
    /// Nothing here throws. A field of an unexpected kind is skipped exactly as
    /// <see cref="ToolFunctionParser"/> skips one, for the reason its class doc gives:
    /// a malformed corner of a request must fail that corner, not the whole completion
    /// (https://github.com/zhongkaifu/TensorSharp/issues/142).
    /// </para>
    /// </summary>
    internal static class SkillSelectionParser
    {
        /// <summary>The request field naming the skills to use.</summary>
        public const string SkillsField = "skills";

        /// <summary>The request field overriding whether unselected skills are advertised.</summary>
        public const string DiscoveryField = "skills_discovery";

        /// <summary>A request may suppress delegation but cannot expand host permissions.</summary>
        public static bool? ParseMultiAgent(JsonElement body)
        {
            if (body.ValueKind != JsonValueKind.Object
                || !body.TryGetProperty("multi_agent", out JsonElement value)) return null;
            return value.ValueKind switch
            {
                JsonValueKind.True => true,
                JsonValueKind.False => false,
                _ => null,
            };
        }

        /// <summary>
        /// Read the selection.
        /// </summary>
        /// <returns>
        /// The names in the order given; null when the field is ABSENT or is not an
        /// array; and an EMPTY list when the field is present and empty.
        ///
        /// <para>
        /// That last distinction is load-bearing. A server started with
        /// <c>--skill pdf</c> applies that selection to any request that does not carry
        /// its own, so a client needs some way to say "none for this one" — and
        /// <c>"skills": []</c> is the only spelling available to it. Collapsing an empty
        /// array to null, the way <see cref="ToolFunctionParser"/> collapses an empty
        /// tools array, would silently ignore the opt-out and hand the model a skill the
        /// caller explicitly declined.
        /// </para>
        /// </returns>
        public static List<string> Parse(JsonElement body)
        {
            if (body.ValueKind != JsonValueKind.Object
                || !body.TryGetProperty(SkillsField, out JsonElement skillsEl))
            {
                return null;
            }

            // A single name rather than an array is a natural thing to write and means
            // exactly one thing, so it is accepted rather than ignored.
            if (skillsEl.ValueKind == JsonValueKind.String)
            {
                string only = skillsEl.GetString();
                return string.IsNullOrWhiteSpace(only) ? null : new List<string> { only.Trim() };
            }

            if (skillsEl.ValueKind != JsonValueKind.Array)
                return null;

            // Present-but-empty is a deliberate "no skills", not "nothing was said".
            var names = new List<string>();
            foreach (JsonElement item in skillsEl.EnumerateArray())
            {
                string name = item.ValueKind switch
                {
                    JsonValueKind.String => item.GetString(),
                    // { "name": "pdf" } is what a client that models skills as objects
                    // sends; reading the name out of it costs nothing and spares the
                    // caller a confusing "no skill called '{...}'".
                    JsonValueKind.Object when item.TryGetProperty("name", out JsonElement n)
                                              && n.ValueKind == JsonValueKind.String
                        => n.GetString(),
                    JsonValueKind.Object when item.TryGetProperty("id", out JsonElement i)
                                              && i.ValueKind == JsonValueKind.String
                        => i.GetString(),
                    _ => null,
                };

                if (!string.IsNullOrWhiteSpace(name))
                    names.Add(name.Trim());
            }

            return names;
        }

        /// <summary>
        /// Read the discovery override.
        /// </summary>
        /// <returns>Null when absent or not a boolean, leaving the server default in force.</returns>
        public static bool? ParseDiscovery(JsonElement body)
        {
            if (body.ValueKind != JsonValueKind.Object
                || !body.TryGetProperty(DiscoveryField, out JsonElement el))
            {
                return null;
            }

            return el.ValueKind switch
            {
                JsonValueKind.True => true,
                JsonValueKind.False => false,
                _ => null,
            };
        }
    }
}
