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

namespace TensorSharp.Server.RequestParsers
{
    /// <summary>
    /// Reads whether an OpenAI chat-completions request asks the model to think.
    ///
    /// <para>
    /// Two spellings mean the same switch. <c>think</c> is TensorSharp's own (and
    /// Ollama's). <c>chat_template_kwargs.enable_thinking</c> is what llama.cpp, vLLM and
    /// SGLang read, and what clients written against them send: pi does in its
    /// <c>qwen-chat-template</c> and <c>chat-template</c> thinking formats. A request
    /// that sets both to different values is refused rather than resolved by a
    /// precedence rule the client cannot see.
    /// </para>
    /// <para>
    /// The other <c>chat_template_kwargs</c> keys are not passed to the template: the
    /// prompt has to be the same for every renderer and for the shared-prefix cache, and
    /// a key only one template reads would split both. They are returned in
    /// <c>ignoredKwargs</c> so the caller can say so.
    /// </para>
    /// </summary>
    internal static class ThinkingSwitchParser
    {
        public const string ThinkField = "think";
        public const string KwargsField = "chat_template_kwargs";
        public const string EnableThinkingKwarg = "enable_thinking";

        /// <summary>
        /// Resolve the switch. Absent everywhere is "off", the endpoint's default.
        /// Returns false with <paramref name="error"/> set for a value of the wrong type
        /// or two values that disagree; the caller answers that with HTTP 400.
        /// </summary>
        public static bool TryParse(JsonElement body, out bool think, out IReadOnlyList<string> ignoredKwargs,
            out string? error)
        {
            think = false;
            ignoredKwargs = Array.Empty<string>();
            error = null;
            if (body.ValueKind != JsonValueKind.Object)
                return true;

            if (!TryReadBool(body, ThinkField, ThinkField, out bool? thinkField, out error))
                return false;

            bool? enableThinking = null;
            if (body.TryGetProperty(KwargsField, out JsonElement kwargs) && kwargs.ValueKind != JsonValueKind.Null)
            {
                if (kwargs.ValueKind != JsonValueKind.Object)
                {
                    error = $"{KwargsField} must be an object.";
                    return false;
                }
                if (!TryReadBool(kwargs, EnableThinkingKwarg, KwargsField + "." + EnableThinkingKwarg,
                        out enableThinking, out error))
                    return false;

                List<string>? ignored = null;
                foreach (JsonProperty property in kwargs.EnumerateObject())
                {
                    if (property.NameEquals(EnableThinkingKwarg)) continue;
                    (ignored ??= new List<string>()).Add(property.Name);
                }
                if (ignored != null) ignoredKwargs = ignored;
            }

            if (thinkField.HasValue && enableThinking.HasValue && thinkField.Value != enableThinking.Value)
            {
                error = $"{ThinkField} ({Json(thinkField.Value)}) and {KwargsField}.{EnableThinkingKwarg} " +
                        $"({Json(enableThinking.Value)}) disagree; send one of them, or the same value in both.";
                return false;
            }

            think = thinkField ?? enableThinking ?? false;
            return true;
        }

        /// <summary>
        /// True when the request explicitly turns thinking off with either spelling (not
        /// merely leaves it out). Used where "off" and "absent" differ, such as the
        /// reasoning level GPT-OSS renders (see <see cref="ReasoningEffortParser"/>).
        /// </summary>
        public static bool ExplicitlyOff(JsonElement body)
        {
            if (body.ValueKind != JsonValueKind.Object)
                return false;
            if (body.TryGetProperty(ThinkField, out JsonElement think) && think.ValueKind == JsonValueKind.False)
                return true;
            return body.TryGetProperty(KwargsField, out JsonElement kwargs)
                && kwargs.ValueKind == JsonValueKind.Object
                && kwargs.TryGetProperty(EnableThinkingKwarg, out JsonElement enable)
                && enable.ValueKind == JsonValueKind.False;
        }

        private static bool TryReadBool(JsonElement obj, string name, string displayName, out bool? value, out string? error)
        {
            value = null;
            error = null;
            if (!obj.TryGetProperty(name, out JsonElement el))
                return true;
            switch (el.ValueKind)
            {
                case JsonValueKind.True: value = true; return true;
                case JsonValueKind.False: value = false; return true;
                case JsonValueKind.Null: return true;
                default:
                    error = $"{displayName} must be a boolean.";
                    return false;
            }
        }

        private static string Json(bool value) => value ? "true" : "false";
    }
}
