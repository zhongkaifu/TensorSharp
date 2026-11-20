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
    /// A request that sends neither asks to think when it names a reasoning level other
    /// than <c>none</c> (<c>reasoning_effort</c>, see <see cref="ReasoningEffortParser"/>):
    /// that is how OpenAI's reasoning models read the field, and what OpenAI SDK clients
    /// send for it. Microsoft.Extensions.AI's <c>ChatOptions.Reasoning</c> sends
    /// <c>reasoning_effort</c> alone, and its thinking was off here (issue #386). An explicit
    /// switch is the more specific instruction, so it decides when both are sent:
    /// <c>"think": false</c> with <c>"reasoning_effort": "high"</c> does not think.
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
        /// Resolve the switch: <c>think</c> or <c>chat_template_kwargs.enable_thinking</c>,
        /// else on when the request names a reasoning level other than <c>none</c>, else
        /// "off", the endpoint's default. Returns false with <paramref name="error"/> set for
        /// a value of the wrong type or two values that disagree; the caller answers that
        /// with HTTP 400.
        /// </summary>
        public static bool TryParse(JsonElement body, out bool think, out IReadOnlyList<string> ignoredKwargs,
            out string? error)
            => TryParse(body, out think, out _, out ignoredKwargs, out error);

        /// <inheritdoc cref="TryParse(JsonElement, out bool, out IReadOnlyList{string}, out string?)"/>
        /// <param name="fromReasoningEffort">True when no switch was sent and
        /// <paramref name="think"/> is on because the request names a reasoning level.</param>
        public static bool TryParse(JsonElement body, out bool think, out bool fromReasoningEffort,
            out IReadOnlyList<string> ignoredKwargs, out string? error)
        {
            think = false;
            fromReasoningEffort = false;
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

            bool? requested = thinkField ?? enableThinking;
            fromReasoningEffort = requested == null && ReasoningEffortParser.AsksToReason(body);
            think = requested ?? fromReasoningEffort;
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
