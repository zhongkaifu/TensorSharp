// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Text.Json;

namespace TensorSharp.Runtime
{
    public static partial class ChatTemplate
    {
        internal const string Inkling2MessageModel = "<|message_model|>";
        internal const string Inkling2EndMessage = "<|end_message|>";
        internal const string Inkling2EndSampling = "<|content_model_end_sampling|>";
        internal const string Inkling2ContentText = "<|content_text|>";
        internal const string Inkling2ContentThinking = "<|content_thinking|>";
        internal const string Inkling2ContentToolJson = "<|content_invoke_tool_json|>";

        /// <summary>The published template's <c>effort_map</c>: a named <c>reasoning_effort</c> as the
        /// <c>Thinking effort level</c> it renders.</summary>
        private static readonly Dictionary<string, double> Inkling2EffortLevels = new(StringComparer.OrdinalIgnoreCase)
        {
            ["none"] = 0.0, ["minimal"] = 0.1, ["low"] = 0.2, ["medium"] = 0.7, ["high"] = 0.9, ["xhigh"] = 0.99, ["max"] = 0.99,
        };

        /// <summary>The level the template renders when the request names none.</summary>
        private const double Inkling2DefaultEffort = 0.9;

        /// <summary>
        /// Inkling 2's chat format, as its published template (the GGUF's <c>tokenizer.chat_template</c>) renders it:
        /// every message is <c>&lt;|message_ROLE|&gt;[name]&lt;|content_KIND|&gt;body&lt;|end_message|&gt;</c>, the
        /// tools are one <c>tool_declare</c> system message of key-sorted compact JSON, a
        /// <c>Thinking effort level</c> system message comes before the first message that is not a system one, an
        /// assistant turn renders its reasoning, text and tool calls (<c>{"name":..,"args":{..}}</c>, the arguments'
        /// keys sorted) and closes with <c>&lt;|content_model_end_sampling|&gt;</c>, and a tool result is addressed to
        /// its call's function. There is no generation prompt: the model opens its own message.
        /// <para>
        /// Rendered here rather than through the lightweight Jinja engine: the template builds each turn in a
        /// <c>{% set %}</c> block inside <c>{% generation %}</c>, and its JSON depends on <c>tojson</c>'s
        /// <c>sort_keys</c> and <c>separators</c> and on <c>dictsort</c>.
        /// </para>
        /// </summary>
        public static string RenderInkling2(List<ChatMessage> messages, List<ToolFunction>? tools = null, string? reasoningEffort = null)
        {
            var sb = new StringBuilder();
            if (tools is { Count: > 0 })
            {
                sb.Append("<|message_system|>tool_declare<|content_xml|>[");
                for (int i = 0; i < tools.Count; i++)
                {
                    if (i > 0) sb.Append(',');
                    AppendInkling2ToolSpec(sb, tools[i]);
                }
                sb.Append(']').Append(Inkling2EndMessage);
            }

            bool effortEmitted = false;
            for (int i = 0; i < messages.Count; i++)
            {
                ChatMessage message = messages[i];
                string header = message.Role switch
                {
                    "user" => "<|message_user|>",
                    "assistant" => Inkling2MessageModel,
                    "system" => "<|message_system|>",
                    "tool" => "<|message_tool|>",
                    _ => throw new ArgumentException($"Inkling 2: unknown message role '{message.Role}'."),
                };
                if (!effortEmitted && message.Role != "system")
                {
                    AppendInkling2Effort(sb, reasoningEffort);
                    effortEmitted = true;
                }
                if (message.Role == "tool")
                    header += Inkling2ToolResultName(messages, i);

                bool assistant = message.Role == "assistant";
                if (assistant && IsRawTokenPlaceholder(message.Content))
                {
                    // KVCachePromptRenderer splices this turn's generated tokens in here. They carry the model's own
                    // message framing (it opens its reply itself), so the turn is the tokens and the end of sampling
                    // the live cache holds after them; the template's framing around them would double it.
                    sb.Append(message.Content).Append(Inkling2EndSampling);
                    continue;
                }
                bool calls = assistant && message.ToolCalls is { Count: > 0 };
                bool reasoning = assistant && !string.IsNullOrEmpty(message.Thinking);
                if (reasoning)
                    sb.Append(header).Append(Inkling2ContentThinking).Append(message.Thinking).Append(Inkling2EndMessage);
                if (message.Content != null && (message.Content.Length > 0 || !(assistant && (calls || reasoning))))
                    sb.Append(header).Append(Inkling2ContentText).Append(message.Content).Append(Inkling2EndMessage);
                if (calls)
                {
                    foreach (ToolCall call in message.ToolCalls!)
                        AppendInkling2ToolCall(sb, call);
                }
                if (assistant)
                    sb.Append(Inkling2EndSampling);
            }
            if (!effortEmitted)
                AppendInkling2Effort(sb, reasoningEffort);
            return sb.ToString();
        }

        /// <summary>True when <paramref name="content"/> is one of <see cref="KVCachePromptRenderer.MakePlaceholder"/>'s
        /// raw-token placeholders and nothing else.</summary>
        private static bool IsRawTokenPlaceholder(string? content)
        {
            char sentinel = KVCachePromptRenderer.PlaceholderSentinel;
            return content is { Length: > 2 } && content[0] == sentinel && content[^1] == sentinel
                && content.IndexOf(sentinel, 1) == content.Length - 1;
        }

        private static void AppendInkling2Effort(StringBuilder sb, string? reasoningEffort)
        {
            double level = Inkling2DefaultEffort;
            if (!string.IsNullOrWhiteSpace(reasoningEffort))
            {
                string key = reasoningEffort.Trim();
                if (!Inkling2EffortLevels.TryGetValue(key, out level)
                    && !(double.TryParse(key, NumberStyles.Float, CultureInfo.InvariantCulture, out level) && level >= 0 && level <= 0.99))
                    throw new ArgumentException($"Inkling 2: unknown reasoning_effort '{reasoningEffort}'.");
            }
            sb.Append("<|message_system|><|content_text|>Thinking effort level: ")
                .Append(level == 0 ? "0" : level.ToString("R", CultureInfo.InvariantCulture))
                .Append(Inkling2EndMessage);
        }

        /// <summary>The function a tool result answers: its call id's, as the template finds it; without an id (or one
        /// no call carries), the call in the same position among the results that follow the last assistant tool round.</summary>
        private static string Inkling2ToolResultName(List<ChatMessage> messages, int index)
        {
            string? id = messages[index].ToolCallId;
            if (!string.IsNullOrEmpty(id))
            {
                string? named = null;
                for (int j = 0; j < messages.Count; j++)
                {
                    if (messages[j].Role != "assistant" || messages[j].ToolCalls == null) continue;
                    foreach (ToolCall call in messages[j].ToolCalls!)
                        if (call.Id == id && !string.IsNullOrEmpty(call.Name)) named = call.Name;   // the template keeps the last
                }
                if (named != null) return named;
            }
            int round = index - 1, position = 0;
            while (round >= 0 && messages[round].Role == "tool") { round--; position++; }
            if (round >= 0 && messages[round].Role == "assistant" && messages[round].ToolCalls is { } roundCalls
                && position < roundCalls.Count)
                return roundCalls[position].Name;
            return "";
        }

        private static void AppendInkling2ToolSpec(StringBuilder sb, ToolFunction tool)
        {
            // {"description", "name", "parameters", "type"} with every object's keys sorted, compact.
            sb.Append("{\"description\":");
            AppendPythonJsonString(sb, tool.Description ?? "");
            sb.Append(",\"name\":");
            AppendPythonJsonString(sb, tool.Name ?? "");
            sb.Append(",\"parameters\":");
            if (tool.ParametersSchemaJson != null)
            {
                using JsonDocument schema = JsonDocument.Parse(tool.ParametersSchemaJson);
                AppendPythonJson(sb, schema.RootElement, compact: true, sortKeys: true);
            }
            else if (tool.Parameters.Count == 0 && tool.Required.Count == 0)
                sb.Append("{}");
            else
            {
                var function = (Dictionary<string, object>)BuildToolDeclaration(tool)["function"];
                AppendPythonJson(sb, JsonSerializer.SerializeToElement(function["parameters"]), compact: true, sortKeys: true);
            }
            sb.Append(",\"type\":\"function\"}");
        }

        /// <summary><c>&lt;|message_model|&gt;NAME&lt;|content_invoke_tool_json|&gt;{"name":NAME,"args":{..}}&lt;|end_message|&gt;</c>:
        /// the arguments' keys sorted (case-sensitively), their values compact JSON in the order given.</summary>
        private static void AppendInkling2ToolCall(StringBuilder sb, ToolCall call)
        {
            sb.Append(Inkling2MessageModel).Append(call.Name).Append(Inkling2ContentToolJson);
            sb.Append("{\"name\":");
            AppendPythonJsonString(sb, call.Name ?? "");
            sb.Append(",\"args\":{");
            bool first = true;
            if (call.Arguments != null)
            {
                foreach (KeyValuePair<string, object?> argument in call.Arguments.OrderBy(a => a.Key, StringComparer.Ordinal))
                {
                    if (!first) sb.Append(',');
                    first = false;
                    AppendPythonJsonString(sb, argument.Key);
                    sb.Append(':');
                    AppendPythonJson(sb, JsonSerializer.SerializeToElement(argument.Value), compact: true);
                }
            }
            sb.Append("}}").Append(Inkling2EndMessage);
        }
    }
}
