// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Text.Json;

namespace TensorSharp.Runtime
{
    /// <summary>
    /// What the MiniMax-M3 generation prompt says about reasoning. The M3.1 template has
    /// no thinking switch: its system block hard-codes "Current thinking mode: adaptive"
    /// and its generation prompt is a bare <c>]~b]ai\n</c>, so the model decides per turn
    /// whether to open <c>&lt;mm:think&gt;</c> or write the bare <c>&lt;/mm:think&gt;</c>
    /// that its history renders for a turn without reasoning. Forcing either way appends
    /// that tag to the generation prompt - an engine convention shared with llama.cpp's
    /// and vLLM's <c>thinking_mode</c> handling of the older M3 template, not part of the
    /// M3.1 template.
    /// </summary>
    public enum MiniMaxM3ThinkingMode
    {
        /// <summary>The template exactly as shipped: the model decides (matches HF and llama.cpp).</summary>
        Adaptive = 0,
        /// <summary>The prompt opens the reasoning block: <c>]~b]ai\n&lt;mm:think&gt;</c>.</summary>
        Enabled,
        /// <summary>The prompt closes it, "begin your response directly after the
        /// &lt;/mm:think&gt; prefix": <c>]~b]ai\n&lt;/mm:think&gt;</c>.</summary>
        Disabled,
    }

    public static partial class ChatTemplate
    {
        public const string MiniMaxM3Namespace = "]<]minimax[>[";
        public const string MiniMaxM3ImagePlaceholder = "]<]image[>[";
        public const string MiniMaxM3ThinkOpen = "<mm:think>";
        public const string MiniMaxM3ThinkClose = "</mm:think>";

        private const string MiniMaxM3TurnStart = "]~b]";
        private const string MiniMaxM3TurnEnd = "[e~[";
        private const string MiniMaxM3DefaultIdentity =
            "Your model version is MiniMax-M3.1, developed by MiniMax. Knowledge cutoff: January 2026. Founded in early 2022, " +
            "MiniMax is a global AI foundation model company committed to advancing the frontiers of AI towards AGI.";
        private const string MiniMaxM3ThinkingInstructions =
            "\n\n<thinking_instructions>\n" +
            "You have a thinking capability that allows you to reason step by step before responding. When thinking is " +
            "enabled, wrap your reasoning in <mm:think></mm:think> tags before your response. When thinking is disabled, " +
            "begin your response directly after the </mm:think> prefix. When thinking is adaptive, decide on your own " +
            "whether to think for the current turn.\n" +
            "Current thinking mode: adaptive. You are encouraged to think for complex decision-making, multi-step " +
            "reasoning, or when analyzing function/tool results.\n" +
            "</thinking_instructions>" +
            "\n\n<model_id>8bb8149fe</model_id>";
        private const string MiniMaxM3DefaultDeveloper = "You are a helpful assistant.";
        private const string MiniMaxM3ToolsHeader =
            "\n\n# Tools\nYou may call one or more tools to assist with the user query.\n" +
            "Here are the tools available in JSONSchema format:\n\n<tools>\n";
        private const string MiniMaxM3ToolsFooter =
            "</tools>\n\n" +
            "To call tools, wrap all invocations in a single ]<]minimax[>[<tool_call>]<]minimax[>[</tool_call> block. " +
            "Parameter values containing nested objects or arrays are recursively expanded into XML elements. Example:\n" +
            "\n]<]minimax[>[<tool_call>\n" +
            "]<]minimax[>[<invoke name=\"tool-name-1\">]<]minimax[>[<param-1>value-1]<]minimax[>[</param-1>" +
            "]<]minimax[>[<param-2>]<]minimax[>[<item>]<]minimax[>[<key-a>val-a]<]minimax[>[</key-a>" +
            "]<]minimax[>[<key-b>val-b]<]minimax[>[</key-b>]<]minimax[>[</item>]<]minimax[>[</param-2>" +
            "]<]minimax[>[</invoke>\n" +
            "]<]minimax[>[<invoke name=\"tool-name-2\">]<]minimax[>[<param-1>value-1]<]minimax[>[</param-1>" +
            "]<]minimax[>[</invoke>\n" +
            "]<]minimax[>[</tool_call>";

        /// <summary>
        /// MiniMax-M3.1 chat format, reproducing the GGUF's (= HF's) <c>chat_template.jinja</c>
        /// byte for byte; InferenceWeb.Tests/Fixtures/MiniMaxM3/chat_reference.json holds the
        /// HF renders it is checked against. The template is bypassed rather than run because
        /// it is built on a recursive macro (<c>to_xml</c>), <c>namespace()</c>, <c>items()</c>
        /// and Python <c>json.dumps</c> spacing, which the lightweight Jinja engine renders
        /// inconsistently.
        /// <list type="bullet">
        /// <item>The header is always <c>]~!b[]~b]system</c> (a leading <c>root</c> message,
        /// else the default identity) + the thinking instructions + <c>&lt;model_id&gt;</c>
        /// [+ <c>&lt;effort&gt;</c>], then <c>]~b]developer</c> (the first system/developer
        /// message, else "You are a helpful assistant.") + the tool declarations.</item>
        /// <item>Every past assistant turn re-renders its reasoning as
        /// <c>&lt;mm:think&gt;R&lt;/mm:think&gt;</c>, or a bare <c>&lt;/mm:think&gt;</c> without
        /// any, so turn N+1's prompt extends turn N's generated text. A turn whose content is a
        /// KV-cache raw-token placeholder gets neither: the spliced tokens already hold the
        /// framing the model wrote.</item>
        /// <item>Consecutive tool results share one <c>]~b]tool</c> turn. Unlike the template,
        /// a result with no preceding call is rendered rather than refused: raw-token splicing
        /// of a tool round clears the structured calls the template would look for.</item>
        /// </list>
        /// Known limits: <c>current_date</c>/<c>current_location</c> on a root message and a
        /// <c>model_identity</c> variable have no TensorSharp request field; a tool declaration
        /// prints <c>name</c>, <c>description</c> (omitted when empty) and <c>parameters</c> in
        /// that order, the shape OpenAI clients send.
        /// </summary>
        public static string RenderMiniMaxM3(List<ChatMessage> messages, bool addGenerationPrompt = true,
            List<ToolFunction>? tools = null, MiniMaxM3ThinkingMode thinking = MiniMaxM3ThinkingMode.Adaptive,
            string? reasoningEffort = null)
        {
            ArgumentNullException.ThrowIfNull(messages);
            List<ChatMessage> conversation = InjectMultimodalTokens(messages, "minimax-m3");

            ChatMessage? system = null, developer = null;
            int first = 0;
            if (conversation.Count > 0 && conversation[0].Role == "root")
            {
                system = conversation[0];
                first = 1;
                if (conversation.Count > 1 && conversation[1].Role is "system" or "developer")
                {
                    developer = conversation[1];
                    first = 2;
                }
            }
            else if (conversation.Count > 0 && conversation[0].Role is "system" or "developer")
            {
                developer = conversation[0];
                first = 1;
            }

            var sb = new StringBuilder();
            sb.Append("]~!b[").Append(MiniMaxM3TurnStart).Append("system\n");
            sb.Append(string.IsNullOrEmpty(system?.Content) ? MiniMaxM3DefaultIdentity : system.Content);
            sb.Append(MiniMaxM3ThinkingInstructions);
            if (!string.IsNullOrEmpty(reasoningEffort))
                sb.Append("\n\n<effort>").Append(reasoningEffort).Append("</effort>");
            sb.Append(MiniMaxM3TurnEnd).Append('\n');

            sb.Append(MiniMaxM3TurnStart).Append("developer\n");
            AppendMiniMaxM3Developer(sb, developer);
            if (tools is { Count: > 0 })
            {
                sb.Append(MiniMaxM3ToolsHeader);
                foreach (ToolFunction tool in tools)
                    sb.Append("<tool>").Append(MiniMaxM3ToolJson(tool)).Append("</tool>\n");
                sb.Append(MiniMaxM3ToolsFooter);
            }
            sb.Append(MiniMaxM3TurnEnd).Append('\n');

            for (int i = first; i < conversation.Count; i++)
            {
                ChatMessage message = conversation[i];
                switch (message.Role)
                {
                    case "assistant":
                        sb.Append(MiniMaxM3TurnStart).Append("ai\n");
                        AppendMiniMaxM3Assistant(sb, message);
                        sb.Append(MiniMaxM3TurnEnd).Append('\n');
                        break;
                    case "tool":
                        if (i == first || conversation[i - 1].Role != "tool")
                            sb.Append(MiniMaxM3TurnStart).Append("tool");
                        sb.Append("\n<response>").Append(message.Content ?? "").Append("</response>");
                        if (i == conversation.Count - 1 || conversation[i + 1].Role != "tool")
                            sb.Append(MiniMaxM3TurnEnd).Append('\n');
                        break;
                    case "system":
                    case "developer":
                        // Mid-conversation system/developer messages keep their position as
                        // the lower-priority developer role.
                        sb.Append(MiniMaxM3TurnStart).Append("developer\n");
                        AppendMiniMaxM3Developer(sb, message);
                        sb.Append(MiniMaxM3TurnEnd).Append('\n');
                        break;
                    case "user":
                        sb.Append(MiniMaxM3TurnStart).Append("user\n").Append(message.Content ?? "")
                            .Append(MiniMaxM3TurnEnd).Append('\n');
                        break;
                    default:
                        // The template would drop the message without a word (a root message
                        // anywhere but first included); the model would then answer a
                        // conversation the caller did not send.
                        throw new ArgumentException(
                            $"Unsupported MiniMax-M3 message role '{message.Role}' at position {i}: the template renders " +
                            "root (first message only), system, developer, user, assistant and tool.", nameof(messages));
                }
            }

            if (addGenerationPrompt)
            {
                sb.Append(MiniMaxM3TurnStart).Append("ai\n");
                if (thinking == MiniMaxM3ThinkingMode.Enabled)
                    sb.Append(MiniMaxM3ThinkOpen);
                else if (thinking == MiniMaxM3ThinkingMode.Disabled)
                    sb.Append(MiniMaxM3ThinkClose);
            }
            return sb.ToString();
        }

        /// <summary>
        /// One <c>]&lt;]image[&gt;[</c> per image, before the message text, as the template's
        /// <c>visible_text</c> renders an image part; the vision injector expands it into
        /// <c>]&lt;]start of image[&gt;[</c> + N rows + <c>]&lt;]end of image[&gt;[</c>.
        /// Sampled video frames are offered as images too, each after the template's video
        /// timestamp text (<c>]&lt;]1.0 seconds[&gt;[</c>): the M3 video path (two frames per
        /// temporal patch, <c>]&lt;]video[&gt;[</c> rows) has no TensorSharp encoder, and an
        /// unexpanded <c>]&lt;]video[&gt;[</c> would drop the frames silently.
        /// </summary>
        internal static void AppendMiniMaxM3MediaPlaceholders(ChatMessage message, StringBuilder text)
        {
            if (message.ImagePaths == null)
                return;
            bool timed = message.ImageTimestamps?.Count == message.ImagePaths.Count;
            for (int i = 0; i < message.ImagePaths.Count; i++)
            {
                if (timed && message.ImageTimestamps![i] is double seconds)
                    text.Append("]<]").Append(seconds.ToString("F1", CultureInfo.InvariantCulture)).Append(" seconds[>[");
                text.Append(MiniMaxM3ImagePlaceholder);
            }
        }

        private static void AppendMiniMaxM3Developer(StringBuilder sb, ChatMessage? message)
            => sb.Append(string.IsNullOrEmpty(message?.Content) ? MiniMaxM3DefaultDeveloper : message.Content);

        private static void AppendMiniMaxM3Assistant(StringBuilder sb, ChatMessage message)
        {
            string content = message.Content ?? "";
            // A cached turn's raw tokens begin with the model's own <mm:think>/</mm:think>
            // (or follow a generation prompt that wrote it; the KV renderer restores that
            // suffix in front of the placeholder).
            if (content.IndexOf(KVCachePromptRenderer.PlaceholderSentinel) < 0)
            {
                string reasoning = "";
                if (message.Thinking != null)
                    reasoning = message.Thinking;
                else if (content.Contains(MiniMaxM3ThinkClose, StringComparison.Ordinal))
                {
                    // content.split('</mm:think>')[0].strip('\n').split('<mm:think>')[-1].strip('\n')
                    // and content.split('</mm:think>')[-1].strip('\n').
                    string head = content.Substring(0, content.IndexOf(MiniMaxM3ThinkClose, StringComparison.Ordinal)).Trim('\n');
                    int open = head.LastIndexOf(MiniMaxM3ThinkOpen, StringComparison.Ordinal);
                    reasoning = (open < 0 ? head : head.Substring(open + MiniMaxM3ThinkOpen.Length)).Trim('\n');
                    content = content.Substring(content.LastIndexOf(MiniMaxM3ThinkClose, StringComparison.Ordinal)
                        + MiniMaxM3ThinkClose.Length).Trim('\n');
                }
                if (reasoning.Length > 0)
                    sb.Append(MiniMaxM3ThinkOpen).Append(reasoning).Append(MiniMaxM3ThinkClose);
                else
                    sb.Append(MiniMaxM3ThinkClose);
            }
            sb.Append(content);

            if (message.ToolCalls is not { Count: > 0 })
                return;
            sb.Append(MiniMaxM3Namespace).Append("<tool_call>\n");
            foreach (ToolCall call in message.ToolCalls)
            {
                sb.Append(MiniMaxM3Namespace).Append("<invoke name=\"").Append(call.Name).Append("\">");
                if (call.Arguments != null)
                {
                    foreach (var argument in call.Arguments)
                    {
                        if (IsMiniMaxM3None(argument.Value))
                            continue;
                        sb.Append(MiniMaxM3Namespace).Append('<').Append(argument.Key).Append('>');
                        AppendMiniMaxM3Xml(sb, argument.Value);
                        sb.Append(MiniMaxM3Namespace).Append("</").Append(argument.Key).Append('>');
                    }
                }
                sb.Append(MiniMaxM3Namespace).Append("</invoke>\n");
            }
            sb.Append(MiniMaxM3Namespace).Append("</tool_call>");
        }

        private static bool IsMiniMaxM3None(object? value)
            => value == null || value is JsonElement { ValueKind: JsonValueKind.Null or JsonValueKind.Undefined };

        /// <summary>The template's <c>to_xml</c> macro: strings raw, booleans through
        /// <c>tojson</c>, numbers through Python <c>str()</c>, mappings as one element per
        /// non-null key and sequences as <c>&lt;item&gt;</c> elements.</summary>
        private static void AppendMiniMaxM3Xml(StringBuilder sb, object? value)
        {
            switch (value)
            {
                case null:
                    return;
                case string text:
                    sb.Append(text);
                    return;
                case bool flag:
                    sb.Append(flag ? "true" : "false");
                    return;
                case JsonElement element:
                    AppendMiniMaxM3Xml(sb, element);
                    return;
                case double number:
                    sb.Append(PythonFloatRepr(number, json: false));
                    return;
                case float number:
                    // Through its shortest spelling: widening 0.1f directly gives 0.10000000149011612.
                    sb.Append(PythonFloatRepr(double.Parse(number.ToString("R", CultureInfo.InvariantCulture),
                        NumberStyles.Float, CultureInfo.InvariantCulture), json: false));
                    return;
                case IDictionary map:
                    foreach (DictionaryEntry entry in map)
                    {
                        if (IsMiniMaxM3None(entry.Value))
                            continue;
                        string key = Convert.ToString(entry.Key, CultureInfo.InvariantCulture) ?? "";
                        sb.Append(MiniMaxM3Namespace).Append('<').Append(key).Append('>');
                        AppendMiniMaxM3Xml(sb, entry.Value);
                        sb.Append(MiniMaxM3Namespace).Append("</").Append(key).Append('>');
                    }
                    return;
                case IEnumerable items:
                    foreach (object? item in items)
                    {
                        sb.Append(MiniMaxM3Namespace).Append("<item>");
                        AppendMiniMaxM3Xml(sb, item);
                        sb.Append(MiniMaxM3Namespace).Append("</item>");
                    }
                    return;
                case IFormattable formattable:
                    sb.Append(formattable.ToString(null, CultureInfo.InvariantCulture));
                    return;
                default:
                    sb.Append(value);
                    return;
            }
        }

        private static void AppendMiniMaxM3Xml(StringBuilder sb, JsonElement element)
        {
            switch (element.ValueKind)
            {
                case JsonValueKind.String:
                    sb.Append(element.GetString());
                    return;
                case JsonValueKind.True:
                    sb.Append("true");
                    return;
                case JsonValueKind.False:
                    sb.Append("false");
                    return;
                case JsonValueKind.Number:
                    sb.Append(PythonNumber(element, json: false));
                    return;
                case JsonValueKind.Object:
                    foreach (JsonProperty property in element.EnumerateObject())
                    {
                        if (IsMiniMaxM3None(property.Value))
                            continue;
                        sb.Append(MiniMaxM3Namespace).Append('<').Append(property.Name).Append('>');
                        AppendMiniMaxM3Xml(sb, property.Value);
                        sb.Append(MiniMaxM3Namespace).Append("</").Append(property.Name).Append('>');
                    }
                    return;
                case JsonValueKind.Array:
                    foreach (JsonElement item in element.EnumerateArray())
                    {
                        sb.Append(MiniMaxM3Namespace).Append("<item>");
                        AppendMiniMaxM3Xml(sb, item);
                        sb.Append(MiniMaxM3Namespace).Append("</item>");
                    }
                    return;
                default:
                    return;
            }
        }

        /// <summary>
        /// <c>tool.function | tojson(ensure_ascii=False)</c> for a declared tool: Python
        /// <c>json.dumps</c> spacing (", " and ": "), non-ASCII kept, schema key order kept.
        /// </summary>
        private static string MiniMaxM3ToolJson(ToolFunction tool)
        {
            var sb = new StringBuilder();
            sb.Append("{\"name\": ");
            AppendPythonJsonString(sb, tool.Name ?? "");
            if (!string.IsNullOrEmpty(tool.Description))
            {
                sb.Append(", \"description\": ");
                AppendPythonJsonString(sb, tool.Description);
            }
            sb.Append(", \"parameters\": ");
            if (tool.ParametersSchemaJson != null)
            {
                using JsonDocument schema = JsonDocument.Parse(tool.ParametersSchemaJson);
                AppendPythonJson(sb, schema.RootElement);
            }
            else
            {
                // A flat declaration (no JSON Schema object) has only the parsed fields
                // left; print them in the OpenAI shape the hardcoded renderers share.
                var function = (Dictionary<string, object>)BuildToolDeclaration(tool)["function"];
                using JsonDocument parameters = JsonDocument.Parse(JsonSerializer.Serialize(function["parameters"]));
                AppendPythonJson(sb, parameters.RootElement);
            }
            sb.Append('}');
            return sb.ToString();
        }

        /// <summary>
        /// Python <c>json.dumps(value, ensure_ascii=False)</c> of a parsed JSON value; with
        /// <paramref name="compact"/> the <c>separators=(",", ":")</c> spelling, and with
        /// <paramref name="sortKeys"/> <c>sort_keys=True</c> (every object's keys in code-point order).
        /// </summary>
        internal static void AppendPythonJson(StringBuilder sb, JsonElement element, bool compact = false, bool sortKeys = false)
        {
            string itemSeparator = compact ? "," : ", ", keySeparator = compact ? ":" : ": ";
            switch (element.ValueKind)
            {
                case JsonValueKind.Object:
                {
                    sb.Append('{');
                    bool first = true;
                    IEnumerable<JsonProperty> properties = element.EnumerateObject();
                    if (sortKeys)
                        properties = properties.OrderBy(p => p.Name, StringComparer.Ordinal);
                    foreach (JsonProperty property in properties)
                    {
                        if (!first) sb.Append(itemSeparator);
                        first = false;
                        AppendPythonJsonString(sb, property.Name);
                        sb.Append(keySeparator);
                        AppendPythonJson(sb, property.Value, compact, sortKeys);
                    }
                    sb.Append('}');
                    return;
                }
                case JsonValueKind.Array:
                {
                    sb.Append('[');
                    bool first = true;
                    foreach (JsonElement item in element.EnumerateArray())
                    {
                        if (!first) sb.Append(itemSeparator);
                        first = false;
                        AppendPythonJson(sb, item, compact, sortKeys);
                    }
                    sb.Append(']');
                    return;
                }
                case JsonValueKind.String:
                    AppendPythonJsonString(sb, element.GetString() ?? "");
                    return;
                case JsonValueKind.Number:
                    sb.Append(PythonNumber(element, json: true));
                    return;
                case JsonValueKind.True:
                    sb.Append("true");
                    return;
                case JsonValueKind.False:
                    sb.Append("false");
                    return;
                default:
                    sb.Append("null");
                    return;
            }
        }

        internal static void AppendPythonJsonString(StringBuilder sb, string text)
        {
            sb.Append('"');
            foreach (char c in text)
            {
                switch (c)
                {
                    case '"': sb.Append("\\\""); break;
                    case '\\': sb.Append("\\\\"); break;
                    case '\n': sb.Append("\\n"); break;
                    case '\r': sb.Append("\\r"); break;
                    case '\t': sb.Append("\\t"); break;
                    case '\b': sb.Append("\\b"); break;
                    case '\f': sb.Append("\\f"); break;
                    default:
                        if (c < 0x20)
                            sb.Append("\\u").Append(((int)c).ToString("x4", CultureInfo.InvariantCulture));
                        else
                            sb.Append(c);
                        break;
                }
            }
            sb.Append('"');
        }

        /// <summary>
        /// A JSON number as Python prints it after <c>json.loads</c>: an integer literal stays
        /// an <c>int</c> (any size), anything with a fraction or exponent becomes a
        /// <c>float</c> and prints as its shortest round-trip <c>repr</c> ("2.0", "1e+20").
        /// </summary>
        private static string PythonNumber(JsonElement element, bool json)
        {
            string raw = element.GetRawText();
            if (raw.AsSpan().IndexOfAny('.', 'e', 'E') < 0)
            {
                // -0 is int 0 in Python.
                return raw.TrimStart('-').TrimStart('0').Length == 0 ? "0" : raw;
            }
            return PythonFloatRepr(double.Parse(raw, NumberStyles.Float, CultureInfo.InvariantCulture), json);
        }

        /// <summary>
        /// Python's <c>repr(float)</c>: the shortest digits that round-trip, positional when
        /// the decimal exponent is in [-4, 16), scientific ("1e-05", "1.5e+300") otherwise, and
        /// always with a fractional part when positional ("3.0"). <paramref name="json"/>
        /// spells the non-finite values as <c>json.dumps</c> does.
        /// </summary>
        internal static string PythonFloatRepr(double value, bool json)
        {
            if (double.IsNaN(value)) return json ? "NaN" : "nan";
            if (double.IsPositiveInfinity(value)) return json ? "Infinity" : "inf";
            if (double.IsNegativeInfinity(value)) return json ? "-Infinity" : "-inf";
            if (value == 0) return double.IsNegative(value) ? "-0.0" : "0.0";

            // "R" is the shortest round-trip form; re-read its digits and exponent.
            string r = Math.Abs(value).ToString("R", CultureInfo.InvariantCulture);
            int exponentAt = r.IndexOfAny(new[] { 'E', 'e' });
            string mantissa = exponentAt < 0 ? r : r.Substring(0, exponentAt);
            int exponent = exponentAt < 0 ? 0 : int.Parse(r.Substring(exponentAt + 1), CultureInfo.InvariantCulture);
            int dot = mantissa.IndexOf('.');
            string digits = dot < 0 ? mantissa : mantissa.Remove(dot, 1);
            int pointPosition = (dot < 0 ? mantissa.Length : dot) + exponent;
            int leadingZeros = 0;
            while (leadingZeros < digits.Length - 1 && digits[leadingZeros] == '0')
                leadingZeros++;
            digits = digits.Substring(leadingZeros).TrimEnd('0');
            pointPosition -= leadingZeros;
            if (digits.Length == 0) digits = "0";

            // value = 0.d1d2... x 10^pointPosition; the scientific exponent is one less.
            int scientific = pointPosition - 1;
            var sb = new StringBuilder();
            if (value < 0) sb.Append('-');
            if (scientific < -4 || scientific >= 16)
            {
                sb.Append(digits[0]);
                if (digits.Length > 1) sb.Append('.').Append(digits, 1, digits.Length - 1);
                sb.Append('e').Append(scientific < 0 ? '-' : '+')
                    .Append(Math.Abs(scientific).ToString("00", CultureInfo.InvariantCulture));
            }
            else if (pointPosition <= 0)
                sb.Append("0.").Append('0', -pointPosition).Append(digits);
            else if (pointPosition >= digits.Length)
                sb.Append(digits).Append('0', pointPosition - digits.Length).Append(".0");
            else
                sb.Append(digits, 0, pointPosition).Append('.').Append(digits, pointPosition, digits.Length - pointPosition);
            return sb.ToString();
        }
    }
}
