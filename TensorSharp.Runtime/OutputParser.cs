// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System;
using System.Collections.Generic;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace TensorSharp.Runtime
{
    /// <summary>
    /// Represents a tool function definition provided to the model.
    /// </summary>
    public class ToolFunction
    {
        public string Name { get; set; } = string.Empty;
        public string Description { get; set; } = string.Empty;
        public Dictionary<string, ToolParameter> Parameters { get; set; } = new();
        public List<string> Required { get; set; } = new();
        public CacheControlMarker? CacheControl { get; set; }
        /// <summary>Original parameter schema for protocols that enforce it during decoding.</summary>
        [System.Text.Json.Serialization.JsonIgnore]
        public string? ParametersSchemaJson { get; set; }

        /// <summary>
        /// Parse a list of tool definitions from JSON, accepting every shape a
        /// caller plausibly writes:
        ///
        /// <list type="bullet">
        /// <item>this type's own flat shape —
        ///   <c>{"name", "description", "parameters": {"city": {...}}, "required": [...]}</c></item>
        /// <item>the JSON Schema shape the OpenAI API uses, where
        ///   <c>parameters</c> is a schema object —
        ///   <c>{"name", "parameters": {"type": "object", "properties": {...}, "required": [...]}}</c></item>
        /// <item>either of those inside the OpenAI tools wrapper —
        ///   <c>{"type": "function", "function": {...}}</c></item>
        /// </list>
        ///
        /// The second is what anyone copying a tool definition out of an API
        /// request writes, and the server has always accepted it; the CLI's
        /// <c>--tools</c> flag used to deserialize straight into this type and
        /// die with an unhandled <c>JsonException</c> ("The JSON value could not
        /// be converted to ToolParameter") on the schema's own <c>"type":
        /// "object"</c>.
        /// </summary>
        /// <exception cref="JsonException">
        /// The document is not valid JSON, or is not an array/object of tool
        /// definitions. The message names what was expected.
        /// </exception>
        public static List<ToolFunction> ParseList(string json)
        {
            if (string.IsNullOrWhiteSpace(json))
                return new List<ToolFunction>();

            using JsonDocument doc = JsonDocument.Parse(json);
            JsonElement root = doc.RootElement;

            // Tolerate a single object, and the OpenAI request shape where the
            // array hangs off a "tools" property.
            if (root.ValueKind == JsonValueKind.Object && root.TryGetProperty("tools", out JsonElement toolsProp))
                root = toolsProp;

            var result = new List<ToolFunction>();
            if (root.ValueKind == JsonValueKind.Object)
            {
                result.Add(ParseOne(root));
                return result;
            }
            if (root.ValueKind != JsonValueKind.Array)
                throw new JsonException(
                    "Tool definitions must be a JSON array of objects (or a single object), " +
                    $"but the document root is {root.ValueKind}.");

            foreach (JsonElement entry in root.EnumerateArray())
            {
                if (entry.ValueKind != JsonValueKind.Object)
                    throw new JsonException(
                        $"Each tool definition must be a JSON object, but found {entry.ValueKind}.");
                result.Add(ParseOne(entry));
            }
            return result;
        }

        private static ToolFunction ParseOne(JsonElement entry)
        {
            // OpenAI wrapper: {"type": "function", "function": {...}}
            if (entry.TryGetProperty("function", out JsonElement inner) && inner.ValueKind == JsonValueKind.Object)
                entry = inner;

            var fn = new ToolFunction
            {
                Name = GetString(entry, "name") ?? string.Empty,
                Description = GetString(entry, "description") ?? string.Empty,
            };

            if (!entry.TryGetProperty("parameters", out JsonElement parameters)
                || parameters.ValueKind != JsonValueKind.Object)
            {
                CollectRequired(entry, fn.Required);
                return fn;
            }

            // JSON Schema shape: the properties live one level down and the
            // required list belongs to the schema, not the function.
            JsonElement propertyBag = parameters;
            if (parameters.TryGetProperty("properties", out JsonElement properties)
                && properties.ValueKind == JsonValueKind.Object)
            {
                fn.ParametersSchemaJson = parameters.GetRawText();
                propertyBag = properties;
                CollectRequired(parameters, fn.Required);
            }

            foreach (JsonProperty prop in propertyBag.EnumerateObject())
            {
                if (prop.Value.ValueKind != JsonValueKind.Object)
                    continue;   // a schema keyword sitting next to "properties" ("type", "$schema", ...)
                var param = new ToolParameter
                {
                    Type = ReadSchemaType(prop.Value) ?? string.Empty,
                    Description = GetString(prop.Value, "description") ?? string.Empty,
                };
                if (prop.Value.TryGetProperty("enum", out JsonElement enumValues)
                    && enumValues.ValueKind == JsonValueKind.Array)
                {
                    // A string member is stored unquoted, because the renderers
                    // add the quotes themselves; anything else keeps its raw JSON
                    // text. GetRawText rather than ToString: ToString renders a
                    // boolean in .NET's casing ("True", which is not JSON) and
                    // renders null as an empty string, which reaches the model as
                    // a meaningless empty choice in the enum.
                    foreach (JsonElement v in enumValues.EnumerateArray())
                        param.Enum.Add(v.ValueKind == JsonValueKind.String ? v.GetString() : v.GetRawText());
                }
                fn.Parameters[prop.Name] = param;
            }

            // A flat definition carries "required" on the function itself.
            if (fn.Required.Count == 0)
                CollectRequired(entry, fn.Required);
            return fn;
        }

        private static string GetString(JsonElement obj, string name)
            => obj.TryGetProperty(name, out JsonElement v) && v.ValueKind == JsonValueKind.String
                ? v.GetString()
                : null;

        /// <summary>
        /// Read a property schema's <c>type</c>. JSON Schema allows a union, and
        /// <c>"type": ["string", "null"]</c> is how every schema generator spells
        /// a nullable field, while <see cref="ToolParameter.Type"/> holds a single
        /// name that the renderers switch on — an unrecognised one degrades the
        /// parameter to <c>any</c> and drops its enum. Keep the first real type
        /// and drop the <c>"null"</c> member, whose meaning <c>required</c>
        /// already carries.
        /// </summary>
        private static string ReadSchemaType(JsonElement schema)
        {
            if (!schema.TryGetProperty("type", out JsonElement type))
                return null;
            if (type.ValueKind == JsonValueKind.String)
                return type.GetString();
            if (type.ValueKind != JsonValueKind.Array)
                return null;

            string first = null;
            foreach (JsonElement v in type.EnumerateArray())
            {
                if (v.ValueKind != JsonValueKind.String)
                    continue;
                string name = v.GetString();
                first ??= name;
                if (name != "null")
                    return name;
            }
            return first;
        }

        private static void CollectRequired(JsonElement obj, List<string> into)
        {
            if (!obj.TryGetProperty("required", out JsonElement req) || req.ValueKind != JsonValueKind.Array)
                return;
            foreach (JsonElement v in req.EnumerateArray())
                if (v.ValueKind == JsonValueKind.String)
                    into.Add(v.GetString());
        }
    }

    public class ToolParameter
    {
        public string Type { get; set; } = string.Empty;
        public string Description { get; set; } = string.Empty;
        public List<string> Enum { get; set; } = new();
    }

    /// <summary>
    /// Represents a tool call extracted from model output.
    /// </summary>
    public class ToolCall
    {
        /// <summary>Source protocol call id, used to associate parallel tool results.</summary>
        public string? Id { get; set; }
        public string Name { get; set; } = string.Empty;
        public Dictionary<string, object> Arguments { get; set; } = new();
        public int Index { get; set; }

        public override string ToString()
        {
            string args = Arguments != null ? JsonSerializer.Serialize(Arguments) : "{}";
            return $"{Name}({args})";
        }
    }

    /// <summary>
    /// Parsed output from a model generation step.
    /// </summary>
    public class ParsedOutput
    {
        public string Content { get; set; } = "";
        public string Thinking { get; set; } = "";
        public List<ToolCall>? ToolCalls { get; set; }

        /// <summary>
        /// Tool-call body text the model wrote since the last <c>Add</c>, while the call
        /// is still incomplete. Progress signal only — a UI shows "the model is writing
        /// code" with it; nothing should try to parse it. A parser that does not track
        /// call bodies incrementally leaves it empty, which is the old behavior: the
        /// whole call surfaces at once in <see cref="ToolCalls"/> when it completes.
        /// </summary>
        public string ToolCallText { get; set; } = "";

        /// <summary>The name of the tool call in progress, once enough of it has been
        /// written to tell. Null outside a call and before the name is complete.</summary>
        public string? ToolCallName { get; set; }
    }

    /// <summary>
    /// Streaming parser that extracts thinking content, regular content, and tool calls
    /// from model output. Handles model-specific tag formats.
    /// </summary>
    public interface IOutputParser : IOutputProtocolParser
    {
        /// <summary>Prime parser state when the prompt already opened a channel.
        /// Call before the first generated piece, after Init.</summary>
        void SetGenerationPromptSuffix(string? suffix) { }
    }

    // ========================================================================
    // ChatML parser: <think>...</think> for thinking, <tool_call>...</tool_call>
    // ========================================================================

    public class ChatMlOutputParser : IOutputParser
    {
        private enum State { CollectingThinking, ThinkingDone, CollectingContent, CollectingTool }

        private State _state;
        private readonly StringBuilder _buffer = new();
        private bool _stripLeadingThinkTag;
        private int _callIndex;

        // Watermark into the in-progress tool-call body already surfaced as
        // ParsedOutput.ToolCallText. The buffer keeps the whole body for the completion
        // parse; this only tracks what a streaming consumer has been shown. Same
        // mechanism as Gemma4OutputParser — a 27B model writes a run_code program for
        // many minutes, and without this the UI has nothing to show for any of it.
        private int _toolReportedChars;

        public bool HasThinkingSupport => true;
        public bool HasToolSupport => true;
        public bool AlwaysRequired => false;

        public void Init(bool enableThinking, List<ToolFunction>? tools)
        {
            _buffer.Clear();
            _callIndex = 0;
            _toolReportedChars = 0;
            if (enableThinking)
            {
                _state = State.CollectingThinking;
                _stripLeadingThinkTag = true;
            }
            else
            {
                _state = State.CollectingContent;
                _stripLeadingThinkTag = false;
            }
        }

        public ParsedOutput Add(string text, bool done)
        {
            _buffer.Append(text);
            var result = new ParsedOutput();
            var thinkingSb = new StringBuilder();
            var contentSb = new StringBuilder();
            var toolCallTextSb = new StringBuilder();
            var toolCalls = new List<ToolCall>();

            bool keepParsing = true;
            while (keepParsing)
            {
                keepParsing = false;
                string buf = _buffer.ToString();

                switch (_state)
                {
                    case State.CollectingThinking:
                        if (_stripLeadingThinkTag)
                        {
                            string trimmed = buf.TrimStart();
                            if (trimmed.StartsWith("<think>"))
                            {
                                buf = trimmed.Substring(7).TrimStart();
                                _buffer.Clear();
                                _buffer.Append(buf);
                                _stripLeadingThinkTag = false;
                                keepParsing = buf.Length > 0;
                                break;
                            }
                            if ("<think>".StartsWith(trimmed) && !done)
                                break;
                            _stripLeadingThinkTag = false;
                        }

                        int closeIdx = buf.IndexOf("</think>", StringComparison.Ordinal);
                        int toolIdx = buf.IndexOf("<tool_call>", StringComparison.Ordinal);

                        if (toolIdx >= 0 && (closeIdx < 0 || toolIdx < closeIdx))
                        {
                            string before = buf.Substring(0, toolIdx).TrimEnd();
                            string after = buf.Substring(toolIdx + 11).TrimStart();
                            _buffer.Clear();
                            _buffer.Append(after);
                            if (before.Length > 0) thinkingSb.Append(before);
                            _state = State.CollectingTool;
                            keepParsing = true;
                        }
                        else if (closeIdx >= 0)
                        {
                            string thinking = buf.Substring(0, closeIdx).TrimEnd();
                            string after = buf.Substring(closeIdx + 8).TrimStart();
                            _buffer.Clear();
                            _buffer.Append(after);
                            if (thinking.Length > 0) thinkingSb.Append(thinking);
                            _state = after.Length > 0 ? State.CollectingContent : State.ThinkingDone;
                            keepParsing = after.Length > 0;
                        }
                        else if (done)
                        {
                            if (buf.Length > 0) thinkingSb.Append(buf);
                            _buffer.Clear();
                        }
                        else
                        {
                            int hold = HoldBackForPartialTag(buf, "</think>", "<tool_call>");
                            if (hold > 0)
                            {
                                string emit = buf.Substring(0, buf.Length - hold);
                                if (emit.Length > 0) thinkingSb.Append(emit);
                                _buffer.Clear();
                                _buffer.Append(buf.Substring(buf.Length - hold));
                            }
                            else
                            {
                                thinkingSb.Append(buf);
                                _buffer.Clear();
                            }
                        }
                        break;

                    case State.ThinkingDone:
                        string td = buf.TrimStart();
                        _buffer.Clear();
                        if (td.Length > 0)
                        {
                            _buffer.Append(td);
                            _state = State.CollectingContent;
                            keepParsing = true;
                        }
                        break;

                    case State.CollectingContent:
                        int tcIdx = buf.IndexOf("<tool_call>", StringComparison.Ordinal);
                        if (tcIdx >= 0)
                        {
                            string before = buf.Substring(0, tcIdx).TrimEnd();
                            string after = buf.Substring(tcIdx + 11).TrimStart();
                            _buffer.Clear();
                            _buffer.Append(after);
                            if (before.Length > 0) contentSb.Append(before);
                            _state = State.CollectingTool;
                            keepParsing = true;
                        }
                        else if (done)
                        {
                            if (buf.Length > 0) contentSb.Append(buf);
                            _buffer.Clear();
                        }
                        else
                        {
                            int hold = HoldBackForPartialTag(buf, "<tool_call>");
                            if (hold > 0)
                            {
                                string emit = buf.Substring(0, buf.Length - hold);
                                if (emit.Length > 0) contentSb.Append(emit);
                                _buffer.Clear();
                                _buffer.Append(buf.Substring(buf.Length - hold));
                            }
                            else
                            {
                                contentSb.Append(buf);
                                _buffer.Clear();
                            }
                        }
                        break;

                    case State.CollectingTool:
                        int endIdx = buf.IndexOf("</tool_call>", StringComparison.Ordinal);
                        if (endIdx >= 0)
                        {
                            string raw = buf.Substring(0, endIdx);
                            string after = buf.Substring(endIdx + 12).TrimStart();
                            _buffer.Clear();
                            _buffer.Append(after);
                            // The body's unreported tail rides out with the completion,
                            // so a consumer rendering the draft ends with all of it.
                            if (endIdx > _toolReportedChars)
                                toolCallTextSb.Append(raw, _toolReportedChars, endIdx - _toolReportedChars);
                            _toolReportedChars = 0;
                            var tc = ParseToolCall(raw);
                            if (tc != null) toolCalls.Add(tc);
                            _state = State.CollectingContent;
                            keepParsing = after.Length > 0;
                        }
                        else if (done && buf.Length > 0)
                        {
                            if (buf.Length > _toolReportedChars)
                                toolCallTextSb.Append(buf, _toolReportedChars, buf.Length - _toolReportedChars);
                            _toolReportedChars = 0;
                            var tc = ParseToolCall(buf);
                            if (tc != null) toolCalls.Add(tc);
                            _buffer.Clear();
                            _state = State.CollectingContent;
                        }
                        else if (!done)
                        {
                            // Mid-call: surface the newly written body as progress,
                            // holding back a possible partial closing tag. Both body
                            // shapes stream the same way — the JSON object and the
                            // XML-ish <function=...> fallback are just text here.
                            int reportable = buf.Length - HoldBackForPartialTag(buf, "</tool_call>");
                            if (reportable > _toolReportedChars)
                            {
                                toolCallTextSb.Append(buf, _toolReportedChars, reportable - _toolReportedChars);
                                _toolReportedChars = reportable;
                            }
                            result.ToolCallName ??= ToolCallNameFrom(buf);
                        }
                        break;
                }
            }

            result.Content = contentSb.ToString();
            result.Thinking = thinkingSb.ToString();
            result.ToolCalls = toolCalls.Count > 0 ? toolCalls : null;
            result.ToolCallText = toolCallTextSb.ToString();
            return result;
        }

        private static readonly Regex JsonToolNameRe = new(
            "^\\s*\\{\\s*\"name\"\\s*:\\s*\"([^\"]+)\"",
            RegexOptions.Compiled | RegexOptions.CultureInvariant);

        /// <summary>The in-progress call's tool name, from whichever body shape has
        /// gotten far enough to carry it.</summary>
        private static string? ToolCallNameFrom(string body)
        {
            Match m = JsonToolNameRe.Match(body);
            if (m.Success)
                return m.Groups[1].Value;

            int fn = body.IndexOf("<function=", StringComparison.Ordinal);
            if (fn >= 0)
            {
                int close = body.IndexOf('>', fn + 10);
                if (close > fn + 10)
                {
                    string name = body.Substring(fn + 10, close - fn - 10).Trim();
                    if (name.Length > 0)
                        return name;
                }
            }
            return null;
        }

        private ToolCall? ParseToolCall(string raw)
        {
            raw = raw.Trim();
            if (raw.Length == 0) return null;
            try
            {
                using var doc = JsonDocument.Parse(raw);
                var root = doc.RootElement;
                string? name = root.GetProperty("name").GetString();
                if (string.IsNullOrEmpty(name)) return null;

                var args = new Dictionary<string, object>();
                if (root.TryGetProperty("arguments", out var argsEl) && argsEl.ValueKind == JsonValueKind.Object)
                {
                    foreach (var prop in argsEl.EnumerateObject())
                        args[prop.Name] = JsonElementToObject(prop.Value);
                }
                return new ToolCall { Name = name, Arguments = args, Index = _callIndex++ };
            }
            catch (JsonException)
            {
                // Qwen 3.5 emits the XML-ish call body instead of a JSON object:
                //   <function=get_weather>
                //   <parameter=city>\nParis\n</parameter>
                //   </function>
                // Dropping it silently loses the whole turn (the text was already
                // consumed as a tool call), so fall back to that form here.
                return ParseXmlToolCall(raw);
            }
        }

        /// <summary>
        /// Parse the `&lt;function=NAME&gt;&lt;parameter=KEY&gt;VALUE&lt;/parameter&gt;&lt;/function&gt;`
        /// tool-call body. Each parameter value is trimmed of the surrounding
        /// newlines the template emits, and parsed as JSON when it is a scalar /
        /// object / array so numbers and booleans do not arrive quoted.
        /// </summary>
        private ToolCall? ParseXmlToolCall(string raw)
        {
            const string fnOpen = "<function=";
            int fnIdx = raw.IndexOf(fnOpen, StringComparison.Ordinal);
            if (fnIdx < 0) return null;
            int nameEnd = raw.IndexOf('>', fnIdx + fnOpen.Length);
            if (nameEnd < 0) return null;

            string name = raw.Substring(fnIdx + fnOpen.Length, nameEnd - fnIdx - fnOpen.Length).Trim();
            if (name.Length == 0) return null;

            var args = new Dictionary<string, object>();
            const string paramOpen = "<parameter=";
            const string paramClose = "</parameter>";
            int pos = nameEnd + 1;
            while (true)
            {
                int pIdx = raw.IndexOf(paramOpen, pos, StringComparison.Ordinal);
                if (pIdx < 0) break;
                int keyEnd = raw.IndexOf('>', pIdx + paramOpen.Length);
                if (keyEnd < 0) break;
                string key = raw.Substring(pIdx + paramOpen.Length, keyEnd - pIdx - paramOpen.Length).Trim();

                int valEnd = raw.IndexOf(paramClose, keyEnd + 1, StringComparison.Ordinal);
                string value = valEnd < 0
                    ? raw.Substring(keyEnd + 1)
                    : raw.Substring(keyEnd + 1, valEnd - keyEnd - 1);
                if (key.Length > 0)
                    args[key] = ParseScalarOrText(value.Trim());

                if (valEnd < 0) break;
                pos = valEnd + paramClose.Length;
            }

            return new ToolCall { Name = name, Arguments = args, Index = _callIndex++ };
        }

        private static object ParseScalarOrText(string value)
        {
            if (value.Length == 0) return value;
            char c = value[0];
            bool looksJson = c == '{' || c == '[' || c == '-' || char.IsDigit(c) ||
                             value == "true" || value == "false" || value == "null";
            if (looksJson)
            {
                try
                {
                    using var doc = JsonDocument.Parse(value);
                    return JsonElementToObject(doc.RootElement);
                }
                catch (JsonException)
                {
                    // Not JSON after all (e.g. a date like 2026-08-01): keep the text.
                }
            }
            return value;
        }

        private static int HoldBackForPartialTag(string buf, params string[] tags)
        {
            int maxOverlap = 0;
            foreach (var tag in tags)
            {
                int max = Math.Min(tag.Length, buf.Length);
                for (int i = max; i > 0; i--)
                {
                    if (buf.EndsWith(tag.Substring(0, i), StringComparison.Ordinal))
                    {
                        maxOverlap = Math.Max(maxOverlap, i);
                        break;
                    }
                }
            }
            return maxOverlap;
        }

        internal static object JsonElementToObject(JsonElement el)
        {
            return el.ValueKind switch
            {
                JsonValueKind.String => el.GetString() ?? string.Empty,
                JsonValueKind.Number => el.TryGetInt64(out long l) ? (object)l : el.GetDouble(),
                JsonValueKind.True => true,
                JsonValueKind.False => false,
                JsonValueKind.Null => null!,
                JsonValueKind.Object => JsonElementToDict(el),
                JsonValueKind.Array => JsonElementToList(el),
                _ => el.GetRawText()
            };
        }

        private static Dictionary<string, object> JsonElementToDict(JsonElement el)
        {
            var d = new Dictionary<string, object>();
            foreach (var p in el.EnumerateObject())
                d[p.Name] = JsonElementToObject(p.Value);
            return d;
        }

        private static List<object> JsonElementToList(JsonElement el)
        {
            var list = new List<object>();
            foreach (var item in el.EnumerateArray())
                list.Add(JsonElementToObject(item));
            return list;
        }
    }

    // ========================================================================
    // Qwen3.5 parser: ChatML thinking and tool-call tags.
    // ========================================================================

    public class Qwen35OutputParser : ChatMlOutputParser
    {
    }

    // ========================================================================
    // Qwen2 / Qwen2.5(-VL) Parser: ChatML with <tool_call> JSON, NO thinking
    // channel. Wraps the ChatML parser with thinking pinned off: initialized
    // with enableThinking=true it would misread the whole answer as thought
    // while waiting for a </think> these models never emit.
    // ========================================================================

    public class Qwen25OutputParser : IOutputParser
    {
        private readonly ChatMlOutputParser _inner = new();

        public bool HasThinkingSupport => false;
        public bool HasToolSupport => true;
        public bool AlwaysRequired => false;

        public void Init(bool enableThinking, List<ToolFunction>? tools) => _inner.Init(false, tools);

        public ParsedOutput Add(string text, bool done) => _inner.Add(text, done);
    }

    // ========================================================================
    // Gemma4 Parser: <|channel>thought\n...<channel|> for thinking,
    //                <|tool_call>call:NAME{args}<tool_call|> for tool calls
    // ========================================================================

    public class Gemma4OutputParser : IOutputParser
    {
        private enum State { CollectingContent, CollectingThinking, CollectingToolCall }

        private State _state;
        private readonly StringBuilder _buffer = new();
        private bool _thinkingEnabled;
        private bool _needsChannelNameStrip;

        // How much of the in-progress tool-call body has been surfaced as
        // ParsedOutput.ToolCallText. The buffer itself is kept whole until the call
        // closes (ParseGemma4ToolCall needs all of it), so progress is a watermark
        // into it rather than a consumption.
        private int _toolReportedChars;

        // Position of the next completed call in this turn. Every other parser
        // numbers its calls so a streaming client can pair the argument deltas of
        // two parallel calls; Gemma 4's were all index 0, and a two-call turn
        // collapsed into one call on the wire.
        private int _callIndex;

        public bool HasThinkingSupport => true;
        public bool HasToolSupport => true;
        public bool AlwaysRequired => true;

        public void SetGenerationPromptSuffix(string? suffix)
        {
            if (suffix?.EndsWith("<|channel>thought\n", StringComparison.Ordinal) == true)
            {
                _state = State.CollectingThinking;
                _needsChannelNameStrip = false;
            }
        }

        public void Init(bool enableThinking, List<ToolFunction>? tools)
        {
            _buffer.Clear();
            _thinkingEnabled = enableThinking;
            _needsChannelNameStrip = false;
            _state = State.CollectingContent;
            _toolReportedChars = 0;
            _callIndex = 0;
        }

        public ParsedOutput Add(string text, bool done)
        {
            _buffer.Append(text);
            var result = new ParsedOutput();
            var thinkingSb = new StringBuilder();
            var contentSb = new StringBuilder();
            var toolCallTextSb = new StringBuilder();
            var toolCalls = new List<ToolCall>();

            bool keepParsing = true;
            while (keepParsing)
            {
                keepParsing = false;
                string buf = _buffer.ToString();
                if (buf.Length == 0) break;

                switch (_state)
                {
                    case State.CollectingContent:
                        int chIdx = buf.IndexOf("<|channel>", StringComparison.Ordinal);
                        int tcIdx = buf.IndexOf("<|tool_call>", StringComparison.Ordinal);
                        // A CLOSING <channel|> with no opener in the generated text:
                        // the block was opened by the prompt's channel primer, so the
                        // model is closing a thinking block we never saw start. Gemma 4
                        // does this even with thinking disabled (the primer is a
                        // complete empty block, but smaller checkpoints still reason
                        // first), and treating it as content is what surfaced the whole
                        // chain of thought — plus the raw marker — as the answer.
                        int strayCloseIdx = buf.IndexOf("<channel|>", StringComparison.Ordinal);

                        if (strayCloseIdx >= 0
                            && (chIdx < 0 || strayCloseIdx < chIdx)
                            && (tcIdx < 0 || strayCloseIdx < tcIdx))
                        {
                            string thought = buf.Substring(0, strayCloseIdx);
                            // The model may re-emit the channel name it is closing.
                            if (thought.StartsWith("thought\n", StringComparison.Ordinal))
                                thought = thought.Substring(8);
                            string after = buf.Substring(strayCloseIdx + 10).TrimStart();
                            _buffer.Clear();
                            _buffer.Append(after);
                            // Only text still buffered can be reclassified: a streaming
                            // consumer already received whatever was flushed before the
                            // marker arrived, and there is no bounded lookahead that
                            // would let us hold an arbitrarily long thought block back.
                            // Batch callers (Add(full, done: true)) buffer the whole
                            // output, so they get the split exactly right.
                            thought = thought.Trim();
                            if (thought.Length > 0 && _thinkingEnabled) thinkingSb.Append(thought);
                            keepParsing = after.Length > 0;
                        }
                        else if (chIdx >= 0 && (tcIdx < 0 || chIdx < tcIdx))
                        {
                            string before = buf.Substring(0, chIdx).TrimEnd();
                            string after = buf.Substring(chIdx + 10);
                            _buffer.Clear();
                            _buffer.Append(after);
                            if (before.Length > 0) contentSb.Append(before);
                            _state = State.CollectingThinking;
                            _needsChannelNameStrip = true;
                            keepParsing = true;
                        }
                        else if (tcIdx >= 0)
                        {
                            string before = buf.Substring(0, tcIdx).TrimEnd();
                            string after = buf.Substring(tcIdx + 12);
                            _buffer.Clear();
                            _buffer.Append(after);
                            if (before.Length > 0) contentSb.Append(before);
                            _state = State.CollectingToolCall;
                            keepParsing = true;
                        }
                        else if (!done)
                        {
                            int hold = HoldBack(buf, "<|channel>", "<|tool_call>", "<channel|>");
                            if (hold > 0)
                            {
                                string emit = buf.Substring(0, buf.Length - hold);
                                if (emit.Length > 0) contentSb.Append(emit);
                                _buffer.Clear();
                                _buffer.Append(buf.Substring(buf.Length - hold));
                            }
                            else
                            {
                                contentSb.Append(buf);
                                _buffer.Clear();
                            }
                        }
                        else
                        {
                            if (buf.Length > 0) contentSb.Append(buf);
                            _buffer.Clear();
                        }
                        break;

                    case State.CollectingThinking:
                        if (_needsChannelNameStrip)
                        {
                            if (buf.StartsWith("thought\n"))
                            {
                                buf = buf.Substring(8);
                                _buffer.Clear();
                                _buffer.Append(buf);
                                _needsChannelNameStrip = false;
                                keepParsing = buf.Length > 0;
                                break;
                            }
                            if (!done && ("thought\n".StartsWith(buf) || buf.StartsWith("thought")))
                                break;
                            _needsChannelNameStrip = false;
                        }

                        int closeIdx = buf.IndexOf("<channel|>", StringComparison.Ordinal);
                        if (closeIdx >= 0)
                        {
                            string thinking = buf.Substring(0, closeIdx).TrimEnd();
                            string after = buf.Substring(closeIdx + 10).TrimStart();
                            _buffer.Clear();
                            _buffer.Append(after);
                            if (thinking.Length > 0 && _thinkingEnabled) thinkingSb.Append(thinking);
                            _state = State.CollectingContent;
                            keepParsing = after.Length > 0;
                        }
                        else if (!done)
                        {
                            int hold = HoldBack(buf, "<channel|>");
                            if (hold > 0)
                            {
                                string emit = buf.Substring(0, buf.Length - hold);
                                if (emit.Length > 0 && _thinkingEnabled) thinkingSb.Append(emit);
                                _buffer.Clear();
                                _buffer.Append(buf.Substring(buf.Length - hold));
                            }
                            else
                            {
                                if (_thinkingEnabled) thinkingSb.Append(buf);
                                _buffer.Clear();
                            }
                        }
                        else
                        {
                            if (buf.Length > 0 && _thinkingEnabled) thinkingSb.Append(buf);
                            _buffer.Clear();
                        }
                        break;

                    case State.CollectingToolCall:
                        int endIdx = buf.IndexOf("<tool_call|>", StringComparison.Ordinal);
                        if (endIdx >= 0)
                        {
                            string raw = buf.Substring(0, endIdx);
                            string after = buf.Substring(endIdx + 12).TrimStart();
                            _buffer.Clear();
                            _buffer.Append(after);
                            // Whatever of the body was not yet surfaced as progress goes
                            // out with the completion, so a consumer that renders the
                            // draft ends with the complete text.
                            if (endIdx > _toolReportedChars)
                                toolCallTextSb.Append(raw, _toolReportedChars, endIdx - _toolReportedChars);
                            _toolReportedChars = 0;
                            AcceptGemma4ToolCall(raw, toolCalls, contentSb);
                            _state = State.CollectingContent;
                            keepParsing = after.Length > 0;
                        }
                        else if (done && buf.Length > 0)
                        {
                            if (buf.Length > _toolReportedChars)
                                toolCallTextSb.Append(buf, _toolReportedChars, buf.Length - _toolReportedChars);
                            _toolReportedChars = 0;
                            AcceptGemma4ToolCall(buf, toolCalls, contentSb);
                            _buffer.Clear();
                            _state = State.CollectingContent;
                        }
                        else
                        {
                            // Mid-call: surface the newly written body text as progress,
                            // holding back what could be the start of the closing tag.
                            // The model spends its longest silent stretches here — a
                            // run_code call carries a whole program — and this is what
                            // lets a UI show the code being written instead of nothing.
                            int reportable = buf.Length - HoldBack(buf, "<tool_call|>");
                            if (reportable > _toolReportedChars)
                            {
                                toolCallTextSb.Append(buf, _toolReportedChars, reportable - _toolReportedChars);
                                _toolReportedChars = reportable;
                            }
                            result.ToolCallName ??= ToolCallNameFrom(buf);
                        }
                        break;
                }
            }

            result.Content = contentSb.ToString();
            result.Thinking = thinkingSb.ToString();
            result.ToolCalls = toolCalls.Count > 0 ? toolCalls : null;
            result.ToolCallText = toolCallTextSb.ToString();
            return result;
        }

        /// <summary>
        /// A completed call body either becomes a structured <see cref="ToolCall"/> or,
        /// when its arguments cannot be read, is surfaced verbatim as CONTENT. Dropping
        /// it produced an empty assistant message with <c>finish_reason=stop</c> and no
        /// tool call: the client saw nothing at all and could not tell a refusal from a
        /// parse failure. The raw text at least shows what the model wrote.
        /// </summary>
        private void AcceptGemma4ToolCall(string raw, List<ToolCall> toolCalls, StringBuilder contentSb)
        {
            var tc = ParseGemma4ToolCall(raw);
            if (tc != null)
            {
                tc.Index = _callIndex++;
                toolCalls.Add(tc);
            }
            else
            {
                contentSb.Append(raw);
            }
        }

        /// <summary>The call's tool name, once <c>call:NAME{</c> has been written.</summary>
        private static string? ToolCallNameFrom(string body)
        {
            if (!body.StartsWith("call:", StringComparison.Ordinal))
                return null;
            int brace = body.IndexOf('{');
            if (brace <= 5)
                return null;
            string name = body.Substring(5, brace - 5).Trim();
            return name.Length > 0 ? name : null;
        }

        private static readonly Regex GemmaQuotedStringRe = new(@"<\|""\|>(.*?)<\|""\|>", RegexOptions.Singleline);
        private static readonly Regex GemmaBareKeyRe = new(@"([,{]\s*)(\w+)\s*:");
        // A JSON number exactly as RFC 8259 spells it; any other bare value is a string.
        private static readonly Regex JsonNumberRe = new(@"^-?(0|[1-9]\d*)(\.\d+)?([eE][+-]?\d+)?$");

        // Tool names whose call bodies already failed to parse. The raw body
        // still reaches the client via ToolCallText, but the dropped structured
        // call must not fail silently — nor warn per retry. This class has no
        // logger, so Console.Error is the channel.
        private static readonly HashSet<string> GemmaParseFailReported = new(StringComparer.Ordinal);

        private static ToolCall? ParseGemma4ToolCall(string content)
        {
            content = content.Trim();
            if (!content.StartsWith("call:")) return null;
            content = content.Substring(5);

            int braceIdx = content.IndexOf('{');
            if (braceIdx < 0) return null;

            string name = content.Substring(0, braceIdx).Trim();
            string argsStr = content.Substring(braceIdx);

            string json = Gemma4ArgsToJson(argsStr);
            try
            {
                using var doc = JsonDocument.Parse(json);
                var args = new Dictionary<string, object>();
                foreach (var prop in doc.RootElement.EnumerateObject())
                    args[prop.Name] = ChatMlOutputParser.JsonElementToObject(prop.Value);
                return new ToolCall { Name = name, Arguments = args };
            }
            catch (Exception ex)
            {
                bool first;
                lock (GemmaParseFailReported) first = GemmaParseFailReported.Add(name);
                if (first)
                    Console.Error.WriteLine(
                        $"[Gemma4OutputParser] Tool call '{name}' has arguments that do not parse as JSON ({ex.Message}); " +
                        "it is dropped from ToolCalls, so the tool will not run — the raw call text is " +
                        "delivered as content and in ToolCallText instead. Reported once per tool name.");
                return null;
            }
        }

        /// <summary>
        /// Turn Gemma 4's call syntax into JSON: keys are bare identifiers, strings are
        /// wrapped in <c>&lt;|"|&gt;</c> ... <c>&lt;|"|&gt;</c>, and - what this used to
        /// miss - the model regularly writes a string value BARE when it looks like an
        /// identifier (<c>{invoice_id:INV-472}</c>, <c>{path:src/main.py}</c>,
        /// <c>{ids:[INV-1, INV-2]}</c>). Such a call parsed only by luck, and
        /// <c>read_invoice{invoice_id:INV-472}</c> was dropped whole. A bare value that
        /// is not a JSON number, <c>true</c>, <c>false</c> or <c>null</c> is quoted,
        /// inside arrays included; numbers stay numbers.
        /// </summary>
        internal static string Gemma4ArgsToJson(string s)
        {
            var quotedStrings = new List<string>();
            string text = GemmaQuotedStringRe.Replace(s, m =>
            {
                quotedStrings.Add(m.Groups[1].Value);
                return "\x00" + (char)(quotedStrings.Count - 1) + "\x00";
            });

            text = GemmaBareKeyRe.Replace(text, "$1\"$2\":");
            text = QuoteGemmaBareValues(text);

            for (int i = 0; i < quotedStrings.Count; i++)
            {
                string escaped = JsonSerializer.Serialize(quotedStrings[i]);
                text = text.Replace("\x00" + (char)i + "\x00", escaped);
            }

            return text;
        }

        /// <summary>
        /// Quote every bare scalar in VALUE position (after a <c>:</c> inside an object,
        /// after <c>[</c> or <c>,</c> inside an array) that is not already a JSON scalar.
        /// Placeholders (<c>\x00</c> i <c>\x00</c>, the strings the model quoted itself)
        /// and real JSON strings pass through untouched.
        /// </summary>
        private static string QuoteGemmaBareValues(string text)
        {
            var sb = new StringBuilder(text.Length + 16);
            var containers = new Stack<char>();
            bool expectValue = false;
            int i = 0;
            while (i < text.Length)
            {
                char c = text[i];
                if (expectValue)
                {
                    if (char.IsWhiteSpace(c)) { sb.Append(c); i++; continue; }
                    expectValue = false;
                    if (c == '{' || c == '[')
                    {
                        containers.Push(c);
                        sb.Append(c);
                        i++;
                        expectValue = c == '[';
                        continue;
                    }
                    if (c == '\x00' || c == '"')
                    {
                        i = CopyOpaque(text, i, sb);
                        continue;
                    }
                    // A bare token runs to the next delimiter at this level.
                    int start = i;
                    while (i < text.Length && text[i] != ',' && text[i] != '}' && text[i] != ']')
                        i++;
                    string run = text.Substring(start, i - start);
                    string token = run.TrimEnd();
                    if (token.Length > 0)
                    {
                        bool scalar = token == "true" || token == "false" || token == "null"
                                      || JsonNumberRe.IsMatch(token);
                        sb.Append(scalar ? token : JsonSerializer.Serialize(token));
                    }
                    sb.Append(run, token.Length, run.Length - token.Length);
                    continue;
                }

                switch (c)
                {
                    case '{':
                    case '[':
                        containers.Push(c);
                        expectValue = c == '[';
                        break;
                    case '}':
                    case ']':
                        if (containers.Count > 0) containers.Pop();
                        break;
                    case ':':
                        expectValue = containers.Count > 0 && containers.Peek() == '{';
                        break;
                    case ',':
                        expectValue = containers.Count > 0 && containers.Peek() == '[';
                        break;
                    case '\x00':
                    case '"':
                        i = CopyOpaque(text, i, sb);
                        continue;
                }
                sb.Append(c);
                i++;
            }
            return sb.ToString();
        }

        /// <summary>Copy the placeholder or JSON string that starts at <paramref name="i"/>
        /// and return the index just past it.</summary>
        private static int CopyOpaque(string text, int i, StringBuilder sb)
        {
            if (text[i] == '\x00')
            {
                // Always exactly three chars: NUL, the string's index as a char, NUL.
                int len = Math.Min(3, text.Length - i);
                sb.Append(text, i, len);
                return i + len;
            }
            int j = i + 1;
            while (j < text.Length && text[j] != '"')
            {
                if (text[j] == '\\' && j + 1 < text.Length) j++;
                j++;
            }
            j = Math.Min(j + 1, text.Length);
            sb.Append(text, i, j - i);
            return j;
        }

        private static int HoldBack(string buf, params string[] tags)
        {
            int maxOverlap = 0;
            foreach (var tag in tags)
            {
                int max = Math.Min(tag.Length, buf.Length);
                for (int i = max; i > 0; i--)
                {
                    if (buf.EndsWith(tag.Substring(0, i), StringComparison.Ordinal))
                    {
                        maxOverlap = Math.Max(maxOverlap, i);
                        break;
                    }
                }
            }
            return maxOverlap;
        }
    }

    // ========================================================================
    // GPT OSS / Harmony Parser
    // Uses <|start|>...<|end|> message framing with <|message|> header end,
    // <|channel|>analysis for thinking, <|channel|>final for content
    // ========================================================================

    public class HarmonyOutputParser : IOutputParser
    {
        private enum HState { LookingForStart, ParsingHeader, ParsingContent }

        private HState _state;
        private readonly StringBuilder _buffer = new();
        private readonly StringBuilder _toolArgs = new();
        private string? _currentChannel;
        private string? _currentRecipient;
        private int _callIndex;

        private const string MsgStartTag = "<|start|>";
        private const string MsgEndTag = "<|end|>";
        private const string CallTag = "<|call|>";
        private const string ReturnTag = "<|return|>";
        private const string HeaderEndTag = "<|message|>";
        private const string ChannelTag = "<|channel|>";
        private const string FunctionPrefix = "functions.";

        // Tags that terminate a content message during generation.
        private static readonly string[] EndTags = { MsgEndTag, CallTag, ReturnTag };
        // Tags whose partial suffixes must be held back while streaming content.
        private static readonly string[] HoldTags = { MsgEndTag, CallTag, ReturnTag, MsgStartTag };

        /// <summary>Safety valve: a "header" this long means the stream never closed one.</summary>
        private const int MaxHeaderChars = 512;

        public bool HasThinkingSupport => true;
        public bool HasToolSupport => true;
        public bool AlwaysRequired => true;

        public void Init(bool enableThinking, List<ToolFunction>? tools)
        {
            _buffer.Clear();
            _toolArgs.Clear();
            _state = HState.LookingForStart;
            _currentChannel = null;
            _currentRecipient = null;
            _callIndex = 0;

            // The prompt's generation marker is "<|start|>assistant", so the
            // model's first emitted token is "<|channel|>". Prime the buffer so
            // the parser is already past the start tag.
            _buffer.Append("<|start|>assistant");
        }

        public ParsedOutput Add(string text, bool done)
        {
            _buffer.Append(text);
            var result = new ParsedOutput();
            var contentSb = new StringBuilder();
            var thinkingSb = new StringBuilder();
            var toolTextSb = new StringBuilder();
            var toolCalls = new List<ToolCall>();

            bool keepParsing = true;
            while (keepParsing)
            {
                keepParsing = false;
                string buf = _buffer.ToString();
                if (buf.Length == 0)
                {
                    // A generation that stops at EOS emits no closing
                    // <|end|>/<|call|>/<|return|> tag, and its last content chunk
                    // may already have been drained into `_toolArgs`. Finalizing
                    // here is what keeps that trailing message — in particular a
                    // commentary tool call, the whole answer for a function-call
                    // turn — from being dropped on the floor.
                    if (done && _state == HState.ParsingContent)
                    {
                        FinalizeMessage(toolCalls);
                        _state = HState.LookingForStart;
                    }
                    break;
                }

                switch (_state)
                {
                    case HState.LookingForStart:
                        int startIdx = buf.IndexOf(MsgStartTag, StringComparison.Ordinal);
                        if (startIdx >= 0)
                        {
                            string after = buf.Substring(startIdx + MsgStartTag.Length);
                            _buffer.Clear();
                            _buffer.Append(after);
                            _state = HState.ParsingHeader;
                            keepParsing = true;
                        }
                        else if (!done)
                        {
                            int hold = HoldBack(buf, MsgStartTag);
                            if (hold > 0)
                            {
                                _buffer.Clear();
                                _buffer.Append(buf.Substring(buf.Length - hold));
                            }
                        }
                        break;

                    case HState.ParsingHeader:
                        int headerEnd = buf.IndexOf(HeaderEndTag, StringComparison.Ordinal);
                        if (headerEnd >= 0)
                        {
                            string header = buf.Substring(0, headerEnd);
                            string after = buf.Substring(headerEnd + HeaderEndTag.Length);
                            _buffer.Clear();
                            _buffer.Append(after);

                            ParseHeader(header);

                            _state = HState.ParsingContent;
                            keepParsing = after.Length > 0;
                        }
                        else if (!done)
                        {
                            // Keep the WHOLE header buffered until <|message|>
                            // arrives. The previous holdback trimmed the buffer to
                            // the partial-tag suffix, which discarded
                            // "<|channel|>analysis" (and any "to=functions.NAME")
                            // whenever a chunk boundary fell inside <|message|> —
                            // the header then parsed as channel "final" and the
                            // model's chain of thought was streamed to the user as
                            // the answer. Headers are short; the cap only guards a
                            // stream that never closes one.
                            if (buf.Length > MaxHeaderChars)
                            {
                                EmitContent(buf, contentSb, thinkingSb, toolTextSb);
                                _buffer.Clear();
                                _state = HState.ParsingContent;
                            }
                        }
                        break;

                    case HState.ParsingContent:
                        int endIdx = FindEarliestEndTag(buf, out int tagLen);
                        if (endIdx >= 0)
                        {
                            string content = buf.Substring(0, endIdx);
                            string after = buf.Substring(endIdx + tagLen);
                            _buffer.Clear();
                            _buffer.Append(after);

                            EmitContent(content, contentSb, thinkingSb, toolTextSb);
                            FinalizeMessage(toolCalls);
                            _state = HState.LookingForStart;
                            keepParsing = after.Length > 0;
                        }
                        else if (!done)
                        {
                            int hold = HoldBack(buf, HoldTags);
                            if (hold > 0)
                            {
                                string emit = buf.Substring(0, buf.Length - hold);
                                if (emit.Length > 0) EmitContent(emit, contentSb, thinkingSb, toolTextSb);
                                _buffer.Clear();
                                _buffer.Append(buf.Substring(buf.Length - hold));
                            }
                            else
                            {
                                EmitContent(buf, contentSb, thinkingSb, toolTextSb);
                                _buffer.Clear();
                            }
                        }
                        else
                        {
                            if (buf.Length > 0) EmitContent(buf, contentSb, thinkingSb, toolTextSb);
                            FinalizeMessage(toolCalls);
                            _buffer.Clear();
                            _state = HState.LookingForStart;
                        }
                        break;
                }
            }

            result.Content = contentSb.ToString();
            result.Thinking = thinkingSb.ToString();
            result.ToolCallText = toolTextSb.ToString();
            if (IsToolCall())
                result.ToolCallName = _currentRecipient!.Substring(FunctionPrefix.Length);
            if (toolCalls.Count > 0)
                result.ToolCalls = toolCalls;
            return result;
        }

        /// <summary>
        /// Parse a message header (the text between &lt;|start|&gt; and &lt;|message|&gt;)
        /// to extract the channel and, for tool calls, the "to=functions.NAME" recipient.
        /// Handles both header orderings (recipient before or after the channel tag).
        /// </summary>
        private void ParseHeader(string header)
        {
            int chIdx = header.IndexOf(ChannelTag, StringComparison.Ordinal);
            if (chIdx >= 0)
            {
                string channelPart = header.Substring(chIdx + ChannelTag.Length);
                int spaceIdx = channelPart.IndexOfAny(new[] { ' ', '\t', '\n', '\r' });
                _currentChannel = spaceIdx >= 0 ? channelPart.Substring(0, spaceIdx) : channelPart;
            }
            else
            {
                _currentChannel = "final";
            }

            _currentRecipient = null;
            int toIdx = header.IndexOf("to=", StringComparison.Ordinal);
            if (toIdx >= 0)
            {
                string rest = header.Substring(toIdx + 3);
                int end = 0;
                while (end < rest.Length && !char.IsWhiteSpace(rest[end]) && rest[end] != '<')
                    end++;
                if (end > 0)
                    _currentRecipient = rest.Substring(0, end);
            }
        }

        private void EmitContent(
            string content, StringBuilder contentSb, StringBuilder thinkingSb, StringBuilder toolTextSb)
        {
            if (content.Length == 0) return;
            if (IsToolCall())
            {
                // The call's body, surfaced live: Harmony already drains it
                // incrementally through the holdback, so the progress signal is
                // simply a copy of what lands in _toolArgs.
                _toolArgs.Append(content);
                toolTextSb.Append(content);
            }
            else if (_currentChannel == "analysis")
                thinkingSb.Append(content);
            else
                contentSb.Append(content);
        }

        /// <summary>Finalize the current message: emit a tool call if it targeted functions.*.</summary>
        private void FinalizeMessage(List<ToolCall> toolCalls)
        {
            if (IsToolCall())
            {
                var tc = BuildToolCall();
                if (tc != null)
                    toolCalls.Add(tc);
            }
            _toolArgs.Clear();
            _currentRecipient = null;
        }

        private bool IsToolCall() =>
            _currentRecipient != null && _currentRecipient.StartsWith(FunctionPrefix, StringComparison.Ordinal);

        private ToolCall? BuildToolCall()
        {
            string name = _currentRecipient!.Substring(FunctionPrefix.Length);
            if (string.IsNullOrEmpty(name)) return null;

            var args = new Dictionary<string, object>();
            string raw = _toolArgs.ToString().Trim();
            if (raw.Length > 0)
            {
                try
                {
                    using var doc = JsonDocument.Parse(raw);
                    if (doc.RootElement.ValueKind == JsonValueKind.Object)
                    {
                        foreach (var prop in doc.RootElement.EnumerateObject())
                            args[prop.Name] = ChatMlOutputParser.JsonElementToObject(prop.Value);
                    }
                }
                catch
                {
                    // Malformed JSON: surface the call with no parsed arguments
                    // rather than dropping it entirely.
                }
            }
            return new ToolCall { Name = name, Arguments = args, Index = _callIndex++ };
        }

        /// <summary>Find the earliest message-terminating tag in the buffer.</summary>
        private static int FindEarliestEndTag(string buf, out int tagLen)
        {
            int best = -1;
            tagLen = 0;
            foreach (var tag in EndTags)
            {
                int idx = buf.IndexOf(tag, StringComparison.Ordinal);
                if (idx >= 0 && (best < 0 || idx < best))
                {
                    best = idx;
                    tagLen = tag.Length;
                }
            }
            return best;
        }

        private static int HoldBack(string buf, params string[] tags)
        {
            int maxOverlap = 0;
            foreach (var tag in tags)
            {
                int max = Math.Min(tag.Length, buf.Length);
                for (int i = max; i > 0; i--)
                {
                    if (buf.EndsWith(tag.Substring(0, i), StringComparison.Ordinal))
                    {
                        maxOverlap = Math.Max(maxOverlap, i);
                        break;
                    }
                }
            }
            return maxOverlap;
        }
    }

    // ========================================================================
    // Passthrough parser (no thinking/tool parsing)
    // ========================================================================

    public class PassthroughOutputParser : IOutputParser
    {
        public bool HasThinkingSupport => false;
        public bool HasToolSupport => false;
        public bool AlwaysRequired => false;

        public void Init(bool enableThinking, List<ToolFunction>? tools) { }

        public ParsedOutput Add(string text, bool done)
        {
            return new ParsedOutput { Content = text };
        }
    }

    // ========================================================================
    // DeepSeek V4 Parser: <think>...</think> for reasoning, and DSML markup for
    // tool calls:
    //     <｜DSML｜tool_calls>
    //     <｜DSML｜invoke name="get_weather">
    //     <｜DSML｜parameter name="city" string="true">Paris</｜DSML｜parameter>
    //     </｜DSML｜invoke>
    //     </｜DSML｜tool_calls>
    // `string="true"` means the value is the raw text between the tags; anything
    // else is JSON. Multiple <invoke> blocks in one call block are parallel calls.
    // ========================================================================

    public class DeepSeek4OutputParser : IOutputParser
    {
        private enum State { Content, Thinking, ToolCalls }

        private const string ThinkOpen = "<think>";
        private const string ThinkClose = "</think>";
        private const string Dsml = "｜DSML｜";
        private readonly string CallsOpen;
        private readonly string CallsClose;
        private readonly string _invokeOpen;
        private readonly string _invokeClose;
        private readonly string _paramOpen;
        private readonly string _paramClose;
        private readonly bool _deepSeek41;

        public DeepSeek4OutputParser() : this(false) { }

        protected DeepSeek4OutputParser(bool deepSeek41)
        {
            _deepSeek41 = deepSeek41;
            string space = deepSeek41 ? " " : "";
            string calls = deepSeek41 ? " calls" : "tool_calls";
            CallsOpen = "<" + Dsml + calls + ">";
            CallsClose = "</" + Dsml + calls + ">";
            _invokeOpen = "<" + Dsml + space + "invoke name=\"";
            _invokeClose = "</" + Dsml + space + "invoke>";
            _paramOpen = "<" + Dsml + space + "parameter name=\"";
            _paramClose = "</" + Dsml + space + "parameter>";
        }

        private State _state;
        private readonly StringBuilder _buffer = new();
        private bool _thinkingEnabled;
        private int _callIndex;

        public bool HasThinkingSupport => true;
        public bool HasToolSupport => true;
        public bool AlwaysRequired => true;

        public void Init(bool enableThinking, List<ToolFunction>? tools)
        {
            _buffer.Clear();
            _thinkingEnabled = enableThinking;
            _callIndex = 0;
            // The generation prompt already emitted `<think>` (thinking) or
            // `</think>` (not), so the model's own output starts inside the
            // reasoning block or straight in content.
            _state = enableThinking ? State.Thinking : State.Content;
        }

        public ParsedOutput Add(string text, bool done)
        {
            _buffer.Append(text);
            var result = new ParsedOutput();
            var contentSb = new StringBuilder();
            var thinkingSb = new StringBuilder();
            var toolCalls = new List<ToolCall>();

            bool keepParsing = true;
            while (keepParsing)
            {
                keepParsing = false;
                string buf = _buffer.ToString();
                if (buf.Length == 0)
                    break;

                switch (_state)
                {
                    case State.Thinking:
                    {
                        int closeIdx = buf.IndexOf(ThinkClose, StringComparison.Ordinal);
                        if (closeIdx >= 0)
                        {
                            thinkingSb.Append(buf, 0, closeIdx);
                            string after = buf.Substring(closeIdx + ThinkClose.Length);
                            _buffer.Clear();
                            _buffer.Append(after);
                            _state = State.Content;
                            keepParsing = after.Length > 0;
                        }
                        else if (done)
                        {
                            thinkingSb.Append(buf);
                            _buffer.Clear();
                        }
                        else
                        {
                            int hold = HoldBackForPartialTag(buf, ThinkClose);
                            if (hold < buf.Length)
                            {
                                thinkingSb.Append(buf, 0, buf.Length - hold);
                                _buffer.Clear();
                                _buffer.Append(buf.Substring(buf.Length - hold));
                            }
                        }
                        break;
                    }

                    case State.Content:
                    {
                        int callIdx = buf.IndexOf(CallsOpen, StringComparison.Ordinal);
                        if (callIdx >= 0)
                        {
                            contentSb.Append(buf, 0, callIdx);
                            string after = buf.Substring(callIdx + CallsOpen.Length);
                            _buffer.Clear();
                            _buffer.Append(after);
                            _state = State.ToolCalls;
                            keepParsing = true;
                            break;
                        }
                        // A late <think> can still open (the model may reason
                        // before answering even when the prompt closed the block).
                        int thinkIdx = _thinkingEnabled ? buf.IndexOf(ThinkOpen, StringComparison.Ordinal) : -1;
                        if (thinkIdx >= 0)
                        {
                            contentSb.Append(buf, 0, thinkIdx);
                            string after = buf.Substring(thinkIdx + ThinkOpen.Length);
                            _buffer.Clear();
                            _buffer.Append(after);
                            _state = State.Thinking;
                            keepParsing = after.Length > 0;
                            break;
                        }
                        if (done)
                        {
                            contentSb.Append(buf);
                            _buffer.Clear();
                        }
                        else
                        {
                            int hold = HoldBackForPartialTag(buf, CallsOpen, ThinkOpen);
                            if (hold < buf.Length)
                            {
                                contentSb.Append(buf, 0, buf.Length - hold);
                                _buffer.Clear();
                                _buffer.Append(buf.Substring(buf.Length - hold));
                            }
                        }
                        break;
                    }

                    case State.ToolCalls:
                    {
                        int endIdx = buf.IndexOf(CallsClose, StringComparison.Ordinal);
                        if (endIdx >= 0)
                        {
                            ParseInvokes(buf.Substring(0, endIdx), toolCalls);
                            string after = buf.Substring(endIdx + CallsClose.Length);
                            _buffer.Clear();
                            _buffer.Append(after);
                            _state = State.Content;
                            keepParsing = after.Length > 0;
                        }
                        else if (done)
                        {
                            // Generation stopped inside the block (hit the token
                            // budget, or EOS right after the last </invoke>):
                            // surface whatever invokes completed.
                            ParseInvokes(buf, toolCalls);
                            _buffer.Clear();
                            _state = State.Content;
                        }
                        break;
                    }
                }
            }

            result.Content = contentSb.ToString();
            result.Thinking = thinkingSb.ToString();
            result.ToolCalls = toolCalls.Count > 0 ? toolCalls : null;
            return result;
        }

        /// <summary>Parse every complete `&lt;invoke&gt;` block in the body.</summary>
        private void ParseInvokes(string body, List<ToolCall> toolCalls)
        {
            string invokeOpen = _invokeOpen;
            string invokeClose = _invokeClose;
            string paramOpen = _paramOpen;
            string paramClose = _paramClose;

            int pos = 0;
            while (true)
            {
                int start = body.IndexOf(invokeOpen, pos, StringComparison.Ordinal);
                if (start < 0)
                    break;
                int nameEnd = body.IndexOf('"', start + invokeOpen.Length);
                if (nameEnd < 0)
                    break;
                string name = body.Substring(start + invokeOpen.Length, nameEnd - start - invokeOpen.Length);

                int end = body.IndexOf(invokeClose, nameEnd, StringComparison.Ordinal);
                if (_deepSeek41)
                {
                    if (end < 0)
                        break;
                    int headerEnd = body.IndexOf('>', nameEnd + 1);
                    if (name.Length > 0 && headerEnd >= 0 && headerEnd < end &&
                        string.IsNullOrWhiteSpace(body.Substring(nameEnd + 1, headerEnd - nameEnd - 1)) &&
                        DeepSeek41OutputParser.TryParseParameters(
                            body.Substring(headerEnd + 1, end - headerEnd - 1), out var v41Args))
                        toolCalls.Add(new ToolCall { Name = name, Arguments = v41Args, Index = _callIndex++ });
                    pos = end + invokeClose.Length;
                    continue;
                }
                string inner = end < 0 ? body.Substring(nameEnd) : body.Substring(nameEnd, end - nameEnd);

                var args = new Dictionary<string, object>();
                int p = 0;
                while (true)
                {
                    int pStart = inner.IndexOf(paramOpen, p, StringComparison.Ordinal);
                    if (pStart < 0)
                        break;
                    int keyEnd = inner.IndexOf('"', pStart + paramOpen.Length);
                    if (keyEnd < 0)
                        break;
                    string key = inner.Substring(pStart + paramOpen.Length, keyEnd - pStart - paramOpen.Length);

                    // string="true|false" decides whether the value is raw text
                    // or JSON; a missing attribute is treated as text.
                    int tagEnd = inner.IndexOf('>', keyEnd);
                    if (tagEnd < 0)
                        break;
                    string attrs = inner.Substring(keyEnd, tagEnd - keyEnd);
                    bool isString = !attrs.Contains("string=\"false\"", StringComparison.Ordinal);

                    int valEnd = inner.IndexOf(paramClose, tagEnd + 1, StringComparison.Ordinal);
                    string raw = valEnd < 0
                        ? inner.Substring(tagEnd + 1)
                        : inner.Substring(tagEnd + 1, valEnd - tagEnd - 1);

                    if (key.Length > 0)
                        args[key] = isString ? raw.Trim() : ParseJsonValue(raw.Trim());

                    if (valEnd < 0)
                        break;
                    p = valEnd + paramClose.Length;
                }

                if (name.Length > 0)
                    toolCalls.Add(new ToolCall { Name = name, Arguments = args, Index = _callIndex++ });

                if (end < 0)
                    break;
                pos = end + invokeClose.Length;
            }
        }

        private static object ParseJsonValue(string value)
        {
            if (value.Length == 0)
                return value;
            try
            {
                using var doc = JsonDocument.Parse(value);
                return ChatMlOutputParser.JsonElementToObject(doc.RootElement);
            }
            catch (JsonException)
            {
                // The model labelled it non-string but did not write JSON; the
                // text is still better than dropping the argument.
                return value;
            }
        }

        private static int HoldBackForPartialTag(string buf, params string[] tags)
        {
            int maxOverlap = 0;
            foreach (var tag in tags)
            {
                int max = Math.Min(tag.Length, buf.Length);
                for (int i = max; i > 0; i--)
                {
                    if (buf.EndsWith(tag.Substring(0, i), StringComparison.Ordinal))
                    {
                        maxOverlap = Math.Max(maxOverlap, i);
                        break;
                    }
                }
            }
            return maxOverlap;
        }
    }

    // ========================================================================
    // Muse-Glimmer Parser
    //
    // Same <|start|>HEADER<|message|>BODY framing as harmony, but the channel is
    // carried by the header's recipient rather than a <|channel|> tag, and a
    // message ends at <|eom|> (more to come this turn) or <|eot|> (turn over):
    //
    //   <|start|>assistant to=self<|message|>...reasoning...<|eom|>
    //   <|start|>assistant to=weather.get<|message|><atem:function_calls>...<|eom|>
    //   <|start|>assistant<|message|>...the answer...<|eot|>
    //
    // Without this the framing and the whole reasoning channel were streamed to
    // the user verbatim, so every reply opened with a literal
    // " to=self<|message|>" followed by the model restating the prompt.
    // ========================================================================

    public class MuseGlimmerOutputParser : IOutputParser
    {
        private enum MState { LookingForStart, ParsingHeader, ParsingContent }

        private MState _state;
        private readonly StringBuilder _buffer = new();
        private readonly StringBuilder _toolArgs = new();
        private string? _currentRecipient;
        private int _callIndex;

        private const string MsgStartTag = "<|start|>";
        private const string HeaderEndTag = "<|message|>";
        private const string EomTag = "<|eom|>";
        private const string EotTag = "<|eot|>";

        private static readonly string[] EndTags = { EomTag, EotTag };
        private static readonly string[] HoldTags = { EomTag, EotTag, MsgStartTag };

        /// <summary>Safety valve: a "header" this long means the stream never closed one.</summary>
        private const int MaxHeaderChars = 512;

        public bool HasThinkingSupport => true;
        public bool HasToolSupport => true;
        // The framing tokens are always emitted, so the parser is never optional:
        // skipping it would leak "<|start|>assistant to=self<|message|>" verbatim.
        public bool AlwaysRequired => true;

        public void Init(bool enableThinking, List<ToolFunction>? tools)
        {
            _buffer.Clear();
            _toolArgs.Clear();
            _state = MState.ParsingHeader;
            _currentRecipient = null;
            _callIndex = 0;
            // The prompt's generation marker is "<|start|>assistant", so the first
            // token the model emits belongs to that message's HEADER (" to=self" or
            // straight to "<|message|>"). Start mid-header rather than hunting for
            // a <|start|> that has already been consumed by the prompt.
        }

        public ParsedOutput Add(string text, bool done)
        {
            _buffer.Append(text);
            var result = new ParsedOutput();
            var contentSb = new StringBuilder();
            var thinkingSb = new StringBuilder();
            var toolTextSb = new StringBuilder();
            var toolCalls = new List<ToolCall>();

            bool keepParsing = true;
            while (keepParsing)
            {
                keepParsing = false;
                string buf = _buffer.ToString();
                if (buf.Length == 0)
                {
                    if (done && _state == MState.ParsingContent)
                    {
                        FinalizeMessage(toolCalls);
                        _state = MState.LookingForStart;
                    }
                    break;
                }

                switch (_state)
                {
                    case MState.LookingForStart:
                        int startIdx = buf.IndexOf(MsgStartTag, StringComparison.Ordinal);
                        if (startIdx >= 0)
                        {
                            _buffer.Clear();
                            _buffer.Append(buf.Substring(startIdx + MsgStartTag.Length));
                            _state = MState.ParsingHeader;
                            keepParsing = true;
                        }
                        else if (!done)
                        {
                            int hold = HarmonyHoldBack(buf, MsgStartTag);
                            if (hold > 0)
                            {
                                _buffer.Clear();
                                _buffer.Append(buf.Substring(buf.Length - hold));
                            }
                            else
                            {
                                _buffer.Clear();
                            }
                        }
                        break;

                    case MState.ParsingHeader:
                        if (CannotBeHeader(buf))
                        {
                            // Headerless output. A structured-output grammar armed from
                            // token 0 forbids the " to=user<|message|>" header, so the
                            // model's first token is already the answer's "{". Waiting
                            // for a <|message|> that can never come swallowed the whole
                            // correct object: the stream delivered content null and the
                            // json_schema path returned 422 on an empty string.
                            _currentRecipient = null;
                            _state = MState.ParsingContent;
                            keepParsing = true;
                            break;
                        }
                        int headerEnd = buf.IndexOf(HeaderEndTag, StringComparison.Ordinal);
                        if (headerEnd >= 0)
                        {
                            ParseHeader(buf.Substring(0, headerEnd));
                            string after = buf.Substring(headerEnd + HeaderEndTag.Length);
                            _buffer.Clear();
                            _buffer.Append(after);
                            _state = MState.ParsingContent;
                            keepParsing = after.Length > 0;
                        }
                        else if (!done)
                        {
                            // Keep the WHOLE header buffered until <|message|>
                            // arrives. Trimming it to the partial-tag suffix (the
                            // holdback the content state uses) would throw away the
                            // "to=..." recipient whenever a chunk boundary lands
                            // inside <|message|>, and an unrecognised recipient
                            // silently routes the reasoning channel to the user.
                            // Headers are a handful of characters; the cap is only
                            // a guard against a stream that never closes one.
                            if (buf.Length > MaxHeaderChars)
                            {
                                EmitContent(buf, contentSb, thinkingSb, toolTextSb);
                                _buffer.Clear();
                                _state = MState.ParsingContent;
                            }
                        }
                        else
                        {
                            // The stream ended inside what was buffered as a header. A
                            // real, unfinished header ("assistant to=self") carries no
                            // text for anyone; anything else is text the model wrote
                            // without framing and must not be dropped on the floor.
                            if (!LooksLikeHeader(buf))
                            {
                                _currentRecipient = null;
                                EmitContent(buf, contentSb, thinkingSb, toolTextSb);
                            }
                            _buffer.Clear();
                            _state = MState.LookingForStart;
                        }
                        break;

                    case MState.ParsingContent:
                        int endIdx = FindEarliestEnd(buf, out int tagLen);
                        if (endIdx >= 0)
                        {
                            EmitContent(buf.Substring(0, endIdx), contentSb, thinkingSb, toolTextSb);
                            string after = buf.Substring(endIdx + tagLen);
                            _buffer.Clear();
                            _buffer.Append(after);
                            FinalizeMessage(toolCalls);
                            _state = MState.LookingForStart;
                            keepParsing = after.Length > 0;
                        }
                        else if (!done)
                        {
                            int hold = HarmonyHoldBack(buf, HoldTags);
                            if (hold > 0)
                            {
                                string emit = buf.Substring(0, buf.Length - hold);
                                if (emit.Length > 0) EmitContent(emit, contentSb, thinkingSb, toolTextSb);
                                _buffer.Clear();
                                _buffer.Append(buf.Substring(buf.Length - hold));
                            }
                            else
                            {
                                EmitContent(buf, contentSb, thinkingSb, toolTextSb);
                                _buffer.Clear();
                            }
                        }
                        else
                        {
                            EmitContent(buf, contentSb, thinkingSb, toolTextSb);
                            _buffer.Clear();
                            FinalizeMessage(toolCalls);
                            _state = MState.LookingForStart;
                        }
                        break;
                }
            }

            result.Content = contentSb.ToString();
            result.Thinking = thinkingSb.ToString();
            result.ToolCallText = toolTextSb.ToString();
            if (IsToolCall())
                result.ToolCallName = _currentRecipient;
            if (toolCalls.Count > 0)
                result.ToolCalls = toolCalls;
            return result;
        }

        /// <summary>Header text between &lt;|start|&gt; and &lt;|message|&gt;, e.g. "assistant to=self".</summary>
        private void ParseHeader(string header)
        {
            _currentRecipient = null;
            int toIdx = header.IndexOf("to=", StringComparison.Ordinal);
            if (toIdx < 0) return;
            string rest = header.Substring(toIdx + 3);
            int end = 0;
            while (end < rest.Length && !char.IsWhiteSpace(rest[end]) && rest[end] != '<')
                end++;
            if (end > 0)
                _currentRecipient = rest.Substring(0, end);
        }

        /// <summary>
        /// A message header is routing text: optional <c>&lt;|start|&gt;</c>, the role
        /// word and a <c>to=RECIPIENT</c>, before <c>&lt;|message|&gt;</c>. It therefore
        /// opens (after whitespace) with a letter or a <c>&lt;</c> of a framing token.
        /// Anything else - JSON's <c>{</c>/<c>[</c>/<c>"</c>, a digit - is body text
        /// written with no header at all.
        /// </summary>
        internal static bool CannotBeHeader(string buf)
        {
            foreach (char c in buf)
            {
                if (char.IsWhiteSpace(c)) continue;
                return !(char.IsLetter(c) || c == '<');
            }
            return false;
        }

        private static readonly System.Text.RegularExpressions.Regex HeaderShape = new(
            @"^\s*(<\|start\|>)?\s*(assistant)?\s*(to=[^\s<]*)?\s*(<\|?[a-z_|]*)?$",
            System.Text.RegularExpressions.RegexOptions.CultureInvariant);

        /// <summary>True when <paramref name="buf"/> is (a prefix of) a routing header.</summary>
        internal static bool LooksLikeHeader(string buf) => HeaderShape.IsMatch(buf);

        private bool IsThinking() => string.Equals(_currentRecipient, "self", StringComparison.Ordinal);

        private bool IsToolCall() =>
            _currentRecipient != null &&
            !string.Equals(_currentRecipient, "self", StringComparison.Ordinal) &&
            !string.Equals(_currentRecipient, "user", StringComparison.Ordinal);

        private void EmitContent(
            string content, StringBuilder contentSb, StringBuilder thinkingSb, StringBuilder toolTextSb)
        {
            if (content.Length == 0) return;
            if (IsToolCall())
            {
                // Same live progress copy the Harmony parser makes: the body is
                // already drained incrementally, so this is the streaming signal.
                _toolArgs.Append(content);
                toolTextSb.Append(content);
            }
            else if (IsThinking()) thinkingSb.Append(content);
            else contentSb.Append(content);
        }

        private void FinalizeMessage(List<ToolCall> toolCalls)
        {
            if (IsToolCall())
            {
                var tc = BuildAtemToolCall(_currentRecipient!, _toolArgs.ToString(), _callIndex);
                if (tc != null) { toolCalls.Add(tc); _callIndex++; }
            }
            _toolArgs.Clear();
            _currentRecipient = null;
        }

        /// <summary>
        /// Parse the ATEM XML block the chat template documents:
        /// <![CDATA[
        /// <atem:function_calls><atem:invoke name="NAME">
        ///   <atem:parameter name="k">v</atem:parameter>
        /// </atem:invoke></atem:function_calls>
        /// ]]>
        /// Values are JSON-decoded when they parse as JSON (lists/objects/numbers/
        /// booleans, which is how the template serialises them) and kept as text
        /// otherwise.
        /// </summary>
        internal static ToolCall? BuildAtemToolCall(string recipient, string body, int index)
        {
            string name = recipient;
            const string invokeOpen = "<atem:invoke name=\"";
            int inv = body.IndexOf(invokeOpen, StringComparison.Ordinal);
            if (inv >= 0)
            {
                int nameStart = inv + invokeOpen.Length;
                int nameEnd = body.IndexOf('"', nameStart);
                if (nameEnd > nameStart) name = body.Substring(nameStart, nameEnd - nameStart);
            }
            if (string.IsNullOrEmpty(name)) return null;

            var args = new Dictionary<string, object>();
            const string paramOpen = "<atem:parameter name=\"";
            const string paramClose = "</atem:parameter>";
            int pos = 0;
            while (true)
            {
                int p = body.IndexOf(paramOpen, pos, StringComparison.Ordinal);
                if (p < 0) break;
                int keyStart = p + paramOpen.Length;
                int keyEnd = body.IndexOf('"', keyStart);
                if (keyEnd < 0) break;
                int valStart = body.IndexOf('>', keyEnd);
                if (valStart < 0) break;
                valStart++;
                int valEnd = body.IndexOf(paramClose, valStart, StringComparison.Ordinal);
                if (valEnd < 0) break;
                string key = body.Substring(keyStart, keyEnd - keyStart);
                string raw = body.Substring(valStart, valEnd - valStart);
                args[key] = ParseAtemValue(raw);
                pos = valEnd + paramClose.Length;
            }
            return new ToolCall { Name = name, Arguments = args, Index = index };
        }

        private static object ParseAtemValue(string raw)
        {
            string t = raw.Trim();
            if (t.Length == 0) return raw;
            if (t == "true") return true;
            if (t == "false") return false;
            if (t == "null") return string.Empty;
            char c0 = t[0];
            if (c0 == '[' || c0 == '{' || c0 == '-' || (c0 >= '0' && c0 <= '9'))
            {
                try
                {
                    using var doc = JsonDocument.Parse(t);
                    return ChatMlOutputParser.JsonElementToObject(doc.RootElement);
                }
                catch
                {
                    // Not JSON after all - fall through to the raw text.
                }
            }
            return raw;
        }

        private static int FindEarliestEnd(string buf, out int tagLen)
        {
            int best = -1;
            tagLen = 0;
            foreach (var tag in EndTags)
            {
                int idx = buf.IndexOf(tag, StringComparison.Ordinal);
                if (idx >= 0 && (best < 0 || idx < best))
                {
                    best = idx;
                    tagLen = tag.Length;
                }
            }
            return best;
        }

        /// <summary>Longest suffix of <paramref name="buf"/> that is a prefix of any tag.</summary>
        private static int HarmonyHoldBack(string buf, params string[] tags)
        {
            int maxOverlap = 0;
            foreach (var tag in tags)
            {
                int max = Math.Min(tag.Length - 1, buf.Length);
                for (int i = max; i > 0; i--)
                {
                    if (string.CompareOrdinal(buf, buf.Length - i, tag, 0, i) == 0)
                    {
                        maxOverlap = Math.Max(maxOverlap, i);
                        break;
                    }
                }
            }
            return maxOverlap;
        }
    }

    // ========================================================================
    // GLM-5.x (glm-dsa): <think>...</think> reasoning, then content, with tool
    // calls as <tool_call>NAME<arg_key>k</arg_key><arg_value>v</arg_value>...</tool_call>
    // ========================================================================

    /// <summary>
    /// Parser for the GLM-5.x reply format.
    ///
    /// <para>The generation prompt already emits the opening <c>&lt;think&gt;</c>
    /// (or an immediately-closed pair when thinking is off), so the model's own
    /// output starts INSIDE the reasoning block and closes it with
    /// <c>&lt;/think&gt;</c>. Everything after that is the answer, except for
    /// <c>&lt;tool_call&gt;</c> blocks, which carry the function name as bare
    /// text followed by alternating key/value tags.</para>
    ///
    /// <para>Unlike Qwen's JSON-bodied tool calls, GLM's arguments arrive as one
    /// XML element per argument, so a value is taken verbatim unless it parses as
    /// JSON — that is what the model emits for numbers, booleans, arrays and
    /// objects (the template renders them with <c>tojson</c>).</para>
    /// </summary>
    public class GlmDsaOutputParser : IOutputParser
    {
        private enum State { Thinking, Content, ToolCall }

        private const string ThinkOpen = "<think>";
        private const string ThinkClose = "</think>";
        private const string CallOpen = "<tool_call>";
        private const string CallClose = "</tool_call>";

        private State _state;
        private readonly StringBuilder _buffer = new();
        private bool _thinkingEnabled;
        private int _callIndex;

        public bool HasThinkingSupport => true;
        public bool HasToolSupport => true;
        public bool AlwaysRequired => true;

        public void Init(bool enableThinking, List<ToolFunction>? tools)
        {
            _buffer.Clear();
            _thinkingEnabled = enableThinking;
            _callIndex = 0;
            _state = enableThinking ? State.Thinking : State.Content;
        }

        public ParsedOutput Add(string text, bool done)
        {
            _buffer.Append(text);
            var result = new ParsedOutput();
            var contentSb = new StringBuilder();
            var thinkingSb = new StringBuilder();
            var toolCalls = new List<ToolCall>();

            bool keepParsing = true;
            while (keepParsing)
            {
                keepParsing = false;
                string buf = _buffer.ToString();
                if (buf.Length == 0)
                    break;

                switch (_state)
                {
                    case State.Thinking:
                    {
                        int closeIdx = buf.IndexOf(ThinkClose, StringComparison.Ordinal);
                        if (closeIdx >= 0)
                        {
                            thinkingSb.Append(buf, 0, closeIdx);
                            string after = buf.Substring(closeIdx + ThinkClose.Length);
                            _buffer.Clear();
                            _buffer.Append(after);
                            _state = State.Content;
                            keepParsing = after.Length > 0;
                        }
                        else if (done)
                        {
                            thinkingSb.Append(buf);
                            _buffer.Clear();
                        }
                        else
                        {
                            int hold = HoldBackForPartialTag(buf, ThinkClose);
                            if (hold < buf.Length)
                            {
                                thinkingSb.Append(buf, 0, buf.Length - hold);
                                _buffer.Clear();
                                _buffer.Append(buf.Substring(buf.Length - hold));
                            }
                        }
                        break;
                    }

                    case State.Content:
                    {
                        int callIdx = buf.IndexOf(CallOpen, StringComparison.Ordinal);
                        if (callIdx >= 0)
                        {
                            contentSb.Append(buf, 0, callIdx);
                            string after = buf.Substring(callIdx + CallOpen.Length);
                            _buffer.Clear();
                            _buffer.Append(after);
                            _state = State.ToolCall;
                            keepParsing = true;
                            break;
                        }
                        // A late <think> can still open: the model may reason again
                        // between answers even though the prompt closed the block.
                        int thinkIdx = _thinkingEnabled ? buf.IndexOf(ThinkOpen, StringComparison.Ordinal) : -1;
                        if (thinkIdx >= 0)
                        {
                            contentSb.Append(buf, 0, thinkIdx);
                            string after = buf.Substring(thinkIdx + ThinkOpen.Length);
                            _buffer.Clear();
                            _buffer.Append(after);
                            _state = State.Thinking;
                            keepParsing = after.Length > 0;
                            break;
                        }
                        if (done)
                        {
                            contentSb.Append(buf);
                            _buffer.Clear();
                        }
                        else
                        {
                            int hold = HoldBackForPartialTag(buf, CallOpen, ThinkOpen);
                            if (hold < buf.Length)
                            {
                                contentSb.Append(buf, 0, buf.Length - hold);
                                _buffer.Clear();
                                _buffer.Append(buf.Substring(buf.Length - hold));
                            }
                        }
                        break;
                    }

                    case State.ToolCall:
                    {
                        int endIdx = buf.IndexOf(CallClose, StringComparison.Ordinal);
                        if (endIdx >= 0)
                        {
                            ParseGlmToolCall(buf.Substring(0, endIdx), toolCalls);
                            string after = buf.Substring(endIdx + CallClose.Length);
                            _buffer.Clear();
                            _buffer.Append(after);
                            _state = State.Content;
                            keepParsing = after.Length > 0;
                        }
                        else if (done)
                        {
                            // Generation stopped inside the block (token budget, or
                            // EOS right after the last argument): surface whatever
                            // completed rather than dropping the call.
                            ParseGlmToolCall(buf, toolCalls);
                            _buffer.Clear();
                            _state = State.Content;
                        }
                        break;
                    }
                }
            }

            result.Content = contentSb.ToString();
            result.Thinking = thinkingSb.ToString();
            result.ToolCalls = toolCalls.Count > 0 ? toolCalls : null;
            return result;
        }

        /// <summary>
        /// Body of one &lt;tool_call&gt; block: the function name, then alternating
        /// &lt;arg_key&gt;/&lt;arg_value&gt; pairs.
        /// </summary>
        private void ParseGlmToolCall(string body, List<ToolCall> toolCalls)
        {
            const string keyOpen = "<arg_key>";
            const string keyClose = "</arg_key>";
            const string valOpen = "<arg_value>";
            const string valClose = "</arg_value>";

            int firstKey = body.IndexOf(keyOpen, StringComparison.Ordinal);
            string name = (firstKey >= 0 ? body.Substring(0, firstKey) : body).Trim();
            if (name.Length == 0)
                return;

            var args = new Dictionary<string, object>();
            int pos = firstKey < 0 ? body.Length : firstKey;
            while (pos < body.Length)
            {
                int ks = body.IndexOf(keyOpen, pos, StringComparison.Ordinal);
                if (ks < 0) break;
                int ke = body.IndexOf(keyClose, ks + keyOpen.Length, StringComparison.Ordinal);
                if (ke < 0) break;
                string key = body.Substring(ks + keyOpen.Length, ke - ks - keyOpen.Length).Trim();

                int vs = body.IndexOf(valOpen, ke + keyClose.Length, StringComparison.Ordinal);
                if (vs < 0) break;
                int ve = body.IndexOf(valClose, vs + valOpen.Length, StringComparison.Ordinal);
                string raw = ve < 0
                    ? body.Substring(vs + valOpen.Length)
                    : body.Substring(vs + valOpen.Length, ve - vs - valOpen.Length);

                if (key.Length > 0)
                    args[key] = ParseJsonValue(raw.Trim());

                if (ve < 0) break;
                pos = ve + valClose.Length;
            }

            toolCalls.Add(new ToolCall { Name = name, Arguments = args, Index = _callIndex++ });
        }

        /// <summary>
        /// A GLM argument is a JSON scalar / array / object when the template
        /// rendered it with <c>tojson</c>, and bare text otherwise. Numbers,
        /// booleans, null and bracketed values are parsed; everything else stays
        /// the literal string the model wrote.
        /// </summary>
        private static object ParseJsonValue(string value)
        {
            if (value.Length == 0)
                return string.Empty;

            char c = value[0];
            bool looksJson = c == '{' || c == '[' || c == '"' || c == '-' || char.IsDigit(c) ||
                             value == "true" || value == "false" || value == "null";
            if (!looksJson)
                return value;

            try
            {
                using var doc = JsonDocument.Parse(value);
                return JsonElementToObject(doc.RootElement);
            }
            catch (JsonException)
            {
                return value;
            }
        }

        private static object JsonElementToObject(JsonElement e)
        {
            switch (e.ValueKind)
            {
                case JsonValueKind.String: return e.GetString() ?? string.Empty;
                // Boxed separately: a `? long : double` ternary would widen every
                // integer to double, so a tool argument of 3 would arrive as 3.0.
                case JsonValueKind.Number:
                    if (e.TryGetInt64(out long l)) return l;
                    return e.GetDouble();
                case JsonValueKind.True: return true;
                case JsonValueKind.False: return false;
                case JsonValueKind.Null: return null;
                case JsonValueKind.Array:
                {
                    var list = new List<object>();
                    foreach (var item in e.EnumerateArray()) list.Add(JsonElementToObject(item));
                    return list;
                }
                case JsonValueKind.Object:
                {
                    var map = new Dictionary<string, object>();
                    foreach (var prop in e.EnumerateObject()) map[prop.Name] = JsonElementToObject(prop.Value);
                    return map;
                }
                default: return e.ToString();
            }
        }

        private static int HoldBackForPartialTag(string buf, params string[] tags)
        {
            int hold = 0;
            foreach (string tag in tags)
            {
                int max = Math.Min(tag.Length - 1, buf.Length);
                for (int len = max; len > 0; len--)
                {
                    if (string.CompareOrdinal(buf, buf.Length - len, tag, 0, len) == 0)
                    {
                        if (len > hold) hold = len;
                        break;
                    }
                }
            }
            return hold;
        }
    }

    // ========================================================================
    // Factory
    // ========================================================================

    public static class OutputParserFactory
    {
        /// <summary>
        /// Parser for a family's raw stream. Which parser, whether it is mandatory and
        /// where a grammar may arm are all declared together in
        /// <see cref="ChatProtocolRegistry"/>, beside that family's prompt renderer -
        /// they are four faces of one text protocol, and splitting them across separate
        /// name chains is how a family used to end up half-added.
        /// </summary>
        public static IOutputParser Create(string architecture)
            => ChatProtocolRegistry.For(architecture)?.CreateOutputParser?.Invoke()
               ?? new PassthroughOutputParser();

        /// <summary>
        /// Text after which a structured-output grammar may start enforcing, or null
        /// when the model's very first token is already part of the answer.
        /// </summary>
        public static string? GrammarActivationTrigger(string architecture, bool enableThinking = false)
        {
            var protocol = ChatProtocolRegistry.For(architecture);
            return enableThinking
                ? protocol?.ThinkingGrammarActivationTrigger ?? protocol?.GrammarActivationTrigger
                : protocol?.GrammarActivationTrigger;
        }

        /// <summary>
        /// True when the reply is unreadable without its parser: the framing tokens and
        /// the whole chain of thought would otherwise stream to the client as if they
        /// were the answer.
        /// </summary>
        public static bool IsAlwaysRequired(string architecture)
            => ChatProtocolRegistry.For(architecture)?.OutputParserAlwaysRequired ?? false;
    }
}
