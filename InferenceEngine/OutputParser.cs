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

namespace InferenceEngine
{
    /// <summary>
    /// Represents a tool function definition provided to the model.
    /// </summary>
    public class ToolFunction
    {
        public string Name { get; set; }
        public string Description { get; set; }
        public Dictionary<string, ToolParameter> Parameters { get; set; }
        public List<string> Required { get; set; }
    }

    public class ToolParameter
    {
        public string Type { get; set; }
        public string Description { get; set; }
        public List<string> Enum { get; set; }
    }

    /// <summary>
    /// Represents a tool call extracted from model output.
    /// </summary>
    public class ToolCall
    {
        public string Name { get; set; }
        public Dictionary<string, object> Arguments { get; set; }
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
        public List<ToolCall> ToolCalls { get; set; }
    }

    /// <summary>
    /// Streaming parser that extracts thinking content, regular content, and tool calls
    /// from model output. Handles model-specific tag formats.
    /// </summary>
    public interface IOutputParser
    {
        void Init(bool enableThinking, List<ToolFunction> tools);
        ParsedOutput Add(string text, bool done);
        bool HasThinkingSupport { get; }
        bool HasToolSupport { get; }
        /// <summary>
        /// True when the model's wire format always requires parsing (e.g. Harmony
        /// framing), even if the caller did not request thinking or tool support.
        /// </summary>
        bool AlwaysRequired { get; }
    }

    // ========================================================================
    // Qwen3 Parser: <think>...</think> for thinking, <tool_call>...</tool_call>
    // ========================================================================

    public class Qwen3OutputParser : IOutputParser
    {
        private enum State { CollectingThinking, ThinkingDone, CollectingContent, CollectingTool }

        private State _state;
        private readonly StringBuilder _buffer = new();
        private bool _stripLeadingThinkTag;
        private int _callIndex;

        public bool HasThinkingSupport => true;
        public bool HasToolSupport => true;
        public bool AlwaysRequired => false;

        public void Init(bool enableThinking, List<ToolFunction> tools)
        {
            _buffer.Clear();
            _callIndex = 0;
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
                            var tc = ParseQwen3ToolCall(raw);
                            if (tc != null) toolCalls.Add(tc);
                            _state = State.CollectingContent;
                            keepParsing = after.Length > 0;
                        }
                        else if (done && buf.Length > 0)
                        {
                            var tc = ParseQwen3ToolCall(buf);
                            if (tc != null) toolCalls.Add(tc);
                            _buffer.Clear();
                            _state = State.CollectingContent;
                        }
                        break;
                }
            }

            result.Content = contentSb.ToString();
            result.Thinking = thinkingSb.ToString();
            result.ToolCalls = toolCalls.Count > 0 ? toolCalls : null;
            return result;
        }

        private ToolCall ParseQwen3ToolCall(string raw)
        {
            raw = raw.Trim();
            if (raw.Length == 0) return null;
            try
            {
                using var doc = JsonDocument.Parse(raw);
                var root = doc.RootElement;
                string name = root.GetProperty("name").GetString();
                if (string.IsNullOrEmpty(name)) return null;

                var args = new Dictionary<string, object>();
                if (root.TryGetProperty("arguments", out var argsEl) && argsEl.ValueKind == JsonValueKind.Object)
                {
                    foreach (var prop in argsEl.EnumerateObject())
                        args[prop.Name] = JsonElementToObject(prop.Value);
                }
                return new ToolCall { Name = name, Arguments = args, Index = _callIndex++ };
            }
            catch
            {
                return null;
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

        internal static object JsonElementToObject(JsonElement el)
        {
            return el.ValueKind switch
            {
                JsonValueKind.String => el.GetString(),
                JsonValueKind.Number => el.TryGetInt64(out long l) ? (object)l : el.GetDouble(),
                JsonValueKind.True => true,
                JsonValueKind.False => false,
                JsonValueKind.Null => null,
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
    // Qwen3.5 Parser: same tags as Qwen3, always starts in thinking mode
    // ========================================================================

    public class Qwen35OutputParser : Qwen3OutputParser
    {
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

        public bool HasThinkingSupport => true;
        public bool HasToolSupport => true;
        public bool AlwaysRequired => false;

        public void Init(bool enableThinking, List<ToolFunction> tools)
        {
            _buffer.Clear();
            _thinkingEnabled = enableThinking;
            _needsChannelNameStrip = false;
            _state = State.CollectingContent;
        }

        public ParsedOutput Add(string text, bool done)
        {
            _buffer.Append(text);
            var result = new ParsedOutput();
            var thinkingSb = new StringBuilder();
            var contentSb = new StringBuilder();
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

                        if (chIdx >= 0 && (tcIdx < 0 || chIdx < tcIdx))
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
                            int hold = HoldBack(buf, "<|channel>", "<|tool_call>");
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
                            var tc = ParseGemma4ToolCall(raw);
                            if (tc != null) toolCalls.Add(tc);
                            _state = State.CollectingContent;
                            keepParsing = after.Length > 0;
                        }
                        else if (done && buf.Length > 0)
                        {
                            var tc = ParseGemma4ToolCall(buf);
                            if (tc != null) toolCalls.Add(tc);
                            _buffer.Clear();
                            _state = State.CollectingContent;
                        }
                        break;
                }
            }

            result.Content = contentSb.ToString();
            result.Thinking = thinkingSb.ToString();
            result.ToolCalls = toolCalls.Count > 0 ? toolCalls : null;
            return result;
        }

        private static readonly Regex GemmaQuotedStringRe = new(@"<\|""\|>(.*?)<\|""\|>", RegexOptions.Singleline);
        private static readonly Regex GemmaBareKeyRe = new(@"([,{])(\w+):");

        private static ToolCall ParseGemma4ToolCall(string content)
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
                    args[prop.Name] = Qwen3OutputParser.JsonElementToObject(prop.Value);
                return new ToolCall { Name = name, Arguments = args };
            }
            catch
            {
                return null;
            }
        }

        internal static string Gemma4ArgsToJson(string s)
        {
            var quotedStrings = new List<string>();
            string text = GemmaQuotedStringRe.Replace(s, m =>
            {
                quotedStrings.Add(m.Groups[1].Value);
                return "\x00" + (char)(quotedStrings.Count - 1) + "\x00";
            });

            text = GemmaBareKeyRe.Replace(text, "$1\"$2\":");

            for (int i = 0; i < quotedStrings.Count; i++)
            {
                string escaped = JsonSerializer.Serialize(quotedStrings[i]);
                text = text.Replace("\x00" + (char)i + "\x00", escaped);
            }

            return text;
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
        private bool _thinkingEnabled;
        private string _currentChannel;

        private const string MsgStartTag = "<|start|>";
        private const string MsgEndTag = "<|end|>";
        private const string HeaderEndTag = "<|message|>";
        private const string ChannelTag = "<|channel|>";

        public bool HasThinkingSupport => true;
        public bool HasToolSupport => false;
        public bool AlwaysRequired => true;

        public void Init(bool enableThinking, List<ToolFunction> tools)
        {
            _buffer.Clear();
            _thinkingEnabled = enableThinking;
            _state = HState.LookingForStart;
            _currentChannel = null;

            _buffer.Append("<|start|>assistant");
        }

        public ParsedOutput Add(string text, bool done)
        {
            _buffer.Append(text);
            var result = new ParsedOutput();
            var contentSb = new StringBuilder();
            var thinkingSb = new StringBuilder();

            bool keepParsing = true;
            while (keepParsing)
            {
                keepParsing = false;
                string buf = _buffer.ToString();
                if (buf.Length == 0) break;

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

                            _state = HState.ParsingContent;
                            keepParsing = after.Length > 0;
                        }
                        else if (!done)
                        {
                            int hold = HoldBack(buf, HeaderEndTag);
                            if (hold > 0 && hold < buf.Length)
                            {
                                _buffer.Clear();
                                _buffer.Append(buf.Substring(buf.Length - hold));
                            }
                        }
                        break;

                    case HState.ParsingContent:
                        int endIdx = buf.IndexOf(MsgEndTag, StringComparison.Ordinal);
                        if (endIdx >= 0)
                        {
                            string content = buf.Substring(0, endIdx);
                            string after = buf.Substring(endIdx + MsgEndTag.Length);
                            _buffer.Clear();
                            _buffer.Append(after);

                            EmitContent(content, contentSb, thinkingSb);
                            _state = HState.LookingForStart;
                            keepParsing = after.Length > 0;
                        }
                        else if (!done)
                        {
                            int hold = HoldBack(buf, MsgEndTag, MsgStartTag);
                            if (hold > 0)
                            {
                                string emit = buf.Substring(0, buf.Length - hold);
                                if (emit.Length > 0) EmitContent(emit, contentSb, thinkingSb);
                                _buffer.Clear();
                                _buffer.Append(buf.Substring(buf.Length - hold));
                            }
                            else
                            {
                                EmitContent(buf, contentSb, thinkingSb);
                                _buffer.Clear();
                            }
                        }
                        else
                        {
                            if (buf.Length > 0) EmitContent(buf, contentSb, thinkingSb);
                            _buffer.Clear();
                        }
                        break;
                }
            }

            result.Content = contentSb.ToString();
            result.Thinking = thinkingSb.ToString();
            return result;
        }

        private void EmitContent(string content, StringBuilder contentSb, StringBuilder thinkingSb)
        {
            if (_currentChannel == "analysis")
                thinkingSb.Append(content);
            else
                contentSb.Append(content);
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

        public void Init(bool enableThinking, List<ToolFunction> tools) { }

        public ParsedOutput Add(string text, bool done)
        {
            return new ParsedOutput { Content = text };
        }
    }

    // ========================================================================
    // Factory
    // ========================================================================

    public static class OutputParserFactory
    {
        public static IOutputParser Create(string architecture)
        {
            return architecture switch
            {
                "gemma4" => new Gemma4OutputParser(),
                "qwen3" => new Qwen3OutputParser(),
                "qwen35" or "qwen35moe" or "qwen3next" or "qwen3vl" or "qwen3vlmoe" => new Qwen35OutputParser(),
                "gptoss" or "gpt-oss" => new HarmonyOutputParser(),
                _ => new PassthroughOutputParser()
            };
        }

        public static bool IsAlwaysRequired(string architecture)
        {
            return architecture is "gptoss" or "gpt-oss";
        }
    }
}
