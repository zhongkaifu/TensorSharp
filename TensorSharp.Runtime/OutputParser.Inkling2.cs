// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Text;
using System.Text.Json;

namespace TensorSharp.Runtime
{
    /// <summary>
    /// Inkling 2's replies: a sequence of <c>&lt;|message_model|&gt;[name]&lt;|content_KIND|&gt;body&lt;|end_message|&gt;</c>
    /// messages, ended by <c>&lt;|content_model_end_sampling|&gt;</c> (the end-of-sequence token). A
    /// <c>content_thinking</c> body is reasoning, a <c>content_text</c> body is the answer, and a
    /// <c>content_invoke_tool_json</c> body is one call, <c>{"name": NAME, "args": {...}}</c>, whose header carries the
    /// function's name too. A body of any other kind is shown as answer text; text outside a message is too.
    /// </summary>
    public class Inkling2OutputParser : IOutputParser
    {
        private enum State { Outside, Header, Body }

        private const string ContentPrefix = "<|content_";
        private const string TagEnd = "|>";

        // Partial suffixes of these are held back while streaming, so a tag split across pieces is still recognized.
        private static readonly string[] HoldTags =
        {
            ChatTemplate.Inkling2MessageModel, ChatTemplate.Inkling2EndMessage, ChatTemplate.Inkling2EndSampling, ContentPrefix,
        };

        /// <summary>A header this long never closed: its text is shown rather than held forever.</summary>
        private const int MaxHeaderChars = 256;

        private readonly StringBuilder _buffer = new();
        private readonly StringBuilder _toolBody = new();
        private State _state;
        private string _kind = "";
        private string _name = "";
        private int _callIndex;

        public bool HasThinkingSupport => true;
        public bool HasToolSupport => true;
        public bool AlwaysRequired => true;

        /// <summary>True while the model writes a reasoning message.</summary>
        public bool IsThinkingChannelOpen => _state == State.Body && _kind == "thinking";

        public void Init(bool enableThinking, List<ToolFunction>? tools)
        {
            _buffer.Clear();
            _toolBody.Clear();
            _state = State.Outside;
            _kind = "";
            _name = "";
            _callIndex = 0;
        }

        public ParsedOutput Add(string text, bool done)
        {
            _buffer.Append(text);
            var content = new StringBuilder();
            var thinking = new StringBuilder();
            var toolText = new StringBuilder();
            var calls = new List<ToolCall>();

            bool more = true;
            while (more)
            {
                more = false;
                string buf = _buffer.ToString();
                switch (_state)
                {
                    case State.Outside:
                    {
                        int open = buf.IndexOf(ChatTemplate.Inkling2MessageModel, StringComparison.Ordinal);
                        int stop = buf.IndexOf(ChatTemplate.Inkling2EndSampling, StringComparison.Ordinal);
                        if (stop >= 0 && (open < 0 || stop < open))
                        {
                            content.Append(Stray(buf.Substring(0, stop)));
                            Keep(buf.Substring(stop + ChatTemplate.Inkling2EndSampling.Length));
                            more = _buffer.Length > 0;
                        }
                        else if (open >= 0)
                        {
                            content.Append(Stray(buf.Substring(0, open)));
                            Keep(buf.Substring(open + ChatTemplate.Inkling2MessageModel.Length));
                            _state = State.Header;
                            more = true;
                        }
                        else
                        {
                            int hold = done ? 0 : HoldBack(buf);
                            content.Append(Stray(buf.Substring(0, buf.Length - hold)));
                            Keep(buf.Substring(buf.Length - hold));
                        }
                        break;
                    }
                    case State.Header:
                    {
                        int tag = buf.IndexOf(ContentPrefix, StringComparison.Ordinal);
                        int tagEnd = tag >= 0 ? buf.IndexOf(TagEnd, tag + ContentPrefix.Length, StringComparison.Ordinal) : -1;
                        if (tagEnd >= 0)
                        {
                            _name = buf.Substring(0, tag).Trim();
                            _kind = buf.Substring(tag + ContentPrefix.Length, tagEnd - tag - ContentPrefix.Length);
                            _toolBody.Clear();
                            Keep(buf.Substring(tagEnd + TagEnd.Length));
                            _state = State.Body;
                            more = true;
                        }
                        else if (done || buf.Length > MaxHeaderChars)
                        {
                            // A header that never names its content: show what it held as answer text.
                            content.Append(buf);
                            _buffer.Clear();
                            _state = State.Outside;
                        }
                        break;
                    }
                    case State.Body:
                    {
                        int end = FirstOf(buf, out int tagLength);
                        if (end >= 0)
                        {
                            Emit(buf.Substring(0, end), content, thinking, toolText);
                            Close(calls);
                            // <|end_message|> is consumed; a following <|message_model|> or end of sampling is the next state's.
                            Keep(buf.Substring(end + (buf.AsSpan(end).StartsWith(ChatTemplate.Inkling2EndMessage) ? tagLength : 0)));
                            _state = State.Outside;
                            more = _buffer.Length > 0;
                        }
                        else if (done)
                        {
                            Emit(buf, content, thinking, toolText);
                            Close(calls);
                            _buffer.Clear();
                            _state = State.Outside;
                        }
                        else
                        {
                            int hold = HoldBack(buf);
                            Emit(buf.Substring(0, buf.Length - hold), content, thinking, toolText);
                            Keep(buf.Substring(buf.Length - hold));
                        }
                        break;
                    }
                }
            }

            var result = new ParsedOutput
            {
                Content = content.ToString(),
                Thinking = thinking.ToString(),
                ToolCallText = toolText.ToString(),
            };
            if (_state == State.Body && _kind == "invoke_tool_json" && _name.Length > 0)
                result.ToolCallName = _name;
            if (calls.Count > 0)
                result.ToolCalls = calls;
            return result;
        }

        private void Keep(string rest)
        {
            _buffer.Clear();
            _buffer.Append(rest);
        }

        // Text outside any message: framing the model was not expected to write there is dropped, the rest shown.
        private static string Stray(string text) => text.Replace(ChatTemplate.Inkling2EndMessage, "");

        private void Emit(string text, StringBuilder content, StringBuilder thinking, StringBuilder toolText)
        {
            if (text.Length == 0) return;
            switch (_kind)
            {
                case "thinking": thinking.Append(text); break;
                case "invoke_tool_json": _toolBody.Append(text); toolText.Append(text); break;
                default: content.Append(text); break;
            }
        }

        private void Close(List<ToolCall> calls)
        {
            if (_kind == "invoke_tool_json")
            {
                ToolCall? call = BuildCall(_name, _toolBody.ToString());
                if (call != null)
                {
                    call.Index = _callIndex++;
                    calls.Add(call);
                }
            }
            _toolBody.Clear();
            _kind = "";
            _name = "";
        }

        /// <summary>A call body <c>{"name": .., "args": {..}}</c> (or <c>"arguments"</c>); the header's name when the
        /// body names none. A body that is not JSON still surfaces the call, with no arguments.</summary>
        internal static ToolCall? BuildCall(string headerName, string body)
        {
            string name = headerName;
            var args = new Dictionary<string, object?>();
            try
            {
                using JsonDocument doc = JsonDocument.Parse(body.Trim());
                JsonElement root = doc.RootElement;
                if (root.ValueKind == JsonValueKind.Object)
                {
                    if (root.TryGetProperty("name", out JsonElement n) && n.ValueKind == JsonValueKind.String && !string.IsNullOrEmpty(n.GetString()))
                        name = n.GetString()!;
                    if ((root.TryGetProperty("args", out JsonElement a) || root.TryGetProperty("arguments", out a)) && a.ValueKind == JsonValueKind.Object)
                        foreach (JsonProperty p in a.EnumerateObject())
                            args[p.Name] = ChatMlOutputParser.JsonElementToObject(p.Value);
                }
            }
            catch (JsonException)
            {
            }
            return string.IsNullOrEmpty(name) ? null : new ToolCall { Name = name, Arguments = args };
        }

        /// <summary>The earliest tag that ends a message body: its own <c>&lt;|end_message|&gt;</c>, or (when the model
        /// skipped it) the next message or the end of sampling.</summary>
        private static int FirstOf(string buf, out int tagLength)
        {
            int best = -1;
            tagLength = 0;
            foreach (string tag in new[] { ChatTemplate.Inkling2EndMessage, ChatTemplate.Inkling2MessageModel, ChatTemplate.Inkling2EndSampling })
            {
                int at = buf.IndexOf(tag, StringComparison.Ordinal);
                if (at >= 0 && (best < 0 || at < best))
                {
                    best = at;
                    tagLength = tag.Length;
                }
            }
            return best;
        }

        private static int HoldBack(string buf)
        {
            int hold = 0;
            foreach (string tag in HoldTags)
                for (int i = Math.Min(tag.Length, buf.Length); i > hold; i--)
                    if (buf.AsSpan(buf.Length - i).SequenceEqual(tag.AsSpan(0, i)))
                    {
                        hold = i;
                        break;
                    }
            // An unclosed <|content_...: keep it whole until its |> arrives.
            int content = buf.LastIndexOf(ContentPrefix, StringComparison.Ordinal);
            if (content >= 0 && buf.IndexOf(TagEnd, content + ContentPrefix.Length, StringComparison.Ordinal) < 0)
                hold = Math.Max(hold, buf.Length - content);
            return hold;
        }
    }
}
