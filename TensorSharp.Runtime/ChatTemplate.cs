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
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;

namespace TensorSharp.Runtime
{
    public class ChatMessage
    {
        public string Role { get; set; } = string.Empty;
        public string? Content { get; set; } = string.Empty;
        /// <summary>
        /// Optional list of image file paths for multimodal messages.
        /// </summary>
        public List<string>? ImagePaths { get; set; }
        /// <summary>Optional source times for sampled video images, aligned with ImagePaths; ordinary images use null.</summary>
        public List<double?>? ImageTimestamps { get; set; }
        /// <summary>
        /// Optional list of audio file paths for multimodal messages.
        /// </summary>
        public List<string>? AudioPaths { get; set; }
        /// <summary>
        /// Optional list of plain-text file paths. Their contents are normally inlined
        /// into <see cref="Content"/> (for example .txt and .md); when
        /// <see cref="HasFileBackedTextAttachments"/> is true, the service first stages
        /// the complete file for tools or restores that inline representation. The paths
        /// themselves are not directly consumed by the model and also let the per-turn
        /// audit log identify the uploads belonging to this message.
        /// </summary>
        public List<string>? TextFilePaths { get; set; }
        /// <summary>
        /// The names the user knows the <see cref="TextFilePaths"/> files by, in the same
        /// order. Uploads are stored under generated names; this keeps the original
        /// "report.md" so code execution can stage the file under the name the model has
        /// seen in the conversation. Absent from older clients, in which case the stored
        /// file name stands in.
        /// </summary>
        public List<string>? TextFileNames { get; set; }
        /// <summary>
        /// True when the client deliberately kept an attached text file out of
        /// <see cref="Content"/> because the complete upload is meant to be staged for
        /// file/code tools. The Web UI service clears this on its inference clone after
        /// staging, or restores the complete inline text when no readable workspace is
        /// available, so the model is never asked to reason about unseen file contents.
        /// </summary>
        public bool HasFileBackedTextAttachments { get; set; }
        /// <summary>
        /// Every file the user attached to this message, by the stored upload name — the
        /// images and the audio as well as the documents, and only the files the user
        /// actually attached (never a frame extracted from one).
        ///
        /// <para>
        /// <see cref="TextFilePaths"/> answers a narrower question: which uploads had
        /// their text folded into <see cref="Content"/>. That made it the wrong list to
        /// stage into a program's working directory, and the gap was not academic — asked
        /// to turn a photo into a PDF, a model was handed a picture it could see and no
        /// file it could open, and spent the turn inventing paths. This list is what a
        /// tool stages, so "the file you were shown is in your working directory" is true
        /// of every attachment rather than only of the text ones.
        /// </para>
        /// </summary>
        public List<string>? AttachmentPaths { get; set; }
        /// <summary>
        /// The names the user knows <see cref="AttachmentPaths"/> by, in the same order.
        /// Uploads are stored under generated names, and a model told to open
        /// "3f2c…d1.png" has been told nothing.
        /// </summary>
        public List<string>? AttachmentNames { get; set; }
        /// <summary>
        /// True when the message contains sampled video frames. ImageTimestamps
        /// distinguishes timed frames from still images in mixed messages.
        /// </summary>
        public bool IsVideo { get; set; }
        /// <summary>
        /// Tool calls made by assistant in this message (for multi-turn tool calling).
        /// </summary>
        public List<ToolCall>? ToolCalls { get; set; }
        /// <summary>
        /// For a <c>role: "tool"</c> message, the id of the assistant tool call this
        /// result answers.
        ///
        /// <para>
        /// DeepSeek V4.1 uses this id to put parallel results back into call order
        /// before rendering them positionally. It is also required by the
        /// OpenAI wire format: a <c>tool</c> message there is rejected
        /// outright without <c>tool_call_id</c>, so anything that speaks to a real
        /// OpenAI-compatible endpoint (see <c>SkillsChatClient</c> under
        /// <c>SkillDelivery.Local</c>) has to carry the id through the conversation
        /// rather than re-derive it.
        /// </para>
        /// </summary>
        public string? ToolCallId { get; set; }
        /// <summary>
        /// Thinking/reasoning content produced by the model in this message.
        /// </summary>
        public string? Thinking { get; set; }
        /// <summary>
        /// Raw output tokens produced directly by the model when this assistant message
        /// was generated (in generation order, INCLUDING any thinking/reasoning tokens
        /// and EXCLUDING the EOS token that terminated generation).
        ///
        /// When present, the KV cache prompt renderer splices these tokens directly into
        /// the rendered token sequence instead of re-tokenizing the assistant content.
        /// This guarantees that re-rendering the conversation produces a token sequence
        /// whose prefix exactly matches what the model previously generated, enabling
        /// reliable KV cache reuse across turns.
        /// </summary>
        public List<int>? RawOutputTokens { get; set; }
        /// <summary>
        /// Exact trailing whitespace of the generation prompt that immediately
        /// preceded <see cref="RawOutputTokens"/> when this assistant round was
        /// produced. <c>null</c> means the round predates boundary tracking;
        /// <see cref="string.Empty"/> is a known boundary with no whitespace.
        ///
        /// Templates do not necessarily frame every generation the same way. Gemma 4,
        /// for example, ends an ordinary model prompt with a newline but continues
        /// directly after a tool response. Keeping this tiny piece of lossless metadata
        /// lets the KV prompt renderer reproduce either boundary without guessing from
        /// the final character of a later, structurally different render.
        /// </summary>
        public string? RawPromptTrailingWhitespace { get; set; }

        /// <summary>
        /// The architecture-specific text the generation prompt ended with when
        /// <see cref="RawOutputTokens"/> were produced — Qwen 3.5's <c>&lt;think&gt;\n</c>
        /// or <c>&lt;think&gt;\n\n&lt;/think&gt;\n\n</c>, Gemma 4's empty thought block — or
        /// null for history that was not tracked by this host. The chat template does
        /// not re-emit that framing for a past turn, so the KV renderer puts it back in
        /// front of the raw tokens; it has to be THIS turn's framing, not the current
        /// request's, or a chat whose thinking toggle changed between turns diverges
        /// from the cache at every earlier answer.
        /// </summary>
        public string? RawGenerationSuffix { get; set; }
        /// <summary>
        /// Render-only placeholder used by the Gemma 4 GGUF template adapter to put a
        /// cached raw tool-calling round at the template's tool-call position while
        /// retaining structured <see cref="ToolCalls"/> for tool-result rendering.
        /// It never belongs to persisted or wire-format history.
        /// </summary>
        internal string? RawToolCallReplayPlaceholder { get; set; }
        /// <summary>
        /// Explicit cache-control marker scoped to the whole message: a prefix
        /// cache breakpoint at the END of <see cref="Content"/>. A marker on an
        /// individual content part belongs in
        /// <see cref="ContentCacheBreakpoints"/> instead, which can express a
        /// position in the middle of the message.
        /// </summary>
        public CacheControlMarker? CacheControl { get; set; }

        /// <summary>
        /// Character offsets into <see cref="Content"/> at which a content part
        /// carried its own <c>cache_control</c> marker, in ascending order.
        /// <para>
        /// A message assembled from parts concatenates them into one
        /// <see cref="Content"/> string, so a marker on part <i>k</i> means "the
        /// cacheable prefix ends where part <i>k</i> ends", not "at the end of
        /// the message". Collapsing those onto <see cref="CacheControl"/> would
        /// push the breakpoint to the end of the concatenated content and cache
        /// more than the client marked, so the offsets are kept separately.
        /// </para>
        /// </summary>
        public List<int>? ContentCacheBreakpoints { get; set; }

        /// <summary>
        /// Record a content-part breakpoint at <paramref name="offset"/> characters
        /// into <see cref="Content"/>. Callers append in ascending order as they
        /// concatenate the parts; a repeat of the previous offset (two markers with
        /// no text between them) collapses into one.
        /// </summary>
        public void AddContentCacheBreakpoint(int offset)
        {
            ContentCacheBreakpoints ??= new List<int>();
            if (ContentCacheBreakpoints.Count > 0 &&
                ContentCacheBreakpoints[ContentCacheBreakpoints.Count - 1] == offset)
            {
                return;
            }
            ContentCacheBreakpoints.Add(offset);
        }
    }

    public static partial class ChatTemplate
    {
        /// <summary>Render the generic ChatML conversation and optional tool declarations.</summary>
        public static string RenderChatMl(List<ChatMessage> messages, bool addGenerationPrompt = true,
            List<ToolFunction>? tools = null, bool enableThinking = false)
        {
            var sb = new StringBuilder();

            if (tools != null && tools.Count > 0)
            {
                sb.Append("<|im_start|>system\nYou are a helpful assistant with access to the following functions. Use them if required -\n");
                sb.Append(JsonSerializer.Serialize(tools, new JsonSerializerOptions { WriteIndented = false }));
                sb.Append("<|im_end|>\n");
            }

            foreach (var msg in messages)
            {
                sb.Append($"<|im_start|>{msg.Role}\n");
                if (msg.ToolCalls != null && msg.ToolCalls.Count > 0)
                {
                    foreach (var tc in msg.ToolCalls)
                    {
                        string tcJson = SerializeToolCall(tc);
                        sb.Append($"\n<tool_call>\n{tcJson}\n</tool_call>");
                    }
                }
                sb.Append($"{msg.Content}<|im_end|>\n");
            }
            if (addGenerationPrompt)
            {
                sb.Append("<|im_start|>assistant\n");
                if (enableThinking)
                    sb.Append("<think>\n");
            }
            return sb.ToString();
        }

        private static string SerializeToolCall(ToolCall tc)
        {
            var obj = new Dictionary<string, object?> { ["name"] = tc.Name, ["arguments"] = tc.Arguments };
            return JsonSerializer.Serialize(obj);
        }

        /// <summary>
        /// Render the NVIDIA Nemotron 3 Nano Omni chat template.
        /// Matches the GGUF jinja template that ships with the model:
        ///   - <|im_start|>{role}\n{content}<|im_end|>\n turn framing
        ///   - For user/system: prepend "<image>\n" per image, "<video>\n" per video,
        ///     "<so_embedding>\n" per audio when those modalities are present.
        ///   - When tools is empty and there's no system message, no preamble.
        ///   - Generation prompt: "&lt;|im_start|&gt;assistant\n&lt;think&gt;\n" when thinking is on,
        ///     otherwise "&lt;|im_start|&gt;assistant\n&lt;think&gt;&lt;/think&gt;".
        /// </summary>
        public static string RenderNemotron(List<ChatMessage> messages, bool addGenerationPrompt = true,
            List<ToolFunction>? tools = null, bool enableThinking = false)
        {
            var sb = new StringBuilder();

            bool hasSystem = messages != null && messages.Count > 0 && messages[0].Role == "system";
            bool hasTools = tools != null && tools.Count > 0;
            int startIdx = 0;

            if (hasSystem)
            {
                sb.Append("<|im_start|>system\n");
                sb.Append(SanitizeNemotronContent(messages![0].Content ?? ""));
                startIdx = 1;
            }
            else if (hasTools)
            {
                sb.Append("<|im_start|>system\n");
            }

            if (hasTools)
            {
                if (hasSystem) sb.Append("\n\n");
                sb.Append(BuildNemotronToolsPreamble(tools!));
            }

            if (hasSystem || hasTools)
                sb.Append("<|im_end|>\n");

            if (messages != null)
            {
                for (int mi = startIdx; mi < messages.Count; mi++)
                {
                    var msg = messages[mi];
                    if (msg.Role == "tool")
                    {
                        // Wrap consecutive tool messages in a single user block as the template does.
                        bool prevIsTool = mi > startIdx && messages[mi - 1].Role == "tool";
                        if (!prevIsTool)
                            sb.Append("<|im_start|>user\n");
                        sb.Append("<tool_response>\n").Append(msg.Content).Append("\n</tool_response>\n");

                        bool nextIsTool = mi + 1 < messages.Count && messages[mi + 1].Role == "tool";
                        if (!nextIsTool)
                            sb.Append("<|im_end|>\n");
                    }
                    else if (msg.Role == "assistant")
                    {
                        sb.Append("<|im_start|>assistant\n");
                        string content = msg.Content ?? string.Empty;
                        if (!content.Contains("<think>") && !content.Contains("</think>"))
                            content = "<think></think>" + content;
                        sb.Append(content.Trim());
                        if (msg.ToolCalls != null && msg.ToolCalls.Count > 0)
                        {
                            foreach (var tc in msg.ToolCalls)
                                AppendNemotronToolCall(sb, tc);
                        }
                        sb.Append("<|im_end|>\n");
                    }
                    else // "user" or "system" appearing later
                    {
                        sb.Append("<|im_start|>").Append(msg.Role).Append('\n');
                        AppendNemotronUserContent(sb, msg);
                        sb.Append("<|im_end|>\n");
                    }
                }
            }

            if (addGenerationPrompt)
            {
                if (enableThinking)
                    sb.Append("<|im_start|>assistant\n<think>\n");
                else
                    sb.Append("<|im_start|>assistant\n<think></think>");
            }

            return sb.ToString();
        }

        private static void AppendNemotronUserContent(StringBuilder sb, ChatMessage msg)
        {
            int imgCount = msg.ImagePaths?.Count ?? 0;
            int audioCount = msg.AudioPaths?.Count ?? 0;

            string textContent = msg.Content ?? string.Empty;
            if (textContent.Contains("<image>")) imgCount = 0;
            if (textContent.Contains("<so_embedding>")) audioCount = 0;

            // Both single images and video frames map onto the per-image format. Video
            // frames are passed in as imgCount frame paths from the caller.
            if (imgCount == 1)
            {
                sb.Append("<image>\n");
            }
            else if (imgCount > 1)
            {
                for (int i = 0; i < imgCount; i++)
                    sb.Append("<image ").Append(i + 1).Append("><image>");
                sb.Append('\n');
            }

            for (int i = 0; i < audioCount; i++) sb.Append("<so_embedding>\n");

            sb.Append(SanitizeNemotronContent(textContent.TrimStart('\n')));
        }

        private static string SanitizeNemotronContent(string content)
        {
            // Mirror the jinja sanitization: strip /think and /no_think directives but
            // keep proper <think>/</think> XML tags intact.
            return content
                .Replace("</think>", "<_end_think>")
                .Replace("/think", "")
                .Replace("/no_think", "")
                .Replace("<_end_think>", "</think>")
                .Trim();
        }

        private static string BuildNemotronToolsPreamble(List<ToolFunction> tools)
        {
            var sb = new StringBuilder();
            sb.Append("# Tools\n\nYou have access to the following functions:\n\n<tools>");
            foreach (var tool in tools)
            {
                sb.Append("\n<function>\n<name>").Append(tool.Name).Append("</name>");
                if (!string.IsNullOrEmpty(tool.Description))
                    sb.Append("\n<description>").Append(tool.Description.Trim()).Append("</description>");
                sb.Append("\n<parameters>");
                if (tool.Parameters != null)
                {
                    foreach (var kv in tool.Parameters)
                    {
                        sb.Append("\n<parameter>");
                        sb.Append("\n<name>").Append(kv.Key).Append("</name>");
                        if (!string.IsNullOrEmpty(kv.Value.Type))
                            sb.Append("\n<type>").Append(kv.Value.Type).Append("</type>");
                        if (!string.IsNullOrEmpty(kv.Value.Description))
                            sb.Append("\n<description>").Append(kv.Value.Description.Trim()).Append("</description>");
                        if (kv.Value.Enum != null && kv.Value.Enum.Count > 0)
                            sb.Append("\n<enum>").Append(JsonSerializer.Serialize(kv.Value.Enum)).Append("</enum>");
                        sb.Append("\n</parameter>");
                    }
                }
                if (tool.Required != null && tool.Required.Count > 0)
                    sb.Append("\n<required>").Append(JsonSerializer.Serialize(tool.Required)).Append("</required>");
                sb.Append("\n</parameters>\n</function>");
            }
            sb.Append("\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n");
            sb.Append("<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n");
            sb.Append("<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n");
            sb.Append("<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>");
            return sb.ToString();
        }

        private static void AppendNemotronToolCall(StringBuilder sb, ToolCall tc)
        {
            sb.Append("<tool_call>\n<function=").Append(tc.Name).Append(">\n");
            if (tc.Arguments != null)
            {
                foreach (var kv in tc.Arguments)
                {
                    sb.Append("<parameter=").Append(kv.Key).Append(">\n");
                    string val = kv.Value is string s ? s : JsonSerializer.Serialize(kv.Value);
                    sb.Append(val).Append("\n</parameter>\n");
                }
            }
            sb.Append("</function>\n</tool_call>\n");
        }

        /// <summary>
        /// Render Qwen3.5 template with optional image support.
        /// Matches the GGUF built-in chat template: for each image in a message,
        /// inserts <|vision_start|><|image_pad|><|vision_end|> markers.
        /// The single <|image_pad|> token is later expanded to N tokens based on image dimensions.
        /// </summary>
        public static string RenderQwen35(List<ChatMessage> messages, bool addGenerationPrompt = true,
            bool enableThinking = false, List<ToolFunction>? tools = null)
        {
            var sb = new StringBuilder();

            bool hasSystem = messages.Count > 0 && messages[0].Role == "system";
            int startIdx = 0;

            if (tools != null && tools.Count > 0)
            {
                sb.Append("<|im_start|>system\n");
                sb.Append("# Tools\n\nYou have access to the following functions:\n\n<tools>");
                foreach (var tool in tools)
                {
                    sb.Append("\n");
                    // Exactly what the GGUF template's `{{ tool | tojson }}` prints, from
                    // the same declaration dictionary the Jinja context is built from.
                    // This renderer used to pretty-print with System.Text.Json instead,
                    // so a prompt rendered here (thinking off) and one rendered by the
                    // template (thinking on) differed from the first tool onwards, and
                    // the KV cache could share nothing past that point between them.
                    sb.Append(Jinja2Template.ToJson(BuildToolDeclaration(tool)));
                }
                sb.Append("\n</tools>\n\n");
                sb.Append("If you choose to call a function ONLY reply in the following format with NO suffix:\n\n");
                sb.Append("<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n");
                sb.Append("<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n");
                sb.Append("<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n");
                sb.Append("- Required parameters MUST be specified\n");
                sb.Append("- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n");
                sb.Append("- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>");

                if (hasSystem)
                {
                    string sysContent = (messages[0].Content ?? "").Trim();
                    if (sysContent.Length > 0)
                    {
                        sb.Append("\n\n");
                        sb.Append(sysContent);
                    }
                    startIdx = 1;
                }
                sb.Append("<|im_end|>\n");
            }
            else if (hasSystem)
            {
                sb.Append($"<|im_start|>system\n{(messages[0].Content ?? "").Trim()}<|im_end|>\n");
                startIdx = 1;
            }

            for (int mi = startIdx; mi < messages.Count; mi++)
            {
                var msg = messages[mi];
                bool lastMessage = mi == messages.Count - 1;
                bool prefill = lastMessage && msg.Role == "assistant";

                if (msg.Role == "assistant")
                {
                    sb.Append("<|im_start|>assistant\n");
                    var (reasoningContent, content) = SplitQwen35AssistantContent(msg);
                    if (!enableThinking || !string.IsNullOrEmpty(reasoningContent))
                    {
                        sb.Append("<think>\n");
                        if (!string.IsNullOrEmpty(reasoningContent))
                            sb.Append(reasoningContent);
                        sb.Append("\n</think>\n\n");
                    }

                    if (msg.ToolCalls != null && msg.ToolCalls.Count > 0)
                    {
                        if (!string.IsNullOrEmpty(content))
                            sb.Append(content);
                        for (int j = 0; j < msg.ToolCalls.Count; j++)
                        {
                            var tc = msg.ToolCalls[j];
                            if (j == 0 && !string.IsNullOrWhiteSpace(content))
                                sb.Append("\n\n");
                            else if (j > 0)
                                sb.Append("\n");
                            sb.Append($"<tool_call>\n<function={tc.Name}>\n");
                            if (tc.Arguments != null)
                            {
                                foreach (var kv in tc.Arguments)
                                {
                                    sb.Append($"<parameter={kv.Key}>\n");
                                    sb.Append(FormatQwen35ToolCallArg(kv.Value));
                                    sb.Append("\n</parameter>\n");
                                }
                            }
                            sb.Append("</function>\n</tool_call>");
                        }
                    }
                    else
                    {
                        sb.Append(content);
                    }
                    if (!prefill)
                        sb.Append("<|im_end|>\n");
                }
                else if (msg.Role == "tool")
                {
                    bool isFirstTool = mi == startIdx || messages[mi - 1].Role != "tool";
                    bool isLastTool = mi == messages.Count - 1 || messages[mi + 1].Role != "tool";
                    if (isFirstTool)
                        sb.Append("<|im_start|>user");
                    sb.Append($"\n<tool_response>\n{(msg.Content ?? "").Trim()}\n</tool_response>");
                    if (isLastTool)
                        sb.Append("<|im_end|>\n");
                }
                else
                {
                    sb.Append($"<|im_start|>{msg.Role}\n");
                    if (msg.ImagePaths != null && msg.ImagePaths.Count > 0)
                    {
                        foreach (var _ in msg.ImagePaths)
                            sb.Append("<|vision_start|><|image_pad|><|vision_end|>");
                    }
                    sb.Append($"{(msg.Content ?? "").Trim()}<|im_end|>\n");
                }

                if (lastMessage && !prefill)
                {
                    sb.Append("<|im_start|>assistant\n");
                    if (enableThinking)
                        sb.Append("<think>\n");
                    else
                        sb.Append("<think>\n\n</think>\n\n");
                }
            }
            return sb.ToString();
        }

        private static (string reasoningContent, string content) SplitQwen35AssistantContent(ChatMessage msg)
        {
            string content = msg?.Content ?? "";
            string reasoningContent = msg?.Thinking ?? "";

            if (!string.IsNullOrEmpty(reasoningContent))
                return (reasoningContent.Trim(), content);

            int closeIdx = content.IndexOf("</think>", StringComparison.Ordinal);
            if (closeIdx < 0)
                return ("", content);

            int openIdx = content.LastIndexOf("<think>", closeIdx, StringComparison.Ordinal);
            if (openIdx < 0)
                return ("", content);

            int reasoningStart = openIdx + "<think>".Length;
            reasoningContent = content.Substring(reasoningStart, closeIdx - reasoningStart).Trim();
            content = content.Substring(closeIdx + "</think>".Length).TrimStart('\r', '\n');
            return (reasoningContent, content);
        }

        private static Dictionary<string, object> BuildToolParamsDict(ToolFunction tool)
            => (Dictionary<string, object>)((Dictionary<string, object>)BuildToolDeclaration(tool)["function"])["parameters"];

        /// <summary>
        /// The OpenAI-shaped declaration of one tool, as the Jinja context hands it to a
        /// GGUF template and as the hardcoded renderers print it. One builder, so a
        /// template path and a hardcoded path cannot describe the same tool differently.
        /// </summary>
        internal static Dictionary<string, object> BuildToolDeclaration(ToolFunction t)
        {
            var props = new Dictionary<string, object>();
            if (t.Parameters != null)
            {
                foreach (var kv in t.Parameters)
                {
                    var pDict = new Dictionary<string, object> { ["type"] = kv.Value.Type ?? "string" };
                    if (!string.IsNullOrEmpty(kv.Value.Description))
                        pDict["description"] = kv.Value.Description;
                    if (kv.Value.Enum != null && kv.Value.Enum.Count > 0)
                        pDict["enum"] = new List<object>(kv.Value.Enum.Select(e => (object)e));
                    props[kv.Key] = pDict;
                }
            }

            var paramsDict = new Dictionary<string, object>
            {
                ["type"] = "object",
                ["properties"] = props,
            };

            if (t.Required != null && t.Required.Count > 0)
                paramsDict["required"] = new List<object>(t.Required.Select(r => (object)r));
            else if (t.Parameters != null && t.Parameters.Count > 0)
                paramsDict["required"] = new List<object>(t.Parameters.Keys.Select(k => (object)k));

            return new Dictionary<string, object>
            {
                ["type"] = "function",
                ["function"] = new Dictionary<string, object>
                {
                    ["name"] = t.Name,
                    ["description"] = t.Description ?? "",
                    ["parameters"] = paramsDict
                }
            };
        }

        private static string FormatQwen35ToolCallArg(object value)
        {
            if (value is string s) return s;
            if (value is bool b) return b.ToString().ToLowerInvariant();
            if (value is Dictionary<string, object> || value is List<object>)
                return JsonSerializer.Serialize(value);
            return value?.ToString() ?? "null";
        }

        /// <summary>
        /// Render a chat prompt using the model's built-in GGUF template if available,
        /// otherwise fall back to hardcoded architecture-specific templates.
        /// Multimodal tokens (image/audio/video) are injected into message content
        /// before rendering so both Jinja2 and hardcoded paths produce correct output.
        /// </summary>
        public static string RenderFromGgufTemplate(string template, List<ChatMessage> messages,
            bool addGenerationPrompt = true, string? architecture = null,
            List<ToolFunction>? tools = null, bool enableThinking = false)
        {
            // Several families ship a template built on Jinja features the lightweight
            // engine renders inconsistently (recursive macros, namespaces, tojson, dict
            // walkers), and their formats are simple enough to render directly. Which
            // ones, and under what condition, is declared once per family in
            // ChatProtocolRegistry rather than as a chain of name comparisons here.
            var protocol = ChatProtocolRegistry.For(architecture);
            if (protocol?.PreferOwnRenderer != null
                && protocol.PreferOwnRenderer(new ChatRenderRequest(
                    messages, addGenerationPrompt, architecture, tools, enableThinking)))
            {
                return RenderHardcoded(messages, addGenerationPrompt, architecture, tools, enableThinking);
            }

            if (!string.IsNullOrWhiteSpace(template))
            {
                try
                {
                    var preprocessed = InjectMultimodalTokens(messages, architecture);
                    string effectiveTemplate = EnableGemma4CachedToolReasoningReplay(
                        template, preprocessed, architecture);
                    var jinja = new Jinja2Template(effectiveTemplate);
                    var context = BuildJinja2Context(
                        preprocessed, addGenerationPrompt, tools, enableThinking, architecture);
                    string result = jinja.Render(context);
                    // Gemma's generation boundary is part of its published template:
                    // preserve its newline and any channel prefix exactly. Adding an
                    // empty thought block can make E4B continue unmarked reasoning.
                    if (architecture != "gemma4" || !addGenerationPrompt)
                        result = result.TrimEnd();
                    if (result.Length > 0)
                    {
                        // Defensive correctness guard: a lightweight Jinja engine that
                        // cannot fully evaluate a feature the template relies on can
                        // SILENTLY render an empty body for a message — most insidiously
                        // when the template captures a message into a block-form
                        // {% set captured %}...{% endset %} (as the Gemma 4 template does
                        // for every message's content). When that capture mis-renders, the
                        // user's question is dropped from the prompt entirely and the model
                        // generates text unrelated to the prompt. Detect that here and fall
                        // back to the hardcoded template, which always emits message content.
                        if (!RenderedContainsLastUserText(result, messages))
                        {
                            WarnTemplateFallbackOnce(architecture, "dropped-user-message",
                                "the rendered prompt dropped the last user message (lightweight-Jinja-engine guard).");
                        }
                        else
                        {
                            result = StripReasoningEndSentinel(result);
                            if (IsQwen35Family(architecture) && addGenerationPrompt)
                                result = enableThinking
                                    ? EnsureQwen35ThinkOpen(result)
                                    : EnsureQwen35ThinkClosed(result);
                            Console.Error.WriteLine($"[ChatTemplate] Jinja2 rendering succeeded for '{architecture}', prompt length={result.Length}");
                            return result;
                        }
                    }
                }
                catch (Exception ex)
                {
                    WarnTemplateFallbackOnce(architecture, "jinja-error",
                        $"Jinja2 rendering threw {ex.GetType().Name}: {ex.Message}.");
                }
            }

            Console.Error.WriteLine($"[ChatTemplate] Using hardcoded template for '{architecture}'");
            return RenderHardcoded(messages, addGenerationPrompt, architecture, tools, enableThinking);
        }

        // (architecture, reason-kind) pairs whose Jinja→hardcoded fallback has
        // already been reported, so the per-request render path warns once
        // instead of per prompt. ChatTemplate is a static class reachable from
        // the CLI, server, and tests with no ILogger to inject — routing this
        // through Microsoft.Extensions.Logging would mean replumbing every
        // RenderFromGgufTemplate call site — so Console.Error is the channel.
        private static readonly HashSet<string> TemplateFallbackReported = new(StringComparer.Ordinal);

        private static void WarnTemplateFallbackOnce(string? architecture, string kind, string detail)
        {
            lock (TemplateFallbackReported)
            {
                if (!TemplateFallbackReported.Add($"{architecture}|{kind}")) return;
            }
            Console.Error.WriteLine(
                $"[ChatTemplate] The model's GGUF Jinja chat template for '{architecture}' was abandoned: {detail} " +
                "Affected requests render with the built-in hardcoded template instead; prompt formatting may " +
                "differ from the model's shipped template. Reported once per architecture.");
        }

        /// <summary>
        /// Verify the rendered prompt still contains the most recent user message's
        /// text. Every supported chat template emits a user message's content
        /// verbatim (at most trimmed), so a missing substring means the renderer
        /// dropped the user's question — see the call site for why that matters.
        /// Returns true (i.e. "looks fine") when there is no user text to verify
        /// (e.g. an image-only turn).
        /// </summary>
        private static bool RenderedContainsLastUserText(string rendered, List<ChatMessage> messages)
        {
            if (messages == null)
                return true;

            string? lastUserText = null;
            for (int i = messages.Count - 1; i >= 0; i--)
            {
                if (messages[i] != null && messages[i].Role == "user")
                {
                    lastUserText = messages[i].Content;
                    break;
                }
            }

            string needle = NormalizeNewlines(lastUserText).Trim();
            if (needle.Length == 0)
                return true;

            return NormalizeNewlines(rendered).Contains(needle, StringComparison.Ordinal);
        }

        private static string NormalizeNewlines(string? s)
        {
            if (string.IsNullOrEmpty(s))
                return string.Empty;
            return s.Replace("\r\n", "\n").Replace('\r', '\n');
        }

        private static string RenderHardcoded(List<ChatMessage> messages,
            bool addGenerationPrompt, string? architecture,
            List<ToolFunction>? tools = null, bool enableThinking = false)
        {
            var request = new ChatRenderRequest(messages, addGenerationPrompt, architecture, tools, enableThinking);
            var render = ChatProtocolRegistry.For(architecture)?.Render;

            // No purpose-built renderer: generic ChatML, which is also what an
            // unrecognised architecture gets.
            return render != null
                ? render(request)
                : RenderChatMl(messages, addGenerationPrompt, tools, enableThinking);
        }

        private const string GlmToolsHeader =
            "<|system|>\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n" +
            "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n";

        private const string GlmToolsFooter =
            "</tools>\n\nFor each function call, output the function name and arguments within the following " +
            "XML format:\n<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}" +
            "</arg_value><arg_key>{arg-key-2}</arg_key><arg_value>{arg-value-2}</arg_value>...</tool_call>";

        /// <summary>
        /// GLM-5.x (glm-dsa) chat format, mirroring the template shipped in the
        /// GGUF:
        /// <code>
        /// [gMASK]&lt;sop&gt;[&lt;|system|&gt;Reasoning Effort: Max][tools block]
        /// &lt;|user|&gt;...&lt;|assistant|&gt;&lt;think&gt;...&lt;/think&gt;...
        /// &lt;|observation|&gt;&lt;tool_response&gt;...&lt;/tool_response&gt;
        /// </code>
        /// <para>The reasoning-effort system line is what turns thinking ON for this
        /// family: the template emits it whenever thinking is not explicitly
        /// disabled, and the generation prompt then opens a <c>&lt;think&gt;</c>
        /// block the model is expected to close itself. With thinking off the
        /// prompt closes the block immediately (<c>&lt;think&gt;&lt;/think&gt;</c>)
        /// so the model answers directly.</para>
        /// <para>Past-turn reasoning is dropped, matching the template's default
        /// (<c>clear_thinking</c>): only the turn currently being generated keeps
        /// its reasoning.</para>
        /// </summary>
        public static string RenderGlmDsa(List<ChatMessage> messages, bool addGenerationPrompt = true,
            bool enableThinking = false, List<ToolFunction>? tools = null)
        {
            var sb = new StringBuilder();
            sb.Append("[gMASK]<sop>");

            if (enableThinking)
                sb.Append("<|system|>Reasoning Effort: Max");

            if (tools != null && tools.Count > 0)
            {
                sb.Append(GlmToolsHeader);
                foreach (var tool in tools)
                    sb.Append(ToolFunctionToJson(tool)).Append('\n');
                sb.Append(GlmToolsFooter);
            }

            bool prevWasTool = false;
            foreach (var m in messages)
            {
                switch (m.Role)
                {
                    case "system":
                        sb.Append("<|system|>").Append(m.Content ?? "");
                        prevWasTool = false;
                        break;
                    case "user":
                    case "developer":
                        sb.Append("<|user|>").Append(m.Content ?? "");
                        prevWasTool = false;
                        break;
                    case "tool":
                        // One <|observation|> opens a RUN of tool results.
                        if (!prevWasTool)
                            sb.Append("<|observation|>");
                        sb.Append("<tool_response>").Append(m.Content ?? "").Append("</tool_response>");
                        prevWasTool = true;
                        break;
                    case "assistant":
                    {
                        sb.Append("<|assistant|>\n");
                        string content = m.Content ?? string.Empty;
                        int close = content.IndexOf("</think>", StringComparison.Ordinal);
                        if (close >= 0)
                        {
                            // Historical reasoning is dropped; the empty block is
                            // still emitted because the model was trained on it.
                            content = content.Substring(close + "</think>".Length);
                        }
                        sb.Append("<think></think>");
                        content = content.Trim();
                        if (content.Length > 0)
                            sb.Append(content);
                        prevWasTool = false;
                        break;
                    }
                }
            }

            if (addGenerationPrompt)
                sb.Append("<|assistant|>").Append(enableThinking ? "<think>" : "<think></think>");

            return sb.ToString();
        }

        /// <summary>
        /// GLM-5.3-Flash (glm5next) chat format, mirroring the template shipped in
        /// the GGUF. Differences from GLM-5.2:
        /// <list type="bullet">
        /// <item>The reasoning-effort system line is ALWAYS emitted (the template
        /// defaults <c>reasoning_effort</c> to <c>max</c>); this family has no
        /// thinking-off prompt shape, so <paramref name="enableThinking"/> only
        /// decides whether the generation prompt's <c>&lt;think&gt;</c> block is
        /// left open or closed immediately.</item>
        /// <item><c>clear_thinking</c> defaults to FALSE: historical assistant
        /// turns KEEP their reasoning when the message still carries it.</item>
        /// <item>No newline after the <c>&lt;|assistant|&gt;</c> tag.</item>
        /// </list>
        /// </summary>
        public static string RenderGlm5Next(List<ChatMessage> messages, bool addGenerationPrompt = true,
            bool enableThinking = true, List<ToolFunction>? tools = null)
        {
            var sb = new StringBuilder();
            sb.Append("[gMASK]<sop>");
            sb.Append("<|system|>Reasoning Effort: Max");

            if (tools != null && tools.Count > 0)
            {
                sb.Append(GlmToolsHeader);
                foreach (var tool in tools)
                    sb.Append(ToolFunctionToJson(tool)).Append('\n');
                sb.Append(GlmToolsFooter);
            }

            bool prevWasTool = false;
            foreach (var m in messages)
            {
                switch (m.Role)
                {
                    case "system":
                        sb.Append("<|system|>").Append(m.Content ?? "");
                        prevWasTool = false;
                        break;
                    case "user":
                    case "developer":
                        sb.Append("<|user|>").Append(m.Content ?? "");
                        prevWasTool = false;
                        break;
                    case "tool":
                        // One <|observation|> opens a RUN of tool results.
                        if (!prevWasTool)
                            sb.Append("<|observation|>");
                        sb.Append("<tool_response>").Append(m.Content ?? "").Append("</tool_response>");
                        prevWasTool = true;
                        break;
                    case "assistant":
                    {
                        sb.Append("<|assistant|>");
                        string content = m.Content ?? string.Empty;
                        int open = content.IndexOf("<think>", StringComparison.Ordinal);
                        int close = content.IndexOf("</think>", StringComparison.Ordinal);
                        if (close >= 0)
                        {
                            // clear_thinking defaults false: past reasoning stays.
                            string reasoning = content.Substring(
                                open >= 0 ? open + "<think>".Length : 0,
                                (close) - (open >= 0 ? open + "<think>".Length : 0));
                            sb.Append("<think>").Append(reasoning).Append("</think>");
                            content = content.Substring(close + "</think>".Length);
                        }
                        else
                        {
                            sb.Append("<think></think>");
                        }
                        content = content.Trim();
                        if (content.Length > 0)
                            sb.Append(content);
                        prevWasTool = false;
                        break;
                    }
                }
            }

            if (addGenerationPrompt)
                sb.Append("<|assistant|>").Append(enableThinking ? "<think>" : "<think></think>");

            return sb.ToString();
        }

        /// <summary>
        /// DeepSeek V4 chat format (mirrors models/templates/deepseek-ai-DeepSeek-V4.jinja):
        /// leading system prompt(s) concatenated after BOS, then
        /// &lt;｜User｜&gt;...&lt;｜Assistant｜&gt;&lt;/think&gt;...&lt;｜end▁of▁sentence｜&gt; turns.
        /// Consecutive user/tool messages merge into one &lt;｜User｜&gt; block. The BOS
        /// token itself is prepended by the tokenizer (add_bos), not emitted here.
        /// </summary>
        public static string RenderDeepSeek4(List<ChatMessage> messages, bool addGenerationPrompt = true,
            bool enableThinking = false, List<ToolFunction>? tools = null)
        {
            var sb = new StringBuilder();

            bool firstSystem = true;
            foreach (var m in messages)
            {
                if (m.Role != "system")
                    continue;
                if (!firstSystem)
                    sb.Append("\n\n");
                sb.Append(m.Content ?? "");
                firstSystem = false;
            }

            // Tools ride on the system prompt, exactly as the model's own template
            // builds it: header (which teaches the DSML call syntax), one JSON
            // schema per function, then the footer. A system message that exists
            // but is empty still contributes its "\n\n" separator.
            if (tools != null && tools.Count > 0)
            {
                if (!firstSystem)
                    sb.Append("\n\n");
                sb.Append(DeepSeek4ToolsHeader);
                foreach (var tool in tools)
                    sb.Append(ToolFunctionToJson(tool)).Append('\n');
                sb.Append(DeepSeek4ToolsFooter);
            }

            bool inUser = false;
            foreach (var m in messages)
            {
                switch (m.Role)
                {
                    case "user":
                    case "developer":
                        sb.Append(inUser ? "\n\n" : "<｜User｜>");
                        inUser = true;
                        sb.Append(m.Content ?? "");
                        break;
                    case "tool":
                        sb.Append(inUser ? "\n\n" : "<｜User｜>");
                        inUser = true;
                        sb.Append("<tool_result>").Append(m.Content ?? "").Append("</tool_result>");
                        break;
                    case "assistant":
                        inUser = false;
                        sb.Append("<｜Assistant｜>");
                        // Past-turn reasoning is dropped (drop_thinking behavior);
                        // non-thinking turns start with a closed think block.
                        sb.Append(enableThinking ? "<think></think>" : "</think>");
                        sb.Append(m.Content ?? "");
                        AppendDeepSeek4ToolCalls(sb, m.ToolCalls);
                        sb.Append("<｜end▁of▁sentence｜>");
                        break;
                }
            }

            if (addGenerationPrompt)
            {
                sb.Append("<｜Assistant｜>");
                sb.Append(enableThinking ? "<think>" : "</think>");
            }

            return sb.ToString();
        }

        // --- DeepSeek V4 tool calling (DSML markup) -------------------------
        //
        // DeepSeek V4 does not use a JSON tool-call block. It marks calls up with
        // its own "DSML" tags, and the model only emits them when the system
        // prompt has taught it the syntax — which is what DeepSeek4ToolsHeader
        // is. Both strings are transcribed from the GGUF's own chat template
        // (`tools_header` / `tools_footer`); keep them byte-identical to it, the
        // model was trained on this exact wording.
        internal const string DsmlToken = "｜DSML｜";

        private const string DeepSeek4ToolsHeader =
            "## Tools\n\nYou have access to a set of tools to help answer the user's question. " +
            "You can invoke tools by writing a \"<" + DsmlToken + "tool_calls>\" block like the following:\n\n" +
            "<" + DsmlToken + "tool_calls>\n" +
            "<" + DsmlToken + "invoke name=\"$TOOL_NAME\">\n" +
            "<" + DsmlToken + "parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</" + DsmlToken + "parameter>\n" +
            "...\n" +
            "</" + DsmlToken + "invoke>\n" +
            "<" + DsmlToken + "invoke name=\"$TOOL_NAME2\">\n" +
            "...\n" +
            "</" + DsmlToken + "invoke>\n" +
            "</" + DsmlToken + "tool_calls>\n\n" +
            "String parameters should be specified as is and set `string=\"true\"`. For all other types " +
            "(numbers, booleans, arrays, objects), pass the value in JSON format and set `string=\"false\"`.\n\n" +
            "If thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside " +
            "<think>...</think> BEFORE any tool calls or final response.\n\n" +
            "Otherwise, output directly after </think> with tool calls or final response.\n\n" +
            "### Available Tool Schemas\n\n";

        private const string DeepSeek4ToolsFooter =
            "\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.\n";

        /// <summary>
        /// Render an assistant turn's tool calls back into DSML, so a follow-up
        /// request that replays the conversation shows the model its own calls in
        /// the form it produced them.
        /// </summary>
        private static void AppendDeepSeek4ToolCalls(StringBuilder sb, List<ToolCall>? toolCalls)
        {
            if (toolCalls == null || toolCalls.Count == 0)
                return;

            sb.Append("\n\n<").Append(DsmlToken).Append("tool_calls>\n");
            foreach (var call in toolCalls)
            {
                sb.Append('<').Append(DsmlToken).Append("invoke name=\"").Append(call.Name).Append("\">\n");
                if (call.Arguments != null)
                {
                    foreach (var kv in call.Arguments)
                    {
                        // `string="true"` means the value is written raw; anything
                        // else is JSON, which is how the model is told to read it.
                        bool isString = kv.Value is string;
                        sb.Append('<').Append(DsmlToken).Append("parameter name=\"").Append(kv.Key)
                          .Append("\" string=\"").Append(isString ? "true" : "false").Append("\">")
                          .Append(isString ? (string)kv.Value : JsonSerializer.Serialize(kv.Value))
                          .Append("</").Append(DsmlToken).Append("parameter>\n");
                    }
                }
                sb.Append("</").Append(DsmlToken).Append("invoke>\n");
            }
            sb.Append("</").Append(DsmlToken).Append("tool_calls>");
        }

        /// <summary>The tool's JSON schema, in the shape the template's
        /// `tool['function'] | tojson` produces.</summary>
        private static string ToolFunctionToJson(ToolFunction tool)
        {
            var fn = new Dictionary<string, object?> { ["name"] = tool.Name };
            if (!string.IsNullOrEmpty(tool.Description))
                fn["description"] = tool.Description;

            var props = new Dictionary<string, object?>();
            foreach (var kv in tool.Parameters ?? new Dictionary<string, ToolParameter>())
            {
                var p = new Dictionary<string, object?> { ["type"] = kv.Value.Type ?? "string" };
                if (!string.IsNullOrEmpty(kv.Value.Description))
                    p["description"] = kv.Value.Description;
                if (kv.Value.Enum != null && kv.Value.Enum.Count > 0)
                    p["enum"] = kv.Value.Enum;
                props[kv.Key] = p;
            }

            fn["parameters"] = new Dictionary<string, object?>
            {
                ["type"] = "object",
                ["properties"] = props,
                ["required"] = tool.Required ?? new List<string>(),
            };
            return JsonSerializer.Serialize(fn);
        }

        private static bool IsQwen35Family(string? architecture)
        {
            return architecture == "qwen35" ||
                   architecture == "qwen35moe" ||
                   architecture == "qwen3next" ||
                   architecture == "qwen3vl" ||
                   architecture == "qwen3vlmoe";
        }

        /// <summary>
        /// A private-use marker for where a re-rendered reasoning block ends, so the
        /// template's own closing newline can be taken back off. Never appears in text a
        /// model or a user can write.
        /// </summary>
        internal const string ReasoningEndSentinel = "\uE000TS_REASONING_END\uE000";

        // The canonical Gemma 4 template only emits an assistant tool round's
        // `reasoning` when that round follows the LAST user message. That is right for
        // ordinary serialized history, but not for TensorSharp's host-cached transcript:
        // after the user sends the next turn, the previously generated tool round is now
        // before the last user even though its exact thought-channel tokens are still in
        // the live KV cache. Mark only those cached rounds and let them bypass the
        // last-user gate. Keeping `tool_calls` in the condition and on the message is
        // essential: the same canonical template uses it to fold the following tool
        // result into this model turn.
        private const string Gemma4CachedToolReasoningMarker =
            "_tensorsharp_cached_tool_reasoning";
        private const string Gemma4RawToolCallReplayMarker =
            "_tensorsharp_raw_tool_call_replay";
        private const string Gemma4CurrentTurnReasoningCondition =
            "thinking_text and loop.index0 > ns_turn.last_user_idx and message.get('tool_calls')";
        private const string Gemma4ToolCallsBranch =
            "{%- if message['tool_calls'] -%}";
        private const string Gemma4ToolResponseMacro =
            "{%- macro format_tool_response_block(tool_name, response) -%}";
        private const string Gemma4MessageStateNamespace =
            "{%- set ns = namespace(prev_message_type=None) -%}";
        private const string Gemma4ToolCallStateAssignment =
            "{%- set ns.prev_message_type = 'tool_call' -%}";
        private const string Gemma4LoopMessagesIteration =
            "{%- for message in loop_messages -%}";
        private const string Gemma4NonToolBranch =
            "{%- if message['role'] != 'tool' -%}";
        private const string Gemma4ForwardToolCallBranch =
            "{%- elif message.get('tool_calls') -%}";
        private const string Gemma4ForwardToolResultScan =
            "{%- for k in range(loop.index0 + 1, loop_messages | length) -%}";
        private const string Gemma4CachedToolReasoningCondition =
            "thinking_text and (loop.index0 > ns_turn.last_user_idx or " +
            "message.get('" + Gemma4CachedToolReasoningMarker + "')) and " +
            "message.get('tool_calls')";
        private const string Gemma4RawToolReplayBranch =
            "{%- if message.get('" + Gemma4RawToolCallReplayMarker + "') -%}" +
            "{{- message.get('" + Gemma4RawToolCallReplayMarker + "') -}}" +
            "{%- set ns.prev_message_type = 'tool_call' -%}" +
            "{%- elif message['tool_calls'] -%}";

        // The second generation of the Gemma 4 template (shipped with the E2B/E4B
        // GGUFs of mid-2026). Same structure, three differences that the recognizer
        // has to know: the tool-call branch reads `message.get('tool_calls')`, the
        // per-message state namespace carries a second field, and the reasoning gate
        // has its own `preserve_thinking` switch for tool-call turns —
        //   thinking_gate = (loop.index0 > ns_turn.last_user_idx)
        //                   or (preserve_thinking and message.get('tool_calls'))
        // which is exactly the cached-round bypass the first generation had to be
        // patched for. Without recognizing it, every agentic turn on these models
        // re-prefilled the whole conversation at its first tool call: the raw round
        // could not be replayed, and the structural re-render dropped the empty thought
        // block the generation prompt had put in front of the call.
        private const string Gemma4V2ToolCallsBranch =
            "{%- if message.get('tool_calls') -%}";
        private const string Gemma4V2MessageStateNamespace =
            "{%- set ns = namespace(prev_message_type=None, prev_non_tool_role=None) -%}";
        private const string Gemma4V2CurrentTurnReasoningGate =
            "(preserve_thinking and message.get('tool_calls'))";
        private const string Gemma4V2CachedToolReasoningGate =
            "((preserve_thinking or message.get('" + Gemma4CachedToolReasoningMarker + "')) " +
            "and message.get('tool_calls'))";
        private const string Gemma4V2RawToolReplayBranch =
            "{%- if message.get('" + Gemma4RawToolCallReplayMarker + "') -%}" +
            "{{- message.get('" + Gemma4RawToolCallReplayMarker + "') -}}" +
            "{%- set ns.prev_message_type = 'tool_call' -%}" +
            "{%- elif message.get('tool_calls') -%}";

        private sealed class Gemma4ReplayTemplateVariants
        {
            public Gemma4ReplayTemplateVariants(string template)
            {
                bool shared =
                    template.Contains(Gemma4ToolResponseMacro, StringComparison.Ordinal)
                    && template.Contains(Gemma4ToolCallStateAssignment, StringComparison.Ordinal)
                    && template.Contains(Gemma4LoopMessagesIteration, StringComparison.Ordinal)
                    && template.Contains(Gemma4NonToolBranch, StringComparison.Ordinal)
                    && template.Contains(Gemma4ForwardToolCallBranch, StringComparison.Ordinal)
                    && template.Contains(Gemma4ForwardToolResultScan, StringComparison.Ordinal);

                if (shared
                    && OccursExactlyOnce(template, Gemma4ToolCallsBranch)
                    && template.Contains(Gemma4CurrentTurnReasoningCondition, StringComparison.Ordinal)
                    && template.Contains(Gemma4MessageStateNamespace, StringComparison.Ordinal))
                {
                    SupportsRawToolCallReplay = true;
                    CachedReasoning = template.Replace(
                        Gemma4CurrentTurnReasoningCondition,
                        Gemma4CachedToolReasoningCondition,
                        StringComparison.Ordinal);
                    RawToolCallReplay = CachedReasoning.Replace(
                        Gemma4ToolCallsBranch,
                        Gemma4RawToolReplayBranch,
                        StringComparison.Ordinal);
                }
                else if (shared
                    && OccursExactlyOnce(template, Gemma4V2ToolCallsBranch)
                    && OccursExactlyOnce(template, Gemma4V2CurrentTurnReasoningGate)
                    && template.Contains(Gemma4V2MessageStateNamespace, StringComparison.Ordinal))
                {
                    SupportsRawToolCallReplay = true;
                    CachedReasoning = template.Replace(
                        Gemma4V2CurrentTurnReasoningGate,
                        Gemma4V2CachedToolReasoningGate,
                        StringComparison.Ordinal);
                    RawToolCallReplay = CachedReasoning.Replace(
                        Gemma4V2ToolCallsBranch,
                        Gemma4V2RawToolReplayBranch,
                        StringComparison.Ordinal);
                }
                else
                {
                    SupportsRawToolCallReplay = false;
                    CachedReasoning = template;
                    RawToolCallReplay = template;
                }
            }

            private static bool OccursExactlyOnce(string template, string needle)
            {
                int first = template.IndexOf(needle, StringComparison.Ordinal);
                return first >= 0
                    && template.IndexOf(needle, first + needle.Length, StringComparison.Ordinal) < 0;
            }

            public bool SupportsRawToolCallReplay { get; }
            public string CachedReasoning { get; }
            public string RawToolCallReplay { get; }
        }

        // The GGUF template is immutable for a loaded model. Cache both recognition and
        // transformed variants by string identity so a long tool loop does not rescan and
        // reallocate the same multi-kilobyte Jinja source on every continuation.
        private static readonly ConditionalWeakTable<string, Gemma4ReplayTemplateVariants>
            Gemma4ReplayTemplates = new();

        /// <summary>
        /// True only for the canonical Gemma 4 template shape whose structured
        /// tool-call branch can be replaced losslessly. Requiring exactly one match
        /// keeps community templates on the existing adaptive fallback.
        /// </summary>
        internal static bool SupportsGemma4RawToolCallReplay(
            string template, string? architecture)
        {
            if (!string.Equals(architecture, "gemma4", StringComparison.Ordinal)
                || string.IsNullOrEmpty(template))
            {
                return false;
            }

            return Gemma4ReplayTemplates.GetValue(
                template,
                static value => new Gemma4ReplayTemplateVariants(value))
                .SupportsRawToolCallReplay;
        }

        private static string EnableGemma4CachedToolReasoningReplay(
            string template, List<ChatMessage> messages, string? architecture)
        {
            if (!string.Equals(architecture, "gemma4", StringComparison.Ordinal)
                || string.IsNullOrEmpty(template)
                || messages == null)
            {
                return template;
            }

            bool hasCachedToolRound = false;
            bool hasRawReplayPlaceholder = false;
            for (int i = 0; i < messages.Count; i++)
            {
                ChatMessage message = messages[i];
                if (message == null)
                    continue;
                if (message.Role == "assistant"
                    && message.ToolCalls is { Count: > 0 }
                    && message.RawOutputTokens is { Count: > 0 })
                {
                    hasCachedToolRound = true;
                }
                if (!string.IsNullOrEmpty(message.RawToolCallReplayPlaceholder))
                    hasRawReplayPlaceholder = true;
            }
            if (!hasCachedToolRound)
                return template;

            Gemma4ReplayTemplateVariants variants = Gemma4ReplayTemplates.GetValue(
                template,
                static value => new Gemma4ReplayTemplateVariants(value));
            if (!variants.SupportsRawToolCallReplay)
                return template;
            return hasRawReplayPlaceholder
                ? variants.RawToolCallReplay
                : variants.CachedReasoning;
        }

        /// <summary>
        /// Remove the sentinel and the framing newline the template emitted after it, so a
        /// re-rendered reasoning block is byte-identical to what the model generated.
        /// See the note in BuildJinja2Context.
        /// </summary>
        internal static string StripReasoningEndSentinel(string rendered)
        {
            if (string.IsNullOrEmpty(rendered)
                || !rendered.Contains(ReasoningEndSentinel, StringComparison.Ordinal))
            {
                return rendered;
            }

            // The newline the template adds between the reasoning and the channel close
            // goes with it. Ordered longest-first; the bare sentinel is the fallback for a
            // template that frames the block some other way, and dropping it alone is
            // still better than leaving a marker in the prompt.
            return rendered
                .Replace(ReasoningEndSentinel + "\r\n", string.Empty, StringComparison.Ordinal)
                .Replace(ReasoningEndSentinel + "\n", string.Empty, StringComparison.Ordinal)
                .Replace(ReasoningEndSentinel, string.Empty, StringComparison.Ordinal);
        }

        private static Dictionary<string, object> BuildJinja2Context(
            List<ChatMessage> messages, bool addGenerationPrompt,
            List<ToolFunction>? tools = null, bool enableThinking = false,
            string? architecture = null)
        {
            // Whether this family's template re-renders an assistant turn's reasoning,
            // and therefore whether handing it over is correct rather than a change in
            // what past turns look like. See ChatProtocol.RendersAssistantReasoning.
            bool passReasoning =
                ChatProtocolRegistry.For(architecture)?.RendersAssistantReasoning ?? false;

            BuildJinjaToolIds(messages, out string[][] callIds, out string?[] resultIds);
            var msgList = new List<object>();
            for (int messageIndex = 0; messageIndex < messages.Count; messageIndex++)
            {
                ChatMessage m = messages[messageIndex];
                var dict = new Dictionary<string, object>
                {
                    ["role"] = m.Role ?? "",
                    ["content"] = m.Content ?? ""
                };
                if (resultIds[messageIndex] != null)
                    dict["tool_call_id"] = resultIds[messageIndex]!;
                if (passReasoning
                    && m.Role == "assistant"
                    && m.ToolCalls is { Count: > 0 }
                    && string.IsNullOrEmpty(m.RawToolCallReplayPlaceholder))
                {
                    // The reasoning, plus a sentinel marking where it ENDS.
                    //
                    // The template frames the channel itself, as
                    // '<|channel>thought(nl)' + text + '(nl)<channel|>'. The model does
                    // not: it writes '<|channel>thought(nl)' + text + '<channel|>', with
                    // whatever trailing newline the text itself happened to have. So the
                    // template's closing newline is one the cache does not contain when
                    // the reasoning ended without one, and one too many when it ended
                    // with one - and no value of `text` fixes both, which is why passing
                    // the reasoning alone still diverged by exactly one token.
                    //
                    // The sentinel is removed after rendering together with the newline
                    // the template put after it, leaving precisely the bytes the model
                    // produced. It also covers the empty case: a round that opened and
                    // closed the channel without thinking renders as the sentinel alone
                    // and collapses to the bare four-token channel, which no non-empty
                    // reasoning value could have produced.
                    string reasoning = (m.Thinking ?? string.Empty) + ReasoningEndSentinel;
                    dict["reasoning"] = reasoning;
                    dict["reasoning_content"] = reasoning;
                    if (m.RawOutputTokens is { Count: > 0 })
                        dict[Gemma4CachedToolReasoningMarker] = true;
                }
                if (m.ToolCalls != null && m.ToolCalls.Count > 0)
                {
                    var tcList = new List<object>();
                    for (int callIndex = 0; callIndex < m.ToolCalls.Count; callIndex++)
                    {
                        ToolCall tc = m.ToolCalls[callIndex];
                        tcList.Add(new Dictionary<string, object>
                        {
                            ["id"] = callIds[messageIndex][callIndex],
                            ["function"] = new Dictionary<string, object>
                            {
                                ["name"] = tc.Name,
                                ["arguments"] = tc.Arguments ?? new Dictionary<string, object>()
                            }
                        });
                    }
                    dict["tool_calls"] = tcList;
                }
                if (!string.IsNullOrEmpty(m.RawToolCallReplayPlaceholder))
                    dict[Gemma4RawToolCallReplayMarker] = m.RawToolCallReplayPlaceholder;
                msgList.Add(dict);
            }

            string bosToken = "";

            var ctx = new Dictionary<string, object>
            {
                ["messages"] = msgList,
                ["add_generation_prompt"] = addGenerationPrompt,
                ["bos_token"] = bosToken,
                ["eos_token"] = "",
            };

            // Always defined, whichever way it is set. Templates written against
            // HF's apply_chat_template(enable_thinking=...) test the flag with
            // `enable_thinking is defined and enable_thinking is false` (Qwen 3 and
            // 3.5 among them): leaving it out when false took the ELSE branch, which
            // is thinking ON, so a request with thinking off rendered exactly the
            // same prompt as one with it on and the model reasoned anyway. The
            // purpose-built Qwen 3.5 renderer had to be substituted for the off case
            // to work around that, and its prompt differed from the template's — so
            // toggling thinking between two turns of one chat diverged the prompt at
            // the tool block and re-prefilled the whole conversation.
            ctx["enable_thinking"] = enableThinking;

            if (tools != null && tools.Count > 0)
            {
                var toolList = new List<object>();
                foreach (var t in tools)
                    toolList.Add(BuildToolDeclaration(t));
                ctx["tools"] = toolList;
            }

            return ctx;
        }

        private static void BuildJinjaToolIds(
            List<ChatMessage> messages, out string[][] callIds, out string?[] resultIds)
        {
            callIds = new string[messages.Count][];
            resultIds = new string?[messages.Count];
            var usedIds = new HashSet<string>(StringComparer.Ordinal);
            foreach (ChatMessage message in messages)
            {
                if (!string.IsNullOrEmpty(message.ToolCallId)) usedIds.Add(message.ToolCallId);
                if (message.ToolCalls != null)
                    foreach (ToolCall call in message.ToolCalls)
                        if (!string.IsNullOrEmpty(call.Id)) usedIds.Add(call.Id);
            }

            int nextId = 0;
            string NewRenderId()
            {
                string id;
                do { id = "__tensorsharp_render_call_" + nextId++; }
                while (!usedIds.Add(id));
                return id;
            }

            for (int i = 0; i < messages.Count; i++)
            {
                ChatMessage message = messages[i];
                if (!string.IsNullOrEmpty(message.ToolCallId)) resultIds[i] = message.ToolCallId;
                if (message.ToolCalls is not { Count: > 0 }) continue;

                var ids = new string[message.ToolCalls.Count];
                for (int c = 0; c < ids.Length; c++)
                    ids[c] = string.IsNullOrEmpty(message.ToolCalls[c].Id)
                        ? NewRenderId() : message.ToolCalls[c].Id!;
                callIds[i] = ids;

                // Legacy local tool loops have no wire IDs and return results in
                // call order. Give their render context an explicit association:
                // canonical Gemma compares IDs, and two missing IDs compare equal,
                // incorrectly naming every result after the last function.
                // Reserve explicit results first; they may arrive out of order.
                int end = i + 1;
                var explicitResults = new HashSet<string>(StringComparer.Ordinal);
                while (end < messages.Count && messages[end].Role == "tool")
                {
                    if (!string.IsNullOrEmpty(messages[end].ToolCallId))
                        explicitResults.Add(messages[end].ToolCallId!);
                    end++;
                }
                var unclaimed = new Queue<string>(ids.Where(id => !explicitResults.Contains(id)));
                for (int r = i + 1; r < end; r++)
                    if (string.IsNullOrEmpty(messages[r].ToolCallId))
                        resultIds[r] = unclaimed.Count > 0 ? unclaimed.Dequeue() : NewRenderId();
            }
        }

        /// <summary>
        /// Pre-process messages to inject multimodal placeholder tokens into the content string
        /// so the Jinja2 template's {{ message['content'] }} renders them correctly.
        /// </summary>
        // internal (not private) so the per-architecture marker mapping can be
        // unit-tested without a GGUF: a missing branch here means the image marker
        // never reaches the prompt, the vision encoder still runs, and its
        // embeddings are silently discarded.
        internal static List<ChatMessage> InjectMultimodalTokens(List<ChatMessage> messages, string? architecture)
        {
            var result = new List<ChatMessage>(messages.Count);
            foreach (var msg in messages)
            {
                bool hasMedia = (msg.ImagePaths != null && msg.ImagePaths.Count > 0) ||
                                (msg.AudioPaths != null && msg.AudioPaths.Count > 0);

                if (!hasMedia)
                {
                    result.Add(msg);
                    continue;
                }

                // Which placeholder tokens a family's template expects is declared
                // once, beside its renderer and its output parser, in
                // ChatProtocolRegistry. A text-only protocol declares none and the
                // message content is passed through unchanged.
                var sb = new StringBuilder();
                ChatProtocolRegistry.For(architecture)?.AppendMediaPlaceholders?.Invoke(msg, sb);

                sb.Append(msg.Content ?? "");

                result.Add(new ChatMessage
                {
                    Role = msg.Role,
                    Content = sb.ToString(),
                    ImagePaths = msg.ImagePaths,
                    ImageTimestamps = msg.ImageTimestamps,
                    AudioPaths = msg.AudioPaths,
                    TextFilePaths = msg.TextFilePaths,
                    TextFileNames = msg.TextFileNames,
                    HasFileBackedTextAttachments = msg.HasFileBackedTextAttachments,
                    IsVideo = msg.IsVideo,
                    ToolCalls = msg.ToolCalls,
                    ToolCallId = msg.ToolCallId,
                    Thinking = msg.Thinking,
                    RawOutputTokens = msg.RawOutputTokens,
                    RawPromptTrailingWhitespace = msg.RawPromptTrailingWhitespace,
                    RawToolCallReplayPlaceholder = msg.RawToolCallReplayPlaceholder,
                    CacheControl = msg.CacheControl,
                    ContentCacheBreakpoints = msg.ContentCacheBreakpoints,
                });
            }
            return result;
        }

        /// <summary>
        /// Render Mistral 3 chat template.
        /// Uses [SYSTEM_PROMPT]...[/SYSTEM_PROMPT] for system messages
        /// and [INST]...[/INST] for user messages.
        /// </summary>
        /// <summary>
        /// Hunyuan dense / Hy-MT2 chat framing, matching
        /// <c>tencent/Hy-MT2-1.8B</c> <c>chat_template.jinja</c>.
        /// Always opens with BOS; user turns do not carry the assistant marker —
        /// that is appended only when <paramref name="addGenerationPrompt"/> is set.
        /// llama.cpp's <c>LLM_CHAT_TEMPLATE_HUNYUAN_DENSE</c> (Hunyuan-4B-Instruct)
        /// omits BOS and glues the assistant marker onto the user turn; Hy-MT2
        /// does not.
        /// </summary>
        public static string RenderHunyuanDense(List<ChatMessage> messages, bool addGenerationPrompt = true)
        {
            var sb = new StringBuilder();
            int startIdx = 0;

            if (messages.Count > 0 && messages[0] != null && messages[0].Role == "system")
            {
                sb.Append("<｜hy_begin▁of▁sentence｜>");
                sb.Append(messages[0].Content);
                sb.Append("<｜hy_place▁holder▁no▁3｜>");
                startIdx = 1;
            }
            else
            {
                sb.Append("<｜hy_begin▁of▁sentence｜>");
            }

            for (int i = startIdx; i < messages.Count; i++)
            {
                ChatMessage msg = messages[i];
                if (msg == null)
                    continue;

                if (msg.Role == "user")
                {
                    sb.Append("<｜hy_User｜>");
                    sb.Append(msg.Content);
                }
                else if (msg.Role == "assistant")
                {
                    sb.Append("<｜hy_Assistant｜>");
                    sb.Append(msg.Content);
                    sb.Append("<｜hy_place▁holder▁no▁2｜>");
                }
            }

            if (addGenerationPrompt)
                sb.Append("<｜hy_Assistant｜>");
            else
                sb.Append("<｜hy_place▁holder▁no▁8｜>");

            return sb.ToString();
        }

        public static string RenderMistral3(List<ChatMessage> messages, bool addGenerationPrompt = true)
        {
            var sb = new StringBuilder();
            int startIdx = 0;

            if (messages.Count > 0 && messages[0].Role == "system")
            {
                sb.Append("[SYSTEM_PROMPT]");
                sb.Append(messages[0].Content);
                sb.Append("[/SYSTEM_PROMPT]");
                startIdx = 1;
            }

            for (int i = startIdx; i < messages.Count; i++)
            {
                var msg = messages[i];
                if (msg.Role == "user")
                {
                    sb.Append("[INST]");
                    sb.Append(msg.Content);
                    sb.Append("[/INST]");
                }
                else if (msg.Role == "assistant")
                {
                    sb.Append(msg.Content);
                }
            }

            return sb.ToString();
        }

        /// <summary>
        /// Render GPT OSS / Harmony chat template.
        /// Matches the GGUF Jinja2 template: system message with model identity / date / channels,
        /// user/assistant messages with &lt;|start|&gt;role&lt;|message|&gt;content&lt;|end|&gt; framing,
        /// and a generation prompt of just &lt;|start|&gt;assistant (model generates channel tags).
        /// </summary>
        public static string RenderHarmony(List<ChatMessage> messages, bool addGenerationPrompt = true,
            List<ToolFunction>? tools = null, bool enableThinking = false)
        {
            var sb = new StringBuilder();
            bool hasTools = tools != null && tools.Count > 0;

            int startIdx = 0;
            string? developerContent = null;
            if (messages.Count > 0 && (messages[0].Role == "system" || messages[0].Role == "developer"))
            {
                developerContent = messages[0].Content;
                startIdx = 1;
            }

            // System message.
            sb.Append("<|start|>system<|message|>");
            sb.Append("You are ChatGPT, a large language model trained by OpenAI.\n");
            sb.Append("Knowledge cutoff: 2024-06\n");
            sb.Append($"Current date: {DateTime.Now:yyyy-MM-dd}\n\n");
            sb.Append("Reasoning: medium\n\n");
            sb.Append("# Valid channels: analysis, commentary, final. Channel must be included for every message.");
            if (hasTools)
                sb.Append("\nCalls to these tools must go to the commentary channel: 'functions'.");
            sb.Append("<|end|>");

            // Developer message carries the instructions and the tool namespace.
            if (!string.IsNullOrEmpty(developerContent) || hasTools)
            {
                sb.Append("<|start|>developer<|message|>");
                if (!string.IsNullOrEmpty(developerContent))
                {
                    sb.Append("# Instructions\n\n");
                    sb.Append(developerContent);
                    sb.Append("\n\n");
                }
                if (hasTools)
                {
                    sb.Append("# Tools\n\n");
                    RenderHarmonyToolNamespace(sb, "functions", tools!);
                }
                sb.Append("<|end|>");
            }

            // The name of the most recent assistant tool call; tool-result
            // messages are attributed to it (Harmony has no per-message tool id).
            string? lastToolName = null;
            for (int i = startIdx; i < messages.Count; i++)
            {
                var msg = messages[i];
                if (msg.Role == "assistant")
                {
                    if (msg.ToolCalls != null && msg.ToolCalls.Count > 0)
                    {
                        // Replay the reasoning that produced the call when available
                        // (Harmony keeps analysis for the active tool-call turn).
                        if (!string.IsNullOrEmpty(msg.Thinking))
                        {
                            sb.Append("<|start|>assistant<|channel|>analysis<|message|>");
                            sb.Append(msg.Thinking);
                            sb.Append("<|end|>");
                        }
                        foreach (var tc in msg.ToolCalls)
                        {
                            sb.Append("<|start|>assistant<|channel|>commentary to=functions.");
                            sb.Append(tc.Name);
                            sb.Append(" <|constrain|>json<|message|>");
                            sb.Append(SerializeToolArguments(tc.Arguments));
                            sb.Append("<|call|>");
                            lastToolName = tc.Name;
                        }
                        if (!string.IsNullOrEmpty(msg.Content))
                        {
                            sb.Append("<|start|>assistant<|channel|>final<|message|>");
                            sb.Append(msg.Content);
                            sb.Append("<|end|>");
                        }
                    }
                    else
                    {
                        sb.Append("<|start|>assistant<|channel|>final<|message|>");
                        sb.Append(msg.Content ?? "");
                        sb.Append("<|end|>");
                        lastToolName = null;
                    }
                }
                else if (msg.Role == "tool")
                {
                    sb.Append("<|start|>functions.");
                    sb.Append(lastToolName ?? "");
                    sb.Append(" to=assistant<|channel|>commentary<|message|>");
                    sb.Append(msg.Content ?? "");
                    sb.Append("<|end|>");
                }
                else
                {
                    sb.Append("<|start|>");
                    sb.Append(msg.Role);
                    sb.Append("<|message|>");
                    sb.Append(msg.Content ?? "");
                    sb.Append("<|end|>");
                }
            }
            if (addGenerationPrompt)
            {
                sb.Append("<|start|>assistant");
            }
            return sb.ToString();
        }

        /// <summary>
        /// Render the Harmony tool namespace, e.g.
        /// <code>## functions\n\nnamespace functions {\n\n// desc\ntype name = (_: {\nparam: string,\n}) => any;\n\n} // namespace functions</code>
        /// Matches the official gpt-oss chat template's TypeScript-style declarations.
        /// </summary>
        private static void RenderHarmonyToolNamespace(StringBuilder sb, string ns, List<ToolFunction> tools)
        {
            sb.Append("## ").Append(ns).Append("\n\n");
            sb.Append("namespace ").Append(ns).Append(" {\n\n");
            foreach (var tool in tools)
            {
                if (!string.IsNullOrEmpty(tool.Description))
                    sb.Append("// ").Append(tool.Description).Append('\n');
                sb.Append("type ").Append(tool.Name).Append(" = ");
                if (tool.Parameters != null && tool.Parameters.Count > 0)
                {
                    sb.Append("(_: {\n");
                    foreach (var kv in tool.Parameters)
                    {
                        var p = kv.Value;
                        if (!string.IsNullOrEmpty(p.Description))
                            sb.Append("// ").Append(p.Description).Append('\n');
                        sb.Append(kv.Key);
                        bool required = tool.Required != null && tool.Required.Contains(kv.Key);
                        if (!required)
                            sb.Append('?');
                        sb.Append(": ");
                        sb.Append(RenderHarmonyTsType(p));
                        sb.Append(",\n");
                    }
                    sb.Append("}) => any;\n\n");
                }
                else
                {
                    sb.Append("() => any;\n\n");
                }
            }
            sb.Append("} // namespace ").Append(ns);
        }

        /// <summary>Map a tool parameter to its Harmony TypeScript type.</summary>
        private static string RenderHarmonyTsType(ToolParameter p)
        {
            string type = (p.Type ?? "").ToLowerInvariant();
            switch (type)
            {
                case "string":
                    if (p.Enum != null && p.Enum.Count > 0)
                        return "\"" + string.Join("\" | \"", p.Enum) + "\"";
                    return "string";
                case "number":
                case "integer":
                    return "number";
                case "boolean":
                    return "boolean";
                case "array":
                    // ToolParameter does not carry item types; fall back to any[].
                    return "any[]";
                case "object":
                    return "object";
                default:
                    return "any";
            }
        }

        /// <summary>Serialize tool-call arguments to compact JSON for the Harmony commentary message.</summary>
        private static string SerializeToolArguments(Dictionary<string, object>? arguments)
        {
            if (arguments == null || arguments.Count == 0)
                return "{}";
            return JsonSerializer.Serialize(arguments);
        }

        /// <summary>
        /// Render Gemma4 chat template.
        /// Uses &lt;|turn&gt;/&lt;turn|&gt; markers. Images use &lt;|image&gt;.
        /// When thinking is disabled the generation prompt includes an empty
        /// thinking block (&lt;|channel&gt;thought\n&lt;channel|&gt;) so the model
        /// skips thinking.
        ///
        /// BOS is NOT emitted here: the tokenizer prepends it (add_bos_token=true,
        /// encode addSpecial=true), exactly like the GGUF Jinja2 path (which renders
        /// an empty bos_token). Emitting a literal
        /// &lt;bos&gt; here too would double the BOS token in the prompt.
        /// </summary>
        public static string RenderGemma4(List<ChatMessage> messages, bool addGenerationPrompt = true,
            List<ToolFunction>? tools = null, bool enableThinking = false)
        {
            var sb = new StringBuilder();

            bool hasTools = tools != null && tools.Count > 0;
            bool hasSystem = messages.Count > 0 && (messages[0].Role == "system" || messages[0].Role == "developer");
            int startIdx = 0;

            if (hasSystem || hasTools || enableThinking)
            {
                sb.Append("<|turn>system\n");
                if (enableThinking)
                    sb.Append("<|think|>\n");
                if (hasSystem)
                {
                    sb.Append(messages[0].Content?.Trim() ?? "");
                    startIdx = 1;
                }
                if (hasTools)
                {
                    foreach (var tool in tools!)
                        sb.Append(RenderGemma4ToolDeclaration(tool));
                }
                sb.Append("<turn|>\n");
            }

            for (int i = startIdx; i < messages.Count; i++)
            {
                var msg = messages[i];
                string role = msg.Role == "assistant" ? "model" : (msg.Role ?? "");
                sb.Append($"<|turn>{role}\n");

                if (msg.Role == "assistant" && msg.ToolCalls != null)
                {
                    foreach (var tc in msg.ToolCalls)
                        sb.Append(FormatGemma4ToolCall(tc));
                    if (!string.IsNullOrEmpty(msg.Content))
                        sb.Append(StripGemma4Thinking(msg.Content));
                }
                else if (msg.Role == "tool")
                {
                    sb.Append(msg.Content?.Trim() ?? "");
                }
                else if (msg.Role == "assistant")
                {
                    sb.Append(StripGemma4Thinking(msg.Content ?? "").Trim());
                }
                else
                {
                    AppendGemma4MediaPlaceholders(msg, sb);
                    sb.Append(msg.Content?.Trim() ?? "");
                }
                sb.Append("<turn|>\n");
            }
            if (addGenerationPrompt)
            {
                sb.Append("<|turn>model\n");
            }
            return sb.ToString();
        }

        internal static void AppendGemma4MediaPlaceholders(ChatMessage message, StringBuilder text)
        {
            bool timed = message.ImagePaths != null && message.ImageTimestamps?.Count == message.ImagePaths.Count;
            // Legacy UI histories may contain frames without source timestamps.
            // Preserve that framing without inventing frame times.
            if (message.IsVideo && message.ImagePaths != null && !timed) text.Append("<|video>");
            if (message.ImagePaths != null)
                for (int i = 0; i < message.ImagePaths.Count; ++i)
                {
                    if (timed && message.ImageTimestamps![i] is double time)
                    {
                        if (!double.IsFinite(time) || time < 0)
                            throw new ArgumentOutOfRangeException(nameof(message.ImageTimestamps));
                        // Gemma4Processor uses integer-truncated mm:ss source times.
                        if (i > 0) text.Append(' ');
                        text.Append(Math.Floor(time / 60).ToString("00", System.Globalization.CultureInfo.InvariantCulture))
                            .Append(':').Append(Math.Floor(time % 60).ToString("00", System.Globalization.CultureInfo.InvariantCulture))
                            .Append(' ');
                    }
                    text.Append("<|image>");
                }
            if (message.AudioPaths != null)
                foreach (var _ in message.AudioPaths) text.Append("<|audio>");
        }

        private static string RenderGemma4ToolDeclaration(ToolFunction tool)
        {
            const string q = "<|\"|>";
            var sb = new StringBuilder();
            sb.Append($"<|tool>declaration:{tool.Name}{{");
            sb.Append($"description:{q}{tool.Description ?? ""}{q}");

            bool hasParams = tool.Parameters != null && tool.Parameters.Count > 0;
            bool hasType = true;

            if (hasParams || hasType)
            {
                sb.Append(",parameters:{");
                bool needsComma = false;

                if (hasParams)
                {
                    sb.Append("properties:{");
                    var sortedKeys = new List<string>(tool.Parameters!.Keys);
                    sortedKeys.Sort(StringComparer.Ordinal);
                    bool first = true;
                    foreach (var key in sortedKeys)
                    {
                        var param = tool.Parameters[key];
                        if (!first) sb.Append(",");
                        first = false;
                        sb.Append($"{key}:{{");

                        bool hasContent = false;
                        if (!string.IsNullOrEmpty(param.Description))
                        {
                            sb.Append($"description:{q}{param.Description}{q}");
                            hasContent = true;
                        }
                        if (param.Enum != null && param.Enum.Count > 0 &&
                            (param.Type ?? "string").Equals("string", StringComparison.OrdinalIgnoreCase))
                        {
                            if (hasContent) sb.Append(",");
                            sb.Append("enum:[");
                            for (int i = 0; i < param.Enum.Count; i++)
                            {
                                if (i > 0) sb.Append(",");
                                sb.Append($"{q}{param.Enum[i]}{q}");
                            }
                            sb.Append("]");
                            hasContent = true;
                        }
                        if (hasContent) sb.Append(",");
                        sb.Append($"type:{q}{(param.Type ?? "string").ToUpper()}{q}}}");
                    }
                    sb.Append("}");
                    needsComma = true;
                }

                var requiredList = tool.Required;
                if (requiredList != null && requiredList.Count > 0)
                {
                    if (needsComma) sb.Append(",");
                    sb.Append("required:[");
                    for (int i = 0; i < requiredList.Count; i++)
                    {
                        if (i > 0) sb.Append(",");
                        sb.Append($"{q}{requiredList[i]}{q}");
                    }
                    sb.Append("]");
                    needsComma = true;
                }

                if (needsComma) sb.Append(",");
                sb.Append($"type:{q}OBJECT{q}}}");
            }
            sb.Append("}<tool|>");
            return sb.ToString();
        }

        private static string FormatGemma4ToolCall(ToolCall tc)
        {
            var sb = new StringBuilder();
            sb.Append($"<|tool_call>call:{tc.Name}{{");
            if (tc.Arguments != null)
            {
                var sortedKeys = new List<string>(tc.Arguments.Keys);
                sortedKeys.Sort(StringComparer.Ordinal);
                bool first = true;
                foreach (var key in sortedKeys)
                {
                    if (!first) sb.Append(",");
                    first = false;
                    sb.Append($"{key}:");
                    sb.Append(FormatGemma4ArgValue(tc.Arguments[key]));
                }
            }
            sb.Append("}<tool_call|>");
            return sb.ToString();
        }

        private static string FormatGemma4ArgValue(object value)
        {
            const string q = "<|\"|>";
            if (value is string s)
                return $"{q}{s}{q}";
            if (value is bool b)
                return b ? "true" : "false";
            if (value is long l)
                return l.ToString();
            if (value is int i)
                return i.ToString();
            if (value is double d)
                return d == Math.Floor(d) ? ((long)d).ToString() : d.ToString(System.Globalization.CultureInfo.InvariantCulture);
            if (value is float f)
                return f == MathF.Floor(f) ? ((long)f).ToString() : f.ToString(System.Globalization.CultureInfo.InvariantCulture);
            if (value is Dictionary<string, object> dict)
            {
                var sb2 = new StringBuilder("{");
                var keys = new List<string>(dict.Keys);
                keys.Sort(StringComparer.Ordinal);
                bool first = true;
                foreach (var k in keys)
                {
                    if (!first) sb2.Append(",");
                    first = false;
                    sb2.Append($"{k}:{FormatGemma4ArgValue(dict[k])}");
                }
                sb2.Append("}");
                return sb2.ToString();
            }
            if (value is List<object> list)
            {
                var sb2 = new StringBuilder("[");
                for (int idx = 0; idx < list.Count; idx++)
                {
                    if (idx > 0) sb2.Append(",");
                    sb2.Append(FormatGemma4ArgValue(list[idx]));
                }
                sb2.Append("]");
                return sb2.ToString();
            }
            return value?.ToString() ?? "null";
        }

        /// <summary>
        /// Ensure a thinking-enabled Qwen 3.5/3.6 generation prompt ends with an OPEN
        /// thinking block "&lt;think&gt;\n" (note the trailing newline).
        ///
        /// The model's own Jinja chat template emits exactly "&lt;think&gt;\n" for the
        /// generation prompt when reasoning is enabled, but the blanket
        /// <c>Render(...).TrimEnd()</c> above strips that trailing newline, leaving a
        /// bare "&lt;think&gt;". This model (matching llama.cpp's behavior) treats a bare
        /// "&lt;think&gt;" as a signal to produce an EMPTY reasoning block: it immediately
        /// emits "\n\n&lt;/think&gt;" and skips chain-of-thought entirely, collapsing the
        /// answer to a short, lower-quality direct reply. Restoring the newline makes the
        /// model actually reason, producing token-for-token the same high-quality output
        /// as llama.cpp for the same prompt.
        /// </summary>
        private static string EnsureQwen35ThinkOpen(string result)
        {
            const string openThink = "<think>";
            if (result.EndsWith(openThink))
                result += "\n";
            return result;
        }

        /// <summary>
        /// Ensure a thinking-disabled Qwen 3.5/3.6 generation prompt ends with the
        /// CLOSED, empty block "&lt;think&gt;\n\n&lt;/think&gt;\n\n" the model's own template
        /// emits. The blanket <c>Render(...).TrimEnd()</c> strips the two trailing
        /// newlines; without them the cache holds a different token run from the one
        /// the purpose-built renderer (and the model's training data) end on.
        /// </summary>
        private static string EnsureQwen35ThinkClosed(string result)
        {
            const string closedThink = "<think>\n\n</think>";
            if (result.EndsWith(closedThink, StringComparison.Ordinal))
                result += "\n\n";
            return result;
        }

        private static string StripGemma4Thinking(string text)
        {
            var result = new StringBuilder();
            while (true)
            {
                int start = text.IndexOf("<|channel>", StringComparison.Ordinal);
                if (start < 0)
                {
                    result.Append(text);
                    break;
                }
                result.Append(text.Substring(0, start));
                int end = text.IndexOf("<channel|>", start, StringComparison.Ordinal);
                if (end < 0)
                    break;
                text = text.Substring(end + 10);
            }
            return result.ToString().Trim();
        }

        /// <summary>
        /// Expand image pad tokens in a token sequence.
        /// Replaces each single imagePadTokenId with tokenCounts[i] copies.
        /// </summary>
        public static List<int> ExpandImageTokens(List<int> tokens, int imagePadTokenId, int[] tokenCounts)
        {
            var result = new List<int>(tokens.Count + 1024);
            int imageIdx = 0;
            foreach (int token in tokens)
            {
                if (token == imagePadTokenId && imageIdx < tokenCounts.Length)
                {
                    int count = tokenCounts[imageIdx++];
                    for (int j = 0; j < count; j++)
                        result.Add(imagePadTokenId);
                }
                else
                {
                    result.Add(token);
                }
            }
            return result;
        }

    }
}
