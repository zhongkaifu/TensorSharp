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
using System.Text;
using System.Text.Json;

namespace TensorSharp.Runtime
{
    public class ChatMessage
    {
        public string Role { get; set; } = string.Empty;
        public string Content { get; set; } = string.Empty;
        /// <summary>
        /// Optional list of image file paths for multimodal messages.
        /// </summary>
        public List<string>? ImagePaths { get; set; }
        /// <summary>
        /// Optional list of audio file paths for multimodal messages.
        /// </summary>
        public List<string>? AudioPaths { get; set; }
        /// <summary>
        /// Optional list of plain-text file paths whose contents have been inlined into
        /// <see cref="Content"/> (e.g. uploaded .txt / .md / .csv attachments). The paths
        /// themselves are not consumed by the model - they exist purely so the per-turn
        /// audit log can record which uploaded files belong to this message even though
        /// their contents have been folded into the prompt text.
        /// </summary>
        public List<string>? TextFilePaths { get; set; }
        /// <summary>
        /// True if ImagePaths represent video frames (inserts &lt;|video&gt; before frame &lt;|image&gt; tokens).
        /// </summary>
        public bool IsVideo { get; set; }
        /// <summary>
        /// Tool calls made by assistant in this message (for multi-turn tool calling).
        /// </summary>
        public List<ToolCall>? ToolCalls { get; set; }
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
    }

    public static class ChatTemplate
    {
        public static string RenderQwen3(List<ChatMessage> messages, bool addGenerationPrompt = true,
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
                    var toolObj = new Dictionary<string, object>
                    {
                        ["type"] = "function",
                        ["function"] = new Dictionary<string, object>
                        {
                            ["name"] = tool.Name,
                            ["description"] = tool.Description ?? "",
                            ["parameters"] = BuildToolParamsDict(tool)
                        }
                    };
                    sb.Append(JsonSerializer.Serialize(toolObj, new JsonSerializerOptions { WriteIndented = true }));
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
        {
            var props = new Dictionary<string, object>();
            if (tool.Parameters != null)
            {
                foreach (var kv in tool.Parameters)
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
            if (tool.Required != null && tool.Required.Count > 0)
                paramsDict["required"] = tool.Required;
            return paramsDict;
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
        /// Render Gemma3 chat template.
        /// Uses &lt;start_of_turn&gt;/&lt;end_of_turn&gt; markers. Images use &lt;start_of_image&gt;.
        /// BOS token is prepended by the tokenizer (add_bos_token=true).
        /// </summary>
        public static string RenderGemma3(List<ChatMessage> messages, bool addGenerationPrompt = true)
        {
            var sb = new StringBuilder();
            for (int i = 0; i < messages.Count; i++)
            {
                var msg = messages[i];
                string role = msg.Role == "assistant" ? "model" : (msg.Role ?? "");
                sb.Append($"<start_of_turn>{role}\n");
                if (msg.ImagePaths != null)
                {
                    foreach (var _ in msg.ImagePaths)
                        sb.Append("<start_of_image>");
                }
                sb.Append($"{msg.Content}<end_of_turn>\n");
            }
            if (addGenerationPrompt)
            {
                sb.Append("<start_of_turn>model\n");
            }
            return sb.ToString();
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
            if (IsQwen35Family(architecture) && !enableThinking)
                return RenderHardcoded(messages, addGenerationPrompt, architecture, tools, enableThinking);

            if (architecture == "mistral3")
                return RenderHardcoded(messages, addGenerationPrompt, architecture, tools, enableThinking);

            if (architecture == "nemotron_h" || architecture == "nemotron_h_moe" || architecture == "nemotron_h_omni")
                return RenderHardcoded(messages, addGenerationPrompt, architecture, tools, enableThinking);

            // gpt-oss / Harmony: the embedded template relies on advanced Jinja
            // features (recursive macros, namespace(), strftime_now, list slicing)
            // that the lightweight engine does not fully support — especially on the
            // tool-rendering path. Use the purpose-built hardcoded renderer instead so
            // tool declarations and tool-call framing are always correct.
            if (architecture == "gptoss" || architecture == "gpt-oss")
                return RenderHardcoded(messages, addGenerationPrompt, architecture, tools, enableThinking);

            if (!string.IsNullOrWhiteSpace(template))
            {
                try
                {
                    var preprocessed = InjectMultimodalTokens(messages, architecture);
                    var jinja = new Jinja2Template(template);
                    var context = BuildJinja2Context(preprocessed, addGenerationPrompt, architecture, tools, enableThinking);
                    string result = jinja.Render(context).TrimEnd();
                    if (result.Length > 0)
                    {
                        if (architecture == "gemma4" && addGenerationPrompt && !enableThinking)
                            result = EnsureGemma4ThinkingBlock(result);
                        Console.Error.WriteLine($"[ChatTemplate] Jinja2 rendering succeeded for '{architecture}', prompt length={result.Length}");
                        return result;
                    }
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine($"[ChatTemplate] Jinja2 rendering failed for architecture '{architecture}': {ex.Message}");
                    Console.Error.WriteLine($"[ChatTemplate] Exception: {ex}");
                    Console.Error.WriteLine($"[ChatTemplate] Falling back to hardcoded template.");
                }
            }

            Console.Error.WriteLine($"[ChatTemplate] Using hardcoded template for '{architecture}'");
            return RenderHardcoded(messages, addGenerationPrompt, architecture, tools, enableThinking);
        }

        private static string RenderHardcoded(List<ChatMessage> messages,
            bool addGenerationPrompt, string? architecture,
            List<ToolFunction>? tools = null, bool enableThinking = false)
        {
            if (architecture == "gemma3")
                return RenderGemma3(messages, addGenerationPrompt);

            if (architecture == "gemma4")
                return RenderGemma4(messages, addGenerationPrompt, tools, enableThinking);

            if (architecture == "gptoss" || architecture == "gpt-oss")
                return RenderHarmony(messages, addGenerationPrompt, tools, enableThinking);

            if (architecture == "qwen35" || architecture == "qwen35moe" || architecture == "qwen3next" ||
                architecture == "qwen3vl" || architecture == "qwen3vlmoe")
            {
                return RenderQwen35(messages, addGenerationPrompt, enableThinking, tools);
            }

            if (architecture == "nemotron_h" || architecture == "nemotron_h_moe" || architecture == "nemotron_h_omni")
                return RenderNemotron(messages, addGenerationPrompt, tools, enableThinking);

            if (architecture == "mistral3")
                return RenderMistral3(messages, addGenerationPrompt);

            return RenderQwen3(messages, addGenerationPrompt, tools, enableThinking);
        }

        private static bool IsQwen35Family(string? architecture)
        {
            return architecture == "qwen35" ||
                   architecture == "qwen35moe" ||
                   architecture == "qwen3next" ||
                   architecture == "qwen3vl" ||
                   architecture == "qwen3vlmoe";
        }

        private static Dictionary<string, object> BuildJinja2Context(
            List<ChatMessage> messages, bool addGenerationPrompt, string? architecture,
            List<ToolFunction>? tools = null, bool enableThinking = false)
        {
            var msgList = new List<object>();
            foreach (var m in messages)
            {
                var dict = new Dictionary<string, object>
                {
                    ["role"] = m.Role ?? "",
                    ["content"] = m.Content ?? ""
                };
                if (m.ToolCalls != null && m.ToolCalls.Count > 0)
                {
                    var tcList = new List<object>();
                    foreach (var tc in m.ToolCalls)
                    {
                        tcList.Add(new Dictionary<string, object>
                        {
                            ["function"] = new Dictionary<string, object>
                            {
                                ["name"] = tc.Name,
                                ["arguments"] = tc.Arguments ?? new Dictionary<string, object>()
                            }
                        });
                    }
                    dict["tool_calls"] = tcList;
                }
                msgList.Add(dict);
            }

            string bosToken = (architecture == "gemma4") ? "<bos>" : "";

            var ctx = new Dictionary<string, object>
            {
                ["messages"] = msgList,
                ["add_generation_prompt"] = addGenerationPrompt,
                ["bos_token"] = bosToken,
                ["eos_token"] = "",
            };

            if (enableThinking)
                ctx["enable_thinking"] = true;

            if (tools != null && tools.Count > 0)
            {
                var toolList = new List<object>();
                foreach (var t in tools)
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

                    toolList.Add(new Dictionary<string, object>
                    {
                        ["type"] = "function",
                        ["function"] = new Dictionary<string, object>
                        {
                            ["name"] = t.Name,
                            ["description"] = t.Description ?? "",
                            ["parameters"] = paramsDict
                        }
                    });
                }
                ctx["tools"] = toolList;
            }

            return ctx;
        }

        /// <summary>
        /// Pre-process messages to inject multimodal placeholder tokens into the content string
        /// so the Jinja2 template's {{ message['content'] }} renders them correctly.
        /// </summary>
        private static List<ChatMessage> InjectMultimodalTokens(List<ChatMessage> messages, string? architecture)
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

                var sb = new StringBuilder();
                if (architecture == "gemma4")
                {
                    if (msg.IsVideo && msg.ImagePaths != null)
                        sb.Append("<|video>");
                    if (msg.ImagePaths != null)
                        foreach (var _ in msg.ImagePaths) sb.Append("<|image>");
                    if (msg.AudioPaths != null)
                        foreach (var _ in msg.AudioPaths) sb.Append("<|audio>");
                }
                else if (architecture == "gemma3")
                {
                    if (msg.ImagePaths != null)
                        foreach (var _ in msg.ImagePaths) sb.Append("<start_of_image>");
                }
                else if (architecture is "qwen35" or "qwen35moe" or "qwen3next" or "qwen3vl" or "qwen3vlmoe")
                {
                    if (msg.ImagePaths != null)
                        foreach (var _ in msg.ImagePaths)
                            sb.Append("<|vision_start|><|image_pad|><|vision_end|>");
                }
                else if (architecture == "mistral3")
                {
                    if (msg.ImagePaths != null)
                        foreach (var _ in msg.ImagePaths)
                            sb.Append("[IMG]");
                }

                sb.Append(msg.Content ?? "");

                result.Add(new ChatMessage
                {
                    Role = msg.Role,
                    Content = sb.ToString(),
                    ImagePaths = msg.ImagePaths,
                    AudioPaths = msg.AudioPaths,
                    IsVideo = msg.IsVideo
                });
            }
            return result;
        }

        /// <summary>
        /// Render Mistral 3 chat template.
        /// Uses [SYSTEM_PROMPT]...[/SYSTEM_PROMPT] for system messages
        /// and [INST]...[/INST] for user messages.
        /// </summary>
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
        /// Matches the HF Jinja2 template: BOS is explicit, and when thinking is
        /// disabled the generation prompt includes an empty thinking block
        /// (&lt;|channel&gt;thought\n&lt;channel|&gt;) so the model skips thinking.
        /// </summary>
        public static string RenderGemma4(List<ChatMessage> messages, bool addGenerationPrompt = true,
            List<ToolFunction>? tools = null, bool enableThinking = false)
        {
            var sb = new StringBuilder();

            sb.Append("<bos>");

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
                    if (msg.ImagePaths != null)
                    {
                        if (msg.IsVideo)
                            sb.Append("<|video>");
                        foreach (var _ in msg.ImagePaths)
                            sb.Append("<|image>");
                    }
                    if (msg.AudioPaths != null)
                    {
                        foreach (var _ in msg.AudioPaths)
                            sb.Append("<|audio>");
                    }
                    sb.Append(msg.Content?.Trim() ?? "");
                }
                sb.Append("<turn|>\n");
            }
            if (addGenerationPrompt)
            {
                sb.Append("<|turn>model\n");
                if (!enableThinking)
                    sb.Append("<|channel>thought\n<channel|>");
            }
            return sb.ToString();
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
        /// Ensure the Gemma 4 prompt ends with an empty thinking block when thinking
        /// is disabled. The GGUF Jinja2 template may not produce it, but the model
        /// expects it to skip the thinking phase and generate content directly.
        /// </summary>
        private static string EnsureGemma4ThinkingBlock(string result)
        {
            const string emptyThinkBlock = "<|channel>thought\n<channel|>";
            if (!result.EndsWith(emptyThinkBlock))
            {
                if (!result.EndsWith("\n"))
                    result += "\n";
                result += emptyThinkBlock;
            }
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

        /// <summary>
        /// Expand Gemma3 image tokens: replace each &lt;start_of_image&gt; token with
        /// \n\n &lt;start_of_image&gt; [pad_tokens...] &lt;end_of_image&gt; \n\n
        /// </summary>
        public static List<int> ExpandGemma3ImageTokens(List<int> tokens, int startOfImageId,
            int endOfImageId, int newlineNewlineId, int padTokenId, int tokensPerImage)
        {
            var result = new List<int>(tokens.Count + tokensPerImage + 10);
            foreach (int token in tokens)
            {
                if (token == startOfImageId)
                {
                    result.Add(newlineNewlineId);
                    result.Add(startOfImageId);
                    for (int j = 0; j < tokensPerImage; j++)
                        result.Add(padTokenId);
                    result.Add(endOfImageId);
                    result.Add(newlineNewlineId);
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

