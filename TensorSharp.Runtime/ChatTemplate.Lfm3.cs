// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace TensorSharp.Runtime
{
    /// <summary>
    /// Liquid LFM3 (LFM3-1.7B) and LFM3-Omni (LFM3-Omni-2B): ChatML turns with a
    /// <c>&lt;think&gt;</c> block, Qwen-Coder XML tool calls and <c>role: "tool"</c> turns
    /// framed as <c>&lt;tool_response&gt;</c>. Both GGUFs carry the publisher template
    /// (<c>general.architecture = lfm3</c> for both); the GGUF Jinja path renders it and is
    /// checked byte for byte against llama.cpp's renders (InferenceWeb.Tests
    /// <c>Lfm3ChatProtocolTests</c>). <see cref="RenderLfm3"/> is the fallback for a GGUF
    /// without a template or a Jinja failure, and reproduces the same text for every message
    /// shape TensorSharp represents.
    /// </summary>
    public static partial class ChatTemplate
    {
        /// <summary>Opens LFM3's reasoning block (a USER_DEFINED single token, 124901).</summary>
        public const string Lfm3ThinkOpen = "<think>";
        /// <summary>Closes LFM3's reasoning block (single token 124902).</summary>
        public const string Lfm3ThinkClose = "</think>";

        /// <summary>
        /// The placeholder for one still image, exactly as the LFM3 / LFM3-Omni templates
        /// render an <c>image</c> content part. The multimodal expander replaces the single
        /// <c>&lt;|image|&gt;</c> (125030) with the encoder's N rows, which gives
        /// llama.cpp's mtmd layout (<c>&lt;|vision_start|&gt;</c> + N rows +
        /// <c>&lt;|vision_end|&gt;</c>) and HF's (<c>&lt;|image|&gt;</c> x N inside the same pair).
        /// </summary>
        public const string Lfm3ImagePlaceholder = "<|vision_start|><|image|><|vision_end|>";

        /// <summary>
        /// The placeholder for one video (its sampled frames, <see cref="ChatMessage.IsVideo"/>).
        /// The expander replaces <c>&lt;|video|&gt;</c> (125029) with the timestamped frame
        /// pairs: per pair <c>"&lt;t.t seconds&gt;"</c> + <c>&lt;|vision_start|&gt;</c> + rows +
        /// <c>&lt;|vision_end|&gt;</c> (+ the pair's soundtrack rows on Omni), all inside this
        /// outer pair, as llama.cpp's mtmd and HF's Lfm3OmniProcessor lay a video out.
        /// </summary>
        public const string Lfm3VideoPlaceholder = "<|vision_start|><|video|><|vision_end|>";

        /// <summary>
        /// The placeholder for one audio clip on LFM3-Omni: the Omni template renders an
        /// <c>audio</c> / <c>input_audio</c> part inside the SAME vision span markers (llama.cpp
        /// <c>mtmd.cpp</c> wraps Omni audio in the image span; HF <c>audio_markers: true</c>).
        /// The expander replaces <c>&lt;|audio|&gt;</c> (125031) with the encoder's rows.
        /// </summary>
        public const string Lfm3AudioPlaceholder = "<|vision_start|><|audio|><|vision_end|>";

        private const string Lfm3ContinueFinalMessageTag = "CONTINUE_FINAL_MESSAGE_TAG ";

        /// <summary>
        /// The media placeholders of one message, before its text, one per
        /// <see cref="Lfm3MediaLayout.Items"/> entry: <see cref="Lfm3ImagePlaceholder"/> per
        /// still image, <see cref="Lfm3VideoPlaceholder"/> per video (all its frames, read from
        /// its file or from the host-sampled frames), then <see cref="Lfm3AudioPlaceholder"/>
        /// per audio clip. TensorSharp keeps a message's media as attachment lists rather
        /// than ordered content parts, so media always precede the text - the order of the
        /// model card's examples and of llama.cpp's render of an <c>[image, audio, text]</c>
        /// request.
        /// </summary>
        public static void AppendLfm3MediaPlaceholders(ChatMessage msg, StringBuilder sb)
        {
            foreach (Lfm3MediaItem item in Lfm3MediaLayout.Items(msg))
                sb.Append(item.Kind switch
                {
                    Lfm3MediaKind.Image => Lfm3ImagePlaceholder,
                    Lfm3MediaKind.Video => Lfm3VideoPlaceholder,
                    _ => Lfm3AudioPlaceholder,
                });
        }

        /// <summary>
        /// LFM3 / LFM3-Omni prompt without the GGUF template: the publisher template's text for
        /// string content (BOS excluded - the tokenizer adds it, as for the Jinja path, whose
        /// <c>bos_token</c> renders empty). Earlier turns lose their reasoning and open with an
        /// empty <c>&lt;think&gt;&lt;/think&gt;</c>; turns after the last user message keep the
        /// reasoning the host passes for a tool-calling round; consecutive tool results that
        /// answer the same call id share one turn; the generation prompt ends
        /// <c>&lt;think&gt;</c> (thinking on) or <c>&lt;think&gt;&lt;/think&gt;</c> (off).
        /// </summary>
        public static string RenderLfm3(List<ChatMessage> messages, bool addGenerationPrompt = true,
            List<ToolFunction>? tools = null, bool enableThinking = false)
        {
            messages ??= new List<ChatMessage>();
            var sb = new StringBuilder();

            int start = 0;
            string system = string.Empty;
            if (messages.Count > 0 && messages[0]?.Role == "system")
            {
                system = Lfm3Content(messages[0]);
                start = 1;
            }
            if (tools is { Count: > 0 })
            {
                var lines = tools.Select(t => Jinja2Template.ToJson(BuildToolDeclaration(t)["function"]));
                system += (system.Length > 0 ? "\n\n" : string.Empty) + "<tools>\n" + string.Join("\n", lines) + "\n</tools>";
            }
            if (system.Length > 0)
                sb.Append("<|im_start|>system\n").Append(system).Append("<|im_end|>\n");

            int lastUser = -1;
            for (int i = start; i < messages.Count; i++)
                if (messages[i]?.Role == "user")
                    lastUser = i;

            var pending = new List<string>();
            for (int i = start; i < messages.Count; i++)
            {
                ChatMessage m = messages[i];
                string role = m?.Role ?? string.Empty;
                string content = m == null ? string.Empty : Lfm3Content(m);
                ChatMessage? next = i + 1 < messages.Count ? messages[i + 1] : null;

                if (role == "tool")
                {
                    bool merge = next?.Role == "tool" && !string.IsNullOrEmpty(m!.ToolCallId)
                        && string.Equals(next.ToolCallId, m.ToolCallId, StringComparison.Ordinal);
                    if (merge)
                    {
                        if (content.Length > 0) pending.Add(content);
                        continue;
                    }
                    sb.Append("<|im_start|>tool\n");
                    if (pending.Count > 0)
                    {
                        if (content.Length > 0) pending.Add(content);
                        sb.Append("<tool_response>\n").Append(string.Join("\n", pending)).Append("\n</tool_response>");
                        pending.Clear();
                    }
                    else if (content.Contains("<tool_response>", StringComparison.Ordinal))
                        sb.Append(content);
                    else
                        sb.Append("<tool_response>\n").Append(content).Append("\n</tool_response>");
                    sb.Append("<|im_end|>\n");
                    continue;
                }

                sb.Append("<|im_start|>").Append(role).Append('\n');
                if (role != "assistant")
                {
                    sb.Append(content).Append("<|im_end|>\n");
                    continue;
                }

                bool keepThinking = i > lastUser;
                // The Jinja path hands a tool-calling round's reasoning to the template
                // (ChatProtocol.RendersAssistantReasoning); no other turn carries it there.
                bool thinkingDefined = m!.ToolCalls is { Count: > 0 } && m.Thinking != null
                    && string.IsNullOrEmpty(m.RawToolCallReplayPlaceholder);
                if (thinkingDefined && keepThinking)
                    sb.Append(Lfm3ThinkOpen).Append(m.Thinking).Append(Lfm3ThinkClose);
                else if (keepThinking && !content.Contains(Lfm3ThinkClose, StringComparison.Ordinal))
                    sb.Append(Lfm3ThinkOpen).Append(Lfm3ThinkClose);

                if (!keepThinking)
                {
                    int close = content.LastIndexOf(Lfm3ThinkClose, StringComparison.Ordinal);
                    if (close >= 0)
                        content = PythonStrip(content.Substring(close + Lfm3ThinkClose.Length));
                    content = Lfm3ThinkOpen + Lfm3ThinkClose + content;
                }
                bool continueFinal = content.EndsWith(Lfm3ContinueFinalMessageTag, StringComparison.Ordinal);
                if (continueFinal)
                    content = content.Substring(0, content.Length - Lfm3ContinueFinalMessageTag.Length);
                sb.Append(content);
                if (m.ToolCalls is { Count: > 0 })
                    AppendLfm3ToolCalls(sb, m.ToolCalls);
                if (continueFinal)
                    sb.Append(Lfm3ContinueFinalMessageTag);
                sb.Append("<|im_end|>\n");
            }

            if (addGenerationPrompt)
                sb.Append("<|im_start|>assistant\n").Append(enableThinking ? Lfm3ThinkOpen : Lfm3ThinkOpen + Lfm3ThinkClose);
            // The Jinja path trims the render (RenderFromGgufTemplate); both paths give one text.
            return sb.ToString().TrimEnd();
        }

        private static string Lfm3Content(ChatMessage message)
        {
            if ((message.ImagePaths is not { Count: > 0 }) && (message.AudioPaths is not { Count: > 0 }))
                return message.Content ?? string.Empty;
            var sb = new StringBuilder();
            AppendLfm3MediaPlaceholders(message, sb);
            return sb.Append(message.Content ?? string.Empty).ToString();
        }

        private static void AppendLfm3ToolCalls(StringBuilder sb, List<ToolCall> calls)
        {
            for (int c = 0; c < calls.Count; c++)
            {
                if (c > 0) sb.Append("\n\n");
                ToolCall call = calls[c];
                sb.Append("<tool_call>\n<function=").Append(call.Name).Append(">\n");
                if (call.Arguments != null)
                    foreach (var arg in call.Arguments)
                        sb.Append("<parameter=").Append(arg.Key).Append(">\n")
                          .Append(Lfm3RenderValue(NormalizeJinjaToolArgument(arg.Value)))
                          .Append("\n</parameter>\n");
                sb.Append("</function>\n</tool_call>");
            }
        }

        // The template's render_value macro: strings verbatim, booleans and None in JSON
        // spelling, mappings and lists through tojson, anything else as Jinja prints it.
        private static string Lfm3RenderValue(object? value) => value switch
        {
            null => "null",
            string s => s,
            bool b => b ? "true" : "false",
            _ => Jinja2Template.ToJson(value),
        };

        // Python's str.strip(): the template trims a past turn's answer after its </think>.
        private static string PythonStrip(string s) => s.Trim();
    }
}
