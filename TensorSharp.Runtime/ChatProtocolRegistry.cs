// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Linq;

namespace TensorSharp.Runtime
{
    /// <summary>
    /// The chat protocols this process knows, keyed by architecture name.
    ///
    /// THIS TABLE IS THE ONLY PLACE A NEW MODEL FAMILY'S TEXT FORMAT IS DECLARED.
    /// Prompt framing, GGUF-template bypass, media placeholders, reply parsing and
    /// grammar arming all read from the same entry, so a family cannot be half-added
    /// the way it could when those five things lived in five separate name chains.
    ///
    /// <see cref="Register"/> is public so a host can add a protocol without forking.
    /// </summary>
    public static class ChatProtocolRegistry
    {
        private static readonly object Gate = new();
        private static readonly Dictionary<string, ChatProtocol> ByArchitecture =
            new(StringComparer.OrdinalIgnoreCase);
        private static readonly List<ChatProtocol> Ordered = new();

        static ChatProtocolRegistry() => RegisterBuiltIns();

        public static void Register(ChatProtocol protocol)
        {
            ArgumentNullException.ThrowIfNull(protocol);
            protocol.Validate();

            lock (Gate)
            {
                foreach (string arch in protocol.Architectures)
                {
                    if (ByArchitecture.TryGetValue(arch, out var existing) && !ReferenceEquals(existing, protocol))
                    {
                        throw new InvalidOperationException(
                            $"Architecture '{arch}' already uses chat protocol '{existing.Id}'; " +
                            $"'{protocol.Id}' cannot claim it too.");
                    }
                }

                if (Ordered.Contains(protocol))
                    return;

                foreach (string arch in protocol.Architectures)
                    ByArchitecture[arch] = protocol;
                Ordered.Add(protocol);
            }
        }

        /// <summary>All registered protocols, in registration order.</summary>
        public static IReadOnlyList<ChatProtocol> All
        {
            get { lock (Gate) return Ordered.ToArray(); }
        }

        /// <summary>The protocol for an architecture, or null when it has none and the
        /// generic ChatML path applies.</summary>
        public static ChatProtocol? For(string? architecture)
        {
            if (string.IsNullOrEmpty(architecture))
                return null;
            lock (Gate)
                return ByArchitecture.TryGetValue(architecture, out var protocol) ? protocol : null;
        }

        private static void RegisterBuiltIns()
        {
            // ---- Gemma ------------------------------------------------------
            Register(new ChatProtocol
            {
                Id = "gemma4",
                Architectures = new[] { "gemma4" },
                Render = r => ChatTemplate.RenderGemma4(r.Messages, r.AddGenerationPrompt, r.Tools, r.EnableThinking),
                AppendMediaPlaceholders = ChatTemplate.AppendGemma4MediaPlaceholders,
                CapsVideoFrames = true,
                CreateOutputParser = () => new Gemma4OutputParser(),
                OutputParserAlwaysRequired = true,
                // The publisher template owns channel priming. Ordinary model turns
                // have no additional suffix; explicitly recorded older suffixes remain
                // authoritative when replaying their raw generated tokens.
                // The template can re-render an in-turn tool round's thinking channel
                // from `reasoning`, and needs `tool_calls` present to render that round's
                // tool RESULT. The KV renderer's canonical-template replay hook keeps
                // `tool_calls` for that result while splicing the exact generated token
                // run; structured reasoning remains the conservative fallback.
                RendersAssistantReasoning = true,
                // ...but only the CANONICAL Gemma 4 template has that reasoning branch.
                // The template shipped in earlier builds - and in the community
                // fine-tunes that inherited it - renders a past model turn as
                // `<|turn>model\n` + tool call, with `strip_thinking` deleting the
                // channel from the content and no `reasoning` field read anywhere. The
                // round's whole thought block (hundreds of tokens) then has no
                // counterpart in the re-render, the prompt diverges from the live cache
                // at the first tool-calling turn, and every following round of an Agent
                // Skills / code-exec turn re-prefills the entire conversation.
                //
                // That template does render `role: "tool"` as its own `<|turn>tool` turn,
                // independent of the assistant's tool_calls, so splicing the round's raw
                // tokens is safe THERE and only there. The renderer decides per prompt by
                // checking what the active template actually produced.
                ToolCallRawSplicing = ToolCallRawSplicing.WhenTemplateLosesTheRound,
                // With thinking on, the reply is `<|channel>thought\n...<channel|>` and
                // then the answer (after a tool result the template primes the opener
                // itself). The channel's own close is where a structured-output grammar
                // may start enforcing; without a trigger response_format + think=true
                // could only be refused.
                ThinkingGrammarActivationTrigger = "<channel|>",
                // The model opens its thought channel itself (only a tool-result
                // continuation with thinking on is primed open), so the budget counts
                // from `<|channel>` and closes with `<channel|>`. With thinking off no
                // Gemma 4 template primes anything after a tool result, and priming a
                // closed `<|channel>thought\n<channel|>` there was measured to make E4B
                // write its reasoning unmarked into the answer - so the prompt stays as
                // the template renders it and the sampler bounds what the model does.
                ThinkingBudgetEndToken = "<channel|>",
                ThinkingBudgetOpenToken = "<|channel>",
                SuppressUnopenedThinkingEndAfter = "<tool_response|>",
            });

            // ---- Qwen -------------------------------------------------------
            Register(new ChatProtocol
            {
                Id = "qwen3",
                Architectures = new[] { "qwen3" },
                CreateOutputParser = () => new ChatMlOutputParser(),
                // Qwen3 generation prompts place the reasoning boundary after the
                // assistant marker. Thinking-capable templates open it; the Bonsai
                // 8B template deliberately emits the closed/empty form every time.
                // Past-turn rendering may omit that boundary, so raw-token replay
                // must put back exactly what the live KV cache saw.
                AssistantGenerationSuffix = thinking => thinking
                    ? "<think>\n"
                    : "<think>\n\n</think>\n\n",
                EmitsEmptyThinkBlockForPastTurns = _ => true,
                // Tool results are rendered solely from role=tool; the preceding
                // structured call is not needed, making lossless raw replay safe.
                ToolCallRawSplicing = ToolCallRawSplicing.Always,
            });

            // Qwen2 / Qwen2.5(-VL): ChatML tool syntax without a thinking
            // channel. Without this entry the family fell through to the passthrough
            // parser, which can never read a tool call back — so skills and run_code
            // were silently withheld from a model that handles them fine. The GGUF's
            // own template renders the prompt; the hardcoded ChatML renderer (thinking
            // off) stands in when that template is missing or misrenders.
            Register(new ChatProtocol
            {
                Id = "qwen25",
                Architectures = new[] { "qwen2", "qwen2vl", "qwen2_vl", "qwen25vl" },
                Render = r => ChatTemplate.RenderChatMl(r.Messages, r.AddGenerationPrompt, r.Tools, enableThinking: false),
                CreateOutputParser = () => new Qwen25OutputParser(),
            });

            Register(new ChatProtocol
            {
                Id = "qwen35",
                Architectures = new[] { "qwen35", "qwen35moe", "qwen3next", "qwen3vl", "qwen3vlmoe" },
                Render = r => ChatTemplate.RenderQwen35(r.Messages, r.AddGenerationPrompt, r.EnableThinking, r.Tools),
                // The GGUF template is used as shipped in BOTH thinking modes. It used
                // to be replaced by the purpose-built renderer with thinking off,
                // because the Jinja context left `enable_thinking` undefined when false
                // and the template then took its thinking-ON branch. That is fixed at
                // the context (it is always defined now), and the two renderers had
                // drifted apart — pretty-printed versus compact tool JSON — so a chat
                // whose thinking toggle changed between turns re-prefilled everything
                // from the first tool declaration on. One renderer, one prompt.
                AppendMediaPlaceholders = AppendQwenVisionPads,
                CreateOutputParser = () => new Qwen35OutputParser(),
                // The template frames the generation prompt as `<think>\n` (thinking on)
                // or `<think>\n\n</think>\n\n` (off) after the assistant marker, and does
                // NOT re-emit either for PAST assistant messages before the last user
                // turn. The cache holds whichever one that turn was generated under, so
                // the renderer puts it back in front of the turn's raw tokens — the
                // turn's own recorded suffix when the transcript remembers it, this
                // request's mode otherwise.
                AssistantGenerationSuffix = thinking => thinking ? "<think>\n" : "<think>\n\n</think>\n\n",
                // For the assistant turn AFTER the last user message the template does
                // emit the empty block; it has to go before the recorded suffix is
                // injected, or the cache's `<think>` meets `<think>\n\n</think>\n\n<think>`.
                EmitsEmptyThinkBlockForPastTurns = _ => true,
                // Its tool-result branch depends only on role=tool, never on the
                // preceding assistant's structured tool_calls field. Keep the exact
                // generated reasoning + call tokens so an agent round extends the live
                // cache instead of re-prefilling the conversation.
                ToolCallRawSplicing = ToolCallRawSplicing.Always,
            });

            // Qwen3.8 Flash Next uses ChatML reasoning and the Qwen XML-style
            // function-call body, with Qwen-VL vision placeholders.
            Register(new ChatProtocol
            {
                Id = "qwen4exp",
                Architectures = new[] { "qwen4exp" },
                CreateOutputParser = () => new Qwen35OutputParser(),
                // The published template emits the closed, empty block when
                // enable_thinking=false. Cache replay must restore that exact suffix.
                AssistantGenerationSuffix = thinking => thinking
                    ? "<think>\n" : "<think>\n\n</think>\n\n",
                EmitsEmptyThinkBlockForPastTurns = _ => true,
                // role=tool renders independently of the assistant tool_calls field.
                ToolCallRawSplicing = ToolCallRawSplicing.Always,
                ThinkingGrammarActivationTrigger = "</think>",
                AppendMediaPlaceholders = AppendQwenVisionPads,
                // A `video_url` part is sampled into timed frames (fps / max_frames
                // in the part, VIDEO_SAMPLE_FPS / VIDEO_MAX_FRAMES defaults), each a
                // full image's worth of tokens, so long clips are capped like Gemma 4
                // and DeepSeek V4.1. The frames render as the Qwen3-VL video layout
                // (see QwenVideoFrames) and the injector merges them in temporal pairs.
                CapsVideoFrames = true,
            });

            // ---- GPT-OSS / Harmony -----------------------------------------
            Register(new ChatProtocol
            {
                Id = "harmony",
                Architectures = new[] { "gptoss", "gpt-oss" },
                Render = r => ChatTemplate.RenderHarmony(
                    r.Messages, r.AddGenerationPrompt, r.Tools, r.EnableThinking, r.ReasoningEffort),
                // The embedded template relies on recursive macros, namespace(),
                // strftime_now and list slicing - especially on the tool-rendering path
                // - which the lightweight Jinja engine does not fully support.
                PreferOwnRenderer = _ => true,
                CreateOutputParser = () => new HarmonyOutputParser(),
                OutputParserAlwaysRequired = true,
                GrammarActivationTrigger = "final<|message|>",
                // GPT-OSS reasons in the analysis channel first whether or not the
                // request asked for thinking, and the final channel opens with exactly
                // this header either way. With only the unconditional trigger declared,
                // the structured-output check read "no delayed trigger for thinking" and
                // refused every response_format request that also set think=true, though
                // the grammar arms at the very same place in both modes.
                ThinkingGrammarActivationTrigger = "final<|message|>",
                // The system message's `Reasoning: low|medium|high` line is the only
                // lever over how long GPT-OSS reasons (see ReasoningEffort).
                RendersReasoningEffort = true,
            });

            // DiffusionGemma writes Gemma 4's channel syntax: an answer may open with
            // the `<|channel>thought\n` primer, or close a thought block the prompt
            // opened with a bare `<channel|>`, before the reply. It had no protocol
            // entry, so no parser ran over the denoised text and OpenAI answers began
            // with the literal channel marker (38/39 JSON checks failed in the release
            // campaign). The GGUF template keeps rendering the prompt - there is
            // deliberately no Render here - and tool calls are refused at the adapter:
            // a block-diffusion turn has no tool-call loop to feed a result back into.
            Register(new ChatProtocol
            {
                Id = "diffusion-gemma",
                Architectures = new[] { "diffusion-gemma", "diffusion_gemma" },
                CreateOutputParser = () => new Gemma4OutputParser(),
                OutputParserAlwaysRequired = true,
                // The denoising pipeline renders every prompt with tools: null, so a
                // declaration never reaches the model. Gemma4OutputParser CAN read a
                // call back, and without this flag registering it flipped
                // SkillCapabilities.ToolsRendered to true: --code-exec and skills
                // discovery began offering the shell / skills_read tools, leasing a
                // workspace and running the skills loop for every diffusion request.
                RendersToolDeclarations = false,
            });

            // ---- Others -----------------------------------------------------
            Register(new ChatProtocol
            {
                Id = "muse-glimmer",
                Architectures = new[] { "muse-glimmer", "muse_glimmer" },
                TemplateAssistantHeaderAnchor = "<|start|>assistant",
                AppendMediaPlaceholders = (msg, sb) =>
                {
                    // The GGUF Jinja template renders an image content part as a single
                    // <|patch|> and a video part as <|video|>. The host later expands
                    // each <|patch|> into <|image_start|> + N filler rows +
                    // <|image_end|>, matching llama.cpp's mtmd chunking for
                    // PROJECTOR_TYPE_MUSE_GLIMMER.
                    if (msg.IsVideo && msg.ImagePaths != null)
                        sb.Append("<|video|>");
                    else if (msg.ImagePaths != null)
                        foreach (var _ in msg.ImagePaths) sb.Append("<|patch|>");
                },
                CreateOutputParser = () => new MuseGlimmerOutputParser(),
                // Every assistant message is wrapped in <|start|>...<|message|>...
                // <|eom|>/<|eot|> framing and its reasoning arrives on the "to=self"
                // channel, so an unparsed stream shows the raw tags and the whole chain
                // of thought as if it were the answer.
                OutputParserAlwaysRequired = true,
                // With thinking off no trigger: a grammar from token 0 makes the model
                // write the object with no header (MuseGlimmerOutputParser reads that as
                // the answer). With thinking on the reply is " to=self<|message|>...",
                // then "<|start|>assistant to=user<|message|>" and the answer - every
                // answer header in the 2026-09-16 --thinking run (65/65) had that
                // recipient - so the grammar arms after it. Without a trigger
                // response_format + think=true could only be refused (HTTP 400).
                ThinkingGrammarActivationTrigger = "to=user<|message|>",
            });

            Register(new ChatProtocol
            {
                Id = "deepseek41",
                Architectures = new[] { "deepseek41", "deepseek_v41" },
                CapsVideoFrames = true,
                AppendMediaPlaceholders = (msg, sb) =>
                {
                    if (msg.ImagePaths != null)
                        for (int i = 0; i < msg.ImagePaths.Count; i++)
                        {
                            if (msg.ImageTimestamps?.Count == msg.ImagePaths.Count && msg.ImageTimestamps[i] is double time)
                                sb.Append("Frame at ").Append(time.ToString("0.###", System.Globalization.CultureInfo.InvariantCulture))
                                    .Append(" seconds: ");
                            sb.Append(ChatTemplate.DeepSeek41ImagePlaceholder);
                            if (msg.ImageTimestamps?.Count == msg.ImagePaths.Count && msg.ImageTimestamps[i].HasValue)
                                sb.Append('\n');
                        }
                },
                Render = r => ChatTemplate.RenderDeepSeek41(r.Messages, r.AddGenerationPrompt, r.EnableThinking, r.Tools),
                PreferOwnRenderer = _ => true,
                CreateOutputParser = () => new DeepSeek41OutputParser(),
                OutputParserAlwaysRequired = true,
                ThinkingGrammarActivationTrigger = "</think>",
                ThinkingBudgetEndToken = "</think>",
                AllowRawAssistantTokenSplicing = false,
            });

            Register(new ChatProtocol
            {
                Id = "deepseek4",
                Architectures = new[] { "deepseek4" },
                Render = r => ChatTemplate.RenderDeepSeek4(r.Messages, r.AddGenerationPrompt, r.EnableThinking, r.Tools),
                // The GGUF-embedded (Unsloth) template leans on Jinja features the
                // lightweight engine handles inconsistently (nested namespaces,
                // from_json, dict.items()); the format itself is simple.
                PreferOwnRenderer = _ => true,
                CreateOutputParser = () => new DeepSeek4OutputParser(),
                // Its reasoning block and its DSML tool calls both arrive as plain text:
                // without the parser the </think> marker and the whole
                // <｜DSML｜tool_calls> block would be streamed to the client as if they
                // were the answer.
                OutputParserAlwaysRequired = true,
            });

            Register(new ChatProtocol
            {
                Id = "glm-dsa",
                Architectures = new[] { "glm-dsa", "glm_dsa" },
                Render = r => ChatTemplate.RenderGlmDsa(r.Messages, r.AddGenerationPrompt, r.EnableThinking, r.Tools),
                // The shipped template is built out of macros, namespaces, tojson and a
                // visible_text walker over structured content - the exact feature set
                // the lightweight Jinja engine renders inconsistently.
                PreferOwnRenderer = _ => true,
                CreateOutputParser = () => new GlmDsaOutputParser(),
                OutputParserAlwaysRequired = true,
            });

            Register(new ChatProtocol
            {
                Id = "glm5next",
                Architectures = new[] { "glm5next" },
                // GLM-5.3-Flash ALWAYS opens a <think> block in the generation prompt
                // (its template has no thinking-off shape), with no newline after it.
                // Re-rendered history goes through the template's empty-<think></think>
                // branch; stripping that restores the half the cache actually holds.
                AssistantGenerationSuffix = _ => "<think>",
                EmitsEmptyThinkBlockForPastTurns = _ => true,
                Render = r => ChatTemplate.RenderGlm5Next(r.Messages, r.AddGenerationPrompt, r.EnableThinking, r.Tools),
                AppendMediaPlaceholders = (msg, sb) =>
                {
                    // The template's emit_image() macro. The host later expands the
                    // single <|image|> into N placeholder tokens matching the merged
                    // patch count.
                    if (msg.ImagePaths != null)
                        foreach (var _ in msg.ImagePaths)
                            sb.Append("<|begin_of_image|><|image|><|end_of_image|>");
                },
                CreateOutputParser = () => new GlmDsaOutputParser(),
                OutputParserAlwaysRequired = true,
            });

            Register(new ChatProtocol
            {
                Id = "nemotron_h",
                Architectures = new[] { "nemotron_h", "nemotron_h_moe", "nemotron_h_omni" },
                Render = r => ChatTemplate.RenderNemotron(r.Messages, r.AddGenerationPrompt, r.Tools, r.EnableThinking),
                PreferOwnRenderer = _ => true,
                CreateOutputParser = () => new ChatMlOutputParser(),
                // Thinking on primes `<think>\n` after the assistant marker (thinking off
                // renders the closed `<think></think>`), so the answer starts after the
                // model's own `</think>` - the same boundary the Nemotron 3.5 GGUF
                // template uses. The JSON grammar arms there.
                ThinkingGrammarActivationTrigger = "</think>",
                // Nemotron 3.5 / Omni vocabularies carry `</think>` as one token, so the
                // thinking budget closes the block and the answer (and an armed grammar)
                // follows inside max_tokens. The Nemotron-H Reasoning-128K GGUFs spell it
                // in several tokens; WithThinkingBudget then declines and the generic
                // explained stop stays in place.
                ThinkingBudgetEndToken = "</think>",
            });

            Register(new ChatProtocol
            {
                Id = "hunyuan-dense",
                Architectures = new[] { "hunyuan-dense" },
                Render = r => ChatTemplate.RenderHunyuanDense(r.Messages, r.AddGenerationPrompt),
                // Official Hy-MT2 jinja (BOS + add_generation_prompt). The
                // renderer never emits tool declarations or role:"tool" results.
                RendersToolDeclarations = false,
                RendersToolResultMessages = false,
                PreferOwnRenderer = _ => true,
            });

            Register(new ChatProtocol
            {
                Id = "mistral3",
                Architectures = new[] { "mistral3" },
                Render = r => ChatTemplate.RenderMistral3(r.Messages, r.AddGenerationPrompt),
                // Two separate losses, both silent. r.Tools is discarded before the
                // renderer is called, so no tool is ever declared; and the renderer's
                // message loop handles only "user" and "assistant", so a role:"tool"
                // message is written nowhere at all - an agentic loop would feed a
                // result back into a prompt that does not contain it and the model
                // would call the same tool again until its budget ran out.
                RendersToolDeclarations = false,
                RendersToolResultMessages = false,
                PreferOwnRenderer = _ => true,
                AppendMediaPlaceholders = (msg, sb) =>
                {
                    if (msg.ImagePaths != null)
                        foreach (var _ in msg.ImagePaths) sb.Append("[IMG]");
                },
            });
        }

        // One <|vision_start|><|image_pad|><|vision_end|> per still image; sampled
        // video frames (frames with a source time) render as the Qwen3-VL video
        // layout, one <|video_pad|> block per temporal pair. QwenVideoFrames is the
        // single definition of that grouping, shared with the injector.
        private static void AppendQwenVisionPads(ChatMessage msg, System.Text.StringBuilder sb)
            => QwenVideoFrames.AppendPlaceholders(msg, sb);
    }
}
