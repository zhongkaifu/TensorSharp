// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Text;

namespace TensorSharp.Runtime
{
    /// <summary>Everything a chat renderer is given for one prompt.</summary>
    /// <param name="ReasoningEffort">The request's <c>reasoning_effort</c> level, or null
    /// for the family's default. See <see cref="Runtime.ReasoningEffort"/>.</param>
    public sealed record ChatRenderRequest(
        List<ChatMessage> Messages,
        bool AddGenerationPrompt,
        string? Architecture,
        List<ToolFunction>? Tools,
        bool EnableThinking,
        string? ReasoningEffort = null);

    /// <summary>
    /// Whether a family may reuse a past TOOL-CALLING round's exact generated tokens
    /// instead of asking its chat template to reconstruct that round.
    ///
    /// <para>
    /// Splicing is the only lossless way to reproduce what the live KV cache holds -
    /// reasoning channel, call syntax, whitespace and tokenization all included - but
    /// it costs the structured <see cref="ChatMessage.ToolCalls"/> field, which some
    /// templates read in order to render or address the tool RESULT that follows. The
    /// two effects pull in opposite directions, so each family declares which one it
    /// can afford.
    /// </para>
    /// </summary>
    public enum ToolCallRawSplicing
    {
        /// <summary>
        /// Never splice a tool-calling round. The template reconstructs it; where the
        /// template cannot, that round re-prefills. The safe default.
        /// </summary>
        Never = 0,

        /// <summary>
        /// Always splice. For families whose tool-result framing depends only on the
        /// <c>role: "tool"</c> message itself and never on the preceding assistant's
        /// structured tool calls - Qwen 3.5 / 3.6.
        /// </summary>
        Always,

        /// <summary>
        /// Splice only when the ACTIVE template demonstrably failed to reproduce the
        /// round, AND splicing it does not drop any tool result from the prompt.
        ///
        /// <para>
        /// Gemma 4 needs this because two different chat templates ship under one
        /// architecture name. The canonical (2026-07-09) template re-renders the round's
        /// thinking channel from <c>reasoning</c> and folds the tool result INTO the same
        /// model turn, gated on <c>tool_calls</c> - blanking that field there deletes
        /// every tool result from the prompt. Earlier builds (and the community
        /// fine-tunes carrying their template) have no reasoning branch at all and render
        /// <c>role: "tool"</c> as its own <c>&lt;|turn&gt;tool</c> turn - so there the
        /// round's whole thought channel is lost on re-render and splicing is both safe
        /// and necessary. Which template a GGUF carries is not knowable from the
        /// architecture name, so the renderer decides per prompt.
        /// </para>
        /// </summary>
        WhenTemplateLosesTheRound,
    }

    /// <summary>
    /// One model family's CHAT PROTOCOL: how its prompts are framed, what media
    /// placeholders its template expects, and how its replies are parsed back apart.
    ///
    /// These four questions used to be answered by four separate chains of
    /// architecture-name comparisons spread across <see cref="ChatTemplate"/> and
    /// <see cref="OutputParserFactory"/> - about two dozen sites in all - so adding a
    /// family meant finding every one of them, and forgetting one failed quietly
    /// (an unparsed reply streams its own reasoning tags to the client as if they
    /// were the answer; a missing media placeholder silently discards the image).
    /// One entry per family answers all four together.
    ///
    /// Deliberately separate from <c>ModelArchitectureDescriptor</c>: a chat protocol
    /// is a text format, not a model. Several architectures share one (GLM-DSA and
    /// GLM-5.3-Flash), the same protocol serves names that have no loader at all
    /// (<c>qwen3vl</c>), and this assembly must not depend on TensorSharp.Models.
    /// </summary>
    public sealed class ChatProtocol
    {
        /// <summary>Canonical protocol id, for diagnostics.</summary>
        public required string Id { get; init; }

        /// <summary>Every <c>general.architecture</c> string this protocol serves.</summary>
        public required IReadOnlyList<string> Architectures { get; init; }

        /// <summary>
        /// Purpose-built renderer for this family. Null means the generic ChatML
        /// renderer, which is the fallback for anything unrecognised.
        /// </summary>
        public Func<ChatRenderRequest, string>? Render { get; init; }

        /// <summary>
        /// True when the GGUF-embedded Jinja template must be bypassed in favour of
        /// <see cref="Render"/>. Several families ship templates built on Jinja
        /// features the lightweight engine renders inconsistently (recursive macros,
        /// namespaces, <c>tojson</c>, dict walkers), and their formats are simple
        /// enough to render directly. A predicate rather than a flag because Qwen 3.5
        /// bypasses only when thinking is disabled.
        /// </summary>
        public Func<ChatRenderRequest, bool>? PreferOwnRenderer { get; init; }

        /// <summary>
        /// Append the media placeholder tokens this family's template expects for one
        /// message, before its text content. Null for text-only protocols.
        ///
        /// Getting this wrong is silent: the encoder runs, and the embeddings are
        /// dropped because no placeholder marks where they go.
        /// </summary>
        public Action<ChatMessage, StringBuilder>? AppendMediaPlaceholders { get; init; }

        /// <summary>Parser that turns this family's raw stream back into content,
        /// reasoning and tool calls. Null means the passthrough parser.</summary>
        public Func<IOutputParser>? CreateOutputParser { get; init; }

        /// <summary>
        /// The reply is unreadable without the parser, so it must run even when the
        /// caller did not ask for reasoning or tool calls - the framing tokens and the
        /// whole chain of thought would otherwise be streamed as the answer.
        /// </summary>
        public bool OutputParserAlwaysRequired { get; init; }

        /// <summary>
        /// Text after which a structured-output grammar may start enforcing, or null
        /// when the model's very first token is already part of the answer.
        ///
        /// GPT-OSS opens every reply with a harmony channel header and reasons in the
        /// <c>analysis</c> channel before answering in <c>final</c>. A grammar armed
        /// from token 0 forbids that header, so the model is pushed straight into a
        /// JSON object having done no reasoning and fills the schema with
        /// placeholders. Arming on the final channel's header instead lets it think
        /// and constrains only the answer. See <c>GrammarConstraint.ActivateAfter</c>.
        /// </summary>
        public string? GrammarActivationTrigger { get; init; }

        /// <summary>
        /// Optional trigger used only when reasoning was enabled for this request.
        /// Families such as DeepSeek V4.1 start directly in JSON when reasoning is
        /// disabled, but must finish their reasoning block before JSON enforcement.
        /// </summary>
        public string? ThinkingGrammarActivationTrigger { get; init; }

        /// <summary>
        /// True when this family's prompt carries the request's
        /// <see cref="ChatRenderRequest.ReasoningEffort"/> (Harmony's
        /// <c>Reasoning: low|medium|high</c> system line). Families that do not render
        /// it ignore the value, and it then stays out of the shared-prefix key so an
        /// explicit <c>think:false</c> and an absent one keep sharing one checkpoint.
        /// </summary>
        public bool RendersReasoningEffort { get; init; }

        /// <summary>Trained single token that may close this family's open
        /// reasoning channel at its budget. Null retains the host's hard stop.</summary>
        public string? ThinkingBudgetEndToken { get; init; }

        /// <summary>
        /// Trained single token with which the MODEL opens its reasoning channel
        /// mid-reply, or null when only the prompt ever opens it. With it declared the
        /// budget counts from the opener instead of from the first generated token, and
        /// it also caps a channel the model opens although the request turned thinking
        /// OFF (see <c>ChatGenerationPipeline.UnrequestedThinkingBudgetFor</c>): that
        /// channel is hidden from the client, and Gemma 4 E4B otherwise spent a whole
        /// 256-token turn in it after a tool result and answered nothing.
        /// </summary>
        public string? ThinkingBudgetOpenToken { get; init; }

        /// <summary>
        /// Prompt tail after which, with thinking off, the
        /// <see cref="ThinkingBudgetEndToken"/> is masked while no channel is open, or
        /// null. Gemma 4's templates continue the model's own turn straight after
        /// <c>&lt;tool_response|&gt;</c> with no channel framing, and E4B there writes
        /// its answer, closes a channel it never opened and writes the answer again; a
        /// stream has already delivered the first copy by the time the stray close
        /// arrives, so the client got both. Scoped to that boundary because masking
        /// costs the device-argmax fast path.
        /// </summary>
        public string? SuppressUnopenedThinkingEndAfter { get; init; }

        /// <summary>
        /// Text the GENERATION PROMPT appends after the assistant role marker that
        /// re-rendering the same turn as HISTORY does not reproduce - given the
        /// thinking flag the turn ran under. Returns null/empty when the family's
        /// template frames past and current assistant turns identically.
        ///
        /// This is a KV-cache prefix-reuse fact, and getting it wrong is silent and
        /// expensive: the re-rendered prefix diverges at the first assistant boundary,
        /// every block hash misses, and each multi-turn request re-prefills the whole
        /// conversation at full cost while still answering correctly.
        /// </summary>
        public Func<bool, string?>? AssistantGenerationSuffix { get; init; }

        /// <summary>
        /// Whether cached assistant tokens may replace the template's history.
        /// Disable when the protocol deliberately changes past reasoning or turn
        /// framing; replaying raw tokens would violate the model's input format.
        /// </summary>
        public bool AllowRawAssistantTokenSplicing { get; init; } = true;

        /// <summary>
        /// True when this family's template emits an EMPTY
        /// <c>&lt;think&gt;&lt;/think&gt;</c> block ahead of a past assistant turn to
        /// say that turn's reasoning was dropped. The KV cache holds no such block -
        /// the original turn forwarded a real <c>&lt;think&gt;</c> plus reasoning
        /// tokens - so it must be removed before the suffix above is injected, or four
        /// spurious tokens land right at the first assistant boundary.
        /// </summary>
        public Func<bool, bool>? EmitsEmptyThinkBlockForPastTurns { get; init; }

        /// <summary>
        /// Marker after which the template emits an assistant HEADER that the raw
        /// generated tokens already carry themselves, or null when the template's
        /// assistant framing matches what the model actually produced.
        ///
        /// Muse-Glimmer is the case that needs this. Its generation prompt is bare -
        /// <c>&lt;|start|&gt;assistant</c> and nothing else - so the model itself emits
        /// the routing header and channel framing that follow. Rendering the SAME turn
        /// again as history takes the template's past-assistant branch, which emits
        /// <c>&lt;|start|&gt;assistant to=user&lt;|message|&gt;</c> before the content;
        /// splicing raw tokens after that header prepends three tokens the cache never
        /// saw, and prefix reuse was 0% for every multi-turn Muse-Glimmer conversation.
        /// </summary>
        public string? TemplateAssistantHeaderAnchor { get; init; }

        /// <summary>
        /// True when this family's chat template re-renders an assistant turn's REASONING
        /// from a <c>reasoning</c> / <c>reasoning_content</c> field, so the host should
        /// hand it over instead of dropping it.
        ///
        /// <para>
        /// It is opt-in per family because it changes the rendered prompt, and only a
        /// template that gates the reasoning correctly may have it: Gemma 4's emits the
        /// thinking channel only for an assistant message that comes AFTER the last user
        /// message and carries tool calls - that is, the rounds of the turn in progress -
        /// and strips it from every earlier turn, which is exactly the rule the KV cache
        /// needs. A template that re-emitted reasoning for past turns would change every
        /// multi-turn prompt, so families are enabled here one at a time, after their
        /// template has been read.
        /// </para>
        /// <para>
        /// What it fixes is not cosmetic. Without it the host had no way to reproduce a
        /// tool-calling round it had already generated, so
        /// <see cref="KVCachePromptRenderer"/> substituted the raw generated tokens
        /// instead - and that substitution blanks <c>tool_calls</c>, which Gemma 4's
        /// template requires in order to render the tool RESULT at all. Every tool result
        /// therefore vanished from the prompt from the second round on: the model asked
        /// for a directory listing, was shown nothing, and answered from invention.
        /// </para>
        /// </summary>
        public bool RendersAssistantReasoning { get; init; }

        /// <summary>
        /// Whether this protocol may splice a past assistant tool-call round's exact
        /// <see cref="ChatMessage.RawOutputTokens"/> into the prompt, and under what
        /// condition. Default <see cref="Runtime.ToolCallRawSplicing.Never"/>.
        ///
        /// <para>
        /// This is opt-in because replacing the message with a raw placeholder clears
        /// <see cref="ChatMessage.ToolCalls"/>: the raw tokens already contain the call
        /// syntax, but some templates need the structured field to place or address the
        /// following result. Qwen 3.5 / 3.6 renders <c>role: "tool"</c> independently,
        /// so splicing is safe there and is the only lossless way to reproduce its
        /// thinking-prefixed live KV-cache stream.
        /// </para>
        /// </summary>
        public ToolCallRawSplicing ToolCallRawSplicing { get; init; } = ToolCallRawSplicing.Never;

        /// <summary>
        /// True when this family turns a video into N evenly spaced FRAME images that
        /// each cost a full image's worth of tokens, so a long clip has to be
        /// downsampled to the configured frame cap before it reaches the model.
        /// Families that render a video as a single placeholder token do not.
        /// </summary>
        public bool CapsVideoFrames { get; init; }

        /// <summary>
        /// False when this family's renderer never puts TOOL DECLARATIONS in the
        /// prompt, so a tool offered to it can never be called.
        ///
        /// <para>
        /// Mistral 3's <see cref="Render"/> delegate takes only the messages and the
        /// generation flag, and the tool list is discarded before the renderer sees
        /// it. Declaring a tool for it is not an error the
        /// caller can see - the request succeeds and the model simply never calls what
        /// it was never told about.
        /// </para>
        /// <para>
        /// Agent Skills reads this to choose how to deliver a skill: where tools are
        /// rendered the skill body is fetched on demand through <c>skills_read</c>
        /// (progressive disclosure); where they are not, the body has to be written
        /// into the prompt up front instead. Without this flag, skills would appear to
        /// work on every model and silently do nothing on this family.
        /// </para>
        /// </summary>
        public bool RendersToolDeclarations { get; init; } = true;

        /// <summary>
        /// False when this family's renderer DROPS <c>role: "tool"</c> messages.
        ///
        /// <para>
        /// Mistral 3's renderer handles only <c>user</c> and <c>assistant</c>, so a tool
        /// result is not merely framed oddly - it is absent from the prompt entirely.
        /// An agentic loop that feeds a result back that way asks the model to continue
        /// from an answer it cannot see, and the usual outcome is that it calls the same
        /// tool again forever.
        /// </para>
        /// </summary>
        public bool RendersToolResultMessages { get; init; } = true;

        internal void Validate()
        {
            if (string.IsNullOrWhiteSpace(Id))
                throw new InvalidOperationException("Chat protocol has no Id.");
            if (Architectures == null || Architectures.Count == 0)
                throw new InvalidOperationException($"Chat protocol '{Id}' serves no architectures.");
            if (OutputParserAlwaysRequired && CreateOutputParser == null)
            {
                throw new InvalidOperationException(
                    $"Chat protocol '{Id}' says its parser is always required but supplies none.");
            }
            if ((ThinkingBudgetOpenToken != null || SuppressUnopenedThinkingEndAfter != null) && ThinkingBudgetEndToken == null)
            {
                throw new InvalidOperationException(
                    $"Chat protocol '{Id}' declares a reasoning-channel opener or stray-close policy without the " +
                    "channel's end token.");
            }
        }
    }
}
