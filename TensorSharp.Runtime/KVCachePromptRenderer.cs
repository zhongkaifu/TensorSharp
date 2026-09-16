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

namespace TensorSharp.Runtime
{
    /// <summary>
    /// Renders a chat history to a token sequence in a way that is COMPATIBLE with
    /// per-turn KV cache reuse.
    ///
    /// Key invariant: when an assistant message has <see cref="ChatMessage.RawOutputTokens"/>
    /// set (i.e. the model previously generated this turn), those raw tokens are spliced
    /// directly into the rendered token sequence INSTEAD OF re-tokenizing the assistant's
    /// content text. A TOOL-CALLING round needs extra care because some templates read
    /// <see cref="ChatMessage.ToolCalls"/> to render the tool RESULT that follows. The
    /// canonical Gemma 4 template has a narrow replay hook that places the raw-token
    /// placeholder at its tool-call branch while retaining the structured field; other
    /// templates follow the family policy in <see cref="ChatProtocol.ToolCallRawSplicing"/>.
    ///
    /// Why this matters: assistant content is typically lossy with respect to raw
    /// generation. Thinking-style models emit <c>&lt;think&gt;...&lt;/think&gt;</c> tokens
    /// that the output parser strips out of <see cref="ChatMessage.Content"/>. Harmony /
    /// channel-based models emit channel framing tokens that the parser collapses into
    /// natural-language output. Even for "plain" models the BPE tokenizer can pick a
    /// different encoding for the same text when context changes (whitespace handling,
    /// re-merged tokens, etc.). All of these effects make naive re-rendering produce
    /// tokens that DIVERGE from what's in the KV cache after a few positions, which kills
    /// the cache hit rate.
    ///
    /// By splicing raw tokens we guarantee that the prefix of the new token sequence
    /// EXACTLY matches the cached tokens for as many tokens as the model has already
    /// produced - which is precisely what we want.
    ///
    /// Implementation strategy:
    ///   1. Replace each cached assistant message's content with a unique placeholder
    ///      string (a Private-Use-Area Unicode character + counter).
    ///   2. Render the chat template normally (text-level).
    ///   3. Walk the rendered text segment by segment, splitting on placeholders.
    ///   4. Tokenize each text segment using the model's tokenizer.
    ///   5. For each placeholder boundary, splice the corresponding raw tokens.
    ///
    /// Because placeholders are surrounded by structural tokens (e.g. an
    /// <c>&lt;|im_start|&gt;assistant\n...&lt;|im_end|&gt;</c> framing in Qwen, a
    /// <c>&lt;|turn&gt;model\n...&lt;turn|&gt;</c> framing in Gemma, etc.), each text
    /// segment is independently tokenizable: the BPE pretokenizer always splits at
    /// whitespace / newline / punctuation, so the encoder gives the same token sequence
    /// whether the segment is encoded alone or as part of the whole prompt.
    ///
    /// Only the first segment encodes BOS; subsequent segments use addSpecial=false so we
    /// never accidentally inject extra special tokens between turns.
    /// </summary>
    public sealed class KVCachePromptRenderer
    {
        // U+E000 is in the Private Use Area: real text never contains it. Each cached
        // assistant message gets a numbered placeholder so we can locate raw tokens in
        // order even if the chat template duplicates content (it doesn't today, but
        // numbering is cheap and defensive).
        internal const char PlaceholderSentinel = '\uE000';

        // U+E001 is used for explicit cache breakpoints.
        internal const char BreakpointSentinel = '\uE001';

        // U+E002 brackets a per-render nonce inserted inside each tool result while
        // proving that a candidate raw-token splice did not make the template drop it.
        // The proof is removed before tokenization and never reaches the model.
        internal const char ToolResultProofSentinel = '\uE002';

        private readonly IPromptRenderer _innerRenderer;

        public KVCachePromptRenderer(IPromptRenderer innerRenderer)
        {
            _innerRenderer = innerRenderer ?? throw new ArgumentNullException(nameof(innerRenderer));
        }

        /// <summary>
        /// Returns the text suffix that the chat template appends AFTER the assistant
        /// role marker but BEFORE the model's generated content, for the given architecture
        /// and thinking mode.
        ///
        /// During the first turn this suffix is part of the rendered prompt (it ends the
        /// "generation prompt"). On subsequent turns we therefore need the renderer to
        /// re-emit this text BEFORE the spliced raw tokens of cached assistant messages,
        /// otherwise the re-rendered token sequence will diverge from what's in the KV
        /// cache (cache has the suffix, naive re-render does not).
        ///
        /// Returns an empty string for architectures whose chat templates already emit
        /// the suffix as part of the standard assistant-message framing.
        /// </summary>
        public static string GetAssistantGenerationSuffix(string architecture, bool enableThinking)
        {
            if (string.IsNullOrEmpty(architecture))
                return string.Empty;

            // WHICH suffix, per family, is declared once in ChatProtocolRegistry beside
            // that family's renderer and output parser - it is a property of the chat
            // format, and it used to be a chain of name comparisons here that a new
            // family had to be remembered into. Families whose template frames past and
            // current-turn assistant messages identically (Harmony, Mistral 3,
            // Nemotron, ...) declare nothing and get an empty suffix.
            return ChatProtocolRegistry.For(architecture)?.AssistantGenerationSuffix?.Invoke(enableThinking)
                   ?? string.Empty;
        }

        /// <summary>
        /// Marker after which the chat template emits an assistant HEADER that the raw
        /// generated tokens already carry themselves, or null when the template's
        /// assistant framing matches what the model actually produced.
        ///
        /// Muse-Glimmer's GGUF template is the case that needs this. Its generation
        /// prompt is bare — <c>&lt;|start|&gt;assistant</c> and nothing else — so the model
        /// itself emits the routing header and channel framing that follow
        /// (<c> to=self&lt;|message|&gt;</c> for its reasoning turn, then
        /// <c>&lt;|eom|&gt;&lt;|start|&gt;assistant to=user&lt;|message|&gt;</c> for the answer).
        /// Rendering the SAME turn again as history, however, takes the template's
        /// past-assistant branch, which emits <c>&lt;|start|&gt;assistant to=user&lt;|message|&gt;</c>
        /// before the content. Splicing the raw tokens after that header prepends three
        /// tokens the KV cache never saw (` to`, `=user`, `&lt;|message|&gt;`), so the very
        /// first follow-up turn diverges at the first assistant boundary and EVERY block
        /// hash misses — prefix reuse was reported as 0% for every multi-turn Muse-Glimmer
        /// conversation. Dropping the template's header restores the exact cached stream:
        /// <c>&lt;|start|&gt;assistant</c> + raw tokens.
        /// </summary>
        internal static string GetTemplateAssistantHeaderAnchor(string architecture)
            => ChatProtocolRegistry.For(architecture)?.TemplateAssistantHeaderAnchor;

        /// <summary>
        /// Delete the text the template emitted between <paramref name="anchor"/> and each
        /// spliced-raw-token placeholder. The scan stops at the nearest anchor before the
        /// placeholder and refuses to cross a turn boundary (another <c>&lt;|start|&gt;</c>),
        /// so a template that legitimately emits no header is left untouched.
        /// </summary>
        private static string StripTemplateAssistantHeaders(string text, string anchor)
        {
            var sb = new System.Text.StringBuilder(text.Length);
            int searchPos = 0;
            while (searchPos < text.Length)
            {
                int sentinel = text.IndexOf(PlaceholderSentinel, searchPos);
                if (sentinel < 0)
                {
                    sb.Append(text, searchPos, text.Length - searchPos);
                    break;
                }

                int decorationsStart = FindBreakpointRunStart(text, searchPos, sentinel);
                int copyEnd = sentinel;
                bool strippedHeader = false;
                int anchorStart = text.LastIndexOf(
                    anchor,
                    decorationsStart - 1 < 0 ? 0 : decorationsStart - 1,
                    StringComparison.Ordinal);
                if (anchorStart >= searchPos)
                {
                    int headerStart = anchorStart + anchor.Length;
                    // Refuse to swallow a whole turn: the header may only be the short
                    // routing/channel run the template adds right before the content.
                    if (headerStart <= decorationsStart
                        && text.IndexOf(
                        "<|start|>", headerStart, decorationsStart - headerStart,
                        StringComparison.Ordinal) < 0)
                    {
                        copyEnd = headerStart;
                        strippedHeader = true;
                    }
                }
                sb.Append(text, searchPos, copyEnd - searchPos);
                if (strippedHeader && decorationsStart < sentinel)
                    sb.Append(text, decorationsStart, sentinel - decorationsStart);

                int sentinelEnd = text.IndexOf(PlaceholderSentinel, sentinel + 1);
                if (sentinelEnd < 0)
                    throw new InvalidOperationException(
                        "Malformed KV-cache placeholder: opening sentinel without matching close.");
                sb.Append(text, sentinel, sentinelEnd - sentinel + 1);
                searchPos = sentinelEnd + 1;
            }
            return sb.ToString();
        }

        /// <summary>
        /// Render <paramref name="messages"/> through the configured chat template into a
        /// token sequence, splicing raw assistant output tokens where available.
        /// </summary>
        /// <param name="tokenizer">Tokenizer to use for encoding text segments.</param>
        /// <param name="chatTemplate">The model's GGUF-embedded Jinja2 template (may be null).</param>
        /// <param name="messages">Chat history (may include assistant messages with <see cref="ChatMessage.RawOutputTokens"/>).</param>
        /// <param name="architecture">Architecture name from <see cref="ModelConfig.Architecture"/>.</param>
        /// <param name="addGenerationPrompt">Whether to append a generation-prompt suffix (e.g. <c>&lt;|im_start|&gt;assistant</c>).</param>
        /// <param name="tools">Optional tool list for tool-calling templates.</param>
        /// <param name="enableThinking">Whether to enable the model's thinking / reasoning channel.</param>
        public List<int> RenderToTokens(
            ITokenizer tokenizer,
            string chatTemplate,
            List<ChatMessage> messages,
            string architecture,
            bool addGenerationPrompt,
            List<ToolFunction>? tools = null,
            bool enableThinking = false,
            string? reasoningEffort = null)
        {
            return RenderToTokens(
                tokenizer, chatTemplate, messages, architecture, addGenerationPrompt, out _, tools, enableThinking,
                reasoningEffort);
        }

        public List<int> RenderToTokens(
            ITokenizer tokenizer,
            string chatTemplate,
            List<ChatMessage> messages,
            string architecture,
            bool addGenerationPrompt,
            out List<int>? explicitBreakpoints,
            List<ToolFunction>? tools = null,
            bool enableThinking = false,
            string? reasoningEffort = null)
        {
            return RenderToTokens(
                tokenizer, chatTemplate, messages, architecture, addGenerationPrompt,
                out explicitBreakpoints, out _, tools, enableThinking, reasoningEffort);
        }

        /// <summary>
        /// Render a prompt and report the exact whitespace at its generation boundary.
        /// The server records that value beside the generated raw tokens so later turns
        /// can replay the boundary without a template-shape heuristic.
        /// </summary>
        public List<int> RenderToTokens(
            ITokenizer tokenizer,
            string chatTemplate,
            List<ChatMessage> messages,
            string architecture,
            bool addGenerationPrompt,
            out List<int>? explicitBreakpoints,
            out string generationPromptTrailingWhitespace,
            List<ToolFunction>? tools = null,
            bool enableThinking = false,
            string? reasoningEffort = null)
        {
            explicitBreakpoints = null;
            generationPromptTrailingWhitespace = string.Empty;
            if (tokenizer == null)
                throw new ArgumentNullException(nameof(tokenizer));
            if (messages == null)
                throw new ArgumentNullException(nameof(messages));

            ToolCallRawSplicing splicing =
                ChatProtocolRegistry.For(architecture)?.ToolCallRawSplicing ?? ToolCallRawSplicing.Never;
            bool hasCachedToolCallRound = false;
            bool isGemma4ReplayCandidate = splicing == ToolCallRawSplicing.WhenTemplateLosesTheRound
                && _innerRenderer is GgufPromptRenderer
                && string.Equals(architecture, "gemma4", StringComparison.Ordinal)
                && !string.IsNullOrEmpty(chatTemplate);
            if (isGemma4ReplayCandidate)
            {
                for (int i = 0; i < messages.Count; i++)
                {
                    ChatMessage message = messages[i];
                    if (message != null
                        && message.Role == "assistant"
                        && message.RawOutputTokens is { Count: > 0 }
                        && message.ToolCalls is { Count: > 0 })
                    {
                        hasCachedToolCallRound = true;
                        break;
                    }
                }
            }
            bool losslessGemma4ToolReplay = hasCachedToolCallRound
                && ChatTemplate.SupportsGemma4RawToolCallReplay(chatTemplate, architecture);

            RenderPass pass;
            try
            {
                pass = Render(
                    tokenizer, chatTemplate, messages, architecture, addGenerationPrompt, tools, enableThinking,
                    reasoningEffort,
                    spliceToolCallRounds: splicing == ToolCallRawSplicing.Always || losslessGemma4ToolReplay,
                    useGemma4RawToolReplay: losslessGemma4ToolReplay,
                    proveToolResults: losslessGemma4ToolReplay);
            }
            catch (RawToolCallReplayUnavailableException)
            {
                // A recognized GGUF template can still be abandoned by the Jinja
                // correctness guard (or a custom renderer can otherwise omit the
                // injected branch). Retry through the established structured/adaptive
                // path instead of turning a cache optimization into a failed request.
                losslessGemma4ToolReplay = false;
                pass = Render(
                    tokenizer, chatTemplate, messages, architecture, addGenerationPrompt, tools, enableThinking,
                    reasoningEffort,
                    spliceToolCallRounds: splicing == ToolCallRawSplicing.Always,
                    useGemma4RawToolReplay: false,
                    proveToolResults: false);
            }

            RenderPass selected;
            if (splicing != ToolCallRawSplicing.WhenTemplateLosesTheRound
                || pass.ToolCallRoundsLeftToTemplate.Count == 0)
            {
                selected = pass;
            }
            else if (ReproducesEveryToolCallRound(
                tokenizer, pass.Tokens, pass.ToolCallRoundsLeftToTemplate))
            {
                // The template was given this family's structured reasoning and tool calls
                // and reproduced the generated runs. Boundary fidelity is handled below by
                // the per-round RawPromptTrailingWhitespace metadata.
                selected = pass;
            }
            else
            {
                // The template dropped part of a generated tool round. Splice it only
                // when doing so leaves every tool result visible to the model.
                RenderPass spliced = Render(
                    tokenizer, chatTemplate, messages, architecture, addGenerationPrompt, tools, enableThinking,
                    reasoningEffort,
                    spliceToolCallRounds: true,
                    useGemma4RawToolReplay: false,
                    proveToolResults: true);

                if (!spliced.ToolResultsProven)
                {
                    WarnToolCallSplicingUnavailableOnce(architecture);
                    selected = pass;
                }
                else
                {
                    selected = spliced;
                }
            }

            explicitBreakpoints = selected.ExplicitBreakpoints;
            generationPromptTrailingWhitespace = TrailingWhitespace(selected.Text);
            return selected.Tokens;
        }

        /// <summary>One rendered prompt, plus what the template was left to reconstruct.</summary>
        private readonly struct RenderPass
        {
            public RenderPass(
                List<int> tokens,
                string text,
                List<UnsplicedToolCallRound> toolCallRoundsLeftToTemplate,
                List<int>? explicitBreakpoints,
                bool toolResultsProven)
            {
                Tokens = tokens;
                Text = text;
                ToolCallRoundsLeftToTemplate = toolCallRoundsLeftToTemplate;
                ExplicitBreakpoints = explicitBreakpoints;
                ToolResultsProven = toolResultsProven;
            }

            /// <summary>The prompt token sequence.</summary>
            public List<int> Tokens { get; }

            /// <summary>The rendered text, before tokenization (placeholders still in it).</summary>
            public string Text { get; }

            /// <summary>
            /// Raw output tokens of each assistant TOOL-CALLING round this pass did NOT
            /// splice, i.e. the rounds the chat template was asked to rebuild itself.
            /// </summary>
            public List<UnsplicedToolCallRound> ToolCallRoundsLeftToTemplate { get; }

            /// <summary>Final token offsets of explicit cache-control markers.</summary>
            public List<int>? ExplicitBreakpoints { get; }

            /// <summary>
            /// Whether every render-time tool-result proof survived exactly once.
            /// Always true for passes that did not request proofing.
            /// </summary>
            public bool ToolResultsProven { get; }
        }

        /// <summary>
        /// Internal control-flow signal: the template renderer did not preserve the
        /// canonical Gemma replay hook, so the caller should use its safe adaptive path.
        /// </summary>
        private sealed class RawToolCallReplayUnavailableException : Exception
        {
        }

        private readonly struct UnsplicedToolCallRound
        {
            public UnsplicedToolCallRound(List<int> rawTokens, string? boundaryWhitespace)
            {
                RawTokens = rawTokens;
                BoundaryWhitespace = boundaryWhitespace;
            }

            public List<int> RawTokens { get; }
            public string? BoundaryWhitespace { get; }
        }

        private readonly struct ToolResultProof
        {
            public ToolResultProof(string marker, string expectedWrappedContent)
            {
                Marker = marker;
                ExpectedWrappedContent = expectedWrappedContent;
                MarkerOffset = expectedWrappedContent.IndexOf(marker, StringComparison.Ordinal);
            }

            public string Marker { get; }
            public string ExpectedWrappedContent { get; }
            public int MarkerOffset { get; }
        }

        private RenderPass Render(
            ITokenizer tokenizer,
            string chatTemplate,
            List<ChatMessage> messages,
            string architecture,
            bool addGenerationPrompt,
            List<ToolFunction>? tools,
            bool enableThinking,
            string? reasoningEffort,
            bool spliceToolCallRounds,
            bool useGemma4RawToolReplay,
            bool proveToolResults)
        {
            // Build a parallel list where each cached assistant message is replaced with a
            // placeholder ChatMessage. Track the raw tokens in render order so we can splice
            // them back in.
            List<ChatMessage>? renderedMessages = null;
            List<List<int>>? rawTokensByPlaceholderIndex = null;
            List<string?>? rawBoundaryWhitespaceByPlaceholderIndex = null;
            List<string?>? rawGenerationSuffixByPlaceholderIndex = null;
            List<int>? rawToolCallReplayPlaceholderIndices = null;
            List<ToolResultProof>? toolResultProofs = null;
            var toolCallRoundsLeftToTemplate = new List<UnsplicedToolCallRound>();
            bool toolResultsProvable = true;
            int placeholderCount = 0;
            int breakpointCount = 0;
            bool allowRawAssistantTokenSplicing =
                ChatProtocolRegistry.For(architecture)?.AllowRawAssistantTokenSplicing ?? true;

            // A marker on any tool means "keep the tool block cached". The chat
            // template renders the whole tool list as one unit, so a marker on
            // one tool and a marker on all of them mean the same thing.
            bool toolsHasMarker = false;
            if (tools != null)
            {
                foreach (var t in tools)
                {
                    if (t.CacheControl != null) { toolsHasMarker = true; break; }
                }
            }

            for (int i = 0; i < messages.Count; i++)
            {
                ChatMessage msg = messages[i];
                bool hasRawTokens = msg != null
                    && msg.Role == "assistant"
                    && allowRawAssistantTokenSplicing
                    && msg.RawOutputTokens != null
                    && msg.RawOutputTokens.Count > 0;

                // A tool-calling round is the contested case. The placeholder below blanks
                // ToolCalls, because the raw tokens already carry the call markup and
                // rendering it twice would duplicate it - but a template may read that same
                // field to place or address the tool RESULT that follows, and Gemma 4's
                // canonical template deletes every result without it. Which families may
                // splice such a round, and under what condition, is declared once per
                // family as ChatProtocol.ToolCallRawSplicing; the caller resolves the
                // condition and passes the answer down.
                bool isToolCallRound = hasRawTokens && msg!.ToolCalls is { Count: > 0 };
                bool useRawToolCallReplayMarker = isToolCallRound
                    && spliceToolCallRounds
                    && useGemma4RawToolReplay;
                if (isToolCallRound && !spliceToolCallRounds)
                {
                    toolCallRoundsLeftToTemplate.Add(new UnsplicedToolCallRound(
                        msg!.RawOutputTokens!, msg.RawPromptTrailingWhitespace));
                    hasRawTokens = false;
                }

                bool hasMarker = msg?.CacheControl != null;
                bool hasPartMarkers = msg?.ContentCacheBreakpoints != null
                    && msg.ContentCacheBreakpoints.Count > 0;
                bool needsToolResultProof = proveToolResults
                    && msg?.Role == "tool"
                    && !string.IsNullOrWhiteSpace(msg.Content);
                if (proveToolResults
                    && msg?.Role == "tool"
                    && string.IsNullOrWhiteSpace(msg.Content))
                {
                    // There is no interior character at which a proof can be inserted
                    // without changing truthiness or edge-trimming behavior. Keep the
                    // structured pass for this rare shape rather than guessing whether
                    // the response framing survived.
                    toolResultsProvable = false;
                }

                // A tool-level marker can only be expressed by prefixing the
                // first message's content, because the tool block itself is
                // emitted by the chat template and there is nowhere else to put
                // a sentinel. Where the template renders tools ahead of the
                // messages (the hardcoded Qwen 3.5 path, which opens its own
                // system turn for them) that lands the breakpoint just after the
                // tool block, which is what the client asked for. Templates that
                // render the tool list *inside* the first system message — which
                // most Jinja templates do — put the tool JSON after that
                // message's content, so the breakpoint lands before the tools
                // and the tool block is left out of the cached prefix. The
                // marker then under-caches rather than caching the wrong thing;
                // placing it correctly needs the sentinel emitted by the
                // template itself, which the Jinja path cannot express.
                bool needsToolsMarker = toolsHasMarker && i == 0;

                if (!hasRawTokens && !hasMarker && !hasPartMarkers
                    && !needsToolsMarker && !needsToolResultProof)
                {
                    if (renderedMessages != null)
                        renderedMessages.Add(msg!);
                    continue;
                }

                if (renderedMessages == null)
                {
                    renderedMessages = new List<ChatMessage>(messages.Count);
                    for (int j = 0; j < i; j++)
                        renderedMessages.Add(messages[j]);
                    rawTokensByPlaceholderIndex = new List<List<int>>();
                    rawBoundaryWhitespaceByPlaceholderIndex = new List<string?>();
                    rawGenerationSuffixByPlaceholderIndex = new List<string?>();
                }

                string newContent = msg!.Content ?? "";
                string? rawToolCallReplayPlaceholder = null;

                if (hasRawTokens)
                {
                    string placeholder = MakePlaceholder(placeholderCount);
                    if (useRawToolCallReplayMarker)
                    {
                        // The canonical Gemma template renders results only while the
                        // structured tool_calls field is present. Put the raw-token
                        // placeholder at that branch and leave Content empty so the call
                        // is not duplicated later in the assistant body.
                        newContent = string.Empty;
                        rawToolCallReplayPlaceholder = placeholder;
                        rawToolCallReplayPlaceholderIndices ??= new List<int>();
                        rawToolCallReplayPlaceholderIndices.Add(placeholderCount);
                    }
                    else
                    {
                        newContent = placeholder;
                    }
                    rawTokensByPlaceholderIndex!.Add(msg.RawOutputTokens!);
                    rawBoundaryWhitespaceByPlaceholderIndex!.Add(msg.RawPromptTrailingWhitespace);
                    rawGenerationSuffixByPlaceholderIndex!.Add(msg.RawGenerationSuffix);
                    placeholderCount++;
                }

                // Breakpoints are NUMBERED in the order they appear in the text,
                // which is what lets the strip pass walk them forwards and record
                // final indices in one go. So claim the indices in that order too:
                // the tool breakpoint sits in front of the content, the part
                // offsets are interior and ascending, and the message-scoped
                // marker closes the content.
                string breakpointPrefix = needsToolsMarker
                    ? MakeBreakpoint(breakpointCount++)
                    : string.Empty;

                if (hasPartMarkers && hasRawTokens)
                {
                    // Character offsets cannot be translated into the original
                    // generated-token stream after content is replaced by a raw
                    // placeholder. Preserve explicit-cache mode conservatively by
                    // collapsing them to one boundary immediately before the raw
                    // tokens; silently dropping every marker would mean cache-all.
                    breakpointPrefix += MakeBreakpoint(breakpointCount++);
                }
                else if (hasPartMarkers)
                {
                    // Part-scoped markers address offsets into the ORIGINAL
                    // content, so exact placement is possible while that content
                    // is still present.
                    newContent = InsertBreakpointsAtOffsets(
                        newContent, msg.ContentCacheBreakpoints!, ref breakpointCount);
                }

                string breakpointSuffix = hasMarker
                    ? MakeBreakpoint(breakpointCount++)
                    : string.Empty;

                if (useRawToolCallReplayMarker)
                {
                    // Content is rendered after the canonical tool/result scan, so a
                    // cache marker left there would point at the wrong part of the
                    // prompt (or be dropped). Keep both sides attached to the relocated
                    // raw-token placeholder instead.
                    rawToolCallReplayPlaceholder = breakpointPrefix
                        + rawToolCallReplayPlaceholder
                        + breakpointSuffix;
                    newContent = string.Empty;
                }
                else
                {
                    newContent = breakpointPrefix + newContent + breakpointSuffix;
                }

                if (needsToolResultProof)
                {
                    string marker = MakeToolResultProof();
                    newContent = InsertToolResultProof(newContent, marker);
                    toolResultProofs ??= new List<ToolResultProof>();
                    toolResultProofs.Add(new ToolResultProof(
                        marker,
                        NormalizeNewlines(StripBreakpointMarkers(newContent)).Trim()));
                }

                renderedMessages.Add(new ChatMessage
                {
                    Role = msg.Role,
                    Content = newContent,
                    // Don't carry Thinking through the template - the raw tokens already contain it.
                    Thinking = hasRawTokens ? null : msg.Thinking,
                    ToolCalls = hasRawTokens && !useRawToolCallReplayMarker
                        ? null
                        : msg.ToolCalls,
                    ToolCallId = msg.ToolCallId,
                    ImagePaths = msg.ImagePaths,
                    ImageTimestamps = msg.ImageTimestamps,
                    AudioPaths = msg.AudioPaths,
                    IsVideo = msg.IsVideo,
                    // Kept for the Jinja context's narrowly scoped Gemma 4 replay.
                    RawOutputTokens = msg.RawOutputTokens,
                    RawPromptTrailingWhitespace = msg.RawPromptTrailingWhitespace,
                    RawGenerationSuffix = msg.RawGenerationSuffix,
                    RawToolCallReplayPlaceholder = rawToolCallReplayPlaceholder,
                });
            }

            List<ChatMessage> messagesForRender = renderedMessages ?? messages;

            string text = _innerRenderer.Render(
                chatTemplate,
                messagesForRender,
                addGenerationPrompt,
                architecture,
                tools,
                enableThinking,
                reasoningEffort);

            text = ValidateAndStripToolResultProofs(
                text, toolResultProofs, out bool renderedToolResultsProven);
            bool toolResultsProven = toolResultsProvable && renderedToolResultsProven;

            if (rawToolCallReplayPlaceholderIndices != null)
            {
                foreach (int index in rawToolCallReplayPlaceholderIndices)
                {
                    string placeholder = MakePlaceholder(index);
                    int first = text.IndexOf(placeholder, StringComparison.Ordinal);
                    if (first < 0
                        || text.IndexOf(
                            placeholder,
                            first + placeholder.Length,
                            StringComparison.Ordinal) >= 0)
                    {
                        throw new RawToolCallReplayUnavailableException();
                    }
                }

                // Verify before tokenization so a rejected optimization does not pay for
                // a full prompt encode that the safe retry will immediately discard.
                if (!toolResultsProven)
                    throw new RawToolCallReplayUnavailableException();
            }

            // Fast path: no placeholders and no breakpoints -> just tokenize the whole rendered string.
            if (placeholderCount == 0 && breakpointCount == 0)
            {
                return new RenderPass(
                    tokenizer.Encode(text, addSpecial: true),
                    text,
                    toolCallRoundsLeftToTemplate,
                    explicitBreakpoints: null,
                    toolResultsProven);
            }

            // Some chat templates (notably Gemma 4) call a strip_thinking filter on
            // assistant content, which would silently delete a prefix injected via the
            // Content field. To work around this AND to keep the renderer template-agnostic,
            // we inject the architecture-specific generation suffix as POST-render text
            // patching: walk the rendered text and prepend the suffix before each placeholder.
            // Templates that carry a thinking channel emit an EMPTY `<think></think>`
            // block ahead of a past assistant turn, to tell the model that turn's
            // reasoning was dropped. The KV cache has no such block: turn N forwarded
            // `<think>` + the model's real reasoning tokens. Left in place, that empty
            // block inserts four tokens (`<think>`, `\n\n`, `</think>`, `\n\n`) right
            // at the first assistant boundary, so the re-rendered prefix diverges there
            // and every multi-turn request re-prefills the whole conversation. Drop it
            // before injecting the suffix that reproduces what the cache actually saw.
            if (ChatProtocolRegistry.For(architecture)?.EmitsEmptyThinkBlockForPastTurns?.Invoke(enableThinking)
                ?? enableThinking)
            {
                text = StripEmptyThinkBlockBeforePlaceholders(text);
            }

            // What this request's generation prompt would end with is the fallback;
            // a turn the transcript tracked knows what ITS prompt ended with, and that
            // is what the cache holds in front of its raw tokens.
            string suffix = GetAssistantGenerationSuffix(architecture, enableThinking);
            IReadOnlyList<string?> recordedSuffixes =
                rawGenerationSuffixByPlaceholderIndex ?? (IReadOnlyList<string?>)Array.Empty<string?>();
            bool anyRecordedSuffix = false;
            for (int i = 0; i < recordedSuffixes.Count && !anyRecordedSuffix; i++)
                anyRecordedSuffix = !string.IsNullOrEmpty(recordedSuffixes[i]);
            if (!string.IsNullOrEmpty(suffix) || anyRecordedSuffix)
                text = InjectSuffixBeforePlaceholders(text, suffix, recordedSuffixes);

            // The mirror image of the suffix injection: some templates emit MORE assistant
            // framing for a past turn than the generation prompt did, and the raw tokens
            // already contain their own. See GetTemplateAssistantHeaderAnchor.
            string headerAnchor = GetTemplateAssistantHeaderAnchor(architecture);
            if (!string.IsNullOrEmpty(headerAnchor))
                text = StripTemplateAssistantHeaders(text, headerAnchor);

            // Reproduce the exact whitespace that preceded each raw generation. Prompt
            // tails are structural: Gemma 4's ordinary generation prompt ends in a newline,
            // while its tool-result continuation ends in a control marker. Inferring every
            // OLD boundary from the NEW prompt's last character made those two shapes flip
            // an old newline on alternate rounds and destroyed the live-cache prefix.
            //
            // RawPromptTrailingWhitespace is authoritative for newly tracked rounds. Null
            // identifies legacy/client-provided history, for which the old final-character
            // heuristic remains as a compatibility fallback.
            bool rendererStrippedTrailingWhitespace =
                text.Length > 0 && !char.IsWhiteSpace(text[text.Length - 1]);
            text = NormalizeWhitespaceBeforeEachPlaceholder(
                text,
                rawBoundaryWhitespaceByPlaceholderIndex is { } boundaries
                    ? boundaries
                    : Array.Empty<string?>(),
                trimUnknownBoundaries: rendererStrippedTrailingWhitespace);

            List<int> tokens = TokenizeAndReplacePlaceholderSpans(
                tokenizer,
                text,
                rawTokensByPlaceholderIndex ?? new List<List<int>>(),
                breakpointCount,
                out List<int>? explicitBreakpoints);

            return new RenderPass(
                tokens,
                text,
                toolCallRoundsLeftToTemplate,
                explicitBreakpoints,
                toolResultsProven);
        }

        /// <summary>
        /// True when every tool-calling round the template was asked to rebuild appears in
        /// the rendered prompt as the exact token run the model generated.
        ///
        /// <para>
        /// This is deliberately measured in TOKENS rather than text. Reproducing the round
        /// as far as the KV cache is concerned means reproducing its tokens; a template
        /// that re-emits the same words with one different merge at a boundary has not
        /// reproduced it, and text comparison would say it had.
        /// </para>
        /// </summary>
        private static bool ReproducesEveryToolCallRound(
            ITokenizer tokenizer,
            List<int> promptTokens,
            List<UnsplicedToolCallRound> toolCallRounds)
        {
            int searchStart = 0;
            for (int i = 0; i < toolCallRounds.Count; i++)
            {
                UnsplicedToolCallRound round = toolCallRounds[i];
                bool found = false;
                while (searchStart + round.RawTokens.Count <= promptTokens.Count)
                {
                    int at = FindSubsequence(promptTokens, round.RawTokens, searchStart);
                    if (at < 0)
                        break;

                    if (round.BoundaryWhitespace == null
                        || string.Equals(
                            TrailingWhitespaceBeforeToken(tokenizer, promptTokens, at),
                            round.BoundaryWhitespace,
                            StringComparison.Ordinal))
                    {
                        searchStart = at + round.RawTokens.Count;
                        found = true;
                        break;
                    }
                    searchStart = at + 1;
                }
                if (!found)
                    return false;
            }
            return true;
        }

        private static string TrailingWhitespaceBeforeToken(
            ITokenizer tokenizer, List<int> tokens, int tokenIndex)
        {
            if (tokenIndex <= 0)
                return string.Empty;

            var reverseChunks = new List<string>();
            for (int i = tokenIndex - 1; i >= 0; i--)
            {
                string piece = tokenizer.Decode(new List<int> { tokens[i] });
                int start = piece.Length;
                while (start > 0 && char.IsWhiteSpace(piece[start - 1]))
                    start--;
                if (start < piece.Length)
                    reverseChunks.Add(piece.Substring(start));
                if (start > 0)
                    break;
            }
            if (reverseChunks.Count == 0)
                return string.Empty;

            var result = new System.Text.StringBuilder();
            for (int i = reverseChunks.Count - 1; i >= 0; i--)
                result.Append(reverseChunks[i]);
            return result.ToString();
        }

        /// <summary>
        /// Put an unpredictable render-only marker INSIDE a tool result, after its first
        /// visible character. Keeping the marker away from either edge preserves the
        /// template's ordinary Trim/TrimEnd behavior. Cache-control sentinels are skipped
        /// because they are zero-width prompt decorations rather than result text.
        /// </summary>
        private static string InsertToolResultProof(string content, string marker)
        {
            int i = 0;
            while (i < content.Length)
            {
                if (content[i] == BreakpointSentinel)
                {
                    int markerEnd = content.IndexOf(BreakpointSentinel, i + 1);
                    if (markerEnd >= 0)
                    {
                        i = markerEnd + 1;
                        continue;
                    }
                }

                if (!char.IsWhiteSpace(content[i]))
                {
                    int insertAt = i + 1;
                    if (char.IsHighSurrogate(content[i])
                        && insertAt < content.Length
                        && char.IsLowSurrogate(content[insertAt]))
                    {
                        insertAt++;
                    }
                    return content.Insert(insertAt, marker);
                }
                i++;
            }

            // The caller excludes empty/all-whitespace results. Keep this fail-safe so
            // future callers cannot accidentally put a proof at a trimming boundary.
            return content;
        }

        private static string MakeToolResultProof()
            => $"{ToolResultProofSentinel}T{Guid.NewGuid():N}{ToolResultProofSentinel}";

        /// <summary>
        /// Prove direct ownership rather than looking for the result text anywhere in
        /// the prompt. A full content run containing its unguessable marker must appear
        /// exactly once and in message order, then remove every proof in one forward
        /// pass. Thus a dropped result cannot be mistaken for the same short word in a
        /// later role header or user message. The work is linear in prompt size plus the
        /// total size of the tool results, even for long code-exec histories.
        /// </summary>
        private static string ValidateAndStripToolResultProofs(
            string renderedText,
            List<ToolResultProof>? proofs,
            out bool proven)
        {
            if (proofs == null || proofs.Count == 0)
            {
                proven = true;
                return renderedText;
            }

            var proofIndexByMarker = new Dictionary<string, int>(
                proofs.Count, StringComparer.Ordinal);
            for (int i = 0; i < proofs.Count; i++)
                proofIndexByMarker.Add(proofs[i].Marker, i);

            string haystack = NormalizeNewlines(StripBreakpointMarkers(renderedText));
            var markerPositions = new int[proofs.Count];
            Array.Fill(markerPositions, -1);

            bool proofsValid = true;
            ScanToolResultProofs(
                haystack,
                proofIndexByMarker,
                (proofIndex, markerAt) =>
                {
                    if (markerPositions[proofIndex] >= 0)
                        proofsValid = false;
                    else
                        markerPositions[proofIndex] = markerAt;
                });

            int previousMarkerAt = -1;
            for (int i = 0; i < proofs.Count; i++)
            {
                ToolResultProof proof = proofs[i];
                int markerAt = markerPositions[i];
                int wrappedAt = markerAt - proof.MarkerOffset;
                if (markerAt < 0
                    || proof.MarkerOffset < 0
                    || markerAt <= previousMarkerAt
                    || wrappedAt < 0
                    || wrappedAt + proof.ExpectedWrappedContent.Length > haystack.Length
                    || string.CompareOrdinal(
                        haystack,
                        wrappedAt,
                        proof.ExpectedWrappedContent,
                        0,
                        proof.ExpectedWrappedContent.Length) != 0)
                {
                    proofsValid = false;
                }
                previousMarkerAt = markerAt;
            }

            proven = proofsValid;

            var sb = new System.Text.StringBuilder(renderedText.Length);
            int copied = 0;
            ScanToolResultProofs(
                renderedText,
                proofIndexByMarker,
                (_, markerAt) =>
                {
                    sb.Append(renderedText, copied, markerAt - copied);
                    copied = markerAt + ToolResultProofLength;
                });
            sb.Append(renderedText, copied, renderedText.Length - copied);
            return sb.ToString();
        }

        private const int ToolResultProofLength = 35;

        /// <summary>
        /// Visit only recognized proof markers in one left-to-right pass. Unknown U+E002
        /// text is ordinary prompt content and is not removed.
        /// </summary>
        private static void ScanToolResultProofs(
            string text,
            Dictionary<string, int> proofIndexByMarker,
            Action<int, int> visit)
        {
            int searchStart = 0;
            while (searchStart < text.Length)
            {
                int markerAt = text.IndexOf(ToolResultProofSentinel, searchStart);
                if (markerAt < 0)
                    return;

                if (markerAt + ToolResultProofLength <= text.Length
                    && text[markerAt + 1] == 'T'
                    && text[markerAt + ToolResultProofLength - 1] == ToolResultProofSentinel)
                {
                    string candidate = text.Substring(markerAt, ToolResultProofLength);
                    if (proofIndexByMarker.TryGetValue(candidate, out int proofIndex))
                    {
                        visit(proofIndex, markerAt);
                        searchStart = markerAt + ToolResultProofLength;
                        continue;
                    }
                }

                searchStart = markerAt + 1;
            }
        }

        private static string StripBreakpointMarkers(string text)
        {
            int markerStart = text.IndexOf(BreakpointSentinel);
            if (markerStart < 0)
                return text;

            var sb = new System.Text.StringBuilder(text.Length);
            int copied = 0;
            while (markerStart >= 0)
            {
                sb.Append(text, copied, markerStart - copied);

                int markerEnd = text.IndexOf(BreakpointSentinel, markerStart + 1);
                if (markerEnd < 0)
                {
                    // A malformed/user-supplied lone sentinel is ordinary prompt text,
                    // not one of MakeBreakpoint's paired markers.
                    sb.Append(text, markerStart, text.Length - markerStart);
                    return sb.ToString();
                }

                copied = markerEnd + 1;
                markerStart = text.IndexOf(BreakpointSentinel, copied);
            }

            sb.Append(text, copied, text.Length - copied);
            return sb.ToString();
        }

        private static string NormalizeNewlines(string? s)
        {
            if (string.IsNullOrEmpty(s))
                return string.Empty;
            return s.Replace("\r\n", "\n").Replace('\r', '\n');
        }

        // Architectures whose "the template lost the round but splicing would lose the
        // tool results" dead end has already been reported. Both halves of the trade are
        // bad; the prompt stays correct and pays the prefill, but a silent choice between
        // two costs is exactly the kind of thing that goes unnoticed for months.
        private static readonly HashSet<string> ToolCallSplicingWarned = new(StringComparer.Ordinal);

        private static void WarnToolCallSplicingUnavailableOnce(string? architecture)
        {
            lock (ToolCallSplicingWarned)
            {
                if (!ToolCallSplicingWarned.Add(architecture ?? string.Empty)) return;
            }
            Console.Error.WriteLine(
                $"[KVCachePromptRenderer] '{architecture}': the chat template did not reproduce a past " +
                "tool-calling round's generated tokens, and splicing them back would drop the tool results " +
                "from the prompt. Keeping the results, so that round re-prefills instead of continuing the " +
                "live KV cache. Reported once per architecture.");
        }

        /// <summary>
        /// Remove an EMPTY thinking block (<c>&lt;think&gt;</c>, only whitespace,
        /// <c>&lt;/think&gt;</c>, then whitespace) that the chat template emitted
        /// immediately before a spliced assistant turn.
        ///
        /// Only an empty block is removed. A block with real reasoning text in it was
        /// not produced by this mechanism and is left alone, so a client that replays
        /// prior reasoning verbatim still renders it.
        /// </summary>
        internal static string StripEmptyThinkBlockBeforePlaceholders(string text)
        {
            const string open = "<think>";
            const string close = "</think>";
            if (text.IndexOf(PlaceholderSentinel) < 0 || text.IndexOf(open, StringComparison.Ordinal) < 0)
                return text;

            var sb = new System.Text.StringBuilder(text.Length);
            int searchPos = 0;
            while (searchPos < text.Length)
            {
                int sentinel = text.IndexOf(PlaceholderSentinel, searchPos);
                if (sentinel < 0)
                {
                    sb.Append(text, searchPos, text.Length - searchPos);
                    break;
                }

                // Cache-control markers immediately before the raw run are zero-width
                // prompt decorations. Look through them when finding the empty block,
                // but preserve them at the raw boundary.
                int decorationsStart = FindBreakpointRunStart(text, searchPos, sentinel);

                // Walk back over trailing whitespace, then require "</think>".
                int cursor = decorationsStart;
                while (cursor > searchPos && char.IsWhiteSpace(text[cursor - 1]))
                    cursor--;
                if (cursor - searchPos >= close.Length
                    && string.CompareOrdinal(text, cursor - close.Length, close, 0, close.Length) == 0)
                {
                    cursor -= close.Length;
                    // Walking back over whitespace has to land exactly on the end of
                    // "<think>" - anything else means the block held real reasoning.
                    while (cursor > searchPos && char.IsWhiteSpace(text[cursor - 1]))
                        cursor--;
                    if (cursor - searchPos >= open.Length
                        && string.CompareOrdinal(text, cursor - open.Length, open, 0, open.Length) == 0)
                    {
                        // Copy up to the "<think>" and drop the whole empty block,
                        // INCLUDING the whitespace the template put after "</think>".
                        // Keeping that whitespace would leave `assistant\n` + `\n\n`
                        // where the cache has `assistant\n` + `<think>\n`, which just
                        // moves the divergence rather than removing it.
                        sb.Append(text, searchPos, (cursor - open.Length) - searchPos);
                        if (decorationsStart < sentinel)
                            sb.Append(text, decorationsStart, sentinel - decorationsStart);
                        sb.Append(text[sentinel]);
                        searchPos = sentinel + 1;
                        continue;
                    }
                }

                sb.Append(text, searchPos, sentinel - searchPos + 1);
                searchPos = sentinel + 1;
            }
            return sb.ToString();
        }

        private static string NormalizeWhitespaceBeforeEachPlaceholder(
            string text,
            IReadOnlyList<string?> boundaryWhitespace,
            bool trimUnknownBoundaries)
        {
            if (!NeedsWhitespaceNormalization(text, boundaryWhitespace, trimUnknownBoundaries))
                return text;

            var sb = new System.Text.StringBuilder(text.Length);
            int searchPos = 0;
            int placeholderIndex = 0;
            while (searchPos < text.Length)
            {
                int sentinel = text.IndexOf(PlaceholderSentinel, searchPos);
                if (sentinel < 0)
                {
                    sb.Append(text, searchPos, text.Length - searchPos);
                    break;
                }

                string? exact = placeholderIndex < boundaryWhitespace.Count
                    ? boundaryWhitespace[placeholderIndex]
                    : null;
                if (!IsValidBoundaryWhitespace(exact))
                    exact = null;
                if (exact != null || trimUnknownBoundaries)
                {
                    // Cache-control markers may be attached immediately before a raw
                    // placeholder. The generation-boundary whitespace belongs BEFORE
                    // those zero-width markers; otherwise the marker stops this trim at
                    // its private-use sentinel and an old template newline survives in
                    // addition to the exact recorded newline.
                    int markerRunStart = FindBreakpointRunStart(text, searchPos, sentinel);
                    int copyEnd = markerRunStart;
                    while (copyEnd > searchPos && char.IsWhiteSpace(text[copyEnd - 1]))
                        copyEnd--;
                    sb.Append(text, searchPos, copyEnd - searchPos);
                    if (exact != null)
                        sb.Append(exact);
                    sb.Append(text, markerRunStart, sentinel - markerRunStart);
                }
                else
                {
                    sb.Append(text, searchPos, sentinel - searchPos);
                }

                int sentinelEnd = text.IndexOf(PlaceholderSentinel, sentinel + 1);
                if (sentinelEnd < 0)
                    throw new InvalidOperationException(
                        "Malformed KV-cache placeholder: opening sentinel without matching close.");
                sb.Append(text, sentinel, sentinelEnd - sentinel + 1);
                searchPos = sentinelEnd + 1;
                placeholderIndex++;
            }
            return sb.ToString();
        }

        private static bool NeedsWhitespaceNormalization(
            string text,
            IReadOnlyList<string?> boundaryWhitespace,
            bool trimUnknownBoundaries)
        {
            int searchPos = 0;
            int placeholderIndex = 0;
            while (searchPos < text.Length)
            {
                int sentinel = text.IndexOf(PlaceholderSentinel, searchPos);
                if (sentinel < 0)
                    return false;

                string? exact = placeholderIndex < boundaryWhitespace.Count
                    ? boundaryWhitespace[placeholderIndex]
                    : null;
                if (!IsValidBoundaryWhitespace(exact))
                    exact = null;
                if (exact != null || trimUnknownBoundaries)
                {
                    int decorationsStart = FindBreakpointRunStart(text, searchPos, sentinel);
                    int whitespaceStart = decorationsStart;
                    while (whitespaceStart > searchPos
                        && char.IsWhiteSpace(text[whitespaceStart - 1]))
                    {
                        whitespaceStart--;
                    }

                    int existingLength = decorationsStart - whitespaceStart;
                    if (exact == null)
                    {
                        if (existingLength > 0)
                            return true;
                    }
                    else if (existingLength != exact.Length
                        || string.CompareOrdinal(
                            text, whitespaceStart,
                            exact, 0, exact.Length) != 0)
                    {
                        return true;
                    }
                }

                int sentinelEnd = text.IndexOf(PlaceholderSentinel, sentinel + 1);
                if (sentinelEnd < 0)
                    throw new InvalidOperationException(
                        "Malformed KV-cache placeholder: opening sentinel without matching close.");
                searchPos = sentinelEnd + 1;
                placeholderIndex++;
            }
            return false;
        }

        private static bool IsValidBoundaryWhitespace(string? value)
        {
            // Generation-prompt tails are tiny structural runs. Treat malformed or
            // externally supplied non-whitespace metadata as legacy/unknown rather than
            // allowing it to inject arbitrary prompt text or a pathological allocation.
            if (value == null)
                return false;
            if (value.Length > 1024)
                return false;
            for (int i = 0; i < value.Length; i++)
                if (!char.IsWhiteSpace(value[i]))
                    return false;
            return true;
        }

        /// <summary>
        /// Walk backwards across adjacent renderer-generated breakpoint markers ending
        /// immediately before a raw-token placeholder.
        /// </summary>
        private static int FindBreakpointRunStart(string text, int lowerBound, int end)
        {
            int cursor = end;
            while (cursor > lowerBound && text[cursor - 1] == BreakpointSentinel)
            {
                if (cursor - 2 < lowerBound)
                    break;
                int opening = text.LastIndexOf(BreakpointSentinel, cursor - 2);
                if (opening < lowerBound)
                    break;

                int markerLength = cursor - opening;
                if (markerLength < 4 || text[opening + 1] != 'B')
                {
                    break;
                }

                bool digits = true;
                for (int i = opening + 2; i < cursor - 1; i++)
                {
                    if (!char.IsDigit(text[i]))
                    {
                        digits = false;
                        break;
                    }
                }
                if (!digits)
                    break;

                cursor = opening;
            }
            return cursor;
        }

        private static string TrailingWhitespace(string text)
        {
            if (string.IsNullOrEmpty(text))
                return string.Empty;
            int start = text.Length;
            while (start > 0 && char.IsWhiteSpace(text[start - 1])) start--;
            return start == text.Length ? string.Empty : text.Substring(start);
        }

        /// <summary>
        /// Put the generation framing back in front of each spliced raw run: the run's
        /// own recorded suffix where the transcript has one (<paramref name="recordedSuffixes"/>,
        /// indexed in placeholder order), <paramref name="defaultSuffix"/> otherwise.
        /// </summary>
        private static string InjectSuffixBeforePlaceholders(
            string text, string defaultSuffix, IReadOnlyList<string?> recordedSuffixes)
        {
            var sb = new System.Text.StringBuilder(text.Length + defaultSuffix.Length * 4);
            int searchPos = 0;
            int placeholderIndex = 0;
            while (searchPos < text.Length)
            {
                int sentinel = text.IndexOf(PlaceholderSentinel, searchPos);
                if (sentinel < 0)
                {
                    sb.Append(text, searchPos, text.Length - searchPos);
                    break;
                }
                // A recorded empty string is a statement — that prompt ended on no
                // suffix at all — and is honoured; only null means "unknown, use the
                // family's default for this request".
                string suffix = placeholderIndex < recordedSuffixes.Count && recordedSuffixes[placeholderIndex] != null
                    ? recordedSuffixes[placeholderIndex]!
                    : defaultSuffix;
                placeholderIndex++;
                int decorationsStart = FindBreakpointRunStart(text, searchPos, sentinel);
                sb.Append(text, searchPos, decorationsStart - searchPos);
                sb.Append(suffix);
                if (decorationsStart < sentinel)
                    sb.Append(text, decorationsStart, sentinel - decorationsStart);
                int sentinelEnd = text.IndexOf(PlaceholderSentinel, sentinel + 1);
                if (sentinelEnd < 0)
                    throw new InvalidOperationException(
                        "Malformed KV-cache placeholder: opening sentinel without matching close.");
                sb.Append(text, sentinel, sentinelEnd - sentinel + 1);
                searchPos = sentinelEnd + 1;
            }
            return sb.ToString();
        }

        internal static string MakePlaceholder(int index)
        {
            // Encoded as PUA-sentinel + DigitsBase32 + PUA-sentinel.
            // Using two sentinels makes the split unambiguous in the tokenizer regex's eyes,
            // and the digits guarantee that two adjacent placeholders never get merged.
            return $"{PlaceholderSentinel}R{index:D4}{PlaceholderSentinel}";
        }

        internal static string MakeBreakpoint(int index)
        {
            return $"{BreakpointSentinel}B{index:D4}{BreakpointSentinel}";
        }

        /// <summary>
        /// Splice a breakpoint sentinel into <paramref name="content"/> at each of
        /// <paramref name="offsets"/>, which are character positions in ascending
        /// order. Offsets are clamped into range and never allowed to move
        /// backwards, so a marker whose offset does not fit the content (a caller
        /// that rewrote the text after parsing it) degrades to a breakpoint at the
        /// nearest legal position rather than throwing.
        /// </summary>
        private static string InsertBreakpointsAtOffsets(
            string content, List<int> offsets, ref int breakpointCount)
        {
            var sb = new System.Text.StringBuilder(content.Length + offsets.Count * 8);
            int copied = 0;
            for (int i = 0; i < offsets.Count; i++)
            {
                int at = offsets[i];
                if (at < copied) at = copied;
                if (at > content.Length) at = content.Length;

                sb.Append(content, copied, at - copied);
                sb.Append(MakeBreakpoint(breakpointCount++));
                copied = at;
            }
            sb.Append(content, copied, content.Length - copied);
            return sb.ToString();
        }

        /// <summary>
        /// Tokenize <paramref name="text"/> as a SINGLE string (so the BPE/SentencePiece
        /// merging decisions at segment boundaries match exactly what the renderer would
        /// have produced for an entire turn-1-style prompt), then replace each occurrence
        /// of the placeholder marker's tokens with the corresponding raw output tokens.
        ///
        /// Tokenizing the whole text in one shot is what makes this approach
        /// renderer-agnostic: it doesn't matter whether the chat template applies a
        /// final TrimEnd, whether it appends additional suffixes, or whether the BPE
        /// tokenizer would have merged the boundary differently between an interior
        /// segment and a trailing one. The placeholder text is built from PUA codepoints
        /// (Unicode <see cref="PlaceholderSentinel"/>) plus ASCII digits/letters so its
        /// tokenization is locally-stable: the BPE pretokenizer regex always isolates
        /// these characters into their own chunks regardless of surrounding context.
        /// </summary>
        private static List<int> TokenizeAndReplacePlaceholderSpans(
            ITokenizer tokenizer,
            string text,
            List<List<int>> rawTokensByPlaceholderIndex,
            int breakpointCount,
            out List<int>? explicitBreakpoints)
        {
            explicitBreakpoints = null;

            // Step 1: tokenize the rendered text as a whole.
            List<int> tokens = tokenizer.Encode(text, addSpecial: true);

            // Step 2: for each placeholder, find its token span and replace.
            // Working backwards (highest-numbered placeholder first) keeps earlier
            // positions stable as we splice (which can lengthen or shorten the list).
            int placeholderCount = rawTokensByPlaceholderIndex.Count;
            for (int i = placeholderCount - 1; i >= 0; i--)
            {
                string placeholder = MakePlaceholder(i);
                List<int> placeholderTokens = tokenizer.Encode(placeholder, addSpecial: false);

                int spanStart = FindSubsequence(tokens, placeholderTokens);
                if (spanStart < 0)
                    throw new InvalidOperationException(
                        $"Could not locate placeholder #{i} ({placeholder.Length} chars, {placeholderTokens.Count} tokens) in tokenized output. " +
                        "This usually means the tokenizer is treating the placeholder differently in context vs in isolation; " +
                        "consider switching to a placeholder character that survives BPE pretokenization.");

                tokens.RemoveRange(spanStart, placeholderTokens.Count);
                tokens.InsertRange(spanStart, rawTokensByPlaceholderIndex[i]);
            }

            // Step 3: strip the explicit cache breakpoints, recording where each
            // one fell.
            //
            // Breakpoints are numbered in render order and walked FORWARDS, which
            // is what makes the recorded indices final. Step 2 goes backwards
            // because it splices raw tokens in place and needs the positions it
            // has not reached yet to stay put; this loop only ever deletes, so
            // the opposite holds. Removing breakpoint i shifts every later
            // breakpoint left by its length, and going forwards means that shift
            // has already been applied by the time we search for i+1 - while
            // deletions after a recorded index cannot disturb it. Walking
            // backwards instead records each index in the coordinate space of an
            // array that still contains all the earlier breakpoints, leaving
            // every breakpoint but the first too large by the length of the ones
            // preceding it.
            //
            // Unlike a placeholder, a breakpoint that cannot be found is skipped
            // rather than fatal: a template is free to drop the content it was
            // attached to (Gemma 4's strip_thinking filter does exactly that),
            // and a cache hint is not worth failing a completion over. Dropping
            // an interior breakpoint only shortens the marked region. Dropping
            // the LAST one raises the ceiling to the next breakpoint down. If
            // every sentinel was dropped, the non-null empty result preserves
            // explicit cache-none mode instead of silently widening to the whole
            // prompt.
            if (breakpointCount > 0)
            {
                explicitBreakpoints = new List<int>(breakpointCount);
                for (int i = 0; i < breakpointCount; i++)
                {
                    string breakpoint = MakeBreakpoint(i);
                    List<int> breakpointTokens = tokenizer.Encode(breakpoint, addSpecial: false);

                    int spanStart = FindSubsequence(tokens, breakpointTokens);
                    if (spanStart >= 0)
                    {
                        tokens.RemoveRange(spanStart, breakpointTokens.Count);
                        explicitBreakpoints.Add(spanStart);
                    }
                }
            }

            return tokens;
        }

        private static int FindSubsequence(List<int> haystack, List<int> needle)
            => FindSubsequence(haystack, needle, 0);

        private static int FindSubsequence(
            List<int> haystack, List<int> needle, int startIndex)
        {
            if (needle.Count == 0 || haystack.Count < needle.Count)
                return -1;

            int last = haystack.Count - needle.Count;
            for (int i = Math.Max(0, startIndex); i <= last; i++)
            {
                bool match = true;
                for (int j = 0; j < needle.Count; j++)
                {
                    if (haystack[i + j] != needle[j])
                    {
                        match = false;
                        break;
                    }
                }
                if (match)
                    return i;
            }
            return -1;
        }
    }
}
