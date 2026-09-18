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
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.AgentHost.CodeExec;
using TensorSharp.AgentHost.Skills;
using TensorSharp.Server.ProtocolAdapters;

namespace TensorSharp.Server.Skills
{
    /// <summary>Runs one generation and reports it as raw text plus metrics.</summary>
    internal delegate IAsyncEnumerable<ChatStreamUpdate> SkillChatGeneration(
        List<ChatMessage> messages,
        List<ToolFunction> tools,
        CancellationToken cancellationToken);

    /// <summary>
    /// Wraps a chat generation in the progressive-disclosure loop: generate, answer any
    /// <c>skills_*</c> call in process, generate again, and stream only the round that
    /// finally answers.
    ///
    /// <para>
    /// <b>It streams.</b> The loop runs its own output parser — it has to, because it is
    /// looking for <c>skills_read</c> calls to answer — and forwards the SEPARATED
    /// pieces rather than the raw text: content as content, reasoning as reasoning, tool
    /// markup not at all. Every update it yields is marked
    /// <see cref="ChatStreamUpdate.IsParsed"/>, and an adapter that sees that must not
    /// parse again.
    /// </para>
    /// <para>
    /// That indirection is the whole design, and it was arrived at the hard way. The
    /// first version buffered each round and replayed only the last one, because
    /// forwarding a round's raw tokens would hand <c>skills_read</c> to a client with no
    /// implementation for it. It worked, and it silently cost every skills request its
    /// streaming: measured on the Web UI, 26 SSE content frames starting 0.17 s in
    /// became 3 frames all arriving at 3.28 s. Stopping the forward at the first tool
    /// token instead is not a fix either — it leaves the ADAPTER's parser inside a
    /// half-open tool-call span, and the next round's text is consumed as that call's
    /// arguments, so the answer vanishes rather than merely arriving late. Parsing once,
    /// here, and handing over the pieces avoids both: the markup never exists as far as
    /// the adapter is concerned, so there is no span to be caught inside.
    /// </para>
    /// <para>
    /// <b>Cost.</b> Every round goes through the same <see cref="ChatSession"/>, so the
    /// pipeline records each assistant turn's generated tokens in the session's tracked
    /// history and the next render splices them back rather than re-tokenizing —
    /// which is what keeps the rendered prefix byte-identical from round to round.
    /// </para>
    /// <para>
    /// A round therefore re-prefills only what it added. Measured on gemma-4-E4B
    /// (ggml_metal), a lookup that appends a 7.9 KB skill body reuses 1983 of the
    /// following round's 4197 prompt tokens (47%) — the whole of the previous round,
    /// with only the newly fetched file left to forward — and its time to first token
    /// drops from 2.3 s to 1.3 s. This needed an engine fix:
    /// <c>BatchExecutor.ComputeLiveContinuationLcp</c> used to require the live KV
    /// cache to be an EXACT prefix of the new prompt, and a turn that ends on a
    /// control token the chat template never re-renders (Gemma 4 answers a tool call by
    /// emitting <c>&lt;|tool_response&gt;</c>) failed that test by one token and
    /// re-prefilled everything. It now rewinds a bounded number of trailing tokens.
    /// </para>
    /// </summary>
    internal static class SkillChatLoop
    {
        /// <summary>
        /// Drive <paramref name="generate"/> until the model stops asking for skill
        /// content, then stream the final round.
        /// </summary>
        /// <param name="architecture">The loaded model's architecture, for the detection parser.</param>
        /// <param name="messages">The conversation, with the skill block already injected.</param>
        /// <param name="plan">The request's skills, tools and bounds.</param>
        /// <param name="enableThinking">Passed to the detection parser so reasoning is not mistaken for content.</param>
        /// <param name="generate">Runs one generation.</param>
        /// <param name="logger">Optional.</param>
        /// <param name="cancellationToken">Checked between rounds and before every tool call.</param>
        public static async IAsyncEnumerable<ChatStreamUpdate> RunAsync(
            string architecture,
            List<ChatMessage> messages,
            SkillRequestPlan plan,
            bool enableThinking,
            SkillChatGeneration generate,
            ILogger logger,
            [EnumeratorCancellation] CancellationToken cancellationToken)
        {
            var working = new List<ChatMessage>(messages);
            int maxRounds = Math.Max(1, plan.LoopOptions.MaxRounds);
            bool guardCompletion = plan.CompletionRequirement != null;
            bool repetitionRetried = false;

            int promptTokens = 0;
            int evalTokens = 0;
            int reusedTokens = 0;
            long promptNs = 0;
            long evalNs = 0;
            long totalNs = 0;

            for (int round = 1; round <= maxRounds; round++)
            {
                cancellationToken.ThrowIfCancellationRequested();

                // The loop parses the stream itself, because it has to: it is looking
                // for skills_read calls to answer. Having parsed, it forwards the
                // SEPARATED pieces rather than the raw text, so tool markup never
                // reaches the adapter and every round streams as it decodes.
                var parser = OutputParserFactory.Create(architecture);
                parser.Init(enableThinking, plan.Tools);

                var content = new StringBuilder();
                var thinking = new StringBuilder();
                var calls = new List<ToolCall>();
                var heldAnswer = guardCompletion ? new List<ChatStreamUpdate>() : null;
                ChatStreamUpdate terminal = default;
                string toolBeingWritten = null;
                // The last few kilobytes of the RAW stream, tool markup included. When
                // the engine ends a round for repeating itself, the repeated text is
                // usually inside a tool call being written, which the parser never
                // surfaces as content -- and the note to the model has to quote it.
                var rawTail = new StringBuilder();

                await foreach (ChatStreamUpdate update in
                    generate(working, plan.Tools, cancellationToken).ConfigureAwait(false))
                {
                    if (update.Done)
                    {
                        terminal = update;
                        continue;
                    }
                    if (update.RawGenerationSuffix != null)
                        parser.SetGenerationPromptSuffix(update.RawGenerationSuffix);
                    if (string.IsNullOrEmpty(update.Piece))
                        continue;

                    rawTail.Append(update.Piece);
                    if (rawTail.Length > RawTailChars * 2)
                        rawTail.Remove(0, rawTail.Length - RawTailChars);

                    ParsedOutput delta = parser.Add(update.Piece, false);
                    Accumulate(delta, content, thinking, calls);

                    // Content and reasoning ordinarily go out the moment they are
                    // decoded. A guarded route holds both: small models sometimes put
                    // user-facing success claims in the reasoning channel, so buffering
                    // only content still leaks the very false claim this guard rejects.
                    // A tool call does not stream as an answer either; it is ours to
                    // answer or the caller's to service.
                    if (guardCompletion)
                    {
                        if (!string.IsNullOrEmpty(delta.Content) || !string.IsNullOrEmpty(delta.Thinking))
                            heldAnswer.Add(ChatStreamUpdate.Parsed(delta.Content, delta.Thinking, null));
                    }
                    else if (!string.IsNullOrEmpty(delta.Content) || !string.IsNullOrEmpty(delta.Thinking))
                    {
                        yield return ChatStreamUpdate.Parsed(delta.Content, delta.Thinking, null);
                    }

                    // The call's BODY does stream — as progress, not as content. A
                    // shell call can be a whole heredoc written in silence otherwise;
                    // the UI shows "writing code" and the draft, and nothing here is
                    // handed to the client as an answer.
                    toolBeingWritten = delta.ToolCallName ?? toolBeingWritten;
                    if (!string.IsNullOrEmpty(delta.ToolCallText))
                        yield return ChatStreamUpdate.ToolProgress("writing", toolBeingWritten, delta.ToolCallText);
                }

                promptTokens += terminal.PromptTokens;
                evalTokens += terminal.EvalTokens;
                reusedTokens += terminal.KvCacheReusedTokens;
                promptNs += terminal.PromptNs;
                evalNs += terminal.EvalNs;
                totalNs += terminal.TotalNs;

                ParsedOutput flushed = parser.Add(string.Empty, true);
                Accumulate(flushed, content, thinking, calls);
                if (guardCompletion)
                {
                    if (!string.IsNullOrEmpty(flushed.Content) || !string.IsNullOrEmpty(flushed.Thinking))
                        heldAnswer.Add(ChatStreamUpdate.Parsed(flushed.Content, flushed.Thinking, null));
                }
                else if (!string.IsNullOrEmpty(flushed.Content) || !string.IsNullOrEmpty(flushed.Thinking))
                {
                    yield return ChatStreamUpdate.Parsed(flushed.Content, flushed.Thinking, null);
                }

                // A round the engine ended for repeating itself is not a round to act on.
                // Whatever calls it holds were written by a model that had already lost
                // the thread, and the text after the loop began is garbage the user
                // watched scroll by. Observed on a phone: a deck-writing script that
                // degenerated into `","+","+"` for five minutes until Stop was tapped.
                // Say what happened, in the conversation, and let the model try ONCE
                // more, differently; a second loop ends the turn with that explanation.
                if (FinishReasonMapper.IsRepetition(terminal.FinishReason))
                {
                    string loopNote = DescribeRepetition(rawTail.ToString());
                    logger?.LogWarning(LogEventIds.SkillToolInvoked,
                        "skills.loop.repetition round={Round} retried={Retried}: {What}",
                        round, !repetitionRetried, loopNote);

                    foreach (ToolCall call in calls)
                        yield return ChatStreamUpdate.ToolProgress("finished", call.Name);

                    if (!repetitionRetried && round < maxRounds)
                    {
                        repetitionRetried = true;
                        working.Add(new ChatMessage
                        {
                            Role = "assistant",
                            Content = content.ToString(),
                            Thinking = thinking.Length == 0 ? null : thinking.ToString(),
                            ToolCalls = calls.Count == 0 ? null : new List<ToolCall>(calls),
                            RawOutputTokens = terminal.RawOutputTokens != null
                                ? new List<int>(terminal.RawOutputTokens)
                                : null,
                            RawPromptTrailingWhitespace = terminal.RawPromptTrailingWhitespace,
                            RawGenerationSuffix = terminal.RawGenerationSuffix,
                        });
                        string feedback =
                            "Host: your previous output was stopped because it began repeating itself -- "
                            + loopNote + ". That was a generation loop, not a result: nothing from it was run "
                            + "or saved. Do not continue or resend that text. Take a different approach in one "
                            + "short step: produce long or repetitive content with a loop or a library rather "
                            + "than spelling it out, keep each command brief, and if a file must be long, write "
                            + "it in parts with apply_patch. Then continue the task.";
                        working.Add(calls.Count == 0
                            ? new HostCompletionCorrectionMessage { Role = "user", Content = feedback }
                            : BuildResult(plan, feedback));
                        yield return ChatStreamUpdate.Parsed(
                            string.Empty,
                            "\n[the output started repeating itself and was stopped; trying a different approach]\n",
                            null);
                        continue;
                    }

                    yield return ChatStreamUpdate.Parsed(
                        (content.Length == 0 ? string.Empty : "\n\n")
                        + "_(The model's output started repeating itself and was stopped: " + loopNote
                        + ". Ask it to try a different approach, or rephrase the request.)_",
                        null, null);
                    // Quoted, so no UI should add the plain version on top of it.
                    yield return Combine(terminal, promptTokens, evalTokens, reusedTokens, promptNs, evalNs, totalNs)
                        with { RepetitionExplained = true };
                    yield break;
                }

                // Three ways, not two. A call that is neither ours nor a tool the CLIENT
                // declared belongs to nobody, and forwarding it was a silent end to the
                // turn: the Web UI declares no tools and has no handler for the tool-call
                // frame, so the reply — thinking, then a tool call, and no content —
                // rendered as nothing at all. Answered here it costs one round.
                SkillTools.Partition(
                    calls, plan.ClientTools,
                    out List<ToolCall> skillCalls,
                    out List<ToolCall> clientCalls,
                    out List<ToolCall> unknownCalls);

                if (skillCalls.Count == 0 && unknownCalls.Count == 0)
                {
                    if (guardCompletion && clientCalls.Count == 0)
                    {
                        WorkspaceArtifactCompletionResult completion =
                            WorkspaceArtifactCompletion.Verify(plan);
                        if (!completion.Complete)
                        {
                            AppendCompletionCorrection(
                                working, plan, content, thinking, calls, terminal, completion.Reason);
                            await foreach (ChatStreamUpdate correction in RunCompletionCorrectionAsync(
                                architecture, working, plan, enableThinking, generate, logger,
                                correctionRound: round + 1,
                                promptTokens, evalTokens, reusedTokens, promptNs, evalNs, totalNs,
                                cancellationToken).ConfigureAwait(false))
                            {
                                yield return correction;
                            }
                            yield break;
                        }

                        plan.VerifiedArtifact = completion.Artifact;
                        foreach (ChatStreamUpdate held in heldAnswer)
                            yield return held;
                        if (completion.Artifact.HasValue
                            && !ContainsArtifactMarkdownLink(
                                content.ToString(), completion.Artifact.Value.Url))
                        {
                            yield return ChatStreamUpdate.Parsed(
                                (content.Length == 0 ? string.Empty : "\n\n")
                                + DescribeCompletedArtifact(completion.Artifact.Value), null, null);
                        }
                    }

                    // Nothing more to fetch. Any tool calls left are the caller's, and
                    // only the caller knows what they do. The progress line still has to
                    // come down: it went up as the call was written, and the caller
                    // servicing it is not something this stream will see.
                    if (clientCalls.Count > 0)
                    {
                        foreach (ToolCall clientCall in clientCalls)
                            yield return ChatStreamUpdate.ToolProgress("finished", clientCall.Name);
                        yield return ChatStreamUpdate.Parsed(string.Empty, null, clientCalls);
                    }

                    yield return Combine(terminal, promptTokens, evalTokens, reusedTokens, promptNs, evalNs, totalNs);
                    yield break;
                }

                working.Add(new ChatMessage
                {
                    Role = "assistant",
                    Content = content.ToString(),
                    Thinking = thinking.Length == 0 ? null : thinking.ToString(),
                    ToolCalls = new List<ToolCall>(calls),
                    // The tokens as GENERATED, so the next round's render reproduces this
                    // one exactly and the live KV cache can be continued rather than
                    // rebuilt. Without it every round after the first re-prefilled the
                    // entire conversation; SkillAgentLoop has always recorded it, and this
                    // loop could not until the terminal update began carrying it.
                    RawOutputTokens = terminal.RawOutputTokens != null
                        ? new List<int>(terminal.RawOutputTokens)
                        : null,
                    RawPromptTrailingWhitespace = terminal.RawPromptTrailingWhitespace,
                    RawGenerationSuffix = terminal.RawGenerationSuffix,
                });

                foreach (ToolCall unknownCall in unknownCalls)
                {
                    logger?.LogWarning(LogEventIds.SkillToolInvoked,
                        "skills.tool round={Round} tool={Tool} skill={SkillId} path={Path} ok={Ok} bytes={Bytes}",
                        round, unknownCall.Name ?? "-", "-", "-", false, 0);

                    string refusal = SkillTools.DescribeUnknownTool(unknownCall.Name, plan.Tools, KnownSkillIds(plan));
                    working.Add(BuildResult(plan, refusal, unknownCall.Name));

                    // Recorded as an invocation like any other, so the UI's trace shows a
                    // failed step rather than a gap: the round happened, it cost a
                    // generation, and the user watching the trace should see why.
                    lock (plan.Invocations)
                    {
                        plan.Invocations.Add(new SkillToolInvocation(
                            round, unknownCall.Name ?? string.Empty, null, null,
                            Ok: false, refusal.Length));
                    }

                    // The "writing <name>…" line went up while the call was being
                    // generated and only ever comes down on a "finished". Answering the
                    // call without one leaves the user watching a progress line for
                    // something that already happened.
                    yield return ChatStreamUpdate.ToolProgress("finished", unknownCall.Name);
                }

                await foreach (ChatStreamUpdate progress in ExecuteSkillCallsAsync(
                    skillCalls, round, plan, working, logger, cancellationToken).ConfigureAwait(false))
                {
                    yield return progress;
                }

                // A routed workflow's writer is itself the last meaningful step. Once
                // its ordered prerequisite runs and immutable artifact pass the host
                // contract, asking the model for another round merely gives it an
                // opportunity to call more tools (or rewrite a file that is already
                // correct). ExecuteSkillCallsAsync verifies after every finished call
                // and stops the current batch too, so a trailing call from the same
                // generated response cannot undo or duplicate the completed work.
                if (plan.VerifiedArtifact is { } completedArtifact)
                {
                    yield return ChatStreamUpdate.Parsed(
                        DescribeCompletedArtifact(completedArtifact), null, null);
                    yield return Combine(
                        terminal,
                        promptTokens, evalTokens, reusedTokens, promptNs, evalNs, totalNs,
                        finishReason: "stop");
                    yield break;
                }
            }

            // Out of rounds. Tell the model so, in the conversation, and let it answer
            // from what it has — the alternative is returning a bare tool call to a
            // client that cannot service it, which shows the user nothing at all.
            logger?.LogWarning(LogEventIds.SkillLoopCapped,
                "skills.loop.capped rounds={Rounds} skills={Skills}", maxRounds, plan.DescribeSelection());

            working.Add(BuildResult(plan,
                "Error: the limit on tool calls for this turn has been reached. Answer now using what you have "
                + "already read, and say which part you could not check."));

            cancellationToken.ThrowIfCancellationRequested();

            // Parsed like every other round. If the model answers the "limit reached"
            // message with yet another skills_read, its markup must still not reach the
            // client — forwarding it raw here would surface a tool call the caller
            // cannot service, which is the exact stall this loop exists to prevent.
            var finalParser = OutputParserFactory.Create(architecture);
            finalParser.Init(enableThinking, plan.Tools);
            var finalCalls = new List<ToolCall>();
            var finalContent = new StringBuilder();
            var finalThinking = new StringBuilder();
            var finalHeldAnswer = guardCompletion ? new List<ChatStreamUpdate>() : null;
            ChatStreamUpdate finalTerminal = default;

            await foreach (ChatStreamUpdate update in
                generate(working, plan.Tools, cancellationToken).ConfigureAwait(false))
            {
                if (update.Done)
                {
                    finalTerminal = update;
                    continue;
                }
                if (update.RawGenerationSuffix != null)
                    finalParser.SetGenerationPromptSuffix(update.RawGenerationSuffix);
                if (string.IsNullOrEmpty(update.Piece))
                    continue;

                ParsedOutput delta = finalParser.Add(update.Piece, false);
                Accumulate(delta, finalContent, finalThinking, finalCalls);
                if (guardCompletion)
                {
                    if (!string.IsNullOrEmpty(delta.Content) || !string.IsNullOrEmpty(delta.Thinking))
                        finalHeldAnswer.Add(ChatStreamUpdate.Parsed(delta.Content, delta.Thinking, null));
                }
                else if (!string.IsNullOrEmpty(delta.Content) || !string.IsNullOrEmpty(delta.Thinking))
                {
                    yield return ChatStreamUpdate.Parsed(delta.Content, delta.Thinking, null);
                }
                if (!string.IsNullOrEmpty(delta.ToolCallText))
                    yield return ChatStreamUpdate.ToolProgress("writing", delta.ToolCallName, delta.ToolCallText);
            }

            ParsedOutput last = finalParser.Add(string.Empty, true);
            Accumulate(last, finalContent, finalThinking, finalCalls);
            if (guardCompletion)
            {
                if (!string.IsNullOrEmpty(last.Content) || !string.IsNullOrEmpty(last.Thinking))
                    finalHeldAnswer.Add(ChatStreamUpdate.Parsed(last.Content, last.Thinking, null));
            }
            else if (!string.IsNullOrEmpty(last.Content) || !string.IsNullOrEmpty(last.Thinking))
            {
                yield return ChatStreamUpdate.Parsed(last.Content, last.Thinking, null);
            }

            // Only the client's own tools may be forwarded here — this is ordinarily
            // the last round, so a name nobody declared would leave the turn empty with
            // no round left to recover in. A guarded route gets its one explicit
            // correction below even when this configured cap has already been reached.
            SkillTools.Partition(
                finalCalls, plan.ClientTools,
                out List<ToolCall> stillOurs, out List<ToolCall> pending, out List<ToolCall> stillUnknown);
            foreach (ToolCall dropped in stillOurs.Concat(stillUnknown))
                yield return ChatStreamUpdate.ToolProgress("finished", dropped.Name);

            bool guardedCompletionVerified = false;
            if (guardCompletion && pending.Count == 0)
            {
                WorkspaceArtifactCompletionResult completion = WorkspaceArtifactCompletion.Verify(plan);
                if (!completion.Complete)
                {
                    AppendCompletionCorrection(
                        working, plan, finalContent, finalThinking, finalCalls, finalTerminal, completion.Reason);
                    await foreach (ChatStreamUpdate correction in RunCompletionCorrectionAsync(
                        architecture, working, plan, enableThinking, generate, logger,
                        correctionRound: maxRounds + 2,
                        promptTokens + finalTerminal.PromptTokens,
                        evalTokens + finalTerminal.EvalTokens,
                        reusedTokens + finalTerminal.KvCacheReusedTokens,
                        promptNs + finalTerminal.PromptNs,
                        evalNs + finalTerminal.EvalNs,
                        totalNs + finalTerminal.TotalNs,
                        cancellationToken).ConfigureAwait(false))
                    {
                        yield return correction;
                    }
                    yield break;
                }

                guardedCompletionVerified = true;
                plan.VerifiedArtifact = completion.Artifact;
                if (stillOurs.Count == 0 && stillUnknown.Count == 0)
                {
                    foreach (ChatStreamUpdate held in finalHeldAnswer)
                        yield return held;
                }
                if (completion.Artifact.HasValue
                    && (stillOurs.Count > 0 || stillUnknown.Count > 0
                        || !ContainsArtifactMarkdownLink(
                            finalContent.ToString(), completion.Artifact.Value.Url)))
                {
                    yield return ChatStreamUpdate.Parsed(
                        stillOurs.Count == 0 && stillUnknown.Count == 0 && finalContent.Length > 0
                            ? "\n\n" + DescribeCompletedArtifact(completion.Artifact.Value)
                            : DescribeCompletedArtifact(completion.Artifact.Value),
                        null, null);
                }
            }

            if (pending.Count > 0)
                yield return ChatStreamUpdate.Parsed(string.Empty, null, pending);

            // The model was told to answer and asked for another tool instead, so its
            // whole reply is markup this loop has just dropped and the user would get
            // an empty bubble. Say what happened and what DID get made. Appended even
            // to a non-empty lead-in: "let me try" plus a dropped call is not an answer.
            if (!guardedCompletionVerified && (stillOurs.Count > 0 || stillUnknown.Count > 0))
            {
                string exhausted = DescribeExhaustedTurn(plan, stillOurs.Concat(stillUnknown));
                yield return ChatStreamUpdate.Parsed(
                    finalContent.Length == 0 ? exhausted : "\n\n" + exhausted, null, null);
            }

            yield return Combine(
                finalTerminal,
                promptTokens + finalTerminal.PromptTokens,
                evalTokens + finalTerminal.EvalTokens,
                reusedTokens + finalTerminal.KvCacheReusedTokens,
                promptNs + finalTerminal.PromptNs,
                evalNs + finalTerminal.EvalNs,
                totalNs + finalTerminal.TotalNs);
        }

        /// <summary>
        /// Every skill this turn showed the model — the selection and the catalog — so a
        /// skill name called as a tool is answered with the calling convention rather
        /// than with "no such tool".
        /// </summary>
        private static IReadOnlyList<string> KnownSkillIds(SkillRequestPlan plan)
        {
            var ids = new List<string>();
            foreach (Skill skill in plan.Prompt.Selected)
                ids.Add(skill.Id);
            foreach (Skill skill in plan.Prompt.Catalog)
                ids.Add(skill.Id);
            return ids;
        }

        /// <summary>How much of the raw stream a round keeps, for naming a loop.</summary>
        private const int RawTailChars = 4096;

        /// <summary>
        /// "the output repeated `","+` 43 times in a row", found in the raw tail of the
        /// round by the same periodicity test the engine applied to tokens, on
        /// characters; or a plain sentence when the tail is too short to show it.
        /// </summary>
        internal static string DescribeRepetition(string rawTail)
        {
            if (RepetitionGuard.TryFindTextLoop(rawTail, out string unit, out int repeats))
            {
                string shown = unit.Replace("\r", "\\r").Replace("\n", "\\n");
                if (shown.Length > 48)
                    shown = shown.Substring(0, 48) + "…";
                return $"the output repeated `{shown}` {repeats} times in a row";
            }
            return "the output repeated the same short sequence over and over";
        }

        /// <summary>
        /// Preserve the rejected assistant turn as evidence, then add one terse host
        /// correction. A user message is used when there was no tool call; inventing a
        /// bare tool result there produces an invalid conversation for strict templates.
        /// </summary>
        private static void AppendCompletionCorrection(
            List<ChatMessage> working,
            SkillRequestPlan plan,
            StringBuilder content,
            StringBuilder thinking,
            IReadOnlyList<ToolCall> calls,
            ChatStreamUpdate terminal,
            string reason)
        {
            working.Add(new ChatMessage
            {
                Role = "assistant",
                Content = content.ToString(),
                Thinking = thinking.Length == 0 ? null : thinking.ToString(),
                ToolCalls = calls.Count == 0 ? null : new List<ToolCall>(calls),
                RawOutputTokens = terminal.RawOutputTokens != null
                    ? new List<int>(terminal.RawOutputTokens)
                    : null,
                RawPromptTrailingWhitespace = terminal.RawPromptTrailingWhitespace,
                    RawGenerationSuffix = terminal.RawGenerationSuffix,
            });

            string feedback =
                "Host completion check: the requested PowerPoint deliverable is not complete. "
                + reason + " The latest tool error and any working research/spec are already in the conversation "
                + "and shared workspace; reuse them. In this one corrective continuation, make the smallest necessary "
                + "edit to the existing spec or repair file. If a required input is absent, create only that missing "
                + "input. Keep supported research claims, source URLs, and dates in the deck, then rerun the bundled "
                + "documents/scripts/make_pptx.py writer. Do not regenerate working files, copy the bundled writer, "
                + "install python-pptx/lxml, or hand-build OOXML. Issue every necessary host tool call now, in order; "
                + "do not merely claim success.";

            working.Add(calls.Count == 0
                ? new HostCompletionCorrectionMessage { Role = "user", Content = feedback }
                : BuildResult(plan, feedback));
        }

        /// <summary>
        /// A host-authored continuation that must use <c>role=user</c> for strict chat
        /// templates, but must never replace the genuine user task as context-compaction
        /// anchor. The runtime type is an in-process marker only; renderers still see an
        /// ordinary <see cref="ChatMessage"/> and no wire or persisted shape changes.
        /// </summary>
        internal sealed class HostCompletionCorrectionMessage : ChatMessage
        {
        }

        /// <summary>
        /// Exactly one extra model generation after an unverified final answer. Calls
        /// from that generation are executed as one bounded batch, then the host checks
        /// the artifact immediately and supplies the final truthful sentence itself —
        /// there is deliberately no third generation and therefore no retry loop.
        /// </summary>
        private static async IAsyncEnumerable<ChatStreamUpdate> RunCompletionCorrectionAsync(
            string architecture,
            List<ChatMessage> working,
            SkillRequestPlan plan,
            bool enableThinking,
            SkillChatGeneration generate,
            ILogger logger,
            int correctionRound,
            int promptTokens,
            int evalTokens,
            int reusedTokens,
            long promptNs,
            long evalNs,
            long totalNs,
            [EnumeratorCancellation] CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var parser = OutputParserFactory.Create(architecture);
            parser.Init(enableThinking, plan.Tools);
            var content = new StringBuilder();
            var thinking = new StringBuilder();
            var calls = new List<ToolCall>();
            var rawTail = new StringBuilder();
            ChatStreamUpdate terminal = default;
            string toolBeingWritten = null;

            await foreach (ChatStreamUpdate update in
                generate(working, plan.Tools, cancellationToken).ConfigureAwait(false))
            {
                if (update.Done)
                {
                    terminal = update;
                    continue;
                }
                if (update.RawGenerationSuffix != null)
                    parser.SetGenerationPromptSuffix(update.RawGenerationSuffix);
                if (string.IsNullOrEmpty(update.Piece))
                    continue;

                rawTail.Append(update.Piece);
                if (rawTail.Length > RawTailChars * 2)
                    rawTail.Remove(0, rawTail.Length - RawTailChars);

                ParsedOutput delta = parser.Add(update.Piece, false);
                Accumulate(delta, content, thinking, calls);
                // The correction's answer and reasoning are still provisional. Tool
                // writing/running progress remains live, which keeps a long writer call
                // visible without leaking another unverified success claim.
                toolBeingWritten = delta.ToolCallName ?? toolBeingWritten;
                if (!string.IsNullOrEmpty(delta.ToolCallText))
                    yield return ChatStreamUpdate.ToolProgress("writing", toolBeingWritten, delta.ToolCallText);
            }

            ParsedOutput flushed = parser.Add(string.Empty, true);
            Accumulate(flushed, content, thinking, calls);

            // The same invariant the main loop holds, and it has to be repeated here:
            // this correction EXECUTES what it parsed, so a round the engine ended for
            // repeating itself must not be acted on. There is no retry left at this
            // point -- the correction is the retry -- so it reports the loop and lets
            // the artifact check below say what was and was not produced.
            bool correctionLooped = FinishReasonMapper.IsRepetition(terminal.FinishReason);
            if (correctionLooped)
            {
                logger?.LogWarning(LogEventIds.SkillLoopCapped,
                    "skills.loop.repetition round={Round} retried=False (artifact correction): {What}",
                    correctionRound, DescribeRepetition(rawTail.ToString()));
                foreach (ToolCall call in calls)
                    yield return ChatStreamUpdate.ToolProgress("finished", call.Name);
                calls.Clear();
            }

            SkillTools.Partition(
                calls, plan.ClientTools,
                out List<ToolCall> skillCalls,
                out List<ToolCall> clientCalls,
                out List<ToolCall> unknownCalls);

            // Completion guards are attached only when the request has no client tools.
            // Still close any unexpected progress line rather than leaking a call this
            // one-shot correction cannot ask the client to service.
            foreach (ToolCall unserviceable in clientCalls.Concat(unknownCalls))
            {
                string refusal = clientCalls.Contains(unserviceable)
                    ? "A caller-owned tool cannot be serviced inside the artifact correction."
                    : SkillTools.DescribeUnknownTool(unserviceable.Name, plan.Tools, KnownSkillIds(plan));
                lock (plan.Invocations)
                {
                    plan.Invocations.Add(new SkillToolInvocation(
                        correctionRound, unserviceable.Name ?? string.Empty, null, null,
                        Ok: false, refusal.Length));
                }
                yield return ChatStreamUpdate.ToolProgress("finished", unserviceable.Name);
            }

            await foreach (ChatStreamUpdate progress in ExecuteSkillCallsAsync(
                skillCalls, correctionRound, plan, working, logger, cancellationToken).ConfigureAwait(false))
            {
                yield return progress;
            }

            WorkspaceArtifactCompletionResult completion = WorkspaceArtifactCompletion.Verify(plan);
            string final;
            if (completion.Complete && completion.Artifact.HasValue)
            {
                plan.VerifiedArtifact = completion.Artifact;
                final = DescribeCompletedArtifact(completion.Artifact.Value);
                logger?.LogInformation(LogEventIds.SkillToolInvoked,
                    "skills.completion.corrected round={Round} artifact={Artifact}",
                    correctionRound, completion.Artifact.Value.Name);
            }
            else
            {
                final = "I couldn't complete the PowerPoint report: no valid downloadable .pptx was produced "
                    + (correctionLooped
                        ? "because the corrective attempt started repeating itself and was stopped. "
                        : "after the one corrective attempt. ")
                    + completion.Reason;
                logger?.LogWarning(LogEventIds.SkillLoopCapped,
                    "skills.completion.failed round={Round} reason={Reason}",
                    correctionRound, completion.Reason);
            }

            yield return ChatStreamUpdate.Parsed(final, null, null);
            // "stop", because this turn DID finish and `final` is its honest answer --
            // nothing was cut off from the caller's side. The flag is belt and braces:
            // `final` already explains a loop, so no UI may add its own note even if
            // this reason is ever changed.
            yield return Combine(
                terminal,
                promptTokens + terminal.PromptTokens,
                evalTokens + terminal.EvalTokens,
                reusedTokens + terminal.KvCacheReusedTokens,
                promptNs + terminal.PromptNs,
                evalNs + terminal.EvalNs,
                totalNs + terminal.TotalNs,
                finishReason: "stop") with { RepetitionExplained = correctionLooped };
        }

        private static string DescribeCompletedArtifact(SkillProducedFile artifact) =>
            "The requested PowerPoint report is ready: "
            + $"[Download the .pptx report]({artifact.Url}).";

        private static bool ContainsArtifactMarkdownLink(string content, string url) =>
            !string.IsNullOrEmpty(content)
            && !string.IsNullOrEmpty(url)
            && content.Contains("](" + url + ")", StringComparison.Ordinal);

        /// <summary>
        /// Execute one generated batch of host-owned calls. Shared by the ordinary
        /// progressive loop and the single artifact correction so their confinement,
        /// trace, heartbeat, and per-round call cap stay identical.
        /// </summary>
        private static async IAsyncEnumerable<ChatStreamUpdate> ExecuteSkillCallsAsync(
            IReadOnlyList<ToolCall> skillCalls,
            int round,
            SkillRequestPlan plan,
            List<ChatMessage> working,
            ILogger logger,
            [EnumeratorCancellation] CancellationToken cancellationToken)
        {
            int executed = 0;
            for (int callIndex = 0; callIndex < skillCalls.Count; callIndex++)
            {
                ToolCall call = skillCalls[callIndex];
                cancellationToken.ThrowIfCancellationRequested();

                if (executed >= plan.LoopOptions.MaxCallsPerRound)
                {
                    working.Add(BuildResult(plan,
                        $"Error: too many tool calls in one turn; only the first {plan.LoopOptions.MaxCallsPerRound} "
                        + "were answered. Ask for one file at a time."));
                    break;
                }

                // Execution runs on a worker so this stream can keep breathing: a
                // shell call that installs packages holds the request for a minute or
                // more. The heartbeat updates keep that wait visible and cancellable.
                ApplyRoutedDefaults(call, plan.CompletionRequirement);
                string callDetail = DescribeCall(call);
                yield return ChatStreamUpdate.ToolProgress("running", call.Name, detail: callDetail);

                var liveOutput = new LiveOutputBuffer();
                // Acquire BEFORE scheduling. Request cancellation can dispose this async
                // iterator before the worker starts; the operation defers workspace
                // deletion until the worker's finally has run.
                IDisposable workspaceOperation = null;
                Task<SkillToolResult> execution;
                if (TryBuildRoutedPrerequisiteFailure(
                    call, plan.CompletionRequirement, out SkillToolResult prerequisiteFailure))
                {
                    // Keep this on the normal result path so the failed attempt is
                    // recorded, shown in the trace, and fed back to the model. The
                    // bundled script itself is deliberately never entered.
                    execution = Task.FromResult(prerequisiteFailure);
                }
                else
                {
                    workspaceOperation = plan.ToolContext?.Workspace?.BeginOperation();
                    try
                    {
                        execution = Task.Run(() =>
                        {
                            using (workspaceOperation)
                                return SkillTools.Execute(call, plan.ToolContext, liveOutput.Add);
                        });
                    }
                    catch
                    {
                        workspaceOperation?.Dispose();
                        throw;
                    }
                }

                var executionClock = Stopwatch.StartNew();
                while (await Task.WhenAny(execution, Task.Delay(1000, cancellationToken)).ConfigureAwait(false) != execution)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    yield return ChatStreamUpdate.ToolProgress(
                        "running", call.Name, piece: liveOutput.Drain(),
                        seconds: executionClock.Elapsed.TotalSeconds, detail: callDetail);
                }
                SkillToolResult result = await execution.ConfigureAwait(false);

                // A run shorter than one heartbeat never hit the loop above; its output
                // still deserves to reach the screen before "finished".
                string trailingOutput = liveOutput.Drain();
                if (!string.IsNullOrEmpty(trailingOutput))
                {
                    yield return ChatStreamUpdate.ToolProgress(
                        "running", call.Name, piece: trailingOutput,
                        seconds: executionClock.Elapsed.TotalSeconds, detail: callDetail);
                }
                executed++;

                var invocation = new SkillToolInvocation(
                    round, call.Name ?? string.Empty, result.SkillId, result.ResourcePath,
                    result.Ok, result.Content?.Length ?? 0)
                { Files = result.Files };
                lock (plan.Invocations)
                    plan.Invocations.Add(invocation);

                logger?.LogInformation(LogEventIds.SkillToolInvoked,
                    "skills.tool round={Round} tool={Tool} skill={SkillId} path={Path} ok={Ok} bytes={Bytes}",
                    round, invocation.Tool, invocation.SkillId ?? "-", invocation.ResourcePath ?? "-",
                    invocation.Ok, invocation.ResultBytes);

                working.Add(BuildResult(plan, result.Content ?? string.Empty, call.Name));

                // Yielded AFTER the invocation is recorded, so the adapter flushes the
                // step's structural result before it removes the progress line.
                yield return ChatStreamUpdate.ToolProgress(
                    "finished", call.Name, seconds: executionClock.Elapsed.TotalSeconds);

                if (TryCompleteRoutedWorkflow(plan, out SkillProducedFile artifact))
                {
                    logger?.LogInformation(LogEventIds.SkillToolInvoked,
                        "skills.completion.early round={Round} artifact={Artifact}",
                        round, artifact.Name);

                    // Parsing happened before execution, so the UI may already have
                    // shown a "writing" phase for calls later in this same response.
                    // Close those transient rows without executing or recording them.
                    for (int skipped = callIndex + 1; skipped < skillCalls.Count; skipped++)
                    {
                        yield return ChatStreamUpdate.ToolProgress(
                            "finished", skillCalls[skipped].Name);
                    }
                    yield break;
                }
            }
        }

        /// <summary>
        /// End only a host-routed, ordered workflow whose artifact passes the complete
        /// structural/evidence contract. A bare completion requirement with no ordered
        /// runs retains the original guard behaviour, and ordinary plans never enter
        /// this path at all.
        /// </summary>
        private static bool TryCompleteRoutedWorkflow(
            SkillRequestPlan plan,
            out SkillProducedFile artifact)
        {
            artifact = default;
            WorkspaceArtifactCompletionRequirement requirement = plan?.CompletionRequirement;
            if (requirement == null || requirement.RequiredRuns.Count == 0)
                return false;

            WorkspaceArtifactCompletionResult completion = WorkspaceArtifactCompletion.Verify(plan);
            if (!completion.Complete || completion.Artifact is not { } verified)
                return false;

            artifact = verified;
            plan.VerifiedArtifact = verified;
            return true;
        }

        /// <summary>
        /// Refuse one exact routed script until its route-owned workspace input exists
        /// and is non-empty. The opt-in path is validated when the route is attached;
        /// checking again here, immediately before launch, prevents a model from spending
        /// a Python invocation on a writer whose JSON spec has not been created yet.
        /// </summary>
        private static bool TryBuildRoutedPrerequisiteFailure(
            ToolCall call,
            WorkspaceArtifactCompletionRequirement requirement,
            out SkillToolResult failure)
        {
            failure = default;
            if (call == null
                || requirement == null
                || !string.Equals(call.Name, SkillTools.RunToolName, StringComparison.Ordinal))
            {
                return false;
            }

            string path = ReadCallString(call, "path") ?? ReadCallString(call, "script");
            string skill = ReadCallString(call, "skill");
            if (string.IsNullOrWhiteSpace(path) || string.IsNullOrWhiteSpace(skill))
                return false;

            string normalizedPath = path.Replace('\\', '/').Trim();
            WorkspaceSkillRunRequirement[] matches = requirement.RequiredRuns
                .Where(run => MatchesRoutedResourcePath(normalizedPath, run)
                    && string.Equals(skill.Trim(), run.SkillId, StringComparison.OrdinalIgnoreCase))
                .ToArray();
            if (matches.Length != 1 || string.IsNullOrEmpty(matches[0].RequiredInputPath))
                return false;

            WorkspaceSkillRunRequirement required = matches[0];
            bool ready = false;
            try
            {
                using (requirement.Workspace.BeginOperation())
                {
                    ready = requirement.Workspace.TryResolve(
                            required.RequiredInputPath, out string fullPath, out _)
                        && File.Exists(fullPath)
                        && new FileInfo(fullPath).Length > 0;
                }
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException
                                          or ArgumentException or NotSupportedException)
            {
                ready = false;
            }

            if (ready)
                return false;

            failure = new SkillToolResult(
                false,
                $"Error: required input '{required.RequiredInputPath}' is missing or empty. "
                    + "Create or repair only that input in the shared workspace, then retry this writer.",
                required.SkillId,
                required.ResourcePath);
            return true;
        }

        /// <summary>
        /// Complete a narrowly routed script call when a model selected the exact
        /// required script but dropped fields from its JSON arguments. Defaults belong
        /// to the route, not to <c>skills_run</c> globally. Ordinary routes preserve a
        /// non-empty model value; a narrowly host-owned route may explicitly require
        /// its bounded canonical argument vector.
        /// </summary>
        internal static bool ApplyRoutedDefaults(
            ToolCall call,
            WorkspaceArtifactCompletionRequirement requirement)
        {
            if (call == null
                || requirement == null
                || !string.Equals(call.Name, SkillTools.RunToolName, StringComparison.Ordinal))
            {
                return false;
            }

            string path = ReadCallString(call, "path");
            if (string.IsNullOrWhiteSpace(path))
                path = ReadCallString(call, "script");
            if (string.IsNullOrWhiteSpace(path))
                return false;

            string normalizedPath = path.Replace('\\', '/').Trim();
            WorkspaceSkillRunRequirement[] matches = requirement.RequiredRuns
                .Where(run => MatchesRoutedResourcePath(normalizedPath, run))
                .ToArray();
            if (matches.Length != 1)
                return false;

            WorkspaceSkillRunRequirement required = matches[0];
            string skill = ReadCallString(call, "skill");
            if (!string.IsNullOrWhiteSpace(skill)
                && !string.Equals(skill.Trim(), required.SkillId, StringComparison.OrdinalIgnoreCase))
            {
                return false;
            }

            call.Arguments ??= new Dictionary<string, object>();
            bool changed = false;

            // The matcher deliberately accepts the two spellings small models emit
            // most often ("./..." and "<skill>/..."). Hand the executor the one
            // unambiguous skill-relative spelling. SkillTools also confines and
            // normalizes these aliases, but canonicalizing here keeps the routed call,
            // its recorded invocation, and completion evidence in exact agreement.
            // If a malformed blank `path` shadows a usable `script` alias, repair the
            // canonical key because ExecuteRun gives `path` precedence.
            string resourceArgument = call.Arguments.ContainsKey("path") ? "path" : "script";
            if (!string.Equals(
                    ReadCallString(call, resourceArgument)?.Trim(),
                    required.ResourcePath,
                    SkillPathGuard.PathComparison))
            {
                call.Arguments[resourceArgument] = required.ResourcePath;
                changed = true;
            }

            if (string.IsNullOrWhiteSpace(skill))
            {
                call.Arguments["skill"] = required.SkillId;
                changed = true;
            }

            if (required.DefaultArguments.Count > 0
                && required.EnforceArguments
                && !ArgumentListEquals(call, "args", required.DefaultArguments))
            {
                // ExecuteRun gives `args` precedence over the legacy `arguments`
                // alias. Installing one canonical vector is therefore sufficient even
                // when a small model emitted both spellings.
                call.Arguments["args"] = required.DefaultArguments.ToArray();
                changed = true;
            }
            else if (required.DefaultArguments.Count > 0
                && !HasUsableArgumentList(call, "args")
                && !HasUsableArgumentList(call, "arguments"))
            {
                call.Arguments["args"] = required.DefaultArguments.ToArray();
                changed = true;
            }

            return changed;
        }

        private static bool ArgumentListEquals(
            ToolCall call,
            string name,
            IReadOnlyList<string> expected)
        {
            if (call?.Arguments == null
                || !call.Arguments.TryGetValue(name, out object value)
                || value == null)
            {
                return false;
            }

            IEnumerable<string> actual = value switch
            {
                JsonElement { ValueKind: JsonValueKind.Array } element =>
                    element.EnumerateArray().Select(item =>
                        item.ValueKind == JsonValueKind.String ? item.GetString() : item.ToString()),
                IEnumerable<string> values => values,
                _ => null,
            };
            return actual != null && actual.SequenceEqual(expected, StringComparer.Ordinal);
        }

        private static bool MatchesRoutedResourcePath(
            string normalizedPath,
            WorkspaceSkillRunRequirement required)
        {
            while (normalizedPath.StartsWith("./", StringComparison.Ordinal))
                normalizedPath = normalizedPath.Substring(2);

            if (string.Equals(normalizedPath, required.ResourcePath, SkillPathGuard.PathComparison))
                return true;

            string skillPrefix = required.SkillId + "/";
            return normalizedPath.StartsWith(skillPrefix, StringComparison.OrdinalIgnoreCase)
                && string.Equals(
                    normalizedPath.Substring(skillPrefix.Length),
                    required.ResourcePath,
                    SkillPathGuard.PathComparison);
        }

        private static string ReadCallString(ToolCall call, string name)
        {
            if (call?.Arguments == null
                || !call.Arguments.TryGetValue(name, out object value)
                || value == null)
            {
                return null;
            }

            return value switch
            {
                string text => text,
                JsonElement { ValueKind: JsonValueKind.String } element => element.GetString(),
                JsonElement { ValueKind: JsonValueKind.Null } => null,
                JsonElement element => element.ToString(),
                _ => Convert.ToString(value, CultureInfo.InvariantCulture),
            };
        }

        private static bool HasUsableArgumentList(ToolCall call, string name)
        {
            if (call?.Arguments == null
                || !call.Arguments.TryGetValue(name, out object value)
                || value == null)
            {
                return false;
            }

            if (value is string text)
                return !string.IsNullOrWhiteSpace(text);

            if (value is JsonElement element)
            {
                return element.ValueKind switch
                {
                    JsonValueKind.Null or JsonValueKind.Undefined => false,
                    JsonValueKind.String => !string.IsNullOrWhiteSpace(element.GetString()),
                    JsonValueKind.Array => element.GetArrayLength() > 0,
                    _ => true,
                };
            }

            if (value is System.Collections.ICollection collection)
                return collection.Count > 0;

            return true;
        }

        /// <summary>
        /// Collects a running tool's stdout/stderr lines from the process reader
        /// threads until the loop's heartbeat drains them into a progress frame.
        ///
        /// <para>
        /// Bounded twice, because the lines come from code the MODEL wrote: the buffer
        /// stops accepting past 64 KB (a print loop must not grow the heap between
        /// drains), and <see cref="Drain"/> stops forwarding past 8 KB total — the live
        /// stream is a window onto the run, not a transcript; the full (32 KB-capped)
        /// output still arrives in the tool result.
        /// </para>
        /// </summary>
        private sealed class LiveOutputBuffer
        {
            private const int MaxBufferedChars = 64 * 1024;
            private const int MaxForwardedChars = 8 * 1024;

            private readonly StringBuilder _pending = new();
            private int _forwarded;
            private bool _truncationReported;

            public void Add(string line)
            {
                lock (_pending)
                {
                    if (_pending.Length < MaxBufferedChars)
                        _pending.Append(line).Append('\n');
                }
            }

            /// <summary>Everything buffered since the last drain, or null when quiet.</summary>
            public string Drain()
            {
                string chunk;
                lock (_pending)
                {
                    if (_pending.Length == 0)
                        return null;
                    chunk = _pending.ToString();
                    _pending.Clear();
                }

                if (_forwarded >= MaxForwardedChars)
                {
                    if (_truncationReported)
                        return null;
                    _truncationReported = true;
                    return "…[further live output not shown; the full output arrives with the result]\n";
                }

                if (_forwarded + chunk.Length > MaxForwardedChars)
                    chunk = chunk.Substring(0, MaxForwardedChars - _forwarded);
                _forwarded += chunk.Length;
                return chunk;
            }
        }

        /// <summary>
        /// One line saying what a tool call is about to do, for the live progress the
        /// user watches while it runs: the command of a <c>shell</c> call, the script and
        /// arguments of a <c>skills_run</c>, the file of a <c>skills_read</c>. Never the
        /// full payload — that streams separately as the call is written.
        ///
        /// <para>
        /// Keyed off the shared name constants rather than string literals. A literal
        /// here does not fail to compile when a tool is renamed; it just quietly stops
        /// matching, and the user is left watching a live line that never says what is
        /// running. That is exactly how this switch went stale once already.
        /// </para>
        /// </summary>
        private static string DescribeCall(ToolCall call)
        {
            if (call?.Arguments == null)
                return null;

            string Arg(string key) =>
                call.Arguments.TryGetValue(key, out object v) && v != null
                    ? (v as string ?? v.ToString())
                    : null;

            // The command itself is the label a user wants: "pip install pandas" says
            // more than "shell · 240 chars" ever could.
            if (string.Equals(call.Name, SkillToolNames.Shell, StringComparison.Ordinal))
            {
                // The RAW value, not Arg()'s string: a Codex-trained model sends command as
                // an argv array, and ToString() on the list yields "System.Object[]" — which
                // is what the user then watched for the whole minute the command ran.
                // ReadCommand understands every shape the runner accepts, so the live line
                // shows exactly what is running.
                object raw = call.Arguments.TryGetValue("command", out object v) ? v : null;
                return Truncate(ShellCommand.Summarize(ShellCommand.ReadCommand(raw)), 64);
            }

            if (SkillToolNames.IsApplyPatchAlias(call.Name)
                || string.Equals(call.Name, SkillToolNames.ApplyPatch, StringComparison.Ordinal))
            {
                int patchChars = Arg("patch")?.Length ?? 0;
                return patchChars.ToString(CultureInfo.InvariantCulture) + " chars";
            }

            switch (call.Name)
            {
                case "skills_run":
                    string script = Arg("path") ?? Arg("script");
                    string args = Arg("args");
                    return script == null ? null : script + (string.IsNullOrEmpty(args) ? "" : " " + args);

                case "skills_read":
                    string skill = Arg("skill");
                    string path = Arg("path") ?? "SKILL.md";
                    return skill == null ? path : skill + "/" + path;

                default:
                    return null;
            }
        }

        /// <summary>
        /// What to tell the USER when the turn ran out of tool calls with the model still
        /// working. Never empty, and never only an apology: the files the turn did
        /// produce are the part worth keeping, and naming them is what makes "ran out of
        /// budget" different from "nothing happened".
        /// </summary>
        private static string DescribeExhaustedTurn(SkillRequestPlan plan, IEnumerable<ToolCall> wanted)
        {
            var sb = new StringBuilder();
            sb.Append("I ran out of the tool-call budget for this turn while still working");

            string[] names;
            lock (plan.Invocations)
                names = plan.Invocations.SelectMany(i => i.Files).Select(f => f.Name).Distinct().ToArray();

            if (names.Length > 0)
            {
                sb.Append(". Finished so far: ").Append(string.Join(", ", names));
            }

            string[] next = wanted.Select(c => c.Name).Where(n => !string.IsNullOrEmpty(n)).Distinct().ToArray();
            if (next.Length > 0)
                sb.Append(". I was about to call ").Append(string.Join(", ", next)).Append(" again");

            sb.Append(". Send \"continue\" and I will pick up from here, or raise --skills-max-rounds to give a "
                      + "turn more steps.");
            return sb.ToString();
        }

        private static string Truncate(string text, int max) =>
            text.Length <= max ? text : text.Substring(0, max - 1) + "\u2026";

        /// <summary>Fold one parser delta into the round's running totals.</summary>
        private static void Accumulate(
            ParsedOutput delta, StringBuilder content, StringBuilder thinking, List<ToolCall> calls)
        {
            if (delta == null)
                return;
            if (!string.IsNullOrEmpty(delta.Content))
                content.Append(delta.Content);
            if (!string.IsNullOrEmpty(delta.Thinking))
                thinking.Append(delta.Thinking);
            if (delta.ToolCalls is { Count: > 0 })
                calls.AddRange(delta.ToolCalls);
        }

        private static ChatStreamUpdate Combine(
            ChatStreamUpdate terminal,
            int promptTokens,
            int evalTokens,
            int reusedTokens,
            long promptNs,
            long evalNs,
            long totalNs,
            string finishReason = null) =>
            new(string.Empty, true, promptTokens, evalTokens, reusedTokens,
                totalNs, promptNs, evalNs, finishReason ?? terminal.FinishReason ?? "stop")
            {
                RawOutputTokens = terminal.RawOutputTokens,
                RawPromptTrailingWhitespace = terminal.RawPromptTrailingWhitespace,
                    RawGenerationSuffix = terminal.RawGenerationSuffix,
            };

        /// <summary>
        /// Wrap a tool result in the message shape this model family renders. Mistral 3
        /// drops <c>role: "tool"</c> messages outright, so on that family the result is
        /// fed back as a user turn instead of vanishing from the prompt.
        /// </summary>
        private static ChatMessage BuildResult(SkillRequestPlan plan, string content, string tool = null)
        {
            if (plan.LoopOptions.ToolResultsAreRendered)
                return new ChatMessage { Role = "tool", Content = content };

            string prefix = tool == null
                ? "Result of the skill lookup you requested:"
                : $"Result of your {tool} call:";
            return new ChatMessage { Role = "user", Content = prefix + "\n\n" + content };
        }
    }
}
