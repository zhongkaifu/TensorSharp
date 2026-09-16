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
using System.Globalization;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

using TensorSharp.Runtime;
namespace TensorSharp.AgentHost.Skills
{
    /// <summary>
    /// One completed generation, as the loop needs to see it.
    /// </summary>
    /// <param name="Parsed">The model's reply, already split into content, reasoning and tool calls.</param>
    /// <param name="RawTokens">
    /// The tokens the model actually emitted this round, in order and excluding the
    /// terminating EOS.
    ///
    /// <para>
    /// Supplying these is what keeps a multi-round skill conversation cheap.
    /// <see cref="KVCachePromptRenderer"/> splices an assistant turn's recorded tokens
    /// back into the next render instead of re-tokenizing its text, so the re-rendered
    /// prefix stays byte-identical to what the cache holds. Passing null still
    /// produces a correct answer, but every round re-prefills the whole conversation
    /// from the first assistant turn onward.
    /// </para>
    /// </param>
    public readonly record struct SkillTurnOutput(ParsedOutput Parsed, IReadOnlyList<int>? RawTokens = null)
    {
        /// <summary>
        /// Exact trailing whitespace of the rendered generation prompt that preceded
        /// <see cref="RawTokens"/>. Empty is a known boundary with no whitespace;
        /// null means the producer does not support boundary tracking yet.
        /// </summary>
        public string? RawPromptTrailingWhitespace { get; init; }

        /// <summary>What the round's generation prompt ended with; see
        /// <see cref="ChatMessage.RawGenerationSuffix"/>.</summary>
        public string? RawGenerationSuffix { get; init; }
    }

    /// <summary>
    /// Runs one generation against whatever execution path the host owns.
    /// </summary>
    /// <remarks>
    /// A delegate rather than an interface because the two hosts share no execution
    /// entry point at all: the server submits a <c>SequenceState</c> to the
    /// <c>InferenceEngine</c>, while the CLI drives <c>Forward</c> directly in its own
    /// decode loops. Everything the loop needs from a host is "given these messages
    /// and these tools, produce one reply", and that is exactly this signature.
    /// </remarks>
    public delegate Task<SkillTurnOutput> SkillTurnGenerator(
        List<ChatMessage> messages,
        List<ToolFunction>? tools,
        CancellationToken cancellationToken);

    /// <summary>Bounds on the loop.</summary>
    public sealed class SkillAgentLoopOptions
    {
        /// <summary>
        /// How many times the model may fetch skill content before it must answer.
        ///
        /// <para>
        /// Eight covers the realistic worst case — read a skill, read two of its
        /// references, page through a long one — with room to spare, while bounding the
        /// damage from a model that loops on a file it keeps mis-naming. Each round is a
        /// full generation, so an unbounded loop is unbounded cost.
        /// </para>
        /// </summary>
        public int MaxRounds { get; init; } = 8;

        /// <summary>
        /// Ceiling on how many skill tool calls are executed in one round. A model that
        /// emits fifty reads in a single turn is malfunctioning, and answering all of
        /// them would blow the context before the next generation starts.
        /// </summary>
        public int MaxCallsPerRound { get; init; } = 8;

        /// <summary>
        /// False when the model's chat format drops <c>role: "tool"</c> messages
        /// (Mistral 3). The results are then fed back as a <c>user</c> turn instead.
        ///
        /// <para>
        /// Without this the loop is silently broken on that family: the renderer drops
        /// every tool message, so the model is asked to continue from an answer that is
        /// not in its prompt, calls the same tool again, and burns the whole round
        /// budget before giving up.
        /// </para>
        /// </summary>
        public bool ToolResultsAreRendered { get; init; } = true;

        /// <summary>Called after each executed tool call, for logging and for streaming a trace to a UI.</summary>
        public Action<SkillToolInvocation>? OnInvocation { get; init; }

        /// <summary>
        /// The CALLER's own tools, when the caller knows which of the declarations were
        /// its own. Given them, the loop can tell a tool the client will answer from a
        /// name nobody declared, and answer the latter in conversation instead of
        /// handing back a call the client has no implementation for.
        ///
        /// <para>
        /// Null keeps the older two-way behaviour: anything not built-in goes back to
        /// the caller. Deliberately NOT the merged tool list — that contains this host's
        /// own declarations, so a host tool the classifier had not learned would be
        /// called the caller's and forwarded, which is the exact failure the third
        /// bucket exists to prevent.
        /// </para>
        /// </summary>
        public IReadOnlyCollection<ToolFunction>? ClientTools { get; init; }

        /// <summary>A copy of these options that also knows the caller's own tools.</summary>
        public SkillAgentLoopOptions WithClientTools(IReadOnlyCollection<ToolFunction>? clientTools) => new()
        {
            MaxRounds = MaxRounds,
            MaxCallsPerRound = MaxCallsPerRound,
            ToolResultsAreRendered = ToolResultsAreRendered,
            OnInvocation = OnInvocation,
            ClientTools = clientTools,
        };

        /// <summary>Defaults.</summary>
        public static SkillAgentLoopOptions Default { get; } = new();
    }

    /// <summary>One executed skill tool call, as reported to <see cref="SkillAgentLoopOptions.OnInvocation"/>.</summary>
    /// <param name="Round">1-based round the call was made in.</param>
    /// <param name="Tool">The tool's name.</param>
    /// <param name="SkillId">The skill it touched, or null.</param>
    /// <param name="ResourcePath">The file it read or ran, or null.</param>
    /// <param name="Ok">False when the model was handed an error instead of content.</param>
    /// <param name="ResultBytes">Size of the result fed back.</param>
    public readonly record struct SkillToolInvocation(
        int Round,
        string Tool,
        string? SkillId,
        string? ResourcePath,
        bool Ok,
        int ResultBytes)
    {
        /// <summary>
        /// Files a <c>shell</c> call produced and kept, so a UI streaming this trace
        /// can offer the downloads itself rather than hoping the model's answer repeats
        /// the links. Empty for every other tool.
        /// </summary>
        public IReadOnlyList<SkillProducedFile> Files { get; init; } = Array.Empty<SkillProducedFile>();
    }

    /// <summary>What the loop ended with.</summary>
    /// <param name="Output">The final generation.</param>
    /// <param name="Messages">
    /// The conversation as it now stands, including every assistant turn and tool
    /// result the loop appended. A host that tracks history should adopt this.
    /// </param>
    /// <param name="Rounds">How many generations ran. 1 means the model answered without fetching anything.</param>
    /// <param name="Invocations">Every skill tool call executed, in order.</param>
    /// <param name="HitRoundLimit">True when the loop stopped because it ran out of rounds.</param>
    /// <param name="PendingClientToolCalls">
    /// Tool calls belonging to the CALLER's own tools, which TensorSharp must not
    /// execute. Non-empty means the host has to return them to its client rather than
    /// treating <paramref name="Output"/> as a finished answer.
    /// </param>
    public sealed record SkillLoopResult(
        SkillTurnOutput Output,
        List<ChatMessage> Messages,
        int Rounds,
        IReadOnlyList<SkillToolInvocation> Invocations,
        bool HitRoundLimit,
        IReadOnlyList<ToolCall> PendingClientToolCalls);

    /// <summary>
    /// The progressive-disclosure loop: generate, answer any skill tool calls in
    /// process, generate again, until the model stops asking.
    ///
    /// <para>
    /// This exists so that a client that knows nothing about skills gets a finished
    /// answer. An ordinary OpenAI client sends <c>skills: ["pdf"]</c> and one user
    /// message; if TensorSharp returned <c>skills_read</c> as a tool call, that client
    /// would have no implementation to service it and the conversation would stall.
    /// Because these particular tools are read-only and confined to a directory the
    /// operator already exposed, TensorSharp can answer them itself — which is what
    /// makes progressive disclosure work over a stateless HTTP API at all.
    /// </para>
    /// <para>
    /// The caller's OWN tools are never executed here. A turn that calls one stops the
    /// loop and reports it through
    /// <see cref="SkillLoopResult.PendingClientToolCalls"/>, because only the client
    /// knows what its tools do.
    /// </para>
    /// </summary>
    public static class SkillAgentLoop
    {
        /// <summary>
        /// Drive <paramref name="generate"/> until the model stops calling skill tools.
        /// </summary>
        /// <param name="messages">
        /// The conversation, with skill instructions already injected by
        /// <see cref="SkillPrompt.Apply(List{ChatMessage}, SkillPlan)"/>. Not mutated:
        /// the loop works on its own copy and returns it.
        /// </param>
        /// <param name="tools">Every tool offered, the built-in skill tools included.</param>
        /// <param name="context">Which skills the tools may reach.</param>
        /// <param name="generate">The host's one-generation callback.</param>
        /// <param name="options">Bounds, or null for the defaults.</param>
        /// <param name="cancellationToken">
        /// Checked between rounds and before each tool call, so a cancelled request
        /// stops at the next boundary rather than after the round budget is spent.
        /// </param>
        public static async Task<SkillLoopResult> RunAsync(
            List<ChatMessage> messages,
            List<ToolFunction>? tools,
            SkillToolContext context,
            SkillTurnGenerator generate,
            SkillAgentLoopOptions? options = null,
            CancellationToken cancellationToken = default)
        {
            ArgumentNullException.ThrowIfNull(messages);
            ArgumentNullException.ThrowIfNull(context);
            ArgumentNullException.ThrowIfNull(generate);
            options ??= SkillAgentLoopOptions.Default;

            var working = new List<ChatMessage>(messages.Count + 8);
            foreach (ChatMessage message in messages)
                working.Add(message);

            var invocations = new List<SkillToolInvocation>();
            int maxRounds = Math.Max(1, options.MaxRounds);
            SkillTurnOutput output = default;

            for (int round = 1; round <= maxRounds; round++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                output = await generate(working, tools, cancellationToken).ConfigureAwait(false);

                List<ToolCall> calls = output.Parsed?.ToolCalls ?? new List<ToolCall>();

                // Three ways, not two: a name that is neither ours nor one the CALLER
                // declared belongs to nobody. Handing it back was a silent end to the
                // turn, since a caller that never declared it has nothing to run and no
                // reason to show it.
                SkillTools.Partition(
                    calls, options.ClientTools,
                    out List<ToolCall> skillCalls,
                    out List<ToolCall> clientCalls,
                    out List<ToolCall> unknownCalls);

                if (skillCalls.Count == 0 && unknownCalls.Count == 0)
                {
                    return new SkillLoopResult(
                        output, working, round, invocations, HitRoundLimit: false, clientCalls);
                }

                // Record what the model said, tool calls and raw tokens included, so the
                // next render splices this turn back rather than re-tokenizing it.
                working.Add(new ChatMessage
                {
                    Role = "assistant",
                    Content = output.Parsed?.Content ?? string.Empty,
                    Thinking = string.IsNullOrEmpty(output.Parsed?.Thinking) ? null : output.Parsed!.Thinking,
                    ToolCalls = new List<ToolCall>(calls),
                    RawOutputTokens = output.RawTokens != null ? new List<int>(output.RawTokens) : null,
                    RawPromptTrailingWhitespace = output.RawPromptTrailingWhitespace,
                    RawGenerationSuffix = output.RawGenerationSuffix,
                });

                foreach (ToolCall unknownCall in unknownCalls)
                {
                    var unknownInvocation = new SkillToolInvocation(
                        round, unknownCall.Name ?? string.Empty, null, null, Ok: false, ResultBytes: 0);
                    invocations.Add(unknownInvocation);
                    options.OnInvocation?.Invoke(unknownInvocation);
                    working.Add(BuildResultMessage(
                        options,
                        // The reachable skills, so a skill name called as if it were a
                        // tool is answered with the calling convention rather than "no
                        // such tool" — the reply that talked a model out of a correct
                        // choice on the chat path.
                        SkillTools.DescribeUnknownTool(
                            unknownCall.Name, tools, ReachableSkillIds(context)),
                        unknownCall.Name, unknownCall.Id));
                }

                int executed = 0;
                foreach (ToolCall call in skillCalls)
                {
                    cancellationToken.ThrowIfCancellationRequested();

                    if (executed >= options.MaxCallsPerRound)
                    {
                        working.Add(BuildResultMessage(options,
                            $"Error: too many tool calls in one turn; only the first "
                            + $"{options.MaxCallsPerRound.ToString(CultureInfo.InvariantCulture)} were answered. "
                            + "Ask for one file at a time.", call.Name, call.Id));
                        continue;
                    }

                    SkillToolResult result = SkillTools.Execute(call, context);
                    executed++;

                    var invocation = new SkillToolInvocation(
                        round, call.Name ?? string.Empty, result.SkillId, result.ResourcePath,
                        result.Ok, result.Content?.Length ?? 0)
                    { Files = result.Files };
                    invocations.Add(invocation);
                    options.OnInvocation?.Invoke(invocation);

                    working.Add(BuildResultMessage(options, result.Content ?? string.Empty, call.Name, call.Id));
                }

                // A caller tool was requested alongside the skill work. Only the client
                // knows what it does, so the loop stops here and hands it back; the skill
                // results it just produced stay in the returned history.
                if (clientCalls.Count > 0)
                {
                    return new SkillLoopResult(
                        output, working, round, invocations, HitRoundLimit: false, clientCalls);
                }
            }

            // Out of rounds with the model still asking for files. Tell it so, in the
            // conversation, and let it answer from what it already has — returning the
            // last tool-call-only turn to the user would show them nothing at all.
            working.Add(BuildResultMessage(options,
                "Error: the limit on tool calls for this turn has been reached. "
                + "Answer now using what you have already read, and say which part you could not check."));

            cancellationToken.ThrowIfCancellationRequested();
            output = await generate(working, tools, cancellationToken).ConfigureAwait(false);

            // Only the caller's OWN tools may ride back out: there is no round left in
            // which the model could recover from a name nobody declared.
            SkillTools.Partition(
                output.Parsed?.ToolCalls, options.ClientTools,
                out List<ToolCall> finalOurs, out List<ToolCall> finalClientCalls,
                out List<ToolCall> finalUnknown);

            // The model was told to answer and asked for another tool instead. Those
            // calls are dropped here — nothing is left to run them — so a reply that was
            // nothing BUT calls would reach the caller as an empty string, which is the
            // same silence the round cap exists to prevent. Say what happened.
            // APPENDED to whatever the model did say, not substituted for it, and not
            // conditional on that being empty. A lead-in sentence followed by a dropped
            // tool call is the COMMON case, not the rare one: 6 of the 12 capped turns in
            // this server's logs looked exactly like that, and this gate fired for none of
            // them — so the caller got a promise, no work, and no notice.
            if (finalOurs.Count > 0 || finalUnknown.Count > 0)
            {
                ParsedOutput parsed = output.Parsed ?? new ParsedOutput();
                string said = parsed.Content ?? string.Empty;
                string exhausted = DescribeExhaustedTurn(finalOurs.Concat(finalUnknown));
                parsed.Content = said.Length == 0 ? exhausted : said.TrimEnd() + "\n\n" + exhausted;
                output = output with { Parsed = parsed };
            }

            return new SkillLoopResult(
                output, working, maxRounds + 1, invocations, HitRoundLimit: true, finalClientCalls);
        }

        /// <summary>
        /// What to tell the USER when the turn ran out of tool calls with the model still
        /// working. Never empty: an empty answer is indistinguishable from a crash, and
        /// the caller has no way to tell that the work was bounded rather than broken.
        /// </summary>
        /// <summary>The ids a skill tool call could actually reach this turn.</summary>
        private static IReadOnlyList<string> ReachableSkillIds(SkillToolContext? context)
        {
            var ids = new List<string>();
            foreach (Skill skill in context?.Reachable ?? (IReadOnlyList<Skill>)Array.Empty<Skill>())
                ids.Add(skill.Id);
            return ids;
        }

        private static string DescribeExhaustedTurn(IEnumerable<ToolCall> wanted)
        {
            string[] names = wanted
                .Select(c => c.Name)
                .Where(n => !string.IsNullOrEmpty(n))
                .Distinct(StringComparer.Ordinal)
                .ToArray();

            string next = names.Length > 0
                ? " I was about to call " + string.Join(", ", names) + " again."
                : string.Empty;

            return "I ran out of the tool-call budget for this turn while still working." + next
                   + " Ask me to continue and I will pick up from here.";
        }

        /// <summary>
        /// Wrap a tool result in the message shape this model family can actually read.
        ///
        /// <para>
        /// A <c>tool</c> role is right almost everywhere, but Mistral 3's renderer
        /// handles only <c>user</c> and <c>assistant</c> and drops everything else on the
        /// floor — silently, with the request still succeeding. Falling back to a
        /// <c>user</c> turn there is not elegant, but it is the difference between skills
        /// working on that family and appearing to work while doing nothing.
        /// </para>
        /// </summary>
        private static ChatMessage BuildResultMessage(SkillAgentLoopOptions options, string content, string? tool = null, string? toolCallId = null)
        {
            // Round-limit guidance is not a result for any outstanding call. A
            // standalone tool message would be invalid on an OpenAI transport.
            if (tool == null && toolCallId == null)
                return new ChatMessage { Role = "user", Content = content };
            if (options.ToolResultsAreRendered)
                return new ChatMessage { Role = "tool", Content = content, ToolCallId = toolCallId };

            string prefix = tool == null
                ? "Result of the skill lookup you requested:"
                : $"Result of your {tool} call:";
            return new ChatMessage { Role = "user", Content = prefix + "\n\n" + content };
        }
    }
}
