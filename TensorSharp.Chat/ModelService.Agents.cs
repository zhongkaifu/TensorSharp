using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TensorSharp.AgentHost.Agents;
using TensorSharp.AgentHost.Skills;
using TensorSharp.Server.Skills;

namespace TensorSharp.Server;

public partial class ModelService
{
    private async IAsyncEnumerable<ChatStreamUpdate> MultiAgentChatStreamAsync(
        List<ChatMessage> history, SkillRequestPlan plan, SkillChatGeneration generate,
        int maxTokens, SamplingConfig turnSampling, SamplingConfig sourceSampling,
        bool enableThinking, ILogger logger, ChatTurnContext rootTurn,
        [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        var sessions = new ConcurrentBag<ChatSession>();
        int promptTokens = 0, evalTokens = 0, reusedTokens = 0;
        long promptNs = 0, evalNs = 0;
        var elapsed = Stopwatch.StartNew();
        List<ChatMessage> messages = MultiAgentPrompt.Apply(history, plan.MultiAgent);
        IReadOnlyList<MultiAgentPromptProfile> publicProfiles = null;

        SkillTurnGenerator CreateChildGenerator(string agentId)
        {
            // Never share the root's transcript or KV continuation scope. The engine
            // can still reuse the public prefix and schedule independent sequences.
            var childSession = new ChatSession();
            sessions.Add(childSession);
            var childTurn = new ChatTurnContext { PublicPrefixCandidates = publicProfiles };
            return async (childMessages, childTools, ct) =>
            {
                var parser = OutputParserFactory.Create(Architecture);
                parser.Init(enableThinking, childTools);
                var content = new StringBuilder();
                var thinking = new StringBuilder();
                var calls = new List<ToolCall>();
                ChatStreamUpdate terminal = default;
                bool historyCompacted = false;
                void Append(ParsedOutput part)
                {
                    if (part == null) return;
                    content.Append(part.Content);
                    thinking.Append(part.Thinking);
                    if (part.ToolCalls != null) calls.AddRange(part.ToolCalls);
                }
                await foreach (ChatStreamUpdate update in _generation.ChatStreamWithMetricsAsync(
                    childSession, childMessages, maxTokens, ct,
                    SamplingForDeepSeek41SkillRound(Architecture, turnSampling, sourceSampling),
                    childTools, enableThinking, childTurn).ConfigureAwait(false))
                {
                    historyCompacted |= update.HistoryCompacted;
                    if (update.Done) { terminal = update; continue; }
                    if (update.RawGenerationSuffix != null)
                        parser.SetGenerationPromptSuffix(update.RawGenerationSuffix);
                    if (update.IsParsed)
                    {
                        content.Append(update.Piece);
                        thinking.Append(update.ThinkingPiece);
                        if (update.ParsedToolCalls != null) calls.AddRange(update.ParsedToolCalls);
                    }
                    else if (!string.IsNullOrEmpty(update.Piece))
                        Append(parser.Add(update.Piece, false));
                }
                Append(parser.Add(string.Empty, true));
                Interlocked.Add(ref promptTokens, terminal.PromptTokens);
                Interlocked.Add(ref evalTokens, terminal.EvalTokens);
                Interlocked.Add(ref reusedTokens, terminal.KvCacheReusedTokens);
                Interlocked.Add(ref promptNs, terminal.PromptNs);
                Interlocked.Add(ref evalNs, terminal.EvalNs);
                return new SkillTurnOutput(new ParsedOutput
                {
                    Content = content.ToString(), Thinking = thinking.ToString(),
                    ToolCalls = calls.Count == 0 ? null : calls,
                }, terminal.RawOutputTokens)
                {
                    FinishReason = terminal.FinishReason,
                    RawPromptTrailingWhitespace = terminal.RawPromptTrailingWhitespace,
                    RawGenerationSuffix = terminal.RawGenerationSuffix,
                    HistoryCompacted = historyCompacted,
                };
            };
        }

        try
        {
            await using var agents = new MultiAgentSession(history, plan.Tools, plan.ToolContext,
                CreateChildGenerator, plan.MultiAgent,
                new SkillAgentLoopOptions
                {
                    MaxRounds = plan.LoopOptions.MaxRounds,
                    MaxCallsPerRound = plan.LoopOptions.MaxCallsPerRound,
                    ToolResultsAreRendered = plan.LoopOptions.ToolResultsAreRendered,
                    ClientTools = plan.ClientTools,
                    OnInvocation = invocation =>
                    {
                        lock (plan.Invocations) plan.Invocations.Add(invocation);
                    },
                }, cancellationToken);
            // Predict the public boundary before the root's first prefill, using
            // the exact policies and tool subsets that children will receive.
            // Include the root so descendants also keep that common ancestor.
            var profiles = new List<MultiAgentPromptProfile>(agents.GetPromptProfiles())
            {
                new(messages.TakeWhile(message => message.Role is "system" or "developer").ToArray(), plan.Tools),
            };
            publicProfiles = profiles;
            rootTurn.PublicPrefixCandidates = profiles;
            plan.Agents = agents;
            await foreach (ChatStreamUpdate update in SkillChatLoop.RunAsync(
                Architecture, messages, plan, enableThinking, generate, logger, cancellationToken)
                .ConfigureAwait(false))
            {
                // Child answers are consumed only as tool results. Usage includes
                // their actual generations, while SSE text stays the parent's.
                yield return update.Done ? update with
                {
                    PromptTokens = update.PromptTokens + Volatile.Read(ref promptTokens),
                    EvalTokens = update.EvalTokens + Volatile.Read(ref evalTokens),
                    KvCacheReusedTokens = update.KvCacheReusedTokens + Volatile.Read(ref reusedTokens),
                    PromptNs = update.PromptNs + Interlocked.Read(ref promptNs),
                    EvalNs = update.EvalNs + Interlocked.Read(ref evalNs),
                    // Generations overlap; summed child durations are compute work,
                    // not end-to-end latency. Report elapsed time for this request.
                    TotalNs = (long)(elapsed.Elapsed.TotalSeconds * 1_000_000_000),
                } : update;
            }
        }
        finally
        {
            plan.Agents = null;
            foreach (ChatSession childSession in sessions) childSession.Dispose();
        }
    }
}
