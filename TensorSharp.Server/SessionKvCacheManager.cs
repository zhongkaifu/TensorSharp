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
using Microsoft.Extensions.Logging;

namespace TensorSharp.Server
{
    internal sealed class SessionKvCacheManager : IDisposable
    {
        private readonly ModelLifecycleService _lifecycle;
        private readonly ILogger _logger;
        private readonly ChatSession _intrinsicSession = new("__svc_intrinsic__");

        public SessionKvCacheManager(ModelLifecycleService lifecycle, ILogger logger)
        {
            _lifecycle = lifecycle ?? throw new ArgumentNullException(nameof(lifecycle));
            _logger = logger ?? Microsoft.Extensions.Logging.Abstractions.NullLogger.Instance;
        }

        /// <summary>
        /// Built-in fallback session for code paths that are not session-aware (e.g.
        /// existing tests or any caller that doesn't plumb a session through). All HTTP
        /// endpoints go through SessionManager and pass their own session explicitly.
        /// </summary>
        public ChatSession IntrinsicSession => _intrinsicSession;

        /// <summary>
        /// Session whose tokens are currently reflected in the model's per-layer K/V
        /// tensors. Only one session at a time can be active.
        /// </summary>
        public ChatSession ActiveSession { get; private set; }

        public KVCache KVCache => ActiveSession?.KVCache ?? _intrinsicSession.KVCache;

        public IReadOnlyList<ChatMessage> TrackedHistory =>
            (ActiveSession ?? _intrinsicSession).TrackedHistory.AsReadOnly();

        public void ClearForModelReload()
        {
            ActiveSession = null;
            _intrinsicSession.TrackedHistory.Clear();
            _intrinsicSession.KVCache.Reset();
        }

        public void InvalidateKVCache()
        {
            ResetSession(ActiveSession ?? _intrinsicSession);
        }

        public void ResetSession(ChatSession session)
        {
            if (session == null)
                return;

            session.TrackedHistory.Clear();
            session.KVCache.Reset();
            if (ReferenceEquals(session, ActiveSession))
                _lifecycle.Model?.ResetKVCache();
        }

        public void DisposeSession(ChatSession session)
        {
            if (session == null || session.IsDisposed)
                return;

            bool wasActive = ReferenceEquals(session, ActiveSession);
            session.Dispose();
            if (wasActive)
            {
                ActiveSession = null;
                _lifecycle.Model?.ResetKVCache();
            }
        }

        public void ActivateSession(ChatSession session)
        {
            if (session == null)
                throw new ArgumentNullException(nameof(session));
            if (session.IsDisposed)
                throw new ObjectDisposedException(nameof(ChatSession), $"Session {session.Id} has been disposed.");

            if (ReferenceEquals(session, ActiveSession))
            {
                session.LastUsedAt = DateTime.UtcNow;
                return;
            }

            string previousSessionId = ActiveSession?.Id;
            if (ActiveSession != null)
            {
                // Old session no longer owns the model's tensors; drop its token
                // bookkeeping so a future ActivateSession(old) rebuilds from scratch.
                ActiveSession.KVCache.Reset();
            }

            _lifecycle.Model?.ResetKVCache();
            session.KVCache.Reset();
            ActiveSession = session;
            session.LastUsedAt = DateTime.UtcNow;

            _logger.LogDebug(LogEventIds.SessionActivated,
                "Activated session {SessionId} (previousSession={PreviousSessionId})",
                session.Id, previousSessionId ?? "(none)");
        }

        /// <summary>
        /// Move the model's KV state to one whose contents are exactly <paramref name="inputTokens"/>,
        /// returning the next-token logits at position <c>inputTokens.Count</c>.
        /// </summary>
        public float[] PrepareForGeneration(ChatSession session, List<int> inputTokens, out int reusedTokens)
        {
            var model = _lifecycle.Model;
            var cache = session.KVCache;
            ReusePlan plan = cache.PlanReuse(inputTokens, model.SupportsKVCacheTruncation);

            switch (plan.Kind)
            {
                case ReusePlanKind.ExactMatch:
                {
                    reusedTokens = inputTokens.Count;
                    _logger.LogDebug(LogEventIds.KvCacheReusePlan,
                        "kv.reuse exact match reusing {ReusedTokens}/{TotalTokens} cached tokens (saved 100%)",
                        inputTokens.Count, inputTokens.Count);
                    model.MultimodalInjector.QueuePromptEmbeddings(inputTokens.Count);
                    return plan.CachedLogits;
                }

                case ReusePlanKind.PartialReuse:
                {
                    int reusedPrefix = plan.ReusedPrefixLength;
                    int suffixLength = plan.TokensToForward;
                    reusedTokens = reusedPrefix;

                    model.TruncateKVCache(reusedPrefix);
                    cache.TruncateTo(reusedPrefix);

                    bool queuedPromptEmbeddings = model.MultimodalInjector.QueuePromptEmbeddings(reusedPrefix);
                    var suffixTokens = CopyTokenRange(inputTokens, reusedPrefix, suffixLength);

                    _logger.LogDebug(LogEventIds.KvCacheReusePlan,
                        "kv.reuse partial keeping {ReusedTokens}/{TotalTokens} tokens, forwarding {NewTokens} new tokens (saved {SavedPercent:F0}%)",
                        reusedPrefix, inputTokens.Count, suffixLength, 100.0 * reusedPrefix / inputTokens.Count);
                    float[] logits = ForwardPromptPrefill(suffixTokens, allowChunking: !queuedPromptEmbeddings);
                    cache.RecordAppend(suffixTokens, logits);
                    return logits;
                }

                case ReusePlanKind.Reset:
                default:
                {
                    reusedTokens = 0;
                    if (!cache.IsEmpty)
                        _logger.LogDebug(LogEventIds.KvCacheReusePlan,
                            "kv.reuse full reset (cached {CachedTokens} tokens, no usable common prefix with {PromptTokens}-token prompt)",
                            cache.Count, inputTokens.Count);
                    else
                        _logger.LogDebug(LogEventIds.KvCacheReusePlan,
                            "kv.reuse full prefill {PromptTokens} tokens", inputTokens.Count);

                    model.ResetKVCache();
                    cache.Reset();
                    bool queuedPromptEmbeddings = model.MultimodalInjector.QueuePromptEmbeddings(0);
                    var allTokens = inputTokens.ToArray();
                    float[] logits = ForwardPromptPrefill(allTokens, allowChunking: !queuedPromptEmbeddings);
                    cache.RecordAppend(allTokens, logits);
                    return logits;
                }
            }
        }

        public float[] ForwardPromptPrefill(int[] tokens, bool allowChunking = true)
        {
            if (tokens == null || tokens.Length == 0)
                throw new ArgumentException("Prompt token list cannot be null or empty.", nameof(tokens));

            var model = _lifecycle.Model;
            if (!allowChunking)
                return model.ForwardRefill(tokens);

            int chunkSize = ResolvePrefillChunkSize(_lifecycle.Backend, tokens.Length);
            if (chunkSize >= tokens.Length)
                return model.ForwardRefill(tokens);

            _logger.LogInformation(LogEventIds.PromptChunking,
                "prompt.chunking total={TotalTokens} chunkSize={ChunkSize} backend={Backend}",
                tokens.Length, chunkSize, _lifecycle.LoadedBackend ?? _lifecycle.Backend.ToString());

            float[] logits = null;
            int chunkIndex = 0;
            for (int start = 0; start < tokens.Length; start += chunkSize)
            {
                int length = Math.Min(chunkSize, tokens.Length - start);
                int[] chunk = new int[length];
                Array.Copy(tokens, start, chunk, 0, length);
                logits = model.ForwardRefill(chunk);
                chunkIndex++;
                _logger.LogTrace(LogEventIds.PromptChunking,
                    "prompt.chunking chunk={ChunkIndex} start={Start} length={Length}",
                    chunkIndex, start, length);
            }

            return logits;
        }

        public List<int> TruncatePromptToContext(ChatSession session, List<int> inputTokens, int maxTokens)
        {
            var model = _lifecycle.Model;
            int maxCtx = model.MaxContextLength;
            if (maxCtx <= 0 || inputTokens == null || inputTokens.Count + maxTokens <= maxCtx)
                return inputTokens;

            int available = maxCtx - maxTokens;
            if (available < 1)
            {
                throw new InvalidOperationException(
                    $"Prompt ({inputTokens.Count} tokens) exceeds the model's context limit ({maxCtx} tokens). " +
                    "Please shorten the input or reduce attached file size.");
            }

            int trimStart = inputTokens.Count - available;
            trimStart = model.MultimodalInjector.ClampTrimStart(trimStart);
            int kept = inputTokens.Count - trimStart;
            if (kept < 1)
            {
                throw new InvalidOperationException(
                    $"Prompt ({inputTokens.Count} tokens) exceeds the model's context limit ({maxCtx} tokens). " +
                    "Please shorten the input or reduce attached file size.");
            }

            _logger.LogWarning(LogEventIds.PromptTruncated,
                "prompt.truncated from {OriginalTokens} to {KeptTokens} tokens (contextLimit={ContextLimit}, generationReserve={MaxTokens}, sessionId={SessionId})",
                inputTokens.Count, kept, maxCtx, maxTokens, session?.Id ?? "(none)");
            model.MultimodalInjector.TrimPreparedPrompt(trimStart);
            session.TrackedHistory.Clear();
            session.KVCache.Reset();
            model.ResetKVCache();
            return inputTokens.GetRange(trimStart, kept);
        }

        public static int ResolvePrefillChunkSize(BackendType backend, int tokenCount)
        {
            if (tokenCount <= 0)
                return 0;

            // Chunked prefill keeps attention score tensors bounded and avoids
            // O(n^2) blowup for sliding-window layers on large prompts.
            return backend == BackendType.GgmlCuda
                ? Math.Min(tokenCount, 5120)
                : Math.Min(tokenCount, 2048);
        }

        public void Dispose()
        {
            ActiveSession = null;
            _intrinsicSession.Dispose();
        }

        private static int[] CopyTokenRange(IList<int> tokens, int start, int length)
        {
            var result = new int[length];
            for (int i = 0; i < length; i++)
                result[i] = tokens[start + i];
            return result;
        }
    }
}
