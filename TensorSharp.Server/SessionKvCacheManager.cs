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
        private readonly PagedKvCacheConfig _pagedConfig = PagedKvCacheConfig.FromEnvironment();
        private PagedKvCacheManager _pagedManager;
        private string _pagedManagerFingerprint;

        public SessionKvCacheManager(ModelLifecycleService lifecycle, ILogger logger)
        {
            _lifecycle = lifecycle ?? throw new ArgumentNullException(nameof(lifecycle));
            _logger = logger ?? Microsoft.Extensions.Logging.Abstractions.NullLogger.Instance;
        }

        public PagedKvCacheManager PagedManager => _pagedManager;
        public PagedKvCacheConfig PagedConfig => _pagedConfig;

        private PagedKvCacheManager EnsurePagedManager()
        {
            if (!_pagedConfig.Enabled)
                return null;
            var model = _lifecycle.Model;
            if (model == null || !model.SupportsKVStateSnapshot)
                return null;

            string fp = model.KVStateFingerprint;
            if (_pagedManager == null || !string.Equals(_pagedManagerFingerprint, fp, StringComparison.Ordinal))
            {
                _pagedManager?.Dispose();
                // The TurboQuant codec is opt-in via TS_KV_PAGED_QUANT_BITS;
                // FromEnvironment(model) returns null both when the env var
                // is unset and when the model has recurrent SSM state that
                // quantization would corrupt (Qwen3.5/3.6 GatedDeltaNet,
                // Nemotron Mamba2). See the codec's docs for the why.
                IKvBlockCodec codec = TurboQuantKvCodec.FromEnvironment(model);
                _pagedManager = new PagedKvCacheManager(_pagedConfig, fp, _logger, codec);
                _pagedManagerFingerprint = fp;
                string codecName = codec != null
                    ? codec.Name
                    : (model.RequiresPerBlockCapture ? "passthrough (recurrent-state model)" : "passthrough");
                _logger.LogInformation(LogEventIds.PagedKvCacheTierInit,
                    "kv.paged manager attached for {Fingerprint} (blockSize={BlockSize}, ramMB={RamMB}, ssd={SsdDir}, codec={Codec})",
                    fp, _pagedConfig.BlockSize, _pagedConfig.MaxRamBytes / (1024 * 1024),
                    string.IsNullOrEmpty(_pagedConfig.SsdDirectory) ? "(disabled)" : _pagedConfig.SsdDirectory,
                    codecName);
            }
            return _pagedManager;
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
            // The paged block tier holds (model, dtype, layout)-fingerprinted blobs;
            // when the underlying model changes the fingerprints are stale and the
            // cached bytes could even be wrong-shape for the new model.
            _pagedManager?.Dispose();
            _pagedManager = null;
            _pagedManagerFingerprint = null;
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

            // Try cross-session paged restore *before* the per-session plan: when
            // another tab / chat earlier paid the prefill cost for the same prompt
            // prefix, the matching K/V blocks are already serialized and ready to
            // inject - even when this session's local cache offers no reuse.
            TryActivatePagedRestore(model, cache, inputTokens);

            ReusePlan plan = cache.PlanReuse(inputTokens, model.SupportsKVCacheTruncation);

            switch (plan.Kind)
            {
                case ReusePlanKind.ExactMatch:
                {
                    reusedTokens = inputTokens.Count;
                    _logger.LogDebug(LogEventIds.KvCacheReusePlan,
                        "kv.reuse exact match reusing {ReusedTokens}/{TotalTokens} cached tokens (saved 100%)",
                        inputTokens.Count, inputTokens.Count);
                    return plan.CachedLogits;
                }

                case ReusePlanKind.PartialReuse:
                {
                    int reusedPrefix = plan.ReusedPrefixLength;
                    int suffixLength = plan.TokensToForward;
                    reusedTokens = reusedPrefix;

                    model.TruncateKVCache(reusedPrefix);
                    cache.TruncateTo(reusedPrefix);

                    var suffixTokens = CopyTokenRange(inputTokens, reusedPrefix, suffixLength);

                    _logger.LogDebug(LogEventIds.KvCacheReusePlan,
                        "kv.reuse partial keeping {ReusedTokens}/{TotalTokens} tokens, forwarding {NewTokens} new tokens (saved {SavedPercent:F0}%)",
                        reusedPrefix, inputTokens.Count, suffixLength, 100.0 * reusedPrefix / inputTokens.Count);
                    Action<int, int> captureCb = BuildPerChunkCaptureCallback(model, cache, inputTokens, suffixTokens, reusedPrefix);
                    float[] logits = ForwardPromptPrefill(suffixTokens, reusedPrefix, allowChunking: true, captureCb);
                    if (captureCb == null)
                    {
                        cache.RecordAppend(suffixTokens, logits);
                        CapturePagedBlocks(model, cache);
                    }
                    else
                    {
                        // The callback streamed tokens into the cache as each chunk
                        // completed; refresh the trailing logits so they reflect the
                        // last forward call rather than the (now-stale) null we
                        // recorded mid-chunk.
                        cache.TruncateTo(reusedPrefix + suffixLength - 1);
                        cache.RecordAppend(suffixTokens[suffixLength - 1], logits);
                    }
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
                    var allTokens = inputTokens.ToArray();
                    Action<int, int> captureCb = BuildPerChunkCaptureCallback(model, cache, inputTokens, allTokens, prefixOffset: 0);
                    float[] logits = ForwardPromptPrefill(allTokens, 0, allowChunking: true, captureCb);
                    if (captureCb == null)
                    {
                        cache.RecordAppend(allTokens, logits);
                        CapturePagedBlocks(model, cache);
                    }
                    else
                    {
                        cache.TruncateTo(allTokens.Length - 1);
                        cache.RecordAppend(allTokens[allTokens.Length - 1], logits);
                    }
                    return logits;
                }
            }
        }

        /// <summary>
        /// Build a per-chunk callback that records the tokens just forwarded into
        /// <paramref name="cache"/> and invokes <see cref="PagedKvCacheManager.Capture"/>
        /// so any block boundaries we just crossed get captured. Returns null
        /// when the model doesn't need per-chunk capture (purely-attention models
        /// can be captured once at the end - the callback would just add overhead).
        /// </summary>
        private Action<int, int> BuildPerChunkCaptureCallback(
            IModelArchitecture model, KVCache cache, List<int> inputTokens, int[] suffixTokens, int prefixOffset)
        {
            var manager = EnsurePagedManager();
            if (manager == null) return null;
            if (!model.RequiresPerBlockCapture) return null;

            return (chunkStart, chunkLength) =>
            {
                int suffixIndex = chunkStart - prefixOffset;
                if (suffixIndex < 0 || suffixIndex + chunkLength > suffixTokens.Length)
                    return;
                // Record the chunk into the session's token list so the manager
                // hashes the right prefix; logits are unknown until the last
                // chunk so we pass null here and patch the final value above.
                var chunkSlice = new int[chunkLength];
                Array.Copy(suffixTokens, suffixIndex, chunkSlice, 0, chunkLength);
                cache.RecordAppend(chunkSlice, null);
                manager.Capture(model, cache.Tokens, cache.Count);
            };
        }

        private void TryActivatePagedRestore(IModelArchitecture model, KVCache cache, List<int> inputTokens)
        {
            var manager = EnsurePagedManager();
            if (manager == null)
                return;

            // Only adopt the paged plan when it strictly beats the in-session prefix
            // length. Reusing the model's existing KV state is always cheaper than
            // resetting + reinjecting bytes, so we never pay the swap cost for free.
            int inSessionPrefix = cache.CommonPrefixLength(inputTokens);
            if (!model.SupportsKVCacheTruncation && inSessionPrefix < cache.Count)
                inSessionPrefix = 0;

            int availableBlocks = manager.CountAvailableBlocks(model, inputTokens);
            int availableTokens = availableBlocks * manager.BlockSize;
            if (availableTokens <= inSessionPrefix)
                return;

            // Swap to the paged plan: drop the current cache and inject the matching
            // blocks. The session's tracked history is preserved because the bytes we
            // restore correspond to the prefix tokens the caller already supplied.
            model.ResetKVCache();
            cache.Reset();
            int restored = manager.TryRestorePrefix(model, inputTokens);
            if (restored > 0)
            {
                var restoredTokens = CopyTokenRange(inputTokens, 0, restored);
                cache.RecordAppend(restoredTokens, null);
                _logger.LogInformation(LogEventIds.PagedKvCacheRestore,
                    "kv.paged adopted restore of {Restored}/{Total} tokens (beat in-session prefix of {InSession})",
                    restored, inputTokens.Count, inSessionPrefix);
            }
        }

        private void CapturePagedBlocks(IModelArchitecture model, KVCache cache)
        {
            var manager = EnsurePagedManager();
            if (manager == null || cache.IsEmpty)
                return;
            manager.Capture(model, cache.Tokens, cache.Count);
        }

        public float[] ForwardPromptPrefill(int[] tokens, bool allowChunking = true)
            => ForwardPromptPrefill(tokens, 0, allowChunking, perChunkCallback: null);

        public float[] ForwardPromptPrefill(int[] tokens, int promptStartToken, bool allowChunking = true)
            => ForwardPromptPrefill(tokens, promptStartToken, allowChunking, perChunkCallback: null);

        /// <summary>
        /// Forward <paramref name="tokens"/> through the model in chunks. When
        /// <paramref name="perChunkCallback"/> is supplied it is invoked after
        /// each chunk's KV state has been written, so callers (notably the paged
        /// cache) can checkpoint mid-prefill. For models with recurrent state
        /// this is essential: the running state at position N must be captured
        /// before the next chunk mutates it.
        /// </summary>
        public float[] ForwardPromptPrefill(int[] tokens, int promptStartToken, bool allowChunking, Action<int, int> perChunkCallback)
        {
            if (tokens == null || tokens.Length == 0)
                throw new ArgumentException("Prompt token list cannot be null or empty.", nameof(tokens));
            if (promptStartToken < 0)
                throw new ArgumentOutOfRangeException(nameof(promptStartToken));

            var model = _lifecycle.Model;
            int chunkSize;
            if (!allowChunking)
            {
                chunkSize = tokens.Length;
            }
            else
            {
                chunkSize = ResolvePrefillChunkSize(_lifecycle.Backend, tokens.Length);
                // Recurrent models need their running state captured at every block
                // boundary - shrink the chunks so callbacks land on those boundaries.
                if (perChunkCallback != null && model.RequiresPerBlockCapture && _pagedConfig.Enabled)
                {
                    int blockSize = _pagedConfig.BlockSize;
                    if (blockSize > 0 && blockSize < chunkSize)
                        chunkSize = blockSize;
                }
            }

            if (chunkSize >= tokens.Length && perChunkCallback == null)
                return ForwardPromptChunk(tokens, promptStartToken);

            _logger.LogInformation(LogEventIds.PromptChunking,
                "prompt.chunking total={TotalTokens} chunkSize={ChunkSize} backend={Backend} capture={Capture}",
                tokens.Length, chunkSize, _lifecycle.LoadedBackend ?? _lifecycle.Backend.ToString(),
                perChunkCallback != null);

            float[] logits = null;
            int chunkIndex = 0;
            for (int start = 0; start < tokens.Length; start += chunkSize)
            {
                int length = Math.Min(chunkSize, tokens.Length - start);
                int[] chunk = new int[length];
                Array.Copy(tokens, start, chunk, 0, length);
                logits = ForwardPromptChunk(chunk, promptStartToken + start);
                chunkIndex++;
                perChunkCallback?.Invoke(promptStartToken + start, length);
                _logger.LogTrace(LogEventIds.PromptChunking,
                    "prompt.chunking chunk={ChunkIndex} start={Start} length={Length}",
                    chunkIndex, start, length);
            }

            return logits;
        }

        private float[] ForwardPromptChunk(int[] tokens, int promptStartToken)
        {
            var model = _lifecycle.Model;
            model.MultimodalInjector.QueuePromptEmbeddingsForSlice(promptStartToken, tokens.Length);
            return model.ForwardRefill(tokens);
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
            => PrefillChunking.ResolveChunkSize(backend, tokenCount);

        public void Dispose()
        {
            ActiveSession = null;
            _intrinsicSession.Dispose();
            _pagedManager?.Dispose();
            _pagedManager = null;
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
