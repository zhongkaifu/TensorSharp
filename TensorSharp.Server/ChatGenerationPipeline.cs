// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Engine-backed implementation. Submits each chat / generate request to the
// shared <see cref="TensorSharp.Runtime.Scheduling.InferenceEngine"/>, then
// streams tokens off the returned <see cref="InferenceRequestHandle"/>. The
// engine owns all KV-state lifecycle; sessions in this layer are pure
// history-tracking containers used by the prompt renderer to reuse raw
// assistant tokens across turns.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TensorSharp.Runtime.Scheduling;

namespace TensorSharp.Server
{
    internal sealed class ChatGenerationPipeline
    {
        private readonly ModelLifecycleService _lifecycle;
        private readonly InferenceEngineHost _engineHost;
        private readonly KVCachePromptRenderer _kvCacheRenderer;
        private readonly InferenceTelemetry _telemetry;
        private readonly ILogger _logger;

        // Per-pipeline lock guarding multimodal-prompt preparation. The
        // multimodal-prep serialisation is now handled by
        // ModelBase.GpuComputeLock (shared with the InferenceEngine worker)
        // so a vision encoder on the request thread can't race the engine's
        // batched forward on the GPU.

        public ChatGenerationPipeline(
            ModelLifecycleService lifecycle,
            InferenceEngineHost engineHost,
            KVCachePromptRenderer kvCacheRenderer,
            InferenceTelemetry telemetry,
            ILogger logger)
        {
            _lifecycle = lifecycle ?? throw new ArgumentNullException(nameof(lifecycle));
            _engineHost = engineHost ?? throw new ArgumentNullException(nameof(engineHost));
            _kvCacheRenderer = kvCacheRenderer ?? throw new ArgumentNullException(nameof(kvCacheRenderer));
            _telemetry = telemetry ?? throw new ArgumentNullException(nameof(telemetry));
            _logger = logger ?? Microsoft.Extensions.Logging.Abstractions.NullLogger.Instance;
        }

        public async IAsyncEnumerable<string> ChatStreamAsync(
            ChatSession session,
            List<ChatMessage> history,
            int maxTokens,
            [EnumeratorCancellation] CancellationToken cancellationToken,
            SamplingConfig samplingConfig = null,
            List<ToolFunction> tools = null,
            bool enableThinking = false)
        {
            await foreach (var (piece, _, _, _, _, _, _, _) in
                ChatStreamWithMetricsAsync(session, history, maxTokens, cancellationToken, samplingConfig, tools, enableThinking))
            {
                if (!string.IsNullOrEmpty(piece))
                    yield return piece;
            }
        }

        public async IAsyncEnumerable<(string piece, bool done, int promptTokens, int evalTokens, int kvCacheReusedTokens, long totalNs, long promptNs, long evalNs)>
            ChatStreamWithMetricsAsync(
                ChatSession session,
                List<ChatMessage> history,
                int maxTokens,
                [EnumeratorCancellation] CancellationToken cancellationToken,
                SamplingConfig samplingConfig = null,
                List<ToolFunction> tools = null,
                bool enableThinking = false)
        {
            session ??= new ChatSession("__svc_intrinsic__");
            var model = _lifecycle.Model
                ?? throw new InvalidOperationException("No model is loaded.");
            var engine = _engineHost.TryGetEngine()
                ?? throw new InvalidOperationException(
                    "Continuous-batching engine is unavailable for this model " +
                    "(the model supports neither IBatchedPagedModel.ForwardBatch " +
                    "nor IModelArchitecture.SupportsKVStateSnapshot).");

            string arch = model.Config.Architecture;
            var preparedHistory = ChatHistoryPreparer.PrepareHistoryForInference(history, arch, _logger);
            var renderHistory = ChatHistoryPreparer.AugmentWithCachedRawTokens(preparedHistory, session.TrackedHistory);

            using var chatScope = _telemetry.BeginInferenceScope(
                session, _lifecycle.LoadedModelName, _lifecycle.LoadedBackend, "chat.stream");
            _telemetry.LogChatStarted(arch, maxTokens, enableThinking, tools, preparedHistory, samplingConfig);

            // Pre-allocate the request id so the multimodal injector can
            // bucket per-request prepared embeddings. Without this, two
            // concurrent multimodal requests would share the same injector
            // state, and either get their image embeddings consumed by the
            // wrong sequence's Forward() call or vanish entirely (because
            // the engine's per-sequence Forward path never queues from a
            // shared bucket).
            string requestId = $"chat-{Guid.NewGuid():N}";
            bool injectorBucketCreated = false;

            var promptSw = Stopwatch.StartNew();
            List<int> inputTokens;
            bool hasMultimodal = RequiresMultimodalPreparation(renderHistory);
            if (hasMultimodal)
            {
                // Multimodal prompt preparation drives the vision/audio
                // encoder, which runs many GGML ops on the backend. Take
                // the model-wide GPU compute lock so we don't race the
                // engine's worker (which is doing the same thing for
                // batched forward) - concurrent GGML on Metal/CUDA from
                // two threads aborts the process via
                // ggml_metal_synchronize. The lock also subsumes the
                // injector-state serialisation that the old
                // _multimodalGate provided, because the prepared-embedding
                // list lives on the model.
                //
                // The encoder forward is long (image 100ms–2s, audio
                // similar, video longer), so to keep concurrent in-flight
                // decode requests from freezing we COOPERATIVELY YIELD
                // the lock between encoder blocks. Each Gemma 4 vision /
                // audio encoder calls ModelBase.YieldGpuComputeLock at
                // its per-block boundary, which releases this lock, lets
                // a waiting engine-worker thread run one ExecuteStep
                // (~50–200ms of inference progress), then re-acquires.
                // The encoder pays a few percent overhead per yield in
                // exchange for in-flight decodes staying responsive.
                // Disable via TS_ENCODER_YIELD=0 for A/B testing.
                //
                // Other models' encoders (Qwen3.5 vision, Mistral 3
                // vision, etc.) currently DON'T yield — they still hold
                // the lock for the full encode. Adding YieldGpuComputeLock
                // calls to their per-layer/per-block loops is the same
                // ~3-line change as for Gemma 4 and recommended.
                lock (model.GpuComputeLock)
                {
                    inputTokens = _kvCacheRenderer.RenderToTokens(
                        model.Tokenizer, model.Config.ChatTemplate, renderHistory, arch,
                        addGenerationPrompt: true, tools: tools, enableThinking: enableThinking);
                    inputTokens = model.MultimodalInjector.ProcessPromptTokens(renderHistory, inputTokens, requestId);
                    injectorBucketCreated = true;
                    inputTokens = TruncatePromptToContext(session, inputTokens, maxTokens, requestId);
                }
            }
            else
            {
                inputTokens = _kvCacheRenderer.RenderToTokens(
                    model.Tokenizer, model.Config.ChatTemplate, renderHistory, arch,
                    addGenerationPrompt: true, tools: tools, enableThinking: enableThinking);
                inputTokens = TruncatePromptToContext(session, inputTokens, maxTokens);
            }

            int promptTokenCount = inputTokens.Count;
            var cfg = samplingConfig ?? SamplingConfig.Default;

            // Fingerprint the media (images/audio/video) folded into this prompt.
            // The image/placeholder token IDs are identical across requests, so the
            // prefix-cache block hashes must be salted with the actual media content
            // — otherwise a later request with the *same* template but a *different*
            // image would adopt the previous image's K/V blocks and describe a stale
            // image. Null for text-only prompts (no change to their cache behavior).
            string mediaFingerprint = BuildMediaFingerprint(renderHistory);

            var seq = new SequenceState(
                requestId: requestId,
                promptTokens: inputTokens,
                maxNewTokens: maxTokens,
                blockSize: engine.PoolStats.blockSize,
                samplingConfig: cfg,
                userTag: session,
                mediaFingerprint: mediaFingerprint);

            promptSw.Stop();
            long promptNs = InferenceTelemetry.ToNanos(promptSw.ElapsedTicks);

            var evalSw = Stopwatch.StartNew();
            var handle = engine.SubmitRequest(seq, cancellationToken);
            var generatedTokens = new List<int>();
            var rawBytes = new List<byte>();
            int prevCharLen = 0;
            string finishReason = "max_tokens";
            bool wasCancelled = false;
            int kvCacheReusedTokens = 0;
            long timeToFirstTokenMs = 0;
            bool firstTokenSampled = false;
            var totalSw = Stopwatch.StartNew();

            try
            {
            // Stream tokens off the engine handle, doing UTF-8-valid piece
            // accumulation and stop-sequence detection in this layer.
            await foreach (var nextToken in handle.Tokens.ReadAllAsync(cancellationToken).ConfigureAwait(false))
            {
                if (cancellationToken.IsCancellationRequested)
                {
                    wasCancelled = true;
                    finishReason = "cancelled";
                    engine.Abort(seq.RequestId);
                    break;
                }

                generatedTokens.Add(nextToken);
                model.Tokenizer.AppendTokenBytes(nextToken, rawBytes);
                int validLen = FindValidUtf8Length(rawBytes);
                string decoded = Encoding.UTF8.GetString(rawBytes.GetRange(0, validLen).ToArray());
                string piece = prevCharLen < decoded.Length ? decoded.Substring(prevCharLen) : "";
                prevCharLen = decoded.Length;

                if (!firstTokenSampled)
                {
                    firstTokenSampled = true;
                    timeToFirstTokenMs = (long)totalSw.Elapsed.TotalMilliseconds;
                }

                bool stopRequested = false;
                if (cfg.StopSequences != null && cfg.StopSequences.Count > 0)
                {
                    var sampler = new TokenSampler(cfg);
                    var (_, shouldStop) = sampler.CheckStopSequences(decoded);
                    if (shouldStop)
                    {
                        stopRequested = true;
                        finishReason = "stop_sequence";
                    }
                }

                if (piece.Length > 0)
                    yield return (piece, false, 0, 0, 0, 0, 0, 0);

                if (stopRequested)
                {
                    engine.Abort(seq.RequestId);
                    break;
                }
            }

            InferenceCompletion completion;
            try
            {
                completion = await handle.Completion.ConfigureAwait(false);
                kvCacheReusedTokens = completion.PrefixCacheReusedTokens;
                if (!wasCancelled && finishReason == "max_tokens")
                {
                    finishReason = completion.FinishReason ?? finishReason;
                }
            }
            catch (OperationCanceledException)
            {
                wasCancelled = true;
                finishReason = "cancelled";
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Engine submission failed for session {SessionId}", session.Id);
                throw;
            }

            string assistantText = Encoding.UTF8.GetString(rawBytes.ToArray());
            evalSw.Stop();
            totalSw.Stop();

            ChatHistoryPreparer.UpdateTrackedHistory(
                session.TrackedHistory, renderHistory, assistantText, generatedTokens);

            double evalSeconds = evalSw.Elapsed.TotalSeconds;
            double tokensPerSecond = (evalSeconds > 0 && generatedTokens.Count > 0)
                ? generatedTokens.Count / evalSeconds
                : 0;
            double kvCacheReusePercent = promptTokenCount > 0
                ? 100.0 * kvCacheReusedTokens / promptTokenCount
                : 0.0;

            _telemetry.LogChatFinished(
                wasCancelled, generatedTokens.Count, promptTokenCount, kvCacheReusedTokens,
                kvCacheReusePercent, timeToFirstTokenMs, totalSw.Elapsed.TotalMilliseconds,
                tokensPerSecond, finishReason, assistantText);

            long evalNs = InferenceTelemetry.ToNanos(evalSw.ElapsedTicks);
            long totalNs = InferenceTelemetry.ToNanos(totalSw.ElapsedTicks);
            yield return ("", true, promptTokenCount, generatedTokens.Count, kvCacheReusedTokens,
                          totalNs, promptNs, evalNs);
            }
            finally
            {
                if (injectorBucketCreated)
                {
                    // Drop the per-request prepared-embedding bucket so it
                    // doesn't leak across requests. Runs on the happy path,
                    // on cancellation, on early-stop, and on iterator
                    // abandonment (the async iterator's Dispose runs the
                    // finally block).
                    model.MultimodalInjector.ClearPreparedPromptState(requestId);
                }
            }
        }

        public async IAsyncEnumerable<(string piece, bool done, int promptTokens, int evalTokens, int kvCacheReusedTokens, long totalNs, long promptNs, long evalNs)>
            GenerateStreamAsync(
                ChatSession session,
                string prompt,
                List<string> imagePaths,
                int maxTokens,
                [EnumeratorCancellation] CancellationToken cancellationToken,
                SamplingConfig samplingConfig = null)
        {
            // Generate uses the same engine path as chat - it just wraps the
            // prompt in a single-message history and skips multi-turn history
            // tracking. We do NOT update session.TrackedHistory here because
            // GenerateStreamAsync is the non-conversational endpoint used by
            // Ollama's /api/generate.
            var oneShot = new List<ChatMessage>
            {
                new ChatMessage { Role = "user", Content = prompt, ImagePaths = imagePaths }
            };
            var freshSession = new ChatSession("__generate_intrinsic__");
            await foreach (var item in ChatStreamWithMetricsAsync(
                freshSession, oneShot, maxTokens, cancellationToken, samplingConfig))
            {
                yield return item;
            }
        }

        /// <summary>Trim the prompt so the prompt+max-tokens fits inside the
        /// model's context window. Multimodal embedding spans are not split.</summary>
        public List<int> TruncatePromptToContext(ChatSession session, List<int> inputTokens, int maxTokens, string requestId = null)
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
            trimStart = model.MultimodalInjector.ClampTrimStart(trimStart, requestId);
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
            model.MultimodalInjector.TrimPreparedPrompt(trimStart, requestId);
            session?.TrackedHistory.Clear();
            return inputTokens.GetRange(trimStart, kept);
        }

        private static bool RequiresMultimodalPreparation(List<ChatMessage> history)
        {
            if (history == null) return false;
            foreach (var m in history)
            {
                if (m == null) continue;
                if (m.ImagePaths != null && m.ImagePaths.Count > 0) return true;
                if (m.AudioPaths != null && m.AudioPaths.Count > 0) return true;
            }
            return false;
        }

        /// <summary>
        /// Build a stable fingerprint of every image/audio attachment in the
        /// prompt, in prompt order. Uploads are stored under content-addressed
        /// filenames, so the path identifies the content: identical media yields
        /// the same fingerprint (prefix cache reused), different media yields a
        /// different one (prefix cache correctly bypassed). Returns null when the
        /// prompt has no media, leaving text-only cache behavior unchanged.
        /// </summary>
        private static string BuildMediaFingerprint(List<ChatMessage> history)
        {
            if (history == null) return null;
            StringBuilder sb = null;
            foreach (var m in history)
            {
                if (m == null) continue;
                if (m.ImagePaths != null)
                {
                    foreach (var p in m.ImagePaths)
                    {
                        if (string.IsNullOrEmpty(p)) continue;
                        (sb ??= new StringBuilder()).Append(m.IsVideo ? "vid:" : "img:").Append(p).Append('\n');
                    }
                }
                if (m.AudioPaths != null)
                {
                    foreach (var p in m.AudioPaths)
                    {
                        if (string.IsNullOrEmpty(p)) continue;
                        (sb ??= new StringBuilder()).Append("aud:").Append(p).Append('\n');
                    }
                }
            }
            return sb?.ToString();
        }

        /// <summary>
        /// Find the length of the longest prefix of the byte buffer that forms valid UTF-8.
        /// Strips any trailing incomplete multi-byte sequence.
        /// </summary>
        private static int FindValidUtf8Length(List<byte> bytes)
        {
            int len = bytes.Count;
            if (len == 0) return 0;

            for (int i = 1; i <= Math.Min(4, len); i++)
            {
                byte b = bytes[len - i];
                if ((b & 0x80) == 0) return len;
                if ((b & 0xE0) == 0xC0) return (i >= 2) ? len : len - i;
                if ((b & 0xF0) == 0xE0) return (i >= 3) ? len : len - i;
                if ((b & 0xF8) == 0xF0) return (i >= 4) ? len : len - i;
                if ((b & 0xC0) == 0x80) continue;
                return len;
            }
            return len;
        }
    }
}
