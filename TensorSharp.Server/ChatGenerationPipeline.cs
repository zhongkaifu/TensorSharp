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
using System.Threading.Channels;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TensorSharp.Runtime.Scheduling;

namespace TensorSharp.Server
{
    /// <summary>A single streaming update from the DiffusionGemma denoising pipeline.
    /// Previews are intermediate best-guess canvases (whole-text "replace" semantics); the final
    /// update carries the trimmed answer; the done update carries metrics.</summary>
    internal readonly record struct DiffusionStreamUpdate(
        string Text, bool IsPreview, bool Done, int Step, int TotalSteps,
        int PromptTokens, int EvalTokens, long TotalNs);

    internal sealed class ChatGenerationPipeline : IDisposable
    {
        private readonly ModelLifecycleService _lifecycle;
        private readonly InferenceEngineHost _engineHost;
        private readonly KVCachePromptRenderer _kvCacheRenderer;
        private readonly InferenceTelemetry _telemetry;
        private readonly ILogger _logger;

        // DiffusionGemma's continuous-batching scheduler (the diffusion analog of the AR InferenceEngine).
        // Created lazily and rebound when the loaded model changes; disposed on model swap / shutdown.
        private readonly object _diffSchedLock = new();
        private DiffusionBatchScheduler _diffScheduler;
        private DiffusionGemmaModel _diffSchedModel;
        // Max canvases denoised together. Each extra concurrent request adds ~one canvas's worth of
        // activation memory, so on a memory-tight box (e.g. 24 GB running a 16.8 GB model) 2 is the safe
        // default; raise via DIFFUSION_MAX_BATCH when there's GPU headroom for more aggregate throughput.
        private static readonly int DiffusionMaxBatch =
            int.TryParse(Environment.GetEnvironmentVariable("DIFFUSION_MAX_BATCH"), out int mb) && mb > 0 ? mb : 2;

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

            // DiffusionGemma does not use the autoregressive continuous-batching engine; it generates a
            // whole block via iterative denoising. Drive it here and surface only the final answer to the
            // append-only protocols (OpenAI/Ollama/non-streaming). The Web UI uses DiffusionChatStreamAsync
            // directly for a live denoising preview.
            if (model is DiffusionGemmaModel)
            {
                await foreach (var u in DiffusionChatStreamAsync(session, history, maxTokens, cancellationToken)
                    .ConfigureAwait(false))
                {
                    if (u.Done)
                        yield return ("", true, u.PromptTokens, u.EvalTokens, 0, u.TotalNs, 0, u.TotalNs);
                    else if (!u.IsPreview && u.Text.Length > 0)
                        yield return (u.Text, false, 0, 0, 0, 0, 0, 0);
                }
                yield break;
            }

            var engine = _engineHost.TryGetEngine()
                ?? throw new InvalidOperationException(
                    "Continuous-batching engine is unavailable for this model " +
                    "(the model supports neither IBatchedPagedModel.ForwardBatch " +
                    "nor IModelArchitecture.SupportsKVStateSnapshot).");

            string arch = model.Config.Architecture;
            var preparedHistory = ChatHistoryPreparer.PrepareHistoryForInference(history, arch, _logger);
            List<ChatMessage> renderHistory;
            lock (session.HistoryLock)
                renderHistory = ChatHistoryPreparer.AugmentWithCachedRawTokens(preparedHistory, session.TrackedHistory);

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

            lock (session.HistoryLock)
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

        /// <summary>
        /// Drives a DiffusionGemma chat turn via the EntropyBound denoising sampler and yields rich
        /// streaming updates: a live preview after every denoising step (the current best-guess canvas,
        /// "replace" semantics), then the final trimmed answer, then a done update with metrics.
        /// The sampler runs on a background thread under <see cref="ModelBase.GpuComputeLock"/> and pushes
        /// updates through a channel so the request thread can stream them without blocking.
        /// </summary>
        public async IAsyncEnumerable<DiffusionStreamUpdate> DiffusionChatStreamAsync(
            ChatSession session,
            List<ChatMessage> history,
            int maxTokens,
            [EnumeratorCancellation] CancellationToken cancellationToken)
        {
            session ??= new ChatSession("__svc_intrinsic__");
            var model = (DiffusionGemmaModel)(_lifecycle.Model
                ?? throw new InvalidOperationException("No model is loaded."));
            string arch = model.Config.Architecture;

            var preparedHistory = ChatHistoryPreparer.PrepareHistoryForInference(history, arch, _logger);
            // Snapshot the (shared, for DefaultSession) tracked history under the session lock so a parallel
            // request's turn-end rewrite can't race this read.
            List<ChatMessage> renderHistory;
            lock (session.HistoryLock)
                renderHistory = ChatHistoryPreparer.AugmentWithCachedRawTokens(preparedHistory, session.TrackedHistory);

            using var chatScope = _telemetry.BeginInferenceScope(
                session, _lifecycle.LoadedModelName, _lifecycle.LoadedBackend, "diffusion.chat.stream");

            var promptSw = Stopwatch.StartNew();
            List<int> inputTokens = _kvCacheRenderer.RenderToTokens(
                model.Tokenizer, model.Config.ChatTemplate, renderHistory, arch,
                addGenerationPrompt: true, tools: null, enableThinking: false);
            inputTokens = TruncatePromptToContext(session, inputTokens, maxTokens);
            int promptTokenCount = inputTokens.Count;
            promptSw.Stop();

            int canvas = model.CanvasLength;
            int blocks = Math.Max(1, (Math.Max(1, maxTokens) + canvas - 1) / canvas);
            var ebParams = new DiffusionEbParams
            {
                MaxDenoisingSteps = DiffusionMaxSteps,
                Seed = Random.Shared.Next(),
                MaxBlocks = blocks,
            };

            // Submit to the shared continuous-batching scheduler. Several concurrent requests are denoised
            // together in one batched forward per step (one background thread owns the GPU lock), so a second
            // parallel request streams immediately instead of waiting for the first to finish.
            var scheduler = GetDiffusionScheduler(model);
            var handle = scheduler.Submit(inputTokens.ToArray(), ebParams, cancellationToken);

            var totalSw = Stopwatch.StartNew();

            // Stream previews as they arrive (cancellation surfaces as OperationCanceledException, which the
            // adapter catches and finalizes).
            await foreach (var preview in handle.Previews.ReadAllAsync(cancellationToken).ConfigureAwait(false))
            {
                string previewText = DecodeDiffusionPreview(model, preview.Tokens);
                yield return new DiffusionStreamUpdate(
                    previewText, IsPreview: true, Done: false, preview.Step + 1, preview.TotalSteps, 0, 0, 0);
            }

            var generated = await handle.Completion.ConfigureAwait(false);
            totalSw.Stop();

            generated ??= new List<int>();
            string finalText = model.Tokenizer.Decode(generated);

            lock (session.HistoryLock)
                ChatHistoryPreparer.UpdateTrackedHistory(
                    session.TrackedHistory, renderHistory, finalText, generated);

            long totalNs = InferenceTelemetry.ToNanos(totalSw.ElapsedTicks);
            _telemetry.LogChatFinished(
                cancellationToken.IsCancellationRequested, generated.Count, promptTokenCount, 0, 0.0,
                0, totalSw.Elapsed.TotalMilliseconds,
                totalSw.Elapsed.TotalSeconds > 0 ? generated.Count / totalSw.Elapsed.TotalSeconds : 0,
                cancellationToken.IsCancellationRequested ? "cancelled" : "stop", finalText);

            // Final answer (replaces the last preview), then the terminal metrics update.
            yield return new DiffusionStreamUpdate(finalText, IsPreview: false, Done: false, 0, 0, 0, 0, 0);
            yield return new DiffusionStreamUpdate("", IsPreview: false, Done: true, 0, 0,
                promptTokenCount, generated.Count, totalNs);
        }

        /// <summary>Get the diffusion batch scheduler bound to the currently-loaded model, (re)creating it
        /// when the model changes. The scheduler owns a single GPU-compute worker thread.</summary>
        private DiffusionBatchScheduler GetDiffusionScheduler(DiffusionGemmaModel model)
        {
            lock (_diffSchedLock)
            {
                if (_diffScheduler != null && ReferenceEquals(_diffSchedModel, model))
                    return _diffScheduler;
                _diffScheduler?.Dispose();
                _diffScheduler = new DiffusionBatchScheduler(model, _logger, DiffusionMaxBatch);
                _diffSchedModel = model;
                _logger.LogInformation("DiffusionGemma batch scheduler constructed (maxBatch={MaxBatch})", DiffusionMaxBatch);
                return _diffScheduler;
            }
        }

        /// <summary>Tear down the diffusion scheduler (joins its worker thread). Called on model swap and
        /// shutdown so the worker doesn't outlive / race the model it references.</summary>
        public void ResetDiffusionScheduler()
        {
            lock (_diffSchedLock)
            {
                _diffScheduler?.Dispose();
                _diffScheduler = null;
                _diffSchedModel = null;
            }
        }

        public void Dispose() => ResetDiffusionScheduler();

        // Default number of denoising steps for server-driven generation (adaptive stop usually
        // terminates earlier). Overridable via the DIFFUSION_STEPS environment variable.
        private static readonly int DiffusionMaxSteps =
            int.TryParse(Environment.GetEnvironmentVariable("DIFFUSION_STEPS"), out int s) && s > 0 ? s : 48;

        /// <summary>Decode a denoising preview canvas for display, trimmed at the first end-of-sequence
        /// token so the live view reads cleanly as it converges.</summary>
        private static string DecodeDiffusionPreview(DiffusionGemmaModel model, int[] tokens)
        {
            int cut = tokens.Length;
            for (int i = 0; i < tokens.Length; i++)
            {
                if (model.Tokenizer.IsEos(tokens[i])) { cut = i; break; }
            }
            var slice = new List<int>(cut);
            for (int i = 0; i < cut; i++) slice.Add(tokens[i]);
            try { return model.Tokenizer.Decode(slice); }
            catch { return string.Empty; }
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
