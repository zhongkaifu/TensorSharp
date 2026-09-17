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
using System.Globalization;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TensorSharp.Models.Architecture;
using TensorSharp.Runtime.Scheduling;

namespace TensorSharp.Server
{
    /// <summary>A single streaming update from the DiffusionGemma denoising pipeline.
    /// Previews are intermediate best-guess canvases (whole-text "replace" semantics); the final
    /// update carries the trimmed answer; the done update carries metrics.</summary>
    /// <param name="Text">The CONTENT channel of the canvas: the denoised text after
    /// the family's output parser has removed channel markers and the thought block.</param>
    /// <param name="Thinking">The thought block of the canvas, only when the request
    /// asked for reasoning; otherwise null.</param>
    internal readonly record struct DiffusionStreamUpdate(
        string Text, bool IsPreview, bool Done, int Step, int TotalSteps,
        int PromptTokens, int EvalTokens, long TotalNs, string? Thinking = null);

    /// <summary>A single streaming update from the autoregressive chat / generate pipeline.
    /// Ordinary updates carry only <see cref="Piece"/> (the text decoded since the last
    /// update); the one terminal update (<see cref="Done"/> = <c>true</c>) carries the token
    /// counts, the timings, and the reason generation stopped.</summary>
    /// <remarks>
    /// This was an 8-tuple until <see cref="FinishReason"/> needed a home. A named type earns
    /// its keep here because most consumers want two or three members out of nine, and a row of
    /// positional <c>_</c> discards had stopped documenting which ones.
    /// </remarks>
    /// <param name="Piece">Text decoded since the previous update; empty on the terminal update.</param>
    /// <param name="Done">True for the single terminal update that carries the metrics below.</param>
    /// <param name="PromptTokens">Prompt tokens evaluated. Terminal update only.</param>
    /// <param name="EvalTokens">Tokens generated. Terminal update only.</param>
    /// <param name="KvCacheReusedTokens">Prompt tokens served from the prefix cache. Terminal update only.</param>
    /// <param name="TotalNs">Wall-clock nanoseconds for the whole request. Terminal update only.</param>
    /// <param name="PromptNs">Nanoseconds spent rendering and preparing the prompt. Terminal update only.</param>
    /// <param name="EvalNs">Nanoseconds spent decoding. Terminal update only.</param>
    /// <param name="FinishReason">Why generation stopped — <c>max_tokens</c>, <c>stop_sequence</c>,
    /// <c>cancelled</c>, or whatever the engine reported (<c>eos</c>, <c>aborted</c>, <c>error</c>).
    /// Null on non-terminal updates. This is the pipeline's own vocabulary, NOT any protocol's:
    /// adapters must translate it through
    /// <see cref="TensorSharp.Server.ProtocolAdapters.FinishReasonMapper"/> rather than putting it
    /// on the wire raw.</param>
    public readonly record struct ChatStreamUpdate(
        string Piece,
        bool Done,
        int PromptTokens,
        int EvalTokens,
        int KvCacheReusedTokens,
        long TotalNs,
        long PromptNs,
        long EvalNs,
        string FinishReason)
    {
        /// <summary>An ordinary (non-terminal) update carrying just newly decoded text.</summary>
        public static ChatStreamUpdate Text(string piece) => new(piece, false, 0, 0, 0, 0, 0, 0, null);

        /// <summary>
        /// True on a terminal update whose own stream has ALREADY told the user, in the
        /// answer, that generation was stopped for repeating itself.
        ///
        /// <para>
        /// The skills loop says it with the repeated text quoted, which is the better
        /// sentence; an adapter that adds its own plain note on top prints the same fact
        /// twice. Set by the producer that wrote the note, read by every UI that would
        /// otherwise add one.
        /// </para>
        /// </summary>
        public bool RepetitionExplained { get; init; }

        /// <summary>
        /// The token ids this round actually generated. Set on the TERMINAL update only,
        /// and null everywhere else.
        ///
        /// <para>
        /// It exists for the skills/code tool loop, which runs several generations inside
        /// one request and re-renders the transcript before each. Re-rendering an assistant
        /// round from its PARSED pieces does not reproduce the tokens that were generated:
        /// the turn header, the channel markers and the tool-call markup are re-derived by
        /// the chat template, and the render diverges from the live KV cache at exactly the
        /// point that round began. The engine rewinds only a handful of trailing tokens, so
        /// every round after the first re-prefilled the whole conversation - measured on
        /// gemma-4-12B at 0% reuse and ~7s to first token per round, against 99.9% and
        /// ~0.3s on the one round where the render happened to line up.
        /// </para>
        /// <para>
        /// <c>SkillAgentLoop</c> - the CLI's copy of the same loop - has always recorded
        /// this; the server's copy had no way to, because the terminal update did not carry
        /// it. Two copies of one algorithm is exactly the shape that lets one of them
        /// quietly lose a property the other has.
        /// </para>
        /// </summary>
        public IReadOnlyList<int> RawOutputTokens { get; init; }

        /// <summary>
        /// Exact whitespace at the end of the prompt that preceded
        /// <see cref="RawOutputTokens"/>. The skills loop and tracked session history
        /// retain it so later renders can reproduce each raw-token boundary exactly.
        /// Empty is a valid, known boundary; null is reserved for legacy updates.
        /// </summary>
        public string? RawPromptTrailingWhitespace { get; init; }

        /// <summary>
        /// What the generation prompt ended with when <see cref="RawOutputTokens"/> were
        /// produced (see <see cref="ChatMessage.RawGenerationSuffix"/>). The skills
        /// loops rebuild each tool round from this update, so without it a round's
        /// framing would be replayed as whatever the NEXT request's mode is.
        /// An open Gemma thought channel is also sent in an empty non-terminal
        /// update before generation, so streaming parsers know its initial state.
        /// </summary>
        public string? RawGenerationSuffix { get; init; }

        /// <summary>
        /// Reasoning text decoded since the last update, already separated from
        /// <see cref="Piece"/>. Only meaningful when <see cref="IsParsed"/> is true.
        /// </summary>
        public string ThinkingPiece { get; init; }

        /// <summary>
        /// Tool calls the CALLER must service, already extracted. Only meaningful when
        /// <see cref="IsParsed"/> is true. Skill tools never appear here — those are
        /// answered in process and never reach a client.
        /// </summary>
        public IReadOnlyList<ToolCall> ParsedToolCalls { get; init; }

        /// <summary>
        /// True when this update has ALREADY been through an output parser, so
        /// <see cref="Piece"/> holds content only, <see cref="ThinkingPiece"/> holds
        /// reasoning, and <see cref="ParsedToolCalls"/> holds whatever the caller must
        /// service. An adapter that sees this must NOT run its own parser over the
        /// update.
        ///
        /// <para>
        /// Only the Agent Skills path sets it. That path has to parse anyway — it is
        /// looking for <c>skills_read</c> calls to answer itself — and once it has, the
        /// tool markup must not be forwarded, because the adapter's own parser would
        /// turn it back into a tool call the client cannot service. Handing over the
        /// already-separated pieces is what lets a skills request stream token by token
        /// instead of buffering the whole answer to check it afterwards.
        /// </para>
        /// </summary>
        public bool IsParsed { get; init; }

        /// <summary>One already-parsed delta: content, reasoning, or caller tool calls.</summary>
        public static ChatStreamUpdate Parsed(
            string content, string thinking, IReadOnlyList<ToolCall> toolCalls) =>
            new(content ?? string.Empty, false, 0, 0, 0, 0, 0, 0, null)
            {
                ThinkingPiece = thinking,
                ParsedToolCalls = toolCalls,
                IsParsed = true,
            };

        /// <summary>
        /// Which stage of an in-process tool call this update reports: <c>writing</c>
        /// while the model is generating the call, <c>running</c> while the host
        /// executes it, <c>finished</c> when execution returned. Null on every other
        /// update. Carried so a UI can show live progress through the two long silent
        /// stretches — a shell call can be a whole heredoc, and executing it can take
        /// minutes — where previously nothing streamed at all.
        /// </summary>
        public string ToolProgressPhase { get; init; }

        /// <summary>The tool being written or run, when known ("shell").</summary>
        public string ToolProgressName { get; init; }

        /// <summary>New tool-call body text (the <c>writing</c> phase), or null.</summary>
        public string ToolProgressPiece { get; init; }

        /// <summary>Seconds the execution has been running (the <c>running</c> and
        /// <c>finished</c> phases).</summary>
        public double ToolProgressSeconds { get; init; }

        /// <summary>
        /// One human-readable line saying WHAT is being run — "python · 2.1 KB code",
        /// "scripts/extract.py 2400" — so the user watching the progress knows more
        /// than the tool's name. Null when there is nothing beyond the name to say.
        /// </summary>
        public string ToolProgressDetail { get; init; }

        /// <summary>A tool-progress event. Piece stays empty and IsParsed is set, so an
        /// adapter that predates the field treats it as a no-op update.</summary>
        public static ChatStreamUpdate ToolProgress(
            string phase, string name, string piece = null, double seconds = 0, string detail = null) =>
            new(string.Empty, false, 0, 0, 0, 0, 0, 0, null)
            {
                IsParsed = true,
                ToolProgressPhase = phase,
                ToolProgressName = name,
                ToolProgressPiece = piece,
                ToolProgressSeconds = seconds,
                ToolProgressDetail = detail,
            };
    }

    /// <summary>
    /// State one client turn carries across the generations it runs: the skills/code
    /// tool loop calls the pipeline once per round, and every round must run in the cache
    /// scope the first one resolved (on a shared session that scope is proved from the
    /// history, and round two's history ends in the loop's own rounds).
    /// </summary>
    internal sealed class ChatTurnContext
    {
        /// <summary>The engine cache scope (<see cref="SequenceState.CacheScope"/>), set by
        /// the first generation of the turn.</summary>
        public string CacheScope { get; set; }
    }

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
            await foreach (var update in
                ChatStreamWithMetricsAsync(session, history, maxTokens, cancellationToken, samplingConfig, tools, enableThinking))
            {
                if (!string.IsNullOrEmpty(update.Piece))
                    yield return update.Piece;
            }
        }

        public async IAsyncEnumerable<ChatStreamUpdate>
            ChatStreamWithMetricsAsync(
                ChatSession session,
                List<ChatMessage> history,
                int maxTokens,
                [EnumeratorCancellation] CancellationToken cancellationToken,
                SamplingConfig samplingConfig = null,
                List<ToolFunction> tools = null,
                bool enableThinking = false,
                ChatTurnContext turnContext = null)
        {
            session ??= new ChatSession("__svc_intrinsic__", sharedAcrossConversations: true);
            var model = _lifecycle.Model
                ?? throw new InvalidOperationException("No model is loaded.");

            // Validate the original history before compaction or media preparation
            // can remove an attachment and accidentally turn it into text-only input.
            string audioError = UnsupportedAudioInputError(model.Config.Architecture, history,
                AudioInputSupport.IsAudioEncoderLoaded(model));
            if (audioError != null)
                throw new InvalidOperationException(audioError);

            // DiffusionGemma does not use the autoregressive continuous-batching engine; it generates a
            // whole block via iterative denoising. Drive it here and surface only the final answer to the
            // append-only protocols (OpenAI/Ollama/non-streaming). The Web UI uses DiffusionChatStreamAsync
            // directly for a live denoising preview.
            if (model is DiffusionGemmaModel)
            {
                await foreach (var u in DiffusionChatStreamAsync(session, history, maxTokens, cancellationToken, enableThinking)
                    .ConfigureAwait(false))
                {
                    if (u.Done)
                    {
                        // Denoising has no token budget to exhaust: the sampler runs its
                        // planned blocks and stops, so the only two outcomes are a natural
                        // finish and a client abort.
                        yield return new ChatStreamUpdate("", true, u.PromptTokens, u.EvalTokens, 0,
                            u.TotalNs, 0, u.TotalNs,
                            cancellationToken.IsCancellationRequested ? "cancelled" : "stop");
                    }
                    else if (!u.IsPreview && (u.Text.Length > 0 || !string.IsNullOrEmpty(u.Thinking)))
                    {
                        // The denoised text has already been through the family's output
                        // parser (channel markers and the thought block are gone), so it
                        // is handed over pre-separated; an adapter must not parse it again.
                        yield return ChatStreamUpdate.Parsed(u.Text, u.Thinking, null);
                    }
                }
                yield break;
            }

            var engine = _engineHost.TryGetEngine()
                ?? throw new InvalidOperationException(
                    "Continuous-batching engine is unavailable for this model " +
                    "(the model supports neither IBatchedPagedModel.ForwardBatch " +
                    "nor IModelArchitecture.SupportsKVStateSnapshot).");
            var enginePoolStats = engine.PoolStats;
            long engineCapacityLong = (long)enginePoolStats.totalBlocks * enginePoolStats.blockSize;
            int engineContextLimit = (int)Math.Min(int.MaxValue, engineCapacityLong);

            string arch = model.Config.Architecture;
            // A prompt fact carried on the sampling config (see SamplingConfig.ReasoningEffort).
            string reasoningEffort = samplingConfig?.ReasoningEffort;
            var preparedHistory = ChatHistoryPreparer.PrepareHistoryForInference(history, arch, _logger);
            TranscriptAugmentation augmentation;
            lock (session.HistoryLock)
                augmentation = session.Transcripts.Augment(preparedHistory);
            List<ChatMessage> renderHistory = augmentation.History;
            // Which conversation's cached state this request may continue past the
            // public prefix: the session's own (Web UI chat), or the one the history
            // just proved it continues (stateless APIs), or a fresh one.
            turnContext ??= new ChatTurnContext();
            turnContext.CacheScope ??= session.ResolveCacheScope(augmentation.InheritedScope);
            string cacheScope = turnContext.CacheScope;
            bool preserveAttachedDocuments = HasTextFileAttachments(renderHistory);

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
            try
            {

            var promptSw = Stopwatch.StartNew();
            int effectiveMaxTokens;
            List<int> explicitBreakpoints = null;
            IReadOnlyList<PromptMediaSpan> mediaSpans = null;
            string generationPromptTrailingWhitespace;
            List<int> inputTokens = _kvCacheRenderer.RenderToTokens(
                model.Tokenizer, model.Config.ChatTemplate, renderHistory, arch,
                addGenerationPrompt: true, out explicitBreakpoints,
                out generationPromptTrailingWhitespace,
                tools: tools, enableThinking: enableThinking, reasoningEffort: reasoningEffort);

            // A raw token suffix keeps the newest tool result, but it also cuts off
            // leading system/developer instructions. Compact complete message ranges
            // before either text or multimodal preparation so both paths preserve the
            // skill/edit contract, latest user task, and newest repair round.
            int contextLimit = model.MaxContextLength;
            if (engineContextLimit > 0 && (contextLimit <= 0 || engineContextLimit < contextLimit))
                contextLimit = engineContextLimit;
            int hardPromptLimit = contextLimit > 1 ? contextLimit - 1 : 0;
            int requestedReserve = contextLimit > 1
                ? HistoryCompactionReserve(maxTokens, contextLimit)
                : 0;
            int targetPromptLimit = contextLimit > 1
                ? contextLimit - requestedReserve
                : 0;

            int CountPromptTokens(List<ChatMessage> candidate) =>
                _kvCacheRenderer.RenderToTokens(
                    model.Tokenizer, model.Config.ChatTemplate, candidate, arch,
                    addGenerationPrompt: true, tools: tools,
                    enableThinking: enableThinking, reasoningEffort: reasoningEffort).Count;

            ContextHistoryWindow window = CompactHistoryForContextBudget(
                renderHistory,
                inputTokens.Count,
                contextLimit,
                maxTokens,
                preserveAttachedDocuments,
                CountPromptTokens);
            if (window.RemovedMessages > 0)
            {
                renderHistory = window.History;
                inputTokens = _kvCacheRenderer.RenderToTokens(
                    model.Tokenizer, model.Config.ChatTemplate, renderHistory, arch,
                    addGenerationPrompt: true, out explicitBreakpoints,
                    out generationPromptTrailingWhitespace,
                    tools: tools, enableThinking: enableThinking, reasoningEffort: reasoningEffort);
                _logger.LogWarning(LogEventIds.PromptTruncated,
                    "prompt.history_compacted from {OriginalTokens} to {KeptTokens} tokens by removing {RemovedMessages} old messages (contextLimit={ContextLimit}, historyReserve={HistoryReserve} for a requested reply of {RequestedTokens}, sessionId={SessionId}); leading instructions, latest user task, and newest repair round were preserved",
                    window.OriginalPromptTokens, inputTokens.Count, window.RemovedMessages,
                    contextLimit, requestedReserve, maxTokens, session?.Id ?? "(none)");
            }

            bool hasMultimodal = RequiresMultimodalPreparation(renderHistory);
            if (hasMultimodal)
            {
                // Projectors are optional at the hosting layer so a multimodal model
                // can legitimately be loaded for text alone. An image request is not
                // legitimate in that state: sending the placeholder through ordinary
                // text inference produces a confident answer about pixels the model
                // never received. WebUiChatService rejects this before streaming;
                // this invariant protects every other caller of the shared pipeline.
                if (HasImageAttachments(renderHistory) && !model.HasVisionEncoder())
                {
                    throw new InvalidOperationException(
                        "Image input cannot be processed because the loaded model has no active vision encoder. " +
                        "Load the matching image projector and retry.");
                }

                // Multimodal prompt preparation drives the vision/audio
                // encoder, which runs many GGML ops on the backend. Take
                // the model-wide GPU compute lock so we don't race the
                // engine's worker (which is doing the same thing for
                // batched forward) - concurrent GGML on Metal/CUDA from
                // two threads aborts the process via
                // ggml_metal_synchronize. The injector keeps preparation state
                // local to each request's execution flow because another encoder
                // can enter during the cooperative yields below.
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
                    List<ChatMessage> historyBeforeMediaCompaction = renderHistory;
                    var unexpandedTokens = inputTokens;
                    int unexpandedBeforeMediaCompaction = unexpandedTokens.Count;
                    string mediaBeforeCompaction = BuildMediaFingerprint(renderHistory);
                    // ClearPreparedPromptState is safe when preparation fails
                    // before creating a bucket. Arm cleanup first so partial
                    // image/audio preparation cannot leak tensors on overflow
                    // or any other exception before engine submission.
                    injectorBucketCreated = true;
                    inputTokens = model.MultimodalInjector.ProcessPromptTokens(renderHistory, inputTokens, requestId);

                    // Media placeholders expand only after the encoder runs. If that
                    // expansion consumed the reply reserve, use the now-known overhead
                    // to remove additional complete old turns and prepare once more.
                    // This prevents the final token fallback from slicing off system
                    // instructions merely because an image added thousands of tokens.
                    int expansionOverhead = Math.Max(0, inputTokens.Count - unexpandedTokens.Count);
                    int adjustedPromptLimit = targetPromptLimit > expansionOverhead
                        ? targetPromptLimit - expansionOverhead
                        : 1;
                    if (!preserveAttachedDocuments && targetPromptLimit > 0
                        && inputTokens.Count > targetPromptLimit
                        && unexpandedTokens.Count > adjustedPromptLimit)
                    {
                        ContextHistoryWindow mediaWindow = CompactHistoryForContext(
                            renderHistory,
                            unexpandedTokens.Count,
                            adjustedPromptLimit,
                            CountPromptTokens);
                        if (mediaWindow.RemovedMessages > 0
                            && mediaWindow.FinalPromptTokens <= hardPromptLimit)
                        {
                            model.MultimodalInjector.ClearPreparedPromptState(requestId);
                            renderHistory = mediaWindow.History;
                            unexpandedTokens = _kvCacheRenderer.RenderToTokens(
                                model.Tokenizer, model.Config.ChatTemplate, renderHistory, arch,
                                addGenerationPrompt: true, out explicitBreakpoints,
                                out generationPromptTrailingWhitespace,
                                tools: tools, enableThinking: enableThinking, reasoningEffort: reasoningEffort);
                            inputTokens = model.MultimodalInjector.ProcessPromptTokens(
                                renderHistory, unexpandedTokens, requestId);

                            // If the discarded prefix itself owned an old image/audio
                            // attachment, its expansion tokens disappeared too. The
                            // first limit conservatively charged that now-absent media
                            // to every remaining text turn and may therefore have
                            // removed more history than necessary. Recompute once with
                            // the measured remaining-media overhead and recover the
                            // largest message-boundary window that does not reintroduce
                            // any removed media. This costs at most one additional media
                            // preparation, only on an already-overflowing prompt.
                            int remainingMediaOverhead = Math.Max(
                                0, inputTokens.Count - unexpandedTokens.Count);
                            string mediaAfterCompaction = BuildMediaFingerprint(renderHistory);
                            if (remainingMediaOverhead < expansionOverhead
                                && !string.Equals(
                                    mediaBeforeCompaction,
                                    mediaAfterCompaction,
                                    StringComparison.Ordinal))
                            {
                                ContextHistoryWindow recoveredWindow = RecoverHistoryAfterRemovedMedia(
                                    historyBeforeMediaCompaction,
                                    unexpandedBeforeMediaCompaction,
                                    targetPromptLimit,
                                    hardPromptLimit,
                                    remainingMediaOverhead,
                                    mediaWindow,
                                    mediaAfterCompaction,
                                    CountPromptTokens);
                                if (recoveredWindow.RemovedMessages < mediaWindow.RemovedMessages)
                                {
                                    model.MultimodalInjector.ClearPreparedPromptState(requestId);
                                    renderHistory = recoveredWindow.History;
                                    unexpandedTokens = _kvCacheRenderer.RenderToTokens(
                                        model.Tokenizer, model.Config.ChatTemplate, renderHistory, arch,
                                        addGenerationPrompt: true, out explicitBreakpoints,
                                        out generationPromptTrailingWhitespace,
                                        tools: tools, enableThinking: enableThinking, reasoningEffort: reasoningEffort);
                                    inputTokens = model.MultimodalInjector.ProcessPromptTokens(
                                        renderHistory, unexpandedTokens, requestId);
                                    mediaWindow = recoveredWindow;
                                    remainingMediaOverhead = Math.Max(
                                        0, inputTokens.Count - unexpandedTokens.Count);
                                }
                            }
                            _logger.LogWarning(LogEventIds.PromptTruncated,
                                "prompt.multimodal_history_compacted from {OriginalTokens} to {KeptTokens} unexpanded tokens by removing {RemovedMessages} old messages after accounting for {MediaTokens} media-expansion tokens (contextLimit={ContextLimit}, sessionId={SessionId})",
                                mediaWindow.OriginalPromptTokens,
                                unexpandedTokens.Count,
                                mediaWindow.RemovedMessages,
                                remainingMediaOverhead,
                                contextLimit,
                                session?.Id ?? "(none)");
                        }
                    }

                    // ProcessPromptTokens expands each single placeholder token
                    // (<|image_pad|>, the audio equivalent) into the encoded
                    // media span. Markers in the byte-identical token prefix are
                    // still exact; later offsets cannot be mapped safely and are
                    // removed while retaining explicit-cache mode.
                    RetainCacheBreakpointsInUnchangedPrefix(
                        unexpandedTokens, inputTokens, explicitBreakpoints);
                    inputTokens = TruncatePromptToContext(
                        session, inputTokens, maxTokens, out effectiveMaxTokens, requestId,
                        preserveAllInput: true,
                        executionContextLimit: engineContextLimit,
                        explicitBreakpoints: explicitBreakpoints,
                        preservedInputKind: preserveAttachedDocuments ? "document and media input" : "media input");

                    // Where each image/audio span landed and what it is, after any trim:
                    // the engine compares these positionally when it reuses a prefix.
                    mediaSpans = model.MultimodalInjector.GetPreparedMediaSpans(requestId);
                }
            }
            else
            {
                inputTokens = TruncatePromptToContext(
                    session, inputTokens, maxTokens, out effectiveMaxTokens, null,
                    preserveAllInput: preserveAttachedDocuments,
                    executionContextLimit: engineContextLimit, explicitBreakpoints: explicitBreakpoints);
            }

            int promptTokenCount = inputTokens.Count;
            var cfg = samplingConfig ?? SamplingConfig.Default;
            int thinkingBudget = ThinkingBudgetFor(effectiveMaxTokens, enableThinking);
            // With thinking off only a family whose model opens its channel itself is
            // given a (small) cap; WithThinkingBudget ignores it for every other family.
            int channelBudget = enableThinking ? thinkingBudget : UnrequestedThinkingBudgetFor(effectiveMaxTokens);
            cfg = WithThinkingBudget(cfg, model.Tokenizer, arch, channelBudget, out bool samplingEndsThinking,
                enableThinking, inputTokens);

            // Where the prompt every conversation on this host shares ends, so the
            // engine can checkpoint its state there once and start the next new chat
            // from a copy (see SequenceState.SharedPrefixTokens).
            int sharedPrefixTokens = ComputeSharedPrefixTokens(
                model, renderHistory, inputTokens, arch, tools, enableThinking, reasoningEffort);

            var seq = new SequenceState(
                requestId: requestId,
                promptTokens: inputTokens,
                maxNewTokens: effectiveMaxTokens,
                blockSize: enginePoolStats.blockSize,
                samplingConfig: cfg,
                userTag: session,
                mediaSpans: mediaSpans,
                cacheBreakpoints: explicitBreakpoints,
                sharedPrefixTokens: sharedPrefixTokens,
                cacheScope: cacheScope);

            promptSw.Stop();
            long promptNs = InferenceTelemetry.ToNanos(promptSw.ElapsedTicks);

            string recordedSuffix = RecordedGenerationSuffix(model.Tokenizer, inputTokens, arch, enableThinking);
            if (SignalsOpenThoughtChannel(arch, recordedSuffix))
                yield return ChatStreamUpdate.Text(string.Empty) with { RawGenerationSuffix = recordedSuffix };

            var evalSw = Stopwatch.StartNew();
            var handle = engine.SubmitRequest(seq, cancellationToken);
            var generatedTokens = new List<int>();
            var rawBytes = new List<byte>();
            int prevValidLen = 0;
            // Stop-sequence matching needs the full decoded text; only the rare
            // request that configures string stop sequences pays for accumulating
            // it. The common path decodes just the newly-completed bytes per token
            // (below) instead of re-decoding the whole buffer every step (O(n^2)).
            bool hasStopSequences = cfg.StopSequences != null && cfg.StopSequences.Count > 0;
            StringBuilder decodedForStops = hasStopSequences ? new StringBuilder() : null;
            TokenSampler stopSampler = hasStopSequences ? new TokenSampler(cfg) : null;
            string finishReason = "max_tokens";

            // The thinking budget. A reasoning model can spend an ENTIRE token
            // allowance inside its thinking channel and emit no answer at all: observed
            // on the algorithmic-art skill, where 8000 tokens — 100% of them thinking —
            // produced an empty response after 888 seconds, reported to the caller as a
            // bare `truncated: true` with nothing to read. Capping thinking turns that
            // silent write-off into a fast, explained stop. Families declaring a
            // trained budget-end token close reasoning in the sampler instead,
            // leaving the remaining allowance available for an answer.
            //
            // Detected from the decoded text rather than the parser, because the parser
            // runs a layer above this loop: while thinking is open the close marker has
            // not appeared, and every reasoning family this host serves closes with
            // </think>. A family that does not is simply never capped, which is the
            // safe direction to be wrong in.
            StringBuilder thinkingScan = thinkingBudget > 0 && !samplingEndsThinking ? new StringBuilder() : null;
            bool thinkingClosed = false;
            int thinkingTokens = 0;
            bool wasCancelled = false;
            int kvCacheReusedTokens = 0;
            long timeToFirstTokenMs = 0;
            bool firstTokenSampled = false;
            var totalSw = Stopwatch.StartNew();

            // Stream tokens off the engine handle, doing UTF-8-valid piece
            // accumulation and stop-sequence detection in this layer.
            //
            // A Stop (the token cancelling while this awaits the next token) used to
            // leave this method through the exception, past the transcript update
            // below - so the stopped turn's raw tokens were never recorded, the next
            // turn re-rendered the half answer from text, and the live cache (which
            // holds every token the engine forwarded, a step or two past what was
            // streamed) no longer matched it. On a model that can rewind a few
            // tokens that mostly went unnoticed; on one that cannot (Qwen 3.5) every
            // Stop cost a full re-prefill on the following turn. The cancellation is
            // observed here instead, the engine's own token list is taken as the
            // record, and the exception is re-raised after the transcript is written.
            // Read WITHOUT the token: the handle registered it at submission, so a
            // Stop aborts the request in the engine, which completes this channel and
            // ends the loop on its own - no exception, and nothing yielded inside a
            // try/catch (which an iterator cannot do).
            OperationCanceledException stopped = null;
            await foreach (var nextToken in handle.Tokens.ReadAllAsync().ConfigureAwait(false))
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
                // Decode only the bytes that completed a UTF-8 boundary since the
                // last token. The prior valid prefix already ended on a character
                // boundary, so this byte slice yields exactly the new characters —
                // identical to substring-ing a full re-decode, but O(new bytes)
                // rather than O(total bytes) per token (and no whole-buffer copy).
                string piece = "";
                if (validLen > prevValidLen)
                {
                    ReadOnlySpan<byte> newBytes = CollectionsMarshal.AsSpan(rawBytes)
                        .Slice(prevValidLen, validLen - prevValidLen);
                    piece = Encoding.UTF8.GetString(newBytes);
                    prevValidLen = validLen;
                }

                if (!firstTokenSampled)
                {
                    firstTokenSampled = true;
                    timeToFirstTokenMs = (long)totalSw.Elapsed.TotalMilliseconds;
                }

                bool stopRequested = false;
                if (hasStopSequences)
                {
                    if (piece.Length > 0)
                        decodedForStops.Append(piece);
                    var (_, shouldStop) = stopSampler.CheckStopSequences(decodedForStops.ToString());
                    if (shouldStop)
                    {
                        stopRequested = true;
                        finishReason = "stop_sequence";
                    }
                }

                if (thinkingScan != null && !thinkingClosed)
                {
                    if (piece.Length > 0)
                        thinkingScan.Append(piece);
                    thinkingTokens++;

                    // Only the tail can contain the marker, and the marker is short.
                    if (thinkingScan.Length > 64)
                        thinkingScan.Remove(0, thinkingScan.Length - 64);

                    if (thinkingScan.ToString().Contains("</think>", StringComparison.Ordinal))
                    {
                        thinkingClosed = true;
                    }
                    else if (thinkingTokens >= thinkingBudget)
                    {
                        // Stop now rather than at the full budget: the answer would be
                        // empty either way, and this way it costs a fraction of the time
                        // and says what happened.
                        stopRequested = true;
                        finishReason = "thinking_budget";
                    }
                }

                if (piece.Length > 0)
                    yield return ChatStreamUpdate.Text(piece);

                if (stopRequested)
                {
                    engine.Abort(seq.RequestId);
                    break;
                }
            }
            if (cancellationToken.IsCancellationRequested && finishReason != "stop_sequence" && finishReason != "thinking_budget")
            {
                wasCancelled = true;
                finishReason = "cancelled";
                stopped = new OperationCanceledException(cancellationToken);
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

            if (wasCancelled)
            {
                // The engine may have forwarded a step or two past the last token that
                // was streamed before the abort landed. Those tokens are in the live
                // cache, so the transcript records them too; otherwise the next
                // render diverges from the cache at the end of this answer. The user
                // never saw their text and the streamed answer stays as it was.
                IReadOnlyList<int> forwarded = seq.OutputTokens;
                for (int i = generatedTokens.Count; i < forwarded.Count; i++)
                    generatedTokens.Add(forwarded[i]);
            }

            string assistantText = Encoding.UTF8.GetString(rawBytes.ToArray());
            evalSw.Stop();
            totalSw.Stop();

            // Record this turn for the next request of the same conversation: the raw
            // tokens the cache holds, and what the client was sent for them - the next
            // request's assistant message must match that to get the tokens back.
            RecordGeneratedTurn(session, preparedHistory, renderHistory, cacheScope,
                new ChatMessage
                {
                    Role = "assistant",
                    Content = assistantText,
                    RawOutputTokens = generatedTokens,
                    RawPromptTrailingWhitespace = generationPromptTrailingWhitespace,
                    RawGenerationSuffix = recordedSuffix,
                },
                BuildEmittedTurn(arch, assistantText, enableThinking, tools,
                    // Parsers are primed with the prompt's open channel exactly when this
                    // pipeline announced it (above); mirror that, or the recorded content
                    // would differ from what the adapters parsed.
                    SignalsOpenThoughtChannel(arch, recordedSuffix) ? recordedSuffix : null,
                    wasCancelled));

            if (stopped != null)
            {
                // Exactly the exception the caller has always seen for a stopped turn,
                // now raised AFTER the transcript knows what the cache holds.
                _telemetry.LogChatFinished(
                    true, generatedTokens.Count, promptTokenCount, kvCacheReusedTokens,
                    promptTokenCount > 0 ? 100.0 * kvCacheReusedTokens / promptTokenCount : 0.0,
                    timeToFirstTokenMs, totalSw.Elapsed.TotalMilliseconds, 0, finishReason, assistantText);
                System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(stopped).Throw();
            }

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
            yield return new ChatStreamUpdate("", true, promptTokenCount, generatedTokens.Count,
                                             kvCacheReusedTokens, totalNs, promptNs, evalNs, finishReason)
            {
                // Carried so the skills loop can splice this round back verbatim on its
                // next render instead of re-tokenizing it. See RawOutputTokens.
                RawOutputTokens = generatedTokens,
                RawPromptTrailingWhitespace = generationPromptTrailingWhitespace,
                RawGenerationSuffix = recordedSuffix,
            };
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
        /// <param name="enableThinking">Whether the caller wants the thought block. DiffusionGemma
        /// writes Gemma 4's channel syntax and its raw canvas used to be delivered verbatim, so every
        /// OpenAI answer opened with the literal <c>&lt;|channel&gt;thought</c> marker; the family's
        /// output parser now separates the channels, and the thought is dropped unless asked for.</param>
        public async IAsyncEnumerable<DiffusionStreamUpdate> DiffusionChatStreamAsync(
            ChatSession session,
            List<ChatMessage> history,
            int maxTokens,
            [EnumeratorCancellation] CancellationToken cancellationToken,
            bool enableThinking = false)
        {
            session ??= new ChatSession("__svc_intrinsic__", sharedAcrossConversations: true);
            var model = (DiffusionGemmaModel)(_lifecycle.Model
                ?? throw new InvalidOperationException("No model is loaded."));
            string arch = model.Config.Architecture;

            var preparedHistory = ChatHistoryPreparer.PrepareHistoryForInference(history, arch, _logger);
            // Read under the session lock so a parallel request's record can't race it.
            List<ChatMessage> renderHistory;
            lock (session.HistoryLock)
                renderHistory = session.Transcripts.Augment(preparedHistory).History;
            bool preserveAttachedDocuments = HasTextFileAttachments(renderHistory);

            using var chatScope = _telemetry.BeginInferenceScope(
                session, _lifecycle.LoadedModelName, _lifecycle.LoadedBackend, "diffusion.chat.stream");

            var promptSw = Stopwatch.StartNew();
            List<int> inputTokens = _kvCacheRenderer.RenderToTokens(
                model.Tokenizer, model.Config.ChatTemplate, renderHistory, arch,
                addGenerationPrompt: true, out _,
                out string generationPromptTrailingWhitespace,
                tools: null, enableThinking: false);
            inputTokens = TruncatePromptToContext(
                session, inputTokens, maxTokens, out _, preserveAllInput: preserveAttachedDocuments);
            int promptTokenCount = inputTokens.Count;
            // The publisher template may leave a thought channel open at the end of the
            // prompt; the parser then has to start inside it, exactly as it does for
            // Gemma 4's autoregressive turns.
            string generationSuffix = RecordedGenerationSuffix(model.Tokenizer, inputTokens, arch, enableThinking: false);
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
                var (previewContent, previewThinking) =
                    SeparateDiffusionChannels(arch, previewText, enableThinking, generationSuffix);
                yield return new DiffusionStreamUpdate(
                    previewContent, IsPreview: true, Done: false, preview.Step + 1, preview.TotalSteps, 0, 0, 0,
                    previewThinking);
            }

            var generated = await handle.Completion.ConfigureAwait(false);
            totalSw.Stop();

            generated ??= new List<int>();
            string finalText = model.Tokenizer.Decode(generated);
            // The tracked history keeps the raw text beside the raw tokens; the caller
            // gets the channels separated.
            var (finalContent, finalThinking) =
                SeparateDiffusionChannels(arch, finalText, enableThinking, generationSuffix);

            RecordGeneratedTurn(session, preparedHistory, renderHistory, cacheScope: null,
                new ChatMessage
                {
                    Role = "assistant",
                    Content = finalText,
                    RawOutputTokens = generated,
                    RawPromptTrailingWhitespace = generationPromptTrailingWhitespace,
                },
                new EmittedAssistantTurn(finalContent, null, finalThinking, finalText,
                    cancellationToken.IsCancellationRequested));

            long totalNs = InferenceTelemetry.ToNanos(totalSw.ElapsedTicks);
            _telemetry.LogChatFinished(
                cancellationToken.IsCancellationRequested, generated.Count, promptTokenCount, 0, 0.0,
                0, totalSw.Elapsed.TotalMilliseconds,
                totalSw.Elapsed.TotalSeconds > 0 ? generated.Count / totalSw.Elapsed.TotalSeconds : 0,
                cancellationToken.IsCancellationRequested ? "cancelled" : "stop", finalText);

            // Final answer (replaces the last preview), then the terminal metrics update.
            yield return new DiffusionStreamUpdate(finalContent, IsPreview: false, Done: false, 0, 0, 0, 0, 0,
                finalThinking);
            yield return new DiffusionStreamUpdate("", IsPreview: false, Done: true, 0, 0,
                promptTokenCount, generated.Count, totalNs);
        }

        /// <summary>
        /// Run the family's output parser over one whole denoised canvas. A diffusion
        /// canvas is re-decoded from scratch at every step rather than appended to, so
        /// each call gets a fresh parser primed with the prompt's open channel, if any.
        /// Returns the content and, only when reasoning was requested, the thought
        /// block (null otherwise: the adapters treat null as "no reasoning to report").
        /// </summary>
        internal static (string Content, string? Thinking) SeparateDiffusionChannels(
            string arch, string rawText, bool enableThinking, string? generationSuffix)
        {
            if (string.IsNullOrEmpty(rawText))
                return (string.Empty, null);
            IOutputParser parser = OutputParserFactory.Create(arch);
            parser.Init(enableThinking, null);
            parser.SetGenerationPromptSuffix(generationSuffix);
            ParsedOutput parsed = parser.Add(rawText, true);
            string content = parsed.Content ?? string.Empty;
            // A diffusion turn has no tool loop, so a call the model wrote anyway is
            // shown as text rather than dropped on the floor.
            if (content.Length == 0 && !string.IsNullOrEmpty(parsed.ToolCallText))
                content = parsed.ToolCallText;
            string? thinking = enableThinking && !string.IsNullOrEmpty(parsed.Thinking) ? parsed.Thinking : null;
            return (content, thinking);
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

        private static bool SignalsOpenThoughtChannel(string arch, string recordedSuffix)
            => arch == "gemma4" && recordedSuffix != null
                && recordedSuffix.EndsWith("<|channel>thought\n", StringComparison.Ordinal);

        private static void RecordGeneratedTurn(
            ChatSession session, List<ChatMessage> preparedHistory, List<ChatMessage> renderHistory,
            string cacheScope, ChatMessage generated, EmittedAssistantTurn emitted)
        {
            lock (session.HistoryLock)
                session.Transcripts.Record(preparedHistory, generated, emitted, cacheScope, renderHistory);
        }

        /// <summary>
        /// What a client received for <paramref name="rawText"/>: the family's output
        /// parser run over it the way the protocol adapters and the Web UI run it
        /// (thinking mode, tools, the prompt's open channel), plus the raw text for a
        /// client that runs no parser. A parser failure leaves the raw text as content.
        /// </summary>
        internal static EmittedAssistantTurn BuildEmittedTurn(
            string arch, string rawText, bool enableThinking, List<ToolFunction> tools,
            string generationSuffix, bool cancelled)
        {
            rawText ??= string.Empty;
            try
            {
                IOutputParser parser = OutputParserFactory.Create(arch);
                parser.Init(enableThinking, tools);
                parser.SetGenerationPromptSuffix(generationSuffix);
                ParsedOutput parsed = parser.Add(rawText, true);
                return new EmittedAssistantTurn(parsed.Content ?? string.Empty, parsed.ToolCalls,
                    parsed.Thinking, rawText, cancelled);
            }
            catch (Exception)
            {
                return new EmittedAssistantTurn(rawText, null, null, rawText, cancelled);
            }
        }

        /// <summary>Below this many shared tokens a checkpoint is not worth its copy.</summary>
        internal const int MinSharedPrefixTokens = 64;

        // The rendered token run of the last few distinct shared prefixes, so a turn
        // does not re-tokenize thousands of tokens of system prompt to find where its
        // own prompt stops sharing them. Keyed by everything the render depends on.
        private readonly object _sharedPrefixLock = new();
        private readonly Dictionary<string, List<int>> _sharedPrefixRenders = new(StringComparer.Ordinal);
        private readonly Queue<string> _sharedPrefixOrder = new();

        /// <summary>
        /// How many leading tokens of <paramref name="promptTokens"/> are the prefix
        /// every conversation shares: the leading system/developer messages plus the
        /// tool declarations, rendered on their own and matched against the prompt.
        /// Zero when there is no such prefix, when it is too short to be worth a
        /// checkpoint, or when anything about working it out fails — the engine then
        /// simply takes no checkpoint, which is what it did before this existed.
        /// </summary>
        internal int ComputeSharedPrefixTokens(
            ModelBase model, List<ChatMessage> history, List<int> promptTokens,
            string arch, List<ToolFunction> tools, bool enableThinking, string reasoningEffort = null)
        {
            try
            {
                if (model?.Tokenizer == null || history == null || promptTokens == null)
                    return 0;
                // Only a family that renders the level has a prefix that depends on it.
                if (ChatProtocolRegistry.For(arch)?.RendersReasoningEffort != true)
                    reasoningEffort = null;
                int leading = 0;
                while (leading < history.Count
                    && (history[leading].Role == "system" || history[leading].Role == "developer"))
                    leading++;
                bool hasTools = tools is { Count: > 0 };
                if (leading == 0 && !hasTools)
                    return 0;
                // Media in the leading messages would make the prefix depend on the
                // attachment, which the checkpoint deliberately ignores.
                for (int i = 0; i < leading; i++)
                    if (ChatHistoryPreparer.HasMultimodalContent(history[i]))
                        return 0;

                var keyBuilder = new StringBuilder();
                keyBuilder.Append(arch).Append('|').Append(enableThinking ? 'T' : 'F').Append('|')
                    .Append(reasoningEffort ?? string.Empty).Append('|');
                for (int i = 0; i < leading; i++)
                    keyBuilder.Append(history[i].Role).Append(':').Append(history[i].Content).Append('\u0001');
                if (hasTools)
                    foreach (var t in tools)
                        keyBuilder.Append(t.Name).Append(':').Append(t.Description).Append(':')
                            .Append(t.Parameters?.Count ?? 0).Append('\u0001');
                string key = keyBuilder.ToString();

                List<int> prefixTokens;
                lock (_sharedPrefixLock)
                    _sharedPrefixRenders.TryGetValue(key, out prefixTokens);
                if (prefixTokens == null)
                {
                    prefixTokens = _kvCacheRenderer.RenderToTokens(
                        model.Tokenizer, model.Config.ChatTemplate, history.GetRange(0, leading), arch,
                        addGenerationPrompt: false, tools: tools, enableThinking: enableThinking, reasoningEffort: reasoningEffort);
                    lock (_sharedPrefixLock)
                    {
                        if (_sharedPrefixRenders.TryAdd(key, prefixTokens))
                        {
                            _sharedPrefixOrder.Enqueue(key);
                            while (_sharedPrefixOrder.Count > 8)
                                _sharedPrefixRenders.Remove(_sharedPrefixOrder.Dequeue());
                        }
                    }
                }

                int lcp = 0;
                int limit = Math.Min(prefixTokens.Count, promptTokens.Count - 1);
                while (lcp < limit && prefixTokens[lcp] == promptTokens[lcp])
                    lcp++;
                return lcp >= MinSharedPrefixTokens ? lcp : 0;
            }
            catch (Exception ex)
            {
                _logger.LogDebug(ex, "shared prefix could not be measured; no checkpoint for this turn");
                return 0;
            }
        }

        /// <summary>
        /// The framing this turn's generation prompt actually ended with, for the
        /// transcript to remember beside the raw tokens.
        ///
        /// <para>
        /// The family declares one suffix per thinking mode, but not every prompt ends
        /// on it: a Gemma 4 round that continues after a tool result ends on the
        /// result's closing marker and the model writes straight on, with no empty
        /// thought block in between. Recording the family's suffix for that round put a
        /// block into the re-render that the cache never held, and the next turn
        /// re-prefilled the conversation from that point. So what is recorded is what
        /// the rendered prompt's tail shows — the suffix when it is there, an explicit
        /// "nothing" when it is not — and the renderer trusts that over its default.
        /// </para>
        /// </summary>
        internal static string RecordedGenerationSuffix(
            ITokenizer tokenizer, List<int> promptTokens, string arch, bool enableThinking)
        {
            string suffix = KVCachePromptRenderer.GetAssistantGenerationSuffix(arch, enableThinking);
            // DiffusionGemma renders through the same channel-priming template family.
            bool gemma = arch == "gemma4" || arch == "diffusion-gemma" || arch == "diffusion_gemma";
            if ((!gemma && string.IsNullOrEmpty(suffix)) || promptTokens == null || promptTokens.Count == 0 || tokenizer == null)
                return string.Empty;
            try
            {
                int take = Math.Min(promptTokens.Count, 64);
                string tail = tokenizer.Decode(promptTokens.GetRange(promptTokens.Count - take, take));
                if (gemma)
                {
                    // A publisher template may prime a channel, and older tracked
                    // turns may contain our former empty-channel prefix. Preserve
                    // the actual boundary without imposing it on ordinary turns.
                    const string closedChannel = "<|channel>thought\n<channel|>";
                    const string openChannel = "<|channel>thought\n";
                    if (tail.TrimEnd().EndsWith(closedChannel, StringComparison.Ordinal))
                        return closedChannel;
                    if (tail.EndsWith(openChannel, StringComparison.Ordinal))
                        return openChannel;
                    return string.Empty;
                }
                if (tail.EndsWith(suffix, StringComparison.Ordinal))
                    return suffix;
                // Every Jinja render is TrimEnd()ed, and only some families put the
                // trailing newline back (Gemma 4, Qwen 3.5). For the rest (Qwen 3,
                // Bonsai, Qwen3.8-Flash-Next) the prompt ends on `</think>` without the
                // suffix's `\n\n`. The framing is still there, so it is the framing that
                // is recorded; the exact boundary whitespace travels separately as
                // RawPromptTrailingWhitespace and is restored by the renderer.
                string trimmedSuffix = suffix.TrimEnd();
                if (trimmedSuffix.Length > 0 && tail.TrimEnd().EndsWith(trimmedSuffix, StringComparison.Ordinal))
                    return suffix;
                return string.Empty;
            }
            catch (Exception)
            {
                // A tokenizer that cannot decode a lone control token: fall back to the
                // family's declared suffix, which is what was recorded before.
                return suffix;
            }
        }

        public async IAsyncEnumerable<ChatStreamUpdate>
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
            var freshSession = new ChatSession("__generate_intrinsic__", sharedAcrossConversations: true);
            await foreach (var item in ChatStreamWithMetricsAsync(
                freshSession, oneShot, maxTokens, cancellationToken, samplingConfig))
            {
                yield return item;
            }
        }

        internal readonly record struct ContextHistoryWindow(
            List<ChatMessage> History,
            int OriginalPromptTokens,
            int FinalPromptTokens,
            int RemovedMessages);

        /// <summary>
        /// How much of the window the compactor sets aside for the reply BEFORE it
        /// starts removing history: the requested reply length, but never more than a
        /// quarter of the window (a 1,024-token floor where the window allows).
        ///
        /// <para>
        /// The reply length is a ceiling the user chose for the answer, not a claim on
        /// the conversation. Honouring it literally here meant that a limit at or above
        /// the window -- the largest rung the phone's settings offer, inside a 32k
        /// window -- left the prompt one token of room, and the compactor removed the
        /// whole conversation on every tool round but the instructions, the latest
        /// request and the newest round: observed on a phone as an agent that fetched
        /// the same page eight times because each round had forgotten the last. What
        /// the reply actually gets is decided AFTER compaction by
        /// <see cref="ClampGenerationReserve"/>: everything the kept prompt leaves,
        /// which in a short conversation is still the whole window.
        /// </para>
        /// </summary>
        internal static int HistoryCompactionReserve(int requestedGenerationTokens, int contextLimit)
        {
            if (contextLimit <= 1)
                return Math.Max(1, requestedGenerationTokens);
            int cap = Math.Max(1024, contextLimit / 4);
            return Math.Clamp(Math.Min(requestedGenerationTokens, cap), 1, contextLimit - 1);
        }

        /// <summary>
        /// Apply the same context/reply budget policy used by live generation before
        /// adopting a message-boundary compaction result. The protected minimum may
        /// exceed the preferred reply reserve, but it must still fit the hard prompt
        /// ceiling so the caller never swaps one silent truncation for another.
        /// </summary>
        internal static ContextHistoryWindow CompactHistoryForContextBudget(
            List<ChatMessage> history,
            int originalPromptTokens,
            int contextLimit,
            int requestedGenerationTokens,
            bool preserveAllInput,
            Func<List<ChatMessage>, int> countPromptTokens)
        {
            if (history == null)
                throw new ArgumentNullException(nameof(history));
            if (countPromptTokens == null)
                throw new ArgumentNullException(nameof(countPromptTokens));
            if (preserveAllInput || contextLimit <= 1)
            {
                return new ContextHistoryWindow(
                    history, originalPromptTokens, originalPromptTokens, RemovedMessages: 0);
            }

            int reserve = HistoryCompactionReserve(requestedGenerationTokens, contextLimit);
            int promptLimit = contextLimit - reserve;
            ContextHistoryWindow window = CompactHistoryForContext(
                history, originalPromptTokens, promptLimit, countPromptTokens);
            int hardPromptLimit = contextLimit - 1;
            if (window.FinalPromptTokens > hardPromptLimit)
            {
                // A raw suffix slice would make the request fit by silently deleting
                // the very policy/task/diagnostic this compactor promises to protect.
                // Make the overflow actionable instead. Callers may start a fresh turn
                // or shorten the newest result, but they must never generate under a
                // context whose higher-priority instructions were cut away.
                throw new PromptContextOverflowException(
                    $"The protected prompt requires {window.FinalPromptTokens} tokens after old conversation "
                    + $"turns are removed, which exceeds the model's {contextLimit}-token context window. "
                    + "Start a new chat or shorten the latest request/tool result; it cannot be safely truncated.");
            }

            return window.RemovedMessages > 0
                ? window
                : new ContextHistoryWindow(
                    history, originalPromptTokens, originalPromptTokens, RemovedMessages: 0);
        }

        /// <summary>
        /// Remove complete old conversation/tool rounds until a rendered prompt fits,
        /// without mutating the caller's list. Every system/developer message, the latest
        /// real user task, and newest assistant/tool repair round are never candidates.
        /// Token counts come from the real renderer supplied by the caller, so chat-template
        /// framing, tool declarations and cached raw output are included.
        /// </summary>
        internal static ContextHistoryWindow CompactHistoryForContext(
            List<ChatMessage> history,
            int originalPromptTokens,
            int promptTokenLimit,
            Func<List<ChatMessage>, int> countPromptTokens)
        {
            if (history == null)
                throw new ArgumentNullException(nameof(history));
            if (countPromptTokens == null)
                throw new ArgumentNullException(nameof(countPromptTokens));
            if (promptTokenLimit <= 0 || originalPromptTokens <= promptTokenLimit || history.Count < 2)
            {
                return new ContextHistoryWindow(
                    history, originalPromptTokens, originalPromptTokens, RemovedMessages: 0);
            }

            int instructionCount = 0;
            while (instructionCount < history.Count && IsInstructionRole(history[instructionCount]?.Role))
                instructionCount++;

            // SkillChatLoop uses role=user for tool results on families whose templates
            // cannot render role=tool. Those messages have a stable host-authored prefix;
            // skip them when locating the actual request that the repair still answers.
            int latestUser = -1;
            for (int i = history.Count - 1; i >= instructionCount; i--)
            {
                if (IsGenuineUserMessage(history, i))
                {
                    latestUser = i;
                    break;
                }
            }

            // With no ordinary user message, retaining the newest message is the safest
            // analogue: it may be a one-shot prompt or a host-authored continuation.
            int anchor = latestUser >= 0 ? latestUser : history.Count - 1;
            var removable = new List<(int Start, int End)>();

            // Completed turns before the latest request. A turn begins at a real user
            // message and includes its assistant/tool traffic up to the next request.
            int start = instructionCount;
            while (start < anchor)
            {
                int end = start + 1;
                while (end < anchor &&
                       !IsGenuineUserMessage(history, end))
                {
                    end++;
                }
                removable.Add((start, end));
                start = end;
            }

            // Within the active request, each assistant generation plus the tool results
            // it requested is one repair round. Retain the newest round (normally the
            // failing command and its diagnostic), and discard older rounds first.
            var activeRounds = new List<(int Start, int End)>();
            start = anchor + 1;
            if (start < history.Count)
            {
                int roundStart = start;
                for (int i = start + 1; i < history.Count; i++)
                {
                    if (IsAssistantRole(history[i]?.Role))
                    {
                        activeRounds.Add((roundStart, i));
                        roundStart = i;
                    }
                }
                activeRounds.Add((roundStart, history.Count));
            }
            for (int i = 0; i + 1 < activeRounds.Count; i++)
                removable.Add(activeRounds[i]);

            if (removable.Count == 0)
            {
                return new ContextHistoryWindow(
                    history, originalPromptTokens, originalPromptTokens, RemovedMessages: 0);
            }

            List<ChatMessage> BuildCandidate(int rangeCount, out int removedMessages)
            {
                var removed = new bool[history.Count];
                removedMessages = 0;
                for (int range = 0; range < rangeCount; range++)
                {
                    (int rangeStart, int rangeEnd) = removable[range];
                    for (int i = rangeStart; i < rangeEnd; i++)
                    {
                        // System and developer instructions are authoritative wherever
                        // they occur. APIs may interleave a policy update between user
                        // turns; treating only the leading block as protected silently
                        // weakens that policy exactly when the context is under stress.
                        if (!removed[i] && !IsInstructionRole(history[i]?.Role))
                        {
                            removed[i] = true;
                            removedMessages++;
                        }
                    }
                }

                var candidate = new List<ChatMessage>(history.Count - removedMessages);
                for (int i = 0; i < history.Count; i++)
                {
                    if (!removed[i])
                        candidate.Add(history[i]);
                }
                return candidate;
            }

            // Rendering and tokenizing the whole prompt once per old turn makes this
            // path quadratic for long chat histories. Removing additional complete
            // ranges cannot add prompt content, so find the smallest fitting prefix of
            // ranges with a bounded binary search. The all-ranges measurement also
            // covers the protected minimum that may still exceed the preferred reserve.
            int maximumRanges = removable.Count;
            List<ChatMessage> maximum = BuildCandidate(maximumRanges, out int maximumRemoved);
            int maximumTokens = countPromptTokens(maximum);
            if (maximumTokens > promptTokenLimit || maximumRanges == 1)
            {
                return new ContextHistoryWindow(
                    maximum, originalPromptTokens, maximumTokens, maximumRemoved);
            }

            int low = 1;
            int high = maximumRanges - 1;
            List<ChatMessage> compacted = maximum;
            int finalTokens = maximumTokens;
            int removedMessages = maximumRemoved;
            while (low <= high)
            {
                int middle = low + ((high - low) / 2);
                List<ChatMessage> candidate = BuildCandidate(middle, out int candidateRemoved);
                int candidateTokens = countPromptTokens(candidate);
                if (candidateTokens <= promptTokenLimit)
                {
                    compacted = candidate;
                    finalTokens = candidateTokens;
                    removedMessages = candidateRemoved;
                    high = middle - 1;
                }
                else
                {
                    low = middle + 1;
                }
            }

            return new ContextHistoryWindow(
                compacted, originalPromptTokens, finalTokens, removedMessages);
        }

        private static bool IsInstructionRole(string role) =>
            string.Equals(role, "system", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(role, "developer", StringComparison.OrdinalIgnoreCase);

        private static bool IsUserRole(string role) =>
            string.Equals(role, "user", StringComparison.OrdinalIgnoreCase);

        private static bool IsAssistantRole(string role) =>
            string.Equals(role, "assistant", StringComparison.OrdinalIgnoreCase);

        private static bool IsGenuineUserMessage(IReadOnlyList<ChatMessage> history, int index) =>
            history != null
            && index >= 0
            && index < history.Count
            && IsUserRole(history[index]?.Role)
            && history[index] is not Skills.SkillChatLoop.HostCompletionCorrectionMessage
            && !IsSyntheticToolResult(history, index);

        private static bool IsSyntheticToolResult(IReadOnlyList<ChatMessage> history, int index)
        {
            if (history == null || index < 0 || index >= history.Count ||
                !HasSyntheticToolResultPrefix(history[index]?.Content))
                return false;

            // Some model templates cannot render role=tool, so SkillChatLoop frames
            // host results as user messages. A prefix alone is not structural evidence:
            // a real user is allowed to begin a request with the same words. Real loop
            // results follow an assistant message that contains the parsed calls; skip
            // sibling result messages from the same call batch while finding it.
            for (int previous = index - 1; previous >= 0; previous--)
            {
                ChatMessage message = history[previous];
                if (IsUserRole(message?.Role) && HasSyntheticToolResultPrefix(message?.Content))
                    continue;

                return IsAssistantRole(message?.Role) && message.ToolCalls is { Count: > 0 };
            }

            return false;
        }

        private static bool HasSyntheticToolResultPrefix(string content)
        {
            if (string.IsNullOrEmpty(content))
                return false;
            return content.StartsWith(
                       "Result of the skill lookup you requested:", StringComparison.Ordinal) ||
                   (content.StartsWith("Result of your ", StringComparison.Ordinal) &&
                    content.IndexOf(" call:", StringComparison.Ordinal) >= 0);
        }

        /// <summary>Trim ordinary conversation history so the prompt plus
        /// generation reserve fits inside the model context. Attached text
        /// documents opt out: silently dropping their leading pages would
        /// produce a deceptively incomplete answer, so a real overflow is
        /// reported instead. Multimodal callers can likewise preserve the complete
        /// prepared input rather than drop leading instructions.</summary>
        public List<int> TruncatePromptToContext(
            ChatSession session,
            List<int> inputTokens,
            int maxTokens,
            out int effectiveMaxTokens,
            string requestId = null,
            bool preserveAllInput = false,
            int executionContextLimit = 0,
            List<int> explicitBreakpoints = null,
            string preservedInputKind = "document")
        {
            var model = _lifecycle.Model;
            int maxCtx = model.MaxContextLength;
            if (executionContextLimit > 0 && (maxCtx <= 0 || executionContextLimit < maxCtx))
                maxCtx = executionContextLimit;
            int inputCount = inputTokens?.Count ?? 0;

            // Shrink the reserve to the room the prompt leaves. The clamped
            // value is what the engine reserves, so it flows back to the caller
            // for maxNewTokens.
            effectiveMaxTokens = ClampGenerationReserve(maxTokens, inputCount, maxCtx);

            RejectAttachedDocumentOverflow(
                inputCount, effectiveMaxTokens, maxCtx, preserveAllInput, preservedInputKind);
            if (maxCtx <= 0 || inputTokens == null || (long)inputCount + effectiveMaxTokens <= maxCtx)
                return inputTokens;

            int available = maxCtx - effectiveMaxTokens;
            if (available < 1)
            {
                throw new PromptContextOverflowException(
                    $"Prompt ({inputTokens.Count} tokens) exceeds the model's context limit ({maxCtx} tokens). " +
                    "Please shorten the input or reduce attached file size.");
            }

            int trimStart = inputTokens.Count - available;
            trimStart = model.MultimodalInjector.ClampTrimStart(trimStart, requestId);
            int kept = inputTokens.Count - trimStart;
            if (kept < 1)
            {
                throw new PromptContextOverflowException(
                    $"Prompt ({inputTokens.Count} tokens) exceeds the model's context limit ({maxCtx} tokens). " +
                    "Please shorten the input or reduce attached file size.");
            }

            _logger.LogWarning(LogEventIds.PromptTruncated,
                "prompt.truncated from {OriginalTokens} to {KeptTokens} tokens (contextLimit={ContextLimit}, generationReserve={MaxTokens}, sessionId={SessionId})",
                inputTokens.Count, kept, maxCtx, effectiveMaxTokens, session?.Id ?? "(none)");
            model.MultimodalInjector.TrimPreparedPrompt(trimStart, requestId);
            // Recorded turns are NOT cleared here. They are keyed by the history that
            // preceded each turn and spliced only where a later request reproduces it, so
            // dropping this prompt's head invalidates none of them; clearing a shared
            // session's store would also have wiped every other conversation's turns.

            // Dropping the first trimStart tokens renumbers everything that
            // survives, so the breakpoints have to move with it (in place - the
            // caller holds this list and passes it on to the sequence). A
            // breakpoint at or before the cut marked a prefix that no longer
            // exists in the prompt at all and is discarded; if that empties the
            // list the request remains in explicit cache-none mode rather than
            // silently widening the boundary to the whole truncated prompt.
            if (explicitBreakpoints != null && explicitBreakpoints.Count > 0)
            {
                for (int i = explicitBreakpoints.Count - 1; i >= 0; i--)
                {
                    if (explicitBreakpoints[i] <= trimStart)
                        explicitBreakpoints.RemoveAt(i);
                    else
                        explicitBreakpoints[i] -= trimStart;
                }
            }

            return inputTokens.GetRange(trimStart, kept);
        }

        /// <summary>Keep only explicit cache boundaries whose token prefix is
        /// unchanged by multimodal placeholder expansion. An empty, non-null
        /// list is intentional: it means the request explicitly caches no
        /// blocks, rather than falling back to implicit cache-all behavior.</summary>
        internal static void RetainCacheBreakpointsInUnchangedPrefix(
            IReadOnlyList<int> beforeExpansion,
            IReadOnlyList<int> afterExpansion,
            List<int> explicitBreakpoints)
        {
            if (explicitBreakpoints == null) return;

            int beforeCount = beforeExpansion?.Count ?? 0;
            int afterCount = afterExpansion?.Count ?? 0;
            int common = Math.Min(beforeCount, afterCount);
            int unchangedPrefix = 0;
            while (unchangedPrefix < common
                && beforeExpansion![unchangedPrefix] == afterExpansion![unchangedPrefix])
            {
                unchangedPrefix++;
            }

            for (int i = explicitBreakpoints.Count - 1; i >= 0; i--)
            {
                if (explicitBreakpoints[i] > unchangedPrefix)
                    explicitBreakpoints.RemoveAt(i);
            }
        }

        /// <summary>
        /// Clamp a generation reserve to the context room the prompt leaves, so
        /// a large default reserve on a small-context model still admits a short
        /// prompt. The reserve is only ever shrunk, never below 1, and never
        /// when the context length is unknown. A prompt that alone overflows the
        /// context is left for the caller's trim/reject logic.
        /// </summary>
        internal static int ClampGenerationReserve(int requestedReserve, int promptTokenCount, int contextLimit)
        {
            if (contextLimit <= 0)
                return requestedReserve;

            int room = Math.Max(1, contextLimit - promptTokenCount);
            return Math.Min(requestedReserve, room);
        }

        internal static void RejectAttachedDocumentOverflow(
            int promptTokens,
            int maxTokens,
            int modelContextLimit,
            bool preserveAllInput,
            string preservedInputKind = "document")
        {
            if (!preserveAllInput || modelContextLimit <= 0 ||
                (long)promptTokens + maxTokens <= modelContextLimit)
            {
                return;
            }

            if (promptTokens >= modelContextLimit)
            {
                string attachmentGuidance = string.Equals(
                    preservedInputKind, "document", StringComparison.Ordinal)
                    ? "For a large CSV or table, enable code execution so the agent can analyze the complete file with tools; for other documents, attach a shorter document or configure a larger effective context limit."
                    : "Attach less media content or configure a larger effective context limit.";
                throw new PromptContextOverflowException(
                    $"The complete attached {preservedInputKind} makes this prompt require {promptTokens} tokens, " +
                    $"which exceeds the effective model/engine context limit of {modelContextLimit} tokens. No " +
                    $"{preservedInputKind} content was truncated. The maxTokens/reply-length setting controls only " +
                    "generated output and cannot enlarge the context window. " + attachmentGuidance);
            }

            throw new PromptContextOverflowException(
                $"The prompt containing the complete attached {preservedInputKind} requires {promptTokens} prompt " +
                $"tokens plus a {maxTokens}-token generation reserve, but the current model/engine " +
                $"configuration allows {modelContextLimit} context tokens. No {preservedInputKind} content was " +
                "truncated. Reduce the reply length, attach a shorter document, or use a model configured " +
                "for more context.");
        }

        internal static bool HasTextFileAttachments(List<ChatMessage> history)
        {
            if (history == null)
                return false;

            foreach (ChatMessage message in history)
            {
                if (message?.TextFilePaths != null && message.TextFilePaths.Count > 0)
                    return true;

                // API clients may inline /api/upload's textContent without also
                // echoing textFilePaths. Recognize the documented envelopes so
                // those documents receive the same no-silent-truncation contract
                // as the bundled Web UI.
                string content = message?.Content;
                if (!string.IsNullOrEmpty(content) &&
                    content.IndexOf("[End of file]", StringComparison.OrdinalIgnoreCase) >= 0 &&
                    (content.IndexOf("[File:", StringComparison.OrdinalIgnoreCase) >= 0 ||
                     content.IndexOf("[Attached file:", StringComparison.OrdinalIgnoreCase) >= 0))
                {
                    return true;
                }
            }

            return false;
        }

        internal const string DeepSeek41AudioInputError = AudioInputSupport.DeepSeek41Message;

        /// <summary>
        /// The refusal an architecture gives audio input, or null when it can consume
        /// audio (or is not a family the table knows): <see cref="AudioInputSupport"/>,
        /// which every entry point that accepts audio consults before writing an
        /// upload or rendering a prompt (the OpenAI chat and Responses parsers, the
        /// Web UI, this pipeline and the CLI). <paramref name="audioEncoderLoaded"/>
        /// is whether the loaded model carries its optional audio tower (Nemotron-H).
        /// </summary>
        internal static string AudioInputErrorFor(string architecture, bool audioEncoderLoaded = false)
            => AudioInputSupport.UnsupportedReasonFor(architecture, audioEncoderLoaded);

        internal static string UnsupportedAudioInputError(string architecture, List<ChatMessage> history,
            bool audioEncoderLoaded = false)
        {
            string error = AudioInputErrorFor(architecture, audioEncoderLoaded);
            if (error == null || history == null)
                return null;
            foreach (ChatMessage message in history)
                if (message?.AudioPaths is { Count: > 0 })
                    return error;
            return null;
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

        internal static bool HasImageAttachments(List<ChatMessage> history)
        {
            if (history == null) return false;
            foreach (ChatMessage message in history)
            {
                if (message?.ImagePaths is { Count: > 0 })
                    return true;
            }
            return false;
        }

        /// <summary>
        /// A string naming every image/audio attachment of a history in order, or null
        /// when there is none. Used only to tell whether history compaction removed
        /// media; prompt reuse compares media positionally by content instead
        /// (<see cref="SequenceState.MediaSpans"/>).
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
        /// Reclaim text turns that a first multimodal compaction discarded only because
        /// it charged the expansion cost of media removed with an older turn. Recovery
        /// may not bring that media back, and the expanded result must still fit the hard
        /// context ceiling. Kept pure so the boundary policy can be regression-tested
        /// without running a vision encoder.
        /// </summary>
        internal static ContextHistoryWindow RecoverHistoryAfterRemovedMedia(
            List<ChatMessage> originalHistory,
            int originalUnexpandedTokens,
            int targetPromptLimit,
            int hardPromptLimit,
            int remainingMediaOverhead,
            ContextHistoryWindow currentWindow,
            string remainingMediaFingerprint,
            Func<List<ChatMessage>, int> countPromptTokens)
        {
            if (originalHistory == null)
                throw new ArgumentNullException(nameof(originalHistory));
            if (countPromptTokens == null)
                throw new ArgumentNullException(nameof(countPromptTokens));
            if (targetPromptLimit <= 0 || hardPromptLimit <= 0 || remainingMediaOverhead < 0)
                return currentWindow;

            int recoveredUnexpandedLimit = targetPromptLimit > remainingMediaOverhead
                ? targetPromptLimit - remainingMediaOverhead
                : 1;
            ContextHistoryWindow recovered = CompactHistoryForContext(
                originalHistory,
                originalUnexpandedTokens,
                recoveredUnexpandedLimit,
                countPromptTokens);
            bool sameRemainingMedia = string.Equals(
                BuildMediaFingerprint(recovered.History),
                remainingMediaFingerprint,
                StringComparison.Ordinal);
            bool fitsHardLimit =
                (long)recovered.FinalPromptTokens + remainingMediaOverhead <= hardPromptLimit;
            return sameRemainingMedia
                && fitsHardLimit
                && recovered.RemovedMessages < currentWindow.RemovedMessages
                    ? recovered
                    : currentWindow;
        }

        /// <summary>
        /// Find the length of the longest prefix of the byte buffer that forms valid UTF-8.
        /// Strips any trailing incomplete multi-byte sequence.
        /// </summary>
        /// <summary>
        /// How many tokens of THINKING a turn may spend before it is stopped, or 0 for
        /// no cap.
        ///
        /// <para>
        /// Default: three quarters of the turn's allowance. Families supporting a
        /// trained budget-end token continue with the remaining answer allowance;
        /// other families retain the existing explained hard stop.
        /// The shape of the failure this prevents is not "the model thought a bit
        /// too long" — it is "the model thought until there was nothing left and returned
        /// an empty string", which reads to a user as the server being broken. A model
        /// that closes its thinking before the cap never notices this exists.
        /// </para>
        /// <para>
        /// TS_THINKING_BUDGET overrides: a token count, or 0 to disable the cap entirely
        /// for a deployment that would rather have long reasoning than a guaranteed answer.
        /// </para>
        /// </summary>
        internal static int ThinkingBudgetFor(int maxTokens, bool enableThinking)
        {
            if (!enableThinking || maxTokens <= 0)
                return 0;

            string configured = Environment.GetEnvironmentVariable("TS_THINKING_BUDGET");
            if (!string.IsNullOrWhiteSpace(configured)
                && int.TryParse(configured, NumberStyles.Integer, CultureInfo.InvariantCulture, out int explicitBudget))
            {
                return explicitBudget > 0 ? explicitBudget : 0;
            }

            // Small allowances are left alone: capping a 200-token turn at 150 would fire
            // on ordinary short reasoning.
            if (maxTokens < 512)
                return 0;

            return (int)(maxTokens * 0.75);
        }

        /// <summary>
        /// Cap for a reasoning channel the model opens although the request turned
        /// thinking OFF (families declaring <see cref="ChatProtocol.ThinkingBudgetOpenToken"/>).
        /// That channel never reaches the client, so every token in it is latency nobody
        /// asked for, and a model that stays in it until <c>max_tokens</c> returns an empty
        /// answer - Gemma 4 E4B did exactly that on the final turn of a tool workflow
        /// (256 tokens of thought, content empty). A short, closed thought still lets it
        /// answer: measured on E4B (Metal, Q8_0) the channel closed at the first line break
        /// past 16, 32, 48 or 64 tokens was followed by the correct result every time,
        /// while closing at exactly 16 or 64 tokens - mid-sentence - produced leaked
        /// reasoning and a tool call. A quarter of the allowance, at most 64 tokens;
        /// <c>TS_THINKING_BUDGET=0</c> disables it together with the thinking cap.
        /// </summary>
        internal static int UnrequestedThinkingBudgetFor(int maxTokens)
        {
            if (maxTokens <= 0)
                return 0;
            string configured = Environment.GetEnvironmentVariable("TS_THINKING_BUDGET");
            if (!string.IsNullOrWhiteSpace(configured)
                && int.TryParse(configured, NumberStyles.Integer, CultureInfo.InvariantCulture, out int explicitBudget)
                && explicitBudget <= 0)
                return 0;
            return Math.Clamp(maxTokens / 4, 1, MaxUnrequestedThinkingTokens);
        }

        internal const int MaxUnrequestedThinkingTokens = 64;

        internal static SamplingConfig WithThinkingBudget(SamplingConfig config, ITokenizer tokenizer,
            string architecture, int tokenBudget, out bool installed,
            bool enableThinking = true, IReadOnlyList<int> promptTokens = null)
        {
            installed = false;
            var protocol = ChatProtocolRegistry.For(architecture);
            string end = protocol?.ThinkingBudgetEndToken;
            if (end == null || tokenizer == null || config.Grammar?.IsActive == true)
                return config;
            int id = TrainedSingleToken(tokenizer, end);
            if (id < 0)
                return config;

            int openId = -1;
            bool openAtStart = true;
            bool suppressUnopenedEnd = false;
            if (protocol.ThinkingBudgetOpenToken != null)
            {
                openId = TrainedSingleToken(tokenizer, protocol.ThinkingBudgetOpenToken);
                if (openId < 0 || openId == id)
                    return config;
                openAtStart = PromptLeavesChannelOpen(promptTokens, openId, id);
                suppressUnopenedEnd = !enableThinking && !openAtStart
                    && PromptEndsWith(tokenizer, promptTokens, protocol.SuppressUnopenedThinkingEndAfter);
            }
            else if (!enableThinking)
            {
                // A family whose channel only the prompt opens has nothing to cap when
                // thinking is off: its prompt closed the channel, and the reply is the answer.
                return config;
            }

            if (tokenBudget <= 0 && !suppressUnopenedEnd)
                return config;
            // The caller's config can be shared by requests. Only this request
            // gets the immutable budget policy and an independent grammar position.
            // OpenAI creates a fresh constraint, but direct ModelService callers
            // may reuse one delayed-grammar config across concurrent requests.
            SamplingConfig result = config.Clone();
            result.Grammar = config.Grammar?.Fork();
            // A channel the request did not ask for closes at a line break (see
            // ThinkingTokenBudget.CloseAtBoundary); a requested one keeps its exact budget.
            Func<int, bool> boundary = enableThinking ? null : EndsLine(tokenizer);
            result.ThinkingBudget = new ThinkingTokenBudget(tokenBudget > 0 ? tokenBudget : int.MaxValue, id,
                closeOnRepetition: true, openTokenId: openId, openAtStart: openAtStart,
                suppressUnopenedEnd: suppressUnopenedEnd, closeAtBoundary: boundary);
            installed = true;
            return result;
        }

        private static Func<int, bool> EndsLine(ITokenizer tokenizer) => token =>
        {
            try
            {
                return tokenizer.Decode(new List<int> { token }).EndsWith('\n');
            }
            catch (Exception)
            {
                return false;
            }
        };

        private static int TrainedSingleToken(ITokenizer tokenizer, string text)
        {
            int id = tokenizer.LookupToken(text);
            return id < 0 || id >= tokenizer.VocabSize || tokenizer.Vocab[id] != text || tokenizer.IsEos(id) ? -1 : id;
        }

        /// <summary>The prompt's tail opens a channel it does not close (Gemma 4 primes
        /// <c>&lt;|channel&gt;thought\n</c> after a tool result with thinking on).</summary>
        internal static bool PromptLeavesChannelOpen(IReadOnlyList<int> promptTokens, int openId, int endId)
        {
            if (promptTokens == null)
                return false;
            int stop = Math.Max(0, promptTokens.Count - 64);
            for (int i = promptTokens.Count - 1; i >= stop; i--)
            {
                if (promptTokens[i] == endId) return false;
                if (promptTokens[i] == openId) return true;
            }
            return false;
        }

        private static bool PromptEndsWith(ITokenizer tokenizer, IReadOnlyList<int> promptTokens, string marker)
        {
            if (string.IsNullOrEmpty(marker) || promptTokens == null || promptTokens.Count == 0)
                return false;
            try
            {
                int take = Math.Min(promptTokens.Count, 16);
                var tail = new List<int>(take);
                for (int i = promptTokens.Count - take; i < promptTokens.Count; i++)
                    tail.Add(promptTokens[i]);
                return tokenizer.Decode(tail).TrimEnd().EndsWith(marker, StringComparison.Ordinal);
            }
            catch (Exception)
            {
                return false;
            }
        }

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
