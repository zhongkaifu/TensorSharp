// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

namespace TensorSharp.Runtime.Telemetry
{
    /// <summary>
    /// Attribute names on TensorSharp spans and metrics. The <c>gen_ai.*</c> names follow the
    /// OpenTelemetry GenAI semantic conventions (development status); the rest are
    /// TensorSharp's own and keep the <c>tensorsharp.</c> prefix so they never collide with a
    /// future convention.
    ///
    /// <para>
    /// Metric attributes must stay low-cardinality: operation, model, backend, protocol,
    /// finish reason, mode, phase, stage, outcome and error type. Ids, prompts, sizes and
    /// counts go on spans only.
    /// </para>
    /// </summary>
    public static class TelemetryAttributes
    {
        // GenAI semantic conventions ---------------------------------------------------
        public const string GenAiOperationName = "gen_ai.operation.name";
        public const string GenAiProviderName = "gen_ai.provider.name";
        public const string GenAiRequestModel = "gen_ai.request.model";
        public const string GenAiResponseModel = "gen_ai.response.model";
        public const string GenAiRequestMaxTokens = "gen_ai.request.max_tokens";
        public const string GenAiRequestTemperature = "gen_ai.request.temperature";
        public const string GenAiRequestTopP = "gen_ai.request.top_p";
        public const string GenAiRequestTopK = "gen_ai.request.top_k";
        public const string GenAiRequestSeed = "gen_ai.request.seed";
        public const string GenAiResponseId = "gen_ai.response.id";
        public const string GenAiResponseFinishReasons = "gen_ai.response.finish_reasons";
        public const string GenAiUsageInputTokens = "gen_ai.usage.input_tokens";
        public const string GenAiUsageOutputTokens = "gen_ai.usage.output_tokens";
        public const string GenAiConversationId = "gen_ai.conversation.id";
        public const string GenAiTokenType = "gen_ai.token.type";
        public const string GenAiToolName = "gen_ai.tool.name";
        public const string GenAiToolCallId = "gen_ai.tool.call.id";
        /// <summary>Prompt messages; only with content capture on.</summary>
        public const string GenAiInputMessages = "gen_ai.input.messages";
        /// <summary>Model output; only with content capture on.</summary>
        public const string GenAiOutputMessages = "gen_ai.output.messages";

        public const string ErrorType = "error.type";

        // gen_ai.operation.name values.
        public const string OperationChat = "chat";
        public const string OperationTextCompletion = "text_completion";
        public const string OperationEmbeddings = "embeddings";
        public const string OperationExecuteTool = "execute_tool";

        /// <summary>Prompt tokens served from the prefix cache, a subset of input tokens (GenAI semconv span attribute).</summary>
        public const string GenAiUsageCacheReadInputTokens = "gen_ai.usage.cache_read.input_tokens";

        // Token metric attribute. TensorSharp's own name: the GenAI conventions removed
        // gen_ai.token.type, and their server metrics define no token counter. Cached prompt
        // tokens are a subset of input, so they get their own counter rather than a third
        // value here (which would count them twice in a sum).
        public const string TokenType = "tensorsharp.token.type";
        public const string TokenTypeInput = "input";
        public const string TokenTypeOutput = "output";
        /// <summary>Kept for source compatibility; cached tokens go to <c>tensorsharp.tokens.cache_read</c>.</summary>
        public const string TokenTypeCached = "cached";
        /// <summary>Time to first token in milliseconds, on the chat span, for percentiles in Application Insights.</summary>
        public const string TimeToFirstTokenMs = "tensorsharp.ttft_ms";

        // TensorSharp ------------------------------------------------------------------
        public const string RequestId = "tensorsharp.request_id";
        /// <summary><c>openai</c>, <c>ollama</c> or <c>webui</c>, as the request logger classifies the path.</summary>
        public const string Protocol = "tensorsharp.protocol";
        public const string Backend = "tensorsharp.backend";
        /// <summary><c>prefix-cache-warmup</c> for the host's own startup request.</summary>
        public const string Origin = "tensorsharp.origin";
        /// <summary><c>completed</c>, <c>cancelled</c>, <c>refused</c> or <c>failed</c>.</summary>
        public const string Outcome = "tensorsharp.outcome";
        public const string PrefixCacheReusedTokens = "tensorsharp.prefix_cache.reused_tokens";
        /// <summary>
        /// The engine's own id for the sequence (<c>chat-…</c>), on the chat span, so engine-thread
        /// log lines (which carry it as <c>{EngineRequestId}</c> and have no trace id) join to
        /// their request. A different id space from <see cref="RequestId"/>.
        /// </summary>
        public const string SequenceId = "tensorsharp.engine_request_id";
        /// <summary>Times the scheduler preempted the sequence to make room for others.</summary>
        public const string Preemptions = "tensorsharp.preemptions";
        /// <summary>1-based round of a skills or multi-agent turn, on each generation's span.</summary>
        public const string Round = "tensorsharp.round";
        /// <summary>The model call was made by TensorSharp itself (knowledge reranking, media extraction, warm-up).</summary>
        public const string Internal = "tensorsharp.internal";
        public const string EmbeddingInputs = "tensorsharp.embedding.inputs";
        public const string EmbeddingBatchSize = "tensorsharp.embedding.batch_size";
        public const string DiffusionPreviews = "tensorsharp.diffusion.previews";
        public const string LoraCount = "tensorsharp.lora.count";
        public const string ComputePrefillMs = "tensorsharp.compute.prefill_ms";
        public const string ComputeDecodeMs = "tensorsharp.compute.decode_ms";
        public const string BatchSize = "tensorsharp.engine.batch_size";
        public const string StepKind = "tensorsharp.engine.step_kind";
        public const string ModelArchitecture = "tensorsharp.model.architecture";
        public const string ModelFile = "tensorsharp.model.file";
        public const string TensorParallelDegree = "tensorsharp.tp.degree";
        public const string TensorParallelNodeId = "tensorsharp.tp.node_id";
        public const string GpuDevice = "tensorsharp.gpu.device";
        /// <summary>Where a routed expert ran: <see cref="MoePlacementDevice"/>, <see cref="MoePlacementStreamed"/>
        /// or <see cref="MoePlacementHost"/>.</summary>
        public const string MoePlacement = "tensorsharp.moe.placement";
        /// <summary>An expert the accelerator already held (a tier slot, a VRAM residency slot).</summary>
        public const string MoePlacementDevice = "device";
        /// <summary>An expert read from storage into an accelerator slot for the route (a window slot).</summary>
        public const string MoePlacementStreamed = "streamed";
        /// <summary>An expert the host's expert kernel computed.</summary>
        public const string MoePlacementHost = "host";

        // Serving signals ----------------------------------------------------------------
        /// <summary>A prefix-cache lookup's result: <c>full</c>, <c>partial</c> or <c>miss</c>.</summary>
        public const string PrefixCacheOutcome = "tensorsharp.prefix_cache.outcome";
        /// <summary>Where a reused prefix came from (<c>end_state</c>, <c>primary</c>, <c>pages</c>,
        /// <c>batched_end_state</c>), or why a miss reused nothing (<c>shared_prefix</c>,
        /// <c>rewind</c>, <c>live_cache</c>, <c>retained_state</c>, <c>pages</c>, <c>breakpoints</c>,
        /// <c>none</c>).</summary>
        public const string PrefixCacheSource = "tensorsharp.prefix_cache.source";
        /// <summary>A sequence slot's state: <c>running</c> (bound to a request) or <c>retained</c>
        /// (kept for a conversation's next turn).</summary>
        public const string SlotState = "tensorsharp.slot.state";
        /// <summary>Why a retained slot was given back: <c>new_slot</c> or <c>growth</c>.</summary>
        public const string SlotReclaimReason = "tensorsharp.slot.reclaim_reason";
        /// <summary>How a slot rewind went: <c>checkpoint</c> (to its checkpoint exactly),
        /// <c>replay</c> (past it, replaying tokens), <c>reset</c> (to 0) or <c>refused</c>.</summary>
        public const string RewindRoute = "tensorsharp.rewind.route";
        /// <summary>A captured CUDA graph's kind: <c>solo</c> (one sequence's decode step) or
        /// <c>batched</c>.</summary>
        public const string GraphKind = "tensorsharp.cuda_graph.kind";
        /// <summary>One stage of a forward, as the engine's stage timer names it (<c>hc</c>,
        /// <c>experts</c>, <c>attncore</c>, ...).</summary>
        public const string EngineStage = "tensorsharp.engine.stage";
        /// <summary>A step's rows, bucketed: <c>1</c>, <c>2-4</c>, <c>5-8</c>, <c>9-16</c>, <c>17+</c>.</summary>
        public const string RowsBucket = "tensorsharp.engine.rows";
        /// <summary>Request span: output tokens a second after the first token.</summary>
        public const string DecodeTokensPerSecond = "tensorsharp.decode.tokens_per_second";
        /// <summary>Request span: prompt tokens prefilled a second, cached ones excluded.</summary>
        public const string PrefillTokensPerSecond = "tensorsharp.prefill.tokens_per_second";
        /// <summary>Request span: prompt tokens that were not served from the prefix cache.</summary>
        public const string UncachedPromptTokens = "tensorsharp.prompt.uncached_tokens";

        /// <summary>The <see cref="RowsBucket"/> value for <paramref name="rows"/>.</summary>
        public static string RowsBucketOf(int rows) => rows switch
        {
            <= 1 => "1",
            <= 4 => "2-4",
            <= 8 => "5-8",
            <= 16 => "9-16",
            _ => "17+",
        };

        // Media -------------------------------------------------------------------------
        public const string MediaKind = "tensorsharp.media.kind";
        public const string VideoMode = "tensorsharp.video.mode";
        public const string VideoWidth = "tensorsharp.video.width";
        public const string VideoHeight = "tensorsharp.video.height";
        public const string VideoFrames = "tensorsharp.video.frames";
        public const string VideoFps = "tensorsharp.video.fps";
        public const string VideoSteps = "tensorsharp.video.steps";
        public const string VideoSeed = "tensorsharp.video.seed";
        public const string VideoAudio = "tensorsharp.video.audio";
        public const string VideoLoraCount = "tensorsharp.video.lora_count";
        public const string VideoPhase = "tensorsharp.video.phase";
        public const string VideoStage = "tensorsharp.video.stage";
        public const string VideoStep = "tensorsharp.video.step";
        public const string VideoCached = "tensorsharp.video.cached";
        public const string VideoCodec = "tensorsharp.video.codec";
        public const string VideoAudioMuxed = "tensorsharp.video.audio_muxed";
        public const string VideoBytes = "tensorsharp.video.bytes";
        /// <summary>The video prompt; only with content capture on.</summary>
        public const string VideoPrompt = "tensorsharp.video.prompt";
        /// <summary>LTX 2.5 found the prompt encoding in its small prompt cache.</summary>
        public const string VideoPromptCacheHit = "tensorsharp.video.prompt_cache_hit";
        public const string VideoTiles = "tensorsharp.video.tiles";
        public const string VideoDetail = "tensorsharp.video.detail";
        /// <summary><c>generate</c> (from words) or <c>edit</c> (from pictures), on the image span and metric.</summary>
        public const string ImageMode = "tensorsharp.image.mode";
        public const string ImageWidth = "tensorsharp.image.width";
        public const string ImageHeight = "tensorsharp.image.height";
        public const string ImageSteps = "tensorsharp.image.steps";
        public const string ImageCfg = "tensorsharp.image.cfg";
        public const string ImageSeed = "tensorsharp.image.seed";
        /// <summary>The number of reference pictures an edit was given.</summary>
        public const string ImageReferences = "tensorsharp.image.references";
        /// <summary>The image prompt; only with content capture on.</summary>
        public const string ImagePrompt = "tensorsharp.image.prompt";

        // Code execution ----------------------------------------------------------------
        public const string CodeExecShell = "tensorsharp.codeexec.shell";
        public const string CodeExecExitCode = "tensorsharp.codeexec.exit_code";
        public const string CodeExecSandbox = "tensorsharp.codeexec.sandbox";
        public const string CodeExecTimedOut = "tensorsharp.codeexec.timed_out";

        // Studio ------------------------------------------------------------------------
        public const string StudioJobId = "tensorsharp.studio.job_id";
        public const string StudioJobKind = "tensorsharp.studio.job_kind";
        public const string StudioProjectId = "tensorsharp.studio.project_id";
        public const string StudioStage = "tensorsharp.studio.stage";

        // Values ------------------------------------------------------------------------
        public const string OutcomeCompleted = "completed";
        public const string OutcomeCancelled = "cancelled";
        public const string OutcomeRefused = "refused";
        public const string OutcomeFailed = "failed";
        public const string OriginPrefixCacheWarmup = "prefix-cache-warmup";
        /// <summary>The CLI's <c>--warmup-runs</c> passes, run before the timed one.</summary>
        public const string OriginWarmup = "warmup";
    }
}
