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
using System.IO;
using System.Linq;
using System.Threading;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Server.Skills;

namespace TensorSharp.Server
{
    public partial class ModelService : IDisposable
    {
        private readonly ILogger _logger;
        private readonly ModelLifecycleService _lifecycle;
        private readonly ChatSession _intrinsicSession;
        private readonly InferenceEngineHost _engineHost;
        private readonly ChatGenerationPipeline _generation;

        public ModelService()
            : this(NullLogger<ModelService>.Instance)
        {
        }

        public ModelService(ILogger<ModelService> logger)
            : this(logger, createModel: null)
        {
        }

        /// <summary>Test seam for exercising the service/API boundary without loading
        /// a production-sized model. Production callers use the public constructor.</summary>
        internal ModelService(
            ILogger<ModelService> logger,
            Func<string, BackendType, ITensorParallelGroup, string, ModelBase> createModel)
        {
            logger ??= NullLogger<ModelService>.Instance;
            _logger = logger;

            var promptRenderer = new GgufPromptRenderer();
            var kvCacheRenderer = new KVCachePromptRenderer(promptRenderer);
            var telemetry = new InferenceTelemetry(logger);

            _lifecycle = createModel == null
                ? new ModelLifecycleService(logger)
                : new ModelLifecycleService(logger, createModel);
            // Shared by every stateless request (OpenAI Chat, Responses, Ollama chat).
            // It holds many conversations: each request's transcripts and cache scope
            // come from what its own history proves, never from the session.
            _intrinsicSession = new ChatSession("__svc_intrinsic__", sharedAcrossConversations: true);
            _engineHost = new InferenceEngineHost(_lifecycle, logger);
            _generation = new ChatGenerationPipeline(_lifecycle, _engineHost, kvCacheRenderer, telemetry, logger);
        }

        /// <summary>The internal lifecycle service. Exposed so the
        /// <see cref="InferenceEngineHost"/> can hook into model load/unload
        /// transitions; do not call this from other code paths.</summary>
        internal ModelLifecycleService LifecycleService => _lifecycle;

        public bool IsLoaded => _lifecycle.IsLoaded;
        public string LoadedModelName => _lifecycle.LoadedModelName;
        public string LoadedModelPath => _lifecycle.LoadedModelPath;
        public string LoadedMmProjName => _lifecycle.LoadedMmProjName;
        public string LoadedMmProjPath => _lifecycle.LoadedMmProjPath;
        public string LoadedBackend => _lifecycle.LoadedBackend;
        public virtual string Architecture => _lifecycle.Architecture;
        /// <summary>The loaded GGUF's embedded chat template, or null. One architecture name
        /// can carry two turn formats, and some parsing decisions depend on which one
        /// (<see cref="OutputParserFactory.IsAlwaysRequired(string, string)"/>).</summary>
        public virtual string ChatTemplate => _lifecycle.Model?.Config?.ChatTemplate;
        /// <summary>Whether the loaded model carries its optional audio tower (see <see cref="TensorSharp.Models.Architecture.AudioInputSupport"/>).</summary>
        public virtual bool IsAudioEncoderLoaded => TensorSharp.Models.Architecture.AudioInputSupport.IsAudioEncoderLoaded(_lifecycle.Model);
        public ModelBase Model => _lifecycle.Model;

        /// <summary>
        /// Loaded text-LoRA adapters, held independently of the base model so they can be
        /// loaded and unloaded at runtime. Written only through <see cref="RegisterLora"/> and
        /// <see cref="UnloadLora"/>. In the server, <c>LoraAdapterDirectory.LoadAsync</c> reads
        /// the file (confined to <c>--loras-dir</c>) and calls <see cref="RegisterLora"/>, for
        /// <c>POST /v1/loras</c> and for the startup load of every adapter at the top of the
        /// directory, and <c>DELETE /v1/loras/{name}</c> calls <see cref="UnloadLora"/>.
        /// <see cref="LoadLora"/> (read any path, then <see cref="RegisterLora"/>) has no server
        /// caller. The protocol adapters only resolve a request's <c>lora</c> name in it
        /// (<see cref="TryResolveLora"/>), and the adapter a request resolved travels with
        /// that request alone.
        /// </summary>
        public TextLoraRegistry LoraRegistry { get; } = new();

        /// <summary>
        /// Resolve a request's <c>lora</c> field to the adapter it runs with: null, empty or
        /// <c>"none"</c> is the base model (<paramref name="adapter"/> null). False, with a
        /// reason for a 400, when no model is loaded, the model cannot take adapters (block
        /// diffusion, an unsupported family, backend or multi-device mode), the name is not
        /// loaded, or any of the adapter's modules cannot be applied to the loaded model.
        /// Nothing here changes what any other request runs with.
        /// </summary>
        public bool TryResolveLora(string? name, out TextLoraAdapter? adapter, out string? error)
        {
            bool resolved = TryResolveLoraRequest(name, out adapter, out LoraRequestError? refusal);
            error = refusal?.Message;
            return resolved;
        }

        /// <summary>
        /// <see cref="TryResolveLora"/> with the refusal as a structured error for the 400:
        /// <c>lora_not_found</c> for a name that is not loaded, <c>model_not_loaded</c>, and
        /// <c>lora_not_applicable</c> (with the modules the loaded model cannot apply) for
        /// everything else.
        /// </summary>
        public bool TryResolveLoraRequest(string? name, out TextLoraAdapter? adapter, out LoraRequestError? error)
        {
            adapter = null;
            error = null;
            if (string.IsNullOrWhiteSpace(name) || string.Equals(name, "none", StringComparison.OrdinalIgnoreCase))
                return true;

            var loaded = LoraRegistry.Get(name);
            if (loaded == null && Model != null)
            {
                error = new LoraRequestError($"No LoRA adapter named '{name}' is loaded.", LoraRequestError.NotFound, Array.Empty<string>());
                return false;
            }
            error = Applicability(loaded);
            if (error != null)
                return false;
            adapter = loaded;
            return true;
        }

        /// <summary>
        /// Every loaded adapter with whether the loaded model can apply it and, when it cannot,
        /// why (for <c>GET /v1/loras</c> and <c>GET /v1/models</c>).
        /// </summary>
        public IReadOnlyList<LoraListing> ListLoras()
        {
            var listings = new List<LoraListing>();
            foreach (var info in LoraRegistry.List())
            {
                // Unloaded between the listing and this read: not listed.
                var adapter = LoraRegistry.Get(info.Name);
                if (adapter == null)
                    continue;
                var refusal = Applicability(adapter);
                listings.Add(new LoraListing(info, refusal == null, refusal?.Message));
            }
            return listings;
        }

        /// <summary>Null when the loaded model can apply <paramref name="adapter"/> in full;
        /// otherwise the refusal. A null adapter reads as "no model" when none is loaded.</summary>
        private LoraRequestError? Applicability(TextLoraAdapter? adapter)
        {
            var model = Model;
            if (model == null)
                return new LoraRequestError("No model is loaded.", LoraRequestError.ModelNotLoaded, Array.Empty<string>());
            IReadOnlyList<string> allModules = adapter == null
                ? Array.Empty<string>()
                : adapter.Deltas.Keys.OrderBy(k => k, StringComparer.Ordinal).ToArray();
            if (IsDiffusionModel)
                return new LoraRequestError(
                    "The loaded model generates by block diffusion, which does not apply LoRA adapters; remove \"lora\" from the request.",
                    LoraRequestError.NotApplicable, allModules);
            if (model is not IAdapterAwareModel)
                return new LoraRequestError("The loaded model cannot apply LoRA adapters.", LoraRequestError.NotApplicable, allModules);

            var plan = model.PlanTextLora(adapter!);
            return plan.IsApplicable
                ? null
                : new LoraRequestError(plan.RefusalMessage!, LoraRequestError.NotApplicable, plan.UnsupportedModules);
        }

        /// <summary>
        /// Load (or reload) a text-LoRA adapter and register it under <paramref name="name"/>.
        /// With a model loaded the adapter is validated and its factors are materialized now,
        /// once (a reload under the same name keeps serving the requests that hold the old
        /// one, which is released after them); an adapter the model cannot apply in full is
        /// refused with <see cref="TextLoraNotApplicableException"/> and nothing is registered.
        /// </summary>
        /// <param name="discoverSiblingConfig">Passed to <see cref="TextLoraLoader.Load"/>: false when
        /// the caller already resolved (or ruled out) an <c>adapter_config.json</c> next to the weights.</param>
        /// <exception cref="InvalidDataException">The file is malformed.</exception>
        /// <exception cref="NotSupportedException">The file uses a LoRA feature that cannot be applied
        /// exactly, or (<see cref="TextLoraNotApplicableException"/>) the loaded model cannot apply it.</exception>
        /// <exception cref="FileNotFoundException">A named file is missing.</exception>
        public TextLoraAdapter LoadLora(
            string name, string path, string? configPath = null, float? scale = null, bool discoverSiblingConfig = true)
        {
            if (string.IsNullOrWhiteSpace(name))
                throw new ArgumentException("A LoRA adapter name is required.", nameof(name));

            return RegisterLora(TextLoraLoader.Load(name, path, configPath, scale, discoverSiblingConfig));
        }

        /// <summary>
        /// The second half of <see cref="LoadLora"/>, for an adapter the caller has already read
        /// with <see cref="TextLoraLoader.Load"/>: bind it to the loaded model (if any) and
        /// register it under its name. The file was read in full before this, so apart from
        /// <see cref="TextLoraNotApplicableException"/> (the model cannot apply the adapter; a
        /// target it does not have, whatever its name or layer number, is one of the
        /// unsupported modules) what this throws is a failure on the server's side, not the
        /// file's. Whatever it throws, the adapter is left neither bound nor registered.
        /// </summary>
        public TextLoraAdapter RegisterLora(TextLoraAdapter adapter)
        {
            ArgumentNullException.ThrowIfNull(adapter);
            string name = adapter.Name;
            var model = Model;
            if (model != null && !model.IsRetiring && model is IAdapterAwareModel aware)
            {
                bool bound;
                string? error;
                IReadOnlyList<string> unsupported = Array.Empty<string>();
                GpuLockWaiters.Enter(model.GpuComputeLock);
                lock (model.GpuComputeLock)
                {
                    GpuLockWaiters.Exit(model.GpuComputeLock);
                    bool keep = false;
                    try
                    {
                        bound = aware.TryBindAdapter(adapter, out error);
                        if (!bound)
                            unsupported = model.PlanTextLora(adapter).UnsupportedModules;
                        keep = bound;
                    }
                    finally
                    {
                        // The plan is stored before the factors are made (a device allocation
                        // can fail there) and before a refusal is described, and an adapter
                        // never registered is never retired, so nothing else would ever
                        // release it. Every way out but a successful bind releases it here.
                        if (!keep)
                            aware.ReleaseAdapter(adapter);
                    }
                }
                if (!bound)
                    throw new TextLoraNotApplicableException(
                        error ?? $"LoRA adapter '{name}' cannot be applied to the loaded model.", unsupported);
            }

            LoraRegistry.Register(adapter);
            // The instance this replaced (if any) is retired: release it once its requests end.
            ReleaseRetiredLoras();
            return adapter;
        }

        /// <summary>
        /// Unload the adapter registered under <paramref name="name"/>. Requests already running
        /// with it finish with it; its factors are released after the last of them. Returns
        /// false when no adapter of that name is loaded.
        /// </summary>
        public bool UnloadLora(string name)
        {
            if (!LoraRegistry.Unload(name))
                return false;
            ReleaseRetiredLoras();
            return true;
        }

        /// <summary>How many requests in flight run with an adapter named <paramref name="name"/>
        /// (any load of it, including one since replaced or unloaded).</summary>
        public int LoraInFlight(string name)
            => _engineHost.CountInFlightRequests(
                adapter => adapter is TextLoraAdapter lora && string.Equals(lora.Name, name, StringComparison.Ordinal));

        private void ReleaseRetiredLoras()
        {
            // The engine releases on its own thread, between steps, once nothing it runs holds
            // the adapter. Without an engine nothing runs, so release here.
            if (_engineHost.ReleaseRetiredAdapters())
                return;
            var model = Model;
            if (model is not IAdapterAwareModel aware)
                return;
            GpuLockWaiters.Enter(model.GpuComputeLock);
            lock (model.GpuComputeLock)
            {
                GpuLockWaiters.Exit(model.GpuComputeLock);
                foreach (var retired in aware.BoundAdapters)
                {
                    if (retired.IsRetired)
                        aware.ReleaseAdapter(retired);
                }
            }
        }

        /// <summary>
        /// Non-null when an explicitly requested <c>--draft-model</c> could not
        /// be activated on the loaded target (see
        /// <see cref="ModelLifecycleService.DraftHeadActivationError"/>). The startup
        /// loader promotes it to a fail-fast error.
        /// </summary>
        public string DraftHeadActivationError => _lifecycle.DraftHeadActivationError;

        /// <summary>The draft was not activated because the loaded model refuses
        /// speculation (see <see cref="ModelLifecycleService.DraftHeadRefusedByModel"/>).</summary>
        public bool DraftHeadRefusedByModel => _lifecycle.DraftHeadRefusedByModel;

        public bool IsModelAlreadyLoaded(string modelName)
        {
            return _lifecycle.IsModelAlreadyLoaded(modelName);
        }

        /// <summary>Engine host exposed for adapters that submit requests
        /// directly to the engine (e.g. multi-turn streaming clients that
        /// want to manage their own session bookkeeping).</summary>
        public InferenceEngineHost EngineHost => _engineHost;

        /// <summary>
        /// Release the memory the engine keeps only for the next request's speed. For a
        /// host that has just been told it is about to be killed for memory; see
        /// <see cref="InferenceEngineHost.TrimIdleMemory"/>. False when no engine stands.
        /// </summary>
        public bool TrimIdleMemory() => _engineHost.TrimIdleMemory();

        /// <summary>
        /// Builds the on-node tensor-parallel group a model load shards across, or null
        /// (the default) for a single-node load. Hosts that carry TensorSharp.Distributed
        /// (the Server, the CLI) set it; this library does not reference that assembly,
        /// so a host without peers — or without a CUDA backend to link — never pays for it.
        /// Read at load time: set it before <see cref="LoadModel"/>.
        /// </summary>
        public Func<BackendType, ITensorParallelGroup?>? TensorParallelGroupFactory
        {
            get => _lifecycle.TensorParallelGroupFactory;
            set => _lifecycle.TensorParallelGroupFactory = value;
        }

        /// <summary>
        /// The operator's <c>--chat-template-file</c> and the model it applies to (the
        /// host's <c>--model</c>); see <see cref="ModelLifecycleService.ChatTemplateOverride"/>.
        /// Read at load time: set it before <see cref="LoadModel"/>.
        /// </summary>
        public (ChatTemplateFile Template, string ModelPath)? ChatTemplateOverride
        {
            get => _lifecycle.ChatTemplateOverride;
            set => _lifecycle.ChatTemplateOverride = value;
        }

        /// <summary>
        /// Sizing for the continuous-batching engine built lazily on the first request
        /// (paged KV pool, batch limits, speculation). Null (the default) keeps
        /// <see cref="SchedulerConfig.FromEnvironment"/>, i.e. the TS_SCHED_* variables;
        /// a host with no shell to set them in — an app sizing the KV pool against its
        /// memory budget — supplies the config directly. Applies to the next engine
        /// construction, so set it before the first request or before a model load.
        /// </summary>
        public SchedulerConfig SchedulerConfigOverride
        {
            get => _engineHost.SchedulerConfigOverride;
            set => _engineHost.SchedulerConfigOverride = value;
        }

        public void LoadModel(string modelPath, string mmProjPath, string backendStr)
            => LoadModelCore(modelPath, mmProjPath, backendStr, startupLoad: false);

        internal void LoadStartupModel(string modelPath, string mmProjPath, string backendStr)
            => LoadModelCore(modelPath, mmProjPath, backendStr, startupLoad: true);

        internal void ValidateDistributedModelSupport(string modelPath, string backendStr)
            => _lifecycle.ValidateDistributedModelSupport(modelPath, backendStr);

        private void LoadModelCore(string modelPath, string mmProjPath, string backendStr, bool startupLoad)
        {
            // Every server load (startup, the Web UI, a protocol adapter's reload) comes
            // through here. Timed from the top, so the wait for a running Jev turn and the
            // engine teardown count: they are part of how long the caller waits for a model.
            using var telemetry = ModelLoadTelemetry.StartLoad(modelPath, mmProjPath, backendStr, startupLoad);
            try
            {
                using var jevLease = _jevExecution.BeginChange();
                if (!_lifecycle.ShouldLoadDistributedModel(modelPath, mmProjPath, backendStr, startupLoad))
                {
                    telemetry.Skipped();
                    return;
                }
                // Tear down the per-model engine and the diffusion batch scheduler BEFORE the model is
                // unloaded so their worker threads don't race the model disposal.
                _engineHost.Reset();
                _generation.ResetDiffusionScheduler();
                lock (_intrinsicSession.HistoryLock)
                    _intrinsicSession.Transcripts.Clear();
                _lifecycle.LoadModel(modelPath, mmProjPath, backendStr);
                telemetry.Loaded(_lifecycle.Architecture, _lifecycle.LoadedBackend,
                    _lifecycle.LoadedTensorParallelDegree, _lifecycle.LoadedLayerSplitDegree);
            }
            catch (Exception ex)
            {
                telemetry.Failed(ex);
                throw;
            }
        }

        /// <summary>
        /// Release the loaded model without loading another, in the same order a swap
        /// uses: the engine and the diffusion scheduler go first (their worker threads
        /// must not race the model's disposal), then the intrinsic history, then the
        /// weights. A no-op when nothing is loaded. Exists so a host under memory
        /// pressure can free the model and keep the service, sessions and skills.
        /// </summary>
        public void UnloadModel()
        {
            using var jevLease = _jevExecution.BeginChange();
            UnloadModelCore();
        }

        private void UnloadModelCore()
        {
            // An explicit unload of a loaded model gets a span; with nothing loaded this
            // only resets idle state, which is not worth one.
            using ModelLoadTelemetry telemetry = _lifecycle.IsLoaded
                ? ModelLoadTelemetry.StartUnload(_lifecycle.LoadedModelPath, _lifecycle.Architecture, _lifecycle.LoadedBackend)
                : null;
            try
            {
                _engineHost.Reset();
                _generation.ResetDiffusionScheduler();
                lock (_intrinsicSession.HistoryLock)
                    _intrinsicSession.Transcripts.Clear();
                _lifecycle.Unload();
            }
            catch (Exception ex)
            {
                telemetry?.Failed(ex);
                throw;
            }
        }

        /// <summary>
        /// Release the loaded model AND throw the GPU backend away, so the next load
        /// builds a new one.
        ///
        /// <para>
        /// For one failure only: a GPU command buffer that came back refused. ggml-metal
        /// latches an error flag when that happens and clears it in exactly one place —
        /// <c>ggml_metal_init</c> — so every graph afterwards fails until the backend is
        /// built again. <see cref="LoadModel"/> alone cannot fix it, because the backend
        /// is a process global that loading weights never touches; the reloaded model
        /// runs on the same poisoned context and fails identically.
        /// </para>
        /// <para>
        /// iOS is where this is reached: an app that is not frontmost may not submit GPU
        /// work, so a turn interrupted by the user switching apps can poison the engine
        /// for the rest of the process. Everywhere else the flag stays terminal, which
        /// is why this is a separate call rather than something a failed graph does by
        /// itself — recreating a backend under a host that has other models on it would
        /// take those down too.
        /// </para>
        /// <para>
        /// The model is released FIRST, and that order is not cosmetic: its tensors live
        /// in buffers the recreate frees, and freeing them twice is a crash.
        /// </para>
        /// </summary>
        /// <returns>Whether a working backend is standing afterwards.</returns>
        public bool UnloadModelAndRecreateBackend()
        {
            using var jevLease = _jevExecution.BeginChange();
            UnloadModelCore();
            return TensorSharp.GGML.GgmlBasicOps.RecreateBackend();
        }

        /// <summary>
        /// Reset the given session's tracked conversation history. Engine-owned
        /// KV blocks are request-scoped and are not reset through this API.
        /// </summary>
        public void ResetSession(ChatSession session)
        {
            if (session == null)
                return;
            // Guard against a concurrent request on the same (e.g. default) session reading/rewriting
            // its transcripts while we clear them.
            // A shared session (the stateless API session, the Web UI default session)
            // holds other clients' conversations too; one client's new chat is simply a
            // request whose history proves no earlier turn, so nothing is cleared there.
            lock (session.HistoryLock)
            {
                if (!session.SharedAcrossConversations)
                    session.ResetConversation();
            }
            session.LastUsedAt = DateTime.UtcNow;
        }

        /// <summary>
        /// Dispose the given session and release its tracked history. The
        /// inference engine releases KV blocks independently of session
        /// disposal.
        /// </summary>
        public void DisposeSession(ChatSession session)
        {
            session?.Dispose();
        }

        /// <summary>
        /// Stream chat inference tokens.
        /// </summary>
        public IAsyncEnumerable<string> ChatStreamAsync(
            List<ChatMessage> history,
            int maxTokens,
            CancellationToken cancellationToken,
            SamplingConfig? samplingConfig = null,
            List<ToolFunction>? tools = null,
            bool enableThinking = false)
        {
            return ChatStreamAsync(_intrinsicSession, history, maxTokens, cancellationToken, samplingConfig, tools, enableThinking);
        }

        /// <summary>
        /// Stream chat inference tokens using the given <paramref name="session"/>'s
        /// tracked history.
        /// </summary>
        public IAsyncEnumerable<string> ChatStreamAsync(
            ChatSession session,
            List<ChatMessage> history,
            int maxTokens,
            CancellationToken cancellationToken,
            SamplingConfig? samplingConfig = null,
            List<ToolFunction>? tools = null,
            bool enableThinking = false)
        {
            return _generation.ChatStreamAsync(session, history, maxTokens, cancellationToken, samplingConfig, tools, enableThinking);
        }

        /// <summary>
        /// Stream chat inference tokens with timing metrics.
        /// </summary>
        public IAsyncEnumerable<ChatStreamUpdate>
            ChatStreamWithMetricsAsync(
                List<ChatMessage> history,
                int maxTokens,
                CancellationToken cancellationToken,
                SamplingConfig? samplingConfig = null,
                List<ToolFunction>? tools = null,
                bool enableThinking = false)
        {
            return ChatStreamWithMetricsAsync(_intrinsicSession, history, maxTokens, cancellationToken, samplingConfig, tools, enableThinking);
        }

        /// <summary>
        /// Session-aware overload of
        /// <see cref="ChatStreamWithMetricsAsync(List{ChatMessage}, int, CancellationToken, SamplingConfig, List{ToolFunction}, bool)"/>.
        /// </summary>
        public IAsyncEnumerable<ChatStreamUpdate>
            ChatStreamWithMetricsAsync(
                ChatSession session,
                List<ChatMessage> history,
                int maxTokens,
                CancellationToken cancellationToken,
                SamplingConfig? samplingConfig = null,
                List<ToolFunction>? tools = null,
                bool enableThinking = false)
        {
            return _generation.ChatStreamWithMetricsAsync(session, history, maxTokens, cancellationToken, samplingConfig, tools, enableThinking);
        }

        /// <summary>
        /// The loaded model's context length, or 0 when nothing is loaded. Used to size
        /// the Agent Skills prompt block: the injected instructions are budgeted as a
        /// fraction of the context so a large skill cannot crowd out the conversation.
        /// </summary>
        public int ContextTokens => _lifecycle.Model?.MaxContextLength ?? 0;

        /// <summary>
        /// Context window declared by the loaded model artifact, or 0 when nothing is
        /// loaded. This can be larger than <see cref="ContextTokens"/> when a host
        /// applies a smaller runtime limit for memory safety.
        /// </summary>
        public int ModelContextTokens => _lifecycle.Model?.Config?.DeclaredContextLength ?? 0;

        /// <summary>Approve optional knowledge expansion against the real rendered
        /// text prompt, preserving the requested reply budget. Media expansion needs
        /// GPU preparation and therefore stays on the ordinary retrieval path.</summary>
        internal bool KnowledgePromptFits(List<ChatMessage> messages, int maxTokens,
            List<ToolFunction>? tools, bool enableThinking)
        {
            var model = Model;
            if (model == null || model.IsRetiring || model.MaxContextLength <= 0
                || messages.Any(message => message.ImagePaths is { Count: > 0 }
                    || message.AudioPaths is { Count: > 0 })) return false;
            var engine = _engineHost.TryGetEngine();
            if (engine == null || !ReferenceEquals(model, Model)) return false;
            var capacity = engine.PoolStats;
            long contextLimit = Math.Min(model.MaxContextLength, (long)capacity.totalBlocks * capacity.blockSize);
            long promptLimit = contextLimit - Math.Max(1, maxTokens) - 1024;
            if (promptLimit <= 0) return false;
            var renderer = new KVCachePromptRenderer(new GgufPromptRenderer());
            return renderer.RenderToTokens(model.Tokenizer, model.Config.ChatTemplate, messages,
                model.Config.Architecture, addGenerationPrompt: true, tools: tools, enableThinking: enableThinking).Count <= promptLimit;
        }

        /// <summary>
        /// Session-aware chat with Agent Skills.
        ///
        /// <para>
        /// With <paramref name="skills"/> null — or with a plan whose model family
        /// cannot carry tool declarations, so nothing will be fetched — this is exactly
        /// <see cref="ChatStreamWithMetricsAsync(ChatSession, List{ChatMessage}, int, CancellationToken, SamplingConfig, List{ToolFunction}, bool)"/>
        /// and costs nothing. Otherwise the request runs through the
        /// progressive-disclosure loop, which answers the model's <c>skills_read</c>
        /// calls in process and forwards every round's content and reasoning as it
        /// decodes while keeping the tool markup to itself, so an ordinary OpenAI client
        /// streams a normal completion rather than stalling on a tool call it has no
        /// implementation for.
        /// </para>
        /// <para>
        /// The updates it yields in that case carry <see cref="ChatStreamUpdate.IsParsed"/>:
        /// the loop has already run the output parser, and an adapter that runs its own
        /// over them would be parsing parsed text.
        /// </para>
        /// </summary>
        internal IAsyncEnumerable<ChatStreamUpdate>
            ChatStreamWithSkillsAsync(
                ChatSession session,
                List<ChatMessage> history,
                int maxTokens,
                CancellationToken cancellationToken,
                SamplingConfig? samplingConfig,
                List<ToolFunction>? tools,
                bool enableThinking,
                SkillRequestPlan? skills,
                ILogger logger = null,
                int? thinkingTokenLimit = null,
                TextLoraAdapter? lora = null)
        {
            if (skills == null || !skills.ToolsOffered)
            {
                return _generation.ChatStreamWithMetricsAsync(
                    session, history, maxTokens, cancellationToken, samplingConfig, tools, enableThinking,
                    thinkingTokenLimit.HasValue || lora != null
                        ? new ChatTurnContext { ThinkingTokenLimit = thinkingTokenLimit, Adapter = lora }
                        : null);
            }

            // One turn, many generations: every round of the loop runs in the cache scope
            // its first round resolved, so round two continues round one's state even on
            // a shared session where the scope is proved from history. Every round, and
            // every sub-agent's, runs with the turn's LoRA adapter.
            var turn = new ChatTurnContext
            { StagedAttachmentNames = skills.StagedAttachmentNames, ThinkingTokenLimit = thinkingTokenLimit, Adapter = lora };

            // Sampling for a turn that can run code. The runner decides, because it is
            // what knows the operator's configuration; SamplingConfig.ForCodingTurn then
            // changes only values still sitting at their built-in default, so a
            // temperature a client or an operator actually chose is never overruled. The
            // caller's instance is cloned, never mutated — it is the server-wide default
            // and is shared across every request.
            SamplingConfig turnSampling =
                skills.ToolContext?.CodeRunner?.ForCodingTurn(samplingConfig) ?? samplingConfig;
            turnSampling = SamplingForBoundedKnowledgePlanning(Architecture, turnSampling,
                skills.ToolContext?.Knowledge != null, enableThinking, thinkingTokenLimit);

            SkillChatGeneration generate =
                (turnMessages, turnTools, ct) => _generation.ChatStreamWithMetricsAsync(
                    session, turnMessages, maxTokens, ct,
                    SamplingForDeepSeek41SkillRound(Architecture, turnSampling, samplingConfig), turnTools, enableThinking,
                    turn);

            if (skills.MultiAgent is { Enabled: true })
                return MultiAgentChatStreamAsync(history, skills, generate, maxTokens,
                    turnSampling, samplingConfig, enableThinking, logger, turn, cancellationToken);

            return SkillChatLoop.RunAsync(
                Architecture,
                history,
                skills,
                enableThinking,
                generate,
                logger,
                cancellationToken);
        }

        internal static SamplingConfig SamplingForBoundedKnowledgePlanning(string architecture,
            SamplingConfig sampling, bool hasKnowledge, bool enableThinking, int? thinkingTokenLimit)
        {
            if (ChatProtocolRegistry.For(architecture)?.Id != "muse-glimmer"
                || !hasKnowledge || !enableThinking || thinkingTokenLimit is not > 0
                || !string.IsNullOrWhiteSpace(sampling?.ReasoningEffort))
                return sampling;
            // The template's default high effort works against an explicitly
            // bounded retrieval plan. Preserve deliberate effort choices and every
            // other request setting, including an independent grammar constraint.
            var result = sampling?.Clone() ?? new SamplingConfig();
            result.ReasoningEffort = "medium";
            result.Grammar = sampling?.Grammar?.Fork();
            return result;
        }

        internal static SamplingConfig SamplingForDeepSeek41SkillRound(string architecture,
            SamplingConfig turnSampling, SamplingConfig sourceSampling)
        {
            if (ChatProtocolRegistry.For(architecture)?.Id != "deepseek41" || sourceSampling?.Grammar == null)
                return turnSampling;
            // Coding defaults may clone sampling settings, deliberately dropping
            // a live grammar. Each internal tool round must begin with its own
            // untouched protocol constraint, just like an external HTTP round.
            var result = (turnSampling ?? sourceSampling).Clone();
            result.Grammar = sourceSampling.Grammar.Fork();
            return result;
        }

        /// <summary>
        /// Stateless overload for the protocols that do not carry a session
        /// (OpenAI chat, Responses, Ollama).
        /// </summary>
        internal IAsyncEnumerable<ChatStreamUpdate>
            ChatStreamWithSkillsAsync(
                List<ChatMessage> history,
                int maxTokens,
                CancellationToken cancellationToken,
                SamplingConfig? samplingConfig,
                List<ToolFunction>? tools,
                bool enableThinking,
                SkillRequestPlan? skills,
                ILogger logger = null,
                TextLoraAdapter? lora = null)
        {
            return ChatStreamWithSkillsAsync(
                _intrinsicSession, history, maxTokens, cancellationToken,
                samplingConfig, tools, enableThinking, skills, logger, lora: lora);
        }

        /// <summary>True when the loaded model is a DiffusionGemma block-diffusion model, which is
        /// generated via an iterative denoising sampler instead of the autoregressive engine.</summary>
        public virtual bool IsDiffusionModel => _lifecycle.Model is DiffusionGemmaModel;

        /// <summary>Stream a DiffusionGemma chat turn as rich denoising updates (live preview canvases +
        /// final answer + metrics). Used by the Web UI for a live denoising view.</summary>
        internal IAsyncEnumerable<DiffusionStreamUpdate> DiffusionChatStreamAsync(
            ChatSession session,
            List<ChatMessage> history,
            int maxTokens,
            CancellationToken cancellationToken,
            bool enableThinking = false,
            SamplingConfig samplingConfig = null)
        {
            return _generation.DiffusionChatStreamAsync(session, history, maxTokens, cancellationToken, enableThinking, samplingConfig);
        }

        /// <summary>
        /// Stream generate tokens.
        /// Intended for one-shot completions and does not update session history.
        /// </summary>
        public IAsyncEnumerable<ChatStreamUpdate>
            GenerateStreamAsync(
                string prompt,
                List<string>? imagePaths,
                int maxTokens,
                CancellationToken cancellationToken,
                SamplingConfig? samplingConfig = null)
        {
            return GenerateStreamAsync(_intrinsicSession, prompt, imagePaths, maxTokens, cancellationToken, samplingConfig);
        }

        /// <summary>
        /// Session-aware streaming generate. Generate requests are treated as
        /// one-shot prompts and do not update tracked chat history.
        /// </summary>
        public IAsyncEnumerable<ChatStreamUpdate>
            GenerateStreamAsync(
                ChatSession session,
                string prompt,
                List<string>? imagePaths,
                int maxTokens,
                CancellationToken cancellationToken,
                SamplingConfig? samplingConfig = null)
        {
            return _generation.GenerateStreamAsync(session, prompt, imagePaths, maxTokens, cancellationToken, samplingConfig);
        }


        internal static int ResolvePrefillChunkSize(BackendType backend, int tokenCount)
            => PrefillChunking.ResolveChunkSize(backend, tokenCount);


        internal static List<ChatMessage> PrepareHistoryForInference(List<ChatMessage> history, string arch)
            => ChatHistoryPreparer.PrepareHistoryForInference(history, arch);

        internal static List<ChatMessage> PrepareHistoryForInference(List<ChatMessage> history, string arch, ILogger logger)
            => ChatHistoryPreparer.PrepareHistoryForInference(history, arch, logger);

        internal static bool HasMultimodalContent(ChatMessage msg)
            => ChatHistoryPreparer.HasMultimodalContent(msg);

        internal static bool HasMultimodalContent(List<ChatMessage> history)
            => ChatHistoryPreparer.HasMultimodalContent(history);

        internal static List<string> GetImagePathsInPromptOrder(List<ChatMessage> history)
            => ChatHistoryPreparer.GetImagePathsInPromptOrder(history);

        internal static string SerializeMessagesForLog(List<ChatMessage> messages)
            => InferenceTelemetry.SerializeMessagesForLog(messages);

        internal static string SerializeUploadsForLog(ChatMessage message)
            => InferenceTelemetry.SerializeUploadsForLog(message);

        public List<string> ScanModels(string directory)
        {
            if (!Directory.Exists(directory)) return new List<string>();
            return Directory.GetFiles(directory, "*.gguf")
                .Select(Path.GetFileName)
                .Where(f => !IsMmProjFile(f))
                .OrderBy(f => f)
                .ToList();
        }

        public List<string> ScanMmProjModels(string directory)
        {
            if (!Directory.Exists(directory)) return new List<string>();
            return Directory.GetFiles(directory, "*.gguf")
                .Select(Path.GetFileName)
                .Where(IsMmProjFile)
                .OrderBy(f => f)
                .ToList();
        }

        public void Dispose()
        {
            using var jevLease = _jevExecution.BeginChange(shutdown: true);
            _engineHost.Dispose();
            _generation.Dispose();
            _lifecycle.Dispose();
            _intrinsicSession.Dispose();
        }

        private static bool IsMmProjFile(string fileName)
        {
            return fileName.IndexOf("mmproj", StringComparison.OrdinalIgnoreCase) >= 0;
        }
    }
}
