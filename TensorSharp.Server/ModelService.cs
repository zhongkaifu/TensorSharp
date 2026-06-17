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

namespace TensorSharp.Server
{
    public class ModelService : IDisposable
    {
        private readonly ModelLifecycleService _lifecycle;
        private readonly ChatSession _intrinsicSession;
        private readonly InferenceEngineHost _engineHost;
        private readonly ChatGenerationPipeline _generation;

        public ModelService()
            : this(NullLogger<ModelService>.Instance)
        {
        }

        public ModelService(ILogger<ModelService> logger)
        {
            logger ??= NullLogger<ModelService>.Instance;

            var promptRenderer = new GgufPromptRenderer();
            var kvCacheRenderer = new KVCachePromptRenderer(promptRenderer);
            var telemetry = new InferenceTelemetry(logger);

            _lifecycle = new ModelLifecycleService(logger);
            _intrinsicSession = new ChatSession("__svc_intrinsic__");
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
        public string Architecture => _lifecycle.Architecture;
        public ModelBase Model => _lifecycle.Model;

        /// <summary>
        /// Non-null when an explicitly requested <c>--mtp-draft-model</c> could not
        /// be activated on the loaded target (see
        /// <see cref="ModelLifecycleService.MtpDraftActivationError"/>). The startup
        /// loader promotes it to a fail-fast error.
        /// </summary>
        public string MtpDraftActivationError => _lifecycle.MtpDraftActivationError;

        /// <summary>
        /// Legacy compatibility shim. The engine owns KV state, so no server
        /// session is ever active in the model.
        /// </summary>
        public ChatSession ActiveSession => null;

        /// <summary>
        /// Legacy compatibility shim. Server-side session KV bookkeeping was
        /// removed; callers receive an isolated empty cache that is never used
        /// by inference.
        /// </summary>
        public KVCache KVCache => new();

        /// <summary>
        /// Snapshot of the intrinsic compatibility session's tracked history.
        /// Session-aware requests use the explicit <see cref="ChatSession"/>
        /// instance passed to the generation methods.
        /// </summary>
        public IReadOnlyList<ChatMessage> TrackedHistory => _intrinsicSession.TrackedHistory.AsReadOnly();

        public bool IsModelAlreadyLoaded(string modelName)
        {
            return _lifecycle.IsModelAlreadyLoaded(modelName);
        }

        /// <summary>Engine host exposed for adapters that submit requests
        /// directly to the engine (e.g. multi-turn streaming clients that
        /// want to manage their own session bookkeeping).</summary>
        public InferenceEngineHost EngineHost => _engineHost;

        public void LoadModel(string modelPath, string mmProjPath, string backendStr)
        {
            // Tear down the per-model engine and the diffusion batch scheduler BEFORE the model is
            // unloaded so their worker threads don't race the model disposal.
            _engineHost.Reset();
            _generation.ResetDiffusionScheduler();
            _intrinsicSession.TrackedHistory.Clear();
            _lifecycle.LoadModel(modelPath, mmProjPath, backendStr);
        }

        /// <summary>
        /// Legacy compatibility shim for older callers. There is no
        /// service-owned KV cache to invalidate; this clears only the intrinsic
        /// tracked history used by non-session-aware overloads.
        /// </summary>
        public void InvalidateKVCache()
        {
            _intrinsicSession.TrackedHistory.Clear();
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
            // TrackedHistory while we clear it.
            lock (session.HistoryLock)
                session.TrackedHistory.Clear();
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
        /// Stream chat inference tokens. Must be called within the InferenceQueue to prevent concurrent access.
        /// </summary>
        public IAsyncEnumerable<string> ChatStreamAsync(
            List<ChatMessage> history,
            int maxTokens,
            CancellationToken cancellationToken,
            SamplingConfig samplingConfig = null,
            List<ToolFunction> tools = null,
            bool enableThinking = false)
        {
            return ChatStreamAsync(_intrinsicSession, history, maxTokens, cancellationToken, samplingConfig, tools, enableThinking);
        }

        /// <summary>
        /// Stream chat inference tokens using the given <paramref name="session"/>'s
        /// tracked history. Must be called within the InferenceQueue.
        /// </summary>
        public IAsyncEnumerable<string> ChatStreamAsync(
            ChatSession session,
            List<ChatMessage> history,
            int maxTokens,
            CancellationToken cancellationToken,
            SamplingConfig samplingConfig = null,
            List<ToolFunction> tools = null,
            bool enableThinking = false)
        {
            return _generation.ChatStreamAsync(session, history, maxTokens, cancellationToken, samplingConfig, tools, enableThinking);
        }

        /// <summary>
        /// Stream chat inference tokens with timing metrics. Must be called within the InferenceQueue.
        /// </summary>
        public IAsyncEnumerable<(string piece, bool done, int promptTokens, int evalTokens, int kvCacheReusedTokens, long totalNs, long promptNs, long evalNs)>
            ChatStreamWithMetricsAsync(
                List<ChatMessage> history,
                int maxTokens,
                CancellationToken cancellationToken,
                SamplingConfig samplingConfig = null,
                List<ToolFunction> tools = null,
                bool enableThinking = false)
        {
            return ChatStreamWithMetricsAsync(_intrinsicSession, history, maxTokens, cancellationToken, samplingConfig, tools, enableThinking);
        }

        /// <summary>
        /// Session-aware overload of
        /// <see cref="ChatStreamWithMetricsAsync(List{ChatMessage}, int, CancellationToken, SamplingConfig, List{ToolFunction}, bool)"/>.
        /// </summary>
        public IAsyncEnumerable<(string piece, bool done, int promptTokens, int evalTokens, int kvCacheReusedTokens, long totalNs, long promptNs, long evalNs)>
            ChatStreamWithMetricsAsync(
                ChatSession session,
                List<ChatMessage> history,
                int maxTokens,
                CancellationToken cancellationToken,
                SamplingConfig samplingConfig = null,
                List<ToolFunction> tools = null,
                bool enableThinking = false)
        {
            return _generation.ChatStreamWithMetricsAsync(session, history, maxTokens, cancellationToken, samplingConfig, tools, enableThinking);
        }

        /// <summary>True when the loaded model is a DiffusionGemma block-diffusion model, which is
        /// generated via an iterative denoising sampler instead of the autoregressive engine.</summary>
        public bool IsDiffusionModel => _lifecycle.Model is DiffusionGemmaModel;

        /// <summary>Stream a DiffusionGemma chat turn as rich denoising updates (live preview canvases +
        /// final answer + metrics). Used by the Web UI for a live denoising view. Must be called within
        /// the InferenceQueue.</summary>
        internal IAsyncEnumerable<DiffusionStreamUpdate> DiffusionChatStreamAsync(
            ChatSession session,
            List<ChatMessage> history,
            int maxTokens,
            CancellationToken cancellationToken)
        {
            return _generation.DiffusionChatStreamAsync(session, history, maxTokens, cancellationToken);
        }

        /// <summary>
        /// Stream generate tokens. Must be called within the InferenceQueue to prevent concurrent access.
        /// Intended for one-shot completions and does not update session history.
        /// </summary>
        public IAsyncEnumerable<(string piece, bool done, int promptTokens, int evalTokens, int kvCacheReusedTokens, long totalNs, long promptNs, long evalNs)>
            GenerateStreamAsync(
                string prompt,
                List<string> imagePaths,
                int maxTokens,
                CancellationToken cancellationToken,
                SamplingConfig samplingConfig = null)
        {
            return GenerateStreamAsync(_intrinsicSession, prompt, imagePaths, maxTokens, cancellationToken, samplingConfig);
        }

        /// <summary>
        /// Session-aware streaming generate. Generate requests are treated as
        /// one-shot prompts and do not update tracked chat history.
        /// </summary>
        public IAsyncEnumerable<(string piece, bool done, int promptTokens, int evalTokens, int kvCacheReusedTokens, long totalNs, long promptNs, long evalNs)>
            GenerateStreamAsync(
                ChatSession session,
                string prompt,
                List<string> imagePaths,
                int maxTokens,
                CancellationToken cancellationToken,
                SamplingConfig samplingConfig = null)
        {
            return _generation.GenerateStreamAsync(session, prompt, imagePaths, maxTokens, cancellationToken, samplingConfig);
        }

        /// <summary>
        /// Instance-friendly shim that augments against the intrinsic compatibility
        /// session's tracked history. Prefer the static overload that takes an
        /// explicit tracked history for deterministic testing.
        /// </summary>
        internal List<ChatMessage> AugmentWithCachedRawTokens(List<ChatMessage> incoming)
        {
            return AugmentWithCachedRawTokens(incoming, _intrinsicSession.TrackedHistory);
        }

        internal static int ResolvePrefillChunkSize(BackendType backend, int tokenCount)
            => PrefillChunking.ResolveChunkSize(backend, tokenCount);

        internal static List<ChatMessage> AugmentWithCachedRawTokens(
            List<ChatMessage> incoming,
            IReadOnlyList<ChatMessage> trackedHistory)
            => ChatHistoryPreparer.AugmentWithCachedRawTokens(incoming, trackedHistory);

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
