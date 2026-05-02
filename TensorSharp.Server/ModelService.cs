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
        private readonly SessionKvCacheManager _sessions;
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
            _sessions = new SessionKvCacheManager(_lifecycle, logger);
            _generation = new ChatGenerationPipeline(_lifecycle, _sessions, kvCacheRenderer, telemetry, logger);
        }

        public bool IsLoaded => _lifecycle.IsLoaded;
        public string LoadedModelName => _lifecycle.LoadedModelName;
        public string LoadedModelPath => _lifecycle.LoadedModelPath;
        public string LoadedMmProjName => _lifecycle.LoadedMmProjName;
        public string LoadedMmProjPath => _lifecycle.LoadedMmProjPath;
        public string LoadedBackend => _lifecycle.LoadedBackend;
        public string Architecture => _lifecycle.Architecture;
        public ModelBase Model => _lifecycle.Model;

        /// <summary>
        /// The session whose tokens are currently held in the model's KV tensors, or
        /// null if no session has been activated yet since the last model load / reset.
        /// </summary>
        public ChatSession ActiveSession => _sessions.ActiveSession;

        /// <summary>
        /// Inspection-only view of the active session's KV cache bookkeeping (for tests
        /// and diagnostics). Returns an empty cache when no session is active.
        /// </summary>
        public KVCache KVCache => _sessions.KVCache;

        /// <summary>
        /// Snapshot of the messages whose tokens are reflected in the current (active)
        /// session's KV state. Returned as a read-only view.
        /// </summary>
        public IReadOnlyList<ChatMessage> TrackedHistory => _sessions.TrackedHistory;

        public bool IsModelAlreadyLoaded(string modelName)
        {
            return _lifecycle.IsModelAlreadyLoaded(modelName);
        }

        public void LoadModel(string modelPath, string mmProjPath, string backendStr)
        {
            _sessions.ClearForModelReload();
            _lifecycle.LoadModel(modelPath, mmProjPath, backendStr);
        }

        /// <summary>
        /// Clear the active session's conversation cache and reset the model's K/V
        /// tensors. Callers that hold a specific session should prefer
        /// <see cref="ResetSession"/>.
        /// </summary>
        public void InvalidateKVCache()
        {
            _sessions.InvalidateKVCache();
        }

        /// <summary>
        /// Reset the given session's conversation cache. If the session is currently
        /// active in the model, the model's K/V tensors are also reset.
        /// </summary>
        public void ResetSession(ChatSession session)
        {
            _sessions.ResetSession(session);
        }

        /// <summary>
        /// Dispose the given session and release any KV state it held. When the
        /// session was active in the model, the model's K/V tensors are reset so
        /// no data leaks to whichever session is activated next.
        /// </summary>
        public void DisposeSession(ChatSession session)
        {
            _sessions.DisposeSession(session);
        }

        /// <summary>
        /// Make <paramref name="session"/> the active session. Switching sessions resets
        /// the model's K/V tensors so no cached data leaks across sessions.
        /// </summary>
        internal void ActivateSession(ChatSession session)
        {
            _sessions.ActivateSession(session);
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
            return ChatStreamAsync(_sessions.IntrinsicSession, history, maxTokens, cancellationToken, samplingConfig, tools, enableThinking);
        }

        /// <summary>
        /// Stream chat inference tokens using the given <paramref name="session"/>'s
        /// KV cache and tracked history. Must be called within the InferenceQueue.
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
            return ChatStreamWithMetricsAsync(_sessions.IntrinsicSession, history, maxTokens, cancellationToken, samplingConfig, tools, enableThinking);
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

        /// <summary>
        /// Stream generate tokens. Must be called within the InferenceQueue to prevent concurrent access.
        /// Always resets the session's KV cache - intended for one-shot completions.
        /// </summary>
        public IAsyncEnumerable<(string piece, bool done, int promptTokens, int evalTokens, int kvCacheReusedTokens, long totalNs, long promptNs, long evalNs)>
            GenerateStreamAsync(
                string prompt,
                List<string> imagePaths,
                int maxTokens,
                CancellationToken cancellationToken,
                SamplingConfig samplingConfig = null)
        {
            return GenerateStreamAsync(_sessions.IntrinsicSession, prompt, imagePaths, maxTokens, cancellationToken, samplingConfig);
        }

        /// <summary>
        /// Session-aware streaming generate. The session's KV cache is reset before the
        /// prefill, so the yielded done tuple always reports <c>kvCacheReusedTokens == 0</c>.
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
        /// Instance-friendly shim that augments against the active session's tracked
        /// history. Prefer the static overload that takes an explicit tracked history
        /// for deterministic testing.
        /// </summary>
        internal List<ChatMessage> AugmentWithCachedRawTokens(List<ChatMessage> incoming)
        {
            return AugmentWithCachedRawTokens(incoming, (_sessions.ActiveSession ?? _sessions.IntrinsicSession).TrackedHistory);
        }

        internal static int ResolvePrefillChunkSize(BackendType backend, int tokenCount)
            => SessionKvCacheManager.ResolvePrefillChunkSize(backend, tokenCount);

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
            _lifecycle.Dispose();
            _sessions.Dispose();
        }

        private static bool IsMmProjFile(string fileName)
        {
            return fileName.IndexOf("mmproj", StringComparison.OrdinalIgnoreCase) >= 0;
        }
    }
}
