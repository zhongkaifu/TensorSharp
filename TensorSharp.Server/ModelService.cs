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
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Encodings.Web;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Runtime.Logging;

namespace TensorSharp.Server
{
    public class ModelService : IDisposable
    {
        private readonly IPromptRenderer _promptRenderer = new GgufPromptRenderer();
        private readonly KVCachePromptRenderer _kvCacheRenderer;
        private readonly ILogger<ModelService> _logger;

        private ModelBase _model;
        private string _loadedModelPath;
        private string _loadedMmProjPath;
        private BackendType _backend;

        // Session whose tokens are currently reflected in the model's per-layer K/V
        // tensors. Only one session at a time can be "active"; switching the active
        // session forces a full reset so no cached K/V data leaks across sessions.
        private ChatSession _activeSession;

        // Built-in fallback session for code paths that are not session-aware (e.g.
        // existing tests or any caller that doesn't plumb a session through). All HTTP
        // endpoints go through SessionManager and pass their own session explicitly.
        private readonly ChatSession _intrinsicSession = new ChatSession("__svc_intrinsic__");

        // Reused JSON options for the per-turn fullInput serializer below. Relaxed
        // escaping keeps non-ASCII content (e.g. Chinese, emoji) readable in the log
        // file instead of expanding it to \\uXXXX escapes; control characters are
        // still escaped by JsonSerializer so each entry stays on a single line.
        private static readonly JsonSerializerOptions FullInputJsonOptions = new()
        {
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
            Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
        };

        public ModelService()
            : this(NullLogger<ModelService>.Instance)
        {
        }

        public ModelService(ILogger<ModelService> logger)
        {
            _logger = logger ?? NullLogger<ModelService>.Instance;
            _kvCacheRenderer = new KVCachePromptRenderer(_promptRenderer);
        }

        public bool IsLoaded => _model != null;
        public string LoadedModelName => _loadedModelPath != null ? Path.GetFileName(_loadedModelPath) : null;
        public string LoadedModelPath => _loadedModelPath;
        public string LoadedMmProjName => _loadedMmProjPath != null ? Path.GetFileName(_loadedMmProjPath) : null;
        public string LoadedMmProjPath => _loadedMmProjPath;
        public string LoadedBackend => _model != null ? BackendCatalog.ToBackendValue(_backend) : null;
        public string Architecture => _model?.Config?.Architecture;
        public ModelBase Model => _model;

        /// <summary>
        /// The session whose tokens are currently held in the model's KV tensors, or
        /// null if no session has been activated yet since the last model load / reset.
        /// </summary>
        public ChatSession ActiveSession => _activeSession;

        /// <summary>
        /// Inspection-only view of the active session's KV cache bookkeeping (for tests
        /// and diagnostics). Returns an empty cache when no session is active.
        /// </summary>
        public KVCache KVCache => _activeSession?.KVCache ?? _intrinsicSession.KVCache;

        /// <summary>
        /// Snapshot of the messages whose tokens are reflected in the current (active)
        /// session's KV state. Returned as a read-only view.
        /// </summary>
        public IReadOnlyList<ChatMessage> TrackedHistory =>
            (_activeSession ?? _intrinsicSession).TrackedHistory.AsReadOnly();

        public bool IsModelAlreadyLoaded(string modelName)
        {
            return _model != null && string.Equals(LoadedModelName, modelName, StringComparison.OrdinalIgnoreCase);
        }

        /// <summary>
        /// Clear the active session's conversation cache and reset the model's K/V
        /// tensors. Callers that hold a specific session should prefer
        /// <see cref="ResetSession"/>.
        /// </summary>
        public void InvalidateKVCache()
        {
            ResetSession(_activeSession ?? _intrinsicSession);
        }

        /// <summary>
        /// Reset the given session's conversation cache. If the session is currently
        /// active in the model, the model's K/V tensors are also reset.
        /// </summary>
        public void ResetSession(ChatSession session)
        {
            if (session == null)
                return;

            session.TrackedHistory.Clear();
            session.KVCache.Reset();
            if (ReferenceEquals(session, _activeSession))
                _model?.ResetKVCache();
        }

        /// <summary>
        /// Dispose the given session and release any KV state it held. When the
        /// session was active in the model, the model's K/V tensors are reset so
        /// no data leaks to whichever session is activated next.
        /// </summary>
        public void DisposeSession(ChatSession session)
        {
            if (session == null || session.IsDisposed)
                return;

            bool wasActive = ReferenceEquals(session, _activeSession);
            session.Dispose();
            if (wasActive)
            {
                _activeSession = null;
                _model?.ResetKVCache();
            }
        }

        /// <summary>
        /// Make <paramref name="session"/> the active session. When the active
        /// session changes, the model's K/V tensors are reset and the previously
        /// active session's KV bookkeeping is cleared (we cannot preserve its
        /// tensors). On the next use of that old session it will simply re-prefill
        /// from its tracked history.
        /// </summary>
        internal void ActivateSession(ChatSession session)
        {
            if (session == null)
                throw new ArgumentNullException(nameof(session));
            if (session.IsDisposed)
                throw new ObjectDisposedException(nameof(ChatSession), $"Session {session.Id} has been disposed.");

            if (ReferenceEquals(session, _activeSession))
            {
                session.LastUsedAt = DateTime.UtcNow;
                return;
            }

            string previousSessionId = _activeSession?.Id;
            if (_activeSession != null)
            {
                // Old session no longer owns the model's tensors; drop its token
                // bookkeeping so a future ActivateSession(old) rebuilds from scratch.
                _activeSession.KVCache.Reset();
            }

            _model?.ResetKVCache();
            session.KVCache.Reset();
            _activeSession = session;
            session.LastUsedAt = DateTime.UtcNow;

            _logger.LogDebug(LogEventIds.SessionActivated,
                "Activated session {SessionId} (previousSession={PreviousSessionId})",
                session.Id, previousSessionId ?? "(none)");
        }

        public void LoadModel(string modelPath, string mmProjPath, string backendStr)
        {
            _logger.LogInformation(LogEventIds.ModelLoadStarted,
                "Loading model {ModelFile} (mmproj={MmProjFile}, backend={Backend}, fullPath={ModelPath}, mmprojPath={MmProjPath})",
                Path.GetFileName(modelPath), Path.GetFileName(mmProjPath ?? string.Empty),
                backendStr ?? "(default)", modelPath, mmProjPath ?? "(none)");

            string previousModel = LoadedModelName;
            _model?.Dispose();
            _model = null;
            _loadedModelPath = null;
            _loadedMmProjPath = null;
            _activeSession = null;
            _intrinsicSession.TrackedHistory.Clear();
            _intrinsicSession.KVCache.Reset();

            if (!string.IsNullOrEmpty(previousModel))
            {
                _logger.LogInformation(LogEventIds.ModelUnloaded,
                    "Unloaded previous model {PreviousModel}", previousModel);
            }

            _backend = BackendCatalog.Canonicalize(backendStr) switch
            {
                "ggml_metal" => BackendType.GgmlMetal,
                "ggml_cpu" => BackendType.GgmlCpu,
                "ggml_cuda" => BackendType.GgmlCuda,
                "cpu" => BackendType.Cpu,
                _ => BackendType.GgmlCpu
            };

            var loadSw = Stopwatch.StartNew();
            try
            {
                _model = ModelBase.Create(modelPath, _backend);
                _loadedModelPath = modelPath;

                if (!string.IsNullOrEmpty(mmProjPath) && File.Exists(mmProjPath))
                {
                    LoadEncoders(mmProjPath);
                    _loadedMmProjPath = mmProjPath;
                }

                loadSw.Stop();
                long modelBytes = SafeGetFileSize(modelPath);
                long mmProjBytes = SafeGetFileSize(mmProjPath);
                _logger.LogInformation(LogEventIds.ModelLoadCompleted,
                    "Loaded model {Model} (architecture={Architecture}, backend={Backend}, modelBytes={ModelBytes}, mmproj={MmProjFile}, mmprojBytes={MmProjBytes}) in {ElapsedMs:F1} ms",
                    LoadedModelName, Architecture ?? "(unknown)", LoadedBackend ?? "(unknown)",
                    modelBytes, LoadedMmProjName ?? "(none)", mmProjBytes, loadSw.Elapsed.TotalMilliseconds);
            }
            catch (Exception ex)
            {
                loadSw.Stop();
                _logger.LogError(LogEventIds.ModelLoadFailed, ex,
                    "Failed to load model {ModelFile} on backend {Backend} after {ElapsedMs:F1} ms",
                    Path.GetFileName(modelPath), backendStr ?? "(default)", loadSw.Elapsed.TotalMilliseconds);
                throw;
            }
        }

        private static long SafeGetFileSize(string path)
        {
            if (string.IsNullOrEmpty(path))
                return 0;
            try
            {
                var fi = new FileInfo(path);
                return fi.Exists ? fi.Length : 0;
            }
            catch
            {
                return 0;
            }
        }

        private void LoadEncoders(string mmProjPath)
        {
            _model?.MultimodalInjector.LoadProjectors(mmProjPath);
        }

        /// <summary>
        /// Stream chat inference tokens. Must be called within the InferenceQueue to prevent concurrent access.
        /// Reuses the KV cache from the previous turn by splicing raw output tokens directly into the
        /// rendered prompt - which guarantees that the cached prefix matches exactly across turns.
        /// </summary>
        public IAsyncEnumerable<string> ChatStreamAsync(
            List<ChatMessage> history,
            int maxTokens,
            CancellationToken cancellationToken,
            SamplingConfig samplingConfig = null,
            List<ToolFunction> tools = null, bool enableThinking = false)
        {
            return ChatStreamAsync(_intrinsicSession, history, maxTokens, cancellationToken, samplingConfig, tools, enableThinking);
        }

        /// <summary>
        /// Stream chat inference tokens using the given <paramref name="session"/>'s
        /// KV cache and tracked history. Must be called within the InferenceQueue to
        /// prevent concurrent access. Switching sessions resets the model's K/V state.
        /// </summary>
        public async IAsyncEnumerable<string> ChatStreamAsync(
            ChatSession session,
            List<ChatMessage> history,
            int maxTokens,
            [EnumeratorCancellation] CancellationToken cancellationToken,
            SamplingConfig samplingConfig = null,
            List<ToolFunction> tools = null, bool enableThinking = false)
        {
            await foreach (var (piece, _, _, _, _, _, _, _) in
                ChatStreamInternalAsync(session, history, maxTokens, cancellationToken, samplingConfig, tools, enableThinking, withMetrics: false))
            {
                if (!string.IsNullOrEmpty(piece))
                    yield return piece;
            }
        }

        /// <summary>
        /// Stream chat inference tokens with timing metrics. Must be called within the InferenceQueue.
        /// Reuses the KV cache from the previous turn when the rendered text prefix matches.
        /// The yielded done tuple's <c>kvCacheReusedTokens</c> reports how many of the
        /// <c>promptTokens</c> were served straight from the prior turn's KV cache (an
        /// exact-match turn yields <c>kvCacheReusedTokens == promptTokens</c>; a full
        /// reset yields zero).
        /// </summary>
        public IAsyncEnumerable<(string piece, bool done, int promptTokens, int evalTokens, int kvCacheReusedTokens, long totalNs, long promptNs, long evalNs)>
            ChatStreamWithMetricsAsync(
                List<ChatMessage> history,
                int maxTokens,
                CancellationToken cancellationToken,
                SamplingConfig samplingConfig = null,
                List<ToolFunction> tools = null, bool enableThinking = false)
        {
            return ChatStreamInternalAsync(_intrinsicSession, history, maxTokens, cancellationToken, samplingConfig, tools, enableThinking, withMetrics: true);
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
                List<ToolFunction> tools = null, bool enableThinking = false)
        {
            return ChatStreamInternalAsync(session, history, maxTokens, cancellationToken, samplingConfig, tools, enableThinking, withMetrics: true);
        }

        private async IAsyncEnumerable<(string piece, bool done, int promptTokens, int evalTokens, int kvCacheReusedTokens, long totalNs, long promptNs, long evalNs)>
            ChatStreamInternalAsync(
                ChatSession session,
                List<ChatMessage> history,
                int maxTokens,
                [EnumeratorCancellation] CancellationToken cancellationToken,
                SamplingConfig samplingConfig,
                List<ToolFunction> tools,
                bool enableThinking,
                bool withMetrics)
        {
            ActivateSession(session ?? _intrinsicSession);
            var activeSession = _activeSession;

            string arch = _model.Config.Architecture;
            var preparedHistory = PrepareHistoryForInference(history, arch, _logger);

            // Project the incoming user-visible history onto our tracked conversation so
            // that any assistant messages we previously generated carry their raw output
            // tokens forward into the renderer.
            var renderHistory = AugmentWithCachedRawTokens(preparedHistory, activeSession.TrackedHistory);

            using var chatScope = _logger.BeginScope(new Dictionary<string, object>(StringComparer.Ordinal)
            {
                [LogScopeKeys.SessionId] = activeSession.Id,
                [LogScopeKeys.Model] = LoadedModelName ?? "(none)",
                [LogScopeKeys.Backend] = LoadedBackend ?? "(none)",
                [LogScopeKeys.Operation] = "chat.stream",
            });

            int userMessageCount = 0;
            int assistantMessageCount = 0;
            int systemMessageCount = 0;
            int imageAttachments = 0;
            int audioAttachments = 0;
            int textFileAttachments = 0;
            ChatMessage lastUserMessage = null;
            if (preparedHistory != null)
            {
                foreach (var m in preparedHistory)
                {
                    if (m == null) continue;
                    if (m.Role == "user")
                    {
                        userMessageCount++;
                        lastUserMessage = m;
                    }
                    else if (m.Role == "assistant") assistantMessageCount++;
                    else if (m.Role == "system") systemMessageCount++;
                    if (m.ImagePaths != null) imageAttachments += m.ImagePaths.Count;
                    if (m.AudioPaths != null) audioAttachments += m.AudioPaths.Count;
                    if (m.TextFilePaths != null) textFileAttachments += m.TextFilePaths.Count;
                }
            }

            // Log the full last user message (control chars escaped, no truncation)
            // so the audit log carries the exact content the model was asked about.
            string lastUserContent = LoggingExtensions.SanitizeForLogFull(lastUserMessage?.Content ?? string.Empty);

            // Per-turn upload manifest: the file paths + filenames + media-type of every
            // attachment the user just sent on this turn. Logged separately from
            // fullInput so operators can see at a glance which uploaded files belong to
            // the current request without parsing the whole conversation JSON.
            string turnUploads = SerializeUploadsForLog(lastUserMessage);

            // Capture EVERY message the caller submitted for this turn (system prompts +
            // every prior user/assistant turn + the new user message + attachment paths)
            // as a single JSON-encoded line. With this we can reproduce the exact prompt
            // the model saw from the log alone - including which uploaded files were
            // attached to each historical turn.
            string fullInput = SerializeMessagesForLog(preparedHistory);

            _logger.LogInformation(LogEventIds.ChatStarted,
                "chat.start arch={Architecture} maxTokens={MaxTokens} thinking={EnableThinking} tools={ToolCount} messages(user={UserMessages},assistant={AssistantMessages},system={SystemMessages}) attachments(image={ImageCount},audio={AudioCount},textFile={TextFileCount}) uploads={Uploads} sampling(temp={Temperature},topK={TopK},topP={TopP},minP={MinP},seed={Seed}) userInput=\"{LastUserContent}\" fullInput={FullInput}",
                arch, maxTokens, enableThinking, tools?.Count ?? 0,
                userMessageCount, assistantMessageCount, systemMessageCount,
                imageAttachments, audioAttachments, textFileAttachments,
                turnUploads,
                samplingConfig?.Temperature ?? 0.8f, samplingConfig?.TopK ?? 40,
                samplingConfig?.TopP ?? 0.9f, samplingConfig?.MinP ?? 0f, samplingConfig?.Seed ?? 0,
                lastUserContent, fullInput);

            var inputTokens = _kvCacheRenderer.RenderToTokens(
                _model.Tokenizer,
                _model.Config.ChatTemplate,
                renderHistory,
                arch,
                addGenerationPrompt: true,
                tools: tools,
                enableThinking: enableThinking);

            inputTokens = _model.MultimodalInjector.ProcessPromptTokens(renderHistory, inputTokens);
            inputTokens = TruncatePromptToContext(activeSession, inputTokens, maxTokens);

            int promptTokenCount = inputTokens.Count;
            var sw = Stopwatch.StartNew();
            float[] logits = PrepareForGeneration(activeSession, inputTokens, out int kvCacheReusedTokens);
            long promptNs = ToNanos(sw.ElapsedTicks);
            double kvCacheReusePercent = promptTokenCount > 0
                ? 100.0 * kvCacheReusedTokens / promptTokenCount
                : 0.0;

            var generatedTokens = new List<int>();
            var cfg = samplingConfig ?? SamplingConfig.Default;
            var sampler = new TokenSampler(cfg);
            var rawBytes = new List<byte>();
            int prevCharLen = 0;

            var evalSw = Stopwatch.StartNew();
            bool firstTokenSampled = false;
            string finishReason = "max_tokens";
            long timeToFirstTokenMs = 0;
            bool wasCancelled = false;

            for (int step = 0; step < maxTokens; step++)
            {
                if (cancellationToken.IsCancellationRequested)
                {
                    wasCancelled = true;
                    finishReason = "cancelled";
                    break;
                }

                int nextToken = sampler.Sample(logits, generatedTokens);
                if (_model.Tokenizer.IsEos(nextToken))
                {
                    finishReason = "eos";
                    break;
                }

                generatedTokens.Add(nextToken);
                _model.Tokenizer.AppendTokenBytes(nextToken, rawBytes);
                int validLen = FindValidUtf8Length(rawBytes);
                string decoded = Encoding.UTF8.GetString(rawBytes.GetRange(0, validLen).ToArray());
                string piece = prevCharLen < decoded.Length ? decoded.Substring(prevCharLen) : "";
                prevCharLen = decoded.Length;

                bool stopRequested = false;
                if (cfg.StopSequences != null && cfg.StopSequences.Count > 0)
                {
                    var (_, shouldStop) = sampler.CheckStopSequences(decoded);
                    if (shouldStop)
                    {
                        stopRequested = true;
                        finishReason = "stop_sequence";
                    }
                }

                if (!firstTokenSampled)
                {
                    firstTokenSampled = true;
                    timeToFirstTokenMs = (long)evalSw.Elapsed.TotalMilliseconds;
                }

                if (piece.Length > 0)
                    yield return (piece, false, 0, 0, 0, 0, 0, 0);

                if (stopRequested)
                    break;

                logits = _model.Forward(new[] { nextToken });
                activeSession.KVCache.RecordAppend(nextToken, logits);
            }

            string assistantText = Encoding.UTF8.GetString(rawBytes.ToArray());
            evalSw.Stop();
            sw.Stop();

            // Use the AUGMENTED history (which has raw tokens spliced into prior assistant
            // turns from our previous tracked history) so that the raw tokens carry forward
            // for ALL past assistants, not just the immediately-previous one. The plain
            // user-visible `history` doesn't carry raw tokens (the WebUI never sees them),
            // so cloning from `history` here would cause the cache to silently re-reset
            // every other turn as the raw tokens of older assistants fell off the tracked
            // record.
            UpdateTrackedHistory(activeSession.TrackedHistory, renderHistory, assistantText, generatedTokens);

            double evalSeconds = evalSw.Elapsed.TotalSeconds;
            double tokensPerSecond = (evalSeconds > 0 && generatedTokens.Count > 0)
                ? generatedTokens.Count / evalSeconds
                : 0;
            // assistantText is the full RAW model output (decoded straight from the
            // generated tokens), so it includes any inline reasoning markers such as
            // <think>...</think> or <|channel|> framing. By logging it without a
            // truncation cap we capture both the reasoning trace and the final
            // user-visible result in a single line.
            string assistantContent = LoggingExtensions.SanitizeForLogFull(assistantText);

            if (wasCancelled)
            {
                _logger.LogWarning(LogEventIds.ChatAborted,
                    "chat.cancelled tokens={Tokens} promptTokens={PromptTokens} kvReused={KvReusedTokens} kvReusePercent={KvReusePercent:F1} ttftMs={TimeToFirstTokenMs} elapsedMs={ElapsedMs:F1} assistantOutput=\"{AssistantContent}\"",
                    generatedTokens.Count, promptTokenCount, kvCacheReusedTokens, kvCacheReusePercent,
                    timeToFirstTokenMs, sw.Elapsed.TotalMilliseconds, assistantContent);
            }
            else
            {
                _logger.LogInformation(LogEventIds.ChatCompleted,
                    "chat.complete tokens={Tokens} promptTokens={PromptTokens} kvReused={KvReusedTokens} kvReusePercent={KvReusePercent:F1} ttftMs={TimeToFirstTokenMs} elapsedMs={ElapsedMs:F1} tokensPerSec={TokensPerSec:F2} finishReason={FinishReason} assistantOutput=\"{AssistantContent}\"",
                    generatedTokens.Count, promptTokenCount, kvCacheReusedTokens, kvCacheReusePercent,
                    timeToFirstTokenMs, sw.Elapsed.TotalMilliseconds, tokensPerSecond, finishReason, assistantContent);
            }

            if (withMetrics)
            {
                long evalNs = ToNanos(evalSw.ElapsedTicks);
                long totalNs = ToNanos(sw.ElapsedTicks);
                yield return ("", true, promptTokenCount, generatedTokens.Count, kvCacheReusedTokens, totalNs, promptNs, evalNs);
            }
        }

        /// <summary>
        /// Instance-friendly shim that augments against the active session's tracked
        /// history. Prefer the static overload that takes an explicit tracked history
        /// for deterministic testing.
        /// </summary>
        internal List<ChatMessage> AugmentWithCachedRawTokens(List<ChatMessage> incoming)
        {
            return AugmentWithCachedRawTokens(incoming, (_activeSession ?? _intrinsicSession).TrackedHistory);
        }

        /// <summary>
        /// Walks <paramref name="incoming"/> alongside <paramref name="trackedHistory"/>
        /// to produce a render-ready history with cached raw output tokens spliced
        /// into assistant messages.
        ///
        /// We CAN'T compare assistant <see cref="ChatMessage.Content"/> directly between the
        /// incoming history and the tracked one: the streaming output parser used by the
        /// HTTP layer (for thinking / Harmony / channel-based architectures) strips
        /// <c>&lt;think&gt;</c> / <c>&lt;|channel|&gt;</c> framing out of the text the UI
        /// receives, so the UI sends back a SHORTER/parsed version of the content while
        /// our tracked entry still has the full raw text the model emitted. A naive
        /// "same content?" check would therefore reject the splice for every cached turn
        /// and force a full reset on every request - the exact symptom that motivated
        /// this fix for Qwen 3.5 / 3.6 thinking models.
        ///
        /// Instead, we match by USER messages (which the UI never modifies between turns
        /// unless the user explicitly edits a previous turn). For every leading position
        /// at which the incoming and tracked histories agree on user content, the
        /// assistant messages in between MUST have come from our previous generation
        /// turns - so we splice their tracked raw tokens in. As soon as a user message
        /// diverges (or roles disagree), we stop splicing for that position and beyond,
        /// which lets the KV cache reuse logic find a partial prefix match if one exists.
        /// </summary>
        internal static List<ChatMessage> AugmentWithCachedRawTokens(List<ChatMessage> incoming, IReadOnlyList<ChatMessage> trackedHistory)
        {
            if (incoming == null)
                return null;

            int matchUntil = 0;
            if (trackedHistory != null)
            {
                int max = Math.Min(incoming.Count, trackedHistory.Count);
                for (int i = 0; i < max; i++)
                {
                    ChatMessage src = incoming[i];
                    ChatMessage tracked = trackedHistory[i];

                    if (src.Role != tracked.Role)
                        break;

                    // Compare on Content for non-assistant roles only. Assistant content can be
                    // legitimately altered by the streaming output parser between turns.
                    if (src.Role != "assistant"
                        && !string.Equals(src.Content ?? string.Empty, tracked.Content ?? string.Empty, StringComparison.Ordinal))
                        break;

                    matchUntil = i + 1;
                }
            }

            var result = new List<ChatMessage>(incoming.Count);
            for (int i = 0; i < incoming.Count; i++)
            {
                ChatMessage src = incoming[i];

                bool useTracked = trackedHistory != null
                    && i < matchUntil
                    && trackedHistory[i].Role == "assistant"
                    && trackedHistory[i].RawOutputTokens != null
                    && trackedHistory[i].RawOutputTokens.Count > 0
                    && (src.RawOutputTokens == null || src.RawOutputTokens.Count == 0);

                if (useTracked)
                {
                    result.Add(new ChatMessage
                    {
                        Role = src.Role,
                        Content = src.Content,
                        ImagePaths = src.ImagePaths,
                        AudioPaths = src.AudioPaths,
                        TextFilePaths = src.TextFilePaths,
                        IsVideo = src.IsVideo,
                        ToolCalls = src.ToolCalls,
                        Thinking = src.Thinking,
                        RawOutputTokens = trackedHistory[i].RawOutputTokens,
                    });
                }
                else
                {
                    result.Add(src);
                }
            }
            return result;
        }

        /// <summary>
        /// Refresh <paramref name="trackedHistory"/> after a generation completes so
        /// the next turn can locate cached raw tokens by message position. Replaces
        /// the tracked list with the user-visible history (without our internal
        /// augmentation) plus the new assistant turn (with raw output tokens
        /// attached).
        /// </summary>
        private static void UpdateTrackedHistory(List<ChatMessage> trackedHistory, List<ChatMessage> incomingHistory, string assistantText, List<int> generatedTokens)
        {
            trackedHistory.Clear();
            if (incomingHistory != null)
            {
                for (int i = 0; i < incomingHistory.Count; i++)
                    trackedHistory.Add(CloneShallow(incomingHistory[i]));
            }

            trackedHistory.Add(new ChatMessage
            {
                Role = "assistant",
                Content = assistantText,
                RawOutputTokens = generatedTokens,
            });
        }

        private static ChatMessage CloneShallow(ChatMessage src)
        {
            return new ChatMessage
            {
                Role = src.Role,
                Content = src.Content,
                ImagePaths = src.ImagePaths,
                AudioPaths = src.AudioPaths,
                TextFilePaths = src.TextFilePaths,
                IsVideo = src.IsVideo,
                ToolCalls = src.ToolCalls,
                Thinking = src.Thinking,
                RawOutputTokens = src.RawOutputTokens,
            };
        }

        /// <summary>
        /// Move the model's KV state to one whose contents are exactly <paramref name="inputTokens"/>,
        /// returning the next-token logits at position <c>inputTokens.Count</c>. Plans the work
        /// via <see cref="KVCache.PlanReuse"/> against <paramref name="session"/>'s cache and
        /// then delegates to either an exact-match, partial-reuse, or full-reset path.
        /// <paramref name="reusedTokens"/> reports how many tokens were served from the
        /// session's cached KV state (full prompt for an exact match, the matching prefix
        /// for a partial reuse, zero for a full reset).
        /// </summary>
        private float[] PrepareForGeneration(ChatSession session, List<int> inputTokens, out int reusedTokens)
        {
            var cache = session.KVCache;
            ReusePlan plan = cache.PlanReuse(inputTokens, _model.SupportsKVCacheTruncation);

            switch (plan.Kind)
            {
                case ReusePlanKind.ExactMatch:
                {
                    reusedTokens = inputTokens.Count;
                    _logger.LogDebug(LogEventIds.KvCacheReusePlan,
                        "kv.reuse exact match reusing {ReusedTokens}/{TotalTokens} cached tokens (saved 100%)",
                        inputTokens.Count, inputTokens.Count);
                    _model.MultimodalInjector.QueuePromptEmbeddings(inputTokens.Count);
                    return plan.CachedLogits;
                }

                case ReusePlanKind.PartialReuse:
                {
                    int reusedPrefix = plan.ReusedPrefixLength;
                    int suffixLength = plan.TokensToForward;
                    reusedTokens = reusedPrefix;

                    _model.TruncateKVCache(reusedPrefix);
                    cache.TruncateTo(reusedPrefix);

                    bool queuedPromptEmbeddings = _model.MultimodalInjector.QueuePromptEmbeddings(reusedPrefix);
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

                    _model.ResetKVCache();
                    cache.Reset();
                    bool queuedPromptEmbeddings = _model.MultimodalInjector.QueuePromptEmbeddings(0);
                    var allTokens = inputTokens.ToArray();
                    float[] logits = ForwardPromptPrefill(allTokens, allowChunking: !queuedPromptEmbeddings);
                    cache.RecordAppend(allTokens, logits);
                    return logits;
                }
            }
        }

        private static long ToNanos(long elapsedTicks)
            => elapsedTicks * (1_000_000_000L / Stopwatch.Frequency);

        private float[] ForwardPromptPrefill(int[] tokens, bool allowChunking = true)
        {
            if (tokens == null || tokens.Length == 0)
                throw new ArgumentException("Prompt token list cannot be null or empty.", nameof(tokens));

            if (!allowChunking)
                return _model.ForwardRefill(tokens);

            int chunkSize = ResolvePrefillChunkSize(_backend, tokens.Length);
            if (chunkSize >= tokens.Length)
                return _model.ForwardRefill(tokens);

            _logger.LogInformation(LogEventIds.PromptChunking,
                "prompt.chunking total={TotalTokens} chunkSize={ChunkSize} backend={Backend}",
                tokens.Length, chunkSize, LoadedBackend ?? _backend.ToString());

            float[] logits = null;
            int chunkIndex = 0;
            for (int start = 0; start < tokens.Length; start += chunkSize)
            {
                int length = Math.Min(chunkSize, tokens.Length - start);
                int[] chunk = new int[length];
                Array.Copy(tokens, start, chunk, 0, length);
                logits = _model.ForwardRefill(chunk);
                chunkIndex++;
                _logger.LogTrace(LogEventIds.PromptChunking,
                    "prompt.chunking chunk={ChunkIndex} start={Start} length={Length}",
                    chunkIndex, start, length);
            }

            return logits;
        }

        internal static int ResolvePrefillChunkSize(BackendType backend, int tokenCount)
        {
            if (tokenCount <= 0)
                return 0;

            // CUDA can OOM on a single huge prefill graph, so we cap the chunk size.
            // CPU / Metal handle a single big graph fine.
            return backend == BackendType.GgmlCuda
                ? Math.Min(tokenCount, 5120)
                : tokenCount;
        }

        private List<int> TruncatePromptToContext(ChatSession session, List<int> inputTokens, int maxTokens)
        {
            int maxCtx = _model.MaxContextLength;
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
            trimStart = _model.MultimodalInjector.ClampTrimStart(trimStart);
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
            _model.MultimodalInjector.TrimPreparedPrompt(trimStart);
            // The prompt has been trimmed at the front; the model state (which assumed a
            // different absolute position) is no longer reusable.
            session.TrackedHistory.Clear();
            session.KVCache.Reset();
            _model.ResetKVCache();
            return inputTokens.GetRange(trimStart, kept);
        }

        private static int[] CopyTokenRange(IList<int> tokens, int start, int length)
        {
            var result = new int[length];
            for (int i = 0; i < length; i++)
                result[i] = tokens[start + i];
            return result;
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
            return GenerateStreamAsync(_intrinsicSession, prompt, imagePaths, maxTokens, cancellationToken, samplingConfig);
        }

        /// <summary>
        /// Session-aware streaming generate. The session's KV cache is reset before the
        /// prefill - one-shot completions intentionally do not share prefixes with prior
        /// turns stored on the session - so the yielded done tuple always reports
        /// <c>kvCacheReusedTokens == 0</c>.
        /// </summary>
        public async IAsyncEnumerable<(string piece, bool done, int promptTokens, int evalTokens, int kvCacheReusedTokens, long totalNs, long promptNs, long evalNs)>
            GenerateStreamAsync(
                ChatSession session,
                string prompt,
                List<string> imagePaths,
                int maxTokens,
                [EnumeratorCancellation] CancellationToken cancellationToken,
                SamplingConfig samplingConfig = null)
        {
            ActivateSession(session ?? _intrinsicSession);
            var activeSession = _activeSession;

            string arch = _model.Config.Architecture;
            var messages = new List<ChatMessage>
            {
                new ChatMessage { Role = "user", Content = prompt, ImagePaths = imagePaths }
            };

            using var generateScope = _logger.BeginScope(new Dictionary<string, object>(StringComparer.Ordinal)
            {
                [LogScopeKeys.SessionId] = activeSession.Id,
                [LogScopeKeys.Model] = LoadedModelName ?? "(none)",
                [LogScopeKeys.Backend] = LoadedBackend ?? "(none)",
                [LogScopeKeys.Operation] = "generate.stream",
            });

            int generateImageCount = imagePaths?.Count ?? 0;
            string promptContent = LoggingExtensions.SanitizeForLogFull(prompt);
            // Reuse the chat-turn upload manifest so /api/generate logs surface the
            // same per-turn upload audit (path + saved filename + media type) as the
            // chat path. messages[0] always holds the single user prompt assembled
            // immediately above, so the manifest covers exactly this request's
            // attachments.
            string turnUploads = SerializeUploadsForLog(messages[0]);
            _logger.LogInformation(LogEventIds.ChatStarted,
                "generate.start arch={Architecture} maxTokens={MaxTokens} imageAttachments={ImageCount} uploads={Uploads} sampling(temp={Temperature},topK={TopK},topP={TopP},seed={Seed}) prompt=\"{Prompt}\"",
                arch, maxTokens, generateImageCount, turnUploads,
                samplingConfig?.Temperature ?? 0.8f, samplingConfig?.TopK ?? 40,
                samplingConfig?.TopP ?? 0.9f, samplingConfig?.Seed ?? 0,
                promptContent);

            var preparedMessages = PrepareHistoryForInference(messages, arch, _logger);
            var inputTokens = _kvCacheRenderer.RenderToTokens(
                _model.Tokenizer,
                _model.Config.ChatTemplate,
                preparedMessages,
                arch,
                addGenerationPrompt: true);

            inputTokens = _model.MultimodalInjector.ProcessPromptTokens(preparedMessages, inputTokens);
            inputTokens = TruncatePromptToContext(activeSession, inputTokens, maxTokens);

            ResetSession(activeSession);

            var sw = Stopwatch.StartNew();
            bool queuedPromptEmbeddings = _model.MultimodalInjector.QueuePromptEmbeddings(0);
            var promptArray = inputTokens.ToArray();
            float[] logits = ForwardPromptPrefill(promptArray, allowChunking: !queuedPromptEmbeddings);
            activeSession.KVCache.RecordAppend(promptArray, logits);
            long promptNs = ToNanos(sw.ElapsedTicks);
            int promptTokenCount = inputTokens.Count;

            var cfg = samplingConfig ?? SamplingConfig.Default;
            var sampler = new TokenSampler(cfg);
            var generatedTokens = new List<int>();
            var rawBytes = new List<byte>();
            int prevCharLen = 0;

            var evalSw = Stopwatch.StartNew();
            string finishReason = "max_tokens";
            bool wasCancelled = false;
            for (int step = 0; step < maxTokens; step++)
            {
                if (cancellationToken.IsCancellationRequested)
                {
                    wasCancelled = true;
                    finishReason = "cancelled";
                    break;
                }

                int nextToken = sampler.Sample(logits, generatedTokens);
                if (_model.Tokenizer.IsEos(nextToken))
                {
                    finishReason = "eos";
                    break;
                }

                generatedTokens.Add(nextToken);
                _model.Tokenizer.AppendTokenBytes(nextToken, rawBytes);
                int validLen = FindValidUtf8Length(rawBytes);
                string decoded = Encoding.UTF8.GetString(rawBytes.GetRange(0, validLen).ToArray());
                string piece = prevCharLen < decoded.Length ? decoded.Substring(prevCharLen) : "";
                prevCharLen = decoded.Length;

                if (cfg.StopSequences != null && cfg.StopSequences.Count > 0)
                {
                    var (_, shouldStop) = sampler.CheckStopSequences(decoded);
                    if (shouldStop)
                    {
                        finishReason = "stop_sequence";
                        break;
                    }
                }

                if (piece.Length > 0)
                    yield return (piece, false, 0, 0, 0, 0, 0, 0);
                logits = _model.Forward(new[] { nextToken });
                activeSession.KVCache.RecordAppend(nextToken, logits);
            }

            evalSw.Stop();
            sw.Stop();
            long evalNs = ToNanos(evalSw.ElapsedTicks);
            long totalNs = ToNanos(sw.ElapsedTicks);

            double evalSeconds = evalSw.Elapsed.TotalSeconds;
            double tokensPerSecond = (evalSeconds > 0 && generatedTokens.Count > 0)
                ? generatedTokens.Count / evalSeconds
                : 0;
            // Decode the entire raw output bytes; this captures both reasoning markers
            // (e.g. <think>...</think>) and the final result text in one go.
            string completionContent = LoggingExtensions.SanitizeForLogFull(Encoding.UTF8.GetString(rawBytes.ToArray()));

            // GenerateStreamAsync always resets the session before prefilling, so the
            // KV reuse fields are constants that we still surface for symmetry with
            // the chat metrics tuple.
            const int kvCacheReusedTokens = 0;
            const double kvCacheReusePercent = 0.0;

            if (wasCancelled)
            {
                _logger.LogWarning(LogEventIds.ChatAborted,
                    "generate.cancelled tokens={Tokens} promptTokens={PromptTokens} kvReused={KvReusedTokens} kvReusePercent={KvReusePercent:F1} elapsedMs={ElapsedMs:F1} completion=\"{Completion}\"",
                    generatedTokens.Count, promptTokenCount, kvCacheReusedTokens, kvCacheReusePercent,
                    sw.Elapsed.TotalMilliseconds, completionContent);
            }
            else
            {
                _logger.LogInformation(LogEventIds.ChatCompleted,
                    "generate.complete tokens={Tokens} promptTokens={PromptTokens} kvReused={KvReusedTokens} kvReusePercent={KvReusePercent:F1} elapsedMs={ElapsedMs:F1} tokensPerSec={TokensPerSec:F2} finishReason={FinishReason} completion=\"{Completion}\"",
                    generatedTokens.Count, promptTokenCount, kvCacheReusedTokens, kvCacheReusePercent,
                    sw.Elapsed.TotalMilliseconds, tokensPerSecond, finishReason, completionContent);
            }

            yield return ("", true, promptTokenCount, generatedTokens.Count, kvCacheReusedTokens, totalNs, promptNs, evalNs);
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

        internal static List<ChatMessage> PrepareHistoryForInference(List<ChatMessage> history, string arch)
            => PrepareHistoryForInference(history, arch, NullLogger.Instance);

        internal static List<ChatMessage> PrepareHistoryForInference(List<ChatMessage> history, string arch, ILogger logger)
        {
            if (history == null || history.Count == 0)
                return history;

            List<ChatMessage> prepared = null;
            for (int i = 0; i < history.Count; i++)
            {
                var normalized = NormalizeMessageForInference(history[i], arch, logger);
                if (ReferenceEquals(normalized, history[i]))
                    continue;

                prepared ??= new List<ChatMessage>(history);
                prepared[i] = normalized;
            }

            return prepared ?? history;
        }

        private static ChatMessage NormalizeMessageForInference(ChatMessage msg, string arch, ILogger logger)
        {
            int maxVideoFrames = MediaHelper.GetConfiguredMaxVideoFrames();
            if (arch != "gemma4" || !msg.IsVideo || msg.ImagePaths == null || msg.ImagePaths.Count <= maxVideoFrames)
                return msg;

            var sampled = MediaHelper.SelectEvenlySpacedIndices(msg.ImagePaths.Count, maxVideoFrames)
                .Select(i => msg.ImagePaths[i])
                .ToList();

            (logger ?? NullLogger.Instance).LogInformation(LogEventIds.VideoFrameDownsample,
                "video.downsample originalFrames={OriginalFrames} sampledFrames={SampledFrames} architecture={Architecture}",
                msg.ImagePaths.Count, sampled.Count, arch);

            return new ChatMessage
            {
                Role = msg.Role,
                Content = msg.Content,
                ImagePaths = sampled,
                AudioPaths = msg.AudioPaths != null ? new List<string>(msg.AudioPaths) : null,
                TextFilePaths = msg.TextFilePaths != null ? new List<string>(msg.TextFilePaths) : null,
                IsVideo = msg.IsVideo,
                ToolCalls = msg.ToolCalls,
                Thinking = msg.Thinking,
                RawOutputTokens = msg.RawOutputTokens,
            };
        }

        internal static bool HasMultimodalContent(ChatMessage msg)
        {
            if (msg == null) return false;
            return (msg.ImagePaths != null && msg.ImagePaths.Count > 0) ||
                   (msg.AudioPaths != null && msg.AudioPaths.Count > 0);
        }

        internal static bool HasMultimodalContent(List<ChatMessage> history)
        {
            if (history == null || history.Count == 0)
                return false;

            return history.Any(HasMultimodalContent);
        }

        internal static List<string> GetImagePathsInPromptOrder(List<ChatMessage> history)
        {
            var imagePaths = new List<string>();
            if (history == null)
                return imagePaths;

            foreach (var msg in history)
            {
                if (msg.ImagePaths == null)
                    continue;

                foreach (var path in msg.ImagePaths)
                {
                    if (!string.IsNullOrEmpty(path))
                        imagePaths.Add(path);
                }
            }

            return imagePaths;
        }

        /// <summary>
        /// Serialize the entire conversation array submitted for this turn into a
        /// single-line JSON string for logging. Each entry carries the role, the
        /// FULL message content (no preview truncation, control characters escaped
        /// by JSON itself), and the FULL list of uploaded file paths (image,
        /// audio, text) so operators can see which uploads belong to which turn
        /// in the audit log. Tool-call arguments and raw output tokens are
        /// deliberately excluded so the log line stays bounded by the message text
        /// rather than per-turn generation byproducts.
        /// </summary>
        internal static string SerializeMessagesForLog(List<ChatMessage> messages)
        {
            if (messages == null || messages.Count == 0)
                return "[]";

            var entries = new List<ChatMessageLogEntry>(messages.Count);
            foreach (var m in messages)
            {
                if (m == null) continue;
                entries.Add(new ChatMessageLogEntry
                {
                    Role = m.Role ?? string.Empty,
                    Content = m.Content ?? string.Empty,
                    Images = ToPathList(m.ImagePaths),
                    Audios = ToPathList(m.AudioPaths),
                    TextFiles = ToPathList(m.TextFilePaths),
                    IsVideo = m.IsVideo ? true : (bool?)null,
                    Thinking = string.IsNullOrEmpty(m.Thinking) ? null : m.Thinking,
                    ToolCallCount = (m.ToolCalls != null && m.ToolCalls.Count > 0) ? m.ToolCalls.Count : (int?)null,
                });
            }

            return JsonSerializer.Serialize(entries, FullInputJsonOptions);
        }

        /// <summary>
        /// Serialize the upload manifest for a single message (typically the latest
        /// user turn) as a single-line JSON array. Each entry records the saved file
        /// path, the saved file name (which equals <c>Path.GetFileName(path)</c> and
        /// is the unique on-disk identifier in the upload directory), and the kind
        /// of media. Returns the literal string <c>"[]"</c> when there are no
        /// uploads so the structured log field is always present and parseable.
        ///
        /// Image, audio and text uploads each surface as an entry; when
        /// <see cref="ChatMessage.IsVideo"/> is set the per-frame paths from
        /// <see cref="ChatMessage.ImagePaths"/> are tagged as <c>video_frame</c>
        /// (since the original video file itself is decomposed into frames on
        /// upload and only the frames flow into the chat message).
        /// </summary>
        internal static string SerializeUploadsForLog(ChatMessage message)
        {
            if (message == null)
                return "[]";

            var entries = new List<UploadLogEntry>();

            string imageType = message.IsVideo ? "video_frame" : "image";
            AppendUploadEntries(entries, message.ImagePaths, imageType);
            AppendUploadEntries(entries, message.AudioPaths, "audio");
            AppendUploadEntries(entries, message.TextFilePaths, "text");

            return entries.Count == 0
                ? "[]"
                : JsonSerializer.Serialize(entries, FullInputJsonOptions);
        }

        private static void AppendUploadEntries(List<UploadLogEntry> sink, List<string> paths, string mediaType)
        {
            if (paths == null || paths.Count == 0)
                return;

            foreach (var path in paths)
            {
                if (string.IsNullOrEmpty(path))
                    continue;
                sink.Add(new UploadLogEntry
                {
                    Path = path,
                    Name = Path.GetFileName(path),
                    MediaType = mediaType,
                });
            }
        }

        private static List<string> ToPathList(List<string> source)
        {
            if (source == null || source.Count == 0)
                return null;

            var result = new List<string>(source.Count);
            foreach (var p in source)
            {
                if (!string.IsNullOrEmpty(p))
                    result.Add(p);
            }
            return result.Count == 0 ? null : result;
        }

        /// <summary>Slim DTO mirroring just the loggable subset of <see cref="ChatMessage"/>.</summary>
        private sealed class ChatMessageLogEntry
        {
            public string Role { get; init; }
            public string Content { get; init; }
            public List<string> Images { get; init; }
            public List<string> Audios { get; init; }
            public List<string> TextFiles { get; init; }
            public bool? IsVideo { get; init; }
            public string Thinking { get; init; }
            public int? ToolCallCount { get; init; }
        }

        /// <summary>Slim DTO for one row of the per-turn upload manifest.</summary>
        private sealed class UploadLogEntry
        {
            public string Path { get; init; }
            public string Name { get; init; }
            public string MediaType { get; init; }
        }

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

        private static bool IsMmProjFile(string fileName)
        {
            return fileName.IndexOf("mmproj", StringComparison.OrdinalIgnoreCase) >= 0;
        }

        public void Dispose()
        {
            _model?.Dispose();
            _model = null;
            _activeSession = null;
            _intrinsicSession.Dispose();
        }
    }
}
