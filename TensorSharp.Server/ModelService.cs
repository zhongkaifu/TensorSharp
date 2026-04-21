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
using System.Threading;
using System.Threading.Tasks;
using TensorSharp;
using TensorSharp.Cpu;

namespace TensorSharp.Server
{
    public class ModelService : IDisposable
    {
        private readonly IPromptRenderer _promptRenderer = new GgufPromptRenderer();
        private readonly KVCachePromptRenderer _kvCacheRenderer;
        private readonly KVCache _kvCache = new();

        // The conversation that the model's KV state currently corresponds to. Each
        // assistant message we generated has its raw output tokens attached so the next
        // turn's render can splice them back in (instead of re-tokenizing the assistant
        // text, which is lossy for thinking / channel-based models).
        private readonly List<ChatMessage> _trackedHistory = new();

        private ModelBase _model;
        private string _loadedModelPath;
        private string _loadedMmProjPath;
        private BackendType _backend;

        public ModelService()
        {
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

        /// <summary>Inspection-only view of the conversation cache (for tests / diagnostics).</summary>
        public KVCache KVCache => _kvCache;

        /// <summary>
        /// Snapshot of the messages whose tokens are reflected in the current KV cache state.
        /// Returned as a copy so callers can't mutate internal bookkeeping.
        /// </summary>
        public IReadOnlyList<ChatMessage> TrackedHistory => _trackedHistory.AsReadOnly();

        public bool IsModelAlreadyLoaded(string modelName)
        {
            return _model != null && string.Equals(LoadedModelName, modelName, StringComparison.OrdinalIgnoreCase);
        }

        public void InvalidateKVCache()
        {
            _trackedHistory.Clear();
            _kvCache.Reset();
            _model?.ResetKVCache();
        }

        public void LoadModel(string modelPath, string mmProjPath, string backendStr)
        {
            _model?.Dispose();
            _model = null;
            _loadedModelPath = null;
            _loadedMmProjPath = null;
            _trackedHistory.Clear();
            _kvCache.Reset();

            _backend = BackendCatalog.Canonicalize(backendStr) switch
            {
                "ggml_metal" => BackendType.GgmlMetal,
                "ggml_cpu" => BackendType.GgmlCpu,
                "ggml_cuda" => BackendType.GgmlCuda,
                "cpu" => BackendType.Cpu,
                _ => BackendType.GgmlCpu
            };

            _model = ModelBase.Create(modelPath, _backend);
            _loadedModelPath = modelPath;

            if (!string.IsNullOrEmpty(mmProjPath) && File.Exists(mmProjPath))
            {
                LoadEncoders(mmProjPath);
                _loadedMmProjPath = mmProjPath;
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
        public async IAsyncEnumerable<string> ChatStreamAsync(
            List<ChatMessage> history,
            int maxTokens,
            [EnumeratorCancellation] CancellationToken cancellationToken,
            SamplingConfig samplingConfig = null,
            List<ToolFunction> tools = null, bool enableThinking = false)
        {
            await foreach (var (piece, _, _, _, _, _, _) in
                ChatStreamInternalAsync(history, maxTokens, cancellationToken, samplingConfig, tools, enableThinking, withMetrics: false))
            {
                if (!string.IsNullOrEmpty(piece))
                    yield return piece;
            }
        }

        /// <summary>
        /// Stream chat inference tokens with timing metrics. Must be called within the InferenceQueue.
        /// Reuses the KV cache from the previous turn when the rendered text prefix matches.
        /// </summary>
        public IAsyncEnumerable<(string piece, bool done, int promptTokens, int evalTokens, long totalNs, long promptNs, long evalNs)>
            ChatStreamWithMetricsAsync(
                List<ChatMessage> history,
                int maxTokens,
                CancellationToken cancellationToken,
                SamplingConfig samplingConfig = null,
                List<ToolFunction> tools = null, bool enableThinking = false)
        {
            return ChatStreamInternalAsync(history, maxTokens, cancellationToken, samplingConfig, tools, enableThinking, withMetrics: true);
        }

        private async IAsyncEnumerable<(string piece, bool done, int promptTokens, int evalTokens, long totalNs, long promptNs, long evalNs)>
            ChatStreamInternalAsync(
                List<ChatMessage> history,
                int maxTokens,
                [EnumeratorCancellation] CancellationToken cancellationToken,
                SamplingConfig samplingConfig,
                List<ToolFunction> tools,
                bool enableThinking,
                bool withMetrics)
        {
            string arch = _model.Config.Architecture;
            var preparedHistory = PrepareHistoryForInference(history, arch);

            // Project the incoming user-visible history onto our tracked conversation so
            // that any assistant messages we previously generated carry their raw output
            // tokens forward into the renderer.
            var renderHistory = AugmentWithCachedRawTokens(preparedHistory);

            Console.Error.WriteLine($"[Prompt] arch={arch}, sampling: temp={samplingConfig?.Temperature ?? 0.8f}, top_k={samplingConfig?.TopK ?? 40}, top_p={samplingConfig?.TopP ?? 0.9f}");

            var inputTokens = _kvCacheRenderer.RenderToTokens(
                _model.Tokenizer,
                _model.Config.ChatTemplate,
                renderHistory,
                arch,
                addGenerationPrompt: true,
                tools: tools,
                enableThinking: enableThinking);

            inputTokens = _model.MultimodalInjector.ProcessPromptTokens(renderHistory, inputTokens);
            inputTokens = TruncatePromptToContext(inputTokens, maxTokens);

            int promptTokenCount = inputTokens.Count;
            var sw = Stopwatch.StartNew();
            float[] logits = PrepareForGeneration(inputTokens);
            long promptNs = ToNanos(sw.ElapsedTicks);

            var generatedTokens = new List<int>();
            var cfg = samplingConfig ?? SamplingConfig.Default;
            var sampler = new TokenSampler(cfg);
            var rawBytes = new List<byte>();
            int prevCharLen = 0;

            var evalSw = Stopwatch.StartNew();
            bool firstTokenSampled = false;

            for (int step = 0; step < maxTokens; step++)
            {
                if (cancellationToken.IsCancellationRequested)
                    break;

                int nextToken = sampler.Sample(logits, generatedTokens);
                if (_model.Tokenizer.IsEos(nextToken))
                    break;

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
                        stopRequested = true;
                }

                if (!firstTokenSampled)
                {
                    firstTokenSampled = true;
                }

                if (piece.Length > 0)
                    yield return (piece, false, 0, 0, 0, 0, 0);

                if (stopRequested)
                    break;

                logits = _model.Forward(new[] { nextToken });
                _kvCache.RecordAppend(nextToken, logits);
            }

            string assistantText = Encoding.UTF8.GetString(rawBytes.ToArray());
            // Use the AUGMENTED history (which has raw tokens spliced into prior assistant
            // turns from our previous tracked history) so that the raw tokens carry forward
            // for ALL past assistants, not just the immediately-previous one. The plain
            // user-visible `history` doesn't carry raw tokens (the WebUI never sees them),
            // so cloning from `history` here would cause the cache to silently re-reset
            // every other turn as the raw tokens of older assistants fell off the tracked
            // record.
            UpdateTrackedHistory(renderHistory, assistantText, generatedTokens);

            if (withMetrics)
            {
                long evalNs = ToNanos(evalSw.ElapsedTicks);
                long totalNs = ToNanos(sw.ElapsedTicks);
                yield return ("", true, promptTokenCount, generatedTokens.Count, totalNs, promptNs, evalNs);
            }
        }

        /// <summary>
        /// Walks <paramref name="incoming"/> alongside <see cref="_trackedHistory"/> to
        /// produce a render-ready history with cached raw output tokens spliced into
        /// assistant messages.
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
        internal List<ChatMessage> AugmentWithCachedRawTokens(List<ChatMessage> incoming)
        {
            if (incoming == null)
                return null;

            int matchUntil = 0;
            int max = Math.Min(incoming.Count, _trackedHistory.Count);
            for (int i = 0; i < max; i++)
            {
                ChatMessage src = incoming[i];
                ChatMessage tracked = _trackedHistory[i];

                if (src.Role != tracked.Role)
                    break;

                // Compare on Content for non-assistant roles only. Assistant content can be
                // legitimately altered by the streaming output parser between turns.
                if (src.Role != "assistant"
                    && !string.Equals(src.Content ?? string.Empty, tracked.Content ?? string.Empty, StringComparison.Ordinal))
                    break;

                matchUntil = i + 1;
            }

            var result = new List<ChatMessage>(incoming.Count);
            for (int i = 0; i < incoming.Count; i++)
            {
                ChatMessage src = incoming[i];

                bool useTracked = i < matchUntil
                    && _trackedHistory[i].Role == "assistant"
                    && _trackedHistory[i].RawOutputTokens != null
                    && _trackedHistory[i].RawOutputTokens.Count > 0
                    && (src.RawOutputTokens == null || src.RawOutputTokens.Count == 0);

                if (useTracked)
                {
                    result.Add(new ChatMessage
                    {
                        Role = src.Role,
                        Content = src.Content,
                        ImagePaths = src.ImagePaths,
                        AudioPaths = src.AudioPaths,
                        IsVideo = src.IsVideo,
                        ToolCalls = src.ToolCalls,
                        Thinking = src.Thinking,
                        RawOutputTokens = _trackedHistory[i].RawOutputTokens,
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
        /// Refresh <see cref="_trackedHistory"/> after a generation completes so the next
        /// turn can locate cached raw tokens by message position. We replace the tracked
        /// list with the user-visible history (without our internal augmentation) plus the
        /// new assistant turn (with raw output tokens attached).
        /// </summary>
        private void UpdateTrackedHistory(List<ChatMessage> incomingHistory, string assistantText, List<int> generatedTokens)
        {
            _trackedHistory.Clear();
            if (incomingHistory != null)
            {
                for (int i = 0; i < incomingHistory.Count; i++)
                    _trackedHistory.Add(CloneShallow(incomingHistory[i]));
            }

            _trackedHistory.Add(new ChatMessage
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
                IsVideo = src.IsVideo,
                ToolCalls = src.ToolCalls,
                Thinking = src.Thinking,
                RawOutputTokens = src.RawOutputTokens,
            };
        }

        /// <summary>
        /// Move the model's KV state to one whose contents are exactly <paramref name="inputTokens"/>,
        /// returning the next-token logits at position <c>inputTokens.Count</c>. Plans the work
        /// via <see cref="KVCache.PlanReuse"/> and then delegates to either an exact-match,
        /// partial-reuse, or full-reset path.
        /// </summary>
        private float[] PrepareForGeneration(List<int> inputTokens)
        {
            ReusePlan plan = _kvCache.PlanReuse(inputTokens, _model.SupportsKVCacheTruncation);

            switch (plan.Kind)
            {
                case ReusePlanKind.ExactMatch:
                {
                    Console.WriteLine($"[KV cache] Exact match: reusing {inputTokens.Count}/{inputTokens.Count} cached tokens (saved 100%)");
                    _model.MultimodalInjector.QueuePromptEmbeddings(inputTokens.Count);
                    return plan.CachedLogits;
                }

                case ReusePlanKind.PartialReuse:
                {
                    int reusedPrefix = plan.ReusedPrefixLength;
                    int suffixLength = plan.TokensToForward;

                    _model.TruncateKVCache(reusedPrefix);
                    _kvCache.TruncateTo(reusedPrefix);

                    bool queuedPromptEmbeddings = _model.MultimodalInjector.QueuePromptEmbeddings(reusedPrefix);
                    var suffixTokens = CopyTokenRange(inputTokens, reusedPrefix, suffixLength);

                    Console.WriteLine($"[KV cache] Partial reuse: keeping {reusedPrefix}/{inputTokens.Count} tokens, forwarding {suffixLength} new tokens (saved {100.0 * reusedPrefix / inputTokens.Count:F0}%)");
                    float[] logits = ForwardPromptPrefill(suffixTokens, allowChunking: !queuedPromptEmbeddings);
                    _kvCache.RecordAppend(suffixTokens, logits);
                    return logits;
                }

                case ReusePlanKind.Reset:
                default:
                {
                    if (!_kvCache.IsEmpty)
                        Console.WriteLine($"[KV cache] Full reset (cached {_kvCache.Count} tokens, no usable common prefix with {inputTokens.Count}-token prompt)");
                    else
                        Console.WriteLine($"[KV cache] Full prompt forward: {inputTokens.Count} tokens");

                    _model.ResetKVCache();
                    _kvCache.Reset();
                    bool queuedPromptEmbeddings = _model.MultimodalInjector.QueuePromptEmbeddings(0);
                    var allTokens = inputTokens.ToArray();
                    float[] logits = ForwardPromptPrefill(allTokens, allowChunking: !queuedPromptEmbeddings);
                    _kvCache.RecordAppend(allTokens, logits);
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

            Console.WriteLine($"[Prompt] Chunking prefill: {tokens.Length} tokens in blocks of {chunkSize} on {LoadedBackend ?? _backend.ToString()}");

            float[] logits = null;
            for (int start = 0; start < tokens.Length; start += chunkSize)
            {
                int length = Math.Min(chunkSize, tokens.Length - start);
                int[] chunk = new int[length];
                Array.Copy(tokens, start, chunk, 0, length);
                logits = _model.ForwardRefill(chunk);
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

        private List<int> TruncatePromptToContext(List<int> inputTokens, int maxTokens)
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

            Console.WriteLine($"[Context] Truncating prompt from {inputTokens.Count} to {kept} tokens (context limit {maxCtx}, reserving {maxTokens} for generation)");
            _model.MultimodalInjector.TrimPreparedPrompt(trimStart);
            // The prompt has been trimmed at the front; the model state (which assumed a
            // different absolute position) is no longer reusable.
            _trackedHistory.Clear();
            _kvCache.Reset();
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
        /// Always resets the KV cache - intended for one-shot completions.
        /// </summary>
        public async IAsyncEnumerable<(string piece, bool done, int promptTokens, int evalTokens, long totalNs, long promptNs, long evalNs)>
            GenerateStreamAsync(
                string prompt,
                List<string> imagePaths,
                int maxTokens,
                [EnumeratorCancellation] CancellationToken cancellationToken,
                SamplingConfig samplingConfig = null)
        {
            string arch = _model.Config.Architecture;
            var messages = new List<ChatMessage>
            {
                new ChatMessage { Role = "user", Content = prompt, ImagePaths = imagePaths }
            };

            var preparedMessages = PrepareHistoryForInference(messages, arch);
            var inputTokens = _kvCacheRenderer.RenderToTokens(
                _model.Tokenizer,
                _model.Config.ChatTemplate,
                preparedMessages,
                arch,
                addGenerationPrompt: true);

            inputTokens = _model.MultimodalInjector.ProcessPromptTokens(preparedMessages, inputTokens);
            inputTokens = TruncatePromptToContext(inputTokens, maxTokens);

            InvalidateKVCache();

            var sw = Stopwatch.StartNew();
            bool queuedPromptEmbeddings = _model.MultimodalInjector.QueuePromptEmbeddings(0);
            var promptArray = inputTokens.ToArray();
            float[] logits = ForwardPromptPrefill(promptArray, allowChunking: !queuedPromptEmbeddings);
            _kvCache.RecordAppend(promptArray, logits);
            long promptNs = ToNanos(sw.ElapsedTicks);
            int promptTokenCount = inputTokens.Count;

            var cfg = samplingConfig ?? SamplingConfig.Default;
            var sampler = new TokenSampler(cfg);
            var generatedTokens = new List<int>();
            var rawBytes = new List<byte>();
            int prevCharLen = 0;

            var evalSw = Stopwatch.StartNew();
            for (int step = 0; step < maxTokens; step++)
            {
                if (cancellationToken.IsCancellationRequested) break;

                int nextToken = sampler.Sample(logits, generatedTokens);
                if (_model.Tokenizer.IsEos(nextToken)) break;

                generatedTokens.Add(nextToken);
                _model.Tokenizer.AppendTokenBytes(nextToken, rawBytes);
                int validLen = FindValidUtf8Length(rawBytes);
                string decoded = Encoding.UTF8.GetString(rawBytes.GetRange(0, validLen).ToArray());
                string piece = prevCharLen < decoded.Length ? decoded.Substring(prevCharLen) : "";
                prevCharLen = decoded.Length;

                if (cfg.StopSequences != null && cfg.StopSequences.Count > 0)
                {
                    var (_, shouldStop) = sampler.CheckStopSequences(decoded);
                    if (shouldStop) break;
                }

                if (piece.Length > 0)
                    yield return (piece, false, 0, 0, 0, 0, 0);
                logits = _model.Forward(new[] { nextToken });
                _kvCache.RecordAppend(nextToken, logits);
            }

            long evalNs = ToNanos(evalSw.ElapsedTicks);
            long totalNs = ToNanos(sw.ElapsedTicks);

            yield return ("", true, promptTokenCount, generatedTokens.Count, totalNs, promptNs, evalNs);
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
        {
            if (history == null || history.Count == 0)
                return history;

            List<ChatMessage> prepared = null;
            for (int i = 0; i < history.Count; i++)
            {
                var normalized = NormalizeMessageForInference(history[i], arch);
                if (ReferenceEquals(normalized, history[i]))
                    continue;

                prepared ??= new List<ChatMessage>(history);
                prepared[i] = normalized;
            }

            return prepared ?? history;
        }

        private static ChatMessage NormalizeMessageForInference(ChatMessage msg, string arch)
        {
            int maxVideoFrames = MediaHelper.GetConfiguredMaxVideoFrames();
            if (arch != "gemma4" || !msg.IsVideo || msg.ImagePaths == null || msg.ImagePaths.Count <= maxVideoFrames)
                return msg;

            var sampled = MediaHelper.SelectEvenlySpacedIndices(msg.ImagePaths.Count, maxVideoFrames)
                .Select(i => msg.ImagePaths[i])
                .ToList();

            Console.WriteLine($"[video] Downsampled {msg.ImagePaths.Count} frames to {sampled.Count} evenly spaced frames for Gemma4 prefill stability.");

            return new ChatMessage
            {
                Role = msg.Role,
                Content = msg.Content,
                ImagePaths = sampled,
                AudioPaths = msg.AudioPaths != null ? new List<string>(msg.AudioPaths) : null,
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
            _trackedHistory.Clear();
            _kvCache.Reset();
        }
    }
}
