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
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading;
using Microsoft.Extensions.Logging;

namespace TensorSharp.Server
{
    internal sealed class ChatGenerationPipeline
    {
        private readonly ModelLifecycleService _lifecycle;
        private readonly SessionKvCacheManager _sessions;
        private readonly KVCachePromptRenderer _kvCacheRenderer;
        private readonly InferenceTelemetry _telemetry;
        private readonly ILogger _logger;

        public ChatGenerationPipeline(
            ModelLifecycleService lifecycle,
            SessionKvCacheManager sessions,
            KVCachePromptRenderer kvCacheRenderer,
            InferenceTelemetry telemetry,
            ILogger logger)
        {
            _lifecycle = lifecycle ?? throw new ArgumentNullException(nameof(lifecycle));
            _sessions = sessions ?? throw new ArgumentNullException(nameof(sessions));
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
            _sessions.ActivateSession(session ?? _sessions.IntrinsicSession);
            var activeSession = _sessions.ActiveSession;
            var model = _lifecycle.Model;

            string arch = model.Config.Architecture;
            var preparedHistory = ChatHistoryPreparer.PrepareHistoryForInference(history, arch, _logger);

            // Project the incoming user-visible history onto our tracked conversation so
            // that any assistant messages we previously generated carry their raw output
            // tokens forward into the renderer.
            var renderHistory = ChatHistoryPreparer.AugmentWithCachedRawTokens(preparedHistory, activeSession.TrackedHistory);

            using var chatScope = _telemetry.BeginInferenceScope(
                activeSession,
                _lifecycle.LoadedModelName,
                _lifecycle.LoadedBackend,
                "chat.stream");

            _telemetry.LogChatStarted(arch, maxTokens, enableThinking, tools, preparedHistory, samplingConfig);

            var inputTokens = _kvCacheRenderer.RenderToTokens(
                model.Tokenizer,
                model.Config.ChatTemplate,
                renderHistory,
                arch,
                addGenerationPrompt: true,
                tools: tools,
                enableThinking: enableThinking);

            inputTokens = model.MultimodalInjector.ProcessPromptTokens(renderHistory, inputTokens);
            inputTokens = _sessions.TruncatePromptToContext(activeSession, inputTokens, maxTokens);

            int promptTokenCount = inputTokens.Count;
            var sw = Stopwatch.StartNew();
            float[] logits = _sessions.PrepareForGeneration(activeSession, inputTokens, out int kvCacheReusedTokens);
            long promptNs = InferenceTelemetry.ToNanos(sw.ElapsedTicks);
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
                if (model.Tokenizer.IsEos(nextToken))
                {
                    finishReason = "eos";
                    break;
                }

                generatedTokens.Add(nextToken);
                model.Tokenizer.AppendTokenBytes(nextToken, rawBytes);
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

                logits = model.Forward(new[] { nextToken });
                activeSession.KVCache.RecordAppend(nextToken, logits);
            }

            string assistantText = Encoding.UTF8.GetString(rawBytes.ToArray());
            evalSw.Stop();
            sw.Stop();

            // Use the AUGMENTED history so raw tokens carry forward for all past
            // assistant turns, not just the immediately previous one.
            ChatHistoryPreparer.UpdateTrackedHistory(
                activeSession.TrackedHistory,
                renderHistory,
                assistantText,
                generatedTokens);

            double evalSeconds = evalSw.Elapsed.TotalSeconds;
            double tokensPerSecond = (evalSeconds > 0 && generatedTokens.Count > 0)
                ? generatedTokens.Count / evalSeconds
                : 0;

            _telemetry.LogChatFinished(
                wasCancelled,
                generatedTokens.Count,
                promptTokenCount,
                kvCacheReusedTokens,
                kvCacheReusePercent,
                timeToFirstTokenMs,
                sw.Elapsed.TotalMilliseconds,
                tokensPerSecond,
                finishReason,
                assistantText);

            long evalNs = InferenceTelemetry.ToNanos(evalSw.ElapsedTicks);
            long totalNs = InferenceTelemetry.ToNanos(sw.ElapsedTicks);
            yield return ("", true, promptTokenCount, generatedTokens.Count, kvCacheReusedTokens, totalNs, promptNs, evalNs);
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
            _sessions.ActivateSession(session ?? _sessions.IntrinsicSession);
            var activeSession = _sessions.ActiveSession;
            var model = _lifecycle.Model;

            string arch = model.Config.Architecture;
            var messages = new List<ChatMessage>
            {
                new ChatMessage { Role = "user", Content = prompt, ImagePaths = imagePaths }
            };

            using var generateScope = _telemetry.BeginInferenceScope(
                activeSession,
                _lifecycle.LoadedModelName,
                _lifecycle.LoadedBackend,
                "generate.stream");

            _telemetry.LogGenerateStarted(
                arch,
                maxTokens,
                imagePaths?.Count ?? 0,
                messages[0],
                samplingConfig);

            var preparedMessages = ChatHistoryPreparer.PrepareHistoryForInference(messages, arch, _logger);
            var inputTokens = _kvCacheRenderer.RenderToTokens(
                model.Tokenizer,
                model.Config.ChatTemplate,
                preparedMessages,
                arch,
                addGenerationPrompt: true);

            inputTokens = model.MultimodalInjector.ProcessPromptTokens(preparedMessages, inputTokens);
            inputTokens = _sessions.TruncatePromptToContext(activeSession, inputTokens, maxTokens);

            _sessions.ResetSession(activeSession);

            var sw = Stopwatch.StartNew();
            bool queuedPromptEmbeddings = model.MultimodalInjector.QueuePromptEmbeddings(0);
            var promptArray = inputTokens.ToArray();
            float[] logits = _sessions.ForwardPromptPrefill(promptArray, allowChunking: !queuedPromptEmbeddings);
            activeSession.KVCache.RecordAppend(promptArray, logits);
            long promptNs = InferenceTelemetry.ToNanos(sw.ElapsedTicks);
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
                if (model.Tokenizer.IsEos(nextToken))
                {
                    finishReason = "eos";
                    break;
                }

                generatedTokens.Add(nextToken);
                model.Tokenizer.AppendTokenBytes(nextToken, rawBytes);
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
                logits = model.Forward(new[] { nextToken });
                activeSession.KVCache.RecordAppend(nextToken, logits);
            }

            evalSw.Stop();
            sw.Stop();
            long evalNs = InferenceTelemetry.ToNanos(evalSw.ElapsedTicks);
            long totalNs = InferenceTelemetry.ToNanos(sw.ElapsedTicks);

            double evalSeconds = evalSw.Elapsed.TotalSeconds;
            double tokensPerSecond = (evalSeconds > 0 && generatedTokens.Count > 0)
                ? generatedTokens.Count / evalSeconds
                : 0;
            string completionText = Encoding.UTF8.GetString(rawBytes.ToArray());

            const int kvCacheReusedTokens = 0;
            const double kvCacheReusePercent = 0.0;
            _telemetry.LogGenerateFinished(
                wasCancelled,
                generatedTokens.Count,
                promptTokenCount,
                kvCacheReusedTokens,
                kvCacheReusePercent,
                sw.Elapsed.TotalMilliseconds,
                tokensPerSecond,
                finishReason,
                completionText);

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
    }
}
