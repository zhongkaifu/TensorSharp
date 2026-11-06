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
using System.Text.Encodings.Web;
using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.Extensions.Logging;

namespace TensorSharp.Server
{
    internal sealed class InferenceTelemetry
    {
        private readonly ILogger _logger;

        private const int MaxLoggedMessageChars = 512;

        // Reused JSON options for the per-turn input-summary serializer below. Relaxed
        // escaping keeps non-ASCII content readable in the log file instead of
        // expanding it to \uXXXX escapes; control characters are still escaped by
        // JsonSerializer so each entry stays on a single line.
        private static readonly JsonSerializerOptions FullInputJsonOptions = new()
        {
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
            Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
        };

        public InferenceTelemetry(ILogger logger)
        {
            _logger = logger ?? Microsoft.Extensions.Logging.Abstractions.NullLogger.Instance;
        }

        public IDisposable BeginInferenceScope(
            ChatSession session,
            string modelName,
            string backend,
            string operation)
        {
            return _logger.BeginScope(new Dictionary<string, object>(StringComparer.Ordinal)
            {
                [LogScopeKeys.SessionId] = session.Id,
                [LogScopeKeys.Model] = modelName ?? "(none)",
                [LogScopeKeys.Backend] = backend ?? "(none)",
                [LogScopeKeys.Operation] = operation,
            });
        }

        /// <summary>
        /// Compact stop-sequence rendering for the per-request log lines: the
        /// count plus the sequences themselves, since a stray stop string is a
        /// common cause of "the model answered with one word".
        /// </summary>
        private static string DescribeStopSequences(SamplingConfig cfg)
        {
            var stops = cfg?.StopSequences;
            if (stops == null || stops.Count == 0)
                return "(none)";
            return "[" + string.Join(",", stops) + "]";
        }

        public void LogChatStarted(
            string arch,
            int maxTokens,
            bool enableThinking,
            List<ToolFunction> tools,
            List<ChatMessage> preparedHistory,
            SamplingConfig samplingConfig)
        {
            if (!_logger.IsEnabled(LogLevel.Information))
                return;

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
                    else if (m.Role is "system" or "developer") systemMessageCount++;
                    if (m.ImagePaths != null) imageAttachments += m.ImagePaths.Count;
                    if (m.AudioPaths != null) audioAttachments += m.AudioPaths.Count;
                    if (m.TextFilePaths != null) textFileAttachments += m.TextFilePaths.Count;
                }
            }

            string lastUserPreview = BuildMessageContentForLog(
                lastUserMessage, out _, out _);
            string lastUserContent = LoggingExtensions.SanitizeForLog(
                lastUserPreview, MaxLoggedMessageChars);
            string turnUploads = SerializeUploadsForLog(lastUserMessage);
            string fullInput = SerializeMessagesForLog(preparedHistory);

            var effective = samplingConfig ?? new SamplingConfig();
            _logger.LogInformation(LogEventIds.ChatStarted,
                "chat.start arch={Architecture}{Origin} maxTokens={MaxTokens} thinking={EnableThinking} tools={ToolCount} messages(user={UserMessages},assistant={AssistantMessages},system={SystemMessages}) attachments(image={ImageCount},audio={AudioCount},textFile={TextFileCount}) uploads={Uploads} sampling(temp={Temperature},topK={TopK},topP={TopP},minP={MinP},repeatPenalty={RepeatPenalty},repeatLastN={RepeatLastN},presencePenalty={PresencePenalty},frequencyPenalty={FrequencyPenalty},seed={Seed},stop={StopSequences}) userInput=\"{LastUserContent}\" fullInput={FullInput}",
                arch, RequestOrigin(), maxTokens, enableThinking, tools?.Count ?? 0,
                userMessageCount, assistantMessageCount, systemMessageCount,
                imageAttachments, audioAttachments, textFileAttachments,
                turnUploads,
                effective.Temperature, effective.TopK, effective.TopP, effective.MinP,
                effective.RepetitionPenalty, effective.PenaltyLastN,
                effective.PresencePenalty, effective.FrequencyPenalty, effective.Seed,
                DescribeStopSequences(effective),
                lastUserContent, fullInput);
        }

        /// <summary>
        /// " origin=prefix-cache-warmup" for the host's own one-token warm-up request
        /// (<see cref="Hosting.PrefixCacheWarmup.IsRunningHere"/>), empty for a client's.
        /// Placed right after the line's first value so parsers keyed on
        /// <c>chat.start arch=</c> / <c>chat.complete tokens=N </c> still match.
        /// </summary>
        private static string RequestOrigin() =>
            Hosting.PrefixCacheWarmup.IsRunningHere ? " origin=prefix-cache-warmup" : string.Empty;

        public void LogChatFinished(
            bool wasCancelled,
            int generatedTokenCount,
            int promptTokenCount,
            int kvCacheReusedTokens,
            double kvCacheReusePercent,
            long timeToFirstTokenMs,
            double elapsedMs,
            double tokensPerSecond,
            string finishReason,
            string assistantText)
        {
            string assistantContent = LoggingExtensions.SanitizeForLogFull(assistantText);

            if (wasCancelled)
            {
                _logger.LogWarning(LogEventIds.ChatAborted,
                    "chat.cancelled tokens={Tokens} promptTokens={PromptTokens} kvReused={KvReusedTokens} kvReusePercent={KvReusePercent:F1} ttftMs={TimeToFirstTokenMs} elapsedMs={ElapsedMs:F1} assistantOutput=\"{AssistantContent}\"",
                    generatedTokenCount, promptTokenCount, kvCacheReusedTokens, kvCacheReusePercent,
                    timeToFirstTokenMs, elapsedMs, assistantContent);
            }
            else
            {
                _logger.LogInformation(LogEventIds.ChatCompleted,
                    "chat.complete tokens={Tokens}{Origin} promptTokens={PromptTokens} kvReused={KvReusedTokens} kvReusePercent={KvReusePercent:F1} ttftMs={TimeToFirstTokenMs} elapsedMs={ElapsedMs:F1} tokensPerSec={TokensPerSec:F2} finishReason={FinishReason} assistantOutput=\"{AssistantContent}\"",
                    generatedTokenCount, RequestOrigin(), promptTokenCount, kvCacheReusedTokens, kvCacheReusePercent,
                    timeToFirstTokenMs, elapsedMs, tokensPerSecond, finishReason, assistantContent);
            }
        }

        public void LogGenerateStarted(
            string arch,
            int maxTokens,
            int imageAttachmentCount,
            ChatMessage promptMessage,
            SamplingConfig samplingConfig)
        {
            if (!_logger.IsEnabled(LogLevel.Information))
                return;

            string promptPreview = BuildMessageContentForLog(
                promptMessage, out _, out _);
            string promptContent = LoggingExtensions.SanitizeForLog(
                promptPreview, MaxLoggedMessageChars);
            string turnUploads = SerializeUploadsForLog(promptMessage);
            var effective = samplingConfig ?? new SamplingConfig();
            _logger.LogInformation(LogEventIds.ChatStarted,
                "generate.start arch={Architecture} maxTokens={MaxTokens} imageAttachments={ImageCount} uploads={Uploads} sampling(temp={Temperature},topK={TopK},topP={TopP},minP={MinP},repeatPenalty={RepeatPenalty},repeatLastN={RepeatLastN},presencePenalty={PresencePenalty},frequencyPenalty={FrequencyPenalty},seed={Seed},stop={StopSequences}) prompt=\"{Prompt}\"",
                arch, maxTokens, imageAttachmentCount, turnUploads,
                effective.Temperature, effective.TopK, effective.TopP, effective.MinP,
                effective.RepetitionPenalty, effective.PenaltyLastN,
                effective.PresencePenalty, effective.FrequencyPenalty, effective.Seed,
                DescribeStopSequences(effective),
                promptContent);
        }

        public void LogGenerateFinished(
            bool wasCancelled,
            int generatedTokenCount,
            int promptTokenCount,
            int kvCacheReusedTokens,
            double kvCacheReusePercent,
            double elapsedMs,
            double tokensPerSecond,
            string finishReason,
            string completionText)
        {
            string completionContent = LoggingExtensions.SanitizeForLogFull(completionText);

            if (wasCancelled)
            {
                _logger.LogWarning(LogEventIds.ChatAborted,
                    "generate.cancelled tokens={Tokens} promptTokens={PromptTokens} kvReused={KvReusedTokens} kvReusePercent={KvReusePercent:F1} elapsedMs={ElapsedMs:F1} completion=\"{Completion}\"",
                    generatedTokenCount, promptTokenCount, kvCacheReusedTokens, kvCacheReusePercent,
                    elapsedMs, completionContent);
            }
            else
            {
                _logger.LogInformation(LogEventIds.ChatCompleted,
                    "generate.complete tokens={Tokens} promptTokens={PromptTokens} kvReused={KvReusedTokens} kvReusePercent={KvReusePercent:F1} elapsedMs={ElapsedMs:F1} tokensPerSec={TokensPerSec:F2} finishReason={FinishReason} completion=\"{Completion}\"",
                    generatedTokenCount, promptTokenCount, kvCacheReusedTokens, kvCacheReusePercent,
                    elapsedMs, tokensPerSecond, finishReason, completionContent);
            }
        }

        public static long ToNanos(long elapsedTicks)
            => elapsedTicks * (1_000_000_000L / Stopwatch.Frequency);

        /// <summary>
        /// Serialize a bounded summary of the conversation submitted for this turn.
        /// Message bodies are capped so a losslessly uploaded document is not copied
        /// wholesale into telemetry (which would add avoidable memory, I/O, and
        /// sensitive-content exposure). Original character counts remain available.
        /// </summary>
        public static string SerializeMessagesForLog(List<ChatMessage> messages)
        {
            if (messages == null || messages.Count == 0)
                return "[]";

            var entries = new List<ChatMessageLogEntry>(messages.Count);
            foreach (var m in messages)
            {
                if (m == null) continue;
                string contentPreview = BuildMessageContentForLog(
                    m, out bool contentOmitted, out bool contentTruncated);
                entries.Add(new ChatMessageLogEntry
                {
                    Role = m.Role ?? string.Empty,
                    Content = contentPreview,
                    ContentChars = m.Content?.Length ?? 0,
                    ContentOmitted = contentOmitted ? true : (bool?)null,
                    ContentTruncated = contentTruncated ? true : (bool?)null,
                    Images = ToPathList(m.ImagePaths),
                    Audios = ToPathList(m.AudioPaths),
                    TextFiles = ToPathList(m.TextFilePaths),
                    IsVideo = m.IsVideo ? true : (bool?)null,
                    Thinking = string.IsNullOrEmpty(m.Thinking) ? null : LimitStructuredLogValue(m.Thinking),
                    ToolCallCount = (m.ToolCalls != null && m.ToolCalls.Count > 0) ? m.ToolCalls.Count : (int?)null,
                });
            }

            return JsonSerializer.Serialize(entries, FullInputJsonOptions);
        }

        /// <summary>
        /// Serialize the upload manifest for a single message as a single-line JSON array.
        /// </summary>
        public static string SerializeUploadsForLog(ChatMessage message)
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

        private static string LimitStructuredLogValue(string value)
        {
            if (string.IsNullOrEmpty(value))
                return string.Empty;
            if (value.Length <= MaxLoggedMessageChars)
                return value;

            return value.Substring(0, MaxLoggedMessageChars) +
                $"...(+{value.Length - MaxLoggedMessageChars} chars)";
        }

        private static string BuildMessageContentForLog(
            ChatMessage message,
            out bool contentOmitted,
            out bool contentTruncated)
        {
            contentOmitted = false;
            contentTruncated = false;
            string content = message?.Content ?? string.Empty;
            if (content.Length == 0)
                return string.Empty;

            const string endMarker = "[End of file]";
            int endMarkerIndex = content.LastIndexOf(endMarker, StringComparison.OrdinalIgnoreCase);
            bool hasDocumentEnvelope = endMarkerIndex >= 0 &&
                (content.IndexOf("[File:", StringComparison.OrdinalIgnoreCase) >= 0 ||
                 content.IndexOf("[Attached file:", StringComparison.OrdinalIgnoreCase) >= 0);
            bool hasTextFilePath = message?.TextFilePaths != null && message.TextFilePaths.Count > 0;

            if (hasDocumentEnvelope || (hasTextFilePath && content.Length > MaxLoggedMessageChars))
            {
                contentOmitted = true;
                string summary = $"[attached document omitted from log; {content.Length} chars]";
                if (hasDocumentEnvelope)
                {
                    int instructionStart = endMarkerIndex + endMarker.Length;
                    while (instructionStart < content.Length && char.IsWhiteSpace(content[instructionStart]))
                        instructionStart++;
                    if (instructionStart < content.Length)
                    {
                        int instructionLength = content.Length - instructionStart;
                        string instruction = instructionLength <= MaxLoggedMessageChars
                            ? content.Substring(instructionStart, instructionLength)
                            : content.Substring(instructionStart, MaxLoggedMessageChars) +
                              $"...(+{instructionLength - MaxLoggedMessageChars} chars)";
                        summary += " instruction=\"" + instruction + "\"";
                    }
                }
                return summary;
            }

            contentTruncated = content.Length > MaxLoggedMessageChars;
            return LimitStructuredLogValue(content);
        }

        private sealed class ChatMessageLogEntry
        {
            public string Role { get; init; }
            public string Content { get; init; }
            public int ContentChars { get; init; }
            public bool? ContentOmitted { get; init; }
            public bool? ContentTruncated { get; init; }
            public List<string> Images { get; init; }
            public List<string> Audios { get; init; }
            public List<string> TextFiles { get; init; }
            public bool? IsVideo { get; init; }
            public string Thinking { get; init; }
            public int? ToolCallCount { get; init; }
        }

        private sealed class UploadLogEntry
        {
            public string Path { get; init; }
            public string Name { get; init; }
            public string MediaType { get; init; }
        }
    }
}
