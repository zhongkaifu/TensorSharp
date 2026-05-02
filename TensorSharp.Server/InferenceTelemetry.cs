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

        // Reused JSON options for the per-turn fullInput serializer below. Relaxed
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

        public void LogChatStarted(
            string arch,
            int maxTokens,
            bool enableThinking,
            List<ToolFunction> tools,
            List<ChatMessage> preparedHistory,
            SamplingConfig samplingConfig)
        {
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

            string lastUserContent = LoggingExtensions.SanitizeForLogFull(lastUserMessage?.Content ?? string.Empty);
            string turnUploads = SerializeUploadsForLog(lastUserMessage);
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
        }

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
                    "chat.complete tokens={Tokens} promptTokens={PromptTokens} kvReused={KvReusedTokens} kvReusePercent={KvReusePercent:F1} ttftMs={TimeToFirstTokenMs} elapsedMs={ElapsedMs:F1} tokensPerSec={TokensPerSec:F2} finishReason={FinishReason} assistantOutput=\"{AssistantContent}\"",
                    generatedTokenCount, promptTokenCount, kvCacheReusedTokens, kvCacheReusePercent,
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
            string promptContent = LoggingExtensions.SanitizeForLogFull(promptMessage?.Content ?? string.Empty);
            string turnUploads = SerializeUploadsForLog(promptMessage);
            _logger.LogInformation(LogEventIds.ChatStarted,
                "generate.start arch={Architecture} maxTokens={MaxTokens} imageAttachments={ImageCount} uploads={Uploads} sampling(temp={Temperature},topK={TopK},topP={TopP},seed={Seed}) prompt=\"{Prompt}\"",
                arch, maxTokens, imageAttachmentCount, turnUploads,
                samplingConfig?.Temperature ?? 0.8f, samplingConfig?.TopK ?? 40,
                samplingConfig?.TopP ?? 0.9f, samplingConfig?.Seed ?? 0,
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
        /// Serialize the entire conversation array submitted for this turn into a
        /// single-line JSON string for logging.
        /// </summary>
        public static string SerializeMessagesForLog(List<ChatMessage> messages)
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

        private sealed class UploadLogEntry
        {
            public string Path { get; init; }
            public string Name { get; init; }
            public string MediaType { get; init; }
        }
    }
}
