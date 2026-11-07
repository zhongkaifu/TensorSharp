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
using System.Threading;
using Microsoft.Extensions.Logging;
using TensorSharp.Runtime.Telemetry;

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
        /// common cause of "the model answered with one word". The strings are the
        /// client's own request text, so they are marked as content; "(none)" is not.
        /// </summary>
        private static object DescribeStopSequences(SamplingConfig cfg)
        {
            var stops = cfg?.StopSequences;
            if (stops == null || stops.Count == 0)
                return "(none)";
            return LogContent.Of("[" + string.Join(",", stops) + "]");
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
            // Prompt text, the upload manifest (saved paths and original names) and the
            // conversation summary are content: LogContent prints them unchanged here and
            // lets the OpenTelemetry exporter redact them.
            _logger.LogInformation(LogEventIds.ChatStarted,
                "chat.start arch={Architecture}{Origin} maxTokens={MaxTokens} thinking={EnableThinking} tools={ToolCount} messages(user={UserMessages},assistant={AssistantMessages},system={SystemMessages}) attachments(image={ImageCount},audio={AudioCount},textFile={TextFileCount}) uploads={Uploads} sampling(temp={Temperature},topK={TopK},topP={TopP},minP={MinP},repeatPenalty={RepeatPenalty},repeatLastN={RepeatLastN},presencePenalty={PresencePenalty},frequencyPenalty={FrequencyPenalty},seed={Seed},stop={StopSequences}) userInput=\"{LastUserContent}\" fullInput={FullInput}",
                arch, RequestOrigin(), maxTokens, enableThinking, tools?.Count ?? 0,
                userMessageCount, assistantMessageCount, systemMessageCount,
                imageAttachments, audioAttachments, textFileAttachments,
                LogContent.Of(turnUploads),
                effective.Temperature, effective.TopK, effective.TopP, effective.MinP,
                effective.RepetitionPenalty, effective.PenaltyLastN,
                effective.PresencePenalty, effective.FrequencyPenalty, effective.Seed,
                DescribeStopSequences(effective),
                LogContent.Of(lastUserContent), LogContent.Of(fullInput));
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
            LogContent assistantContent = LogContent.Of(LoggingExtensions.SanitizeForLogFull(assistantText));

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
                arch, maxTokens, imageAttachmentCount, LogContent.Of(turnUploads),
                effective.Temperature, effective.TopK, effective.TopP, effective.MinP,
                effective.RepetitionPenalty, effective.PenaltyLastN,
                effective.PresencePenalty, effective.FrequencyPenalty, effective.Seed,
                DescribeStopSequences(effective),
                LogContent.Of(promptContent));
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
            LogContent completionContent = LogContent.Of(LoggingExtensions.SanitizeForLogFull(completionText));

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

    /// <summary>
    /// Sets <see cref="TelemetryContext.Round"/> or <see cref="TelemetryContext.Origin"/>
    /// for the model calls a block makes, and puts the previous value back on dispose.
    /// <see cref="GenAiRequestTelemetry"/> reads both when a generation starts.
    ///
    /// <para>
    /// Inside an async iterator a value set here is only reliably visible until the
    /// iterator's next <c>yield</c> (the next <c>MoveNextAsync</c> runs on the consumer's
    /// execution context), so set it immediately before the call that starts the
    /// generation. That is enough: the generation reads it when it starts.
    /// </para>
    /// </summary>
    internal readonly struct TelemetryContextScope : IDisposable
    {
        // 0 (default): the scope set nothing and restores nothing.
        private const byte RoundKind = 1, OriginKind = 2;

        private readonly byte _kind;
        private readonly int? _outerRound;
        private readonly string? _outerOrigin;

        private TelemetryContextScope(byte kind, int? outerRound, string? outerOrigin)
        {
            _kind = kind;
            _outerRound = outerRound;
            _outerOrigin = outerOrigin;
        }

        /// <summary>
        /// Mark the generations started in this scope as round <paramref name="round"/>
        /// (1-based) of a skills or multi-agent turn. Only spans carry the round, so with
        /// nothing listening for spans this sets nothing (an AsyncLocal write allocates).
        /// </summary>
        public static TelemetryContextScope Round(int round)
        {
            if (!TensorSharpTelemetry.Inference.HasListeners())
                return default;
            var scope = new TelemetryContextScope(RoundKind, TelemetryContext.Round, null);
            TelemetryContext.Round = round;
            return scope;
        }

        /// <summary>
        /// Mark the generations started in this scope as TensorSharp's own
        /// (<c>tensorsharp.origin</c>, <c>tensorsharp.internal=true</c>). The origin is a
        /// metric attribute too, so it is set whenever spans or the request metrics are
        /// being listened to.
        /// </summary>
        public static TelemetryContextScope Origin(string origin)
        {
            if (!TensorSharpTelemetry.Inference.HasListeners() && !TensorSharpMetrics.RequestDuration.Enabled)
                return default;
            var scope = new TelemetryContextScope(OriginKind, null, TelemetryContext.Origin);
            TelemetryContext.Origin = origin;
            return scope;
        }

        public void Dispose()
        {
            if (_kind == RoundKind)
                TelemetryContext.Round = _outerRound;
            else if (_kind == OriginKind)
                TelemetryContext.Origin = _outerOrigin;
        }
    }

    /// <summary>
    /// The <c>model.load</c> span and <c>tensorsharp.model.load.duration</c> sample of one
    /// model load, or the <c>model.unload</c> span of an explicit unload. A load runs
    /// synchronously on its caller's thread, so the span is simply a child of whatever is
    /// current (the Web UI request, nothing at startup) and is itself current while the
    /// load runs, which puts the loader's own log lines in its trace.
    ///
    /// <para>
    /// Start one, report <see cref="Loaded"/>, <see cref="Skipped"/> or
    /// <see cref="Failed"/>, and dispose it on every exit; the duration is recorded once,
    /// in <see cref="Dispose"/>.
    /// </para>
    /// </summary>
    internal sealed class ModelLoadTelemetry : IDisposable
    {
        // Span-only facts the shared attribute list has no name for.
        private const string MmProjFile = "tensorsharp.model.mmproj_file";
        private const string StartupLoad = "tensorsharp.model.startup_load";
        private const string LoadSkipped = "tensorsharp.model.load.skipped";
        private const string LayerSplitDegree = "tensorsharp.layer_split.degree";

        private readonly Activity? _activity;
        private readonly long _startTimestamp = Stopwatch.GetTimestamp();
        private readonly bool _recordDuration;
        private readonly string _model;
        private string _backend;
        private string _outcome = TelemetryAttributes.OutcomeCompleted;
        private string? _errorType;
        private bool _skipped;
        private int _disposed;

        private ModelLoadTelemetry(string spanName, bool recordDuration, string? modelPath, string? backend)
        {
            _recordDuration = recordDuration;
            _model = GenAiRequestTelemetry.ModelNameFromPath(modelPath);
            _backend = TelemetryBackend(backend);
            _activity = TensorSharpTelemetry.Models.StartActivity(spanName, ActivityKind.Internal);
            if (_activity is { IsAllDataRequested: true })
            {
                // The file name, never the path: a path names the operator's directories.
                if (FileName(modelPath) is { } file)
                    _activity.SetTag(TelemetryAttributes.ModelFile, file);
                _activity.SetTag(TelemetryAttributes.GenAiRequestModel, _model);
                _activity.SetTag(TelemetryAttributes.Backend, _backend);
            }
        }

        /// <summary>
        /// Start a <c>model.load</c>. <paramref name="backend"/> is the requested backend as
        /// the caller spelled it; <see cref="Loaded"/> replaces it with the one that loaded.
        /// </summary>
        public static ModelLoadTelemetry StartLoad(string? modelPath, string? mmProjPath, string? backend, bool startupLoad = false)
        {
            var telemetry = new ModelLoadTelemetry("model.load", recordDuration: true, modelPath, backend);
            if (telemetry._activity is { IsAllDataRequested: true } activity)
            {
                if (FileName(mmProjPath) is { } mmProj)
                    activity.SetTag(MmProjFile, mmProj);
                if (startupLoad)
                    activity.SetTag(StartupLoad, true);
            }
            return telemetry;
        }

        /// <summary>Start a <c>model.unload</c> of the model that is loaded now. Records no metric.</summary>
        public static ModelLoadTelemetry StartUnload(string? modelPath, string? architecture, string? backend)
        {
            var telemetry = new ModelLoadTelemetry("model.unload", recordDuration: false, modelPath, backend);
            if (!string.IsNullOrEmpty(architecture))
                telemetry.SetTag(TelemetryAttributes.ModelArchitecture, architecture);
            return telemetry;
        }

        /// <summary>
        /// The model loaded. Degrees of 1 (no tensor parallelism, no layer split) are left
        /// off the span.
        /// </summary>
        public void Loaded(string? architecture, string? backend, int tensorParallelDegree = 1, int layerSplitDegree = 1)
        {
            if (!string.IsNullOrEmpty(backend))
                _backend = TelemetryBackend(backend);
            if (_activity is not { IsAllDataRequested: true })
                return;
            _activity.SetTag(TelemetryAttributes.Backend, _backend);
            _activity.SetTag(TelemetryAttributes.ModelArchitecture, string.IsNullOrEmpty(architecture) ? "unknown" : architecture);
            if (tensorParallelDegree > 1)
                _activity.SetTag(TelemetryAttributes.TensorParallelDegree, tensorParallelDegree);
            if (layerSplitDegree > 1)
                _activity.SetTag(LayerSplitDegree, layerSplitDegree);
        }

        /// <summary>
        /// Nothing was loaded because the requested model already serves (a distributed
        /// host asked for the checkpoint it runs). No duration is recorded: a no-op is not
        /// a load time.
        /// </summary>
        public void Skipped()
        {
            _skipped = true;
            SetTag(LoadSkipped, true);
        }

        /// <summary>
        /// The load threw. Records the exception type, never its message (a message can
        /// name the operator's paths). A refusal (<see cref="ModelLoadRefusal"/>: a missing
        /// or invalid file, not enough memory, an unsupported layout) is <c>refused</c>,
        /// cancellation <c>cancelled</c>, anything else <c>failed</c>. Both refused and
        /// failed set the span's status to Error: unlike a refused chat request, which the
        /// server answered correctly, a refused load leaves the requested model unloaded.
        /// </summary>
        public void Failed(Exception exception)
        {
            if (exception is OperationCanceledException)
            {
                _outcome = TelemetryAttributes.OutcomeCancelled;
                SetTag(TelemetryAttributes.Outcome, _outcome);
                return;
            }
            Type type = exception.GetType();
            _outcome = ModelLoadRefusal.IsRefusal(exception) ? TelemetryAttributes.OutcomeRefused : TelemetryAttributes.OutcomeFailed;
            _errorType = type.FullName ?? type.Name;
            if (_activity is null)
                return;
            _activity.SetStatus(ActivityStatusCode.Error, type.Name);
            if (_activity.IsAllDataRequested)
            {
                _activity.SetTag(TelemetryAttributes.Outcome, _outcome);
                _activity.SetTag(TelemetryAttributes.ErrorType, _errorType);
            }
        }

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
                return;
            if (_recordDuration && !_skipped)
            {
                var tags = new TagList
                {
                    { TelemetryAttributes.GenAiRequestModel, _model },
                    { TelemetryAttributes.Backend, _backend },
                    { TelemetryAttributes.Outcome, _outcome },
                };
                if (_errorType != null)
                    tags.Add(TelemetryAttributes.ErrorType, _errorType);
                TensorSharpMetrics.ModelLoadDuration.Record(Stopwatch.GetElapsedTime(_startTimestamp).TotalSeconds, tags);
            }
            if (_activity is null)
                return;
            if (_activity.IsAllDataRequested && _outcome == TelemetryAttributes.OutcomeCompleted)
                _activity.SetTag(TelemetryAttributes.Outcome, _outcome);
            _activity.Stop();
        }

        private void SetTag(string key, object value)
        {
            if (_activity is { IsAllDataRequested: true })
                _activity.SetTag(key, value);
        }

        private static string? FileName(string? path)
        {
            if (string.IsNullOrWhiteSpace(path))
                return null;
            string name = Path.GetFileName(path.Trim());
            return string.IsNullOrEmpty(name) ? null : name;
        }

        /// <summary>
        /// A backend value fit for a metric attribute: the canonical name of a backend this
        /// host knows, <c>unknown</c> for anything else (so a mistyped name cannot add a
        /// series), and the lifecycle's default (<c>ggml_cpu</c>) when none was named.
        /// </summary>
        private static string TelemetryBackend(string? backend)
        {
            if (string.IsNullOrWhiteSpace(backend))
                return BackendCatalog.ToBackendValue(BackendType.GgmlCpu);
            string? canonical = BackendCatalog.Canonicalize(backend);
            return canonical != null && BackendCatalog.IsKnownBackend(canonical) ? canonical : "unknown";
        }
    }
}
