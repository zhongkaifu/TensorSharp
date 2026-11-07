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
using System.Threading;

namespace TensorSharp.Runtime.Telemetry
{
    /// <summary>
    /// The span and the metrics of one generation request (chat, completion or embeddings),
    /// in one object so a call site states facts and this class decides how they are
    /// recorded.
    ///
    /// <para>
    /// The span is held here rather than left in <see cref="Activity.Current"/>: the chat
    /// pipeline is an async iterator, and a value set in <c>Activity.Current</c> inside one is
    /// not reliably current after a <c>yield</c>, because the next <c>MoveNextAsync</c> can
    /// resume under the consumer's execution context. Children are therefore parented
    /// explicitly on <see cref="Context"/>.
    /// </para>
    /// <para>
    /// The engine runs on its own worker thread, so its phases are not spans that thread
    /// opens. <see cref="RecordEnginePhases"/> builds them when the request finishes, from
    /// the timestamps the scheduler already records.
    /// </para>
    /// <para>
    /// Every exit records <c>gen_ai.server.request.duration</c> exactly once, in
    /// <see cref="Dispose"/>, with the outcome the call site reported. Thread-safe for one
    /// writer at a time, which is how a request uses it.
    /// </para>
    /// </summary>
    public sealed class GenAiRequestTelemetry : IDisposable
    {
        private readonly Activity? _activity;
        private readonly long _startTimestamp;
        private readonly DateTime _startUtc;
        private readonly string _operation;
        private readonly string _model;
        private readonly string? _backend;
        private readonly string? _protocol;
        private string _outcome = TelemetryAttributes.OutcomeCompleted;
        private string? _errorType;
        private string? _origin;
        private int _disposed;

        private GenAiRequestTelemetry(string operation, string model, string? backend, ActivityContext parent)
        {
            _operation = operation;
            _model = model;
            _backend = backend;
            _protocol = TelemetryContext.Protocol;
            _startTimestamp = Stopwatch.GetTimestamp();
            _startUtc = DateTime.UtcNow;
            if (TelemetryContext.Origin is { Length: > 0 } contextOrigin)
                _origin = contextOrigin;

            // With nobody listening (telemetry off), skip the tag collection and the span
            // name: StartActivity would return null anyway, and "off" should cost nothing.
            if (!TensorSharpTelemetry.Inference.HasListeners())
                return;

            var tags = new ActivityTagsCollection
            {
                { TelemetryAttributes.GenAiOperationName, operation },
                { TelemetryAttributes.GenAiProviderName, TensorSharpTelemetry.ProviderName },
                { TelemetryAttributes.GenAiRequestModel, model },
            };
            if (!string.IsNullOrEmpty(backend)) tags.Add(TelemetryAttributes.Backend, backend);
            if (!string.IsNullOrEmpty(_protocol)) tags.Add(TelemetryAttributes.Protocol, _protocol);
            if (TelemetryContext.RequestId is { Length: > 0 } requestId) tags.Add(TelemetryAttributes.RequestId, requestId);
            if (TelemetryContext.Round is int round) tags.Add(TelemetryAttributes.Round, round);
            if (_origin != null)
            {
                tags.Add(TelemetryAttributes.Origin, _origin);
                tags.Add(TelemetryAttributes.Internal, true);
            }

            // A default parent context means "whatever is current", normally the HTTP server span.
            _activity = TensorSharpTelemetry.Inference.StartActivity($"{operation} {model}", ActivityKind.Internal, parent, tags);
        }

        /// <summary>
        /// Start the request. <paramref name="operation"/> is a <c>gen_ai.operation.name</c>
        /// value (<see cref="TelemetryAttributes.OperationChat"/> and friends);
        /// <paramref name="model"/> the model name dashboards should group by (a file name,
        /// never a path). With no explicit <paramref name="parent"/> the span is a child of
        /// whatever is current (normally the HTTP server span).
        /// </summary>
        public static GenAiRequestTelemetry Start(string operation, string? model, string? backend, ActivityContext parent = default) =>
            new(operation, string.IsNullOrEmpty(model) ? "unknown" : model, backend, parent);

        /// <summary>The span, or null when nobody records it.</summary>
        public Activity? Activity => _activity;

        /// <summary>Parent for this request's child spans; <c>default</c> when the span is not recorded.</summary>
        public ActivityContext Context => _activity?.Context ?? default;

        /// <summary>Whether attributes set now will be exported. Guard expensive values with it.</summary>
        public bool IsRecording => _activity is { IsAllDataRequested: true };

        /// <summary>When the request started, for time-to-first-token.</summary>
        public DateTime StartedAtUtc => _startUtc;

        public void SetTag(string key, object? value)
        {
            if (_activity is { IsAllDataRequested: true })
                _activity.SetTag(key, value);
        }

        /// <summary>The sampling the request asked for. Nulls are left off.</summary>
        public void SetSampling(int? maxTokens, double? temperature, double? topP, int? topK, long? seed = null)
        {
            if (!IsRecording) return;
            if (maxTokens is > 0) _activity!.SetTag(TelemetryAttributes.GenAiRequestMaxTokens, maxTokens.Value);
            if (temperature.HasValue && double.IsFinite(temperature.Value)) _activity!.SetTag(TelemetryAttributes.GenAiRequestTemperature, temperature.Value);
            if (topP.HasValue && double.IsFinite(topP.Value)) _activity!.SetTag(TelemetryAttributes.GenAiRequestTopP, topP.Value);
            if (topK is > 0) _activity!.SetTag(TelemetryAttributes.GenAiRequestTopK, topK.Value);
            if (seed.HasValue && seed.Value >= 0) _activity!.SetTag(TelemetryAttributes.GenAiRequestSeed, seed.Value);
        }

        /// <summary>Tag the host's own startup warm-up request so dashboards can leave it out.</summary>
        public void SetOrigin(string? origin)
        {
            if (string.IsNullOrEmpty(origin)) return;
            _origin = origin;
            SetTag(TelemetryAttributes.Origin, origin);
            SetTag(TelemetryAttributes.Internal, true);
        }

        /// <summary>
        /// Prompt messages, produced only when content capture is on and the span is recorded
        /// (the factory is never called otherwise).
        /// </summary>
        public void SetInputMessages(Func<string?> messagesJson)
        {
            if (TensorSharpTelemetry.CaptureMessageContent && IsRecording && messagesJson() is { } json)
                _activity!.SetTag(TelemetryAttributes.GenAiInputMessages, json);
        }

        /// <summary>Model output; same rules as <see cref="SetInputMessages"/>.</summary>
        public void SetOutputMessages(Func<string?> messagesJson)
        {
            if (TensorSharpTelemetry.CaptureMessageContent && IsRecording && messagesJson() is { } json)
                _activity!.SetTag(TelemetryAttributes.GenAiOutputMessages, json);
        }

        /// <summary>
        /// A child span of this request, started now. Null when this request is not recorded,
        /// so a child is never accidentally started as a root of its own trace.
        /// </summary>
        public Activity? StartChild(string name, ActivitySource? source = null)
        {
            if (_activity is null) return null;
            return (source ?? TensorSharpTelemetry.Inference).StartActivity(name, ActivityKind.Internal, _activity.Context);
        }

        /// <summary>
        /// Build <c>inference.queue</c>, <c>inference.prefill</c> and <c>inference.decode</c>
        /// from the scheduler's timestamps (UTC). A phase whose start is unknown (a request
        /// aborted before it was scheduled, or before its first token) is left out. Wall time
        /// and compute time differ under continuous batching, so the compute share
        /// (<paramref name="prefillTicks"/> / <paramref name="decodeTicks"/>, in
        /// <see cref="Stopwatch"/> ticks) goes on the spans as attributes.
        /// </summary>
        public void RecordEnginePhases(DateTime submittedAtUtc, DateTime? firstScheduledAtUtc, DateTime? firstTokenAtUtc,
            DateTime endUtc, long prefillTicks = 0, long decodeTicks = 0, int preemptions = 0)
        {
            if (_activity is null) return;
            if (preemptions > 0) SetTag(TelemetryAttributes.Preemptions, preemptions);
            if (firstScheduledAtUtc is { } scheduled)
            {
                AddPhase("inference.queue", submittedAtUtc, scheduled, null, 0);
                // A first token before the first admission means the timestamps were not both
                // from the first admission; leave the prefill and decode split out rather than
                // draw a negative span.
                if (firstTokenAtUtc is { } firstToken && firstToken >= scheduled)
                {
                    AddPhase("inference.prefill", scheduled, firstToken, TelemetryAttributes.ComputePrefillMs, prefillTicks);
                    AddPhase("inference.decode", firstToken, endUtc, TelemetryAttributes.ComputeDecodeMs, decodeTicks);
                }
                else if (firstTokenAtUtc is null)
                {
                    AddPhase("inference.prefill", scheduled, endUtc, TelemetryAttributes.ComputePrefillMs, prefillTicks);
                }
            }
            else
            {
                AddPhase("inference.queue", submittedAtUtc, endUtc, null, 0);
            }
        }

        private void AddPhase(string name, DateTime startUtc, DateTime endUtc, string? computeKey, long computeTicks)
        {
            // A child never starts before its parent: a caller that built its sequence (which
            // stamps SubmittedAt) a moment before starting this request's span would otherwise
            // draw the queue phase a few microseconds outside the chat span.
            DateTime parentStart = _activity!.StartTimeUtc;
            if (startUtc < parentStart) startUtc = parentStart;
            if (endUtc < startUtc) endUtc = startUtc;
            var phase = TensorSharpTelemetry.Inference.StartActivity(
                name, ActivityKind.Internal, _activity!.Context,
                startTime: TensorSharpTelemetry.ToUtcOffset(startUtc));
            if (phase is null) return;
            if (computeKey != null && computeTicks > 0 && phase.IsAllDataRequested)
                phase.SetTag(computeKey, Math.Round(computeTicks * 1000.0 / Stopwatch.Frequency, 3));
            phase.SetEndTime(TensorSharpTelemetry.ToUtcOffset(endUtc).UtcDateTime);
            phase.Stop();
        }

        private TimeSpan? _timeToFirstToken;
        private double? _timePerOutputToken;

        /// <summary>
        /// Token usage and timing of a finished request. Token counters are recorded now, for
        /// every outcome (the tokens were processed either way). Time to first token (from this
        /// request's start to <paramref name="firstTokenAtUtc"/>) and time per output token (wall
        /// time after the first token divided by the tokens after it) are recorded by
        /// <see cref="Dispose"/>, and only for a completed request: the GenAI conventions define
        /// both for successful responses.
        /// </summary>
        public void RecordUsage(int inputTokens, int outputTokens, int cachedTokens, string? finishReason,
            DateTime? firstTokenAtUtc, DateTime? endUtc = null)
        {
            DateTime end = endUtc ?? DateTime.UtcNow;
            if (firstTokenAtUtc is { } first && first >= _startUtc)
            {
                _timeToFirstToken = first - _startUtc;
                if (outputTokens > 1 && end > first)
                    _timePerOutputToken = (end - first).TotalSeconds / (outputTokens - 1);
            }

            if (_activity is { IsAllDataRequested: true })
            {
                _activity.SetTag(TelemetryAttributes.GenAiUsageInputTokens, inputTokens);
                _activity.SetTag(TelemetryAttributes.GenAiUsageOutputTokens, outputTokens);
                if (cachedTokens > 0)
                {
                    _activity.SetTag(TelemetryAttributes.GenAiUsageCacheReadInputTokens, cachedTokens);
                    _activity.SetTag(TelemetryAttributes.PrefixCacheReusedTokens, cachedTokens);
                }
                if (!string.IsNullOrEmpty(finishReason))
                    _activity.SetTag(TelemetryAttributes.GenAiResponseFinishReasons, new[] { finishReason });
                if (_timeToFirstToken is { } ttft)
                    _activity.SetTag(TelemetryAttributes.TimeToFirstTokenMs, Math.Round(ttft.TotalMilliseconds, 1));
            }

            TagList tags = MetricTags(includeOutcome: false);
            if (inputTokens > 0) Tokens(inputTokens, TelemetryAttributes.TokenTypeInput, tags);
            if (outputTokens > 0) Tokens(outputTokens, TelemetryAttributes.TokenTypeOutput, tags);
            if (cachedTokens > 0) TensorSharpMetrics.TokensCacheRead.Add(cachedTokens, tags);
        }

        /// <summary>
        /// Time to first token for a path that measures it itself (no scheduler timestamps).
        /// Recorded by <see cref="Dispose"/> if the request completes.
        /// </summary>
        public void RecordTimeToFirstToken(TimeSpan timeToFirstToken)
        {
            if (timeToFirstToken < TimeSpan.Zero) return;
            _timeToFirstToken = timeToFirstToken;
            SetTag(TelemetryAttributes.TimeToFirstTokenMs, Math.Round(timeToFirstToken.TotalMilliseconds, 1));
        }

        public void SetResponseId(string? id)
        {
            if (!string.IsNullOrEmpty(id)) SetTag(TelemetryAttributes.GenAiResponseId, id);
        }

        /// <summary>The client went away or the request was stopped. Not an error.</summary>
        public void MarkCancelled()
        {
            _outcome = TelemetryAttributes.OutcomeCancelled;
            SetTag(TelemetryAttributes.Outcome, _outcome);
        }

        /// <summary>
        /// The request was refused before generating (a 4xx the caller can fix).
        /// <paramref name="errorType"/> is a short stable code, never a message.
        /// </summary>
        public void MarkRefused(string errorType)
        {
            _outcome = TelemetryAttributes.OutcomeRefused;
            _errorType = errorType;
            SetTag(TelemetryAttributes.Outcome, _outcome);
            SetTag(TelemetryAttributes.ErrorType, errorType);
        }

        /// <summary>
        /// The request failed. Records the exception type, never its message: messages can
        /// quote the prompt. Cancellation exceptions count as <see cref="MarkCancelled"/>.
        /// </summary>
        public void MarkFailed(Exception exception)
        {
            if (exception is OperationCanceledException)
            {
                MarkCancelled();
                return;
            }
            _outcome = TelemetryAttributes.OutcomeFailed;
            _errorType = exception.GetType().FullName ?? exception.GetType().Name;
            if (_activity is not null)
            {
                _activity.SetStatus(ActivityStatusCode.Error, exception.GetType().Name);
                if (_activity.IsAllDataRequested)
                {
                    _activity.SetTag(TelemetryAttributes.Outcome, _outcome);
                    _activity.SetTag(TelemetryAttributes.ErrorType, _errorType);
                }
            }
        }

        /// <summary>Stop the span and record the request duration, once.</summary>
        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
            double seconds = Stopwatch.GetElapsedTime(_startTimestamp).TotalSeconds;
            TensorSharpMetrics.RequestDuration.Record(seconds, MetricTags(includeOutcome: true));
            if (_outcome == TelemetryAttributes.OutcomeCompleted)
            {
                TagList tags = MetricTags(includeOutcome: false);
                if (_timeToFirstToken is { } ttft)
                    TensorSharpMetrics.TimeToFirstToken.Record(ttft.TotalSeconds, tags);
                if (_timePerOutputToken is { } tpot)
                    TensorSharpMetrics.TimePerOutputToken.Record(tpot, tags);
            }
            if (_activity is not null)
            {
                if (_activity.IsAllDataRequested && _outcome == TelemetryAttributes.OutcomeCompleted)
                    _activity.SetTag(TelemetryAttributes.Outcome, _outcome);
                _activity.Stop();
            }
        }

        private static void Tokens(int count, string type, TagList baseTags)
        {
            TagList tags = baseTags;
            tags.Add(TelemetryAttributes.TokenType, type);
            TensorSharpMetrics.Tokens.Add(count, tags);
        }

        private TagList MetricTags(bool includeOutcome)
        {
            var tags = new TagList
            {
                { TelemetryAttributes.GenAiOperationName, _operation },
                { TelemetryAttributes.GenAiProviderName, TensorSharpTelemetry.ProviderName },
                { TelemetryAttributes.GenAiRequestModel, _model },
            };
            if (!string.IsNullOrEmpty(_backend)) tags.Add(TelemetryAttributes.Backend, _backend);
            if (!string.IsNullOrEmpty(_protocol)) tags.Add(TelemetryAttributes.Protocol, _protocol);
            if (_origin != null) tags.Add(TelemetryAttributes.Origin, _origin);
            if (includeOutcome)
            {
                tags.Add(TelemetryAttributes.Outcome, _outcome);
                if (_errorType != null) tags.Add(TelemetryAttributes.ErrorType, _errorType);
            }
            return tags;
        }

        /// <summary>A model file name for <c>gen_ai.request.model</c>: no directory, no extension.</summary>
        public static string ModelNameFromPath(string? path)
        {
            if (string.IsNullOrWhiteSpace(path)) return "unknown";
            string name = System.IO.Path.GetFileNameWithoutExtension(path.Trim());
            return string.IsNullOrEmpty(name) ? "unknown" : name;
        }
    }
}
