// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
//
// Traces and metrics for one generated clip, shared by every host that calls
// IVideoGenerationModel.GenerateVideo (the server's plain and streaming paths, Studio
// through GenerateVideoAsync, the CLI). The models only report progress; every span is
// opened here, with an explicit parent, because on MLX a model re-enters on the MLX worker
// thread and reports progress from there, where Activity.Current belongs to nobody.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;
using Microsoft.Extensions.Logging;
using TensorSharp.Models.Telemetry;
using TensorSharp.Models.WanVideo;
using TensorSharp.Runtime.Logging;
using TensorSharp.Runtime.Telemetry;

namespace TensorSharp.Models.Video
{
    /// <summary>
    /// The <c>generate_video &lt;model&gt;</c> span and the
    /// <c>tensorsharp.video.generation.duration</c> metric of one clip, from the moment a host
    /// accepts it (including its wait for the video lock) until it is saved.
    ///
    /// <para>
    /// Held by the host rather than left in <see cref="System.Diagnostics.Activity.Current"/>:
    /// the streaming route is an async iterator and the generation runs on another thread, so
    /// children are parented on <see cref="Context"/>. Every exit records the duration exactly
    /// once, in <see cref="Dispose"/>, with the outcome the host reported. One writer at a time.
    /// </para>
    /// </summary>
    public sealed class VideoClipTelemetry : IDisposable
    {
        private const string MediaKindVideo = "video";

        private readonly Activity? _activity;
        private readonly long _startTimestamp;
        private string _outcome = TelemetryAttributes.OutcomeCompleted;
        private string? _errorType;
        private int _disposed;

        private VideoClipTelemetry(string model, string? backend, VideoGenerationParams parameters, string? prompt, ActivityContext parent)
        {
            Model = model;
            Backend = string.IsNullOrEmpty(backend) ? null : backend;
            Mode = NormalizeMode(parameters);
            Audio = parameters?.GenerateAudio ?? true;
            _startTimestamp = Stopwatch.GetTimestamp();
            GpuMemoryTelemetry.EnsureRegistered();

            // A default parent context means "whatever is current", normally the HTTP server span.
            if (!TensorSharpTelemetry.Media.HasListeners())
                return;
            _activity = TensorSharpTelemetry.Media.StartActivity($"generate_video {model}", ActivityKind.Internal, parent);
            if (_activity is not { IsAllDataRequested: true })
                return;
            _activity.SetTag(TelemetryAttributes.MediaKind, MediaKindVideo);
            _activity.SetTag(TelemetryAttributes.GenAiRequestModel, model);
            if (Backend != null) _activity.SetTag(TelemetryAttributes.Backend, Backend);
            if (TelemetryContext.Protocol is { Length: > 0 } protocol) _activity.SetTag(TelemetryAttributes.Protocol, protocol);
            if (TelemetryContext.RequestId is { Length: > 0 } requestId) _activity.SetTag(TelemetryAttributes.RequestId, requestId);
            _activity.SetTag(TelemetryAttributes.VideoMode, Mode);
            _activity.SetTag(TelemetryAttributes.VideoAudio, Audio);
            // What the request asked for; 0 and -1 mean "the model's default" and are left off.
            // The values the model actually used replace them when the clip is generated.
            if (parameters != null)
            {
                if (parameters.Width > 0) _activity.SetTag(TelemetryAttributes.VideoWidth, parameters.Width);
                if (parameters.Height > 0) _activity.SetTag(TelemetryAttributes.VideoHeight, parameters.Height);
                if (parameters.Frames > 0) _activity.SetTag(TelemetryAttributes.VideoFrames, parameters.Frames);
                if (parameters.Fps > 0) _activity.SetTag(TelemetryAttributes.VideoFps, parameters.Fps);
                if (parameters.Steps > 0) _activity.SetTag(TelemetryAttributes.VideoSteps, parameters.Steps);
                if (parameters.Seed >= 0) _activity.SetTag(TelemetryAttributes.VideoSeed, parameters.Seed);
            }
            if (prompt != null && TensorSharpTelemetry.CaptureMessageContent)
                _activity.SetTag(TelemetryAttributes.VideoPrompt, prompt);
        }

        /// <summary>
        /// Start the clip. <paramref name="model"/> is the name dashboards group by (the model's
        /// <see cref="IVideoGenerationModel.VideoModelFamily"/>, or a file name, never a path);
        /// <paramref name="backend"/> the <c>--backend</c> spelling. The prompt is put on the span
        /// only when content capture is on. With no explicit <paramref name="parent"/> the span
        /// is a child of whatever is current.
        /// </summary>
        public static VideoClipTelemetry Start(string model, string? backend, VideoGenerationParams parameters,
            string? prompt = null, ActivityContext parent = default) =>
            new(string.IsNullOrEmpty(model) ? "unknown" : model, backend, parameters, prompt, parent);

        /// <summary>The span, or null when nobody records it.</summary>
        public Activity? Activity => _activity;

        /// <summary>Parent for this clip's child spans; <c>default</c> when the span is not recorded.</summary>
        public ActivityContext Context => _activity?.Context ?? default;

        /// <summary>The model name the span and the metrics carry.</summary>
        public string Model { get; }

        /// <summary>The backend the metrics carry, or null.</summary>
        internal string? Backend { get; }

        /// <summary>The generation mode, normalized to a short fixed set for metrics.</summary>
        internal string Mode { get; }

        /// <summary>Whether the request asked for the soundtrack.</summary>
        internal bool Audio { get; }

        public void SetTag(string key, object? value)
        {
            if (_activity is { IsAllDataRequested: true })
                _activity.SetTag(key, value);
        }

        /// <summary>
        /// A child span of this clip, started now (an export, a LoRA swap). Null when the clip is
        /// not recorded, so a child never starts a trace of its own.
        /// </summary>
        public Activity? StartChild(string name)
        {
            if (_activity is null) return null;
            return TensorSharpTelemetry.Media.StartActivity(name, ActivityKind.Internal, _activity.Context);
        }

        /// <summary>
        /// A child span built after the fact from UTC timestamps (the wait for the video lock).
        /// An end before the start is clamped to a zero-length span.
        /// </summary>
        public void AddChild(string name, DateTime startUtc, DateTime endUtc, params KeyValuePair<string, object?>[] tags)
        {
            if (_activity is null) return;
            if (endUtc < startUtc) endUtc = startUtc;
            var child = TensorSharpTelemetry.Media.StartActivity(name, ActivityKind.Internal, _activity.Context,
                tags: tags is { Length: > 0 } ? tags : null, startTime: TensorSharpTelemetry.ToUtcOffset(startUtc));
            if (child is null) return;
            child.SetEndTime(TensorSharpTelemetry.ToUtcOffset(endUtc).UtcDateTime);
            child.Stop();
        }

        /// <summary>The client went away or the clip was stopped. Not an error.</summary>
        public void MarkCancelled()
        {
            _outcome = TelemetryAttributes.OutcomeCancelled;
            SetTag(TelemetryAttributes.Outcome, _outcome);
        }

        /// <summary>
        /// The clip was refused before it was generated. <paramref name="errorType"/> is a short
        /// stable code (<c>video_export_unavailable</c>), never a message.
        /// </summary>
        public void MarkRefused(string errorType)
        {
            _outcome = TelemetryAttributes.OutcomeRefused;
            _errorType = errorType;
            SetTag(TelemetryAttributes.Outcome, _outcome);
            SetTag(TelemetryAttributes.ErrorType, errorType);
        }

        /// <summary>
        /// The clip failed. Records the exception type, never its message: messages can quote
        /// the prompt or an uploaded file's path. A cancellation exception counts as
        /// <see cref="MarkCancelled"/>.
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

        /// <summary>Stop the span and record the clip's duration, once.</summary>
        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
            double seconds = Stopwatch.GetElapsedTime(_startTimestamp).TotalSeconds;
            TagList tags = MetricTags();
            tags.Add(TelemetryAttributes.VideoMode, Mode);
            tags.Add(TelemetryAttributes.VideoAudio, Audio);
            tags.Add(TelemetryAttributes.Outcome, _outcome);
            if (_errorType != null) tags.Add(TelemetryAttributes.ErrorType, _errorType);
            TensorSharpMetrics.VideoGenerationDuration.Record(seconds, tags);
            if (_activity is not null)
            {
                if (_activity.IsAllDataRequested && _outcome == TelemetryAttributes.OutcomeCompleted)
                    _activity.SetTag(TelemetryAttributes.Outcome, _outcome);
                _activity.Stop();
            }
        }

        /// <summary>The model and backend every video metric of this clip carries.</summary>
        internal TagList MetricTags()
        {
            var tags = new TagList { { TelemetryAttributes.GenAiRequestModel, Model } };
            if (Backend != null) tags.Add(TelemetryAttributes.Backend, Backend);
            return tags;
        }

        /// <summary>
        /// Replace the requested size, length, rate and seed with what the model produced: the
        /// pipelines pick a random seed, a keyframe's canvas or a plug-in's step count themselves.
        /// </summary>
        internal void SetEffective(GeneratedVideo video, int totalSteps)
        {
            if (_activity is not { IsAllDataRequested: true } || video == null) return;
            if (video.Frames is { Length: > 0 } frames)
            {
                _activity.SetTag(TelemetryAttributes.VideoFrames, frames.Length);
                if (frames[0] != null)
                {
                    _activity.SetTag(TelemetryAttributes.VideoWidth, frames[0].Width);
                    _activity.SetTag(TelemetryAttributes.VideoHeight, frames[0].Height);
                }
            }
            if (video.Fps > 0) _activity.SetTag(TelemetryAttributes.VideoFps, video.Fps);
            if (video.Seed >= 0) _activity.SetTag(TelemetryAttributes.VideoSeed, video.Seed);
            if (totalSteps > 0) _activity.SetTag(TelemetryAttributes.VideoSteps, totalSteps);
        }

        /// <summary>
        /// The request's mode as one of <c>t2v</c>, <c>i2v</c>, <c>fl2v</c>, <c>ref</c>,
        /// <c>keyframes</c> or <c>other</c>. The mode is free text in a request, so it is never
        /// put on a metric as given; an absent one is inferred from the inputs supplied, the
        /// way the models infer it (approximately: each model decides for itself).
        /// </summary>
        internal static string NormalizeMode(VideoGenerationParams parameters)
        {
            if (parameters == null) return "t2v";
            string requested = parameters.Mode?.Trim();
            if (!string.IsNullOrEmpty(requested))
            {
                return requested.ToLowerInvariant() switch
                {
                    "t2v" or "t2va" or "text" or "text-to-video" => "t2v",
                    "i2v" or "i2va" or "image" or "image-to-video" or "first-frame" => "i2v",
                    "fl2v" or "fl2va" or "firstlast" or "first-last" or "first-last-frame" => "fl2v",
                    "ref" or "reference" or "ref2va" => "ref",
                    "keyframes" => "keyframes",
                    _ => "other",
                };
            }
            bool end = parameters.EndImage != null || !string.IsNullOrWhiteSpace(parameters.EndImagePath)
                || parameters.ContinuationVideoPaths is { Count: > 1 };
            if (end) return "fl2v";
            bool start = parameters.Image != null || parameters.ImageBytes is { Length: > 0 }
                || !string.IsNullOrWhiteSpace(parameters.ImagePath) || parameters.ContinuationVideoPaths is { Count: > 0 };
            if (start) return "i2v";
            if (parameters.ImageConditions is { Count: > 0 }) return "keyframes";
            if (parameters.ReferenceImagePaths is { Count: > 0 } || parameters.ReferenceVideoPaths is { Count: > 0 }
                || parameters.ReferenceAudioPaths is { Count: > 0 })
                return "ref";
            return "t2v";
        }
    }

    /// <summary>
    /// Runs <see cref="IVideoGenerationModel.GenerateVideo"/> with its progress reports turned
    /// into spans, metrics and one log line per phase.
    /// </summary>
    public static class VideoGenerationTelemetry
    {
        private const string PhaseDenoise = "denoise";
        private const string PhaseDone = "done";
        private const string StageDitLoad = "dit-load";
        /// <summary>Seconds one denoising step took, on its <c>step</c> span event.</summary>
        private const string StepSeconds = "tensorsharp.video.step_seconds";

        /// <summary>
        /// Generate a clip through <paramref name="model"/>, recording it under
        /// <paramref name="clip"/> (null: metrics and log lines only, no spans).
        ///
        /// <para>
        /// <see cref="VideoGenerationParams.OnProgress"/> is wrapped for the duration of the call
        /// and restored afterwards. The caller's callback always runs first and its exceptions
        /// propagate unchanged, since they are how a host stops a clip (MiniMax-H3 never reads
        /// the cancellation token) and how Studio refuses on a full disk; the telemetry after it
        /// never throws. Heartbeats (Wan's, from a timer thread) go to the caller only.
        /// </para>
        /// <para>
        /// Each phase becomes a span under the clip, each <see cref="VideoGenerationProgress.Stage"/>
        /// a span under its phase carrying the report's <see cref="VideoGenerationProgress.Attributes"/>,
        /// and each finished denoising step a <c>step</c> event with its duration. The
        /// <paramref name="logger"/> writes one line per phase, with the phase span as the current
        /// activity when the model reports on the calling thread, so a long clip can be followed
        /// while it runs.
        /// </para>
        /// </summary>
        public static GeneratedVideo Generate(IVideoGenerationModel model, string prompt, VideoGenerationParams parameters,
            VideoClipTelemetry? clip, ILogger? logger)
        {
            ArgumentNullException.ThrowIfNull(model);
            parameters ??= new VideoGenerationParams();
            // The per-phase lines exist to follow a long clip in an exported log. With logs not
            // exported they would only add lines to the local console and file logs, which the
            // telemetry work must leave as they were.
            if (!TensorSharpTelemetry.LogsExported)
                logger = null;
            // Captured once, here, on the calling thread: on MLX the model re-enters on the MLX
            // worker and reports progress there, where the ambient activity is not this clip's.
            var tracker = new ClipTracker(clip, model, logger, Environment.CurrentManagedThreadId);
            if (model is ModelBase loaded)
                GpuMemoryTelemetry.Track(loaded);

            Action<VideoGenerationProgress> original = parameters.OnProgress;
            parameters.OnProgress = progress =>
            {
                original?.Invoke(progress);
                // Before any shared state: heartbeats arrive concurrently from a timer thread.
                if (progress == null || progress.Heartbeat)
                    return;
                tracker.OnReport(progress);
            };
            bool generated = false;
            try
            {
                GeneratedVideo video = model.GenerateVideo(prompt, parameters);
                generated = true;
                clip?.SetEffective(video, tracker.TotalSteps);
                return video;
            }
            catch (Exception exception)
            {
                tracker.Fail(exception);
                if (exception is OperationCanceledException)
                    clip?.MarkCancelled();
                else
                    clip?.MarkFailed(exception);
                throw;
            }
            finally
            {
                parameters.OnProgress = original;
                // LTX can throw from its own report before the callback sees it, so a phase or
                // stage can still be open here.
                tracker.Finish(generated);
            }
        }

        /// <summary>
        /// The phase and stage spans of one clip. Reports come from one thread at a time (the
        /// generating thread, or the MLX worker); the gate covers <see cref="Finish"/> running
        /// on the calling thread after the model returns.
        /// </summary>
        private sealed class ClipTracker
        {
            private readonly object _gate = new();
            private readonly Activity? _clipSpan;
            private readonly ActivityContext _parent;
            private readonly bool _traced;
            private readonly string _model;
            private readonly TagList _baseTags;
            private readonly ILogger? _logger;
            private readonly int _callingThread;

            private string? _phase;
            private Activity? _phaseSpan;
            private long _phaseStart;
            private string? _stage;
            private Activity? _stageSpan;
            private long _stageStart;
            private int _lastStep;
            private long _lastStepAt;
            private bool _finished;

            public ClipTracker(VideoClipTelemetry? clip, IVideoGenerationModel model, ILogger? logger, int callingThread)
            {
                _clipSpan = clip?.Activity;
                _parent = clip?.Context ?? default;
                // Only an explicit parent: a default context would fall back to Activity.Current,
                // which on the MLX worker is not this clip, and would start a trace of its own.
                _traced = _parent != default;
                string family = null;
                try { family = model.VideoModelFamily; } catch (Exception) { }
                _model = clip?.Model ?? (string.IsNullOrEmpty(family) ? "unknown" : family);
                _baseTags = clip?.MetricTags() ?? new TagList { { TelemetryAttributes.GenAiRequestModel, _model } };
                _logger = logger;
                _callingThread = callingThread;
            }

            /// <summary>Total denoising steps from the last report that gave them.</summary>
            public int TotalSteps { get; private set; }

            public void OnReport(VideoGenerationProgress p)
            {
                try
                {
                    lock (_gate)
                    {
                        if (_finished) return;
                        long now = Stopwatch.GetTimestamp();
                        if (p.TotalSteps > 0) TotalSteps = p.TotalSteps;

                        bool phaseChanged = !string.Equals(p.Phase, _phase, StringComparison.Ordinal);
                        if (phaseChanged)
                        {
                            EndStage();
                            EndPhase(now, record: true);
                            _phase = p.Phase;
                            _phaseStart = now;
                            _lastStep = p.Step;
                            _lastStepAt = now;
                            // "done" closes the last phase; nothing runs in it.
                            if (_traced && p.Phase != null && p.Phase != PhaseDone)
                                _phaseSpan = StartSpan(p.Phase, _parent, p.Phase, null, p.Stage == null ? p.Detail : null);
                            GpuMemoryTelemetry.Snapshot();
                            LogPhase(p);
                        }

                        if (!string.Equals(p.Stage, _stage, StringComparison.Ordinal))
                        {
                            EndStage();
                            _stage = p.Stage;
                            _stageStart = now;
                            if (p.Stage != null)
                            {
                                if (_phaseSpan != null)
                                    _stageSpan = StartSpan(p.Stage, _phaseSpan.Context, p.Phase, p.Stage, p.Detail);
                                if (p.Stage == StageDitLoad && TryGetBool(p.Attributes, TelemetryAttributes.VideoCached, out bool cached))
                                {
                                    TagList tags = _baseTags;
                                    tags.Add(TelemetryAttributes.VideoCached, cached);
                                    TensorSharpMetrics.VideoDitLoads.Add(1, tags);
                                }
                            }
                            if (!phaseChanged)
                                GpuMemoryTelemetry.Snapshot();
                        }

                        if (p.Attributes is { Count: > 0 } && (_stageSpan ?? _phaseSpan) is { IsAllDataRequested: true } target)
                        {
                            foreach (var attribute in p.Attributes)
                                target.SetTag(attribute.Key, attribute.Value);
                        }

                        // The first report of the phase set the baseline. A later rise in Step is
                        // a finished step, timed from the previous one or from the start of its
                        // stage, whichever is later: between LTX's passes the x2 upscale and the
                        // refinement image encode are their own stages, not part of step 9.
                        if (!phaseChanged && p.Phase == PhaseDenoise && p.Step > _lastStep)
                        {
                            double seconds = Stopwatch.GetElapsedTime(Math.Max(_lastStepAt, _stageStart), now).TotalSeconds;
                            if (TensorSharpMetrics.VideoDenoiseStepDuration.Enabled)
                            {
                                TagList tags = _baseTags;
                                if (_stage != null) tags.Add(TelemetryAttributes.VideoStage, _stage);
                                TensorSharpMetrics.VideoDenoiseStepDuration.Record(seconds, tags);
                            }
                            if ((_stageSpan ?? _phaseSpan) is { IsAllDataRequested: true } span)
                            {
                                span.AddEvent(new ActivityEvent("step", tags: new ActivityTagsCollection
                                {
                                    { TelemetryAttributes.VideoStep, p.Step },
                                    { TelemetryAttributes.VideoSteps, p.TotalSteps },
                                    { StepSeconds, Math.Round(seconds, 3) },
                                }));
                            }
                            _lastStep = p.Step;
                            _lastStepAt = now;
                        }
                    }
                }
                catch (Exception)
                {
                    // Telemetry must never fail a generation.
                }
            }

            /// <summary>Mark the phase and stage the clip stopped in.</summary>
            public void Fail(Exception exception)
            {
                try
                {
                    lock (_gate)
                    {
                        bool cancelled = exception is OperationCanceledException;
                        foreach (Activity span in new[] { _stageSpan, _phaseSpan })
                        {
                            if (span == null) continue;
                            if (cancelled)
                            {
                                if (span.IsAllDataRequested) span.SetTag(TelemetryAttributes.Outcome, TelemetryAttributes.OutcomeCancelled);
                                continue;
                            }
                            span.SetStatus(ActivityStatusCode.Error, exception.GetType().Name);
                            if (span.IsAllDataRequested)
                            {
                                span.SetTag(TelemetryAttributes.Outcome, TelemetryAttributes.OutcomeFailed);
                                span.SetTag(TelemetryAttributes.ErrorType, exception.GetType().FullName ?? exception.GetType().Name);
                            }
                        }
                    }
                }
                catch (Exception)
                {
                }
            }

            /// <summary>
            /// End whatever is still open. A phase cut short by an exception keeps its span but
            /// is left out of the phase-duration metric, which describes phases that finished.
            /// </summary>
            public void Finish(bool generated)
            {
                try
                {
                    lock (_gate)
                    {
                        if (_finished) return;
                        _finished = true;
                        EndStage();
                        EndPhase(Stopwatch.GetTimestamp(), record: generated);
                    }
                }
                catch (Exception)
                {
                }
            }

            private void EndStage()
            {
                if (_stageSpan != null)
                {
                    StopSpan(_stageSpan);
                    _stageSpan = null;
                }
                _stage = null;
            }

            private void EndPhase(long now, bool record)
            {
                if (_phaseSpan != null)
                {
                    StopSpan(_phaseSpan);
                    _phaseSpan = null;
                }
                if (record && _phase != null && _phase != PhaseDone && TensorSharpMetrics.VideoPhaseDuration.Enabled)
                {
                    TagList tags = _baseTags;
                    tags.Add(TelemetryAttributes.VideoPhase, _phase);
                    TensorSharpMetrics.VideoPhaseDuration.Record(Stopwatch.GetElapsedTime(_phaseStart, now).TotalSeconds, tags);
                }
            }

            private void LogPhase(VideoGenerationProgress p)
            {
                if (_logger == null || !_logger.IsEnabled(LogLevel.Information))
                    return;
                // The record takes its trace from Activity.Current. That is only safe to point at
                // the phase span on the thread that called Generate, which is running this clip
                // and nothing else; the MLX worker's ambient context outlives the work item.
                Activity? span = _phaseSpan ?? _clipSpan;
                bool swap = span != null && Environment.CurrentManagedThreadId == _callingThread;
                Activity? saved = swap ? Activity.Current : null;
                if (swap) Activity.Current = span;
                try
                {
                    _logger.LogInformation(LogEventIds.VideoGeneratePhase,
                        "video.phase {Phase} model={Model} step={Step}/{TotalSteps}", p.Phase, _model, p.Step, p.TotalSteps);
                }
                finally
                {
                    if (swap) Activity.Current = saved;
                }
            }

            /// <summary>
            /// Start a span with an explicit parent and leave <see cref="Activity.Current"/> as it
            /// was: <c>StartActivity</c> makes the new span current on the reporting thread, which
            /// on MLX is the shared worker.
            /// </summary>
            private static Activity? StartSpan(string name, ActivityContext parent, string? phase, string? stage, string? detail)
            {
                Activity? saved = Activity.Current;
                Activity? span = TensorSharpTelemetry.Media.StartActivity(name, ActivityKind.Internal, parent);
                if (span == null)
                    return null;
                Activity.Current = saved;
                if (span.IsAllDataRequested)
                {
                    if (phase != null) span.SetTag(TelemetryAttributes.VideoPhase, phase);
                    if (stage != null) span.SetTag(TelemetryAttributes.VideoStage, stage);
                    // Detail is the model's own status text ("tile 3/7", "first pass at 512x512"),
                    // never the prompt or a file name.
                    if (!string.IsNullOrEmpty(detail)) span.SetTag(TelemetryAttributes.VideoDetail, detail);
                }
                return span;
            }

            /// <summary>Stop a span without moving <see cref="Activity.Current"/>: <c>Stop</c>
            /// restores the activity that was current where the span started, which may be
            /// another thread's.</summary>
            private static void StopSpan(Activity span)
            {
                Activity? saved = Activity.Current;
                span.Stop();
                if (!ReferenceEquals(saved, span))
                    Activity.Current = saved;
            }

            private static bool TryGetBool(IReadOnlyDictionary<string, object> attributes, string key, out bool value)
            {
                if (attributes != null && attributes.TryGetValue(key, out object raw) && raw is bool b)
                {
                    value = b;
                    return true;
                }
                value = false;
                return false;
            }
        }
    }
}
