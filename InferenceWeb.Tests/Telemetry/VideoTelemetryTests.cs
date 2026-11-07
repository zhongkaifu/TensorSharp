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
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.Metrics;
using System.Linq;
using System.Threading;
using Microsoft.Extensions.Logging;
using TensorSharp.Models.QwenImage;
using TensorSharp.Models.Video;
using TensorSharp.Models.WanVideo;
using TensorSharp.Runtime.Telemetry;

namespace InferenceWeb.Tests.Telemetry;

/// <summary>
/// One clip's spans, metrics and log lines, from <see cref="VideoGenerationTelemetry.Generate"/>
/// driving a scripted <see cref="IVideoGenerationModel"/> that replays the progress reports
/// MiniMax-H3 and LTX 2.5 make. No model, no GPU: the script is the model. In the global-state
/// collection because the content-capture and log-export switches are process-wide; each test
/// still filters by its own model name and trace id.
/// </summary>
[Collection(TelemetryGlobalStateCollection.Name)]
public sealed class VideoTelemetryTests : IDisposable
{
    private const string GenerationDuration = "tensorsharp.video.generation.duration";
    private const string PhaseDuration = "tensorsharp.video.phase.duration";
    private const string StepDuration = "tensorsharp.video.denoise.step.duration";
    private const string DitLoads = "tensorsharp.video.dit.loads";
    private const string StepSeconds = "tensorsharp.video.step_seconds";
    // Stage facts a model reports; the helper copies whatever it is given onto the stage span.
    private const string EncoderOpenSeconds = "tensorsharp.video.test.open_seconds";
    private const string ReleaseReason = "tensorsharp.video.test.release_reason";

    private const int FrameWidth = 4;
    private const int FrameHeight = 2;
    private const int FrameCount = 3;
    private const int OutputFps = 24;
    private const long OutputSeed = 1234;

    private const int LtxFirstPass = 8;
    private const int LtxRefinement = 3;
    private const int LtxSteps = LtxFirstPass + LtxRefinement;

    private static readonly string[] MiniMaxPhases = { "text-encode", "denoise", "vae-decode", "audio-decode" };

    private readonly object _gate = new();
    private readonly List<Activity> _started = new();
    private readonly List<Activity> _stopped = new();
    private readonly ConcurrentQueue<(string Instrument, double Value, Dictionary<string, object?> Tags)> _measurements = new();
    private readonly ActivityListener _activities;
    private readonly MeterListener _meters;
    private readonly bool _logsExportedBefore;

    public VideoTelemetryTests()
    {
        _logsExportedBefore = TensorSharpTelemetry.LogsExported;
        TensorSharpTelemetry.ConfigureLogsExported(false);
        TensorSharpTelemetry.ConfigureCaptureMessageContent(false);

        _activities = new ActivityListener
        {
            ShouldListenTo = s => s.Name.StartsWith(TensorSharpTelemetry.NamePrefix, StringComparison.Ordinal),
            Sample = (ref ActivityCreationOptions<ActivityContext> _) => ActivitySamplingResult.AllDataAndRecorded,
            ActivityStarted = a => { lock (_gate) _started.Add(a); },
            ActivityStopped = a => { lock (_gate) _stopped.Add(a); },
        };
        ActivitySource.AddActivityListener(_activities);

        _meters = new MeterListener
        {
            InstrumentPublished = (instrument, listener) =>
            {
                if (instrument.Meter.Name == TensorSharpTelemetry.MeterName)
                    listener.EnableMeasurementEvents(instrument);
            },
        };
        _meters.SetMeasurementEventCallback<double>((i, v, tags, _) => Record(i, v, tags));
        _meters.SetMeasurementEventCallback<long>((i, v, tags, _) => Record(i, v, tags));
        _meters.SetMeasurementEventCallback<int>((i, v, tags, _) => Record(i, v, tags));
        _meters.Start();
    }

    public void Dispose()
    {
        _activities.Dispose();
        _meters.Dispose();
        TensorSharpTelemetry.ConfigureCaptureMessageContent(null);
        TensorSharpTelemetry.ConfigureLogsExported(_logsExportedBefore);
    }

    // ---------------------------------------------------------------- MiniMax-H3

    [Fact]
    public void AMiniMaxClipIsOneSpanPerPhaseAndStageWithItsStepsAndMetrics()
    {
        string model = NewModel();
        const int steps = 4;
        var script = MiniMaxScript(steps, ditCached: false);

        var run = RunClip(model, script);

        Assert.Null(run.Thrown);
        Assert.Equal(script.Select(s => s.Report), run.Seen);
        AssertCallerRanFirst(run);

        var trace = SpansOf(run.TraceId);
        var clip = ClipSpan(trace, model);
        Assert.Equal(TensorSharpTelemetry.MediaSourceName, clip.Source.Name);
        Assert.Equal(ActivityStatusCode.Unset, clip.Status);
        Assert.Equal(TelemetryAttributes.OutcomeCompleted, clip.GetTagItem(TelemetryAttributes.Outcome));
        Assert.Equal("video", clip.GetTagItem(TelemetryAttributes.MediaKind));
        Assert.Equal(model, clip.GetTagItem(TelemetryAttributes.GenAiRequestModel));
        Assert.Equal("cuda", clip.GetTagItem(TelemetryAttributes.Backend));
        Assert.Equal("t2v", clip.GetTagItem(TelemetryAttributes.VideoMode));
        Assert.Equal(true, clip.GetTagItem(TelemetryAttributes.VideoAudio));
        // What the model produced replaces what the request left to it.
        Assert.Equal(steps, clip.GetTagItem(TelemetryAttributes.VideoSteps));
        Assert.Equal(FrameCount, clip.GetTagItem(TelemetryAttributes.VideoFrames));
        Assert.Equal(FrameWidth, clip.GetTagItem(TelemetryAttributes.VideoWidth));
        Assert.Equal(FrameHeight, clip.GetTagItem(TelemetryAttributes.VideoHeight));
        Assert.Equal(OutputFps, clip.GetTagItem(TelemetryAttributes.VideoFps));
        Assert.Equal(OutputSeed, clip.GetTagItem(TelemetryAttributes.VideoSeed));

        // generate_video -> one span per phase, in order, none for "done".
        var phases = ChildrenOf(trace, clip);
        Assert.Equal(MiniMaxPhases, Names(phases));
        Assert.All(phases, p =>
        {
            Assert.Equal(ActivityKind.Internal, p.Kind);
            Assert.Equal(p.DisplayName, p.GetTagItem(TelemetryAttributes.VideoPhase));
            Assert.Equal(ActivityStatusCode.Unset, p.Status);
        });
        for (int i = 1; i < phases.Count; i++)
            Assert.True(phases[i - 1].StartTimeUtc + phases[i - 1].Duration <= phases[i].StartTimeUtc,
                $"{phases[i - 1].DisplayName} overlaps {phases[i].DisplayName}");

        // text-encode -> its stages, with the facts the model attached to each.
        var textStages = ChildrenOf(trace, phases[0]);
        Assert.Equal(new[] { "text-encoder", "prefault-wait", "dit-load" }, Names(textStages));
        Assert.All(textStages, s =>
        {
            Assert.Equal("text-encode", s.GetTagItem(TelemetryAttributes.VideoPhase));
            Assert.Equal(s.DisplayName, s.GetTagItem(TelemetryAttributes.VideoStage));
        });
        Assert.Equal(1.25, textStages[0].GetTagItem(EncoderOpenSeconds));
        Assert.Equal(false, textStages[2].GetTagItem(TelemetryAttributes.VideoCached));

        // denoise -> no stages, one step event per finished step.
        Assert.Empty(ChildrenOf(trace, phases[1]));
        var stepEvents = StepEvents(phases[1]);
        Assert.Equal(Enumerable.Range(1, steps).Cast<object>(), stepEvents.Select(e => Tag(e, TelemetryAttributes.VideoStep)));
        Assert.All(stepEvents, e =>
        {
            Assert.Equal(steps, Tag(e, TelemetryAttributes.VideoSteps));
            Assert.IsType<double>(Tag(e, StepSeconds));
        });

        // vae-decode -> the release, the open and the decode, each with its facts.
        var vaeStages = ChildrenOf(trace, phases[2]);
        Assert.Equal(new[] { "denoiser-release", "vae-open", "vae-decode" }, Names(vaeStages));
        Assert.Equal("vae-would-not-fit", vaeStages[0].GetTagItem(ReleaseReason));
        Assert.Equal(true, vaeStages[1].GetTagItem(TelemetryAttributes.VideoCached));
        Assert.Equal(2, vaeStages[2].GetTagItem(TelemetryAttributes.VideoTiles));
        Assert.Empty(ChildrenOf(trace, phases[3]));

        // Nothing else in the trace: no stray phase, stage or "done" span.
        Assert.Equal(1 + 4 + 3 + 3, trace.Count);

        var duration = Assert.Single(Measurements(model, GenerationDuration));
        Assert.Equal(TelemetryAttributes.OutcomeCompleted, duration.Tags[TelemetryAttributes.Outcome]);
        Assert.False(duration.Tags.ContainsKey(TelemetryAttributes.ErrorType));
        Assert.Equal("cuda", duration.Tags[TelemetryAttributes.Backend]);
        Assert.Equal("t2v", duration.Tags[TelemetryAttributes.VideoMode]);
        Assert.Equal(true, duration.Tags[TelemetryAttributes.VideoAudio]);

        Assert.Equal(MiniMaxPhases, Measurements(model, PhaseDuration).Select(m => (string?)m.Tags[TelemetryAttributes.VideoPhase]));
        var stepDurations = Measurements(model, StepDuration);
        Assert.Equal(steps, stepDurations.Count);
        Assert.All(stepDurations, m => Assert.False(m.Tags.ContainsKey(TelemetryAttributes.VideoStage)));

        var load = Assert.Single(Measurements(model, DitLoads));
        Assert.Equal(1, load.Value);
        Assert.Equal(false, load.Tags[TelemetryAttributes.VideoCached]);
        Assert.Equal("cuda", load.Tags[TelemetryAttributes.Backend]);
    }

    [Fact]
    public void DenoiserLoadsAreCountedByWhetherTheDenoiserWasReused()
    {
        string model = NewModel();

        var cold = RunClip(model, MiniMaxScript(2, ditCached: false));
        var warm = RunClip(model, MiniMaxScript(2, ditCached: true));

        Assert.Null(cold.Thrown);
        Assert.Null(warm.Thrown);
        var loads = Measurements(model, DitLoads);
        Assert.Equal(new object?[] { false, true }, loads.Select(m => m.Tags[TelemetryAttributes.VideoCached]));
        Assert.All(loads, m => Assert.Equal(1, m.Value));
        Assert.Equal(false, DitLoadSpan(SpansOf(cold.TraceId)).GetTagItem(TelemetryAttributes.VideoCached));
        Assert.Equal(true, DitLoadSpan(SpansOf(warm.TraceId)).GetTagItem(TelemetryAttributes.VideoCached));
        // The VAE's cached flag is an attribute of its own stage, not a denoiser load.
        Assert.Equal(2, Measurements(model, GenerationDuration).Count);
    }

    // ---------------------------------------------------------------- LTX 2.5

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void AnLtxTwoPassClipKeepsItsPassesAndUpscaleAsStagesOfDenoise(bool heartbeats)
    {
        string model = NewModel();
        var script = LtxScript(heartbeats);

        var run = RunClip(model, script);

        Assert.Null(run.Thrown);
        // The caller saw every report, heartbeats included (some from another thread), in order.
        Assert.Equal(script.Select(s => s.Report), run.Seen);
        Assert.Equal(heartbeats ? 4 : 0, run.Seen.Count(p => p.Heartbeat));
        AssertCallerRanFirst(run);

        // Heartbeats carry a phase, a stage and a step that would each have changed this tree.
        var trace = SpansOf(run.TraceId);
        var clip = ClipSpan(trace, model);
        Assert.Equal(LtxSteps, clip.GetTagItem(TelemetryAttributes.VideoSteps));
        var phases = ChildrenOf(trace, clip);
        Assert.Equal(MiniMaxPhases, Names(phases));
        Assert.Equal("Gemma 4 and learned AV conditioning", phases[0].GetTagItem(TelemetryAttributes.VideoDetail));
        Assert.Empty(ChildrenOf(trace, phases[0]));

        // The first denoise report named a stage, so its detail is the stage's, not the phase's.
        var denoise = phases[1];
        Assert.Null(denoise.GetTagItem(TelemetryAttributes.VideoDetail));
        Assert.Empty(StepEvents(denoise));
        var passes = ChildrenOf(trace, denoise);
        Assert.Equal(new[] { "first-pass", "upscale", "refinement" }, Names(passes));
        Assert.All(passes, s => Assert.Equal("denoise", s.GetTagItem(TelemetryAttributes.VideoPhase)));
        Assert.Equal("first pass at 512x288", passes[0].GetTagItem(TelemetryAttributes.VideoDetail));
        Assert.Equal("x2 latent upscale to 1024x576", passes[1].GetTagItem(TelemetryAttributes.VideoDetail));
        Assert.Equal(Enumerable.Range(1, LtxFirstPass).Cast<object>(),
            StepEvents(passes[0]).Select(e => Tag(e, TelemetryAttributes.VideoStep)));
        Assert.Empty(StepEvents(passes[1]));
        Assert.Equal(Enumerable.Range(LtxFirstPass + 1, LtxRefinement).Cast<object>(),
            StepEvents(passes[2]).Select(e => Tag(e, TelemetryAttributes.VideoStep)));
        Assert.Equal(1 + 4 + 3, trace.Count);

        var stepDurations = Measurements(model, StepDuration);
        Assert.Equal(LtxSteps, stepDurations.Count);
        Assert.Equal(LtxFirstPass, stepDurations.Count(m => Equals(m.Tags.GetValueOrDefault(TelemetryAttributes.VideoStage), "first-pass")));
        Assert.Equal(LtxRefinement, stepDurations.Count(m => Equals(m.Tags.GetValueOrDefault(TelemetryAttributes.VideoStage), "refinement")));
        Assert.Equal(MiniMaxPhases, Measurements(model, PhaseDuration).Select(m => (string?)m.Tags[TelemetryAttributes.VideoPhase]));
        Assert.Empty(Measurements(model, DitLoads));
        Assert.Equal(TelemetryAttributes.OutcomeCompleted,
            Assert.Single(Measurements(model, GenerationDuration)).Tags[TelemetryAttributes.Outcome]);
    }

    [Fact]
    public void AStepAfterAnotherStageIsTimedFromThatStageNotFromThePreviousStep()
    {
        string model = NewModel();
        const int upscaleMs = 250;

        var run = RunClip(model, LtxScript(heartbeats: false, upscaleSleepMs: upscaleMs));

        Assert.Null(run.Thrown);
        var trace = SpansOf(run.TraceId);
        var passes = ChildrenOf(trace, ChildrenOf(trace, ClipSpan(trace, model))[1]);
        Assert.Equal("upscale", passes[1].DisplayName);
        Assert.True(passes[1].Duration >= TimeSpan.FromMilliseconds(upscaleMs - 20), $"upscale took {passes[1].Duration}");

        // Step 9 follows the upscale, but the upscale is not part of it.
        var stepDurations = Measurements(model, StepDuration);
        var firstRefinement = stepDurations[LtxFirstPass];
        Assert.Equal("refinement", firstRefinement.Tags[TelemetryAttributes.VideoStage]);
        Assert.InRange(firstRefinement.Value, 0, 0.15);
        var firstRefinementEvent = StepEvents(passes[2])[0];
        Assert.InRange((double)Tag(firstRefinementEvent, StepSeconds)!, 0, 0.15);
        // The phase as a whole does include it.
        var denoisePhase = Measurements(model, PhaseDuration).Single(m => Equals(m.Tags[TelemetryAttributes.VideoPhase], "denoise"));
        Assert.True(denoisePhase.Value >= (upscaleMs - 20) / 1000.0, $"denoise phase {denoisePhase.Value}s");
    }

    // ---------------------------------------------------------------- failures

    [Fact]
    public void ACallbackExceptionPropagatesUnchangedAndFailsTheClip()
    {
        string model = NewModel();
        var refusal = new InvalidOperationException(@"No room left on D:\secret-takes for the clip");

        var run = RunClip(model, MiniMaxScript(4, ditCached: false),
            alsoOnProgress: p => { if (p.Phase == "denoise" && p.Step == 2) throw refusal; });

        Assert.Same(refusal, run.Thrown);
        AssertCallerRanFirst(run);
        var trace = SpansOf(run.TraceId);
        var clip = ClipSpan(trace, model);
        Assert.Equal(ActivityStatusCode.Error, clip.Status);
        Assert.Equal(nameof(InvalidOperationException), clip.StatusDescription);
        Assert.Equal(TelemetryAttributes.OutcomeFailed, clip.GetTagItem(TelemetryAttributes.Outcome));
        Assert.Equal(typeof(InvalidOperationException).FullName, clip.GetTagItem(TelemetryAttributes.ErrorType));
        // The type, never the message.
        Assert.DoesNotContain(trace.SelectMany(a => a.TagObjects), t => t.Value is string s && s.Contains("secret", StringComparison.Ordinal));

        var phases = ChildrenOf(trace, clip);
        Assert.Equal(new[] { "text-encode", "denoise" }, Names(phases));
        Assert.Equal(ActivityStatusCode.Unset, phases[0].Status);
        Assert.Equal(ActivityStatusCode.Error, phases[1].Status);
        Assert.Equal(TelemetryAttributes.OutcomeFailed, phases[1].GetTagItem(TelemetryAttributes.Outcome));
        Assert.Equal(typeof(InvalidOperationException).FullName, phases[1].GetTagItem(TelemetryAttributes.ErrorType));
        // The report the caller threw on never reached the telemetry.
        Assert.Equal(new object[] { 1 }, StepEvents(phases[1]).Select(e => Tag(e, TelemetryAttributes.VideoStep)));

        // Only the phase that finished has a duration; the clip has exactly one, failed.
        Assert.Equal(new[] { "text-encode" }, Measurements(model, PhaseDuration).Select(m => (string?)m.Tags[TelemetryAttributes.VideoPhase]));
        var duration = Assert.Single(Measurements(model, GenerationDuration));
        Assert.Equal(TelemetryAttributes.OutcomeFailed, duration.Tags[TelemetryAttributes.Outcome]);
        Assert.Equal(typeof(InvalidOperationException).FullName, duration.Tags[TelemetryAttributes.ErrorType]);
    }

    [Fact]
    public void ACancellationThrownByTheCallerEndsTheClipAsCancelledNotFailed()
    {
        string model = NewModel();
        var stop = new OperationCanceledException("The client went away.");

        var run = RunClip(model, MiniMaxScript(4, ditCached: false),
            alsoOnProgress: p => { if (p.Phase == "denoise" && p.Step == 2) throw stop; });

        Assert.Same(stop, run.Thrown);
        var trace = SpansOf(run.TraceId);
        var clip = ClipSpan(trace, model);
        Assert.NotEqual(ActivityStatusCode.Error, clip.Status);
        Assert.Equal(TelemetryAttributes.OutcomeCancelled, clip.GetTagItem(TelemetryAttributes.Outcome));
        Assert.Null(clip.GetTagItem(TelemetryAttributes.ErrorType));

        var denoise = ChildrenOf(trace, clip)[1];
        Assert.Equal("denoise", denoise.DisplayName);
        Assert.NotEqual(ActivityStatusCode.Error, denoise.Status);
        Assert.Equal(TelemetryAttributes.OutcomeCancelled, denoise.GetTagItem(TelemetryAttributes.Outcome));

        Assert.Equal(new[] { "text-encode" }, Measurements(model, PhaseDuration).Select(m => (string?)m.Tags[TelemetryAttributes.VideoPhase]));
        var duration = Assert.Single(Measurements(model, GenerationDuration));
        Assert.Equal(TelemetryAttributes.OutcomeCancelled, duration.Tags[TelemetryAttributes.Outcome]);
        Assert.False(duration.Tags.ContainsKey(TelemetryAttributes.ErrorType));
    }

    [Fact]
    public void AModelFailureMarksTheStageAndPhaseItStoppedIn()
    {
        string model = NewModel();
        var script = MiniMaxScript(2, ditCached: true);
        // Fail right after the decode stage opened, the way LTX can throw from its own report.
        int decodeStage = script.FindIndex(s => s.Report.Phase == "vae-decode" && s.Report.Stage == "vae-decode");
        var failure = new InvalidOperationException("VAE output was not finite");

        var run = RunClip(model, script, modelFailsAt: decodeStage + 1, modelFailure: failure);

        Assert.Same(failure, run.Thrown);
        Assert.Equal(decodeStage + 1, run.Seen.Count);
        var trace = SpansOf(run.TraceId);
        var clip = ClipSpan(trace, model);
        Assert.Equal(TelemetryAttributes.OutcomeFailed, clip.GetTagItem(TelemetryAttributes.Outcome));
        var phases = ChildrenOf(trace, clip);
        Assert.Equal(new[] { "text-encode", "denoise", "vae-decode" }, Names(phases));
        Assert.Equal(ActivityStatusCode.Error, phases[2].Status);
        var stages = ChildrenOf(trace, phases[2]);
        Assert.Equal(new[] { "denoiser-release", "vae-open", "vae-decode" }, Names(stages));
        Assert.Equal(ActivityStatusCode.Unset, stages[0].Status);
        Assert.Equal(ActivityStatusCode.Unset, stages[1].Status);
        Assert.Equal(ActivityStatusCode.Error, stages[2].Status);
        Assert.Equal(TelemetryAttributes.OutcomeFailed, stages[2].GetTagItem(TelemetryAttributes.Outcome));
        Assert.Equal(new[] { "text-encode", "denoise" }, Measurements(model, PhaseDuration).Select(m => (string?)m.Tags[TelemetryAttributes.VideoPhase]));
    }

    // ---------------------------------------------------------------- callback, clip, content, logs

    [Fact]
    public void WithoutAClipThereAreMetricsButNoSpansAndANullCallbackStaysNull()
    {
        string family = NewModel();
        var parameters = new VideoGenerationParams();
        var video = new ScriptedVideoModel(family, MiniMaxScript(2, ditCached: true));
        using var ambient = new Activity("request without a clip").Start();
        int startedBefore;
        lock (_gate) startedBefore = _started.Count;

        var result = VideoGenerationTelemetry.Generate(video, "a prompt", parameters, clip: null, logger: null);

        Assert.Equal(FrameCount, result.Frames.Length);
        Assert.Null(parameters.OnProgress);
        Assert.Same(ambient, Activity.Current);
        var spanNames = new HashSet<string>(MiniMaxPhases) { "text-encoder", "prefault-wait", "dit-load", "denoiser-release", "vae-open" };
        lock (_gate)
            Assert.DoesNotContain(_started.Skip(startedBefore), a => spanNames.Contains(a.DisplayName));

        // The metrics carry the model's family and no backend.
        Assert.Equal(MiniMaxPhases, Measurements(family, PhaseDuration).Select(m => (string?)m.Tags[TelemetryAttributes.VideoPhase]));
        Assert.Equal(2, Measurements(family, StepDuration).Count);
        var load = Assert.Single(Measurements(family, DitLoads));
        Assert.Equal(true, load.Tags[TelemetryAttributes.VideoCached]);
        Assert.False(load.Tags.ContainsKey(TelemetryAttributes.Backend));
        // The clip's own duration belongs to the clip.
        Assert.Empty(Measurements(family, GenerationDuration));
    }

    [Fact]
    public void ThePromptIsOnTheClipSpanOnlyWithContentCaptureOn()
    {
        const string prompt = "a red fox crossing a frozen river at dawn";

        TensorSharpTelemetry.ConfigureCaptureMessageContent(false);
        string off = NewModel();
        var offRun = RunClip(off, MiniMaxScript(2, ditCached: false), prompt: prompt);
        Assert.Null(offRun.Thrown);
        Assert.Equal(prompt, offRun.Model.Prompt);
        var offTrace = SpansOf(offRun.TraceId);
        Assert.Null(ClipSpan(offTrace, off).GetTagItem(TelemetryAttributes.VideoPrompt));
        Assert.DoesNotContain(offTrace.SelectMany(a => a.TagObjects), t => t.Value is string s && s.Contains("fox", StringComparison.Ordinal));

        TensorSharpTelemetry.ConfigureCaptureMessageContent(true);
        string on = NewModel();
        var onRun = RunClip(on, MiniMaxScript(2, ditCached: false), prompt: prompt);
        Assert.Null(onRun.Thrown);
        var onTrace = SpansOf(onRun.TraceId);
        var clip = ClipSpan(onTrace, on);
        Assert.Equal(prompt, clip.GetTagItem(TelemetryAttributes.VideoPrompt));
        Assert.DoesNotContain(onTrace.Where(a => !ReferenceEquals(a, clip)).SelectMany(a => a.TagObjects),
            t => t.Value is string s && s.Contains("fox", StringComparison.Ordinal));

        // Never on a metric, whatever the switch says.
        Assert.DoesNotContain(_measurements, m => m.Tags.Values.Any(v => v is string s && s.Contains("fox", StringComparison.Ordinal)));
    }

    [Fact]
    public void PhaseLinesAreWrittenOnlyWhenLogsAreExported()
    {
        TensorSharpTelemetry.ConfigureLogsExported(false);
        var local = new CapturingLogger();
        var quiet = RunClip(NewModel(), MiniMaxScript(2, ditCached: false), logger: local);
        Assert.Null(quiet.Thrown);
        Assert.Empty(local.Lines);

        TensorSharpTelemetry.ConfigureLogsExported(true);
        var exported = new CapturingLogger();
        string model = NewModel();
        var run = RunClip(model, MiniMaxScript(2, ditCached: false), logger: exported);
        Assert.Null(run.Thrown);

        var lines = exported.Lines;
        Assert.Equal(new[] { "text-encode", "denoise", "vae-decode", "audio-decode", "done" },
            lines.Select(l => l.Message.Split(' ')[1]));
        Assert.All(lines, l =>
        {
            Assert.Equal(LogEventIds.VideoGeneratePhase.Id, l.EventId.Id);
            Assert.Equal(LogLevel.Information, l.Level);
            Assert.Contains("model=" + model, l.Message, StringComparison.Ordinal);
        });
        // Each line is written with its phase span current ("done" has none: the clip's), so an
        // exported record lands in the clip's trace.
        var trace = SpansOf(run.TraceId);
        var clip = ClipSpan(trace, model);
        var expectedSpans = ChildrenOf(trace, clip).Select(p => p.SpanId).Append(clip.SpanId);
        Assert.Equal(expectedSpans, lines.Select(l => l.CurrentSpanId));
    }

    // ---------------------------------------------------------------- the clip itself

    [Fact]
    public void ARefusedClipRecordsItsCodeAndItsDurationOnce()
    {
        string model = NewModel();
        var clip = VideoClipTelemetry.Start(model, null, new VideoGenerationParams
        {
            Mode = "I2V", GenerateAudio = false, Width = 640, Height = 384, Frames = 49, Fps = 24, Steps = 8, Seed = 7,
        });
        Assert.NotNull(clip.Activity);
        var traceId = clip.Activity!.TraceId;

        clip.MarkRefused("video_export_unavailable");
        clip.Dispose();
        clip.Dispose();

        var span = ClipSpan(SpansOf(traceId), model);
        Assert.Equal(TelemetryAttributes.OutcomeRefused, span.GetTagItem(TelemetryAttributes.Outcome));
        Assert.Equal("video_export_unavailable", span.GetTagItem(TelemetryAttributes.ErrorType));
        Assert.Equal("i2v", span.GetTagItem(TelemetryAttributes.VideoMode));
        Assert.Equal(false, span.GetTagItem(TelemetryAttributes.VideoAudio));
        Assert.Equal(640, span.GetTagItem(TelemetryAttributes.VideoWidth));
        Assert.Equal(384, span.GetTagItem(TelemetryAttributes.VideoHeight));
        Assert.Equal(49, span.GetTagItem(TelemetryAttributes.VideoFrames));
        Assert.Equal(24, span.GetTagItem(TelemetryAttributes.VideoFps));
        Assert.Equal(8, span.GetTagItem(TelemetryAttributes.VideoSteps));
        Assert.Equal(7L, span.GetTagItem(TelemetryAttributes.VideoSeed));
        Assert.Null(span.GetTagItem(TelemetryAttributes.Backend));

        var duration = Assert.Single(Measurements(model, GenerationDuration));
        Assert.Equal(TelemetryAttributes.OutcomeRefused, duration.Tags[TelemetryAttributes.Outcome]);
        Assert.Equal("video_export_unavailable", duration.Tags[TelemetryAttributes.ErrorType]);
        Assert.Equal("i2v", duration.Tags[TelemetryAttributes.VideoMode]);
        Assert.Equal(false, duration.Tags[TelemetryAttributes.VideoAudio]);
        Assert.False(duration.Tags.ContainsKey(TelemetryAttributes.Backend));
    }

    [Fact]
    public void ARequestLeavingSettingsToTheModelPutsNoPlaceholdersOnTheSpan()
    {
        string model = NewModel();
        var clip = VideoClipTelemetry.Start(model, "", new VideoGenerationParams());
        var traceId = clip.Activity!.TraceId;
        clip.Dispose();

        var span = ClipSpan(SpansOf(traceId), model);
        foreach (string key in new[]
        {
            TelemetryAttributes.VideoWidth, TelemetryAttributes.VideoHeight, TelemetryAttributes.VideoFrames,
            TelemetryAttributes.VideoFps, TelemetryAttributes.VideoSteps, TelemetryAttributes.VideoSeed,
            TelemetryAttributes.Backend, TelemetryAttributes.VideoPrompt,
        })
            Assert.Null(span.GetTagItem(key));
        Assert.Equal(TelemetryAttributes.OutcomeCompleted, span.GetTagItem(TelemetryAttributes.Outcome));
    }

    [Fact]
    public void ChildrenOfAClipNestUnderIt()
    {
        string model = NewModel();
        var clip = VideoClipTelemetry.Start(model, null, new VideoGenerationParams());
        var traceId = clip.Activity!.TraceId;
        DateTime queued = DateTime.UtcNow.AddSeconds(-3);

        clip.AddChild("video.queue", queued, queued.AddSeconds(1.5), new KeyValuePair<string, object?>("tensorsharp.test.position", 2));
        using (var export = clip.StartChild("video.export"))
            Assert.NotNull(export);
        Assert.Same(clip.Activity, Activity.Current);
        clip.Dispose();

        var trace = SpansOf(traceId);
        var span = ClipSpan(trace, model);
        var children = ChildrenOf(trace, span);
        Assert.Equal(new[] { "video.queue", "video.export" }, Names(children));
        Assert.Equal(queued, children[0].StartTimeUtc);
        Assert.Equal(TimeSpan.FromSeconds(1.5), children[0].Duration);
        Assert.Equal(2, children[0].GetTagItem("tensorsharp.test.position"));
    }

    [Theory]
    [InlineData("T2V", "t2v")]
    [InlineData(" image-to-video ", "i2v")]
    [InlineData("first-last-frame", "fl2v")]
    [InlineData("Ref2VA", "ref")]
    [InlineData("keyframes", "keyframes")]
    [InlineData("a mode nobody has heard of", "other")]
    public void TheModeOnMetricsIsOneOfAFixedSet(string requested, string expected) =>
        Assert.Equal(expected, VideoClipTelemetry.NormalizeMode(new VideoGenerationParams { Mode = requested }));

    [Fact]
    public void AnAbsentModeIsInferredFromTheInputs()
    {
        Assert.Equal("t2v", VideoClipTelemetry.NormalizeMode(null!));
        Assert.Equal("t2v", VideoClipTelemetry.NormalizeMode(new VideoGenerationParams()));
        Assert.Equal("i2v", VideoClipTelemetry.NormalizeMode(new VideoGenerationParams { ImagePath = "first.png" }));
        Assert.Equal("fl2v", VideoClipTelemetry.NormalizeMode(new VideoGenerationParams { ImagePath = "first.png", EndImagePath = "last.png" }));
        Assert.Equal("i2v", VideoClipTelemetry.NormalizeMode(new VideoGenerationParams { ContinuationVideoPaths = new[] { "a.mp4" } }));
        Assert.Equal("fl2v", VideoClipTelemetry.NormalizeMode(new VideoGenerationParams { ContinuationVideoPaths = new[] { "a.mp4", "b.mp4" } }));
        Assert.Equal("keyframes", VideoClipTelemetry.NormalizeMode(new VideoGenerationParams
        {
            ImageConditions = new List<VideoImageCondition> { new() { ImagePath = "key.png", FrameIndex = 8 } },
        }));
        Assert.Equal("ref", VideoClipTelemetry.NormalizeMode(new VideoGenerationParams { ReferenceImagePaths = new[] { "face.png" } }));
    }

    // ---------------------------------------------------------------- scripts

    private static VideoGenerationProgress P(string? phase, int step, int total, string? stage = null, string? detail = null,
        IReadOnlyDictionary<string, object>? attributes = null, bool heartbeat = false) =>
        new()
        {
            Phase = phase, Step = step, TotalSteps = total, Stage = stage, Detail = detail,
            Attributes = attributes, Heartbeat = heartbeat,
        };

    /// <summary>The reports MiniMaxH3Pipeline makes, stage for stage.</summary>
    private static List<ScriptItem> MiniMaxScript(int steps, bool ditCached)
    {
        var reports = new List<VideoGenerationProgress>
        {
            P("text-encode", 0, steps),
            P("text-encode", 0, steps, "text-encoder"),
            P("text-encode", 0, steps, "text-encoder", attributes: new Dictionary<string, object> { [EncoderOpenSeconds] = 1.25 }),
            P("text-encode", 0, steps, "prefault-wait"),
            P("text-encode", 0, steps, "dit-load", attributes: new Dictionary<string, object> { [TelemetryAttributes.VideoCached] = ditCached }),
            P("denoise", 0, steps),
        };
        for (int s = 1; s <= steps; s++)
            reports.Add(P("denoise", s, steps));
        reports.Add(P("vae-decode", steps, steps));
        reports.Add(P("vae-decode", steps, steps, "denoiser-release",
            attributes: new Dictionary<string, object> { [ReleaseReason] = "vae-would-not-fit" }));
        reports.Add(P("vae-decode", steps, steps, "vae-open",
            attributes: new Dictionary<string, object> { [TelemetryAttributes.VideoCached] = true }));
        reports.Add(P("vae-decode", steps, steps, "vae-decode"));
        reports.Add(P("vae-decode", steps, steps, "vae-decode", "tile 1/2", new Dictionary<string, object> { [TelemetryAttributes.VideoTiles] = 2 }));
        reports.Add(P("vae-decode", steps, steps, "vae-decode", "tile 2/2", new Dictionary<string, object> { [TelemetryAttributes.VideoTiles] = 2 }));
        reports.Add(P("audio-decode", steps, steps));
        reports.Add(P("done", steps, steps));
        return reports.Select(r => new ScriptItem(r)).ToList();
    }

    /// <summary>
    /// The reports Ltx25Pipeline makes for a two-stage clip. With <paramref name="heartbeats"/>,
    /// periodic ticks are interleaved, two of them from another thread; each names a phase, a
    /// stage or a step that would change the trace if it were taken for a report.
    /// </summary>
    private static List<ScriptItem> LtxScript(bool heartbeats, int upscaleSleepMs = 0)
    {
        var items = new List<ScriptItem>();
        void Add(VideoGenerationProgress p, bool otherThread = false, int sleepMs = 0) => items.Add(new ScriptItem(p, otherThread, sleepMs));

        if (heartbeats) Add(P("text-encode", 0, LtxSteps, "bogus-stage", heartbeat: true));
        Add(P("text-encode", 0, LtxSteps, detail: "Gemma 4 and learned AV conditioning"));
        Add(P("denoise", 0, LtxSteps, "first-pass", "first pass at 512x288"));
        for (int s = 1; s <= LtxFirstPass; s++)
        {
            Add(P("denoise", s, LtxSteps, "first-pass", $"step {s}/{LtxSteps}"));
            if (heartbeats && s == 3) Add(P("denoise", 6, LtxSteps, "first-pass", heartbeat: true));
        }
        Add(P("denoise", LtxFirstPass, LtxSteps, "upscale", "x2 latent upscale to 1024x576"), sleepMs: upscaleSleepMs);
        if (heartbeats) Add(P("vae-decode", LtxFirstPass, LtxSteps, heartbeat: true), otherThread: true);
        Add(P("denoise", LtxFirstPass, LtxSteps, "refinement", "refinement pass at 1024x576"));
        for (int s = LtxFirstPass + 1; s <= LtxSteps; s++)
        {
            Add(P("denoise", s, LtxSteps, "refinement", $"step {s}/{LtxSteps}"));
            if (heartbeats && s == LtxFirstPass + 2) Add(P(null, 0, 0, detail: "still working", heartbeat: true), otherThread: true);
        }
        Add(P("vae-decode", LtxSteps, LtxSteps, detail: "convolutional video decoder"));
        Add(P("audio-decode", LtxSteps, LtxSteps, detail: "audio VAE, vocoder and 48 kHz bandwidth extension"));
        Add(P("done", LtxSteps, LtxSteps, detail: "1024x576, 121 frames at 24 fps"));
        return items;
    }

    // ---------------------------------------------------------------- running a clip

    private sealed record ScriptItem(VideoGenerationProgress Report, bool FromOtherThread = false, int SleepAfterMs = 0);

    private sealed record ClipRun(
        ActivityTraceId TraceId, List<VideoGenerationProgress> Seen, List<int> StartedAtCall, Exception? Thrown,
        ScriptedVideoModel Model);

    /// <summary>
    /// Start a clip, generate it through <see cref="VideoGenerationTelemetry.Generate"/> with a
    /// caller callback that records each report and how many of the clip's spans existed when
    /// it ran, and dispose the clip, as a host does.
    /// </summary>
    private ClipRun RunClip(string model, IReadOnlyList<ScriptItem> script, string? prompt = null, ILogger? logger = null,
        Action<VideoGenerationProgress>? alsoOnProgress = null, int modelFailsAt = -1, Exception? modelFailure = null)
    {
        var parameters = new VideoGenerationParams();
        var video = new ScriptedVideoModel("scripted-family", script, modelFailsAt, modelFailure);
        var clip = VideoClipTelemetry.Start(model, "cuda", parameters, prompt);
        Assert.NotNull(clip.Activity);
        Activity clipSpan = clip.Activity!;
        ActivityTraceId traceId = clipSpan.TraceId;

        var seen = new List<VideoGenerationProgress>();
        var startedAtCall = new List<int>();
        Action<VideoGenerationProgress> callback = p =>
        {
            int started = StartedIn(traceId);
            lock (seen)
            {
                seen.Add(p);
                startedAtCall.Add(started);
            }
            alsoOnProgress?.Invoke(p);
        };
        parameters.OnProgress = callback;

        Exception? thrown = null;
        Activity? currentAfter = null;
        try
        {
            VideoGenerationTelemetry.Generate(video, prompt ?? "a lighthouse at dusk", parameters, clip, logger);
        }
        catch (Exception exception)
        {
            thrown = exception;
        }
        finally
        {
            currentAfter = Activity.Current;
            clip.Dispose();
        }

        // The caller's callback is back, and the phase spans never became the ambient activity.
        Assert.Same(callback, parameters.OnProgress);
        Assert.Same(clipSpan, currentAfter);
        return new ClipRun(traceId, seen, startedAtCall, thrown, video);
    }

    /// <summary>
    /// The caller's callback ran before the telemetry saw each report: when it ran, exactly the
    /// spans opened by the reports before it existed (plus the clip's own).
    /// </summary>
    private static void AssertCallerRanFirst(ClipRun run)
    {
        string? phase = null, stage = null;
        int expected = 1;
        for (int i = 0; i < run.Seen.Count; i++)
        {
            var report = run.Seen[i];
            if (report.Heartbeat)
                continue;
            Assert.True(expected == run.StartedAtCall[i],
                $"report {i} ({report.Phase}/{report.Stage}): {run.StartedAtCall[i]} spans existed when the caller saw it, expected {expected}");
            expected += SpansOpenedBy(report, ref phase, ref stage);
        }
    }

    /// <summary>How many spans the helper opens for a report: its phase's and its stage's, when new.</summary>
    private static int SpansOpenedBy(VideoGenerationProgress p, ref string? phase, ref string? stage)
    {
        int opened = 0;
        if (p.Phase != phase)
        {
            phase = p.Phase;
            stage = null;
            if (phase is not null and not "done") opened++;
        }
        if (p.Stage != stage)
        {
            stage = p.Stage;
            if (stage != null && phase is not null and not "done") opened++;
        }
        return opened;
    }

    /// <summary>
    /// A model that replays a script of progress reports through <see cref="VideoGenerationParams.OnProgress"/>
    /// and returns a tiny clip, or throws at <c>failAt</c> (before that report).
    /// </summary>
    private sealed class ScriptedVideoModel : IVideoGenerationModel
    {
        private readonly IReadOnlyList<ScriptItem> _script;
        private readonly int _failAt;
        private readonly Exception? _failure;

        public ScriptedVideoModel(string family, IReadOnlyList<ScriptItem> script, int failAt = -1, Exception? failure = null)
        {
            VideoModelFamily = family;
            _script = script;
            _failAt = failure == null ? -1 : failAt;
            _failure = failure;
        }

        public string VideoModelFamily { get; }
        public bool SupportsAudio => true;
        public bool SupportsImageConditioning => false;
        public bool SupportsEndImageConditioning => false;
        public bool SupportsReferenceConditioning => false;

        /// <summary>The prompt the last generation was given.</summary>
        public string? Prompt { get; private set; }

        public GeneratedVideo GenerateVideo(string prompt, VideoGenerationParams? p = null)
        {
            Prompt = prompt;
            for (int i = 0; i < _script.Count; i++)
            {
                if (i == _failAt)
                    throw _failure!;
                ScriptItem item = _script[i];
                if (item.FromOtherThread)
                {
                    // Wan's heartbeats come from a timer thread.
                    var thread = new Thread(() => p?.OnProgress?.Invoke(item.Report)) { IsBackground = true };
                    thread.Start();
                    thread.Join();
                }
                else
                {
                    p?.OnProgress?.Invoke(item.Report);
                }
                if (item.SleepAfterMs > 0)
                    Thread.Sleep(item.SleepAfterMs);
            }
            if (_failAt >= _script.Count)
                throw _failure!;

            var frames = new RgbImage[FrameCount];
            for (int f = 0; f < FrameCount; f++)
                frames[f] = new RgbImage(FrameWidth, FrameHeight, new float[FrameWidth * FrameHeight * 3]);
            return new GeneratedVideo { Frames = frames, Fps = OutputFps, Seed = OutputSeed };
        }
    }

    private sealed class CapturingLogger : ILogger
    {
        private readonly List<(LogLevel Level, EventId EventId, string Message, ActivitySpanId CurrentSpanId)> _lines = new();

        public List<(LogLevel Level, EventId EventId, string Message, ActivitySpanId CurrentSpanId)> Lines
        {
            get { lock (_lines) return _lines.ToList(); }
        }

        public IDisposable BeginScope<TState>(TState state) where TState : notnull => null!;
        public bool IsEnabled(LogLevel logLevel) => true;

        public void Log<TState>(LogLevel logLevel, EventId eventId, TState state, Exception? exception,
            Func<TState, Exception?, string> formatter)
        {
            var current = Activity.Current?.SpanId ?? default;
            lock (_lines) _lines.Add((logLevel, eventId, formatter(state, exception), current));
        }
    }

    // ---------------------------------------------------------------- observation

    private void Record(Instrument instrument, double value, ReadOnlySpan<KeyValuePair<string, object?>> tags)
    {
        var map = new Dictionary<string, object?>(StringComparer.Ordinal);
        foreach (var tag in tags) map[tag.Key] = tag.Value;
        _measurements.Enqueue((instrument.Name, value, map));
    }

    private int StartedIn(ActivityTraceId traceId)
    {
        lock (_gate) return _started.Count(a => a.TraceId == traceId);
    }

    /// <summary>The stopped spans of one trace, in the order they stopped.</summary>
    private List<Activity> SpansOf(ActivityTraceId traceId)
    {
        lock (_gate) return _stopped.Where(a => a.TraceId == traceId).ToList();
    }

    private static Activity ClipSpan(List<Activity> trace, string model) =>
        Assert.Single(trace, a => a.DisplayName == $"generate_video {model}");

    private static List<Activity> ChildrenOf(List<Activity> trace, Activity parent) =>
        trace.Where(a => a.ParentSpanId == parent.SpanId).ToList();

    private static Activity DitLoadSpan(List<Activity> trace) => Assert.Single(trace, a => a.DisplayName == "dit-load");

    private static IEnumerable<string> Names(IEnumerable<Activity> spans) => spans.Select(a => a.DisplayName);

    private static List<ActivityEvent> StepEvents(Activity span) => span.Events.Where(e => e.Name == "step").ToList();

    private static object? Tag(ActivityEvent e, string key) => e.Tags.FirstOrDefault(t => t.Key == key).Value;

    private List<(string Instrument, double Value, Dictionary<string, object?> Tags)> Measurements(string model, string instrument) =>
        _measurements.Where(m => m.Instrument == instrument
            && Equals(m.Tags.GetValueOrDefault(TelemetryAttributes.GenAiRequestModel), model)).ToList();

    private static string NewModel() => "video-test-" + Guid.NewGuid().ToString("N")[..8];
}
