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
using TensorSharp.Models.QwenImage;
using TensorSharp.Runtime.Telemetry;

namespace InferenceWeb.Tests.Telemetry;

/// <summary>
/// The <c>generate_image</c> span and duration metric of one picture, from
/// <see cref="ImageGenerationTelemetry"/> as the server's image routes and the CLI drive it. No
/// model: the test plays the request. In the global-state collection because the
/// content-capture switch is process-wide; each test filters by its own model name.
/// </summary>
[Collection(TelemetryGlobalStateCollection.Name)]
public sealed class ImageTelemetryTests : IDisposable
{
    private const string GenerationDuration = "tensorsharp.image.generation.duration";
    private const string Prompt = "A small orange cat beside a blue ceramic vase";

    private readonly object _gate = new();
    private readonly List<Activity> _stopped = new();
    private readonly ConcurrentQueue<(string Instrument, double Value, Dictionary<string, object?> Tags)> _measurements = new();
    private readonly ActivityListener _activities;
    private readonly MeterListener _meters;

    public ImageTelemetryTests()
    {
        TensorSharpTelemetry.ConfigureCaptureMessageContent(false);
        _activities = new ActivityListener
        {
            ShouldListenTo = s => s.Name.StartsWith(TensorSharpTelemetry.NamePrefix, StringComparison.Ordinal),
            Sample = (ref ActivityCreationOptions<ActivityContext> _) => ActivitySamplingResult.AllDataAndRecorded,
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
        _meters.SetMeasurementEventCallback<double>((i, v, tags, _) =>
        {
            var copy = new Dictionary<string, object?>();
            foreach (var t in tags) copy[t.Key] = t.Value;
            _measurements.Enqueue((i.Name, v, copy));
        });
        _meters.Start();
    }

    public void Dispose()
    {
        _activities.Dispose();
        _meters.Dispose();
        TensorSharpTelemetry.ConfigureCaptureMessageContent(null);
    }

    [Fact]
    public void AGeneratedPictureCarriesWhatItAskedForAndWhatItWasMadeWith()
    {
        string model = NewModel();
        using (var image = ImageGenerationTelemetry.Start(model, "ggml_cuda", generate: true))
        {
            // Size given, steps and CFG left to the model, seed 0 (a seed like any other).
            image.SetRequest(Prompt, new QwenImageParams { Width = 1024, Height = 1024, Seed = 0 }, references: 0);
            for (int step = 1; step <= 8; step++) image.RecordStep(8);
            image.SetResult(1024, 1024);
        }

        var span = Span(model);
        Assert.Equal("generate_image " + model, span.DisplayName);
        Assert.Equal("image", span.GetTagItem(TelemetryAttributes.MediaKind));
        Assert.Equal("generate", span.GetTagItem(TelemetryAttributes.ImageMode));
        Assert.Equal(model, span.GetTagItem(TelemetryAttributes.GenAiRequestModel));
        Assert.Equal("ggml_cuda", span.GetTagItem(TelemetryAttributes.Backend));
        Assert.Equal(1024, span.GetTagItem(TelemetryAttributes.ImageWidth));
        Assert.Equal(1024, span.GetTagItem(TelemetryAttributes.ImageHeight));
        Assert.Equal(8, span.GetTagItem(TelemetryAttributes.ImageSteps));
        Assert.Equal(0L, span.GetTagItem(TelemetryAttributes.ImageSeed));
        Assert.Null(span.GetTagItem(TelemetryAttributes.ImageCfg));
        Assert.Null(span.GetTagItem(TelemetryAttributes.ImageReferences));
        Assert.Null(span.GetTagItem(TelemetryAttributes.ImagePrompt));
        Assert.Equal(TelemetryAttributes.OutcomeCompleted, span.GetTagItem(TelemetryAttributes.Outcome));

        var sample = Assert.Single(Measurements(model));
        Assert.Equal("generate", sample.Tags[TelemetryAttributes.ImageMode]);
        Assert.Equal(TelemetryAttributes.OutcomeCompleted, sample.Tags[TelemetryAttributes.Outcome]);
    }

    [Fact]
    public void ThePromptIsOnTheSpanOnlyWithContentCaptureOn()
    {
        TensorSharpTelemetry.ConfigureCaptureMessageContent(false);
        string off = NewModel();
        using (var image = ImageGenerationTelemetry.Start(off, null, generate: false))
            image.SetRequest(Prompt, new QwenImageParams { Steps = 8, CfgScale = 1f, Seed = 42 }, references: 2);
        Assert.Null(Span(off).GetTagItem(TelemetryAttributes.ImagePrompt));

        TensorSharpTelemetry.ConfigureCaptureMessageContent(true);
        string on = NewModel();
        using (var image = ImageGenerationTelemetry.Start(on, null, generate: false))
            image.SetRequest(Prompt, new QwenImageParams { Steps = 8, CfgScale = 1f, Seed = 42 }, references: 2);
        var span = Span(on);
        Assert.Equal(Prompt, span.GetTagItem(TelemetryAttributes.ImagePrompt));
        Assert.Equal("edit", span.GetTagItem(TelemetryAttributes.ImageMode));
        Assert.Equal(2, span.GetTagItem(TelemetryAttributes.ImageReferences));
        Assert.Equal(8, span.GetTagItem(TelemetryAttributes.ImageSteps));
        Assert.Equal(1.0, span.GetTagItem(TelemetryAttributes.ImageCfg));
        Assert.Equal(42L, span.GetTagItem(TelemetryAttributes.ImageSeed));
        Assert.Null(span.GetTagItem(TelemetryAttributes.Backend));

        // Never on a metric, whatever the switch says.
        Assert.DoesNotContain(_measurements, m => m.Tags.Values.Any(v => v is string s && s.Contains("cat", StringComparison.Ordinal)));
    }

    [Fact]
    public void TheFirstOutcomeMarkedIsTheOneRecorded()
    {
        string refused = NewModel();
        using (var image = ImageGenerationTelemetry.Start(refused, null, generate: true))
        {
            image.MarkRefused("invalid_request");
            image.MarkFailed(new InvalidOperationException("later"));
        }
        Assert.Equal(TelemetryAttributes.OutcomeRefused, Span(refused).GetTagItem(TelemetryAttributes.Outcome));
        Assert.Equal("invalid_request", Span(refused).GetTagItem(TelemetryAttributes.ErrorType));
        Assert.Equal(ActivityStatusCode.Unset, Span(refused).Status);
        Assert.Equal("invalid_request", Assert.Single(Measurements(refused)).Tags[TelemetryAttributes.ErrorType]);

        string cancelled = NewModel();
        using (var image = ImageGenerationTelemetry.Start(cancelled, null, generate: true))
            image.MarkFailed(new OperationCanceledException());
        Assert.Equal(TelemetryAttributes.OutcomeCancelled, Span(cancelled).GetTagItem(TelemetryAttributes.Outcome));
        Assert.Null(Span(cancelled).GetTagItem(TelemetryAttributes.ErrorType));

        string failed = NewModel();
        using (var image = ImageGenerationTelemetry.Start(failed, null, generate: true))
            image.MarkFailed(new InvalidOperationException("out of device memory"));
        var span = Span(failed);
        Assert.Equal(TelemetryAttributes.OutcomeFailed, span.GetTagItem(TelemetryAttributes.Outcome));
        Assert.Equal(typeof(InvalidOperationException).FullName, span.GetTagItem(TelemetryAttributes.ErrorType));
        Assert.Equal(ActivityStatusCode.Error, span.Status);
        Assert.DoesNotContain(span.TagObjects, t => t.Value is string s && s.Contains("memory", StringComparison.Ordinal));
    }

    [Fact]
    public void ItIsRecordedOnceAndAMarkAfterTheEndIsIgnored()
    {
        string model = NewModel();
        var image = ImageGenerationTelemetry.Start(model, null, generate: true);
        image.Dispose();
        image.Dispose();
        image.MarkFailed(new InvalidOperationException());
        Assert.Single(Measurements(model));
        Assert.Equal(TelemetryAttributes.OutcomeCompleted, Span(model).GetTagItem(TelemetryAttributes.Outcome));
    }

    [Fact]
    public void AStepCountOfZeroLeavesTheRequestedOneInPlace()
    {
        string model = NewModel();
        using (var image = ImageGenerationTelemetry.Start(model, null, generate: true))
        {
            image.SetRequest(null, new QwenImageParams { Steps = 25, Seed = 1 }, references: 0);
            image.RecordStep(0);
            image.SetResult(0, 0);
        }
        var span = Span(model);
        Assert.Equal(25, span.GetTagItem(TelemetryAttributes.ImageSteps));
        Assert.Null(span.GetTagItem(TelemetryAttributes.ImageWidth));
        Assert.Null(span.GetTagItem(TelemetryAttributes.ImagePrompt));
    }

    private static string NewModel() => $"qwen-image-test-{Guid.NewGuid():N}.gguf";

    private Activity Span(string model)
    {
        lock (_gate)
            return Assert.Single(_stopped, a => a.DisplayName == "generate_image " + model);
    }

    private List<(string Instrument, double Value, Dictionary<string, object?> Tags)> Measurements(string model) =>
        _measurements.Where(m => m.Instrument == GenerationDuration
            && m.Tags.TryGetValue(TelemetryAttributes.GenAiRequestModel, out var v) && Equals(v, model)).ToList();
}
