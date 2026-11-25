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
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Telemetry;
using TensorSharp.Server;
using Xunit;

namespace InferenceWeb.Tests.Telemetry;

public class PromptMediaTelemetryTests
{
    private static List<ChatMessage> VideoHistory() => new()
    {
        new() { Role = "system", Content = "describe" },
        new() { Role = "user", Content = "a picture", ImagePaths = ["a.png"] },
        // A video message with a still image among its frames: the untimed one is an image.
        new()
        {
            Role = "user", Content = "a clip", IsVideo = true, ImagePaths = ["f0.png", "f1.png", "still.png"],
            ImageTimestamps = [0.0, 0.5, null], VideoFilePaths = ["clip.mp4"],
        },
        // Sampled frames without times are frames too.
        new() { Role = "user", Content = "another clip", IsVideo = true, ImagePaths = ["g0.png", "g1.png"] },
        new() { Role = "user", Content = "a recording", AudioPaths = ["a.wav"] },
    };

    [Fact]
    public void TheRecordCountsAVideosFramesApartFromStillImagesAndAudio()
    {
        var record = ChatGenerationPipeline.MediaPreparationRecord.Begin(VideoHistory());

        Assert.Equal((2, 4, 1, 1), (record.Images, record.VideoFrames, record.Videos, record.AudioClips));
        Assert.Null(record.EndUtc);
    }

    private static (Activity Render, ConcurrentQueue<Activity> Media, ActivityListener Listener) StartRender(DateTime start)
    {
        var parent = new ActivityContext(ActivityTraceId.CreateRandom(), ActivitySpanId.CreateRandom(), ActivityTraceFlags.Recorded);
        var media = new ConcurrentQueue<Activity>();
        var listener = new ActivityListener
        {
            ShouldListenTo = s => s.Name == TensorSharpTelemetry.InferenceSourceName,
            Sample = (ref ActivityCreationOptions<ActivityContext> _) => ActivitySamplingResult.AllDataAndRecorded,
            ActivityStopped = a =>
            {
                if (a.TraceId == parent.TraceId && a.DisplayName == "prompt.media")
                    media.Enqueue(a);
            },
        };
        ActivitySource.AddActivityListener(listener);
        var render = TensorSharpTelemetry.Inference.StartActivity("prompt.render", ActivityKind.Internal, parent,
            startTime: TensorSharpTelemetry.ToUtcOffset(start))!;
        return (render, media, listener);
    }

    [Fact]
    public void PromptMediaIsAChildOfPromptRenderWithWhatThePreparationProduced()
    {
        var record = ChatGenerationPipeline.MediaPreparationRecord.Begin(VideoHistory());
        record.LockedUtc = record.StartUtc.AddMilliseconds(40);
        record.Preparations = 2;
        record.Finish(new[] { new PromptMediaSpan(10, 110, "a"), new PromptMediaSpan(110, 160, "b") });
        DateTime renderStart = record.StartUtc.AddMilliseconds(-200);
        DateTime renderEnd = record.EndUtc!.Value.AddMilliseconds(300);
        var (render, media, listener) = StartRender(renderStart);
        using (listener)
        {
            record.Record(render, renderStart, renderEnd);
            render.Stop();
        }

        var span = Assert.Single(media);
        Assert.Equal(render.SpanId, span.ParentSpanId);
        Assert.Equal(record.StartUtc, span.StartTimeUtc);
        Assert.Equal(record.EndUtc!.Value, span.StartTimeUtc + span.Duration);
        Assert.Equal(2, span.GetTagItem(TelemetryAttributes.MediaImages));
        Assert.Equal(4, span.GetTagItem(TelemetryAttributes.MediaVideoFrames));
        Assert.Equal(1, span.GetTagItem(TelemetryAttributes.MediaVideos));
        Assert.Equal(1, span.GetTagItem(TelemetryAttributes.MediaAudio));
        Assert.Equal(40.0, span.GetTagItem(TelemetryAttributes.MediaLockWaitMs));
        Assert.Equal(2, span.GetTagItem(TelemetryAttributes.MediaSpans));
        Assert.Equal(150, span.GetTagItem(TelemetryAttributes.MediaTokens));
        Assert.Equal(2, span.GetTagItem(TelemetryAttributes.MediaPreparations));
    }

    [Fact]
    public void APreparationThatDidNotFinishEndsWithTheRenderWithoutSpansOrTokens()
    {
        var record = ChatGenerationPipeline.MediaPreparationRecord.Begin(VideoHistory());
        record.Preparations = 1;
        DateTime renderStart = record.StartUtc.AddMilliseconds(-50);
        DateTime renderEnd = record.StartUtc.AddMilliseconds(500);
        var (render, media, listener) = StartRender(renderStart);
        using (listener)
        {
            record.Record(render, renderStart, renderEnd);
            render.Stop();
        }

        var span = Assert.Single(media);
        Assert.Equal(renderEnd, span.StartTimeUtc + span.Duration);
        Assert.Null(span.GetTagItem(TelemetryAttributes.MediaLockWaitMs));
        Assert.Null(span.GetTagItem(TelemetryAttributes.MediaSpans));
        Assert.Null(span.GetTagItem(TelemetryAttributes.MediaTokens));
        Assert.Null(span.GetTagItem(TelemetryAttributes.MediaPreparations));
    }
}
