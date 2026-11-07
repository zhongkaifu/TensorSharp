// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Diagnostics;
using System.Diagnostics.Metrics;
using System.Text.Json;
using System.Text.Json.Nodes;
using TensorSharp.Studio;

namespace TensorSharp.Studio.Tests.Telemetry;

/// <summary>Tests that listen to Studio's activity source or meter. Run on their own: a listener makes every job in the
/// process traced, and a test without one asserts that nothing is recorded.</summary>
[CollectionDefinition(Name, DisableParallelization = true)]
public sealed class StudioTelemetryCollection
{
    public const string Name = "Studio telemetry";
}

/// <summary>Studio's job spans and job metric, built from what a job records (its times and stages) and from the trace
/// context kept in <c>job.json</c>. Driven through <see cref="StudioTelemetry"/> with hand-built job records, and once
/// through a restart that finds a job unfinished. No media and no model.</summary>
[Collection(StudioTelemetryCollection.Name)]
public sealed class StudioTelemetryTests : IDisposable
{
    const string ErrorCodeAttribute = "tensorsharp.studio.error_code";
    static readonly DateTimeOffset T0 = new(2026, 3, 1, 9, 0, 0, TimeSpan.Zero);

    readonly List<Activity> stopped = [];
    readonly ActivityListener listener;
    readonly string root = Path.Combine(Path.GetTempPath(), "studio-telemetry-" + StudioJson.NewId());

    public StudioTelemetryTests()
    {
        listener = new ActivityListener
        {
            ShouldListenTo = source => source.Name == StudioTelemetry.SourceName,
            Sample = (ref ActivityCreationOptions<ActivityContext> _) => ActivitySamplingResult.AllDataAndRecorded,
            ActivityStopped = activity => { lock (stopped) stopped.Add(activity); },
        };
        ActivitySource.AddActivityListener(listener);
    }

    public void Dispose()
    {
        listener.Dispose();
        if (Directory.Exists(root)) Directory.Delete(root, true);
    }

    // ------------------------------------------------------------ a job that never ran

    [Fact]
    public void AJobThatNeverRanIsAnInstantAtItsEndLinkedToTheRequestThatQueuedIt()
    {
        var queuing = NewQueuingContext();
        var finished = T0.AddSeconds(42);
        var record = new StudioJobRecord
        {
            Job = new StudioJob { Kind = "generate", State = "cancelled", ProjectId = StudioJson.NewId(), CreatedAt = T0, FinishedAt = finished },
            TraceParent = queuing.TraceParent,
            TraceState = "vendor=value",
        };
        // Whoever noticed the end (here, a cancel request) is not the job's parent, and keeps its own context.
        using var cancelRequest = new Activity("POST /studio/jobs/cancel").Start();

        StudioTelemetry.TraceJobNotRun(record);

        Assert.Same(cancelRequest, Activity.Current);
        var span = JobSpan(record.Job.Id);
        Assert.Equal("studio.job generate", span.DisplayName);
        Assert.Equal(ActivityKind.Internal, span.Kind);
        Assert.Equal(default(ActivitySpanId), span.ParentSpanId);
        Assert.NotEqual(cancelRequest.TraceId, span.TraceId);
        Assert.NotEqual(queuing.TraceId, span.TraceId);
        Assert.Equal(finished.UtcDateTime, span.StartTimeUtc);
        // Stop() would stretch a zero duration to "now"; an instant is one tick.
        Assert.Equal(TimeSpan.FromTicks(1), span.Duration);
        Assert.Equal(ActivityStatusCode.Unset, span.Status);
        Assert.Equal("cancelled", span.GetTagItem(StudioTelemetry.OutcomeAttribute));
        Assert.Equal(record.Job.Id, span.GetTagItem(StudioTelemetry.JobIdAttribute));
        Assert.Equal("generate", span.GetTagItem(StudioTelemetry.JobKindAttribute));
        Assert.Equal(record.Job.ProjectId, span.GetTagItem(StudioTelemetry.ProjectIdAttribute));

        var link = Assert.Single(span.Links);
        Assert.Equal(queuing.TraceId, link.Context.TraceId);
        Assert.Equal(queuing.SpanId, link.Context.SpanId);
        Assert.Equal("vendor=value", link.Context.TraceState);
        Assert.DoesNotContain(Stopped(), a => a.ParentSpanId == span.SpanId);
    }

    [Theory]
    [InlineData("generate", "studio.job generate")]
    [InlineData("render", "studio.job render")]
    [InlineData("anything a file on disk says", "studio.job")]
    public void JobSpanNamesAreFixedWhateverTheRecordSays(string kind, string expected)
    {
        var record = new StudioJobRecord { Job = new StudioJob { Kind = kind, State = "interrupted", CreatedAt = T0, FinishedAt = T0.AddSeconds(1) } };

        StudioTelemetry.TraceJobNotRun(record);

        Assert.Equal(expected, JobSpan(record.Job.Id).DisplayName);
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("not a traceparent")]
    public void AJobWithoutAUsableStoredContextHasNoLink(string? traceParent)
    {
        var record = new StudioJobRecord
        {
            Job = new StudioJob { Kind = "render", State = "cancelled", CreatedAt = T0, FinishedAt = T0.AddSeconds(2) },
            TraceParent = traceParent,
        };

        StudioTelemetry.TraceJobNotRun(record);

        Assert.Empty(JobSpan(record.Job.Id).Links);
    }

    // ------------------------------------------------------------ a job the worker ran

    [Fact]
    public void ARunJobSpansItsQueueWaitAndStudioStagesFromTheTimesItRecorded()
    {
        var job = new StudioJob
        {
            Kind = "render", State = "running", Stage = "starting", ProjectId = StudioJson.NewId(),
            CreatedAt = T0, StartedAt = T0.AddSeconds(5),
        };
        var record = new StudioJobRecord { Job = job, TraceParent = NewQueuingContext().TraceParent };
        // The worker inherited the context of whoever constructed the service; the job is a trace of its own.
        using var constructor = new Activity("constructed the service").Start();

        var activity = StudioTelemetry.StartJob(record);

        Assert.NotNull(activity);
        Assert.Same(activity, Activity.Current);
        Assert.Equal(default(ActivitySpanId), activity.ParentSpanId);
        Assert.NotEqual(constructor.TraceId, activity.TraceId);
        Assert.Single(activity.Links);

        // The stages the job left, in order; text-encode is a model phase the generation's spans already show, and the
        // last stage ran past the recorded end, so it is cut there.
        job.Stages = [new("starting", 2), new("model-identity", 1), new("text-encode", 10), new("encoding", 3), new("rendering", 10)];
        job.State = "completed";
        job.FinishedAt = T0.AddSeconds(25);
        StudioTelemetry.EndJob(activity, record, null);

        var spans = Stopped();
        var span = JobSpan(job.Id);
        AssertSpan(span, "studio.job render", T0, T0.AddSeconds(25));
        Assert.Equal("completed", span.GetTagItem(StudioTelemetry.OutcomeAttribute));
        Assert.Equal(ActivityStatusCode.Unset, span.Status);

        var children = spans.Where(a => a.ParentSpanId == span.SpanId).ToList();
        Assert.Equal(new[] { "studio.queue", "studio.stage starting", "studio.stage model-identity", "studio.stage encoding", "studio.stage rendering" },
            children.Select(c => c.DisplayName));
        AssertSpan(children[0], "studio.queue", T0, T0.AddSeconds(5));
        AssertSpan(children[1], "studio.stage starting", T0.AddSeconds(5), T0.AddSeconds(7));
        AssertSpan(children[2], "studio.stage model-identity", T0.AddSeconds(7), T0.AddSeconds(8));
        AssertSpan(children[3], "studio.stage encoding", T0.AddSeconds(18), T0.AddSeconds(21));
        AssertSpan(children[4], "studio.stage rendering", T0.AddSeconds(21), T0.AddSeconds(25));
        Assert.Null(children[0].GetTagItem(StudioTelemetry.StageAttribute));
        Assert.Equal(new object?[] { "starting", "model-identity", "encoding", "rendering" },
            children.Skip(1).Select(c => c.GetTagItem(StudioTelemetry.StageAttribute)));
        Assert.All(children, c => Assert.Equal(span.TraceId, c.TraceId));
        // The job span ends after the spans built under it.
        Assert.True(spans.IndexOf(span) > children.Max(c => spans.IndexOf(c)));
    }

    [Fact]
    public void AFailedJobRecordsTheExceptionTypeAndStudioCodeNeverTheMessage()
    {
        var studioFailure = RunAndFail("generate", new StudioException("unsafe_generation_output", "The take 'secret-garden.mp4' was rejected"));
        Assert.Equal(ActivityStatusCode.Error, studioFailure.Status);
        Assert.Equal(nameof(StudioException), studioFailure.StatusDescription);
        Assert.Equal("failed", studioFailure.GetTagItem(StudioTelemetry.OutcomeAttribute));
        Assert.Equal(typeof(StudioException).FullName, studioFailure.GetTagItem(StudioTelemetry.ErrorTypeAttribute));
        Assert.Equal("unsafe_generation_output", studioFailure.GetTagItem(ErrorCodeAttribute));

        var otherFailure = RunAndFail("render", new IOException("FFmpeg could not write C:\\secret\\out.mp4"));
        Assert.Equal(ActivityStatusCode.Error, otherFailure.Status);
        Assert.Equal(typeof(IOException).FullName, otherFailure.GetTagItem(StudioTelemetry.ErrorTypeAttribute));
        Assert.Null(otherFailure.GetTagItem(ErrorCodeAttribute));

        Assert.DoesNotContain(Stopped().SelectMany(a => a.TagObjects), t => t.Value is string s && s.Contains("secret", StringComparison.Ordinal));
        Assert.DoesNotContain(Stopped(), a => a.StatusDescription?.Contains("secret", StringComparison.Ordinal) == true);
    }

    [Fact]
    public void EndingAJobThatWasNotRecordedDoesNothing()
    {
        var record = new StudioJobRecord { Job = new StudioJob { Kind = "render", State = "completed", CreatedAt = T0, StartedAt = T0, FinishedAt = T0.AddSeconds(1) } };

        StudioTelemetry.EndJob(null, record, new InvalidOperationException());

        Assert.DoesNotContain(Stopped(), a => Equals(a.GetTagItem(StudioTelemetry.JobIdAttribute), record.Job.Id));
    }

    // ------------------------------------------------------------ the stored trace context

    [Fact]
    public void TheQueuingRequestsContextIsKeptOnTheRecordAndSurvivesJobJson()
    {
        using var queuing = new Activity("POST /studio/jobs").SetIdFormat(ActivityIdFormat.W3C);
        queuing.TraceStateString = "tensorsharp=1";
        queuing.Start();
        var record = new StudioJobRecord { Job = new StudioJob { Kind = "render", CreatedAt = T0 } };

        StudioTelemetry.CaptureQueuingContext(record);

        Assert.Equal(queuing.Id, record.TraceParent);
        Assert.Equal("tensorsharp=1", record.TraceState);

        // job.json, as Studio writes and reads it.
        var path = Path.Combine(root, "jobs", record.Job.Id, "job.json");
        StudioStorage.Save(path, record);
        var json = JsonNode.Parse(File.ReadAllText(path))!;
        Assert.Equal(queuing.Id, (string?)json["traceParent"]);
        Assert.Equal("tensorsharp=1", (string?)json["traceState"]);
        var reloaded = StudioStorage.Read<StudioJobRecord>(path);
        Assert.Equal(record.TraceParent, reloaded.TraceParent);
        Assert.Equal(record.TraceState, reloaded.TraceState);

        // After a restart the job's span still links to the request.
        reloaded.Job.State = "interrupted";
        reloaded.Job.FinishedAt = T0.AddSeconds(9);
        StudioTelemetry.TraceJobNotRun(reloaded);
        var link = Assert.Single(JobSpan(record.Job.Id).Links);
        Assert.Equal(queuing.TraceId, link.Context.TraceId);
        Assert.Equal(queuing.SpanId, link.Context.SpanId);
        Assert.Equal("tensorsharp=1", link.Context.TraceState);
    }

    [Fact]
    public void OnlyAW3CContextIsKept()
    {
        var none = new StudioJobRecord();
        Activity.Current = null;
        StudioTelemetry.CaptureQueuingContext(none);
        Assert.Null(none.TraceParent);
        Assert.Null(none.TraceState);

        using var hierarchical = new Activity("legacy client").SetIdFormat(ActivityIdFormat.Hierarchical).Start();
        var legacy = new StudioJobRecord();
        StudioTelemetry.CaptureQueuingContext(legacy);
        Assert.Null(legacy.TraceParent);
        Assert.Null(legacy.TraceState);
    }

    [Fact]
    public void AnOlderJobJsonWithoutTheFieldsStillLoadsAndNullFieldsAreLeftOut()
    {
        var jobId = StudioJson.NewId();
        var path = Path.Combine(root, "jobs", jobId, "job.json");
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        // What a host from before the trace fields wrote.
        File.WriteAllText(path, JsonSerializer.Serialize(new
        {
            job = new StudioJob { Id = jobId, ProjectId = StudioJson.NewId(), Kind = "render", State = "completed", Stage = "completed", CreatedAt = T0 },
            fingerprint = "saved",
        }, StudioJson.Options));

        var loaded = StudioStorage.Read<StudioJobRecord>(path);

        Assert.Equal(jobId, loaded.Job.Id);
        Assert.Equal("saved", loaded.Fingerprint);
        Assert.Null(loaded.TraceParent);
        Assert.Null(loaded.TraceState);

        // Saved again without a context, the file has no trace fields for an older host to trip over.
        StudioStorage.Save(path, loaded);
        var json = JsonNode.Parse(File.ReadAllText(path))!.AsObject();
        Assert.False(json.ContainsKey("traceParent"));
        Assert.False(json.ContainsKey("traceState"));
        var text = JsonSerializer.Serialize(new StudioJobRecord(), StudioJson.Options);
        Assert.DoesNotContain("traceParent", text, StringComparison.Ordinal);
        Assert.DoesNotContain("traceState", text, StringComparison.Ordinal);
    }

    [Fact]
    public async Task AJobFoundUnfinishedAtStartupIsTracedAsInterruptedWithItsStoredLink()
    {
        string projectId;
        await using (var first = new StudioService(root)) projectId = (await first.CreateProjectAsync()).Id;
        var queuing = NewQueuingContext();
        var jobId = StudioJson.NewId();
        var directory = Path.Combine(root, "jobs", jobId);
        Directory.CreateDirectory(directory);
        var lastSaved = T0.AddSeconds(3);
        // A render the previous process queued and never started.
        File.WriteAllText(Path.Combine(directory, "job.json"), JsonSerializer.Serialize(new
        {
            job = new StudioJob
            {
                Id = jobId, ProjectId = projectId, RequestId = StudioJson.NewId(), Kind = "render",
                State = "queued", Stage = "queued", CreatedAt = T0, UpdatedAt = lastSaved,
            },
            fingerprint = "saved",
            traceParent = queuing.TraceParent,
            traceState = "vendor=1",
        }, StudioJson.Options));

        await using (var restarted = new StudioService(root))
            Assert.Equal("interrupted", Assert.Single(await restarted.ListJobsAsync(projectId)).State);

        var span = JobSpan(jobId);
        Assert.Equal("studio.job render", span.DisplayName);
        Assert.Equal("interrupted", span.GetTagItem(StudioTelemetry.OutcomeAttribute));
        Assert.Equal(default(ActivitySpanId), span.ParentSpanId);
        // The last save is the latest moment the job is known to have been alive.
        Assert.Equal(lastSaved.UtcDateTime, span.StartTimeUtc);
        Assert.Equal(TimeSpan.FromTicks(1), span.Duration);
        var link = Assert.Single(span.Links);
        Assert.Equal(queuing.TraceId, link.Context.TraceId);
        Assert.Equal(queuing.SpanId, link.Context.SpanId);
        Assert.Equal("vendor=1", link.Context.TraceState);

        // The rewritten job.json keeps the context.
        var json = JsonNode.Parse(File.ReadAllText(Path.Combine(directory, "job.json")))!;
        Assert.Equal("interrupted", (string?)json["job"]!["state"]);
        Assert.Equal(queuing.TraceParent, (string?)json["traceParent"]);
        Assert.Equal("vendor=1", (string?)json["traceState"]);
    }

    // ------------------------------------------------------------ the job metric

    [Fact]
    public void AJobRecordsItsDurationFromQueuingToItsEnd()
    {
        var measurements = new List<(double Value, Dictionary<string, object?> Tags)>();
        using var meters = new MeterListener
        {
            InstrumentPublished = (instrument, l) =>
            {
                if (instrument.Meter.Name == StudioTelemetry.MeterName && instrument.Name == "tensorsharp.studio.job.duration")
                    l.EnableMeasurementEvents(instrument);
            },
        };
        meters.SetMeasurementEventCallback<double>((_, value, tags, _) =>
        {
            var map = new Dictionary<string, object?>(StringComparer.Ordinal);
            foreach (var tag in tags) map[tag.Key] = tag.Value;
            lock (measurements) measurements.Add((value, map));
        });
        meters.Start();

        StudioTelemetry.RecordJobDuration(new StudioJob { Kind = "render", State = "completed", CreatedAt = T0, FinishedAt = T0.AddSeconds(1234.5) });
        // A clock that moved back records no negative time.
        StudioTelemetry.RecordJobDuration(new StudioJob { Kind = "generate", State = "cancelled", CreatedAt = T0, FinishedAt = T0.AddSeconds(-2) });
        // A job that has not ended records nothing.
        StudioTelemetry.RecordJobDuration(new StudioJob { Kind = "render", State = "running", CreatedAt = T0 });

        List<(double Value, Dictionary<string, object?> Tags)> recorded;
        lock (measurements) recorded = [.. measurements];
        Assert.Equal(2, recorded.Count);
        Assert.Equal(1234.5, recorded[0].Value);
        Assert.Equal("render", recorded[0].Tags[StudioTelemetry.JobKindAttribute]);
        Assert.Equal("completed", recorded[0].Tags[StudioTelemetry.OutcomeAttribute]);
        Assert.Equal(0.0, recorded[1].Value);
        Assert.Equal("generate", recorded[1].Tags[StudioTelemetry.JobKindAttribute]);
        Assert.Equal("cancelled", recorded[1].Tags[StudioTelemetry.OutcomeAttribute]);
        // Identifiers stay on spans.
        Assert.All(recorded, m => Assert.Equal(2, m.Tags.Count));
    }

    [Theory]
    [InlineData("1", true)]
    [InlineData(" TRUE ", true)]
    [InlineData("yes", true)]
    [InlineData("On", true)]
    [InlineData("0", false)]
    [InlineData("false", false)]
    [InlineData("", false)]
    [InlineData(null, false)]
    public void TheContentSwitchReadsTheSpellingsTheRestOfTensorSharpReads(string? value, bool expected) =>
        Assert.Equal(expected, StudioTelemetry.IsTruthy(value));

    // ------------------------------------------------------------ helpers

    Activity RunAndFail(string kind, Exception failure)
    {
        var job = new StudioJob { Kind = kind, State = "running", CreatedAt = T0, StartedAt = T0.AddSeconds(1) };
        var record = new StudioJobRecord { Job = job };
        var activity = StudioTelemetry.StartJob(record);
        Assert.NotNull(activity);
        job.Stages = [new("starting", 1)];
        job.State = "failed";
        job.FinishedAt = T0.AddSeconds(4);
        StudioTelemetry.EndJob(activity, record, failure);
        return JobSpan(job.Id);
    }

    static (string TraceParent, ActivityTraceId TraceId, ActivitySpanId SpanId) NewQueuingContext()
    {
        var traceId = ActivityTraceId.CreateRandom();
        var spanId = ActivitySpanId.CreateRandom();
        return ($"00-{traceId.ToHexString()}-{spanId.ToHexString()}-01", traceId, spanId);
    }

    static void AssertSpan(Activity span, string name, DateTimeOffset start, DateTimeOffset end)
    {
        Assert.Equal(name, span.DisplayName);
        Assert.Equal(start.UtcDateTime, span.StartTimeUtc);
        Assert.Equal(end - start, span.Duration);
    }

    List<Activity> Stopped() { lock (stopped) return [.. stopped]; }

    Activity JobSpan(string jobId) => Assert.Single(Stopped(), a => a.DisplayName.StartsWith("studio.job", StringComparison.Ordinal)
        && Equals(a.GetTagItem(StudioTelemetry.JobIdAttribute), jobId));
}

/// <summary>With nothing listening to Studio's source, a job keeps no context and leaves no span.</summary>
[Collection(StudioTelemetryCollection.Name)]
public sealed class StudioTelemetryWithoutListenerTests
{
    [Fact]
    public void NothingIsKeptOrStartedWhileNothingRecordsStudioTraces()
    {
        using var queuing = new Activity("POST /studio/jobs").SetIdFormat(ActivityIdFormat.W3C).Start();
        var record = new StudioJobRecord { Job = new StudioJob { Kind = "render", State = "cancelled", FinishedAt = DateTimeOffset.UtcNow } };

        StudioTelemetry.CaptureQueuingContext(record);
        Assert.Null(record.TraceParent);
        Assert.Null(record.TraceState);

        StudioTelemetry.TraceJobNotRun(record);
        Assert.Same(queuing, Activity.Current);

        Assert.Null(StudioTelemetry.StartJob(record));
        // StartJob always clears the inherited context: the worker's jobs never nest under its constructor.
        Assert.Null(Activity.Current);
    }
}
