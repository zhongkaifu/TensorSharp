// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Diagnostics;
using System.Diagnostics.Metrics;
using System.Reflection;
using System.Runtime.CompilerServices;

namespace TensorSharp.Studio;

/// <summary>Studio's traces and metrics. Studio references no other TensorSharp project, so it declares its own
/// source and meter; the host's OpenTelemetry setup subscribes to them by name. With nothing listening,
/// <see cref="ActivitySource.StartActivity(string, ActivityKind)"/> returns null and the instruments drop their
/// measurements, so a host without telemetry pays nothing.
/// <para>Each job is a trace of its own. The worker runs a job long after the request that queued it has returned, so
/// the job span is a root with a link to that request, whose context is kept in <c>job.json</c> and so survives a
/// restart. Spans carry identifiers, kinds, stages and outcomes only: never prompts, take or file names, or job errors
/// (exception messages, FFmpeg stderr), which quote them. A failure is recorded by its exception type.</para></summary>
internal static class StudioTelemetry
{
    public const string SourceName = "TensorSharp.Studio";
    public const string MeterName = "TensorSharp.Studio";
    /// <summary>The opt-in for exporting prompt and output text that the rest of TensorSharp reads.</summary>
    public const string CaptureMessageContentEnvironmentVariable = "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT";

    public const string JobIdAttribute = "tensorsharp.studio.job_id";
    public const string JobKindAttribute = "tensorsharp.studio.job_kind";
    public const string ProjectIdAttribute = "tensorsharp.studio.project_id";
    public const string StageAttribute = "tensorsharp.studio.stage";
    public const string OutcomeAttribute = "tensorsharp.outcome";
    public const string ErrorTypeAttribute = "error.type";
    // A StudioException's code: one of Studio's fixed identifiers (invalid_resolution, unsafe_generation_output, ...),
    // which says what failed where the exception type alone does not. Never its message.
    const string ErrorCodeAttribute = "tensorsharp.studio.error_code";

    static readonly string Version = ResolveVersion();
    public static ActivitySource Source { get; } = new(SourceName, Version);
    public static Meter Meter { get; } = new(MeterName, Version);

    /// <summary>Whether prompt and output text may be exported: <see cref="CaptureMessageContentEnvironmentVariable"/>
    /// set to true, 1, yes or on. Studio's spans carry no such text even then (the generation span under a job records
    /// the prompt under this switch); it is read here so that anything Studio adds later honours the same opt-in.</summary>
    public static bool CaptureMessageContent => IsTruthy(Environment.GetEnvironmentVariable(CaptureMessageContentEnvironmentVariable));

    // A job runs from seconds (a short render) to hours (long clips queued behind each other).
    static readonly double[] DurationBuckets = { 0.5, 1, 2, 5, 10, 20, 30, 60, 120, 300, 600, 1200, 1800, 3600, 7200, 14400 };
    static readonly Histogram<double> JobDuration = Meter.CreateHistogram<double>("tensorsharp.studio.job.duration", "s",
        "Time from queuing a Studio job to its end, by kind and outcome. A job that ends while queued records its wait.",
        advice: new InstrumentAdvice<double> { HistogramBucketBoundaries = DurationBuckets });

    // The services whose jobs the gauges count. Weak, so the gauges never keep a service that was not disposed alive.
    static readonly ConditionalWeakTable<StudioService, object> Services = new();
    static readonly object Tracked = new();

    static StudioTelemetry()
    {
        Meter.CreateObservableGauge("tensorsharp.studio.jobs.queued", () => CountJobs(running: false), "{job}",
            "Studio jobs waiting for the worker.");
        Meter.CreateObservableGauge("tensorsharp.studio.jobs.running", () => CountJobs(running: true), "{job}",
            "Studio jobs the worker is running, including one being cancelled.");
    }

    /// <summary>Counts <paramref name="service"/>'s jobs in the gauges until <see cref="Untrack"/>.</summary>
    public static void Track(StudioService service) => Services.AddOrUpdate(service, Tracked);
    public static void Untrack(StudioService service) => Services.Remove(service);

    // Runs on the metrics collection thread.
    static int CountJobs(bool running)
    {
        int count = 0;
        foreach (var (service, _) in Services)
        {
            var (queued, active) = service.CountActiveJobs();
            count += running ? active : queued;
        }
        return count;
    }

    /// <summary>Keeps the queuing request's W3C trace context on the job record, which is saved in <c>job.json</c>, so
    /// the job's trace links back to it, also after a restart. Nothing is kept while nothing records Studio traces.</summary>
    public static void CaptureQueuingContext(StudioJobRecord record)
    {
        if (!Source.HasListeners() || Activity.Current is not { IdFormat: ActivityIdFormat.W3C } queuing) return;
        record.TraceParent = queuing.Id; record.TraceState = queuing.TraceStateString;
    }

    /// <summary>Starts the span of a job the worker is about to run. It is the root of the job's own trace: the context
    /// the worker inherited from whoever constructed the service is cleared first. It is dated from when the job was
    /// queued, so the queue wait fits inside it, and is left current, so the work the job awaits (the generation's
    /// spans) nests under it. Null when nothing records it.</summary>
    public static Activity? StartJob(StudioJobRecord record)
    {
        Activity.Current = null;
        if (!Source.HasListeners()) return null;
        var job = record.Job;
        var activity = Source.StartActivity(JobSpanName(job.Kind), ActivityKind.Internal, default(ActivityContext), links: Link(record), startTime: job.CreatedAt);
        if (activity is { IsAllDataRequested: true }) Describe(activity, job);
        return activity;
    }

    /// <summary>Ends a span from <see cref="StartJob"/> when the job has finished, at its recorded end and with its
    /// outcome; <paramref name="failure"/> is what failed it, if anything. The queue wait and Studio's own stages are
    /// added as child spans from the times the job recorded. The other stages of a generation are the model's phases,
    /// which the generation's own spans already show.</summary>
    public static void EndJob(Activity? activity, StudioJobRecord record, Exception? failure)
    {
        if (activity == null) return;
        var job = record.Job; var finished = job.FinishedAt ?? DateTimeOffset.UtcNow;
        Conclude(activity, job, failure);
        if (activity.IsAllDataRequested && job.StartedAt is { } started)
        {
            Child(activity, "studio.queue", null, job.CreatedAt, started);
            // The stages are contiguous from the start, each beginning where the one before it ended.
            var cursor = started;
            foreach (var stage in job.Stages)
            {
                var end = cursor.AddSeconds(stage.Seconds);
                if (StageSpanName(stage.Stage) is { } name) Child(activity, name, stage.Stage, cursor, end < finished ? end : finished);
                cursor = end;
            }
        }
        End(activity, finished);
    }

    /// <summary>Records a job that ended without the worker running it in this process: cancelled or stopped while
    /// queued, or found unfinished at startup. Its span is an instant at the job's recorded end, in a trace of its own,
    /// with the outcome Studio gave it and the link to the request that queued it.</summary>
    public static void TraceJobNotRun(StudioJobRecord record)
    {
        if (!Source.HasListeners()) return;
        // Whoever noticed the end (a cancel request, startup or shutdown) is not the job's parent.
        var previous = Activity.Current; Activity.Current = null;
        try
        {
            var job = record.Job; var at = job.FinishedAt ?? DateTimeOffset.UtcNow;
            var activity = Source.StartActivity(JobSpanName(job.Kind), ActivityKind.Internal, default(ActivityContext), links: Link(record), startTime: at);
            if (activity == null) return;
            if (activity.IsAllDataRequested) Describe(activity, job);
            Conclude(activity, job, null);
            End(activity, at);
        }
        finally { Activity.Current = previous; }
    }

    /// <summary>Records a job's time from queuing to its end. Called once as it ends, whatever ended it.</summary>
    public static void RecordJobDuration(StudioJob job)
    {
        if (!JobDuration.Enabled || job.FinishedAt is not { } finished) return;
        JobDuration.Record(Math.Max(0, (finished - job.CreatedAt).TotalSeconds),
            new KeyValuePair<string, object?>(JobKindAttribute, job.Kind), new KeyValuePair<string, object?>(OutcomeAttribute, job.State));
    }

    static void Describe(Activity activity, StudioJob job)
    {
        activity.SetTag(JobIdAttribute, job.Id);
        activity.SetTag(JobKindAttribute, job.Kind);
        activity.SetTag(ProjectIdAttribute, job.ProjectId);
    }

    static void Conclude(Activity activity, StudioJob job, Exception? failure)
    {
        if (failure != null) activity.SetStatus(ActivityStatusCode.Error, failure.GetType().Name);
        if (!activity.IsAllDataRequested) return;
        activity.SetTag(OutcomeAttribute, job.State);
        if (failure == null) return;
        activity.SetTag(ErrorTypeAttribute, failure.GetType().FullName ?? failure.GetType().Name);
        if (failure is StudioException studio) activity.SetTag(ErrorCodeAttribute, studio.Code);
    }

    // A span built after the fact, between times the job recorded.
    static void Child(Activity parent, string name, string? stage, DateTimeOffset start, DateTimeOffset end)
    {
        var child = Source.StartActivity(name, ActivityKind.Internal, parent.Context, startTime: start);
        if (child == null) return;
        if (stage != null && child.IsAllDataRequested) child.SetTag(StageAttribute, stage);
        End(child, end);
    }

    // Stop() replaces a zero duration with "until now", so an instant is given one tick.
    static void End(Activity activity, DateTimeOffset end)
    {
        var at = end.UtcDateTime;
        activity.SetEndTime(at > activity.StartTimeUtc ? at : activity.StartTimeUtc.AddTicks(1));
        activity.Stop();
    }

    static ActivityLink[]? Link(StudioJobRecord record) =>
        ActivityContext.TryParse(record.TraceParent, record.TraceState, out var queuing) ? new[] { new ActivityLink(queuing) } : null;

    // Fixed names, so a job record read from disk cannot add span names.
    static string JobSpanName(string kind) => kind switch
    {
        "generate" => "studio.job generate",
        "render" => "studio.job render",
        _ => "studio.job",
    };

    // Studio's own stages. A generation's other stages are the model's phases (text-encode, denoise, vae-decode, ...),
    // which the generation's spans already trace.
    static string? StageSpanName(string stage) => stage switch
    {
        "starting" => "studio.stage starting",
        "model-identity" => "studio.stage model-identity",
        "encoding" => "studio.stage encoding",
        "importing" => "studio.stage importing",
        "cancelling" => "studio.stage cancelling",
        "rendering" => "studio.stage rendering",
        "validating" => "studio.stage validating",
        _ => null,
    };

    /// <summary>True for 1, true, yes and on (any case, surrounding whitespace ignored), as TensorSharp's other
    /// switches read them.</summary>
    internal static bool IsTruthy(string? value)
    {
        if (string.IsNullOrWhiteSpace(value)) return false;
        var v = value.Trim();
        return v == "1" || v.Equals("true", StringComparison.OrdinalIgnoreCase) || v.Equals("yes", StringComparison.OrdinalIgnoreCase)
            || v.Equals("on", StringComparison.OrdinalIgnoreCase);
    }

    static string ResolveVersion()
    {
        var assembly = typeof(StudioTelemetry).Assembly;
        var informational = assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion;
        if (!string.IsNullOrEmpty(informational))
        {
            // Without the "+<commit>" Source Link appends: dashboards group by the version.
            int plus = informational.IndexOf('+');
            return plus > 0 ? informational[..plus] : informational;
        }
        return assembly.GetName().Version?.ToString() ?? "0.0.0";
    }
}
