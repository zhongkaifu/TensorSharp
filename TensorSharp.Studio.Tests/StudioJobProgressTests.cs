using System.Diagnostics;
using System.Text.Json.Nodes;
using TensorSharp.Studio;

namespace TensorSharp.Studio.Tests;

/// <summary>What a job tells the person waiting on it: stage, steps, elapsed and remaining time. The estimator runs on a
/// fixed clock; the lifecycle tests drive the real worker with a test double, not a model.</summary>
public sealed class StudioJobProgressTests : IDisposable
{
    readonly string root = Path.Combine(Path.GetTempPath(), "studio-progress-" + StudioJson.NewId());
    public void Dispose() { if (Directory.Exists(root)) Directory.Delete(root, true); }

    static readonly DateTimeOffset T0 = new(2026, 10, 17, 12, 0, 0, TimeSpan.Zero);
    static DateTimeOffset At(double seconds) => T0.AddSeconds(seconds);
    // A completed run of the same settings: 15 s before denoising, 100 s of it, 47 s after.
    static StudioJob Previous() => new()
    {
        State = "completed", FinishedAt = At(-10),
        Stages = [new("starting", 5), new("text-encode", 10), new("denoise", 100), new("vae-decode", 40), new("encoding", 5), new("importing", 2)]
    };
    // Running since T0, denoising since 10 s, with `done` of 10 steps finished one every 10 s after the first at 20 s.
    static StudioJobRecord Denoising(int done)
    {
        var record = new StudioJobRecord
        {
            Job = new() { Kind = "generate", State = "running", Stage = "denoise", CreatedAt = At(-3), StartedAt = T0, StageStartedAt = At(10), Step = done, TotalSteps = 10,
                Stages = [new("starting", 4), new("text-encode", 6)] },
            DenoiseStartedAt = At(10)
        };
        if (done > 0) { record.FirstStep = 1; record.FirstStepAt = At(20); record.LastStep = done; record.LastStepAt = At(20 + 10 * (done - 1)); }
        return record;
    }

    [Fact]
    public void DenoisingCountsDownFromTheStepRateAndSaysWhenTheDecodeIsUnmeasured()
    {
        var record = Denoising(3); // steps at 20, 30, 40 s: 10 s each
        var timing = StudioService.Measure(record, At(44), null, null);
        Assert.Equal(44, timing.ElapsedSeconds); Assert.Equal(34, timing.StageElapsedSeconds); Assert.Equal(3, timing.QueuedSeconds);
        Assert.Equal(66, timing.RemainingSeconds!.Value, 3); // 7 steps at 10 s, 4 s into the next
        Assert.Equal("denoise", timing.RemainingScope);
        Assert.Null(timing.PreviousRunSeconds); Assert.Null(timing.JobsAhead);
    }

    [Fact]
    public void ASlowStepNeverCountsBelowTheStepsStillToStart()
    {
        var timing = StudioService.Measure(Denoising(3), At(65), null, null); // 25 s into a 10 s step
        Assert.Equal(60, timing.RemainingSeconds!.Value, 3); // the six steps after it
    }

    [Fact]
    public void TheFirstStepAloneIsTimedFromTheStartOfDenoising()
    {
        var timing = StudioService.Measure(Denoising(1), At(25), null, null); // one 10 s step, from 10 s to 20 s
        Assert.Equal(85, timing.RemainingSeconds!.Value, 3);
    }

    [Fact]
    public void BeforeTheFirstStepOnlyAPreviousRunGivesAnEstimate()
    {
        Assert.Null(StudioService.Measure(Denoising(0), At(12), null, null).RemainingSeconds);
        var timing = StudioService.Measure(Denoising(0), At(12), Previous(), null); // 10 s per step last time, 2 s into the first
        Assert.Equal(98 + 47, timing.RemainingSeconds!.Value, 3); Assert.Equal("job", timing.RemainingScope);
    }

    [Fact]
    public void APreviousRunOfTheSameSettingsAddsItsDecodeToTheDenoisingEstimate()
    {
        var timing = StudioService.Measure(Denoising(3), At(44), Previous(), null);
        Assert.Equal(66 + 47, timing.RemainingSeconds!.Value, 3); Assert.Equal("job", timing.RemainingScope);
        Assert.Equal(162, timing.PreviousRunSeconds!.Value, 3);
    }

    [Fact]
    public void BeforeDenoisingTheEstimateIsThePreviousRunLessTheTimeAlreadySpent()
    {
        var record = new StudioJobRecord { Job = new() { Kind = "generate", State = "running", Stage = "text-encode", CreatedAt = T0, StartedAt = T0, StageStartedAt = At(5), TotalSteps = 10, Stages = [new("starting", 5)] } };
        var timing = StudioService.Measure(record, At(8), Previous(), null);
        Assert.Equal(7 + 100 + 47, timing.RemainingSeconds!.Value, 3);
        Assert.Null(StudioService.Measure(record, At(8), null, null).RemainingSeconds);
    }

    [Fact]
    public void AfterDenoisingThePreviousDecodeCountsDownAndStopsWhenThisRunOutlastsIt()
    {
        var record = new StudioJobRecord { Job = new() { Kind = "generate", State = "running", Stage = "vae-decode", CreatedAt = T0, StartedAt = T0, StageStartedAt = At(120), Step = 10, TotalSteps = 10,
            Stages = [new("starting", 5), new("text-encode", 10), new("denoise", 105)] } };
        var timing = StudioService.Measure(record, At(150), Previous(), null);
        Assert.Equal(17, timing.RemainingSeconds!.Value, 3); Assert.Equal("job", timing.RemainingScope);
        Assert.Null(StudioService.Measure(record, At(170), Previous(), null).RemainingSeconds);
        Assert.Null(StudioService.Measure(record, At(150), null, null).RemainingSeconds);
    }

    [Fact]
    public void QueuedEndedAndCancellingJobsReportTimesButNoEstimate()
    {
        var queued = new StudioJobRecord { Job = new() { Kind = "generate", State = "queued", CreatedAt = T0 } };
        var timing = StudioService.Measure(queued, At(30), Previous(), 2);
        Assert.Equal(30, timing.QueuedSeconds); Assert.Equal(2, timing.JobsAhead); Assert.Null(timing.ElapsedSeconds); Assert.Null(timing.RemainingSeconds);
        Assert.Equal(162, timing.PreviousRunSeconds!.Value, 3);

        var ended = new StudioJobRecord { Job = new() { Kind = "generate", State = "failed", CreatedAt = T0, StartedAt = At(4), FinishedAt = At(64) } };
        timing = StudioService.Measure(ended, At(500), null, null);
        Assert.Equal(60, timing.ElapsedSeconds); Assert.Equal(4, timing.QueuedSeconds); Assert.Null(timing.StageElapsedSeconds); Assert.Null(timing.RemainingSeconds);

        var cancelling = Denoising(3); cancelling.Job.State = "cancelling"; cancelling.Job.Stage = "cancelling"; cancelling.Job.StageStartedAt = At(50);
        timing = StudioService.Measure(cancelling, At(52), Previous(), null);
        Assert.Equal(2, timing.StageElapsedSeconds); Assert.Null(timing.RemainingSeconds);
    }

    [Fact]
    public void ARenderIsEstimatedFromTheOutputPositionFfmpegHasReached()
    {
        var record = new StudioJobRecord { Job = new() { Kind = "render", State = "running", Stage = "rendering", Progress = .5, CreatedAt = T0, StartedAt = T0, StageStartedAt = At(2) } };
        var timing = StudioService.Measure(record, At(12), null, null);
        Assert.Equal(10, timing.RemainingSeconds!.Value, 3); Assert.Equal("job", timing.RemainingScope);
        record.Job.Progress = .05; Assert.Null(StudioService.Measure(record, At(12), null, null).RemainingSeconds);
    }

    /// <summary>Reports a text encode, then denoising steps with details, and parks after the second step until released.</summary>
    sealed class SteppedDouble(string source) : IStudioGenerationAdapter
    {
        public TaskCompletionSource Parked = new(TaskCreationOptions.RunContinuationsAsynchronously), Release = new(TaskCreationOptions.RunContinuationsAsynchronously);
        public Task<IReadOnlyList<StudioModelCapability>> GetCapabilitiesAsync(CancellationToken ct) => Task.FromResult<IReadOnlyList<StudioModelCapability>>(
            [new() { ModelId = "stepped-double", Available = true, TextToVideo = true, AudioOutput = true, DefaultFrameCount = 49, DefaultSteps = 4 }]);
        public Task ValidateAsync(StudioGenerationRequest r, IReadOnlyDictionary<string, string> assets, CancellationToken ct) => Task.CompletedTask;
        public async Task<StudioGenerationResult> GenerateAsync(StudioGenerationRequest r, string dir, IReadOnlyDictionary<string, string> assets, IProgress<StudioProgress> progress, CancellationToken ct)
        {
            async Task Report(string stage, int step, string? detail) { progress.Report(new(stage, .1 + .8 * step / 4, detail) { Step = step, TotalSteps = 4 }); await Task.Delay(30, ct); }
            await Report("text-encode", 0, null);
            await Report("denoise", 0, "first pass");
            await Report("denoise", 1, "step 1/4");
            await Report("denoise", 2, "step 2/4");
            Parked.TrySetResult(); await Release.Task.WaitAsync(ct);
            await Report("denoise", 3, "step 3/4");
            await Report("denoise", 4, "step 4/4");
            await Report("vae-decode", 4, "tile 1/2");
            var output = Path.Combine(dir, "generated.mp4"); File.Copy(source, output);
            return new(output, "TEST DOUBLE; no model inference", new());
        }
    }
    async Task<string> FixtureAsync()
    {
        var video = Path.Combine(root, "fixture.mp4");
        using var p = new Process { StartInfo = new("ffmpeg") { UseShellExecute = false, RedirectStandardError = true } };
        foreach (var a in new[] { "-v", "error", "-f", "lavfi", "-i", "color=c=blue:s=64x64:r=24:d=3", "-frames:v", "49", "-c:v", "mpeg4", video }) p.StartInfo.ArgumentList.Add(a);
        p.Start(); string error = await p.StandardError.ReadToEndAsync(); await p.WaitForExitAsync(); Assert.True(p.ExitCode == 0, error);
        return video;
    }
    static async Task<StudioJob> Until(StudioService service, string projectId, string jobId, Func<StudioJob, bool> done)
    {
        using var deadline = new CancellationTokenSource(TimeSpan.FromSeconds(20));
        while (true)
        {
            var job = (await service.ListJobsAsync(projectId, deadline.Token)).Single(j => j.Id == jobId);
            if (done(job)) return job;
            await Task.Delay(20, deadline.Token);
        }
    }
    static async Task RequireMediaAsync(StudioService service)
    {
        var capability = await service.Media.GetCapabilitiesAsync();
        Skip.IfNot(capability.Available && capability.ProbeAvailable, "The generation lifecycle needs installed ffprobe and FFmpeg H.264/AAC: " + capability.Reason);
    }

    [SkippableFact]
    public async Task AGenerationReportsItsStageStepsAndTimesAndKeepsAStageBreakdown()
    {
        Directory.CreateDirectory(root);
        var adapter = new SteppedDouble(await FixtureAsync());
        await using var service = new StudioService(root, adapter); await RequireMediaAsync(service);
        var p = await service.CreateProjectAsync("Progress", 64, 64);
        var request = new StudioGenerationRequest { ProjectId = p.Id, ModelId = "stepped-double", Prompt = "A slow pan across a quiet harbour at dawn.", Width = 64, Height = 64, Seed = 7, GenerateAudio = true };

        var job = await service.StartGenerationAsync(StudioJson.Clone(request));
        Assert.Equal("t2v", job.Summary!.Mode); Assert.Equal(request.Prompt, job.Summary.Prompt); Assert.Equal(4, job.Summary.Steps); Assert.Equal(7, job.Summary.Seed); Assert.True(job.Summary.GenerateAudio);
        await adapter.Parked.Task.WaitAsync(TimeSpan.FromSeconds(10));
        var running = await Until(service, p.Id, job.Id, j => j.Step == 2);
        Assert.Equal(("running", "denoise", "step 2/4", 4), (running.State, running.Stage, running.Detail, running.TotalSteps));
        Assert.NotNull(running.StartedAt); Assert.NotNull(running.StageStartedAt); Assert.Null(running.FinishedAt);
        Assert.Equal(new[] { "starting", "text-encode" }, running.Stages.Select(s => s.Stage));
        Assert.True(running.Timing!.ElapsedSeconds > 0); Assert.True(running.Timing.StageElapsedSeconds > 0);
        // Nothing has measured the decode yet, so the estimate says it covers the denoising alone.
        Assert.True(running.Timing.RemainingSeconds > 0); Assert.Equal("denoise", running.Timing.RemainingScope); Assert.Null(running.Timing.PreviousRunSeconds);

        adapter.Release.TrySetResult();
        var completed = await Until(service, p.Id, job.Id, j => j.State is "completed" or "failed");
        Assert.Equal("completed", completed.State); Assert.Null(completed.Error);
        Assert.Equal(new[] { "starting", "text-encode", "denoise", "vae-decode", "importing" }, completed.Stages.Select(s => s.Stage));
        Assert.Equal(4, completed.Step); Assert.Null(completed.StageStartedAt); Assert.Null(completed.Detail); Assert.NotNull(completed.FinishedAt);
        Assert.Null(completed.Timing!.RemainingSeconds); Assert.Equal((completed.FinishedAt - completed.StartedAt)!.Value.TotalSeconds, completed.Timing.ElapsedSeconds!.Value, 3);

        // The next run of the same settings estimates through the decode, from the run that finished.
        adapter.Parked = new(TaskCreationOptions.RunContinuationsAsynchronously); adapter.Release = new(TaskCreationOptions.RunContinuationsAsynchronously);
        var again = StudioJson.Clone(request); again.RequestId = StudioJson.NewId();
        var second = await service.StartGenerationAsync(again);
        await adapter.Parked.Task.WaitAsync(TimeSpan.FromSeconds(10));
        var estimated = await Until(service, p.Id, second.Id, j => j.Step == 2);
        Assert.Equal("job", estimated.Timing!.RemainingScope); Assert.Equal(completed.Stages.Sum(s => s.Seconds), estimated.Timing.PreviousRunSeconds!.Value, 3);
        await service.CancelJobAsync(second.Id);
        var cancelled = await Until(service, p.Id, second.Id, j => j.State == "cancelled");
        Assert.Equal(new[] { "starting", "text-encode", "denoise", "cancelling" }, cancelled.Stages.Select(s => s.Stage));
        Assert.Equal(2, cancelled.Step);
    }

    [SkippableFact]
    public async Task ARestartRecordsTheStageAnInterruptedJobStoppedIn()
    {
        Directory.CreateDirectory(root);
        string projectId, jobId;
        var adapter = new SteppedDouble(await FixtureAsync());
        await using (var service = new StudioService(root, adapter))
        {
            await RequireMediaAsync(service); projectId = (await service.CreateProjectAsync()).Id;
            jobId = (await service.StartGenerationAsync(new() { ProjectId = projectId, ModelId = "stepped-double", Prompt = "Rain on a window.", Width = 64, Height = 64 })).Id;
            await adapter.Parked.Task.WaitAsync(TimeSpan.FromSeconds(10));
            await Until(service, projectId, jobId, j => j.Step == 2);
        }
        // Emulate an abrupt process loss mid-denoise: the durable record still says running.
        var path = Path.Combine(root, "jobs", jobId, "job.json"); var json = JsonNode.Parse(await File.ReadAllTextAsync(path))!;
        var saved = json["job"]!;
        var stopped = DateTimeOffset.Parse(saved["finishedAt"]!.GetValue<string>());
        var stages = saved["stages"]!.AsArray(); stages.RemoveAt(stages.Count - 1); // undo the shutdown's own close
        saved["state"] = "running"; saved["stage"] = "denoise"; saved["stageStartedAt"] = stopped.AddSeconds(-5).ToString("O"); saved["finishedAt"] = null; saved["updatedAt"] = stopped.ToString("O");
        await File.WriteAllTextAsync(path, json.ToJsonString());

        await using var reopened = new StudioService(root, new SteppedDouble(""));
        var job = Assert.Single(await reopened.ListJobsAsync(projectId));
        Assert.Equal("interrupted", job.State); Assert.Equal(stopped, job.FinishedAt);
        Assert.Equal("denoise", job.Stages[^1].Stage); Assert.Equal(5, job.Stages[^1].Seconds, 3);
        Assert.Equal(2, job.Step); Assert.Null(job.Timing!.RemainingSeconds);
    }
}
