using System.Text.Json.Nodes;
using TensorSharp.Studio;

namespace TensorSharp.Studio.Tests;

public sealed class StudioJobTests : IDisposable
{
    readonly string root = Path.Combine(Path.GetTempPath(), "studio-jobs-" + StudioJson.NewId());
    public void Dispose() { if (Directory.Exists(root)) Directory.Delete(root, true); }
    static StudioGenerationRequest Request(string id) => new() { ProjectId = id, ModelId = "fixture-test-double", Prompt = "A test", Width = 64, Height = 64, FrameCount = 17 };
    static async Task<StudioJob> Wait(StudioService service, string projectId, string jobId, params string[] states)
    {
        using var deadline = new CancellationTokenSource(TimeSpan.FromSeconds(15));
        while (true)
        {
            var job = (await service.ListJobsAsync(projectId, deadline.Token)).Single(j => j.Id == jobId);
            if (states.Contains(job.State)) return job;
            await Task.Delay(20, deadline.Token);
        }
    }
    sealed class BlockingAdapter : IStudioGenerationAdapter
    {
        public int Calls; public int ValidationCalls; public bool Active; public StudioGenerationRequest? Validated, Received, Generated;
        public Action<StudioGenerationRequest>? OnValidate;
        public StudioModelCapability Capability = new() { ModelId = "fixture-test-double", Family = "fixture-family", Available = true, TextToVideo = true, MaxDimension = 1024 }; public TaskCompletionSource Started = new(TaskCreationOptions.RunContinuationsAsynchronously);
        public Task<IReadOnlyList<StudioModelCapability>> GetCapabilitiesAsync(CancellationToken ct) => Task.FromResult<IReadOnlyList<StudioModelCapability>>([Capability]);
        public Task ValidateAsync(StudioGenerationRequest r, IReadOnlyDictionary<string, string> assets, CancellationToken ct) { ValidationCalls++; Received = StudioJson.Clone(r); OnValidate?.Invoke(r); Validated = StudioJson.Clone(r); return Task.CompletedTask; }
        public async Task<StudioGenerationResult> GenerateAsync(StudioGenerationRequest r, string directory, IReadOnlyDictionary<string, string> assets, IProgress<StudioProgress> progress, CancellationToken ct)
        {
            Calls++; Generated = StudioJson.Clone(r); Active = true; Started.TrySetResult(); progress.Report(new("fixture-blocked", .25));
            try { await Task.Delay(Timeout.Infinite, ct); throw new InvalidOperationException(); }
            finally { await Task.Delay(40); Active = false; }
        }
    }
    static async Task RequireMediaAsync(StudioService service)
    {
        var capability = await service.Media.GetCapabilitiesAsync();
        Skip.IfNot(capability.Available && capability.ProbeAvailable, "Generation lifecycle admission requires installed ffprobe and FFmpeg H.264/AAC: " + capability.Reason);
    }
    [Theory]
    [InlineData("mobile")]
    [InlineData("missing-probe")]
    [InlineData("missing-encoder")]
    public async Task GenerationIsUnavailableBeforeInferenceWhenOutputCannotBeIngested(string mode)
    {
        var options = new StudioOptions();
        if (mode == "mobile") options.EnableExternalProcesses = false;
        if (mode == "missing-probe") options.FfprobePath = Path.Combine(root, "missing-ffprobe");
        if (mode == "missing-encoder") options.FfmpegPath = Path.Combine(root, "missing-ffmpeg");
        var adapter = new BlockingAdapter(); await using var service = new StudioService(root, adapter, options);
        var p = await service.CreateProjectAsync();
        var capabilities = System.Text.Json.JsonSerializer.SerializeToElement(await service.GetCapabilitiesAsync(), StudioJson.Options);
        Assert.False(capabilities.GetProperty("generationAvailable").GetBoolean());
        var model = Assert.Single(capabilities.GetProperty("models").EnumerateArray());
        Assert.False(model.GetProperty("available").GetBoolean());
        Assert.Equal("fixture-family", model.GetProperty("family").GetString());
        Assert.True(model.GetProperty("textToVideo").GetBoolean());
        Assert.Contains("before native inference can start", model.GetProperty("unavailableReason").GetString());
        Assert.True(adapter.Capability.Available); // The provider's shared metadata is not mutated.
        var error = await Assert.ThrowsAsync<StudioException>(() => service.StartGenerationAsync(Request(p.Id)));
        Assert.Equal("generation_output_unavailable", error.Code);
        Assert.Equal(0, adapter.ValidationCalls); Assert.Equal(0, adapter.Calls);
        Assert.Empty(await service.ListJobsAsync(p.Id)); Assert.Empty((await service.GetProjectAsync(p.Id)).Assets);
    }
    [SkippableFact] public async Task CancellationWaitsForWorkerCleanupAndRequestIdIsIdempotent()
    {
        var adapter = new BlockingAdapter(); await using var service = new StudioService(root, adapter);
        await RequireMediaAsync(service); var p = await service.CreateProjectAsync(); var request = Request(p.Id);
        var job = await service.StartGenerationAsync(request); await adapter.Started.Task.WaitAsync(TimeSpan.FromSeconds(5));
        Assert.Equal(job.Id, (await service.StartGenerationAsync(request)).Id); Assert.Equal(1, adapter.Calls);
        var cancelling = await service.CancelJobAsync(job.Id); Assert.Equal("cancelling", cancelling.State); Assert.True(adapter.Active);
        var terminal = await Wait(service, p.Id, job.Id, "cancelled"); Assert.False(adapter.Active); Assert.Null(terminal.OutputAssetId);
        Assert.Empty((await service.GetProjectAsync(p.Id)).Assets);
    }
    [SkippableTheory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task ModelDefaultsAreResolvedBeforeAdmissionWithoutOverridingExplicitChoices(bool explicitChoices)
    {
        var adapter = new BlockingAdapter();
        adapter.Capability.DefaultWidth = 1024; adapter.Capability.DefaultHeight = 768;
        adapter.Capability.DefaultFps = 30; adapter.Capability.DefaultFrameCount = 145;
        adapter.Capability.AudioOutput = true; adapter.Capability.DefaultGenerateAudio = true;
        await using var service = new StudioService(root, adapter); await RequireMediaAsync(service);
        var project = await service.CreateProjectAsync();
        var request = new StudioGenerationRequest { ProjectId = project.Id, ModelId = adapter.Capability.ModelId, Prompt = "Keep detailed faces and hands." };
        if (explicitChoices) { request.Width = request.Height = 512; request.Fps = 24; request.GenerateAudio = false; }
        var job = await service.StartGenerationAsync(request);
        var effective = Assert.IsType<StudioGenerationRequest>(adapter.Validated);
        Assert.Equal(explicitChoices ? 512 : 1024, effective.Width);
        Assert.Equal(explicitChoices ? 512 : 768, effective.Height);
        Assert.Equal(explicitChoices ? 24 : 30, effective.Fps);
        Assert.Equal(!explicitChoices, effective.GenerateAudio);
        Assert.Equal(145, effective.FrameCount);
        Assert.Equal(adapter.Capability.DefaultSteps, effective.Steps);
        Assert.Equal(adapter.Capability.DefaultCfgScale, effective.CfgScale);
        Assert.Equal(job.Id, (await service.StartGenerationAsync(request)).Id);
        if (!explicitChoices) { Assert.Equal(0, request.Width); Assert.Null(request.GenerateAudio); }
        await service.CancelJobAsync(job.Id);
    }

    static async Task<string> ImportStillAsync(StudioService service, string projectId, string root)
    {
        string png = Path.Combine(root, "still-" + StudioJson.NewId() + ".png");
        using (var process = new System.Diagnostics.Process())
        {
            process.StartInfo = new("ffmpeg") { UseShellExecute = false, RedirectStandardError = true };
            foreach (var arg in new[] { "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=96x64:d=1", "-frames:v", "1", png }) process.StartInfo.ArgumentList.Add(arg);
            process.Start(); string error = await process.StandardError.ReadToEndAsync(); await process.WaitForExitAsync();
            Assert.True(process.ExitCode == 0, error);
        }
        await using var input = File.OpenRead(png);
        return (await service.ImportAssetAsync(projectId, input, "first.png")).Assets.Last().Id;
    }
    [SkippableTheory]
    [InlineData("i2v-filled")]
    [InlineData("t2v")]
    [InlineData("i2v-unfilled")]
    public async Task FirstFrameCanvasModesLeaveAZeroSizeToTheAdapterAndQueueTheSizeItChose(string scenario)
    {
        // In a mode that keeps the first frame's framing, 0x0 asks the model to size the canvas from that image.
        // Only its adapter can read the image, so admission must not substitute the default size first, and the
        // queued job must carry the concrete size the adapter chose. Any other mode still gets the default.
        var adapter = new BlockingAdapter();
        adapter.Capability.ImageConditioning = true; adapter.Capability.SpatialMultiple = 32;
        adapter.Capability.DefaultWidth = 640; adapter.Capability.DefaultHeight = 384;
        adapter.Capability.ImageCanvasModes = [StudioGenerationModes.ImageToVideo];
        if (scenario != "i2v-unfilled") adapter.OnValidate = r => { if (r.Width == 0 && r.Height == 0) { r.Width = 96; r.Height = 64; } };
        await using var service = new StudioService(root, adapter); await RequireMediaAsync(service);
        var project = await service.CreateProjectAsync();
        var request = new StudioGenerationRequest { ProjectId = project.Id, ModelId = adapter.Capability.ModelId, Prompt = "The picture slowly comes alive.", FrameCount = 17 };
        if (scenario != "t2v") { request.Mode = StudioGenerationModes.ImageToVideo; request.ConditioningAssetId = await ImportStillAsync(service, project.Id, root); }
        if (scenario == "i2v-unfilled")
        {
            Assert.Equal("invalid_resolution", (await Assert.ThrowsAsync<StudioException>(() => service.StartGenerationAsync(request))).Code);
            Assert.Empty(await service.ListJobsAsync(project.Id)); Assert.Equal(0, adapter.Calls);
            return;
        }
        var job = await service.StartGenerationAsync(request);
        var received = Assert.IsType<StudioGenerationRequest>(adapter.Received);
        Assert.Equal(scenario == "t2v" ? (640, 384) : (0, 0), (received.Width, received.Height));
        await adapter.Started.Task.WaitAsync(TimeSpan.FromSeconds(5));
        var generated = Assert.IsType<StudioGenerationRequest>(adapter.Generated);
        Assert.Equal(scenario == "t2v" ? (640, 384) : (96, 64), (generated.Width, generated.Height));
        Assert.Equal(job.Id, (await service.StartGenerationAsync(request)).Id); // the 0x0 request is still the same request
        await service.CancelJobAsync(job.Id);
    }

    [Fact]
    public async Task GenerationDraftCanDeferModelDefaultsUntilAdmission()
    {
        await using var service = new StudioService(root);
        var project = await service.CreateProjectAsync();
        var draft = await service.CreateGenerationDraftAsync(project.Id, project.Revision,
            new() { Prompt = "A person smiles gently." }, "Use the loaded model's quality defaults.");
        Assert.Equal(0, draft.Request.Width); Assert.Equal(0, draft.Request.Height); Assert.Equal(0, draft.Request.Fps);
        Assert.Null(draft.Request.GenerateAudio);
        var saved = Assert.Single(await service.ListGenerationDraftsAsync(project.Id));
        Assert.Equal(draft.Request.Prompt, saved.Request.Prompt);
        Assert.Equal(0, saved.Request.Width); Assert.Equal(0, saved.Request.Height); Assert.Equal(0, saved.Request.Fps);
        Assert.Null(saved.Request.GenerateAudio);
    }
    [SkippableFact] public async Task QueueIsBoundedAndCancellingQueuedJobsNeverStartsInference()
    {
        var adapter = new BlockingAdapter(); await using var service = new StudioService(root, adapter, new() { MaxQueuedJobs = 2 });
        await RequireMediaAsync(service); var p = await service.CreateProjectAsync(); var first = await service.StartGenerationAsync(Request(p.Id)); await adapter.Started.Task;
        var second = await service.StartGenerationAsync(Request(p.Id));
        Assert.Equal("queue_full", (await Assert.ThrowsAsync<StudioException>(() => service.StartGenerationAsync(Request(p.Id)))).Code);
        Assert.Equal("cancelled", (await service.CancelJobAsync(second.Id)).State);
        await service.CancelJobAsync(first.Id); await Wait(service, p.Id, first.Id, "cancelled"); Assert.Equal(1, adapter.Calls);
    }
    [SkippableFact] public async Task RestartMarksRunningWorkInterruptedWithoutPretendingToResumeDenoising()
    {
        string projectId, jobId;
        var adapter = new BlockingAdapter();
        await using (var service = new StudioService(root, adapter))
        {
            await RequireMediaAsync(service); projectId = (await service.CreateProjectAsync()).Id; jobId = (await service.StartGenerationAsync(Request(projectId))).Id; await adapter.Started.Task;
        }
        Assert.False(adapter.Active);
        // Emulate an abrupt process loss leaving the durable record in running state.
        var path = Path.Combine(root, "jobs", jobId, "job.json"); var json = JsonNode.Parse(await File.ReadAllTextAsync(path))!;
        json["job"]!["state"] = "running"; await File.WriteAllTextAsync(path, json.ToJsonString());
        await using var reopened = new StudioService(root, new BlockingAdapter());
        var job = Assert.Single(await reopened.ListJobsAsync(projectId)); Assert.Equal("interrupted", job.State); Assert.Contains("mid-denoising", job.Error);
    }
    [SkippableTheory]
    [InlineData("resolution")]
    [InlineData("frames")]
    [InlineData("image")]
    [InlineData("guidance")]
    [InlineData("audio")]
    public async Task UnsupportedGenerationOptionsFailBeforeExecution(string field)
    {
        var adapter = new BlockingAdapter(); await using var service = new StudioService(root, adapter); await RequireMediaAsync(service); var p = await service.CreateProjectAsync(); var r = Request(p.Id);
        switch (field)
        {
            case "resolution": r.Width = 65; break;
            case "frames": r.FrameCount = 18; break;
            case "image": r.ConditioningAssetId = StudioJson.NewId(); break;
            case "guidance": r.TemporalConditions.Add(new(StudioJson.NewId(), 2)); break;
            case "audio": r.GenerateAudio = true; break;
        }
        // An image or temporal guide implies i2v/keyframes, which a text-only checkpoint does not run.
        var expectedCode = field switch { "resolution" => "invalid_resolution", "frames" => "invalid_frame_count", "image" or "guidance" => "unsupported_mode", _ => "unsupported_conditioning" };
        Assert.Equal(expectedCode, (await Assert.ThrowsAsync<StudioException>(() => service.StartGenerationAsync(r))).Code); Assert.Equal(0, adapter.Calls); Assert.Empty(await service.ListJobsAsync(p.Id));
    }
    [SkippableFact] public async Task JobRequestIdentifierCannotBeReusedForDifferentParameters()
    {
        var adapter = new BlockingAdapter(); await using var service = new StudioService(root, adapter); await RequireMediaAsync(service); var p = await service.CreateProjectAsync(); var request = Request(p.Id);
        await service.StartGenerationAsync(request); request.Prompt = "A different request";
        Assert.Equal("idempotency_conflict", (await Assert.ThrowsAsync<StudioException>(() => service.StartGenerationAsync(request))).Code);
    }
}
