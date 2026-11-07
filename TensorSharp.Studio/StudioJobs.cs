// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Text;
using System.Text.Json.Serialization;
using System.Threading.Channels;

namespace TensorSharp.Studio;

internal sealed class StudioJobRecord
{
    public StudioJob Job { get; set; } = new();
    public StudioProject? Snapshot { get; set; }
    public StudioGenerationRequest? Generation { get; set; }
    public string Fingerprint { get; set; } = "";
    /// <summary>The take's last frame, saved losslessly in the job directory before its video was encoded; null when
    /// the adapter kept none. <see cref="StudioService.ExtractFrameAsync"/> starts the next segment of a chain on it.</summary>
    public string? LastFrameFileName { get; set; }
    /// <summary>The W3C trace context (<c>traceparent</c>, <c>tracestate</c>) of the request that queued the job, when
    /// Studio traces were being recorded. The job runs in a trace of its own that links back to it, also after a
    /// restart. Left out of the file when null, and an older host skips them.</summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? TraceParent { get; set; }
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? TraceState { get; set; }
    // Live step timing behind the remaining-time estimate. Fields are not serialized: a restart interrupts running work.
    internal int FirstStep, LastStep;
    internal DateTimeOffset FirstStepAt, LastStepAt;
    internal DateTimeOffset? DenoiseStartedAt;
}

public sealed partial class StudioService
{
    readonly object jobSync = new();
    readonly Dictionary<string, StudioJobRecord> jobs = [];
    readonly Dictionary<string, CancellationTokenSource> jobCancellations = [];
    readonly CancellationTokenSource shutdown = new();
    Channel<string> queue = null!;
    Task worker = null!;
    bool stopping;
    string JobDirectory(string jobId) => StudioPaths.SafeChild(RootPath, "jobs", StudioPaths.Id(jobId));
    void SaveJob(StudioJobRecord record) { record.Job.UpdatedAt = DateTimeOffset.UtcNow; StudioStorage.Save(StudioPaths.SafeChild(JobDirectory(record.Job.Id), "job.json"), record); }

    void InitializeJobs()
    {
        queue = Channel.CreateBounded<string>(new BoundedChannelOptions(Options.MaxQueuedJobs) { SingleReader = true, FullMode = BoundedChannelFullMode.Wait });
        int cleanupBudget = StartupCleanupBudget;
        CleanupPortableScratch(ref cleanupBudget);
        foreach (var directory in Directory.EnumerateDirectories(StudioPaths.SafeChild(RootPath, "jobs")).Take(10000))
        {
            var id = Path.GetFileName(directory);
            if (!IsManagedId(id)) continue; // Preserve unrelated directories in the configured root.
            var path = StudioPaths.SafeChild(JobDirectory(id), "job.json");
            CleanupAtomicTemporaries(directory, "job.json", ref cleanupBudget);
            if (!File.Exists(path))
            {
                CleanupScratchTree(StudioPaths.SafeChild(directory, "work"), ref cleanupBudget);
                continue;
            }
            var record = StudioStorage.Read<StudioJobRecord>(path);
            if (record.Job.Id != id) throw new StudioException("invalid_job", "Job identity does not match its directory.");
            if (record.Job.State is "queued" or "running" or "cancelling")
            {
                var adopted = record.Job.Kind == "generate" ? LoadDocument(record.Job.ProjectId).Project.Assets
                    .FirstOrDefault(a => a.Generation?.Request.RequestId == record.Job.RequestId && File.Exists(StudioPaths.SafeChild(ProjectDirectory(record.Job.ProjectId), "assets", a.FileName))) : null;
                // The last save is the latest moment the job is known to have been working.
                if (adopted != null)
                {
                    Finish(record.Job, "completed", record.Job.UpdatedAt); record.Job.Progress = 1;
                    record.Job.OutputAssetId = adopted.Id; record.Job.Error = null;
                    // The last frame is moved into the job directory before the take is imported.
                    if (File.Exists(StudioPaths.SafeChild(directory, LastFrameFile))) record.LastFrameFileName = LastFrameFile;
                }
                else
                {
                    Finish(record.Job, "interrupted", record.Job.UpdatedAt);
                    record.Job.Error = "The host stopped before this job finished. Retry with a new request identifier. Inference cannot resume mid-denoising.";
                }
                SaveJob(record);
                // Any span the job had was lost with the process that queued or ran it, so this one records how it ended.
                StudioTelemetry.TraceJobNotRun(record);
            }
            jobs.Add(id, record);
            // Partial work is never presented as a completed artifact. A restart can safely discard only job-owned scratch.
            var scratch = StudioPaths.SafeChild(directory, "work");
            if (Directory.Exists(scratch) && record.Job.State != "running") CleanupScratchTree(scratch, ref cleanupBudget);
        }
        StudioTelemetry.Track(this);
        worker = Task.Run(ProcessJobsAsync);
    }
    public Task<IReadOnlyList<StudioJob>> ListJobsAsync(string projectId, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested(); LoadDocument(projectId);
        lock (jobSync)
        {
            var now = DateTimeOffset.UtcNow;
            return Task.FromResult<IReadOnlyList<StudioJob>>(jobs.Values.Where(r => r.Job.ProjectId == projectId).OrderByDescending(r => r.Job.CreatedAt).Select(r => Present(r, now)).ToArray());
        }
    }
    public Task<StudioJob> CancelJobAsync(string jobId, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested(); StudioPaths.Id(jobId);
        lock (jobSync)
        {
            if (!jobs.TryGetValue(jobId, out var record)) throw new StudioException("job_not_found", "Job does not exist.");
            var now = DateTimeOffset.UtcNow;
            if (record.Job.State == "queued") { Finish(record.Job, "cancelled", now); SaveJob(record); StudioTelemetry.TraceJobNotRun(record); }
            else if (record.Job.State == "running")
            {
                record.Job.State = "cancelling"; EnterStage(record.Job, "cancelling", now); SaveJob(record);
                if (jobCancellations.TryGetValue(jobId, out var source)) source.Cancel();
            }
            return Task.FromResult(Present(record, now));
        }
    }
    public string GetJobOutputPath(string projectId, string jobId)
    {
        LoadDocument(projectId); StudioPaths.Id(jobId);
        lock (jobSync)
        {
            if (!jobs.TryGetValue(jobId, out var record) || record.Job.ProjectId != projectId) throw new StudioException("job_not_found", "Job does not belong to this project.");
            if (record.Job.State != "completed" || record.Job.OutputFileName == null) throw new StudioException("output_not_ready", "The job does not have a completed render output.");
            var path = StudioPaths.SafeChild(JobDirectory(jobId), record.Job.OutputFileName);
            if (!File.Exists(path)) throw new StudioException("missing_output", "Rendered output is missing; render the project again.");
            return path;
        }
    }
    public async Task<StudioJob> StartRenderAsync(string projectId, long expectedRevision, string requestId, CancellationToken cancellationToken = default)
    {
        var project = await GetProjectAsync(projectId, cancellationToken); CheckRevision(project, expectedRevision);
        if (project.DurationFrames <= 0) throw new StudioException("empty_timeline", "Add clips to the timeline before rendering.");
        foreach (var c in project.Captions) StudioCaptionBitmap.Validate(c.Text);
        var capability = await Media.GetCapabilitiesAsync(cancellationToken);
        if (!capability.Available) throw new StudioException("render_unavailable", capability.Reason!);
        foreach (var clip in project.Clips) GetSnapshotAssetPath(project, clip.AssetId);
        return Enqueue(new() { Job = new() { ProjectId = projectId, RequestId = requestId, Kind = "render", ProjectRevision = project.Revision }, Snapshot = project,
            Fingerprint = Hash(new { kind = "render", projectId, expectedRevision }) });
    }
    public async Task<StudioJob> StartGenerationAsync(StudioGenerationRequest request, CancellationToken cancellationToken = default)
    {
        if (generation == null) throw new StudioException("generation_unavailable", "No native generation adapter is connected to this host.");
        request = StudioJson.Clone(request);
        if (request.TemporalConditions is null || request.ReferenceAssetIds is null || request.ReferenceVideoAssetIds is null ||
            request.TemporalConditions.Any(c => c is null) || request.ReferenceAssetIds.Any(id => id is null) || request.ReferenceVideoAssetIds.Any(id => id is null))
            throw new StudioException("invalid_conditioning", "Conditioning collections must not be null.");
        var project = await GetProjectAsync(request.ProjectId, cancellationToken);
        var (capability, imageCanvas) = await AdmitAsync(request, cancellationToken);
        var fingerprint = Hash(new { kind = "generate", request });
        lock (jobSync)
        {
            // An accepted request owns its captured inputs, even after a library deletion.
            // Reconnecting clients may retrieve that job without re-admitting removed assets.
            var existing = FindExistingJob(request.ProjectId, request.RequestId, fingerprint);
            if (existing != null) return existing;
        }
        var paths = ResolveConditioning(request, project);
        await generation.ValidateAsync(request, paths, cancellationToken);
        // The adapter has replaced a first-frame 0x0 with the concrete canvas; the job records that size.
        if (imageCanvas) RequireResolution(request, capability);
        // Bound after the fingerprint, so resubmitting the same unbound request finds this job. The run refuses a model
        // whose recipe has changed since (the chat service re-checks it under the model's GPU lock).
        request.SamplingRecipe ??= capability.SamplingRecipe;
        return Enqueue(new() { Job = new() { ProjectId = project.Id, RequestId = request.RequestId, Kind = "generate", ProjectRevision = project.Revision }, Generation = request, Snapshot = project,
            Fingerprint = fingerprint });
    }
    /// <summary>The checks a request passes before its inputs are read: the host can ingest the output, the model is
    /// loaded and runs the request's mode with its settings. Fills in the model's defaults in place; with width = height = 0
    /// in an image-canvas mode the size is left for the adapter. Returns the capability and whether that is the case.</summary>
    async Task<(StudioModelCapability Capability, bool ImageCanvas)> AdmitAsync(StudioGenerationRequest request, CancellationToken cancellationToken)
    {
        var generationMediaReason = GenerationMediaUnavailableReason(await Media.GetCapabilitiesAsync(cancellationToken));
        if (generationMediaReason != null) throw new StudioException("generation_output_unavailable", generationMediaReason);
        var capabilities = await generation!.GetCapabilitiesAsync(cancellationToken);
        var capability = capabilities.FirstOrDefault(c => c.ModelId == request.ModelId) ?? throw new StudioException("model_not_found", "Selected video model is not available from this host registry.");
        if (!capability.Available) throw new StudioException("model_unavailable", capability.UnavailableReason ?? "Selected video model is not installed.");
        // A shot planned for another sampling recipe carries that recipe's steps: refuse it before defaults or the fingerprint.
        capability.RequireSamplingRecipe(request);
        string mode = StudioGenerationModes.Resolve(request);
        // In a mode that keeps the first frame's framing, 0x0 asks the model to size the canvas from that image.
        // Only its adapter can read the image's upright size, so the size is filled in after it validates.
        bool imageCanvas = request.Width == 0 && request.Height == 0 && capability.ImageCanvasModes.Contains(mode);
        if (!imageCanvas && request.Width == 0) request.Width = capability.DefaultWidth;
        if (!imageCanvas && request.Height == 0) request.Height = capability.DefaultHeight;
        if (request.Fps == 0) request.Fps = capability.DefaultFps;
        request.GenerateAudio ??= capability.DefaultGenerateAudio;
        if (request.FrameCount == 0) request.FrameCount = capability.DefaultFrameCount;
        if (request.Steps == 0) request.Steps = capability.DefaultSteps;
        if (request.CfgScale == 0) request.CfgScale = capability.DefaultCfgScale;
        if (string.IsNullOrWhiteSpace(request.Prompt) || request.Prompt.Length > 16000 || request.NegativePrompt?.Length > 16000)
            throw new StudioException("invalid_prompt", "Generation prompt must contain 1–16000 characters; negative prompt is limited to 16000.");
        if (!imageCanvas) RequireResolution(request, capability);
        if (request.FrameCount < capability.TemporalOffset || request.FrameCount > capability.MaxFrames || (request.FrameCount - capability.TemporalOffset) % capability.TemporalMultiple != 0)
            throw new StudioException("invalid_frame_count", $"This model requires {capability.TemporalMultiple}k+{capability.TemporalOffset} frames up to {capability.MaxFrames}.");
        if (request.Fps is < 1 or > 120 || capability.FixedFps is { } fixedFps && request.Fps != fixedFps)
            throw new StudioException("invalid_fps", capability.FixedFps is { } fixedRate ? $"This model requires {fixedRate} fps." : "Frame rate must be between 1 and 120.");
        if (capability.FixedSteps is { } steps && request.Steps != steps)
            throw new StudioException("invalid_sampling", capability.FixedStepsRefusal());
        if (request.Steps is < 1 or > 200 || !float.IsFinite(request.CfgScale) || request.CfgScale is < 1 or > 30 || capability.FixedCfgScale is { } cfg && Math.Abs(request.CfgScale - cfg) > 0.0001)
            throw new StudioException("invalid_sampling", "Steps or CFG do not match the selected model's supported sampling settings.");
        // First, so a chain segment that keeps a character hears why this family cannot do both, not "load the reference
        // checkpoint", which would refuse the pinned first frame in turn.
        capability.RequireReferencesWithFirstFrame(request, mode);
        if (!capability.Modes.Contains(mode))
            throw new StudioException("unsupported_mode", capability.UnavailableModes.TryGetValue(mode, out var reason) ? reason
                : !capability.TextToVideo && mode == StudioGenerationModes.TextToVideo ? "This checkpoint requires first-frame image conditioning."
                : $"The selected model does not support {mode} generation. Supported: {string.Join(", ", capability.Modes)}.");
        if (mode == StudioGenerationModes.Reference && capability.ReferenceMinFrames is { } minimumFrames && request.FrameCount < minimumFrames)
            throw new StudioException("invalid_frame_count", $"This model's reference workflow requires at least {minimumFrames} frames.");
        if (request.EndConditioningAssetId != null && !capability.LastFrameConditioning)
            throw new StudioException("unsupported_conditioning", capability.UnavailableModes.TryGetValue(StudioGenerationModes.FirstLastFrame, out var lastReason)
                ? lastReason : "The selected model cannot pin a last frame.");
        if (request.ConditioningAssetId != null && !capability.ImageConditioning || request.GenerateAudio == true && !capability.AudioOutput)
            throw new StudioException("unsupported_conditioning", "The selected model does not support a requested image/audio conditioning mode.");
        if (request.TemporalConditions.Count > 0 && !capability.TemporalGuidance ||
            request.ReferenceAssetIds.Count > 0 && !capability.IdentityReferences || request.ReferenceAssetIds.Count > capability.MaxReferences ||
            request.ReferenceVideoAssetIds.Count > 0 && !capability.ReferenceVideos || request.ReferenceVideoAssetIds.Count > capability.MaxReferenceVideos ||
            request.TemporalConditions.Count > 32)
            throw new StudioException("unsupported_conditioning", "Temporal guidance or identity-reference media exceed this model's capabilities.");
        foreach (var guide in request.TemporalConditions)
            if (guide.FrameIndex < -1 || guide.FrameIndex >= request.FrameCount || !float.IsFinite(guide.Strength) || guide.Strength is <= 0 or > 1)
                throw new StudioException("invalid_conditioning", "Temporal guide indices and strengths must fit the output frame range and (0, 1].");
        return (capability, imageCanvas);
    }
    static void RequireResolution(StudioGenerationRequest request, StudioModelCapability capability)
    {
        if (request.Width < 16 || request.Height < 16 || request.Width > capability.MaxDimension || request.Height > capability.MaxDimension ||
            request.Width % capability.SpatialMultiple != 0 || request.Height % capability.SpatialMultiple != 0)
            throw new StudioException("invalid_resolution", $"This model requires dimensions divisible by {capability.SpatialMultiple}, up to {capability.MaxDimension} pixels.");
    }
    Dictionary<string, string> ResolveConditioning(StudioGenerationRequest request, StudioProject project)
    {
        var imageIds = request.ReferenceAssetIds.Concat(request.TemporalConditions.Select(c => c.AssetId))
            .Concat(new[] { request.ConditioningAssetId, request.EndConditioningAssetId }.OfType<string>());
        var ids = imageIds.Select(id => (Id: id, Kind: "image"))
            .Concat(request.ReferenceVideoAssetIds.Select(id => (Id: id, Kind: "video")))
            .GroupBy(item => item.Id, StringComparer.Ordinal).Select(group =>
            {
                if (group.Select(item => item.Kind).Distinct(StringComparer.Ordinal).Count() != 1)
                    throw new StudioException("invalid_conditioning", "A reference asset cannot be used as both an image and a video.");
                return group.First();
            });
        var paths = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var (id, kind) in ids)
        {
            var asset = project.Assets.FirstOrDefault(a => a.Id == id) ?? throw new StudioException("asset_not_found", "A conditioning asset does not belong to the project.");
            if (asset.Kind != kind) throw new StudioException("invalid_conditioning", kind == "video"
                ? "Video reference assets must be full video clips." : "Frame and image-reference assets must be still images.");
            paths.Add(id, GetSnapshotAssetPath(project, id));
        }
        return paths;
    }
    StudioJob Enqueue(StudioJobRecord record)
    {
        StudioPaths.Id(record.Job.RequestId); CheckDisk();
        lock (jobSync)
        {
            if (stopping) throw new StudioException("host_stopping", "Studio is stopping; restart the host before submitting work.");
            var existing = FindExistingJob(record.Job.ProjectId, record.Job.RequestId, record.Fingerprint);
            if (existing != null) return existing;
            if (jobs.Values.Count(r => r.Job.State is "queued" or "running" or "cancelling") >= Options.MaxQueuedJobs)
                throw new StudioException("queue_full", "The local media job queue is full. Wait or cancel queued work before retrying.");
            // The worker runs the job outside this request, so its trace can only link back to it.
            StudioTelemetry.CaptureQueuingContext(record);
            jobs.Add(record.Job.Id, record); SaveJob(record);
            if (!queue.Writer.TryWrite(record.Job.Id)) { jobs.Remove(record.Job.Id); Directory.Delete(JobDirectory(record.Job.Id), true); throw new StudioException("queue_full", "The local job queue is full."); }
            return Present(record, DateTimeOffset.UtcNow);
        }
    }
    // Call under jobSync.
    StudioJob? FindExistingJob(string projectId, string requestId, string fingerprint)
    {
        var existing = jobs.Values.FirstOrDefault(r => r.Job.ProjectId == projectId && r.Job.RequestId == requestId);
        if (existing == null) return null;
        if (existing.Fingerprint != fingerprint) throw new StudioException("idempotency_conflict", "This job request identifier was already used for different parameters.");
        return Present(existing, DateTimeOffset.UtcNow);
    }
    // Call under jobSync. Closes the stage the job is in, recording how long it ran, and opens the next one.
    static void EnterStage(StudioJob job, string stage, DateTimeOffset now)
    {
        if (job.Stage == stage) return;
        if (job.StageStartedAt is { } since) job.Stages.Add(new(job.Stage, Math.Max(0, (now - since).TotalSeconds)));
        job.Stage = stage; job.StageStartedAt = now; job.Detail = null;
    }
    // Call under jobSync. A job that has ended is in no stage. Every way a job ends comes through here.
    static void Finish(StudioJob job, string state, DateTimeOffset now)
    {
        EnterStage(job, state, now);
        job.State = state; job.StageStartedAt = null; job.FinishedAt = now;
        StudioTelemetry.RecordJobDuration(job);
    }
    // Call under jobSync. Returns whether the report opened a new stage.
    static bool Advance(StudioJobRecord record, StudioProgress p, DateTimeOffset now)
    {
        var job = record.Job; bool entered = job.Stage != p.Stage;
        EnterStage(job, p.Stage, now);
        if (p.Stage == "denoise") record.DenoiseStartedAt ??= now;
        // A report without step totals carries no fraction; it must not move the bar back.
        job.Detail = p.Detail; job.Progress = Math.Max(job.Progress, Math.Clamp(p.Fraction, 0, .99));
        if (p.TotalSteps > 0) job.TotalSteps = p.TotalSteps;
        if (p.Step > record.LastStep)
        {
            if (record.FirstStep == 0) { record.FirstStep = p.Step; record.FirstStepAt = now; }
            record.LastStep = job.Step = p.Step; record.LastStepAt = now;
        }
        return entered;
    }
    // Call under jobSync. A copy of the job with its clock-derived view as of now.
    StudioJob Present(StudioJobRecord record, DateTimeOffset now)
    {
        var job = StudioJson.Clone(record.Job);
        job.Summary = Summarize(record.Generation);
        bool active = job.State is "queued" or "running" or "cancelling";
        int? ahead = job.State == "queued" ? jobs.Values.Count(r => r.Job.State is "running" or "cancelling" || r.Job.State == "queued" && r.Job.CreatedAt < job.CreatedAt) : null;
        job.Timing = Measure(record, now, active ? PreviousRun(record)?.Job : null, ahead);
        return job;
    }
    /// <summary>The jobs waiting for the worker and those it is running (a job being cancelled still holds it), for
    /// Studio's gauges. Called on the metrics collection thread.</summary>
    internal (int Queued, int Running) CountActiveJobs()
    {
        lock (jobSync)
        {
            int queued = 0, running = 0;
            foreach (var r in jobs.Values)
                if (r.Job.State == "queued") queued++;
                else if (r.Job.State is "running" or "cancelling") running++;
            return (queued, running);
        }
    }
    // Call under jobSync. The latest completed generation of the same model and settings: the only measure of the
    // stages before and after denoising, which a live step rate cannot predict.
    StudioJobRecord? PreviousRun(StudioJobRecord record)
    {
        var r = record.Generation; if (r == null) return null;
        string mode = r.Mode ?? StudioGenerationModes.Infer(r);
        StudioJobRecord? latest = null;
        foreach (var other in jobs.Values)
        {
            var o = other.Generation;
            if (o == null || other == record || other.Job.State != "completed" || other.Job.FinishedAt == null || other.Job.Stages.Count == 0) continue;
            if (o.ModelId != r.ModelId || o.Width != r.Width || o.Height != r.Height || o.FrameCount != r.FrameCount || o.Steps != r.Steps || o.CfgScale != r.CfgScale ||
                o.GenerateAudio != r.GenerateAudio || o.SamplingRecipe != r.SamplingRecipe || (o.Mode ?? StudioGenerationModes.Infer(o)) != mode) continue;
            if (latest == null || other.Job.FinishedAt > latest.Job.FinishedAt) latest = other;
        }
        return latest;
    }
    static StudioJobSummary? Summarize(StudioGenerationRequest? r)
    {
        if (r == null) return null;
        int cut = r.Prompt.Length <= 240 ? r.Prompt.Length : char.IsHighSurrogate(r.Prompt[239]) ? 239 : 240;
        return new()
        {
            ModelId = r.ModelId, Mode = r.Mode ?? StudioGenerationModes.Infer(r), Prompt = cut < r.Prompt.Length ? r.Prompt[..cut] + "…" : r.Prompt,
            Width = r.Width, Height = r.Height, FrameCount = r.FrameCount, Fps = r.Fps, Steps = r.Steps, Seed = r.Seed, GenerateAudio = r.GenerateAudio,
            FirstFrameAssetId = r.ConditioningAssetId, LastFrameAssetId = r.EndConditioningAssetId,
            GuideCount = r.TemporalConditions.Count, ReferenceCount = r.ReferenceAssetIds.Count,
            ReferenceVideoCount = r.ReferenceVideoAssetIds.Count
        };
    }

    /// <summary>A take's library name: the start of its prompt, numbered among the takes of its shot, so variants are
    /// told apart at a glance instead of all arriving as "generated.mp4". The extension keeps the output's media type.</summary>
    internal static string TakeFileName(StudioGenerationRequest request, StudioProject project, string outputPath)
    {
        // Whole runes only, so a cut never leaves half of a surrogate pair (written as U+FFFD).
        var words = new StringBuilder();
        foreach (Rune r in request.Prompt.EnumerateRunes())
        {
            if (words.Length + r.Utf16SequenceLength > 48) break;
            bool space = Rune.IsWhiteSpace(r) || Rune.IsControl(r) || r.Value is '/' or '\\'
                         || r.IsBmp && Array.IndexOf(Path.GetInvalidFileNameChars(), (char)r.Value) >= 0;
            if (space) { if (words.Length > 0 && words[^1] != ' ') words.Append(' '); }
            else words.Append(r.ToString());
        }
        string title = words.ToString();
        // A prompt longer than the name ends at its last whole word, not mid-word.
        if (title.Length >= 47 && title.LastIndexOf(' ') is int cut and >= 20) title = title[..cut];
        title = title.Trim().TrimEnd('.', ',', ';', ':');
        if (title.Length == 0) title = "Generated take";
        // A segment that continues a video says which part of the chain it is, so the parts are told apart in the library.
        int part = ChainPart(request, project);
        if (part > 1) title += $" · part {part}";
        // Numbered after the highest take of the shot, so deleting one never makes the next repeat a name.
        int take = 1;
        if (request.ShotId != null)
            foreach (var asset in project.Assets.Where(a => a.Generation?.Request.ShotId == request.ShotId))
            {
                var number = System.Text.RegularExpressions.Regex.Match(Path.GetFileNameWithoutExtension(asset.Name), @" · take (\d+)$");
                take = Math.Max(take, (number.Success ? int.Parse(number.Groups[1].Value, System.Globalization.CultureInfo.InvariantCulture) : 1) + 1);
            }
        if (take > 1) title += $" · take {take}";
        return title + Path.GetExtension(outputPath);
    }
    /// <summary>Which segment of a chain a shot is: 1 unless it starts on a still taken from a video, then one more than
    /// that video (an imported video counts as part 1), following each take's first frame back.</summary>
    internal static int ChainPart(StudioGenerationRequest request, StudioProject project)
    {
        int part = 1; var seen = new HashSet<string>(StringComparer.Ordinal);
        for (string? first = request.ConditioningAssetId; first != null && part < 999;)
        {
            var still = project.Assets.FirstOrDefault(a => a.Id == first);
            if (still?.DerivedFrom is not { } source) break;
            part++;
            var video = project.Assets.FirstOrDefault(a => a.Id == source.AssetId);
            if (video?.Generation is not { } generation || !seen.Add(video.Id)) break;
            first = generation.Request.ConditioningAssetId;
        }
        return part;
    }
    /// <summary>Elapsed, queued and remaining time as of <paramref name="now"/>. Denoising is estimated from the steps this
    /// run has finished; the stages around it only from <paramref name="previous"/>, a completed run of the same model and
    /// settings. Without one, a denoising estimate says it stops at the decode (scope <c>denoise</c>), and none is made
    /// before denoising or after it.</summary>
    internal static StudioJobTiming Measure(StudioJobRecord record, DateTimeOffset now, StudioJob? previous, int? jobsAhead)
    {
        static double Seconds(DateTimeOffset from, DateTimeOffset to) => Math.Max(0, (to - from).TotalSeconds);
        var job = record.Job; var end = job.FinishedAt ?? now;
        var timing = new StudioJobTiming
        {
            QueuedSeconds = Seconds(job.CreatedAt, job.StartedAt ?? end),
            ElapsedSeconds = job.StartedAt is { } started ? Seconds(started, end) : null,
            PreviousRunSeconds = previous?.Stages.Sum(s => s.Seconds),
        };
        if (job.State == "queued") { timing.JobsAhead = jobsAhead; return timing; }
        if (job.State is not ("running" or "cancelling") || job.StageStartedAt is not { } stageStarted) return timing;
        double inStage = Seconds(stageStarted, now); timing.StageElapsedSeconds = inStage;
        if (job.State == "cancelling") return timing;
        if (job.Kind == "render")
        {
            // ffmpeg reports the output position it has reached, which advances close to linearly in time.
            if (job.Stage == "rendering" && job.Progress >= .1) { timing.RemainingSeconds = inStage * (1 - job.Progress) / job.Progress; timing.RemainingScope = "job"; }
            return timing;
        }
        var (before, denoise, after) = previous == null ? default : Split(previous.Stages);
        bool measured = denoise > 0; // the previous run recorded its denoising
        int total = job.TotalSteps, done = record.LastStep;
        bool denoising = job.Stage == "denoise", denoised = !denoising && (job.Stages.Exists(s => s.Stage == "denoise") || total > 0 && done >= total);
        // Seconds per step: between this run's finished steps (the first carries one-time uploads and graph builds, so it
        // is the baseline rather than a sample), then the first step alone, then the previous run's average.
        double rate = record.LastStep > record.FirstStep ? Seconds(record.FirstStepAt, record.LastStepAt) / (record.LastStep - record.FirstStep)
            : done > 0 && record.DenoiseStartedAt is { } begun && record.FirstStepAt > begun ? Seconds(begun, record.FirstStepAt) / done
            : measured && total > 0 ? denoise / total : 0;
        if (denoising)
        {
            if (rate <= 0 || total <= 0) return timing;
            int left = Math.Max(0, total - done);
            // Count down through the step running now, but never below the steps still to start.
            double denoiseLeft = Math.Max(rate * left - Seconds(done > 0 ? record.LastStepAt : stageStarted, now), rate * Math.Max(0, left - 1));
            timing.RemainingSeconds = denoiseLeft + after; timing.RemainingScope = measured ? "job" : "denoise";
        }
        else if (!measured) return timing;
        else if (denoised)
        {
            int last = job.Stages.FindLastIndex(s => s.Stage == "denoise");
            double since = (last < 0 ? 0 : job.Stages.Skip(last + 1).Sum(s => s.Seconds)) + inStage;
            // Past the previous run's time there is nothing left to estimate from.
            if (after - since > 0) { timing.RemainingSeconds = after - since; timing.RemainingScope = "job"; }
        }
        else { timing.RemainingSeconds = Math.Max(0, before - (timing.ElapsedSeconds ?? 0)) + denoise + after; timing.RemainingScope = "job"; }
        return timing;
    }
    // A completed run's seconds before its first denoising stage, through its last, and after it.
    static (double Before, double Denoise, double After) Split(List<StudioJobStage> stages)
    {
        int first = stages.FindIndex(s => s.Stage == "denoise"), last = stages.FindLastIndex(s => s.Stage == "denoise");
        if (first < 0) return default;
        return (stages.Take(first).Sum(s => s.Seconds), stages.Skip(first).Take(last - first + 1).Sum(s => s.Seconds), stages.Skip(last + 1).Sum(s => s.Seconds));
    }
    sealed class InlineProgress(Action<StudioProgress> callback) : IProgress<StudioProgress> { public void Report(StudioProgress value) => callback(value); }
    /// <summary>The job-directory name of a take's lossless last frame.</summary>
    internal const string LastFrameFile = "last-frame.png";
    internal static string FailureDiagnostic(StudioJob job, Exception error)
    {
        var diagnostic = new StringBuilder($"Studio job {job.Id} ({job.Kind}) failed during {job.Stage}.");
        // The job document retains the user-facing error. The durable host log adds the call sites without copying
        // exception messages, which adapters or media tools may populate with prompts, file names or command lines.
        for (Exception? cause = error; cause != null; cause = cause.InnerException)
        {
            diagnostic.AppendLine().Append(cause.GetType().FullName);
            if (cause is StudioException studio) diagnostic.Append(" [").Append(studio.Code).Append(']');
            if (cause.StackTrace is { } stack) diagnostic.AppendLine().Append(stack);
        }
        return diagnostic.ToString();
    }
    // An adapter's output must be a regular file inside the job's own work directory.
    static string ManagedOutput(string path, string workDirectory)
    {
        var actualPath = Path.GetFullPath(path);
        if (!actualPath.StartsWith(workDirectory + Path.DirectorySeparatorChar, StringComparison.Ordinal) || File.GetAttributes(actualPath).HasFlag(FileAttributes.ReparsePoint))
            throw new StudioException("unsafe_generation_output", "Generation adapter returned a file outside its managed job directory.");
        return actualPath;
    }
    /// <summary>Moves the take's lossless last frame into the job directory when it is the take's size, so a chain can
    /// continue from the exact pixels rather than a decode of the encoded video. A frame of any other size is not kept,
    /// and the take is then continued from its decoded video (the still records which).</summary>
    async Task KeepLastFrameAsync(StudioJobRecord record, StudioGenerationResult output, string workDirectory, StudioGenerationRequest request, CancellationToken ct)
    {
        if (output.LastFramePath == null || !File.Exists(output.LastFramePath)) return;
        var path = ManagedOutput(output.LastFramePath, workDirectory);
        if (!string.Equals(Path.GetExtension(path), ".png", StringComparison.OrdinalIgnoreCase))
            throw new StudioException("unsafe_generation_output", "A take's last frame must be a PNG image.");
        var probe = await Media.ProbeAsync(path, LastFrameFile, ct);
        if (probe.Kind != "image" || probe.Width != request.Width || probe.Height != request.Height) return;
        File.Move(path, StudioPaths.SafeChild(JobDirectory(record.Job.Id), LastFrameFile), true);
        record.LastFrameFileName = LastFrameFile;
    }
    async Task ProcessJobsAsync()
    {
        try
        {
            await foreach (var id in queue.Reader.ReadAllAsync(shutdown.Token))
            {
                StudioJobRecord record; CancellationTokenSource cancellation;
                lock (jobSync)
                {
                    record = jobs[id]; if (record.Job.State != "queued") continue;
                    cancellation = CancellationTokenSource.CreateLinkedTokenSource(shutdown.Token); jobCancellations[id] = cancellation;
                    var started = DateTimeOffset.UtcNow;
                    record.Job.State = "running"; record.Job.StartedAt = started; EnterStage(record.Job, "starting", started); SaveJob(record);
                }
                var workDirectory = StudioPaths.SafeChild(JobDirectory(id), "work"); Directory.CreateDirectory(workDirectory);
                DateTimeOffset lastSave = DateTimeOffset.MinValue;
                var progress = new InlineProgress(p =>
                {
                    lock (jobSync)
                    {
                        if (record.Job.State != "running") return;
                        var now = DateTimeOffset.UtcNow;
                        // A stage change is saved at once, so a restart reports the stage the job stopped in.
                        if (Advance(record, p, now) || now - lastSave > TimeSpan.FromMilliseconds(300)) { CheckDisk(); SaveJob(record); lastSave = now; }
                    }
                });
                // The job's own trace, current while it runs, so the generation's spans nest under it.
                var activity = StudioTelemetry.StartJob(record); Exception? failure = null;
                try
                {
                    CheckDisk();
                    if (record.Job.Kind == "render")
                    {
                        var output = await Media.RenderAsync(record.Snapshot!, a => GetSnapshotAssetPath(record.Snapshot!, a), workDirectory, progress, cancellation.Token);
                        cancellation.Token.ThrowIfCancellationRequested();
                        var target = StudioPaths.SafeChild(JobDirectory(id), "output.mp4"); File.Move(output, target, true);
                        record.Job.OutputFileName = "output.mp4";
                    }
                    else
                    {
                        var request = record.Generation!;
                        var snapshot = record.Snapshot ?? await GetProjectAsync(request.ProjectId, cancellation.Token);
                        var output = await generation!.GenerateAsync(request, workDirectory, ResolveConditioning(request, snapshot), progress, cancellation.Token);
                        cancellation.Token.ThrowIfCancellationRequested();
                        lock (jobSync) if (record.Job.State == "running") { EnterStage(record.Job, "importing", DateTimeOffset.UtcNow); SaveJob(record); }
                        var actualPath = ManagedOutput(output.FilePath, workDirectory);
                        // Kept before the import: the work directory is deleted once the job ends.
                        await KeepLastFrameAsync(record, output, workDirectory, request, cancellation.Token);
                        await using var stream = File.OpenRead(actualPath);
                        string takeName = TakeFileName(request, await GetProjectAsync(request.ProjectId, cancellation.Token), actualPath);
                        var imported = await ImportAssetInternalAsync(request.ProjectId, stream, takeName, new()
                        { Request = StudioJson.Clone(request), ModelVersion = output.ModelVersion, EffectiveParameters = output.EffectiveParameters }, cancellation.Token);
                        record.Job.OutputAssetId = imported.AssetId;
                    }
                    lock (jobSync) { Finish(record.Job, "completed", DateTimeOffset.UtcNow); record.Job.Progress = 1; SaveJob(record); }
                }
                catch (OperationCanceledException)
                {
                    lock (jobSync)
                    {
                        Finish(record.Job, shutdown.IsCancellationRequested ? "interrupted" : "cancelled", DateTimeOffset.UtcNow);
                        record.Job.Error = shutdown.IsCancellationRequested ? "Host stopped. Retry with a new request identifier; no mid-inference resume is available." : null; SaveJob(record);
                    }
                }
                catch (Exception e)
                {
                    failure = e;
                    try { Options.LogJobFailure?.Invoke(FailureDiagnostic(record.Job, e)); }
                    catch (Exception) { /* A diagnostic sink must not stop this worker or prevent the durable job save. */ }
                    lock (jobSync) { Finish(record.Job, "failed", DateTimeOffset.UtcNow); record.Job.Error = e is OutOfMemoryException ? "Insufficient memory. Release other loaded models or explicitly choose lower settings, then retry." : e.Message; SaveJob(record); }
                }
                finally
                {
                    lock (jobSync) jobCancellations.Remove(id);
                    cancellation.Dispose();
                    try { if (Directory.Exists(workDirectory)) Directory.Delete(workDirectory, true); } catch (IOException) { }
                    // After the cleanup, so telemetry can never leave a work directory behind. The job has ended by now.
                    StudioTelemetry.EndJob(activity, record, failure);
                }
            }
        }
        catch (OperationCanceledException) when (shutdown.IsCancellationRequested) { }
    }
    async Task StopJobsAsync()
    {
        lock (jobSync) { if (stopping) return; stopping = true; queue.Writer.TryComplete(); shutdown.Cancel(); }
        await worker;
        lock (jobSync)
        {
            foreach (var record in jobs.Values.Where(r => r.Job.State == "queued")) { Finish(record.Job, "interrupted", DateTimeOffset.UtcNow); record.Job.Error = "Host stopped before this queued job started. Retry with a new request identifier."; SaveJob(record); StudioTelemetry.TraceJobNotRun(record); }
        }
        StudioTelemetry.Untrack(this);
        shutdown.Dispose();
    }
}
