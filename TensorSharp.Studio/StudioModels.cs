// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Text.Json;
using System.Text.Json.Serialization;

namespace TensorSharp.Studio;

public static class StudioJson
{
    public static readonly JsonSerializerOptions Options = new(JsonSerializerDefaults.Web)
    {
        WriteIndented = true,
        Converters = { new JsonStringEnumConverter(JsonNamingPolicy.CamelCase) }
    };
    public static T Clone<T>(T value) => JsonSerializer.Deserialize<T>(JsonSerializer.Serialize(value, Options), Options)!;
    public static string NewId() => Guid.NewGuid().ToString("N");
}

public sealed class StudioException(string code, string message) : Exception(message)
{
    public string Code { get; } = code;
}

/// <summary>Timeline timestamps are integer frames. Seconds convert using midpoint away from zero.
/// Media render sampling uses presentation timestamps and the project's rational output frame rate.</summary>
public sealed record StudioFrameRate(int Numerator = 24, int Denominator = 1)
{
    public double Value => (double)Numerator / Denominator;
    public double Seconds(long frame) => (double)frame * Denominator / Numerator;
    public long Frames(double seconds) => checked((long)Math.Round(seconds * Numerator / Denominator, MidpointRounding.AwayFromZero));
    public void Validate()
    {
        if (Numerator is < 1 or > 240000 || Denominator is < 1 or > 10000 || Value is < 1 or > 240)
            throw new StudioException("invalid_frame_rate", "Frame rate must be between 1 and 240 fps with positive rational terms.");
    }
}

public sealed class StudioProject
{
    public int SchemaVersion { get; set; } = 1;
    public string Id { get; set; } = StudioJson.NewId();
    public string Name { get; set; } = "Untitled video";
    public long Revision { get; set; }
    public int Width { get; set; } = 1280;
    public int Height { get; set; } = 720;
    public StudioFrameRate FrameRate { get; set; } = new();
    public string? ConversationId { get; set; }
    public DateTimeOffset UpdatedAt { get; set; } = DateTimeOffset.UtcNow;
    public List<StudioAsset> Assets { get; set; } = [];
    public List<StudioTrack> Tracks { get; set; } = [];
    public List<StudioClip> Clips { get; set; } = [];
    public List<StudioCaption> Captions { get; set; } = [];
    public long DurationFrames => Math.Max(Clips.Select(c => checked(c.StartFrame + c.DurationFrames)).DefaultIfEmpty(0).Max(),
        Captions.Select(c => checked(c.StartFrame + c.DurationFrames)).DefaultIfEmpty(0).Max());
}

public sealed class StudioAsset
{
    public string Id { get; set; } = StudioJson.NewId();
    public string Name { get; set; } = "Media";
    public string FileName { get; set; } = "";
    public string Kind { get; set; } = "video";
    public string ContentType { get; set; } = "video/mp4";
    public long Bytes { get; set; }
    public string Sha256 { get; set; } = "";
    public double DurationSeconds { get; set; }
    public int Width { get; set; }
    public int Height { get; set; }
    public bool HasAudio { get; set; }
    public string? SourceFrameRate { get; set; }
    public bool VariableFrameRate { get; set; }
    public bool Missing { get; set; }
    public StudioGenerationProvenance? Generation { get; set; }
}

public sealed class StudioTrack
{
    public string Id { get; set; } = StudioJson.NewId();
    public string Name { get; set; } = "Video";
    public string Kind { get; set; } = "video";
    public bool Muted { get; set; }
    public bool Visible { get; set; } = true;
    public double Volume { get; set; } = 1;
}

public sealed class StudioClip
{
    public string Id { get; set; } = StudioJson.NewId();
    public string AssetId { get; set; } = "";
    public string TrackId { get; set; } = "";
    public long StartFrame { get; set; }
    public long SourceStartFrame { get; set; }
    public long DurationFrames { get; set; }
    public double Volume { get; set; } = 1;
    public long FadeInFrames { get; set; }
    public long FadeOutFrames { get; set; }
    /// <summary>Additional audio-only fade-out, used by a dissolve while the outgoing picture remains opaque.</summary>
    public long AudioFadeOutFrames { get; set; }
    /// <summary>Clips with the same non-empty group move, trim, and split together.</summary>
    public string? LinkGroup { get; set; }
    public string Fit { get; set; } = "contain";
    public string? ShotId { get; set; }
}

public sealed class StudioCaption
{
    public string Id { get; set; } = StudioJson.NewId();
    public string Text { get; set; } = "";
    public long StartFrame { get; set; }
    public long DurationFrames { get; set; }
}

/// <summary>One command shape for UI and agents. Only fields used by the selected Type are accepted semantically.</summary>
public sealed class StudioCommand
{
    public string Type { get; set; } = "";
    public string? ConversationId { get; set; }
    public string? TrackId { get; set; }
    public string? ClipId { get; set; }
    public string? OtherClipId { get; set; }
    public string? AssetId { get; set; }
    public string? CaptionId { get; set; }
    public string? Name { get; set; }
    public string? Kind { get; set; }
    public string? Text { get; set; }
    public string? LinkGroup { get; set; }
    public string? ShotId { get; set; }
    public string? Fit { get; set; }
    public long? StartFrame { get; set; }
    public long? SourceStartFrame { get; set; }
    public long? DurationFrames { get; set; }
    public long? AtFrame { get; set; }
    public long? FadeInFrames { get; set; }
    public long? FadeOutFrames { get; set; }
    public long? AudioFadeOutFrames { get; set; }
    public double? Volume { get; set; }
    public bool? Muted { get; set; }
    public bool? Visible { get; set; }
    /// <summary>Explicit confirmation to remove timeline uses when deleting an asset from the library.</summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public bool? RemoveTimelineClips { get; set; }
    public int? Index { get; set; }
    public int? Width { get; set; }
    public int? Height { get; set; }
    public bool ApplyToLinked { get; set; } = true;
}

public sealed class StudioProposal
{
    public string Id { get; set; } = StudioJson.NewId();
    public string ProjectId { get; set; } = "";
    public long ExpectedRevision { get; set; }
    public string Summary { get; set; } = "";
    public DateTimeOffset CreatedAt { get; set; } = DateTimeOffset.UtcNow;
    public List<StudioCommand> Commands { get; set; } = [];
}

public sealed class StudioGenerationRequest
{
    public string ProjectId { get; set; } = "";
    public string RequestId { get; set; } = StudioJson.NewId();
    public string ModelId { get; set; } = "";
    public string Prompt { get; set; } = "";
    public string? NegativePrompt { get; set; }
    /// <summary>Zero uses the selected model's host-configured generation size, except that width = height = 0 in one of
    /// the model's <see cref="StudioModelCapability.ImageCanvasModes"/> sizes the canvas from the first-frame image, or
    /// from the last-frame image when there is no first.</summary>
    public int Width { get; set; }
    public int Height { get; set; }
    public int FrameCount { get; set; }
    public int Fps { get; set; }
    public int Steps { get; set; }
    public float CfgScale { get; set; }
    /// <summary>The <see cref="StudioModelCapability.SamplingRecipe"/> this shot was planned for. Null binds the model's
    /// recipe at admission, after the request's fingerprint, so a resubmission stays idempotent; a request planned for
    /// another recipe is refused (<c>sampling_recipe_conflict</c>), never resampled. Requests saved before recipes omit it.</summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? SamplingRecipe { get; set; }
    public long Seed { get; set; } = -1;
    /// <summary>Null follows the model default; explicit false retains silent generation.</summary>
    public bool? GenerateAudio { get; set; }
    /// <summary>t2v, i2v, fl2v, keyframes or ref (see <see cref="StudioGenerationModes"/>).
    /// Null infers the mode from the supplied inputs, as requests saved before explicit modes did.</summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Mode { get; set; }
    /// <summary>The pinned first frame (i2v, optionally fl2v and keyframes). Optional in fl2v.</summary>
    public string? ConditioningAssetId { get; set; }
    /// <summary>The pinned last frame (fl2v, optionally keyframes). A last frame alone is fl2v.</summary>
    public string? EndConditioningAssetId { get; set; }
    public string? ShotId { get; set; }
    public List<StudioTemporalCondition> TemporalConditions { get; set; } = [];
    public List<string> ReferenceAssetIds { get; set; } = [];
}

public sealed record StudioTemporalCondition(string AssetId, int FrameIndex, float Strength = 1);

public sealed class StudioGenerationProvenance
{
    public StudioGenerationRequest Request { get; set; } = new();
    public string ModelVersion { get; set; } = "";
    public Dictionary<string, string> EffectiveParameters { get; set; } = [];
    public DateTimeOffset CreatedAt { get; set; } = DateTimeOffset.UtcNow;
}

public sealed class StudioModelCapability
{
    public string ModelId { get; set; } = "";
    public string DisplayName { get; set; } = "";
    public string Family { get; set; } = "";
    public bool Available { get; set; }
    public string? UnavailableReason { get; set; }
    public bool TextToVideo { get; set; }
    public bool ImageConditioning { get; set; }
    public bool LastFrameConditioning { get; set; }
    public bool AudioOutput { get; set; }
    public bool TemporalGuidance { get; set; }
    public bool IdentityReferences { get; set; }
    public int MaxReferences { get; set; }
    /// <summary>The generation modes this checkpoint runs, derived from the conditioning flags above.</summary>
    public string[] Modes => StudioGenerationModes.For(this);
    /// <summary>Modes of this model family that the loaded checkpoint or host cannot run, with the reason
    /// (for example, MiniMax-H3 identity references need its separate Ref2VA checkpoint).</summary>
    public Dictionary<string, string> UnavailableModes { get; set; } = [];
    /// <summary>Minimum output frames for ref mode, when the reference workflow needs a longer clip.</summary>
    public int? ReferenceMinFrames { get; set; }
    public int DefaultWidth { get; set; } = 512;
    public int DefaultHeight { get; set; } = 512;
    /// <summary>Modes in which a request with width = height = 0 takes its canvas from its first-frame image
    /// (the last-frame image when it has no first), by the model's own recipe, instead of from
    /// <see cref="DefaultWidth"/>/<see cref="DefaultHeight"/>. These modes keep the image's framing, so the canvas sets
    /// how many pixels its faces get. The host fills in the exact size before the job is queued.</summary>
    public string[] ImageCanvasModes { get; set; } = [];
    /// <summary>The recipe <see cref="ImageCanvasModes"/> follow, so a page can show the size before the host confirms it:
    /// the image's aspect at this short edge, shrunk to <see cref="ImageCanvasMaxPixels"/> when larger, each side on the nearest
    /// <see cref="SpatialMultiple"/> step (ties to even). Aspect ratios beyond 1:<see cref="ImageCanvasMaxAspect"/> or
    /// <see cref="ImageCanvasMaxAspect"/>:1 need an explicit size. All zero when <see cref="ImageCanvasModes"/> is empty.</summary>
    public int ImageCanvasShortEdge { get; set; }
    public long ImageCanvasMaxPixels { get; set; }
    public double ImageCanvasMaxAspect { get; set; }
    /// <summary>The clip lengths, in seconds, the checkpoint's publisher trained and recommends (MiniMax-H3: 4–15 s);
    /// null when it states none. A page can say when a take falls outside it: a much shorter take that ends on a photo
    /// tends to cut or dissolve to it rather than move there. Never enforced.</summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public double? RecommendedMinSeconds { get; set; }
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public double? RecommendedMaxSeconds { get; set; }
    public int DefaultFps { get; set; } = 24;
    public bool DefaultGenerateAudio { get; set; }
    public int DefaultFrameCount { get; set; } = 17;
    public int DefaultSteps { get; set; } = 20;
    public int? FixedSteps { get; set; }
    /// <summary>The id of a model's own built-in schedule (the host model's <c>VideoSamplingContract.NativeId</c>).</summary>
    public const string NativeSamplingRecipe = "native";
    /// <summary>The sampling recipe the model is bound to: <see cref="NativeSamplingRecipe"/>, or the id of a recipe that
    /// fixes the steps, sigma grid, shift and sampler (a step-distillation plug-in's, for example
    /// <c>minimax-h3-lora:minimax-h3-fl2v-turbo-8step</c>). Admission binds each job to it.</summary>
    public string SamplingRecipe { get; set; } = NativeSamplingRecipe;
    /// <summary>The recipe's display name; null for <see cref="NativeSamplingRecipe"/>.</summary>
    public string? SamplingRecipeName { get; set; }
    public float DefaultCfgScale { get; set; } = 5;
    public float? FixedCfgScale { get; set; }
    public string[] DeferredCapabilities { get; set; } = [];
    public int SpatialMultiple { get; set; } = 16;
    public int TemporalMultiple { get; set; } = 4;
    public int TemporalOffset { get; set; } = 1;
    public int? FixedFps { get; set; }
    public int MaxFrames { get; set; } = 257;
    public int MaxDimension { get; set; } = 2048;

    /// <summary>Why a step count other than <see cref="FixedSteps"/> is refused, naming what fixes it.</summary>
    public string FixedStepsRefusal() =>
        $"This model samples exactly {FixedSteps} steps ({SamplingRecipeName ?? "fixed by the checkpoint"}); leave steps 0 for its default.";

    /// <summary>Refuse a request planned for a recipe other than <see cref="SamplingRecipe"/>: its steps belong to
    /// that recipe (code <c>sampling_recipe_conflict</c>, a 409). A request that names none is bound later.</summary>
    public void RequireSamplingRecipe(StudioGenerationRequest request)
    {
        if (request.SamplingRecipe == null || request.SamplingRecipe == SamplingRecipe) return;
        string planned = request.SamplingRecipe == NativeSamplingRecipe ? "the model's own schedule" : request.SamplingRecipe;
        throw new StudioException("sampling_recipe_conflict", $"This shot was planned for {planned} but the loaded model now samples with " +
            $"{SamplingRecipeName ?? "its own schedule"}. Refresh the model settings and review the steps.");
    }
}

public sealed record StudioGenerationResult(string FilePath, string ModelVersion, Dictionary<string, string> EffectiveParameters);
public sealed record StudioProgress(string Stage, double Fraction, string? Detail = null)
{
    /// <summary>Denoising steps finished so far (never the step still running) and the steps planned; both 0 for a
    /// stage that has no steps.</summary>
    public int Step { get; init; }
    public int TotalSteps { get; init; }
}

/// <summary>Host adapter must use its existing model registry/resource leases. Output must be inside outputDirectory.
/// Cancellation must not return until inference and subprocesses have stopped using files or buffers.</summary>
public interface IStudioGenerationAdapter
{
    Task<IReadOnlyList<StudioModelCapability>> GetCapabilitiesAsync(CancellationToken cancellationToken);
    Task ValidateAsync(StudioGenerationRequest request, IReadOnlyDictionary<string, string> conditioningAssets, CancellationToken cancellationToken);
    Task<StudioGenerationResult> GenerateAsync(StudioGenerationRequest request, string outputDirectory,
        IReadOnlyDictionary<string, string> conditioningAssets, IProgress<StudioProgress> progress, CancellationToken cancellationToken);
}

public sealed class StudioJob
{
    public string Id { get; set; } = StudioJson.NewId();
    public string ProjectId { get; set; } = "";
    public string RequestId { get; set; } = "";
    public string Kind { get; set; } = "render";
    public string State { get; set; } = "queued";
    public string Stage { get; set; } = "queued";
    public double Progress { get; set; }
    public long ProjectRevision { get; set; }
    public string? Error { get; set; }
    public string? OutputAssetId { get; set; }
    public string? OutputFileName { get; set; }
    public DateTimeOffset CreatedAt { get; set; } = DateTimeOffset.UtcNow;
    public DateTimeOffset UpdatedAt { get; set; } = DateTimeOffset.UtcNow;
    /// <summary>The worker's latest note on its stage, e.g. "cond pass" or "tile 3/12".</summary>
    public string? Detail { get; set; }
    /// <summary>Denoising steps finished and planned; both 0 for a job or stage without steps.</summary>
    public int Step { get; set; }
    public int TotalSteps { get; set; }
    public DateTimeOffset? StartedAt { get; set; }
    public DateTimeOffset? StageStartedAt { get; set; }
    public DateTimeOffset? FinishedAt { get; set; }
    /// <summary>The stages the job has left, in order, with how long each ran.</summary>
    public List<StudioJobStage> Stages { get; set; } = [];
    /// <summary>Clock-derived view, filled in on the copies Studio returns and never stored. Durations are computed on
    /// the host, so a client with a skewed clock still shows the right elapsed and remaining time.</summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public StudioJobTiming? Timing { get; set; }
    /// <summary>What a generation job makes, from its request; null for a render.</summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public StudioJobSummary? Summary { get; set; }
}

public sealed record StudioJobStage(string Stage, double Seconds);

public sealed class StudioJobTiming
{
    /// <summary>Seconds since the worker started the job, or its whole run once it has ended; null before it starts.</summary>
    public double? ElapsedSeconds { get; set; }
    /// <summary>Seconds in the current stage; null unless the job is running.</summary>
    public double? StageElapsedSeconds { get; set; }
    /// <summary>Seconds spent waiting in the queue: so far while queued, the whole wait once started.</summary>
    public double QueuedSeconds { get; set; }
    /// <summary>Jobs the worker runs before this one; null unless queued.</summary>
    public int? JobsAhead { get; set; }
    /// <summary>Estimated seconds still to run; null when nothing measured supports an estimate.</summary>
    public double? RemainingSeconds { get; set; }
    /// <summary><c>job</c> when <see cref="RemainingSeconds"/> runs to the finished take, <c>denoise</c> when it covers
    /// only the denoising steps left because nothing has measured the decode that follows them.</summary>
    public string? RemainingScope { get; set; }
    /// <summary>How long the last completed run of the same model and settings took; the basis for every estimate
    /// beyond the live step rate.</summary>
    public double? PreviousRunSeconds { get; set; }
}

public sealed class StudioJobSummary
{
    public string ModelId { get; set; } = "";
    public string Mode { get; set; } = "";
    public string Prompt { get; set; } = "";
    public int Width { get; set; }
    public int Height { get; set; }
    public int FrameCount { get; set; }
    public int Fps { get; set; }
    public int Steps { get; set; }
    /// <summary>The requested seed; -1 means the model draws one.</summary>
    public long Seed { get; set; }
    public bool? GenerateAudio { get; set; }
    /// <summary>The photos the shot starts and ends on, when it pins them, so a page can say "Ends on beach.jpg".</summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? FirstFrameAssetId { get; set; }
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? LastFrameAssetId { get; set; }
    /// <summary>How many timed keyframes and identity references the shot uses.</summary>
    public int GuideCount { get; set; }
    public int ReferenceCount { get; set; }
}

public sealed class StudioOptions
{
    public string FfmpegPath { get; set; } = "ffmpeg";
    public string FfprobePath { get; set; } = "ffprobe";
    public long MaxImportBytes { get; set; } = 2L * 1024 * 1024 * 1024;
    public int MaxQueuedJobs { get; set; } = 8;
    public int MaxHistory { get; set; } = 40;
    public long MaxTimelineFrames { get; set; } = 24 * 60 * 60;
    public long MinimumFreeDiskBytes { get; set; } = 256L * 1024 * 1024;
    public bool EnableExternalProcesses { get; set; } = !OperatingSystem.IsIOS() && !OperatingSystem.IsTvOS() && !OperatingSystem.IsAndroid();
    public string? CaptionFontPath { get; set; }
}
