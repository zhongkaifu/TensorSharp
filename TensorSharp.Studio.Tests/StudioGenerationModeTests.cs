using System.Text.Json;
using TensorSharp.Studio;

namespace TensorSharp.Studio.Tests;

/// <summary>Mode contract only: what each mode accepts and how admission checks it against
/// a checkpoint. A test double stands in for the model; no inference runs.</summary>
public sealed class StudioGenerationModeTests : IDisposable
{
    readonly string root = Path.Combine(Path.GetTempPath(), "studio-modes-" + StudioJson.NewId());
    public void Dispose() { if (Directory.Exists(root)) Directory.Delete(root, true); }

    static StudioGenerationRequest Inputs(string? mode, string inputs) => new()
    {
        Prompt = "A shot", Mode = mode,
        ConditioningAssetId = inputs.Contains('F') ? "first" : null,
        EndConditioningAssetId = inputs.Contains('L') ? "last" : null,
        TemporalConditions = inputs.Contains('G') ? [new("guide", 8)] : [],
        ReferenceAssetIds = inputs.Contains('R') ? ["ref-a", "ref-b"] : [],
    };

    [Theory]
    [InlineData("t2v", "")]
    [InlineData("i2v", "F")]
    [InlineData("fl2v", "FL")]
    [InlineData("fl2v", "L")]  // The first frame is optional: a clip may only END on an image.
    [InlineData("keyframes", "G")]
    [InlineData("keyframes", "FLG")]
    [InlineData("ref", "R")]
    public void EachModeAcceptsExactlyItsOwnInputs(string mode, string inputs) =>
        Assert.Equal(mode, StudioGenerationModes.Resolve(Inputs(mode, inputs)));

    [Theory]
    [InlineData("t2v", "F")]   // Text only takes no images.
    [InlineData("t2v", "R")]
    [InlineData("i2v", "")]    // i2v animates a first frame it must be given.
    [InlineData("i2v", "FL")]  // A last frame is fl2v, not i2v.
    [InlineData("i2v", "FR")]
    [InlineData("fl2v", "F")]  // fl2v pins the last frame; a first frame alone is i2v.
    [InlineData("fl2v", "")]
    [InlineData("fl2v", "FLG")]
    [InlineData("fl2v", "LR")]
    [InlineData("keyframes", "F")]
    [InlineData("keyframes", "GR")]
    [InlineData("ref", "")]
    [InlineData("ref", "FR")]  // A reference is never also a pinned frame.
    [InlineData("ref", "RG")]
    public void AModeRefusesInputsThatBelongToAnotherMode(string mode, string inputs) =>
        Assert.Equal("invalid_mode_inputs", Assert.Throws<StudioException>(() => StudioGenerationModes.Resolve(Inputs(mode, inputs))).Code);

    [Theory]
    [InlineData("", "t2v")]
    [InlineData("F", "i2v")]
    [InlineData("FL", "fl2v")]
    [InlineData("L", "fl2v")]
    [InlineData("FG", "keyframes")]
    [InlineData("R", "ref")]
    public void RequestsSavedWithoutAModeRunAsTheirInputsImply(string inputs, string expected)
    {
        var request = Inputs(null, inputs);
        Assert.Equal(expected, StudioGenerationModes.Resolve(request));
        Assert.Null(request.Mode); // Admission leaves legacy requests (and their idempotency fingerprint) unchanged.
    }

    [Fact]
    public void ExplicitModesAreNormalizedAndUnknownModesRefused()
    {
        var request = Inputs(" I2V ", "F");
        Assert.Equal("i2v", StudioGenerationModes.Resolve(request));
        Assert.Equal("i2v", request.Mode);
        Assert.Equal("invalid_mode", Assert.Throws<StudioException>(() => StudioGenerationModes.Resolve(Inputs("v2v", ""))).Code);
        var duplicate = Inputs("ref", ""); duplicate.ReferenceAssetIds = ["same", "same"];
        Assert.Equal("invalid_mode_inputs", Assert.Throws<StudioException>(() => StudioGenerationModes.Resolve(duplicate)).Code);
    }

    [Fact]
    public void LegacyRequestsSerializeWithoutAModeSoReplayedJobsKeepTheirFingerprint()
    {
        var legacy = JsonSerializer.SerializeToElement(Inputs(null, "F"), StudioJson.Options);
        Assert.False(legacy.TryGetProperty("mode", out _));
        var explicitMode = JsonSerializer.SerializeToElement(Inputs("i2v", "F"), StudioJson.Options);
        Assert.Equal("i2v", explicitMode.GetProperty("mode").GetString());
    }

    [Theory]
    [InlineData(true, false, false, false, false, "t2v")]
    [InlineData(false, true, false, false, false, "i2v")]
    [InlineData(true, true, true, false, false, "t2v,i2v,fl2v")]
    [InlineData(true, true, true, true, false, "t2v,i2v,fl2v,keyframes")]
    [InlineData(true, false, false, false, true, "t2v,ref")]
    [InlineData(true, true, true, true, true, "t2v,i2v,fl2v,keyframes,ref")]
    public void CapabilityModesFollowTheConditioningFlags(bool text, bool first, bool last, bool timed, bool references, string expected)
    {
        var capability = new StudioModelCapability
        {
            TextToVideo = text, ImageConditioning = first, LastFrameConditioning = last, TemporalGuidance = timed,
            IdentityReferences = references, MaxReferences = references ? 9 : 0,
        };
        Assert.Equal(expected, string.Join(",", capability.Modes));
        var json = JsonSerializer.SerializeToElement(capability, StudioJson.Options);
        Assert.Equal(expected, string.Join(",", json.GetProperty("modes").EnumerateArray().Select(m => m.GetString())));
    }

    [Fact]
    public void TakesAreNamedAfterTheirPromptAndNumberedWithinTheirShot()
    {
        var project = new StudioProject();
        var request = new StudioGenerationRequest { Prompt = "  Waves roll in at sunrise / warm light, the camera drifts slowly along the shore line", ShotId = "shot" };
        Assert.Equal("Waves roll in at sunrise warm light, the camera.mp4", StudioJobs_TakeName(request, project));
        // Cut at the last whole word that fits, never mid-word.
        Assert.Equal("The camera drifts slowly and settles on the.mp4",
            StudioJobs_TakeName(new() { Prompt = "The camera drifts slowly and settles on the colour bars." }, project));
        project.Assets.Add(new() { Generation = new() { Request = new() { ShotId = "shot" } } });
        Assert.EndsWith(" · take 2.mp4", StudioJobs_TakeName(request, project));
        Assert.Equal("Generated take.mp4", StudioJobs_TakeName(new() { Prompt = "\t\n" }, project));
        // A deleted take never makes the next one repeat a number: numbering follows the highest take of the shot.
        project.Assets.Add(new() { Name = "Waves · take 3.mp4", Generation = new() { Request = new() { ShotId = "shot" } } });
        Assert.EndsWith(" · take 4.mp4", StudioJobs_TakeName(request, project));
        // An emoji straddling the cut is dropped whole, never split into half a surrogate pair.
        var emoji = StudioJobs_TakeName(new() { Prompt = new string('a', 47) + "\U0001F30A waves" }, project);
        Assert.Equal(new string('a', 47) + ".mp4", emoji);
        Assert.DoesNotContain('\uFFFD', emoji);
        Assert.Equal("A cat.mp4", StudioJobs_TakeName(new() { Prompt = "A cat." }, project));
    }

    static string StudioJobs_TakeName(StudioGenerationRequest request, StudioProject project) =>
        StudioService.TakeFileName(request, project, "/work/job/generated.mp4");

    [Fact]
    public void JobSummariesSayWhichPhotosAShotUses()
    {
        var summary = JsonSerializer.SerializeToElement(new StudioJobSummary { Mode = "fl2v", LastFrameAssetId = "beach", ReferenceCount = 0 }, StudioJson.Options);
        Assert.Equal("beach", summary.GetProperty("lastFrameAssetId").GetString());
        Assert.False(summary.TryGetProperty("firstFrameAssetId", out _)); // a last frame alone: nothing stands in for the first
    }

    [Fact]
    public void AReferenceCapabilityWithoutAnyReferenceSlotDoesNotAdvertiseRef() =>
        Assert.DoesNotContain("ref", new StudioModelCapability { TextToVideo = true, IdentityReferences = true, MaxReferences = 0 }.Modes);

    sealed class RecordingAdapter(StudioModelCapability capability) : IStudioGenerationAdapter
    {
        public int Validations, Generations;
        public Task<IReadOnlyList<StudioModelCapability>> GetCapabilitiesAsync(CancellationToken ct) => Task.FromResult<IReadOnlyList<StudioModelCapability>>([capability]);
        public Task ValidateAsync(StudioGenerationRequest r, IReadOnlyDictionary<string, string> assets, CancellationToken ct) { Validations++; return Task.CompletedTask; }
        public Task<StudioGenerationResult> GenerateAsync(StudioGenerationRequest r, string directory, IReadOnlyDictionary<string, string> assets, IProgress<StudioProgress> progress, CancellationToken ct)
        { Generations++; return Task.FromException<StudioGenerationResult>(new OperationCanceledException()); }
    }

    static StudioModelCapability FirstLastCheckpoint() => new()
    {
        ModelId = "fl2va-double", Family = "fixture-family", Available = true, TextToVideo = true, ImageConditioning = true,
        LastFrameConditioning = true, MaxDimension = 1024, DefaultFrameCount = 17,
        UnavailableModes = { ["ref"] = "Identity references need the Ref2VA checkpoint." },
    };

    static async Task<StudioService> AdmittingService(string root, IStudioGenerationAdapter adapter)
    {
        var service = new StudioService(root, adapter);
        var media = await service.Media.GetCapabilitiesAsync();
        if (!(media.Available && media.ProbeAvailable)) { await service.DisposeAsync(); Skip.If(true, "Generation admission requires installed ffprobe and FFmpeg H.264/AAC: " + media.Reason); }
        return service;
    }

    [SkippableFact]
    public async Task AModeTheCheckpointCannotRunIsRefusedWithItsReasonBeforeTheModelSeesIt()
    {
        var adapter = new RecordingAdapter(FirstLastCheckpoint());
        await using var service = await AdmittingService(root, adapter);
        var project = await service.CreateProjectAsync();
        var request = new StudioGenerationRequest { ProjectId = project.Id, ModelId = "fl2va-double", Prompt = "New scene", Width = 64, Height = 64, Mode = "ref", ReferenceAssetIds = [StudioJson.NewId()] };
        var error = await Assert.ThrowsAsync<StudioException>(() => service.StartGenerationAsync(request));
        Assert.Equal("unsupported_mode", error.Code);
        Assert.Equal("Identity references need the Ref2VA checkpoint.", error.Message);
        request.Mode = "keyframes"; request.ReferenceAssetIds = []; request.TemporalConditions = [new(StudioJson.NewId(), 4)];
        error = await Assert.ThrowsAsync<StudioException>(() => service.StartGenerationAsync(request));
        Assert.Equal("unsupported_mode", error.Code);
        Assert.Contains("t2v, i2v, fl2v", error.Message);
        Assert.Equal(0, adapter.Validations); Assert.Empty(await service.ListJobsAsync(project.Id));
    }

    [SkippableFact]
    public async Task MismatchedModeInputsAndShortReferenceClipsFailAdmission()
    {
        var capability = FirstLastCheckpoint(); capability.UnavailableModes.Clear();
        capability.IdentityReferences = true; capability.MaxReferences = 4; capability.ReferenceMinFrames = 121; capability.TemporalMultiple = 8; capability.MaxFrames = 257;
        var adapter = new RecordingAdapter(capability);
        await using var service = await AdmittingService(root, adapter);
        var project = await service.CreateProjectAsync();
        var mismatched = new StudioGenerationRequest { ProjectId = project.Id, ModelId = "fl2va-double", Prompt = "Pinned ends", Width = 64, Height = 64, Mode = "fl2v", ConditioningAssetId = StudioJson.NewId() };
        Assert.Equal("invalid_mode_inputs", (await Assert.ThrowsAsync<StudioException>(() => service.StartGenerationAsync(mismatched))).Code);
        var shortReference = new StudioGenerationRequest { ProjectId = project.Id, ModelId = "fl2va-double", Prompt = "New scene", Width = 64, Height = 64, FrameCount = 17, Mode = "ref", ReferenceAssetIds = [StudioJson.NewId()] };
        var error = await Assert.ThrowsAsync<StudioException>(() => service.StartGenerationAsync(shortReference));
        Assert.Equal("invalid_frame_count", error.Code); Assert.Contains("121", error.Message);
        Assert.Equal(0, adapter.Validations); Assert.Empty(await service.ListJobsAsync(project.Id));
    }

    [Fact]
    public async Task PlannedShotsCheckTheirModeWhenSavedAndKeepItForTheModelToCheckLater()
    {
        await using var service = new StudioService(root);
        var project = await service.CreateProjectAsync();
        var draft = await service.CreateGenerationDraftAsync(project.Id, project.Revision,
            new() { Prompt = "A drone shot over hills.", Mode = "T2V" }, "Opening establishing shot.");
        Assert.Equal("t2v", draft.Request.Mode);
        Assert.Equal("t2v", Assert.Single(await service.ListGenerationDraftsAsync(project.Id)).Request.Mode);
        var error = await Assert.ThrowsAsync<StudioException>(() => service.CreateGenerationDraftAsync(project.Id, project.Revision,
            new() { Prompt = "Animate the photo.", Mode = "i2v" }, "Missing its first frame."));
        Assert.Equal("invalid_mode_inputs", error.Code);
    }
}
