using System.Text.Json;
using System.Text.Json.Nodes;
using TensorSharp.Studio;

namespace TensorSharp.Studio.Tests;

/// <summary>
/// Admission under a sampling recipe (a step-distillation plug-in such as MiniMax-H3 Turbo): the recipe's fixed step
/// count fills an unset one and refuses any other by name; the job is bound to the recipe after its fingerprint, so the
/// same unbound request finds the same job; a shot planned for another recipe is a 409 with no job. The adapter is a
/// test double that advertises a recipe; no model, plug-in or inference is involved.
/// </summary>
public sealed class StudioSamplingRecipeTests : IDisposable
{
    const string TurboRecipe = "minimax-h3-lora:minimax-h3-fl2v-turbo-8step", TurboName = "LightX2V Turbo 8-step (FL2V v1.0, 768p)";
    const string TurboStepsRefusal = "This model samples exactly 8 steps (" + TurboName + "); leave steps 0 for its default.";
    readonly string root = Path.Combine(Path.GetTempPath(), "studio-recipe-" + StudioJson.NewId());
    public void Dispose() { if (Directory.Exists(root)) Directory.Delete(root, true); }

    sealed class RecipeAdapter : IStudioGenerationAdapter
    {
        public readonly StudioModelCapability Capability = new() { ModelId = "h3", Family = "minimax-h3", Available = true, TextToVideo = true, MaxDimension = 1024 };
        public int ValidationCalls, Calls; public StudioGenerationRequest? Validated, Generated;
        public readonly TaskCompletionSource Started = new(TaskCreationOptions.RunContinuationsAsynchronously);
        public RecipeAdapter LoadTurbo()
        {
            Capability.FixedSteps = Capability.DefaultSteps = 8;
            Capability.SamplingRecipe = TurboRecipe; Capability.SamplingRecipeName = TurboName; return this;
        }
        public Task<IReadOnlyList<StudioModelCapability>> GetCapabilitiesAsync(CancellationToken ct) => Task.FromResult<IReadOnlyList<StudioModelCapability>>([Capability]);
        public Task ValidateAsync(StudioGenerationRequest r, IReadOnlyDictionary<string, string> assets, CancellationToken ct) { ValidationCalls++; Validated = StudioJson.Clone(r); return Task.CompletedTask; }
        public async Task<StudioGenerationResult> GenerateAsync(StudioGenerationRequest r, string directory, IReadOnlyDictionary<string, string> assets, IProgress<StudioProgress> progress, CancellationToken ct)
        {
            Calls++; Generated = StudioJson.Clone(r); Started.TrySetResult();
            await Task.Delay(Timeout.Infinite, ct); throw new InvalidOperationException();
        }
    }

    static StudioGenerationRequest Shot(string projectId, int steps = 0, string? recipe = null) =>
        new() { ProjectId = projectId, ModelId = "h3", Prompt = "A kite rises over the dunes.", Width = 64, Height = 64, FrameCount = 17, Steps = steps, SamplingRecipe = recipe };

    static async Task RequireMediaAsync(StudioService service)
    {
        var capability = await service.Media.GetCapabilitiesAsync();
        Skip.IfNot(capability.Available && capability.ProbeAvailable, "Generation admission requires installed ffprobe and FFmpeg H.264/AAC: " + capability.Reason);
    }

    JsonNode? StoredRequest(string jobId) => JsonNode.Parse(File.ReadAllText(Path.Combine(root, "jobs", jobId, "job.json")))!["generation"];

    [SkippableFact]
    public async Task ARecipeFillsItsStepsAndBindsTheJobAfterTheFingerprint()
    {
        var adapter = new RecipeAdapter().LoadTurbo();
        await using var service = new StudioService(root, adapter); await RequireMediaAsync(service);
        var project = await service.CreateProjectAsync(); var shot = Shot(project.Id);
        var job = await service.StartGenerationAsync(shot);

        var validated = Assert.IsType<StudioGenerationRequest>(adapter.Validated);
        Assert.Equal(8, validated.Steps);
        Assert.Null(validated.SamplingRecipe); // bound once the adapter has validated the request
        await adapter.Started.Task.WaitAsync(TimeSpan.FromSeconds(5));
        Assert.Equal(TurboRecipe, adapter.Generated!.SamplingRecipe);
        Assert.Equal(TurboRecipe, (string?)StoredRequest(job.Id)!["samplingRecipe"]);
        Assert.Null(shot.SamplingRecipe); // the caller's request is not mutated

        // The binding is not part of the fingerprint: the same unbound request, or one repeating the recipe's own step
        // count, is the same job, admitted once.
        Assert.Equal(job.Id, (await service.StartGenerationAsync(shot)).Id);
        var explicitSteps = StudioJson.Clone(shot); explicitSteps.Steps = 8;
        Assert.Equal(job.Id, (await service.StartGenerationAsync(explicitSteps)).Id);
        Assert.Equal(1, adapter.ValidationCalls); Assert.Equal(1, adapter.Calls);
        Assert.Single(await service.ListJobsAsync(project.Id));
        await service.CancelJobAsync(job.Id);
    }

    [SkippableFact]
    public async Task TheModelsOwnScheduleIsBoundAsNative()
    {
        var adapter = new RecipeAdapter();
        await using var service = new StudioService(root, adapter); await RequireMediaAsync(service);
        var project = await service.CreateProjectAsync();
        var job = await service.StartGenerationAsync(Shot(project.Id));
        await adapter.Started.Task.WaitAsync(TimeSpan.FromSeconds(5));
        Assert.Equal(20, adapter.Generated!.Steps);
        Assert.Equal(StudioModelCapability.NativeSamplingRecipe, adapter.Generated.SamplingRecipe);
        Assert.Equal("native", (string?)StoredRequest(job.Id)!["samplingRecipe"]);
        await service.CancelJobAsync(job.Id);
    }

    [SkippableTheory]
    [InlineData(true, 20, TurboStepsRefusal)]
    [InlineData(true, 4, TurboStepsRefusal)]
    [InlineData(false, 20, "This model samples exactly 8 steps (fixed by the checkpoint); leave steps 0 for its default.")]
    public async Task AnotherStepCountIsRefusedByNameBeforeAJobExists(bool turbo, int steps, string message)
    {
        var adapter = new RecipeAdapter();
        if (turbo) adapter.LoadTurbo(); else adapter.Capability.FixedSteps = adapter.Capability.DefaultSteps = 8; // a distilled checkpoint
        await using var service = new StudioService(root, adapter); await RequireMediaAsync(service);
        var project = await service.CreateProjectAsync();
        var error = await Assert.ThrowsAsync<StudioException>(() => service.StartGenerationAsync(Shot(project.Id, steps)));
        Assert.Equal("invalid_sampling", error.Code); Assert.Equal(message, error.Message);
        Assert.Equal(0, adapter.ValidationCalls); Assert.Empty(await service.ListJobsAsync(project.Id));
    }

    [SkippableTheory]
    [InlineData(true, "native", 0, "This shot was planned for the model's own schedule but the loaded model now samples with " + TurboName + ".")]
    [InlineData(true, "native", 20, "This shot was planned for the model's own schedule but the loaded model now samples with " + TurboName + ".")]
    [InlineData(false, TurboRecipe, 8, "This shot was planned for " + TurboRecipe + " but the loaded model now samples with its own schedule.")]
    public async Task AShotPlannedForAnotherRecipeIsAConflictWithNoJob(bool turbo, string planned, int steps, string message)
    {
        var adapter = new RecipeAdapter(); if (turbo) adapter.LoadTurbo();
        await using var service = new StudioService(root, adapter); await RequireMediaAsync(service);
        var project = await service.CreateProjectAsync();
        // The conflict comes first: a base shot's 20 steps are the recipe's problem, not a step-count typo.
        var error = await Assert.ThrowsAsync<StudioException>(() => service.StartGenerationAsync(Shot(project.Id, steps, planned)));
        Assert.Equal("sampling_recipe_conflict", error.Code);
        Assert.Equal(message + " Refresh the model settings and review the steps.", error.Message);
        Assert.Equal(0, adapter.ValidationCalls); Assert.Equal(0, adapter.Calls);
        Assert.Empty(await service.ListJobsAsync(project.Id));
        // The same shot planned for the loaded recipe is admitted and keeps the recipe it named.
        var job = await service.StartGenerationAsync(Shot(project.Id, recipe: adapter.Capability.SamplingRecipe));
        await adapter.Started.Task.WaitAsync(TimeSpan.FromSeconds(5));
        Assert.Equal(adapter.Capability.SamplingRecipe, adapter.Generated!.SamplingRecipe);
        await service.CancelJobAsync(job.Id);
    }

    [Fact]
    public void RequestsWithoutARecipeSerializeAsTheyDidBeforeRecipes()
    {
        // Saved jobs, drafts and takes, and every fingerprint, are byte-identical unless a recipe is named.
        var legacy = new StudioGenerationRequest { ProjectId = "p", RequestId = "r", Prompt = "A kite.", Steps = 20 };
        string json = JsonSerializer.Serialize(legacy, StudioJson.Options);
        Assert.DoesNotContain("samplingRecipe", json);
        Assert.Null(JsonSerializer.Deserialize<StudioGenerationRequest>(json, StudioJson.Options)!.SamplingRecipe);
        legacy.SamplingRecipe = TurboRecipe;
        var bound = JsonNode.Parse(JsonSerializer.Serialize(legacy, StudioJson.Options))!;
        Assert.Equal(TurboRecipe, (string?)bound["samplingRecipe"]);
        Assert.Equal(TurboRecipe, StudioJson.Clone(legacy).SamplingRecipe);

        // A capability names its recipe for the page; one from a host without recipes reads as the model's own schedule.
        var advertised = JsonNode.Parse(JsonSerializer.Serialize(new StudioModelCapability(), StudioJson.Options))!;
        Assert.Equal("native", (string?)advertised["samplingRecipe"]);
        Assert.Null(advertised["samplingRecipeName"]);
        var older = JsonSerializer.Deserialize<StudioModelCapability>("""{"modelId":"h3","available":true}""", StudioJson.Options)!;
        Assert.Equal(StudioModelCapability.NativeSamplingRecipe, older.SamplingRecipe);
        Assert.Null(older.SamplingRecipeName);
    }

    [SkippableFact]
    public async Task APlanStartsUnderTheLoadedRecipeAndStaysAPlan()
    {
        var adapter = new RecipeAdapter().LoadTurbo();
        await using var service = new StudioService(root, adapter); await RequireMediaAsync(service);
        var project = await service.CreateProjectAsync();
        StudioGenerationRequest Planned(int steps = 0, string? recipe = null) =>
            new() { Prompt = "A kite rises over the dunes.", Width = 64, Height = 64, FrameCount = 17, Steps = steps, SamplingRecipe = recipe };
        var draft = await service.CreateGenerationDraftAsync(project.Id, project.Revision, Planned(), "A fast shot.");
        Assert.Null(draft.Request.SamplingRecipe);

        var job = await service.StartGenerationDraftAsync(project.Id, draft.Id, project.Revision, "h3");
        await adapter.Started.Task.WaitAsync(TimeSpan.FromSeconds(5));
        Assert.Equal(8, adapter.Generated!.Steps); Assert.Equal(TurboRecipe, adapter.Generated.SamplingRecipe);
        // The saved plan keeps its own (unset) choices for a later model; starting it again is the same job.
        var saved = Assert.Single(await service.ListGenerationDraftsAsync(project.Id));
        Assert.Equal(job.Id, saved.JobId); Assert.Equal(0, saved.Request.Steps); Assert.Null(saved.Request.SamplingRecipe);
        Assert.Equal(job.Id, (await service.StartGenerationDraftAsync(project.Id, draft.Id, project.Revision, "h3")).Id);

        // A plan that asks for another step count, or was planned for another recipe, is refused before a job exists.
        var slow = await service.CreateGenerationDraftAsync(project.Id, project.Revision, Planned(steps: 20), "A slow shot.");
        var refused = await Assert.ThrowsAsync<StudioException>(() => service.StartGenerationDraftAsync(project.Id, slow.Id, project.Revision, "h3"));
        Assert.Equal("invalid_sampling", refused.Code); Assert.Equal(TurboStepsRefusal, refused.Message);
        var native = await service.CreateGenerationDraftAsync(project.Id, project.Revision, Planned(recipe: "native"), "A base shot.");
        var conflict = await Assert.ThrowsAsync<StudioException>(() => service.StartGenerationDraftAsync(project.Id, native.Id, project.Revision, "h3"));
        Assert.Equal("sampling_recipe_conflict", conflict.Code);
        Assert.Single(await service.ListJobsAsync(project.Id)); Assert.Equal(1, adapter.Calls);
        await service.CancelJobAsync(job.Id);
    }
}
