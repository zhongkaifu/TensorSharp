// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
namespace TensorSharp.Studio;

/// <summary>A reviewable shot request authored while a conversational model is loaded.
/// Starting it requires a deliberate model selection; no model is automatically switched.</summary>
public sealed class StudioGenerationDraft
{
    public string Id { get; set; } = StudioJson.NewId();
    public string ProjectId { get; set; } = "";
    public long ExpectedRevision { get; set; }
    public string Summary { get; set; } = "";
    public StudioGenerationRequest Request { get; set; } = new();
    public string? JobId { get; set; }
    public DateTimeOffset CreatedAt { get; set; } = DateTimeOffset.UtcNow;
}

public sealed partial class StudioService
{
    string DraftDirectory(string projectId) => StudioPaths.SafeChild(ProjectDirectory(projectId), "generation-drafts");
    string DraftPath(string projectId, string draftId) => StudioPaths.SafeChild(DraftDirectory(projectId), StudioPaths.Id(draftId) + ".json");

    public async Task<StudioGenerationDraft> CreateGenerationDraftAsync(string projectId, long expectedRevision,
        StudioGenerationRequest request, string summary, CancellationToken cancellationToken = default)
    {
        if (request == null || request.TemporalConditions == null || request.ReferenceAssetIds == null || request.ProjectId == null
            || request.Prompt == null || summary == null || request.TemporalConditions.Any(condition => condition == null))
            throw new StudioException("invalid_draft", "Generation plans require non-null request fields, conditioning arrays, and a summary.");
        request = StudioJson.Clone(request);
        await projectGate.WaitAsync(cancellationToken);
        try
        {
            var project = LoadDocument(projectId).Project; CheckRevision(project, expectedRevision);
            if (request.ProjectId.Length > 0 && request.ProjectId != projectId)
                throw new StudioException("project_conflict", "A generation plan cannot target another project.");
            request.ProjectId = projectId; StudioPaths.Id(request.RequestId);
            if (string.IsNullOrWhiteSpace(request.Prompt) || request.Prompt.Length > 16000 || summary.Length is < 1 or > 1000)
                throw new StudioException("invalid_draft", "Generation plans require a bounded prompt and short summary.");
            if (request.Width != 0 && request.Width is < 32 or > 2048 || request.Height != 0 && request.Height is < 32 or > 2048 || request.FrameCount is < 0 or > 257 || request.Steps is < 0 or > 100 || request.Fps is < 0 or > 240 || !float.IsFinite(request.CfgScale) || request.CfgScale is < 0 or > 30)
                throw new StudioException("invalid_draft", "Suggested generation settings exceed Studio admission limits.");
            // The mode must match its inputs now; whether a checkpoint runs it is checked when the plan starts.
            _ = StudioGenerationModes.Resolve(request);
            // Assets are capabilities scoped to this project, even before a model can validate their exact combination.
            _ = ResolveConditioning(request, project);
            Directory.CreateDirectory(DraftDirectory(projectId));
            if (Directory.EnumerateFiles(DraftDirectory(projectId), "*.json").Take(101).Count() >= 100)
                throw new StudioException("draft_limit", "This project already contains 100 generation plans.");
            var draft = new StudioGenerationDraft { ProjectId = projectId, ExpectedRevision = expectedRevision, Summary = summary, Request = request };
            StudioStorage.Save(DraftPath(projectId, draft.Id), draft); return draft;
        }
        finally { projectGate.Release(); }
    }

    public async Task<IReadOnlyList<StudioGenerationDraft>> ListGenerationDraftsAsync(string projectId, CancellationToken cancellationToken = default)
    {
        await projectGate.WaitAsync(cancellationToken);
        try
        {
            _ = LoadDocument(projectId);
            if (!Directory.Exists(DraftDirectory(projectId))) return [];
            return Directory.EnumerateFiles(DraftDirectory(projectId), "*.json").Take(100)
                .Select(path => StudioStorage.Read<StudioGenerationDraft>(DraftPath(projectId, Path.GetFileNameWithoutExtension(path))))
                .OrderBy(d => d.CreatedAt).ToArray();
        }
        finally { projectGate.Release(); }
    }

    public async Task<StudioJob> StartGenerationDraftAsync(string projectId, string draftId, long expectedRevision,
        string modelId, CancellationToken cancellationToken = default)
    {
        StudioGenerationDraft draft;
        await projectGate.WaitAsync(cancellationToken);
        try
        {
            CheckRevision(LoadDocument(projectId).Project, expectedRevision);
            draft = StudioStorage.Read<StudioGenerationDraft>(DraftPath(projectId, draftId));
            if (draft.ProjectId != projectId) throw new StudioException("project_conflict", "Generation plan belongs to another project.");
            if (string.IsNullOrWhiteSpace(modelId)) throw new StudioException("model_required", "Explicitly select and load a video model before starting this plan.");
            if (!string.IsNullOrEmpty(draft.Request.ModelId) && draft.Request.ModelId != modelId)
                throw new StudioException("model_conflict", "The plan names another model. Create a new plan to change it explicitly.");
            draft.Request.ModelId = modelId;
        }
        finally { projectGate.Release(); }
        // Admission resolves model defaults and validates the exact checkpoint before queueing.
        // RequestId makes retries idempotent even if the host exits before saving JobId below.
        var job = await StartGenerationAsync(draft.Request, cancellationToken);
        await projectGate.WaitAsync(cancellationToken);
        try { draft.JobId = job.Id; StudioStorage.Save(DraftPath(projectId, draftId), draft); }
        finally { projectGate.Release(); }
        return job;
    }
}
