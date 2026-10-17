// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Text.Json;
using System.Text.Json.Serialization;

namespace TensorSharp.Studio;

/// <summary>One host-independent dispatch surface for HTTP, WebView, and structured agent tools.</summary>
public sealed class StudioApi(StudioService service)
{
    static readonly JsonSerializerOptions StrictJson = new(StudioJson.Options) { UnmappedMemberHandling = JsonUnmappedMemberHandling.Disallow };
    public async Task<JsonElement> DispatchAsync(string action, JsonElement payload, CancellationToken cancellationToken = default)
    {
        if (payload.ValueKind != JsonValueKind.Object) throw new StudioException("invalid_request", "Studio payload must be an object.");
        if (payload.GetRawText().Length > 1024 * 1024) throw new StudioException("request_too_large", "Studio command payload exceeds one MiB.");
        object result;
        switch (action)
        {
            case "capabilities": Fields(payload); result = await service.GetCapabilitiesAsync(cancellationToken); break;
            case "projects/list": Fields(payload); result = await service.ListProjectsAsync(cancellationToken); break;
            case "projects/create":
                Fields(payload, "name", "width", "height", "fpsNumerator", "fpsDenominator", "conversationId");
                result = await service.CreateProjectAsync(Text(payload, "name", "Untitled video"), Int(payload, "width", 1280), Int(payload, "height", 720), Int(payload, "fpsNumerator", 24), Int(payload, "fpsDenominator", 1), OptionalText(payload, "conversationId"), cancellationToken); break;
            case "projects/get": Fields(payload, "projectId"); result = await service.GetProjectAsync(Text(payload, "projectId"), cancellationToken); break;
            case "commands/apply":
                Fields(payload, "projectId", "expectedRevision", "commandId", "commands");
                result = await service.ApplyCommandsAsync(Text(payload, "projectId"), Long(payload, "expectedRevision"), Text(payload, "commandId"), Commands(payload), cancellationToken); break;
            case "commands/undo":
            case "commands/redo":
                Fields(payload, "projectId", "expectedRevision", "commandId");
                result = await service.UndoAsync(Text(payload, "projectId"), Long(payload, "expectedRevision"), Text(payload, "commandId"), action == "commands/redo", cancellationToken); break;
            case "proposals/create":
                Fields(payload, "projectId", "expectedRevision", "commands", "summary");
                result = await service.ProposeAsync(Text(payload, "projectId"), Long(payload, "expectedRevision"), Commands(payload), Text(payload, "summary"), cancellationToken); break;
            case "proposals/list": Fields(payload, "projectId"); result = await service.ListProposalsAsync(Text(payload, "projectId"), cancellationToken); break;
            case "proposals/apply":
                Fields(payload, "projectId", "proposalId", "commandId");
                result = await service.ApplyProposalAsync(Text(payload, "projectId"), Text(payload, "proposalId"), Text(payload, "commandId"), cancellationToken); break;
            case "generation-drafts/create":
                Fields(payload, "projectId", "expectedRevision", "request", "summary");
                result = await service.CreateGenerationDraftAsync(Text(payload, "projectId"), Long(payload, "expectedRevision"), Deserialize<StudioGenerationRequest>(payload.GetProperty("request")), Text(payload, "summary"), cancellationToken); break;
            case "generation-drafts/list":
                Fields(payload, "projectId"); result = await service.ListGenerationDraftsAsync(Text(payload, "projectId"), cancellationToken); break;
            case "generation-drafts/start":
                Fields(payload, "projectId", "draftId", "expectedRevision", "modelId");
                result = await service.StartGenerationDraftAsync(Text(payload, "projectId"), Text(payload, "draftId"), Long(payload, "expectedRevision"), Text(payload, "modelId"), cancellationToken); break;
            case "jobs/list": Fields(payload, "projectId"); result = await service.ListJobsAsync(Text(payload, "projectId"), cancellationToken); break;
            case "jobs/cancel": Fields(payload, "jobId"); result = await service.CancelJobAsync(Text(payload, "jobId"), cancellationToken); break;
            case "jobs/render":
                Fields(payload, "projectId", "expectedRevision", "requestId");
                result = await service.StartRenderAsync(Text(payload, "projectId"), Long(payload, "expectedRevision"), Text(payload, "requestId"), cancellationToken); break;
            case "jobs/generate":
                result = await service.StartGenerationAsync(Deserialize<StudioGenerationRequest>(payload), cancellationToken); break;
            default: throw new StudioException("unknown_action", $"Studio action '{action}' is not supported.");
        }
        return JsonSerializer.SerializeToElement(result, StudioJson.Options);
    }
    static T Deserialize<T>(JsonElement payload)
    {
        RejectDuplicateProperties(payload);
        try { return payload.Deserialize<T>(StrictJson) ?? throw new JsonException("Empty request"); }
        catch (JsonException e) { throw new StudioException("invalid_request", e.Message); }
    }
    static void RejectDuplicateProperties(JsonElement value)
    {
        if (value.ValueKind == JsonValueKind.Object)
        {
            var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            foreach (var property in value.EnumerateObject())
            {
                if (!seen.Add(property.Name)) throw new StudioException("invalid_request", $"Duplicate property '{property.Name}' is not permitted.");
                RejectDuplicateProperties(property.Value);
            }
        }
        else if (value.ValueKind == JsonValueKind.Array)
            foreach (var item in value.EnumerateArray()) RejectDuplicateProperties(item);
    }
    static List<StudioCommand> Commands(JsonElement payload) => payload.TryGetProperty("commands", out var c) ? Deserialize<List<StudioCommand>>(c) : throw new StudioException("missing_parameter", "commands is required.");
    static void Fields(JsonElement payload, params string[] allowed)
    {
        var seen = new HashSet<string>(StringComparer.Ordinal);
        foreach (var property in payload.EnumerateObject())
            if (!seen.Add(property.Name) || !allowed.Contains(property.Name, StringComparer.Ordinal))
                throw new StudioException("invalid_request", $"Unexpected or duplicate property '{property.Name}'. Unsupported parameters are never silently ignored.");
    }
    static string Text(JsonElement payload, string name, string? fallback = null) => OptionalText(payload, name) ?? fallback ?? throw new StudioException("missing_parameter", $"{name} is required.");
    static string? OptionalText(JsonElement payload, string name)
    {
        if (!payload.TryGetProperty(name, out var value) || value.ValueKind == JsonValueKind.Null) return null;
        if (value.ValueKind != JsonValueKind.String) throw new StudioException("invalid_request", $"{name} must be text.");
        return value.GetString();
    }
    static int Int(JsonElement payload, string name, int fallback) => payload.TryGetProperty(name, out var v) ? v.TryGetInt32(out var n) ? n : throw new StudioException("invalid_request", $"{name} must be an integer.") : fallback;
    static long Long(JsonElement payload, string name) => payload.TryGetProperty(name, out var v) && v.TryGetInt64(out var n) ? n : throw new StudioException("missing_parameter", $"{name} must be an integer.");
}
