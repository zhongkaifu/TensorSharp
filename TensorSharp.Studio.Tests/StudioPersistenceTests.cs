using System.Text.Json;
using TensorSharp.Studio;

namespace TensorSharp.Studio.Tests;

public sealed class StudioPersistenceTests : IDisposable
{
    readonly string root = Path.Combine(Path.GetTempPath(), "studio-tests-" + StudioJson.NewId());
    StudioService Service() => new(root, options: new() { MinimumFreeDiskBytes = 0 });
    public void Dispose() { if (Directory.Exists(root)) Directory.Delete(root, true); }

    [Fact] public async Task AtomicBatchesRevisionChecksAndIdempotencyPersistAcrossRestart()
    {
        string id; string commandId = StudioJson.NewId();
        var commands = new List<StudioCommand> { new() { Type = "settings", Name = "Saved title" }, new() { Type = "addTrack", Name = "Overlay" } };
        await using (var service = Service())
        {
            var p = await service.CreateProjectAsync(); id = p.Id;
            p = await service.ApplyCommandsAsync(id, 0, commandId, commands); Assert.Equal(1, p.Revision); Assert.Equal(3, p.Tracks.Count);
            var replay = await service.ApplyCommandsAsync(id, 0, commandId, commands); Assert.Equal(1, replay.Revision);
            Assert.Equal("idempotency_conflict", (await Assert.ThrowsAsync<StudioException>(() => service.ApplyCommandsAsync(id, 0, commandId, [new() { Type = "settings", Name = "Different" }]))).Code);
            Assert.Equal("revision_conflict", (await Assert.ThrowsAsync<StudioException>(() => service.ApplyCommandsAsync(id, 0, StudioJson.NewId(), commands))).Code);
            await Assert.ThrowsAsync<StudioException>(() => service.ApplyCommandsAsync(id, 1, StudioJson.NewId(), [new() { Type = "settings", Name = "Must roll back" }, new() { Type = "removeClip", ClipId = StudioJson.NewId() }]));
            Assert.Equal("Saved title", (await service.GetProjectAsync(id)).Name);
        }
        await using (var service = Service())
        {
            var p = await service.GetProjectAsync(id); Assert.Equal("Saved title", p.Name);
            p = await service.ApplyCommandsAsync(id, 0, commandId, commands); Assert.Equal(1, p.Revision);
            p = await service.UndoAsync(id, 1, StudioJson.NewId()); Assert.Equal(2, p.Revision); Assert.Equal("Untitled video", p.Name); Assert.Equal(2, p.Tracks.Count);
            p = await service.UndoAsync(id, 2, StudioJson.NewId(), redo: true); Assert.Equal(3, p.Revision); Assert.Equal("Saved title", p.Name);
        }
        Assert.Empty(Directory.EnumerateFiles(root, "*.tmp", SearchOption.AllDirectories));
    }
    [Fact] public async Task StaleAgentProposalCannotOverwriteManualEdits()
    {
        await using var service = Service(); var p = await service.CreateProjectAsync();
        var proposal = await service.ProposeAsync(p.Id, 0, [new() { Type = "settings", Name = "Agent title" }], "Rename project");
        p = await service.ApplyCommandsAsync(p.Id, 0, StudioJson.NewId(), [new() { Type = "settings", Name = "Manual title" }]);
        Assert.Equal("revision_conflict", (await Assert.ThrowsAsync<StudioException>(() => service.ApplyProposalAsync(p.Id, proposal.Id, StudioJson.NewId()))).Code);
        Assert.Equal("Manual title", (await service.GetProjectAsync(p.Id)).Name);
        var fresh = await service.ProposeAsync(p.Id, 1, [new() { Type = "settings", Name = "Accepted" }, new() { Type = "addTrack", Name = "Agent track" }], "Rename and add a track");
        p = await service.ApplyProposalAsync(p.Id, fresh.Id, StudioJson.NewId()); Assert.Equal(3, p.Tracks.Count);
        p = await service.UndoAsync(p.Id, p.Revision, StudioJson.NewId()); Assert.Equal(2, p.Tracks.Count); Assert.Equal("Manual title", p.Name);
    }
    [Fact] public async Task RemovingATrackIsOneUndoStepThatRestoresItsPlace()
    {
        await using var service = Service(); var p = await service.CreateProjectAsync(); var order = p.Tracks.Select(t => t.Id).ToArray();
        p = await service.ApplyCommandsAsync(p.Id, 0, StudioJson.NewId(), [new() { Type = "removeTrack", TrackId = order[0] }]);
        Assert.Equal(new[] { order[1] }, p.Tracks.Select(t => t.Id));
        p = await service.UndoAsync(p.Id, p.Revision, StudioJson.NewId()); Assert.Equal(order, p.Tracks.Select(t => t.Id));
        p = await service.UndoAsync(p.Id, p.Revision, StudioJson.NewId(), redo: true); Assert.Equal(new[] { order[1] }, p.Tracks.Select(t => t.Id));
    }
    [Fact] public async Task SingleWriterAndPathIsolationAreEnforced()
    {
        await using var service = Service(); var p = await service.CreateProjectAsync();
        Assert.Equal("storage_in_use", Assert.Throws<StudioException>(() => Service()).Code);
        Assert.Equal("invalid_id", (await Assert.ThrowsAsync<StudioException>(() => service.GetProjectAsync("../other"))).Code);
        Assert.Equal("asset_not_found", Assert.Throws<StudioException>(() => service.GetAssetPath(p.Id, StudioJson.NewId())).Code);
    }
    [Fact] public async Task ApiRejectsUnknownGenerationFieldsAndMissingRevision()
    {
        await using var service = Service(); var api = new StudioApi(service);
        using var unknown = JsonDocument.Parse("{\"prompt\":\"hello\",\"arbitraryShell\":\"bad\"}");
        Assert.Equal("invalid_request", (await Assert.ThrowsAsync<StudioException>(() => api.DispatchAsync("jobs/generate", unknown.RootElement))).Code);
        using var duplicate = JsonDocument.Parse("{\"prompt\":\"hello\",\"Prompt\":\"replacement\"}");
        Assert.Equal("invalid_request", (await Assert.ThrowsAsync<StudioException>(() => api.DispatchAsync("jobs/generate", duplicate.RootElement))).Code);
        using var missing = JsonDocument.Parse("{\"projectId\":\"" + StudioJson.NewId() + "\",\"requestId\":\"" + StudioJson.NewId() + "\"}");
        Assert.Equal("missing_parameter", (await Assert.ThrowsAsync<StudioException>(() => api.DispatchAsync("jobs/render", missing.RootElement))).Code);
    }
    [Fact] public async Task ConversationBindingUsesTheSameRevisionAndUndoBoundary()
    {
        await using var service = Service(); var p = await service.CreateProjectAsync();
        p = await service.ApplyCommandsAsync(p.Id, p.Revision, StudioJson.NewId(), [new() { Type = "settings", ConversationId = "saved-conversation" }]);
        Assert.Equal("saved-conversation", p.ConversationId);
        Assert.Equal("revision_conflict", (await Assert.ThrowsAsync<StudioException>(() => service.ApplyCommandsAsync(p.Id, 0, StudioJson.NewId(), [new() { Type = "settings", ConversationId = "wrong-stale-conversation" }]))).Code);
        p = await service.UndoAsync(p.Id, p.Revision, StudioJson.NewId()); Assert.Null(p.ConversationId);
        await Assert.ThrowsAsync<StudioException>(() => service.ApplyCommandsAsync(p.Id, p.Revision, StudioJson.NewId(), [new() { Type = "settings", ConversationId = new string('x', 201) }]));
    }
    [Fact] public async Task FutureSchemaIsRejectedWithoutRewritingTheDocument()
    {
        string id;
        await using (var service = Service()) id = (await service.CreateProjectAsync()).Id;
        var path = Path.Combine(root, "projects", id, "project.json"); var json = await File.ReadAllTextAsync(path); json = json.Replace("\"schemaVersion\": 1", "\"schemaVersion\": 99"); await File.WriteAllTextAsync(path, json);
        await using var reopened = Service();
        Assert.Equal("unsupported_schema", (await Assert.ThrowsAsync<StudioException>(() => reopened.GetProjectAsync(id))).Code);
        Assert.Equal(json, await File.ReadAllTextAsync(path));
    }
}
