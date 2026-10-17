using System.IO.Compression;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.RegularExpressions;
using TensorSharp.AgentHost.Skills;
using TensorSharp.Chat.Studio;
using TensorSharp.Runtime;
using TensorSharp.Server;
using TensorSharp.Studio;

namespace InferenceWeb.Tests;

/// <summary>The agent's edit declaration and the command interpreter are one surface; neither may drift from the other.</summary>
public sealed class StudioToolDeclarationTests : IAsyncLifetime
{
    readonly string root = Path.Combine(Path.GetTempPath(), "studio-tool-declaration-" + Guid.NewGuid().ToString("N"));
    StudioService? service;

    public Task InitializeAsync() => Task.CompletedTask;
    public async Task DisposeAsync()
    {
        if (service != null) await service.DisposeAsync();
        if (Directory.Exists(root)) Directory.Delete(root, true);
    }

    [Fact]
    public void StudioProposeDeclaresExactlyTheCommandsTheInterpreterAccepts()
    {
        var propose = StudioTools.Merge(new()).Single(t => t.Name == "studio_propose");
        var declared = Regex.Matches(propose.Description, @"\b([a-z][A-Za-z]*)\(").Select(m => m.Groups[1].Value).ToHashSet(StringComparer.Ordinal);
        Assert.Equal(StudioCommands.CommandTypes.Order(StringComparer.Ordinal), declared.Order(StringComparer.Ordinal));
    }

    [Fact]
    public async Task AgentSeesEachTracksClipCountAndTrackRemovalIsOnlyAProposalUntilAppliedThenOneUndoStep()
    {
        service = new StudioService(Path.Combine(root, "studio"), options: new StudioOptions { EnableExternalProcesses = false, MinimumFreeDiskBytes = 0 });
        var project = await SeededProject(service);
        var video = project.Tracks.Single(t => t.Kind == "video"); var audio = project.Tracks.Single(t => t.Kind == "audio"); var asset = project.Assets.Single().Id;
        project = await service.ApplyCommandsAsync(project.Id, project.Revision, StudioJson.NewId(), [
            new() { Type = "insertClip", TrackId = video.Id, AssetId = asset, DurationFrames = 24 },
            new() { Type = "insertClip", TrackId = audio.Id, AssetId = asset, DurationFrames = 24 },
            new() { Type = "insertClip", TrackId = audio.Id, AssetId = asset, StartFrame = 24, DurationFrames = 24 }]);
        var videoClip = project.Clips.Single(c => c.TrackId == video.Id).Id;
        using var model = new TextModelService();
        var scope = StudioChatScope.Create(service, model, JsonSerializer.SerializeToElement(new
        { studio = new { projectId = project.Id, expectedRevision = project.Revision, selectedClipIds = Array.Empty<string>() } }), root);
        var context = new SkillToolContext(Array.Empty<Skill>()) { Studio = scope };

        // The declaration asks the model to state the removed clip count; inspection is where it learns it.
        var inspected = SkillTools.Execute(new ToolCall { Name = "studio_inspect", Arguments = new() }, context);
        Assert.True(inspected.Ok, inspected.Content);
        using (var inspection = JsonDocument.Parse(inspected.Content))
        {
            var counts = inspection.RootElement.GetProperty("tracks").EnumerateArray().ToDictionary(t => t.GetProperty("id").GetString()!, t => t.GetProperty("clipCount").GetInt32());
            Assert.Equal(1, counts[video.Id]); Assert.Equal(2, counts[audio.Id]);
        }

        var proposal = SkillTools.Execute(new ToolCall { Name = "studio_propose", Arguments = new()
        { ["summary"] = "Remove the audio track and its 2 clips", ["commands"] = "[{\"type\":\"removeTrack\",\"trackId\":\"" + audio.Id + "\"}]" } }, context);
        Assert.True(proposal.Ok, proposal.Content);
        Assert.Equal(3, (await service.GetProjectAsync(project.Id)).Clips.Count); // A proposal is not an approval.

        var saved = JsonSerializer.Deserialize<StudioProposal>(proposal.Content, StudioJson.Options)!;
        var applied = await service.ApplyProposalAsync(project.Id, saved.Id, StudioJson.NewId());
        Assert.Equal(video.Id, Assert.Single(applied.Tracks).Id); Assert.Equal(videoClip, Assert.Single(applied.Clips).Id); Assert.Single(applied.Assets);
        var undone = await service.UndoAsync(project.Id, applied.Revision, StudioJson.NewId());
        Assert.Equal(project.Tracks.Select(t => t.Id), undone.Tracks.Select(t => t.Id));
        Assert.Equal(project.Clips.Select(c => c.Id), undone.Clips.Select(c => c.Id));
    }

    // A portable package carries asset metadata, so the project has media without an ffprobe-backed import.
    static async Task<StudioProject> SeededProject(StudioService service)
    {
        var project = new StudioProject { Name = "Agent track removal", Tracks = [new() { Name = "Video" }, new() { Name = "Audio", Kind = "audio" }] };
        byte[] bytes = [1, 2, 3, 4];
        var asset = new StudioAsset { Name = "Interview", FileName = StudioJson.NewId() + ".mp4", DurationSeconds = 10, HasAudio = true, Bytes = bytes.Length, Sha256 = Convert.ToHexStringLower(SHA256.HashData(bytes)) };
        project.Assets.Add(asset);
        using var package = new MemoryStream();
        using (var zip = new ZipArchive(package, ZipArchiveMode.Create, true))
        {
            await using (var stream = zip.CreateEntry("assets/" + asset.FileName).Open()) await stream.WriteAsync(bytes);
            await using var metadata = zip.CreateEntry("project.json").Open(); await JsonSerializer.SerializeAsync(metadata, project, StudioJson.Options);
        }
        package.Position = 0; return await service.ImportPackageAsync(package);
    }

    sealed class TextModelService : ModelService { public override string Architecture => "qwen35"; }
}
