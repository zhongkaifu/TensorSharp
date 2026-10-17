using System.IO.Compression;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Nodes;
using TensorSharp.Studio;

namespace TensorSharp.Studio.Tests;

public sealed class StudioAssetDeletionTests : IDisposable
{
    readonly string root = Path.Combine(Path.GetTempPath(), "studio-asset-delete-" + StudioJson.NewId());
    public void Dispose() { if (Directory.Exists(root)) Directory.Delete(root, true); }
    StudioService Service(bool external = false, int history = 40) => new(root, options: new() { EnableExternalProcesses = external, MinimumFreeDiskBytes = 0, MaxHistory = history });

    // Verified portable-package metadata fixtures exercise offline editing without a decoder.
    static async Task<StudioProject> Project(StudioService service, int count = 2)
    {
        var project = new StudioProject { Tracks = [new() { Name = "Video" }, new() { Name = "Audio", Kind = "audio" }] };
        using var package = new MemoryStream();
        using (var zip = new ZipArchive(package, ZipArchiveMode.Create, true))
        {
            for (int i = 0; i < count; i++)
            {
                byte[] bytes = [(byte)i, 1, 2, 3];
                var asset = new StudioAsset { Name = "Asset " + i, FileName = StudioJson.NewId() + ".mp4", DurationSeconds = 10, HasAudio = true, Bytes = bytes.Length, Sha256 = Convert.ToHexStringLower(SHA256.HashData(bytes)) };
                project.Assets.Add(asset);
                using var stream = zip.CreateEntry("assets/" + asset.FileName).Open(); await stream.WriteAsync(bytes);
            }
            using var metadata = zip.CreateEntry("project.json").Open(); await JsonSerializer.SerializeAsync(metadata, project, StudioJson.Options);
        }
        package.Position = 0; return await service.ImportPackageAsync(package);
    }
    static StudioCommand Delete(string id, bool? removeClips = null) => new StudioCommand { Type = "deleteAsset", AssetId = id, RemoveTimelineClips = removeClips };
    static Task<StudioProject> Edit(StudioService service, StudioProject p, params StudioCommand[] commands) => service.ApplyCommandsAsync(p.Id, p.Revision, StudioJson.NewId(), commands);
    static Task<StudioProject> Undo(StudioService service, StudioProject p, bool redo = false) => service.UndoAsync(p.Id, p.Revision, StudioJson.NewId(), redo);

    [Fact] public async Task UnusedAssetDeletionIsUndoableKeepsFilesAndExcludesTheAssetFromPackages()
    {
        await using var service = Service(); var p = await Project(service); var asset = p.Assets[0]; var other = p.Assets[1];
        var path = service.GetAssetPath(p.Id, asset.Id); var original = await File.ReadAllBytesAsync(path);
        p = await Edit(service, p, Delete(asset.Id));
        Assert.Equal(other.Id, Assert.Single(p.Assets).Id); Assert.Equal(original, await File.ReadAllBytesAsync(path));
        Assert.Equal("asset_not_found", Assert.Throws<StudioException>(() => service.GetAssetPath(p.Id, asset.Id)).Code);
        using (var zip = ZipFile.OpenRead(await service.ExportPackageAsync(p.Id)))
        {
            Assert.Null(zip.GetEntry("assets/" + asset.FileName)); Assert.NotNull(zip.GetEntry("assets/" + other.FileName));
            using var stream = zip.GetEntry("project.json")!.Open(); var exported = await JsonSerializer.DeserializeAsync<StudioProject>(stream, StudioJson.Options);
            Assert.Equal(other.Id, Assert.Single(exported!.Assets).Id);
        }
        p = await Undo(service, p); Assert.Equal(new[] { asset.Id, other.Id }, p.Assets.Select(a => a.Id)); Assert.Equal(path, service.GetAssetPath(p.Id, asset.Id));
        p = await Undo(service, p, true); Assert.Equal(other.Id, Assert.Single(p.Assets).Id); Assert.True(File.Exists(path));
    }

    [Theory]
    [InlineData(null)]
    [InlineData(false)]
    public async Task UsedAssetsRequireExplicitConfirmationAndFailedTransactionsAreAtomic(bool? confirmation)
    {
        await using var service = Service(); var p = await Project(service);
        p = await Edit(service, p, new StudioCommand { Type = "insertClip", AssetId = p.Assets[0].Id, TrackId = p.Tracks[0].Id, DurationFrames = 24 });
        var error = await Assert.ThrowsAsync<StudioException>(() => Edit(service, p, new StudioCommand { Type = "settings", Name = "Must roll back" }, Delete(p.Assets[0].Id, confirmation)));
        Assert.Equal("asset_in_use", error.Code); Assert.Contains("1 timeline clip", error.Message);
        var saved = await service.GetProjectAsync(p.Id); Assert.Equal(p.Revision, saved.Revision); Assert.Equal(p.Name, saved.Name); Assert.Equal(2, saved.Assets.Count); Assert.Single(saved.Clips);
    }

    [Fact] public async Task ConfirmedDeleteRemovesAllUsesOnEveryTrackButPreservesOtherLinkedAssets()
    {
        await using var service = Service(); var p = await Project(service); var asset = p.Assets[0]; var other = p.Assets[1]; var link = StudioJson.NewId();
        p = await Edit(service, p,
            new StudioCommand { Type = "insertClip", AssetId = asset.Id, TrackId = p.Tracks[0].Id, DurationFrames = 24, LinkGroup = link },
            new StudioCommand { Type = "insertClip", AssetId = asset.Id, TrackId = p.Tracks[1].Id, DurationFrames = 24, LinkGroup = link },
            new StudioCommand { Type = "insertClip", AssetId = asset.Id, TrackId = p.Tracks[0].Id, StartFrame = 48, DurationFrames = 24 },
            new StudioCommand { Type = "insertClip", AssetId = other.Id, TrackId = p.Tracks[1].Id, StartFrame = 72, DurationFrames = 24, LinkGroup = link });
        var before = StudioJson.Clone(p); p = await Edit(service, p, Delete(asset.Id, true));
        Assert.Equal(other.Id, Assert.Single(p.Assets).Id); Assert.Equal(other.Id, Assert.Single(p.Clips).AssetId);
        Assert.Equal(72, p.Clips[0].StartFrame); Assert.Equal(link, p.Clips[0].LinkGroup);
        p = await Undo(service, p); Assert.Equal(JsonSerializer.Serialize(before.Clips, StudioJson.Options), JsonSerializer.Serialize(p.Clips, StudioJson.Options));
        Assert.Equal(new[] { asset.Id, other.Id }, p.Assets.Select(a => a.Id));
        p = await Undo(service, p, true); Assert.Equal(other.Id, Assert.Single(p.Clips).AssetId);
    }

    [Fact] public async Task FailedDeleteBatchCannotCommitAnEarlierDeletion()
    {
        await using var service = Service(); var p = await Project(service);
        await Assert.ThrowsAsync<StudioException>(() => Edit(service, p, Delete(p.Assets[0].Id), new StudioCommand { Type = "removeClip", ClipId = StudioJson.NewId() }));
        var saved = await service.GetProjectAsync(p.Id); Assert.Equal(p.Revision, saved.Revision); Assert.Equal(2, saved.Assets.Count);
        Assert.Equal("history_empty", (await Assert.ThrowsAsync<StudioException>(() => Undo(service, saved))).Code);
    }

    [Fact] public async Task DeletionHonorsRevisionsIdempotencyAndProjectIsolation()
    {
        await using var service = Service(); var p = await Project(service); var id = p.Assets[0].Id; var command = StudioJson.NewId(); var revision = p.Revision;
        var other = await service.CreateProjectAsync();
        Assert.Equal("asset_not_found", (await Assert.ThrowsAsync<StudioException>(() => Edit(service, other, Delete(id, true)))).Code);
        Assert.Equal("revision_conflict", (await Assert.ThrowsAsync<StudioException>(() => service.ApplyCommandsAsync(p.Id, revision + 1, command, [Delete(id)]))).Code);
        p = await service.ApplyCommandsAsync(p.Id, revision, command, [Delete(id)]);
        var retry = await service.ApplyCommandsAsync(p.Id, revision, command, [Delete(id)]); Assert.Equal(p.Revision, retry.Revision);
        Assert.Equal("idempotency_conflict", (await Assert.ThrowsAsync<StudioException>(() => service.ApplyCommandsAsync(p.Id, revision, command, [Delete(id, true)]))).Code);
        p = await Undo(service, p); retry = await service.ApplyCommandsAsync(p.Id, revision, command, [Delete(id)]);
        Assert.Equal(p.Revision, retry.Revision); Assert.Contains(retry.Assets, a => a.Id == id); // A transport retry never replays a command that was undone.
    }

    [Fact] public async Task MissingAssetsCanBeDeletedAndUndoPreservesTheMissingState()
    {
        await using var service = Service(); var p = await Project(service, 1); var id = p.Assets[0].Id;
        File.Delete(service.GetAssetPath(p.Id, id)); p = await service.GetProjectAsync(p.Id); Assert.True(p.Assets[0].Missing);
        p = await Edit(service, p, Delete(id)); Assert.Empty(p.Assets);
        p = await Undo(service, p); Assert.True(Assert.Single(p.Assets).Missing);
    }

    [Fact] public async Task UnrelatedUndoDoesNotResurrectAssetsAndDeletionSurvivesRestart()
    {
        string projectId, assetId, otherId;
        await using (var service = Service())
        {
            var p = await Project(service); projectId = p.Id; assetId = p.Assets[0].Id; otherId = p.Assets[1].Id;
            p = await Edit(service, p, Delete(assetId)); p = await Edit(service, p, new StudioCommand { Type = "settings", Name = "A later edit" });
            p = await Undo(service, p); Assert.Equal(otherId, Assert.Single(p.Assets).Id);
        }
        await using (var service = Service())
        {
            var p = await service.GetProjectAsync(projectId); Assert.Equal(otherId, Assert.Single(p.Assets).Id);
            p = await Undo(service, p); Assert.Equal(new[] { assetId, otherId }, p.Assets.Select(a => a.Id));
        }
        await using (var service = Service())
        {
            var p = await service.GetProjectAsync(projectId); p = await Undo(service, p, true); Assert.Equal(otherId, Assert.Single(p.Assets).Id);
            p = await Undo(service, p, true); Assert.Equal("A later edit", p.Name); Assert.Equal(otherId, Assert.Single(p.Assets).Id);
        }
    }

    [Fact] public async Task LegacyEmptyAssetHistoryStillRestoresTimelineWithoutChangingLibrary()
    {
        string id, assetId;
        await using (var service = Service())
        {
            var p = await Project(service); id = p.Id; assetId = p.Assets[0].Id;
            await Edit(service, p, new StudioCommand { Type = "insertClip", AssetId = assetId, TrackId = p.Tracks[0].Id, DurationFrames = 24 });
        }
        var path = Path.Combine(root, "projects", id, "project.json"); var document = JsonNode.Parse(await File.ReadAllTextAsync(path))!;
        document.AsObject().Remove("assetHistory"); Assert.Empty(document["undo"]![0]!["assets"]!.AsArray());
        await File.WriteAllTextAsync(path, document.ToJsonString());
        await using var reopened = Service(); var project = await reopened.GetProjectAsync(id);
        project = await Edit(reopened, project, Delete(assetId, true)); project = await Undo(reopened, project); Assert.Single(project.Clips);
        project = await Undo(reopened, project); Assert.Empty(project.Clips); Assert.Equal(2, project.Assets.Count);
        project = await Undo(reopened, project, true); Assert.Single(project.Clips); Assert.Equal(2, project.Assets.Count);
    }

    [Fact] public async Task AssetHistoryIsPrunedWithItsUndoAndRedoEntries()
    {
        await using var service = Service(history: 1); var p = await Project(service); p = await Edit(service, p, Delete(p.Assets[0].Id));
        p = await Edit(service, p, new StudioCommand { Type = "settings", Name = "Discard old undo" });
        var path = Path.Combine(root, "projects", p.Id, "project.json"); var doc = JsonNode.Parse(await File.ReadAllTextAsync(path))!;
        Assert.Empty(doc["assetHistory"]!.AsObject());
        p = await Edit(service, p, Delete(p.Assets[0].Id)); p = await Undo(service, p);
        p = await Edit(service, p, new StudioCommand { Type = "settings", Name = "Discard redo" });
        doc = JsonNode.Parse(await File.ReadAllTextAsync(path))!; Assert.Empty(doc["assetHistory"]!.AsObject());
    }

    [Fact] public async Task DeleteCommandSupportsSharedApiAndRejectsIgnoredParameters()
    {
        await using var service = Service(); var p = await Project(service); var api = new StudioApi(service);
        Assert.Equal("unsupported_parameter", (await Assert.ThrowsAsync<StudioException>(() => Edit(service, p, new StudioCommand { Type = "deleteAsset", AssetId = p.Assets[0].Id, StartFrame = 0 }))).Code);
        Assert.Equal("unsupported_parameter", (await Assert.ThrowsAsync<StudioException>(() => Edit(service, p, new StudioCommand { Type = "settings", Name = "No", RemoveTimelineClips = true }))).Code);
        var result = await api.DispatchAsync("commands/apply", JsonSerializer.SerializeToElement(new { projectId = p.Id, expectedRevision = p.Revision, commandId = StudioJson.NewId(), commands = new[] { Delete(p.Assets[0].Id, true) } }, StudioJson.Options));
        Assert.Single(result.GetProperty("assets").EnumerateArray());
    }

    [Fact] public void NullableConfirmationDoesNotChangeExistingCommandFingerprints()
    {
        var json = JsonSerializer.Serialize(new StudioCommand { Type = "settings", Name = "old request" }, StudioJson.Options);
        Assert.DoesNotContain("removeTimelineClips", json); // Existing persisted request hashes predate this optional field.
        Assert.Contains("\"removeTimelineClips\": true", JsonSerializer.Serialize(Delete(StudioJson.NewId(), true), StudioJson.Options));
    }

    [StudioMediaFact] public async Task ImportsAfterDeletionOrUndoRemainVisibleAcrossUndoRedoAndRestart()
    {
        // Use real deterministic WAV imports to exercise non-undoable asset arrivals through the normal API.
        static MemoryStream Wave(short sample)
        {
            var stream = new MemoryStream(); using (var writer = new BinaryWriter(stream, System.Text.Encoding.UTF8, true))
            {
                writer.Write("RIFF"u8); writer.Write(36 + 1600); writer.Write("WAVEfmt "u8); writer.Write(16); writer.Write((short)1); writer.Write((short)1);
                writer.Write(8000); writer.Write(16000); writer.Write((short)2); writer.Write((short)16); writer.Write("data"u8); writer.Write(1600);
                for (int i = 0; i < 800; i++) writer.Write(sample);
            }
            stream.Position = 0; return stream;
        }
        string id, deletedId, secondId, thirdId;
        await using (var service = Service(true))
        {
            var p = await service.CreateProjectAsync(); id = p.Id;
            using (var input = Wave(1)) p = await service.ImportAssetAsync(id, input, "original.wav"); deletedId = p.Assets.Single().Id;
            p = await Edit(service, p, Delete(deletedId));
            using (var input = Wave(2)) p = await service.ImportAssetAsync(id, input, "arrived-after-delete.wav"); secondId = p.Assets.Single().Id;
            p = await Undo(service, p); Assert.Equal(new[] { deletedId, secondId }, p.Assets.Select(a => a.Id));
            using (var input = Wave(3)) p = await service.ImportAssetAsync(id, input, "arrived-after-undo.wav"); thirdId = p.Assets.Last().Id;
        }
        await using var reopened = Service(true); var saved = await reopened.GetProjectAsync(id);
        saved = await Undo(reopened, saved, true); Assert.Equal(new[] { secondId, thirdId }, saved.Assets.Select(a => a.Id));
        saved = await Undo(reopened, saved); Assert.Equal(new[] { deletedId, secondId, thirdId }, saved.Assets.Select(a => a.Id));
    }
}
