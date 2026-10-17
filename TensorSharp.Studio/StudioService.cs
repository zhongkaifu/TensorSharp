// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Security.Cryptography;
using System.Text.Json;

namespace TensorSharp.Studio;

/// <summary>Single authoritative project owner. The browser submits revisions and commands; it never saves its own project copy.</summary>
public sealed partial class StudioService : IAsyncDisposable
{
    public string RootPath { get; }
    public StudioOptions Options { get; }
    public StudioMedia Media { get; }
    readonly IStudioGenerationAdapter? generation;
    internal readonly SemaphoreSlim projectGate = new(1, 1);
    readonly SemaphoreSlim importGate = new(2, 2);
    readonly FileStream ownerLease;

    public StudioService(string rootPath, IStudioGenerationAdapter? generation = null, StudioOptions? options = null)
    {
        RootPath = Path.GetFullPath(rootPath); Options = options ?? new(); this.generation = generation;
        if (Options.MaxQueuedJobs is < 1 or > 100 || Options.MaxHistory is < 1 or > 100 || Options.MaxImportBytes <= 0)
            throw new ArgumentException("Invalid Studio limits.", nameof(options));
        Directory.CreateDirectory(RootPath);
        try { ownerLease = new FileStream(Path.Combine(RootPath, ".owner"), FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None); }
        catch (IOException) { throw new StudioException("storage_in_use", "This Studio storage is already owned by another active host."); }
        Directory.CreateDirectory(StudioPaths.SafeChild(RootPath, "projects"));
        Directory.CreateDirectory(StudioPaths.SafeChild(RootPath, "jobs"));
        Media = new(Options); InitializeJobs();
    }

    internal string ProjectDirectory(string projectId) => StudioPaths.SafeChild(RootPath, "projects", StudioPaths.Id(projectId));
    internal StudioDocument LoadDocument(string projectId)
    {
        var doc = StudioStorage.Read<StudioDocument>(StudioPaths.SafeChild(ProjectDirectory(projectId), "project.json"));
        if (doc.Project.Id != projectId) throw new StudioException("invalid_document", "Project identity does not match its storage directory.");
        StudioCommands.Validate(doc.Project, Options); return doc;
    }
    internal void SaveDocument(StudioDocument doc) => StudioStorage.Save(StudioPaths.SafeChild(ProjectDirectory(doc.Project.Id), "project.json"), doc);
    static void CheckRevision(StudioProject p, long expected)
    {
        if (p.Revision != expected) throw new StudioException("revision_conflict", $"Project changed: expected revision {expected}, current revision is {p.Revision}. Refresh and revalidate the edit.");
    }
    static void Touch(StudioProject p) { p.Revision = checked(p.Revision + 1); p.UpdatedAt = DateTimeOffset.UtcNow; }
    static string Hash<T>(T value) => Convert.ToHexStringLower(SHA256.HashData(JsonSerializer.SerializeToUtf8Bytes(value, StudioJson.Options)));
    static bool AlreadyApplied(StudioDocument doc, string commandId, string fingerprint)
    {
        StudioPaths.Id(commandId);
        if (!doc.AppliedCommands.TryGetValue(commandId, out var previous)) return false;
        if (previous != fingerprint) throw new StudioException("idempotency_conflict", "This command identifier was already used for a different request.");
        return true;
    }
    void RecordEdit(StudioDocument doc, StudioProject previous, string commandId, string fingerprint)
    {
        RecordAssetHistory(doc, previous, doc.Project);
        previous.Assets = []; // Imports/generations survive unrelated timeline undo.
        doc.Undo.Add(previous);
        if (doc.Undo.Count > Options.MaxHistory) { doc.AssetHistory.Remove(doc.Undo[0].Revision); doc.Undo.RemoveAt(0); }
        foreach (var entry in doc.Redo) doc.AssetHistory.Remove(entry.Revision);
        doc.Redo.Clear(); doc.AppliedCommands.Add(commandId, fingerprint);
        // Reject admission rather than evicting old keys and accidentally replaying an old request.
        if (doc.AppliedCommands.Count > 100_000) throw new StudioException("command_limit", "Project command ledger is full; export/import as a new project to continue.");
        Touch(doc.Project); SaveDocument(doc);
    }
    static void RecordAssetHistory(StudioDocument doc, StudioProject target, StudioProject current)
    {
        var currentIds = current.Assets.Select(a => a.Id).ToHashSet(StringComparer.Ordinal);
        var targetIds = target.Assets.Select(a => a.Id).ToHashSet(StringComparer.Ordinal);
        var restore = target.Assets.Select((asset, index) => new StudioAssetRestore { Asset = asset, Index = index }).Where(a => !currentIds.Contains(a.Asset.Id)).ToList();
        var remove = current.Assets.Where(a => !targetIds.Contains(a.Id)).Select(a => a.Id).ToList();
        if (restore.Count > 0 || remove.Count > 0)
            doc.AssetHistory[target.Revision] = new() { Restore = restore, Remove = remove };
    }
    StudioProject PublicProject(StudioProject project)
    {
        var result = StudioJson.Clone(project);
        foreach (var asset in result.Assets) asset.Missing = !File.Exists(StudioPaths.SafeChild(ProjectDirectory(result.Id), "assets", asset.FileName));
        return result;
    }
    internal void CheckDisk()
    {
        var root = Path.GetPathRoot(RootPath)!;
        try { if (new DriveInfo(root).AvailableFreeSpace < Options.MinimumFreeDiskBytes) throw new StudioException("insufficient_disk", "Free disk space is below the Studio safety reserve. Free space before importing or starting a job."); }
        catch (ArgumentException) { /* Some virtual platforms do not expose a drive; writes still report OS errors. */ }
    }

    public async Task<StudioProject> CreateProjectAsync(string name = "Untitled video", int width = 1280, int height = 720, int fpsNumerator = 24, int fpsDenominator = 1, string? conversationId = null, CancellationToken cancellationToken = default)
    {
        await projectGate.WaitAsync(cancellationToken);
        try
        {
            CheckDisk();
            var project = new StudioProject { Name = name, Width = width, Height = height, FrameRate = new(fpsNumerator, fpsDenominator), ConversationId = conversationId,
                Tracks = [new() { Name = "Video 1", Kind = "video" }, new() { Name = "Audio 1", Kind = "audio" }] };
            StudioCommands.Validate(project, Options); SaveDocument(new() { Project = project }); return PublicProject(project);
        }
        finally { projectGate.Release(); }
    }
    public async Task<StudioProject> GetProjectAsync(string projectId, CancellationToken cancellationToken = default)
    {
        await projectGate.WaitAsync(cancellationToken);
        try { return PublicProject(LoadDocument(projectId).Project); }
        finally { projectGate.Release(); }
    }
    public async Task<IReadOnlyList<StudioProject>> ListProjectsAsync(CancellationToken cancellationToken = default)
    {
        await projectGate.WaitAsync(cancellationToken);
        try
        {
            var result = new List<StudioProject>();
            foreach (var path in Directory.EnumerateDirectories(StudioPaths.SafeChild(RootPath, "projects")).Take(10000))
            {
                cancellationToken.ThrowIfCancellationRequested();
                if (!File.Exists(Path.Combine(path, "project.json"))) continue;
                result.Add(PublicProject(LoadDocument(Path.GetFileName(path)).Project));
            }
            return result.OrderByDescending(p => p.UpdatedAt).ToArray();
        }
        finally { projectGate.Release(); }
    }
    public async Task<StudioProject> ApplyCommandsAsync(string projectId, long expectedRevision, string commandId, IReadOnlyList<StudioCommand> commands, CancellationToken cancellationToken = default)
    {
        await projectGate.WaitAsync(cancellationToken);
        try
        {
            var doc = LoadDocument(projectId); var fingerprint = Hash(new { kind = "apply", expectedRevision, commands });
            if (AlreadyApplied(doc, commandId, fingerprint)) return PublicProject(doc.Project);
            CheckRevision(doc.Project, expectedRevision); var previous = StudioJson.Clone(doc.Project);
            StudioCommands.Apply(doc.Project, commands, Options); RecordEdit(doc, previous, commandId, fingerprint); return PublicProject(doc.Project);
        }
        finally { projectGate.Release(); }
    }
    public async Task<StudioProject> UndoAsync(string projectId, long expectedRevision, string commandId, bool redo = false, CancellationToken cancellationToken = default)
    {
        await projectGate.WaitAsync(cancellationToken);
        try
        {
            var doc = LoadDocument(projectId); var fingerprint = Hash(new { kind = redo ? "redo" : "undo", expectedRevision });
            if (AlreadyApplied(doc, commandId, fingerprint)) return PublicProject(doc.Project);
            CheckRevision(doc.Project, expectedRevision); var source = redo ? doc.Redo : doc.Undo; var target = redo ? doc.Undo : doc.Redo;
            if (source.Count == 0) throw new StudioException("history_empty", redo ? "Nothing to redo." : "Nothing to undo.");
            var previous = doc.Project; var assets = previous.Assets.ToList();
            doc.Project = source[^1]; source.RemoveAt(source.Count - 1); target.Add(previous);
            // Restore only the asset changes belonging to this transaction, so late imports
            // and completed generations remain present and unrelated undo cannot resurrect deletions.
            if (doc.AssetHistory.Remove(doc.Project.Revision, out var assetHistory))
            {
                assets.RemoveAll(a => assetHistory.Remove.Contains(a.Id, StringComparer.Ordinal));
                foreach (var restored in assetHistory.Restore.OrderBy(a => a.Index))
                    if (assets.All(a => a.Id != restored.Asset.Id)) assets.Insert(Math.Clamp(restored.Index, 0, assets.Count), restored.Asset);
            }
            doc.Project.Assets = assets;
            RecordAssetHistory(doc, previous, doc.Project);
            previous.Assets = [];
            doc.Project.Revision = previous.Revision;
            Touch(doc.Project); doc.AppliedCommands.Add(commandId, fingerprint); StudioCommands.Validate(doc.Project, Options); SaveDocument(doc); return PublicProject(doc.Project);
        }
        finally { projectGate.Release(); }
    }
    public async Task<StudioProposal> ProposeAsync(string projectId, long expectedRevision, IReadOnlyList<StudioCommand> commands, string summary, CancellationToken cancellationToken = default)
    {
        await projectGate.WaitAsync(cancellationToken);
        try
        {
            var doc = LoadDocument(projectId); CheckRevision(doc.Project, expectedRevision);
            StudioCommands.Apply(StudioJson.Clone(doc.Project), commands, Options);
            if (string.IsNullOrWhiteSpace(summary) || summary.Length > 4000) throw new StudioException("invalid_summary", "Proposal summary must contain 1–4000 characters.");
            var proposal = new StudioProposal { ProjectId = projectId, ExpectedRevision = expectedRevision, Commands = StudioJson.Clone(commands.ToList()), Summary = summary };
            doc.Proposals.Add(proposal); if (doc.Proposals.Count > 100) doc.Proposals.RemoveAt(0); SaveDocument(doc); return StudioJson.Clone(proposal);
        }
        finally { projectGate.Release(); }
    }
    public async Task<IReadOnlyList<StudioProposal>> ListProposalsAsync(string projectId, CancellationToken cancellationToken = default)
    {
        await projectGate.WaitAsync(cancellationToken);
        try { return StudioJson.Clone(LoadDocument(projectId).Proposals); }
        finally { projectGate.Release(); }
    }
    public async Task<StudioProject> ApplyProposalAsync(string projectId, string proposalId, string commandId, CancellationToken cancellationToken = default)
    {
        StudioProposal proposal;
        await projectGate.WaitAsync(cancellationToken);
        try { proposal = LoadDocument(projectId).Proposals.FirstOrDefault(p => p.Id == StudioPaths.Id(proposalId)) ?? throw new StudioException("proposal_not_found", "Proposal is no longer available."); }
        finally { projectGate.Release(); }
        return await ApplyCommandsAsync(projectId, proposal.ExpectedRevision, commandId, proposal.Commands, cancellationToken);
    }

    public string GetAssetPath(string projectId, string assetId)
    {
        return GetSnapshotAssetPath(LoadDocument(projectId).Project, assetId);
    }
    // Captured jobs/packages own an immutable, validated project snapshot. Only internal callers
    // may resolve its retained media; public media access still requires current library membership.
    string GetSnapshotAssetPath(StudioProject snapshot, string assetId)
    {
        var asset = snapshot.Assets.FirstOrDefault(a => a.Id == StudioPaths.Id(assetId)) ?? throw new StudioException("asset_not_found", "Asset does not belong to this project.");
        var path = StudioPaths.SafeChild(ProjectDirectory(snapshot.Id), "assets", asset.FileName);
        if (!File.Exists(path)) throw new StudioException("missing_media", $"Media '{asset.Name}' is missing. Relink the original media to continue.");
        return path;
    }
    public async Task<StudioProject> ImportAssetAsync(string projectId, Stream input, string fileName, CancellationToken cancellationToken = default)
    {
        await importGate.WaitAsync(cancellationToken);
        try { return await ImportAssetInternalAsync(projectId, input, fileName, null, cancellationToken); }
        finally { importGate.Release(); }
    }

    async Task<StudioProject> ImportAssetInternalAsync(string projectId, Stream input, string fileName, StudioGenerationProvenance? provenance, CancellationToken cancellationToken)
    {
        StudioPaths.Id(projectId); LoadDocument(projectId); CheckDisk();
        var extension = Path.GetExtension(fileName).ToLowerInvariant(); StudioMedia.ContentType(extension);
        var displayName = Path.GetFileName(fileName).Replace('\\', '_'); if (displayName.Length > 200) displayName = displayName[..200];
        var directory = StudioPaths.SafeChild(ProjectDirectory(projectId), "assets"); Directory.CreateDirectory(directory);
        var target = StudioPaths.SafeChild(directory, StudioJson.NewId() + extension); var keep = false;
        try
        {
            string hash;
            using (var sha = IncrementalHash.CreateHash(HashAlgorithmName.SHA256))
            {
                await using var output = new FileStream(target, FileMode.CreateNew, FileAccess.Write, FileShare.None, 81920, true);
                var buffer = new byte[81920]; long count = 0;
                while (true)
                {
                    var read = await input.ReadAsync(buffer, cancellationToken); if (read == 0) break;
                    count = checked(count + read); if (count > Options.MaxImportBytes) throw new StudioException("import_too_large", "Media exceeds the configured import size limit.");
                    if (count % (64L * 1024 * 1024) < read) CheckDisk();
                    sha.AppendData(buffer, 0, read); await output.WriteAsync(buffer.AsMemory(0, read), cancellationToken);
                }
                if (count == 0) throw new StudioException("empty_media", "The media file is empty.");
                await output.FlushAsync(cancellationToken); output.Flush(true); hash = Convert.ToHexStringLower(sha.GetHashAndReset());
            }
            var asset = await Media.ProbeAsync(target, displayName, cancellationToken); asset.Sha256 = hash; asset.Generation = provenance;
            if (provenance is { } generated && (asset.Kind != "video" || asset.Width != generated.Request.Width || asset.Height != generated.Request.Height ||
                Math.Abs(asset.DurationSeconds - (double)generated.Request.FrameCount / generated.Request.Fps) > 2d / generated.Request.Fps + .01))
                throw new StudioException("generation_output_mismatch", "Decoded generation output does not match its requested dimensions, duration, or media type.");
            await projectGate.WaitAsync(cancellationToken);
            try
            {
                var doc = LoadDocument(projectId);
                if (provenance == null && doc.Project.Assets.Any(a => a.Sha256 == hash && File.Exists(StudioPaths.SafeChild(directory, a.FileName)))) return PublicProject(doc.Project);
                doc.Project.Assets.Add(asset); Touch(doc.Project); StudioCommands.Validate(doc.Project, Options); SaveDocument(doc); keep = true; return PublicProject(doc.Project);
            }
            finally { projectGate.Release(); }
        }
        finally { if (!keep && File.Exists(target)) File.Delete(target); }
    }

    /// <summary>Relinking must match the original cryptographic hash, so edits and captured render revisions cannot change meaning.</summary>
    public async Task<StudioProject> RelinkAssetAsync(string projectId, string assetId, Stream input, CancellationToken cancellationToken = default)
    {
        await projectGate.WaitAsync(cancellationToken);
        try
        {
            var doc = LoadDocument(projectId); var asset = doc.Project.Assets.FirstOrDefault(a => a.Id == StudioPaths.Id(assetId)) ?? throw new StudioException("asset_not_found", "Asset does not belong to this project.");
            var destination = StudioPaths.SafeChild(ProjectDirectory(projectId), "assets", asset.FileName); Directory.CreateDirectory(Path.GetDirectoryName(destination)!);
            var temporary = destination + "." + StudioJson.NewId() + ".tmp";
            try
            {
                await using (var output = File.Create(temporary))
                {
                    var buffer = new byte[81920]; long total = 0;
                    while (true)
                    {
                        int n = await input.ReadAsync(buffer, cancellationToken); if (n == 0) break;
                        total += n; if (total > Options.MaxImportBytes || total > asset.Bytes) throw new StudioException("relink_mismatch", "Relinked media does not match the original file.");
                        await output.WriteAsync(buffer.AsMemory(0, n), cancellationToken);
                    }
                }
                await using (var verify = File.OpenRead(temporary))
                    if (Convert.ToHexStringLower(await SHA256.HashDataAsync(verify, cancellationToken)) != asset.Sha256)
                        throw new StudioException("relink_mismatch", "Relinked media must have the same SHA-256 as the original file.");
                File.Move(temporary, destination, true); return PublicProject(doc.Project);
            }
            finally { if (File.Exists(temporary)) File.Delete(temporary); }
        }
        finally { projectGate.Release(); }
    }

    public async Task<object> GetCapabilitiesAsync(CancellationToken cancellationToken = default)
    {
        var render = await Media.GetCapabilitiesAsync(cancellationToken);
        var generationMediaReason = GenerationMediaUnavailableReason(render);
        var providerModels = generation == null ? [] : await generation.GetCapabilitiesAsync(cancellationToken);
        // Capabilities describe the complete Studio workflow, not inference alone. Keep family
        // metadata visible while disabling expensive work whose output cannot be safely ingested.
        // Never mutate a provider's shared capability instances.
        var models = providerModels.Select(StudioJson.Clone).ToArray();
        if (generationMediaReason != null)
            foreach (var model in models)
            {
                model.Available = false;
                model.UnavailableReason = string.IsNullOrWhiteSpace(model.UnavailableReason) ? generationMediaReason : model.UnavailableReason + " " + generationMediaReason;
            }
        return new { models, renderAvailable = render.Available, renderUnavailableReason = render.Reason,
            importAvailable = Options.EnableExternalProcesses && render.ProbeAvailable, captionAvailable = true,
            captionCharacterSet = "Basic Latin (rendered uppercase)", exportCodec = render.Codec,
            externalProcesses = Options.EnableExternalProcesses, packageSupported = true,
            transcriptionAvailable = false, transcriptionUnavailableReason = "No timestamped transcription adapter is connected to Studio.",
            generationAvailable = models.Any(m => m.Available), generationUnavailableReason = generationMediaReason,
            timelineTimebase = "integer frames at project rational frame rate; source presentation timestamps resampled to that rate" };
    }

    string? GenerationMediaUnavailableReason(StudioRenderCapabilities media)
    {
        if (Options.EnableExternalProcesses && media.ProbeAvailable && media.Available) return null;
        return "Studio generation requires local ffprobe and FFmpeg with H.264/AAC encoding to validate, preview, and export its output before native inference can start. " +
            (Options.EnableExternalProcesses ? media.Reason : "This host does not permit the required media subprocesses; project review and editing remain available.");
    }

    public async ValueTask DisposeAsync()
    {
        await StopJobsAsync(); ownerLease.Dispose(); projectGate.Dispose(); importGate.Dispose();
    }
}
