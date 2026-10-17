// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.IO.Compression;
using System.Security.Cryptography;
using System.Text.Json;

namespace TensorSharp.Studio;

public sealed partial class StudioService
{
    private readonly SemaphoreSlim packageGate = new(1, 1);

    public async Task<string> ExportPackageAsync(string projectId, CancellationToken ct = default)
    {
        await packageGate.WaitAsync(ct).ConfigureAwait(false);
        try { CheckDisk(); return await ExportPackageCoreAsync(projectId, ct).ConfigureAwait(false); }
        finally { packageGate.Release(); }
    }

    private async Task<string> ExportPackageCoreAsync(string projectId, CancellationToken ct)
    {
        var project = await GetProjectAsync(projectId, ct).ConfigureAwait(false);
        project.ConversationId = null; // A portable creative project does not export private chat linkage/history.
        var directory = StudioPaths.SafeChild(RootPath, "packages");
        Directory.CreateDirectory(directory);
        var target = StudioPaths.SafeChild(directory, StudioPaths.Id(projectId) + "-" + project.Revision + ".tsstudio");
        if (File.Exists(target)) return target;
        var temporary = target + "." + StudioJson.NewId() + ".tmp";
        try
        {
            long bytes = project.Assets.Sum(a => a.Bytes);
            EnsurePackageCapacity(bytes);
            await using (var file = new FileStream(temporary, FileMode.CreateNew, FileAccess.Write, FileShare.None, 65536, true))
            {
                using (var archive = new ZipArchive(file, ZipArchiveMode.Create, leaveOpen: true))
                {
                    var metadata = archive.CreateEntry("project.json", CompressionLevel.Optimal);
                    await using (var stream = metadata.Open())
                        await JsonSerializer.SerializeAsync(stream, project, StudioJson.Options, ct).ConfigureAwait(false);
                    foreach (var asset in project.Assets)
                    {
                        ct.ThrowIfCancellationRequested();
                        string source = GetSnapshotAssetPath(project, asset.Id);
                        var entry = archive.CreateEntry("assets/" + StudioPaths.FileName(asset.FileName), CompressionLevel.NoCompression);
                        await using var input = new FileStream(source, FileMode.Open, FileAccess.Read, FileShare.Read, 65536, true);
                        await using var destination = entry.Open();
                        var hash = await CopyAndHashAsync(input, destination, asset.Bytes, ct).ConfigureAwait(false);
                        if (hash.Bytes != asset.Bytes || !string.Equals(hash.Hash, asset.Sha256, StringComparison.OrdinalIgnoreCase))
                            throw new StudioException("media_changed", "A source asset changed or is missing. Relink it before exporting a package.");
                    }
                }
                await file.FlushAsync(ct).ConfigureAwait(false);
                file.Flush(true);
            }
            File.Move(temporary, target, true);
            return target;
        }
        finally { if (File.Exists(temporary)) File.Delete(temporary); }
    }

    public async Task<StudioProject> ImportPackageAsync(Stream input, CancellationToken ct = default)
    {
        await packageGate.WaitAsync(ct).ConfigureAwait(false);
        try { CheckDisk(); return await ImportPackageCoreAsync(input, ct).ConfigureAwait(false); }
        finally { packageGate.Release(); }
    }

    private async Task<StudioProject> ImportPackageCoreAsync(Stream input, CancellationToken ct)
    {
        var id = StudioJson.NewId();
        var stagingRoot = StudioPaths.SafeChild(RootPath, "imports");
        Directory.CreateDirectory(stagingRoot);
        var staging = StudioPaths.SafeChild(stagingRoot, id);
        Directory.CreateDirectory(staging);
        var archivePath = StudioPaths.SafeChild(stagingRoot, id + ".zip");
        try
        {
            await using (var file = new FileStream(archivePath, FileMode.CreateNew, FileAccess.Write, FileShare.None, 65536, true))
                await CopyAndHashAsync(input, file, Options.MaxImportBytes, ct).ConfigureAwait(false);
            using var archive = ZipFile.OpenRead(archivePath);
            if (archive.Entries.Count is < 1 or > 2001)
                throw new StudioException("invalid_package", "A Studio package may contain at most 2000 media files and one project document.");
            var entries = new Dictionary<string, ZipArchiveEntry>(StringComparer.OrdinalIgnoreCase);
            long expanded = 0;
            foreach (var entry in archive.Entries)
            {
                ct.ThrowIfCancellationRequested();
                // Never extract arbitrary paths. Links, folders, alternate separators, duplicate
                // names, device paths and any entry outside the exact allow-list are rejected.
                string name = entry.FullName;
                if (name.Contains('\\') || name.Contains(':') || name.StartsWith('/') || name.EndsWith('/') ||
                    (entry.ExternalAttributes >> 16 & 0xF000) == 0xA000 || !entries.TryAdd(name, entry))
                    throw new StudioException("unsafe_package", "Project package contains an unsafe or duplicate entry.");
                if (name != "project.json" && (!name.StartsWith("assets/", StringComparison.Ordinal) || name[7..] != StudioPaths.FileName(name[7..])))
                    throw new StudioException("unsafe_package", "Only project.json and flat assets/ media entries are permitted.");
                if (entry.Length < 0 || entry.Length > Options.MaxImportBytes - expanded)
                    throw new StudioException("package_too_large", "Expanded package exceeds the configured import limit.");
                expanded += entry.Length;
            }
            EnsurePackageCapacity(expanded);
            if (!entries.TryGetValue("project.json", out var metadata) || metadata.Length > 4 * 1024 * 1024)
                throw new StudioException("invalid_package", "Package requires a project.json no larger than 4 MiB.");
            StudioProject project;
            await using (var stream = metadata.Open())
                project = await JsonSerializer.DeserializeAsync<StudioProject>(stream, StudioJson.Options, ct).ConfigureAwait(false)
                    ?? throw new StudioException("invalid_package", "Package project is empty.");
            StudioCommands.Validate(project, Options);
            if (project.Assets.Select(a => a.FileName).Distinct(StringComparer.OrdinalIgnoreCase).Count() != project.Assets.Count || entries.Count != project.Assets.Count + 1)
                throw new StudioException("invalid_package", "Package assets must match the project exactly, without duplicates or extra files.");
            var mediaDirectory = StudioPaths.SafeChild(staging, "assets");
            Directory.CreateDirectory(mediaDirectory);
            foreach (var asset in project.Assets)
            {
                ct.ThrowIfCancellationRequested();
                if (!entries.TryGetValue("assets/" + asset.FileName, out var entry) || entry.Length != asset.Bytes ||
                    asset.Sha256.Length != 64 || asset.Sha256.Any(c => !Uri.IsHexDigit(c)))
                    throw new StudioException("invalid_package", "An asset is missing or its length/hash metadata is invalid.");
                string path = StudioPaths.SafeChild(mediaDirectory, asset.FileName);
                await using (var source = entry.Open())
                await using (var destination = new FileStream(path, FileMode.CreateNew, FileAccess.Write, FileShare.None, 65536, true))
                {
                    var copied = await CopyAndHashAsync(source, destination, entry.Length, ct).ConfigureAwait(false);
                    if (copied.Bytes != asset.Bytes || !string.Equals(copied.Hash, asset.Sha256, StringComparison.OrdinalIgnoreCase))
                        throw new StudioException("invalid_package", "Media hash verification failed; the package was not imported.");
                }
                // Metadata is untrusted. Desktop re-probes bytes; mobile can open a verified
                // package for review/editing, but cannot claim decoder validation/export.
                if (Options.EnableExternalProcesses)
                {
                    var measured = await Media.ProbeAsync(path, asset.Name, ct).ConfigureAwait(false);
                    asset.Kind = measured.Kind; asset.Width = measured.Width; asset.Height = measured.Height;
                    asset.DurationSeconds = measured.DurationSeconds; asset.HasAudio = measured.HasAudio;
                    asset.SourceFrameRate = measured.SourceFrameRate; asset.VariableFrameRate = measured.VariableFrameRate;
                    asset.ContentType = measured.ContentType;
                }
                asset.Missing = false;
                if (asset.Generation != null) asset.Generation.Request.ProjectId = id;
            }
            project.Id = id; project.Revision = 0; project.ConversationId = null; project.UpdatedAt = DateTimeOffset.UtcNow;
            StudioCommands.Validate(project, Options);
            StudioStorage.Save(StudioPaths.SafeChild(staging, "project.json"), new StudioDocument { Project = project });
            await projectGate.WaitAsync(ct).ConfigureAwait(false);
            try
            {
                var projectsRoot = StudioPaths.SafeChild(RootPath, "projects");
                Directory.CreateDirectory(projectsRoot);
                Directory.Move(staging, StudioPaths.SafeChild(projectsRoot, id));
            }
            finally { projectGate.Release(); }
            return StudioJson.Clone(project);
        }
        catch (InvalidDataException e) { throw new StudioException("invalid_package", "Invalid Studio ZIP package: " + e.Message); }
        catch (JsonException e) { throw new StudioException("invalid_package", "Invalid project JSON: " + e.Message); }
        finally
        {
            if (Directory.Exists(staging)) Directory.Delete(staging, recursive: true);
            if (File.Exists(archivePath)) File.Delete(archivePath);
        }
    }

    private void EnsurePackageCapacity(long bytes)
    {
        if (bytes < 0 || bytes > Options.MaxImportBytes)
            throw new StudioException("package_too_large", "Project media exceeds the configured package limit.");
        var drive = new DriveInfo(Path.GetPathRoot(RootPath)!);
        if (drive.IsReady && drive.AvailableFreeSpace < bytes + Options.MinimumFreeDiskBytes)
            throw new StudioException("insufficient_disk", "There is insufficient disk space for this project package.");
    }

    private async Task<(long Bytes, string Hash)> CopyAndHashAsync(Stream input, Stream output, long limit, CancellationToken ct)
    {
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        byte[] buffer = new byte[65536];
        long total = 0, nextDiskCheck = 0;
        while (true)
        {
            int count = await input.ReadAsync(buffer, ct).ConfigureAwait(false);
            if (count == 0) break;
            if (count > limit - total) throw new StudioException("package_too_large", "Package exceeds the configured size limit.");
            if (total >= nextDiskCheck) { CheckDisk(); nextDiskCheck = total + 16L * 1024 * 1024; }
            total += count; hash.AppendData(buffer.AsSpan(0, count));
            await output.WriteAsync(buffer.AsMemory(0, count), ct).ConfigureAwait(false);
        }
        return (total, Convert.ToHexStringLower(hash.GetHashAndReset()));
    }
}
