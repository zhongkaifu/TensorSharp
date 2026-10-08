// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Buffers;
using System.Security.Cryptography;
using TensorAgent.Core.Downloads;
using TensorAgent.Core.Interop;
using TensorAgent.Core.Localization;

namespace TensorAgent.Core.Catalog;

/// <summary>Where a catalog entry stands on this device.</summary>
public enum InstallState
{
    /// <summary>Nothing of it is on disk.</summary>
    NotInstalled,
    /// <summary>Some required files are missing or partial.</summary>
    Partial,
    /// <summary>Every required file is present at its expected size.</summary>
    Installed,
}

/// <summary>Progress of a whole entry's download (several files).</summary>
/// <param name="FileName">The file being transferred now.</param>
/// <param name="FileIndex">1-based index of that file among the files this download
/// completes, the linked ones first.</param>
/// <param name="FileCount">How many files this download completes, by transfer or by link.</param>
/// <param name="BytesReceived">Bytes on disk across all files of the entry.</param>
/// <param name="TotalBytes">Bytes the entry needs in total.</param>
/// <param name="BytesPerSecond">Current rate.</param>
/// <param name="Phase">"downloading" or "verifying".</param>
public readonly record struct ModelDownloadProgress(
    string FileName, int FileIndex, int FileCount, long BytesReceived, long TotalBytes, double BytesPerSecond, string Phase)
{
    public double Fraction => TotalBytes > 0 ? Math.Min(1.0, (double)BytesReceived / TotalBytes) : 0;
    public TimeSpan? Eta => BytesPerSecond > 1 ? TimeSpan.FromSeconds((TotalBytes - BytesReceived) / BytesPerSecond) : null;
}

/// <summary>
/// The on-device model library: one folder per catalog entry under a root the app chooses
/// (Application Support, excluded from iCloud backup by the app's platform hook), files
/// stored under their catalog names so the engine's companion discovery (a projector
/// beside its model, a VAE beside its DiT) works unchanged.
///
/// <para>
/// A file that two entries list is stored once. The MiniMax-H3 keyframes and references
/// entries share 24.0 GB of companions -- the text encoder, the video and audio VAEs and
/// the tokenizer files -- and differ only in an 11.4 GB denoiser, so fetching every file
/// of every entry into its own folder cost 70.9 GB for the pair. A download that finds a
/// file already complete under another entry links it from there instead (see
/// <see cref="TryLinkSharedCopy"/>): 46.8 GB for both, and the second install fetches only
/// its denoiser. Each entry still has every file in its own folder under its own name,
/// which is what keeps companion discovery, <see cref="Delete"/> and the orphan sweep as
/// they were: removing a folder removes that entry's names, and the bytes stay on disk for
/// as long as another entry names them.
/// </para>
/// </summary>
public sealed class ModelStore
{
    private readonly ResumableDownloader _downloader;
    private readonly IReadOnlyList<CatalogModel> _catalog;
    private readonly SemaphoreSlim _gate = new(1, 1);
    private string _root;

    /// <summary>Root directory holding one sub-folder per catalog entry.</summary>
    public string Root => Volatile.Read(ref _root);

    /// <summary>Called for every file the store creates, so a platform can mark it (e.g. as
    /// excluded from backup). Best effort; exceptions are swallowed.</summary>
    public Action<string>? OnFileCreated { get; set; }

    /// <summary>
    /// Makes the hard link through which an entry takes a file another entry already holds:
    /// <see cref="HardLinks.Create"/>. Replaced only by tests, to fail the way a file system
    /// without hard links does.
    /// </summary>
    internal Action<string, string> CreateHardLink { get; init; } = HardLinks.Create;

    /// <param name="root">Directory holding one sub-folder per catalog entry.</param>
    /// <param name="downloader">Fetches the files; a default one when null.</param>
    /// <param name="catalog">
    /// The entries a download may link a shared file from and whose folders the orphan
    /// sweep keeps; <see cref="ModelCatalog.BuiltIn"/> when null. The whole catalog, not
    /// <see cref="ModelCatalog.ForDevice"/>, for the reason <see cref="SweepOrphanedModels"/>
    /// gives.
    /// </param>
    public ModelStore(string root, ResumableDownloader? downloader = null, IReadOnlyList<CatalogModel>? catalog = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(root);
        _root = Path.GetFullPath(root);
        Directory.CreateDirectory(Root);
        _downloader = downloader ?? new ResumableDownloader();
        _catalog = catalog ?? ModelCatalog.BuiltIn;
    }

    /// <summary>
    /// Switch the library once downloads/imports are idle. The caller can persist the
    /// setting after the destination is checked and before the live root changes.
    /// Existing model files stay in their current directory.
    /// </summary>
    internal void ChangeRoot(string absoluteRoot, Action? beforeChange = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(absoluteRoot);
        if (string.Equals(Root, absoluteRoot,
            OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal))
        {
            beforeChange?.Invoke();
            return;
        }

        if (!_gate.Wait(0))
            throw new InvalidOperationException(Loc.T("settings.storage.modelCache.busy"));
        try
        {
            Directory.CreateDirectory(absoluteRoot);
            string probe = Path.Combine(absoluteRoot, ".tensoragent-write-test-" + Guid.NewGuid().ToString("N"));
            using (var stream = new FileStream(probe, FileMode.CreateNew, FileAccess.Write, FileShare.None,
                bufferSize: 1, FileOptions.DeleteOnClose))
            {
                stream.WriteByte(0);
            }

            beforeChange?.Invoke();
            Volatile.Write(ref _root, absoluteRoot);
        }
        finally
        {
            _gate.Release();
        }
    }

    public string DirectoryFor(CatalogModel model) => Path.Combine(Root, model.Id);

    public string PathFor(CatalogModel model, CatalogFile file) => Path.Combine(DirectoryFor(model), file.FileName);

    /// <summary>Path of the loadable weights, or null when not installed.</summary>
    public string? WeightsPath(CatalogModel model)
    {
        string directory = DirectoryFor(model);
        return StateOf(model, directory) == InstallState.Installed
            ? Path.Combine(directory, model.Weights.FileName) : null;
    }

    /// <summary>Path of an installed optional/required companion by role, or null.</summary>
    public string? CompanionPath(CatalogModel model, CatalogFileRole role)
    {
        CatalogFile? file = model.Files.FirstOrDefault(f => f.Role == role);
        if (file is null)
            return null;
        string path = PathFor(model, file);
        return IsComplete(path, file) ? path : null;
    }

    public InstallState StateOf(CatalogModel model) => StateOf(model, DirectoryFor(model));

    private static InstallState StateOf(CatalogModel model, string directory)
    {
        bool any = false, all = true;
        foreach (CatalogFile file in model.Files)
        {
            if (file.Optional)
                continue;
            bool present = IsComplete(Path.Combine(directory, file.FileName), file);
            any |= present;
            all &= present;
        }
        if (all) return InstallState.Installed;
        if (any || Directory.Exists(directory) && Directory.EnumerateFileSystemEntries(directory).Any())
            return InstallState.Partial;
        return InstallState.NotInstalled;
    }

    public bool IsFileInstalled(CatalogModel model, CatalogFile file) => IsComplete(PathFor(model, file), file);

    /// <summary>Bytes on disk for the entry, including partial files.</summary>
    public long InstalledBytes(CatalogModel model)
    {
        string dir = DirectoryFor(model);
        if (!Directory.Exists(dir))
            return 0;
        long total = 0;
        foreach (string f in Directory.EnumerateFiles(dir))
            total += new FileInfo(f).Length;
        return total;
    }

    /// <summary>
    /// Bytes still to transfer for the given files (required ones by default). A file that
    /// another entry already holds complete is not counted, because the download links it
    /// rather than fetching it: with one MiniMax-H3 entry installed, the other is an 11.4 GB
    /// download, not 35 GB. <see cref="CatalogModel.TotalBytes"/> stays the entry's own size.
    /// </summary>
    public long RemainingBytes(CatalogModel model, bool includeOptional = false) =>
        RemainingBytes(model, file => !file.Optional || includeOptional);

    /// <summary>Bytes to fetch for the same required and explicitly chosen optional
    /// files as <see cref="DownloadAsync"/>, including resumable and shared copies.</summary>
    public long RemainingBytes(CatalogModel model, IReadOnlyCollection<CatalogFileRole> optionalRoles) =>
        RemainingBytes(model, file => !file.Optional || optionalRoles.Contains(file.Role));

    private long RemainingBytes(CatalogModel model, Func<CatalogFile, bool> include)
    {
        string root = Root;
        string directory = Path.Combine(root, model.Id);
        long remaining = 0;
        foreach (CatalogFile file in model.Files)
        {
            if (!include(file))
                continue;
            string path = Path.Combine(directory, file.FileName);
            if (IsComplete(path, file) || SharedCopy(model, file, root) is not null)
                continue;
            string part = ResumableDownloader.PartPath(path);
            long have = File.Exists(part) ? new FileInfo(part).Length : 0;
            remaining += Math.Max(0, file.Bytes - have);
        }
        return remaining;
    }

    /// <summary>
    /// Download every required file (and the optional ones named in
    /// <paramref name="optionalRoles"/>) that is not already complete. Files are fetched one
    /// after another - a phone's link is the bottleneck, not the server - and each is
    /// verified against its SHA-256 before it is renamed into place. A file that another
    /// entry already holds complete is linked from that copy, before the first transfer
    /// starts, instead of being fetched (see <see cref="TryLinkSharedCopy"/>).
    /// </summary>
    public async Task DownloadAsync(
        CatalogModel model,
        IProgress<ModelDownloadProgress>? progress,
        CancellationToken ct,
        IReadOnlyCollection<CatalogFileRole>? optionalRoles = null)
    {
        ArgumentNullException.ThrowIfNull(model);
        if (model.SideloadOnly)
        {
            throw new InvalidOperationException(
                Loc.T("host.models.noDownloadUrl", ("model", model.DisplayName), ("file", model.Weights.FileName)));
        }
        await _gate.WaitAsync(ct).ConfigureAwait(false);
        try
        {
            Directory.CreateDirectory(DirectoryFor(model));
            var wanted = model.Files
                .Where(f => !f.Optional || (optionalRoles?.Contains(f.Role) ?? false))
                .ToList();
            long total = wanted.Sum(f => f.Bytes);
            long doneBefore = 0;
            var pending = new List<CatalogFile>();
            foreach (CatalogFile file in wanted)
            {
                if (IsComplete(PathFor(model, file), file))
                    doneBefore += file.Bytes;
                else
                    pending.Add(file);
            }

            // Every link is made before the first transfer starts, not as each file's turn
            // comes. A MiniMax-H3 entry lists its 11.4 GB denoiser first -- an hour at
            // 25 Mbit/s -- and with the links waiting behind it, deleting the entry that
            // held the shared copies during that hour meant fetching all 24 GB of them
            // again. Linked first, the progress also starts from what is already here, so
            // the time left it implies is the denoiser's, not that of a 35 GB download.
            var transfers = new List<CatalogFile>(pending.Count);
            int completed = 0;
            foreach (CatalogFile file in pending)
            {
                string path = PathFor(model, file);
                if (!TryLinkSharedCopy(model, file, path))
                {
                    transfers.Add(file);
                    continue;
                }
                // Nothing to transfer: the whole file arrives at once.
                doneBefore += file.Bytes;
                progress?.Report(new ModelDownloadProgress(
                    file.FileName, ++completed, pending.Count, doneBefore, total, 0, "downloading"));
                Notify(path);
            }

            foreach (CatalogFile file in transfers)
            {
                string path = PathFor(model, file);
                long baseBytes = doneBefore;
                int index = ++completed;
                var fileProgress = new Progress<DownloadProgress>(p =>
                    progress?.Report(new ModelDownloadProgress(
                        file.FileName, index, pending.Count, baseBytes + p.BytesReceived, total, p.BytesPerSecond, p.Phase)));
                await _downloader.DownloadAsync(file.Url, path, file.Bytes, file.Sha256, fileProgress, ct).ConfigureAwait(false);
                doneBefore += file.Bytes;
                Notify(path);
            }
            progress?.Report(new ModelDownloadProgress(string.Empty, pending.Count, pending.Count, total, total, 0, "downloading"));
        }
        finally
        {
            _gate.Release();
        }
    }

    /// <summary>
    /// Import the exact local artifact described by a sideload-only catalog card.
    /// Bytes are copied to a same-directory staging file, checked for both length and
    /// SHA-256, and only then atomically replace the loadable destination. A wrong pick,
    /// cancellation, or read error therefore cannot damage a previously imported model.
    /// </summary>
    public async Task ImportAsync(
        CatalogModel model,
        Stream source,
        IProgress<long>? progress = null,
        CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(source);
        if (!model.SideloadOnly)
            throw new InvalidOperationException(Loc.T("host.models.notImported", ("model", model.DisplayName)));
        if (!source.CanRead)
            throw new ArgumentException(Loc.T("host.import.unreadable"), nameof(source));

        CatalogFile weights = model.Weights;

        await _gate.WaitAsync(ct).ConfigureAwait(false);
        byte[]? buffer = null;
        string? staging = null;
        try
        {
            string directory = DirectoryFor(model);
            string destination = Path.Combine(directory, weights.FileName);
            staging = destination + ".import-" + Guid.NewGuid().ToString("N");
            buffer = ArrayPool<byte>.Shared.Rent(1024 * 1024);
            Directory.CreateDirectory(directory);
            using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
            long copied = 0;
            await using (var target = new FileStream(
                staging, FileMode.CreateNew, FileAccess.Write, FileShare.None,
                bufferSize: 1024 * 1024, FileOptions.Asynchronous | FileOptions.SequentialScan))
            {
                while (true)
                {
                    int read = await source.ReadAsync(buffer.AsMemory(0, buffer.Length), ct).ConfigureAwait(false);
                    if (read == 0)
                        break;

                    copied = checked(copied + read);
                    if (copied > weights.Bytes)
                    {
                        throw new InvalidDataException(Loc.T("host.import.tooLarge",
                            ("file", Path.GetFileName(weights.FileName)), ("bytes", weights.Bytes.ToString("N0", Loc.Culture))));
                    }

                    hash.AppendData(buffer, 0, read);
                    await target.WriteAsync(buffer.AsMemory(0, read), ct).ConfigureAwait(false);
                    progress?.Report(copied);
                }
                await target.FlushAsync(ct).ConfigureAwait(false);
                target.Flush(flushToDisk: true);
            }

            if (copied != weights.Bytes)
            {
                throw new InvalidDataException(Loc.T("host.import.wrongSize",
                    ("size", copied.ToString("N0", Loc.Culture)), ("file", weights.FileName),
                    ("bytes", weights.Bytes.ToString("N0", Loc.Culture))));
            }

            string actualHash = Convert.ToHexString(hash.GetHashAndReset()).ToLowerInvariant();
            if (!string.Equals(actualHash, weights.Sha256, StringComparison.Ordinal))
            {
                throw new InvalidDataException(
                    Loc.T("host.import.wrongHash", ("actual", actualHash), ("expected", weights.Sha256)));
            }

            File.Move(staging, destination, overwrite: true);
            Notify(destination);
        }
        finally
        {
            if (buffer is not null)
                ArrayPool<byte>.Shared.Return(buffer);
            try { if (staging is not null && File.Exists(staging)) File.Delete(staging); }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                // An abandoned staging file is never loadable and must not mask the
                // validation/read error that caused this cleanup path.
            }
            _gate.Release();
        }
    }

    /// <summary>
    /// Remove every file of the entry, partial downloads included. Only this entry's names
    /// go: a file it shares with another entry is one set of bytes under two names, and
    /// those bytes stay on disk under the other entry's name, which stays installed.
    /// </summary>
    public void Delete(CatalogModel model)
    {
        string dir = DirectoryFor(model);
        if (Directory.Exists(dir))
            Directory.Delete(dir, recursive: true);
    }

    /// <summary>
    /// Delete the model directories of retired catalog entries, and say how much that freed.
    ///
    /// <para>
    /// A directory is named by its entry's id, and an id changes whenever the entry
    /// changes which FILE it points at -- swapping Gemma 4 12B from UD-IQ3_XXS to
    /// UD-IQ2_M turns <c>gemma-4-12b-iq3xxs</c> into <c>gemma-4-12b-iq2m</c>. The old
    /// directory then belongs to no entry, so the Models list cannot show it and the
    /// user cannot delete it: 4.6 GB of a superseded quantization, invisible,
    /// on a device where storage is the scarcest thing there is. This is the only place
    /// that can reclaim it.
    /// </para>
    /// <para>
    /// Only an id the catalog RETIRED is reclaimed (<see cref="ModelCatalog.Retired"/>),
    /// not every id this build does not know. An unknown id may be a NEWER build's entry:
    /// on a Mac the Debug and Release builds share this directory, and a Release build
    /// from the day before, sweeping everything its own catalog did not list, deleted the
    /// installed models of the five entries the Debug build had just added. Such a
    /// directory is kept, and said so; the build that knows it shows it again.
    /// </para>
    /// <para>
    /// Checked against the WHOLE catalog rather than what this device is offered
    /// (<see cref="ModelCatalog.ForDevice"/>), because an entry gated to a larger device
    /// is still a real entry -- deleting weights for a model an iPad can run, because a
    /// phone cannot, would be a data-loss bug wearing a tidy-up's clothes.
    /// </para>
    /// <para>
    /// A folder holds names, not bytes. An orphan that a claimed entry linked a shared file
    /// from, back when the orphan was itself an entry, loses only its own name for that
    /// file, so the sweep can no more take a file a claimed entry lists than
    /// <see cref="Delete"/> can. The bytes it reports freed then overstate: they include
    /// files whose space stays in use under the other name.
    /// </para>
    /// </summary>
    /// <param name="catalog">The entries whose folders are kept; the store's catalog when null.</param>
    /// <param name="retired">The ids whose folders are reclaimed; <see cref="ModelCatalog.Retired"/> when null.</param>
    /// <returns>Bytes freed.</returns>
    public long SweepOrphanedModels(IReadOnlyList<CatalogModel>? catalog = null, IReadOnlyCollection<string>? retired = null)
    {
        string root = Root;
        var known = new HashSet<string>(
            (catalog ?? _catalog).Select(m => m.Id), StringComparer.OrdinalIgnoreCase);
        var reclaimable = new HashSet<string>(retired ?? ModelCatalog.Retired, StringComparer.OrdinalIgnoreCase);

        long freed = 0;
        IEnumerable<string> directories;
        try { directories = Directory.EnumerateDirectories(root); }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { return 0; }

        foreach (string directory in directories.ToList())
        {
            string id = Path.GetFileName(directory);
            if (known.Contains(id))
                continue;
            if (!reclaimable.Contains(id))
            {
                Console.WriteLine(
                    $"TensorAgent: kept {id}: no entry of this build's catalog claims it and none was "
                    + "retired under that name, so it is most likely a newer build's model");
                continue;
            }
            try
            {
                long bytes = Directory.EnumerateFiles(directory, "*", SearchOption.AllDirectories)
                    .Sum(f => new FileInfo(f).Length);
                Directory.Delete(directory, recursive: true);
                freed += bytes;
                Console.WriteLine(
                    $"TensorAgent: removed {id}, which no catalog entry "
                    + $"claims any more ({bytes / (1024.0 * 1024.0 * 1024.0):0.0} GB freed)");
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                // A sweep that cannot delete is not a reason to fail a launch.
                Console.WriteLine($"TensorAgent: could not remove {directory}: {ex.Message}");
            }
        }

        // And loose FILES, which belong to no entry by construction: this directory
        // holds one sub-folder per catalog id (see DirectoryFor) and nothing else ever
        // writes into it. One can still arrive -- a weights file pushed onto the device
        // by hand, landing beside the per-model folders instead of inside one -- and it
        // is worse off than an orphaned directory: the Models list is built from entries,
        // so a stray file has no row, no size against any model, and no delete button,
        // while being the largest kind of file this app deals in.
        IEnumerable<string> strays;
        try { strays = Directory.EnumerateFiles(root); }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { return freed; }

        foreach (string file in strays.ToList())
        {
            try
            {
                long bytes = new FileInfo(file).Length;
                File.Delete(file);
                freed += bytes;
                Console.WriteLine(
                    $"TensorAgent: removed the stray file {Path.GetFileName(file)} from the models "
                    + $"directory, which no catalog entry claims ({bytes / (1024.0 * 1024.0 * 1024.0):0.0} GB freed)");
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                Console.WriteLine($"TensorAgent: could not remove {file}: {ex.Message}");
            }
        }
        return freed;
    }

    /// <summary>Remove one optional companion (e.g. a projector) to free memory/disk.</summary>
    public void DeleteFile(CatalogModel model, CatalogFile file)
    {
        string path = PathFor(model, file);
        if (File.Exists(path)) File.Delete(path);
        string part = ResumableDownloader.PartPath(path);
        if (File.Exists(part)) File.Delete(part);
    }

    /// <summary>
    /// Give <paramref name="model"/> the copy of <paramref name="file"/> that another entry
    /// already holds, as a hard link at <paramref name="path"/>. False means download it: no
    /// other entry has a complete copy, or the link could not be made.
    ///
    /// <para>
    /// The copy is not hashed again. It was verified when it was downloaded, and from then
    /// on the store trusts a complete file's size, as every other check here does
    /// (<see cref="IsComplete"/>).
    /// </para>
    /// </summary>
    private bool TryLinkSharedCopy(CatalogModel model, CatalogFile file, string path)
    {
        if (SharedCopy(model, file) is not { } source)
            return false;
        string sourceName = Path.GetRelativePath(Root, source);

        try
        {
            // A wrong-sized file under this name is a broken download, not a copy: the
            // downloader's first act would be to delete it, and left here it would refuse
            // the link.
            if (File.Exists(path))
                File.Delete(path);
            CreateHardLink(source, path);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            // Another volume, a file system without hard links, a sandbox that forbids
            // them: reasons to spend the bandwidth, never reasons to fail the download.
            Console.WriteLine(
                $"TensorAgent: could not link {model.Id}/{file.FileName} to {sourceName} ({ex.Message}); downloading it instead");
            return false;
        }

        // Removed only once the link exists, so a link that fails keeps the part file and
        // the download it falls back to resumes from it instead of starting over.
        string part = ResumableDownloader.PartPath(path);
        try
        {
            if (File.Exists(part))
                File.Delete(part);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            Console.WriteLine($"TensorAgent: could not remove the stale {part}: {ex.Message}");
        }

        string size = file.Bytes >= 1L << 30
            ? $"{file.Bytes / (1024.0 * 1024.0 * 1024.0):0.0} GB"
            : $"{file.Bytes / (1024.0 * 1024.0):0.##} MB";
        Console.WriteLine(
            $"TensorAgent: linked {model.Id}/{file.FileName} to {sourceName} instead of downloading {size} again");
        return true;
    }

    /// <summary>
    /// A complete copy of <paramref name="file"/> held by another entry of the catalog, or
    /// null. The same artifact is the same SHA-256 and size, whatever either entry names
    /// it, and complete means what it means everywhere here, the exact size -- so a part
    /// file, or a truncated copy left by an interrupted transfer, is never linked.
    /// </summary>
    private string? SharedCopy(CatalogModel model, CatalogFile file) => SharedCopy(model, file, Root);

    private string? SharedCopy(CatalogModel model, CatalogFile file, string root)
    {
        foreach (CatalogModel other in _catalog)
        {
            if (string.Equals(other.Id, model.Id, StringComparison.OrdinalIgnoreCase))
                continue;
            foreach (CatalogFile candidate in other.Files)
            {
                if (candidate.Bytes != file.Bytes
                    || !string.Equals(candidate.Sha256, file.Sha256, StringComparison.OrdinalIgnoreCase))
                    continue;
                string path = Path.Combine(root, other.Id, candidate.FileName);
                if (IsComplete(path, candidate))
                    return path;
            }
        }
        return null;
    }

    private static bool IsComplete(string path, CatalogFile file) =>
        File.Exists(path) && new FileInfo(path).Length == file.Bytes;

    private void Notify(string path)
    {
        try { OnFileCreated?.Invoke(path); }
        catch { /* a backup-exclusion failure must not fail the download */ }
    }
}
