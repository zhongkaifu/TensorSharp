// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using TensorAgent.Core.Downloads;
using TensorSharp.Runtime;

namespace TensorAgent.Core.Catalog;

/// <summary>
/// The LoRA plug-ins on this device: one folder per <see cref="CatalogLora"/> under a root
/// beside the models directory, not inside a model's own folder. A model's folder is its
/// files and nothing else (anything in it makes an uninstalled model read as partly
/// downloaded), and a plug-in outlives a re-download of its base model.
/// </summary>
public sealed class LoraStore
{
    // Stores in separate chats may refer to the same directory. Keep the read/check and
    // atomic replacement together: Windows cannot replace a file another reader has open.
    private static readonly object RecipeGate = new();
    private readonly ResumableDownloader _downloader;
    private readonly SemaphoreSlim _gate = new(1, 1);

    public string Root { get; }

    public LoraStore(string root, ResumableDownloader? downloader = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(root);
        Root = Path.GetFullPath(root);
        _downloader = downloader ?? new ResumableDownloader();
    }

    /// <summary>
    /// The license acceptances a download of a plug-in with <see cref="CatalogLora.LicenseTerms"/> is checked against
    /// (<see cref="DownloadAsync"/>), with the catalog entries whose acceptance covers it (<see cref="Catalog"/>). Null
    /// keeps none: such a plug-in then never downloads (fail closed). Plug-ins without terms are unaffected.
    /// </summary>
    public LicenseAcceptanceStore? Licenses { get; init; }

    /// <summary>The entries a plug-in's base-model acceptance is looked up among; <see cref="ModelCatalog.BuiltIn"/> by default.</summary>
    public IReadOnlyList<CatalogModel> Catalog { get; init; } = ModelCatalog.BuiltIn;

    /// <summary>Whether <paramref name="lora"/> declares terms that no acceptance of its base entries covers yet.</summary>
    public bool NeedsLicenseAcceptance(CatalogLora lora)
    {
        ArgumentNullException.ThrowIfNull(lora);
        return lora.LicenseTerms is not null && !(Licenses?.IsAccepted(lora, Catalog) ?? false);
    }

    /// <summary>Refuse, before anything is fetched, a plug-in download whose base entry's terms are not accepted.</summary>
    /// <exception cref="LoraLicenseAcceptanceRequiredException">They are not.</exception>
    public void EnsureLicenseAccepted(CatalogLora lora)
    {
        if (NeedsLicenseAcceptance(lora))
            throw new LoraLicenseAcceptanceRequiredException(lora, LicenseAcceptanceStore.BaseEntries(lora, Catalog).FirstOrDefault());
    }

    public string DirectoryFor(CatalogLora lora) => Path.Combine(Root, lora.Id);

    public string PathFor(CatalogLora lora, LoraFile file) => Path.Combine(DirectoryFor(lora), file.FileName);

    /// <summary>Every file at its pinned size; a part file or a missing file is not installed.</summary>
    public InstallState StateOf(CatalogLora lora)
    {
        bool any = false, all = true;
        foreach (LoraFile file in lora.Files)
        {
            bool present = IsComplete(PathFor(lora, file), file);
            any |= present;
            all &= present;
        }
        if (all)
            return InstallState.Installed;
        return any || Directory.Exists(DirectoryFor(lora)) && Directory.EnumerateFileSystemEntries(DirectoryFor(lora)).Any()
            ? InstallState.Partial
            : InstallState.NotInstalled;
    }

    public bool IsInstalled(CatalogLora lora) => StateOf(lora) == InstallState.Installed;

    /// <summary>Bytes on disk for the plug-in, part files included.</summary>
    public long InstalledBytes(CatalogLora lora)
    {
        string dir = DirectoryFor(lora);
        return Directory.Exists(dir) ? Directory.EnumerateFiles(dir).Sum(f => new FileInfo(f).Length) : 0;
    }

    /// <summary>
    /// Download every file that is not complete, one after another, each verified against its
    /// SHA-256 before it is renamed into place. A <c>.part</c> left by an earlier attempt is resumed.
    /// A plug-in under its base model's license downloads only once that license is accepted
    /// (<see cref="EnsureLicenseAccepted"/>), before anything is fetched.
    /// </summary>
    public async Task DownloadAsync(CatalogLora lora, IProgress<ModelDownloadProgress>? progress, CancellationToken ct)
    {
        ArgumentNullException.ThrowIfNull(lora);
        EnsureLicenseAccepted(lora);
        await _gate.WaitAsync(ct).ConfigureAwait(false);
        try
        {
            Directory.CreateDirectory(DirectoryFor(lora));
            long total = lora.TotalBytes;
            long done = 0;
            var pending = new List<LoraFile>();
            foreach (LoraFile file in lora.Files)
            {
                if (IsComplete(PathFor(lora, file), file))
                    done += file.Bytes;
                else
                    pending.Add(file);
            }

            int index = 0;
            foreach (LoraFile file in pending)
            {
                long before = done;
                int number = ++index;
                var fileProgress = new Progress<DownloadProgress>(p => progress?.Report(new ModelDownloadProgress(
                    file.FileName, number, pending.Count, before + p.BytesReceived, total, p.BytesPerSecond, p.Phase)));
                await _downloader.DownloadAsync(file.Url, PathFor(lora, file), file.Bytes, file.Sha256, fileProgress, ct)
                    .ConfigureAwait(false);
                done += file.Bytes;
            }
            progress?.Report(new ModelDownloadProgress(string.Empty, pending.Count, pending.Count, total, total, 0, "downloading"));
        }
        finally
        {
            _gate.Release();
        }
    }

    /// <summary>Remove the plug-in's folder, part files included.</summary>
    public void Delete(CatalogLora lora)
    {
        string dir = DirectoryFor(lora);
        if (Directory.Exists(dir))
            Directory.Delete(dir, recursive: true);
    }

    /// <summary>
    /// What the engine is handed for this plug-in: its weights, the strength, and the config
    /// that carries what the weights do not (a speed plug-in's sampling schedule). The paths
    /// are absolute, which is how the engine records the set it applies, so asking for the
    /// same set again is recognised as no change.
    /// </summary>
    /// <param name="lora">An installed plug-in.</param>
    /// <param name="strength">The strength to apply; a speed plug-in is always used at its own.</param>
    public LoraSpec SpecFor(CatalogLora lora, float strength)
    {
        ArgumentNullException.ThrowIfNull(lora);
        string? config = null;
        if (lora.RecipeConfig is not null)
            config = MaterializeRecipe(lora);
        else if (lora.ConfigFile is { } name)
            config = Path.Combine(DirectoryFor(lora), name);
        float scale = lora.StrengthAdjustable ? strength : lora.DefaultStrength;
        return new LoraSpec(PathFor(lora, lora.Weights), scale, config);
    }

    /// <summary>
    /// Write the engine's plug-in config for <paramref name="lora"/> (embedded in this
    /// assembly) beside its weights, unless the same text is already there, and return its path.
    /// The engine reads only its schedule and strength when it is handed as a config; its
    /// download entry, which points into the repository's model tree, is not used.
    /// </summary>
    private string MaterializeRecipe(CatalogLora lora)
    {
        string path = Path.Combine(DirectoryFor(lora), lora.RecipeConfig!);
        string text = LoraCatalog.RecipeText(lora);
        lock (RecipeGate)
        {
            if (!File.Exists(path) || File.ReadAllText(path) != text)
            {
                Directory.CreateDirectory(DirectoryFor(lora));
                string tmp = $"{path}.{Guid.NewGuid():N}.tmp";
                try
                {
                    File.WriteAllText(tmp, text);
                    File.Move(tmp, path, overwrite: true);
                }
                finally
                {
                    File.Delete(tmp);
                }
            }
        }
        return path;
    }

    private static bool IsComplete(string path, LoraFile file) =>
        File.Exists(path) && new FileInfo(path).Length == file.Bytes;
}
