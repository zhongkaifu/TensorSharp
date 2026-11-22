// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp.Runtime.Logging;

namespace TensorSharp.AgentHost.Skills
{
    /// <summary>A skill directory that could not be loaded, and why.</summary>
    /// <param name="Path">The directory (or <c>SKILL.md</c>) that failed.</param>
    /// <param name="Message">What was wrong, phrased for the skill's author.</param>
    public readonly record struct SkillLoadError(string Path, string Message);

    /// <summary>The result of one scan, for logging and for the management API.</summary>
    /// <param name="Skills">Every skill that loaded, sorted by id.</param>
    /// <param name="Errors">Every directory that looked like a skill but did not load.</param>
    /// <param name="ScannedRoots">The roots that were walked.</param>
    public sealed record SkillScanResult(
        IReadOnlyList<Skill> Skills,
        IReadOnlyList<SkillLoadError> Errors,
        IReadOnlyList<string> ScannedRoots);

    /// <summary>
    /// How a <see cref="SkillRegistry"/> finds and stores skills.
    /// </summary>
    public sealed class SkillRegistryOptions
    {
        /// <summary>Optional marketplace transport, for hosts with their own HTTP policy or tests.
        /// Network access is still gated separately by the request's host options.</summary>
        public SkillMarketplace? Marketplace { get; init; }

        /// <summary>Optional host network policy, evaluated for each marketplace operation.
        /// Hosts can use this to apply their current domain allow-list and network switch.</summary>
        public Func<string, bool>? MarketplaceHostAllowed { get; init; }

        /// <summary>
        /// Directories to scan, in precedence order. A root may be a single skill
        /// (it contains <c>SKILL.md</c> directly) or a directory of them.
        /// </summary>
        public IReadOnlyList<string> Roots { get; init; } = Array.Empty<string>();

        /// <summary>
        /// Where skills installed at runtime are written. Also scanned, and always
        /// first in precedence so an uploaded skill shadows a stale copy of the same
        /// name in a read-only operator root rather than being shadowed by it. Null
        /// disables installation, leaving the registry read-only.
        /// </summary>
        public string? InstallDirectory { get; init; }

        /// <summary>
        /// A file recording skills the user has deleted, so they stay deleted.
        ///
        /// <para>
        /// A skill that ships INSIDE the application cannot be removed by deleting it:
        /// the bundle is read-only, and it is recreated by the next install of the app
        /// anyway. Without somewhere to write down "the user got rid of this", deleting
        /// a built-in skill either fails outright or works until the next launch, and
        /// both of those are worse than not offering the button. Ids listed here are
        /// skipped by discovery. Null keeps the old behaviour, which is what the
        /// desktop server wants: an operator manages that directory with a shell.
        /// </para>
        /// </summary>
        public string? RemovedRecordFile { get; init; }

        /// <summary>
        /// How deep to walk under a root looking for <c>SKILL.md</c>. Two levels
        /// covers both conventions in the wild — <c>root/&lt;skill&gt;/SKILL.md</c> and
        /// the layout of <see href="https://github.com/anthropics/skills"/>, where
        /// pointing at the repository root has to reach <c>skills/&lt;skill&gt;/SKILL.md</c>
        /// — without turning a mistyped root such as <c>$HOME</c> into a full-disk walk.
        /// </summary>
        public int MaxDepth { get; init; } = 3;

        /// <summary>
        /// Upper bound on skills loaded from all roots together. A root pointed at
        /// the wrong directory should fail visibly rather than exhaust memory.
        /// </summary>
        public int MaxSkills { get; init; } = 512;

        /// <summary>
        /// Largest <c>SKILL.md</c> that will be parsed. The published skills top out
        /// around 75 KB; a megabyte of Markdown is a mistake, not a skill.
        /// </summary>
        public int MaxManifestBytes { get; init; } = 4 * 1024 * 1024;

        /// <summary>Largest total size of one skill directory.</summary>
        public long MaxSkillBytes { get; init; } = 256L * 1024 * 1024;

        /// <summary>Largest number of files in one skill directory.</summary>
        public int MaxSkillFiles { get; init; } = 4096;
    }

    /// <summary>
    /// The set of skills a host knows about.
    ///
    /// <para>
    /// The registry owns discovery (walking configured roots for <c>SKILL.md</c>
    /// files), installation (copying a directory in, or extracting an uploaded ZIP),
    /// and lookup. It is deliberately the only component that touches skill storage,
    /// so the containment rules in <see cref="SkillPathGuard"/> are enforced in one
    /// place rather than at each of the CLI, server and public API surfaces.
    /// </para>
    /// <para>
    /// Reads are lock-free: the whole index is an immutable snapshot swapped under a
    /// lock by writers. Chat requests hit <see cref="TryGet"/> and
    /// <see cref="Skills"/> on every turn, and a rescan triggered by an upload must
    /// not make them wait.
    /// </para>
    /// <para>
    /// <b>Name collisions.</b> Two roots may both ship a skill called <c>pdf</c>.
    /// Precedence is root order — the install directory first, then the configured
    /// roots as given — and the loser is not renamed but reported as an error, so
    /// selecting <c>pdf</c> is never ambiguous and the operator is told which copy
    /// won. Renaming the loser to something like <c>pdf-2</c> would be worse: the
    /// name is what a user types and what a <c>SKILL.md</c> cross-references, and it
    /// would silently change when an unrelated root was added.
    /// </para>
    /// </summary>
    public sealed class SkillRegistry
    {
        private readonly SkillRegistryOptions _options;
        private readonly ILogger _logger;
        private readonly object _writeGate = new();
        private volatile Index _index;

        /// <summary>
        /// Create a registry and scan its roots immediately, so the caller can log a
        /// startup summary and so the first chat request never pays for the walk.
        /// </summary>
        public SkillRegistry(SkillRegistryOptions options, ILogger? logger = null)
        {
            _options = options ?? throw new ArgumentNullException(nameof(options));
            _logger = logger ?? NullLogger.Instance;
            _index = Index.Empty;
            Refresh();
        }

        /// <summary>Every loaded skill, sorted by id. A stable snapshot; never null.</summary>
        public IReadOnlyList<Skill> Skills => _index.Ordered;

        /// <summary>Every skill directory that failed to load, with the reason.</summary>
        public IReadOnlyList<SkillLoadError> Errors => _index.Errors;

        /// <summary>The roots that were scanned, install directory first.</summary>
        public IReadOnlyList<string> Roots => _index.Roots;

        /// <summary>Where installed skills are written, or null when installation is disabled.</summary>
        public string? InstallDirectory => _options.InstallDirectory;

        /// <summary>True when this registry can accept new skills.</summary>
        public bool CanInstall => !string.IsNullOrWhiteSpace(_options.InstallDirectory);

        /// <summary>The shared discovery client. Merely accessing it performs no network I/O.</summary>
        public SkillMarketplace Marketplace => _options.Marketplace ?? SkillMarketplace.Shared;

        internal bool MarketplaceHostAllowed(string host) => _options.MarketplaceHostAllowed?.Invoke(host) ?? true;

        /// <summary>Look up a skill by id, case-insensitively.</summary>
        public bool TryGet(string? id, out Skill skill)
        {
            skill = null!;
            if (string.IsNullOrWhiteSpace(id))
                return false;
            return _index.ById.TryGetValue(id.Trim(), out skill!);
        }

        /// <summary>
        /// Resolve a caller-supplied list of skill ids. Unknown ids are returned
        /// separately rather than ignored: a request naming a skill the server does
        /// not have is a client bug worth a 400, and a model naming one is worth an
        /// error it can read and correct.
        /// </summary>
        public IReadOnlyList<Skill> Resolve(IEnumerable<string>? ids, out IReadOnlyList<string> unknown)
        {
            var resolved = new List<Skill>();
            var missing = new List<string>();
            if (ids == null)
            {
                unknown = missing;
                return resolved;
            }

            var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            foreach (string raw in ids)
            {
                string id = (raw ?? string.Empty).Trim();
                if (id.Length == 0 || !seen.Add(id))
                    continue;
                if (TryGet(id, out Skill skill))
                    resolved.Add(skill);
                else
                    missing.Add(id);
            }

            // Sorting here is what makes the rendered catalog byte-stable across
            // requests: the KV prefix cache chains its block hashes from token 0, so a
            // selection that arrives in a different order would otherwise re-render the
            // whole system block and miss every cached block. See SkillCatalog.
            resolved.Sort(static (a, b) => string.CompareOrdinal(a.Id, b.Id));
            unknown = missing;
            return resolved;
        }

        /// <summary>
        /// Re-walk every root and publish a new snapshot. Safe to call concurrently
        /// with reads.
        /// </summary>
        public SkillScanResult Refresh()
        {
            lock (_writeGate)
            {
                Index built = Build();
                _index = built;
                return new SkillScanResult(built.Ordered, built.Errors, built.Roots);
            }
        }

        /// <summary>
        /// Copy a skill directory into the install directory and register it.
        /// </summary>
        /// <param name="sourceDirectory">A directory containing <c>SKILL.md</c>.</param>
        /// <param name="overwrite">Replace an existing installed skill of the same name.</param>
        /// <param name="source">
        /// Hub provenance to record with it (<see cref="SkillMarketplace"/>), or null for a
        /// skill that did not come from the hub; null also clears any earlier record.
        /// </param>
        /// <exception cref="InvalidOperationException">Installation is disabled, or the name is already taken.</exception>
        /// <exception cref="SkillInstallException">The directory is not a usable skill.</exception>
        public Skill InstallFromDirectory(string sourceDirectory, bool overwrite = false, SkillSource? source = null)
        {
            if (!CanInstall)
                throw new InvalidOperationException("This registry is read-only: no install directory is configured.");
            if (string.IsNullOrWhiteSpace(sourceDirectory) || !Directory.Exists(sourceDirectory))
                throw new SkillInstallException($"'{sourceDirectory}' is not a directory.");

            SkillSource? provenance = source;
            string sourceRoot = SkillPathGuard.NormalizeDirectory(sourceDirectory);
            string manifestPath = Path.Combine(sourceRoot, SkillManifestParser.SkillFileName);
            if (!File.Exists(manifestPath))
                throw new SkillInstallException($"'{sourceDirectory}' has no {SkillManifestParser.SkillFileName}.");

            if (!TryLoad(sourceRoot, SkillOrigin.Installed, null, out Skill? staged, out string? error))
                throw new SkillInstallException(error ?? "the skill could not be read");

            lock (_writeGate)
            {
                // Installing a skill is the user asking for it back, so any record of
                // them having deleted it is stale from this moment. Cleared BEFORE the
                // copy, so the rebuild at the end of this block already sees it.
                ForgetRemoved(staged!.Manifest.Name);
                string destination = ReserveInstallDirectory(staged!.Manifest.Name, overwrite);
                try
                {
                    CopyTree(sourceRoot, destination, staged.Manifest.Name);
                }
                catch
                {
                    TryDeleteDirectory(destination);
                    throw;
                }
                RecordSource(staged.Manifest.Name, provenance);

                Index built = Build();
                _index = built;
                if (built.ById.TryGetValue(staged.Manifest.Name, out Skill? installed))
                {
                    _logger.LogInformation(LogEventIds.SkillInstalled,
                        "skills.installed id={SkillId} files={FileCount} bytes={Bytes} source=directory",
                        installed.Id, installed.Files.Count, installed.TotalBytes);
                    return installed;
                }

                // The scan that just ran should have found what was written; if it did
                // not, the copy is half-formed and must not be left behind pretending to
                // be a skill.
                TryDeleteDirectory(destination);
                _index = Build();
                throw new SkillInstallException(
                    $"'{staged.Manifest.Name}' was copied but did not load back; the partial copy was removed.");
            }
        }

        /// <summary>
        /// Extract a skill bundle (a ZIP holding one skill directory, or a
        /// <c>SKILL.md</c> at its own root) into the install directory and register it.
        /// </summary>
        /// <param name="archive">The ZIP bytes. Read once, forward-only.</param>
        /// <param name="overwrite">Replace an existing installed skill of the same name.</param>
        /// <param name="limits">Extraction limits, or null for the registry's own.</param>
        /// <exception cref="InvalidOperationException">Installation is disabled.</exception>
        /// <exception cref="SkillInstallException">The archive is unusable or exceeds a limit.</exception>
        public Skill InstallFromZip(Stream archive, bool overwrite = false, SkillArchiveLimits? limits = null)
        {
            if (!CanInstall)
                throw new InvalidOperationException("This registry is read-only: no install directory is configured.");
            ArgumentNullException.ThrowIfNull(archive);

            SkillArchiveLimits effective = limits ?? new SkillArchiveLimits
            {
                MaxTotalBytes = _options.MaxSkillBytes,
                MaxEntries = _options.MaxSkillFiles,
            };

            lock (_writeGate)
            {
                string staging = Path.Combine(
                    EnsureInstallDirectory(),
                    ".staging-" + Guid.NewGuid().ToString("N"));

                try
                {
                    SkillArchive.Extract(archive, staging, effective);

                    string skillRoot = SkillArchive.LocateSkillRoot(staging)
                        ?? throw new SkillInstallException(
                            $"the archive contains no {SkillManifestParser.SkillFileName}.");

                    if (!TryLoad(skillRoot, SkillOrigin.Installed, null, out Skill? staged, out string? error))
                        throw new SkillInstallException(error ?? "the skill could not be read");

                    // Installing a skill is the user asking for it back, so any record of
                // them having deleted it is stale from this moment. Cleared BEFORE the
                // copy, so the rebuild at the end of this block already sees it.
                ForgetRemoved(staged!.Manifest.Name);
                string destination = ReserveInstallDirectory(staged!.Manifest.Name, overwrite);
                    Directory.Move(skillRoot, destination);
                    RecordSource(staged.Manifest.Name, null);

                    Index built = Build();
                    _index = built;
                    if (built.ById.TryGetValue(staged.Manifest.Name, out Skill? installed))
                    {
                        _logger.LogInformation(LogEventIds.SkillInstalled,
                            "skills.installed id={SkillId} files={FileCount} bytes={Bytes} source=zip",
                            installed.Id, installed.Files.Count, installed.TotalBytes);
                        return installed;
                    }

                    TryDeleteDirectory(destination);
                    _index = Build();
                    throw new SkillInstallException(
                        $"'{staged.Manifest.Name}' was extracted but did not load back; the partial copy was removed.");
                }
                finally
                {
                    TryDeleteDirectory(staging);
                }
            }
        }

        /// <summary>
        /// Delete an installed skill. Refuses to touch a skill discovered under an
        /// operator-configured root — the management API must never delete files the
        /// operator put there by hand.
        /// </summary>
        public bool Remove(string id)
        {
            lock (_writeGate)
            {
                if (!_index.ById.TryGetValue((id ?? string.Empty).Trim(), out Skill? skill))
                    return false;
                if (skill.Origin == SkillOrigin.Installed)
                {
                    TryDeleteDirectory(skill.RootDirectory);
                    RecordSource(skill.Id, null);
                }
                else if (_options.RemovedRecordFile is { Length: > 0 })
                {
                    // It lives in a read-only root -- inside the app bundle, on a phone.
                    // The bytes cannot go, so the ID is written down instead and
                    // discovery skips it from now on, including after the app is
                    // reinstalled and the bundle puts the files back.
                    RecordRemoved(skill.Id);
                }
                else
                {
                    throw new InvalidOperationException(
                        $"'{skill.Id}' was discovered under {skill.DiscoveredUnder} and is not managed by TensorSharp; " +
                        "remove it from that directory instead.");
                }

                _index = Build();
                _logger.LogInformation(LogEventIds.SkillRemoved, "skills.removed id={SkillId}", skill.Id);
                return true;
            }
        }

        /// <summary>
        /// The installed skill that came from hub package <paramref name="package"/>
        /// (<c>owner/repository@skill-name</c>), whatever name it was installed under.
        /// </summary>
        public bool TryGetInstalledPackage(string? package, out Skill skill)
        {
            skill = null!;
            if (string.IsNullOrWhiteSpace(package))
                return false;
            string wanted = package.Trim();
            foreach (Skill candidate in _index.Ordered)
            {
                if (candidate.Source is { } source
                    && string.Equals(source.Package, wanted, StringComparison.OrdinalIgnoreCase))
                {
                    skill = candidate;
                    return true;
                }
            }
            return false;
        }

        /// <summary>
        /// The provenance record for hub installs: one JSON file in the install directory,
        /// keyed by skill id. It sits beside the skill directories rather than inside one, so
        /// it is never part of a skill's own files, and a root-level file is not a skill.
        /// </summary>
        internal const string SourcesFileName = ".hub-sources.json";

        private string? SourcesFile => CanInstall ? Path.Combine(_options.InstallDirectory!, SourcesFileName) : null;

        private Dictionary<string, SkillSource> ReadSources()
        {
            var sources = new Dictionary<string, SkillSource>(StringComparer.OrdinalIgnoreCase);
            string? file = SourcesFile;
            if (file == null || !File.Exists(file))
                return sources;
            try
            {
                var stored = System.Text.Json.JsonSerializer.Deserialize<Dictionary<string, SkillSource>>(
                    File.ReadAllText(file));
                if (stored != null)
                {
                    foreach ((string id, SkillSource source) in stored)
                    {
                        if (source != null && !string.IsNullOrWhiteSpace(source.Package))
                            sources[id] = source;
                    }
                }
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or System.Text.Json.JsonException)
            {
                // An unreadable record loses the provenance line, not the skills.
                _logger.LogWarning(LogEventIds.SkillRejected,
                    "skills.sources.unreadable file={File} reason={Reason}", file, ex.Message);
            }
            return sources;
        }

        /// <summary>Record (or, with null, forget) the hub provenance of one installed skill. Call under the write gate.</summary>
        private void RecordSource(string id, SkillSource? source)
        {
            string? file = SourcesFile;
            if (file == null)
                return;
            Dictionary<string, SkillSource> sources = ReadSources();
            bool changed = source == null ? sources.Remove(id) : true;
            if (source != null)
                sources[id] = source;
            if (!changed)
                return;
            try
            {
                Directory.CreateDirectory(_options.InstallDirectory!);
                string temporary = file + "." + Guid.NewGuid().ToString("N") + ".tmp";
                File.WriteAllText(temporary, System.Text.Json.JsonSerializer.Serialize(sources,
                    new System.Text.Json.JsonSerializerOptions { WriteIndented = true }));
                File.Move(temporary, file, overwrite: true);
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                _logger.LogWarning(LogEventIds.SkillRejected,
                    "skills.sources.unwritable file={File} reason={Reason}", file, ex.Message);
            }
        }

        /// <summary>Ids the user has deleted. Empty when no record file is configured.</summary>
        private HashSet<string> RemovedIds()
        {
            var removed = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            string? file = _options.RemovedRecordFile;
            if (string.IsNullOrEmpty(file) || !File.Exists(file))
                return removed;
            try
            {
                foreach (string line in File.ReadAllLines(file))
                {
                    string id = line.Trim();
                    if (id.Length > 0 && !id.StartsWith('#'))
                        removed.Add(id);
                }
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                // A record that cannot be read must not take the registry down with it;
                // the cost is a deleted skill coming back, which is visible and fixable.
                _logger.LogWarning(LogEventIds.SkillRejected,
                    "skills.removed.unreadable file={File} reason={Reason}", file, ex.Message);
            }
            return removed;
        }

        private void RecordRemoved(string id)
        {
            string file = _options.RemovedRecordFile!;
            HashSet<string> removed = RemovedIds();
            if (!removed.Add(id))
                return;
            try
            {
                string? directory = Path.GetDirectoryName(file);
                if (!string.IsNullOrEmpty(directory))
                    Directory.CreateDirectory(directory);
                File.WriteAllLines(file, removed.OrderBy(x => x, StringComparer.Ordinal));
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                throw new InvalidOperationException(
                    $"'{id}' could not be removed: the record of deleted skills at {file} is not writable.", ex);
            }
        }

        /// <summary>
        /// Forget that <paramref name="id"/> was ever deleted, so installing it again
        /// works. Without this a user who removed a built-in skill and then installed
        /// their own copy of it would watch the install succeed and the skill not
        /// appear.
        /// </summary>
        private void ForgetRemoved(string id)
        {
            string? file = _options.RemovedRecordFile;
            if (string.IsNullOrEmpty(file) || !File.Exists(file))
                return;
            HashSet<string> removed = RemovedIds();
            if (!removed.Remove(id))
                return;
            try { File.WriteAllLines(file, removed.OrderBy(x => x, StringComparer.Ordinal)); }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                _logger.LogWarning(LogEventIds.SkillRejected,
                    "skills.removed.unwritable file={File} reason={Reason}", file, ex.Message);
            }
        }

        // ---- discovery ---------------------------------------------------------

        private Index Build()
        {
            var byId = new Dictionary<string, Skill>(StringComparer.OrdinalIgnoreCase);
            var ordered = new List<Skill>();
            var errors = new List<SkillLoadError>();
            var roots = new List<string>();
            var visited = new HashSet<string>(SkillPathGuard.PathComparison == StringComparison.Ordinal
                ? StringComparer.Ordinal
                : StringComparer.OrdinalIgnoreCase);
            HashSet<string> removed = RemovedIds();
            Dictionary<string, SkillSource> sources = ReadSources();

            foreach ((string root, SkillOrigin origin) in EnumerateRoots())
            {
                string normalized;
                try
                {
                    normalized = SkillPathGuard.NormalizeDirectory(root);
                }
                catch (Exception ex) when (ex is ArgumentException or NotSupportedException or PathTooLongException)
                {
                    errors.Add(new SkillLoadError(root, $"not a usable path: {ex.Message}"));
                    continue;
                }

                if (!visited.Add(normalized))
                    continue;
                roots.Add(normalized);

                if (!Directory.Exists(normalized))
                {
                    // A missing install directory is normal on a fresh machine; a
                    // missing operator root is a configuration mistake worth reporting.
                    if (origin == SkillOrigin.Discovered)
                        errors.Add(new SkillLoadError(normalized, "directory does not exist"));
                    continue;
                }

                foreach (string dir in FindSkillDirectories(normalized, errors))
                {
                    if (ordered.Count >= _options.MaxSkills)
                    {
                        errors.Add(new SkillLoadError(dir,
                            $"skipped: the registry already holds its maximum of {_options.MaxSkills} skills"));
                        break;
                    }

                    if (!TryLoad(dir, origin, normalized, out Skill? skill, out string? error))
                    {
                        errors.Add(new SkillLoadError(dir, error!));
                        continue;
                    }

                    if (removed.Contains(skill!.Id))
                    {
                        // Deleted by the user. Not an error and not reported as one:
                        // this is the skill being gone, which is what was asked for.
                        continue;
                    }

                    if (byId.TryGetValue(skill!.Id, out Skill? winner))
                    {
                        errors.Add(new SkillLoadError(dir,
                            $"skipped: '{skill.Id}' is already provided by {winner.RootDirectory}, which takes precedence"));
                        continue;
                    }

                    // Only a skill in the install directory can carry hub provenance: the
                    // record names a directory this registry wrote, never an operator root.
                    if (origin == SkillOrigin.Installed
                        && sources.TryGetValue(skill.Id, out SkillSource? provenance)
                        && string.Equals(Path.GetFileName(dir), skill.Id, StringComparison.OrdinalIgnoreCase))
                    {
                        skill.Source = provenance;
                    }

                    byId[skill.Id] = skill;
                    ordered.Add(skill);
                }
            }

            ordered.Sort(static (a, b) => string.CompareOrdinal(a.Id, b.Id));
            return new Index(byId, ordered, errors, roots);
        }

        private IEnumerable<(string Root, SkillOrigin Origin)> EnumerateRoots()
        {
            if (!string.IsNullOrWhiteSpace(_options.InstallDirectory))
                yield return (_options.InstallDirectory!, SkillOrigin.Installed);

            foreach (string root in _options.Roots)
            {
                if (!string.IsNullOrWhiteSpace(root))
                    yield return (root, SkillOrigin.Discovered);
            }
        }

        /// <summary>
        /// Breadth-first walk for directories holding a <c>SKILL.md</c>. A directory
        /// that has one is a skill and is not descended into: a skill's own
        /// <c>references/</c> may legitimately contain example skills, and treating
        /// those as installed skills would put documentation in the model's catalog.
        /// </summary>
        private IEnumerable<string> FindSkillDirectories(string root, List<SkillLoadError> errors)
        {
            var queue = new Queue<(string Dir, int Depth)>();
            queue.Enqueue((root, 0));

            while (queue.Count > 0)
            {
                (string dir, int depth) = queue.Dequeue();

                if (File.Exists(Path.Combine(dir, SkillManifestParser.SkillFileName)))
                {
                    yield return dir;
                    continue;
                }

                if (depth >= _options.MaxDepth)
                    continue;

                string[] children;
                try
                {
                    children = Directory.GetDirectories(dir);
                }
                catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
                {
                    errors.Add(new SkillLoadError(dir, $"could not be listed: {ex.Message}"));
                    continue;
                }

                Array.Sort(children, StringComparer.Ordinal);
                foreach (string child in children)
                {
                    string name = Path.GetFileName(child);
                    if (IsIgnoredDirectory(name))
                        continue;
                    queue.Enqueue((child, depth + 1));
                }
            }
        }

        /// <summary>
        /// Directories a skills walk must never enter. <c>.git</c> alone can hold tens
        /// of thousands of files, and a dependency tree checked in next to a skill is
        /// not part of it.
        /// </summary>
        private static bool IsIgnoredDirectory(string name) =>
            name.Length == 0
            || name[0] == '.'
            || name.Equals("node_modules", StringComparison.OrdinalIgnoreCase)
            || name.Equals("__pycache__", StringComparison.OrdinalIgnoreCase);

        private bool TryLoad(
            string directory,
            SkillOrigin origin,
            string? discoveredUnder,
            out Skill? skill,
            out string? error)
        {
            skill = null;
            error = null;

            string manifestPath = Path.Combine(directory, SkillManifestParser.SkillFileName);
            string document;
            DateTime newest;
            try
            {
                var info = new FileInfo(manifestPath);
                if (!info.Exists)
                {
                    error = $"no {SkillManifestParser.SkillFileName}";
                    return false;
                }
                if (info.Length > _options.MaxManifestBytes)
                {
                    error = $"{SkillManifestParser.SkillFileName} is {SkillTextBudget.FormatBytes(info.Length)}, "
                          + $"over the {SkillTextBudget.FormatBytes(_options.MaxManifestBytes)} limit";
                    return false;
                }
                document = File.ReadAllText(manifestPath);
                newest = info.LastWriteTimeUtc;
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                error = $"{SkillManifestParser.SkillFileName} could not be read: {ex.Message}";
                return false;
            }

            if (!SkillManifestParser.TryParse(document, Path.GetFileName(directory), out SkillManifest? manifest, out error))
                return false;

            if (!TryIndexFiles(directory, out List<SkillFile> files, ref newest, out error))
                return false;

            skill = new Skill(manifest!.Name, manifest, directory, origin, discoveredUnder, files, newest);
            return true;
        }

        private bool TryIndexFiles(string directory, out List<SkillFile> files, ref DateTime newest, out string? error)
        {
            files = new List<SkillFile>();
            error = null;
            long total = 0;

            IEnumerable<string> found;
            try
            {
                found = Directory.EnumerateFiles(directory, "*", new EnumerationOptions
                {
                    RecurseSubdirectories = true,
                    // A symlinked directory inside a skill could point anywhere, and the
                    // walk itself must not follow it out of the tree (SkillPathGuard
                    // stops the reads, but the walk would still stat the whole target).
                    // The same option also ends the cycle a self-referential link creates.
                    AttributesToSkip = FileAttributes.ReparsePoint | FileAttributes.System,
                    IgnoreInaccessible = true,
                });
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                error = $"could not be listed: {ex.Message}";
                return false;
            }

            try
            {
                foreach (string path in found)
                {
                    if (files.Count >= _options.MaxSkillFiles)
                    {
                        error = $"holds more than {_options.MaxSkillFiles} files";
                        return false;
                    }

                    var info = new FileInfo(path);
                    total += info.Length;
                    if (total > _options.MaxSkillBytes)
                    {
                        error = $"is larger than the {SkillTextBudget.FormatBytes(_options.MaxSkillBytes)} limit";
                        return false;
                    }
                    if (info.LastWriteTimeUtc > newest)
                        newest = info.LastWriteTimeUtc;

                    string relative = SkillPathGuard.ToSkillRelative(directory, info.FullName);
                    files.Add(new SkillFile(
                        relative,
                        info.Length,
                        Skill.ClassifyFile(relative),
                        Skill.IsTextExtension(Path.GetExtension(relative))));
                }
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                error = $"could not be listed: {ex.Message}";
                return false;
            }

            files.Sort(static (a, b) => string.CompareOrdinal(a.Path, b.Path));
            return true;
        }

        // ---- installation ------------------------------------------------------

        private string EnsureInstallDirectory()
        {
            string dir = SkillPathGuard.NormalizeDirectory(_options.InstallDirectory!);
            Directory.CreateDirectory(dir);
            return dir;
        }

        private string ReserveInstallDirectory(string name, bool overwrite)
        {
            string installRoot = EnsureInstallDirectory();
            if (!SkillPathGuard.TryResolve(installRoot, name, out string? destination, out string? guardError))
                throw new SkillInstallException($"'{name}' is not a usable directory name: {guardError}");

            if (Directory.Exists(destination))
            {
                if (!overwrite)
                {
                    throw new SkillInstallException(
                        $"a skill called '{name}' is already installed; pass overwrite to replace it.");
                }
                TryDeleteDirectory(destination!);
            }

            // A skill of this name discovered under an operator root would be shadowed
            // rather than replaced, and silently shadowing the operator's copy is worse
            // than refusing: say so and let them choose.
            if (_index.ById.TryGetValue(name, out Skill? existing)
                && existing.Origin == SkillOrigin.Discovered
                && !overwrite)
            {
                throw new SkillInstallException(
                    $"'{name}' is already provided by {existing.RootDirectory}; " +
                    "pass overwrite to install a copy that takes precedence over it.");
            }

            return destination!;
        }

        private void CopyTree(string source, string destination, string skillName)
        {
            Directory.CreateDirectory(destination);
            long total = 0;
            int count = 0;

            foreach (string path in Directory.EnumerateFiles(source, "*", new EnumerationOptions
            {
                RecurseSubdirectories = true,
                AttributesToSkip = FileAttributes.ReparsePoint | FileAttributes.System,
                IgnoreInaccessible = true,
            }))
            {
                var info = new FileInfo(path);
                total += info.Length;
                count++;
                if (count > _options.MaxSkillFiles)
                    throw new SkillInstallException($"'{skillName}' holds more than {_options.MaxSkillFiles} files.");
                if (total > _options.MaxSkillBytes)
                {
                    throw new SkillInstallException(
                        $"'{skillName}' is larger than the {SkillTextBudget.FormatBytes(_options.MaxSkillBytes)} limit.");
                }

                string relative = SkillPathGuard.ToSkillRelative(source, info.FullName);
                if (!SkillPathGuard.TryResolve(destination, relative, out string? target, out string? guardError))
                    throw new SkillInstallException($"'{relative}' cannot be copied: {guardError}");

                Directory.CreateDirectory(Path.GetDirectoryName(target!)!);
                File.Copy(info.FullName, target!, overwrite: true);
            }
        }

        private static void TryDeleteDirectory(string directory)
        {
            try
            {
                if (Directory.Exists(directory))
                    Directory.Delete(directory, recursive: true);
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                // Best effort: a leftover staging directory is noise, not corruption,
                // and throwing here would mask whatever failure led us to clean up.
            }
        }

        /// <summary>An immutable published view. Replaced wholesale; never mutated in place.</summary>
        private sealed class Index
        {
            public static readonly Index Empty = new(
                new Dictionary<string, Skill>(StringComparer.OrdinalIgnoreCase),
                Array.Empty<Skill>(),
                Array.Empty<SkillLoadError>(),
                Array.Empty<string>());

            public Index(
                Dictionary<string, Skill> byId,
                IReadOnlyList<Skill> ordered,
                IReadOnlyList<SkillLoadError> errors,
                IReadOnlyList<string> roots)
            {
                ById = byId;
                Ordered = ordered;
                Errors = errors;
                Roots = roots;
            }

            public Dictionary<string, Skill> ById { get; }
            public IReadOnlyList<Skill> Ordered { get; }
            public IReadOnlyList<SkillLoadError> Errors { get; }
            public IReadOnlyList<string> Roots { get; }
        }
    }

    /// <summary>Thrown when a skill cannot be installed. The message is safe to show a client.</summary>
    public sealed class SkillInstallException : Exception
    {
        public SkillInstallException(string message) : base(message) { }

        public SkillInstallException(string message, Exception inner) : base(message, inner) { }

        /// <summary>The HTTP status a remote source answered with, when that is what failed.</summary>
        public int? HttpStatus { get; init; }
    }
}
