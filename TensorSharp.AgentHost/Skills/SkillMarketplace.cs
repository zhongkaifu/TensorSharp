// Copyright (c) Zhongkai Fu. Licensed under the repository's BSD-3-Clause license.
using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;

namespace TensorSharp.AgentHost.Skills;

/// <summary>A public skills.sh search result. Package is the exact install selector.</summary>
public sealed record SkillSearchResult(string Name, string Source, string Package, string Url, long Installs);

/// <summary>An inspected skill at an immutable Git commit. Instructions remain untrusted source content.</summary>
public sealed record SkillPackageInspection(
    string Package, string Source, string Revision, SkillManifest Manifest,
    IReadOnlyList<string> Files, long TotalBytes)
{
    /// <summary>The skill's directory inside the repository at <see cref="Revision"/>; empty at the root.</summary>
    public string Directory { get; init; } = string.Empty;

    /// <summary>The skill's files on GitHub, at the inspected commit.</summary>
    public string BrowseUrl => "https://github.com/" + Source + "/tree/" + Revision
        + (Directory.Length == 0 ? string.Empty : "/" + Directory);
}

/// <summary>Registration and provenance of one newly installed skill.</summary>
public sealed record SkillMarketplaceInstall(Skill Skill, string Package, string Source, string Revision);

/// <summary>Resource limits for the public marketplace transport.</summary>
public sealed class SkillMarketplaceOptions
{
    public int MaxJsonBytes { get; init; } = 1024 * 1024;
    public int MaxArchiveBytes { get; init; } = 32 * 1024 * 1024;
    public int MaxManifestBytes { get; init; } = 1024 * 1024;
    public SkillArchiveLimits ArchiveLimits { get; init; } = new()
    {
        MaxTotalBytes = 128L * 1024 * 1024,
        MaxEntryBytes = 16L * 1024 * 1024,
        MaxEntries = 8192,
    };
    public TimeSpan RequestTimeout { get; init; } = TimeSpan.FromSeconds(60);
    public TimeSpan CacheLifetime { get; init; } = TimeSpan.FromMinutes(5);
    public int SearchCacheEntries { get; init; } = 64;
    public int PackageCacheEntries { get; init; } = 2;
    public string? TemporaryDirectory { get; init; }

    /// <summary>Optional live host policy, checked before every HTTP request, including cached operations.</summary>
    public Func<Uri, bool>? IsRequestAllowed { get; init; }
}

/// <summary>
/// Finds public skills through skills.sh and installs their original GitHub files
/// without running npm, git, install hooks, or repository programs. A host must
/// separately authorize network access and installation before offering this client
/// to a model. Inspect-then-install pins the inspected commit even if HEAD changes.
/// </summary>
public sealed class SkillMarketplace : IDisposable
{
    public static SkillMarketplace Shared { get; } = new();

    private static readonly Regex OwnerPattern = new("^[A-Za-z0-9][A-Za-z0-9-]{0,38}\\z", RegexOptions.CultureInvariant);
    private static readonly Regex RepoPattern = new("^[A-Za-z0-9][A-Za-z0-9._-]{0,99}\\z", RegexOptions.CultureInvariant);
    private static readonly Regex SkillPattern = new("^[a-z0-9]+(?:-[a-z0-9]+)*\\z", RegexOptions.CultureInvariant);
    private static readonly Regex RevisionPattern = new("^[a-fA-F0-9]{40}\\z", RegexOptions.CultureInvariant);
    private static readonly Regex DevicePattern = new("^(CON|PRN|AUX|NUL|CONIN\\$|CONOUT\\$|COM[1-9¹²³]|LPT[1-9¹²³])(?:\\..*)?\\z",
        RegexOptions.IgnoreCase | RegexOptions.CultureInvariant);
    private readonly HttpClient _http;
    private readonly bool _ownsHttp;
    private readonly SkillMarketplaceOptions _options;
    private readonly object _gate = new();
    private readonly Dictionary<string, SearchCache> _searches = new(StringComparer.Ordinal);
    private readonly Dictionary<string, PackageCache> _packages = new(StringComparer.OrdinalIgnoreCase);
    private bool _disposed;

    /// <param name="httpClient">
    /// Optional host/test transport. It must disable automatic redirects; the default
    /// does so. No credentials or authorization headers are added by this client.
    /// </param>
    public SkillMarketplace(HttpClient? httpClient = null, SkillMarketplaceOptions? options = null)
    {
        _options = options ?? new SkillMarketplaceOptions();
        if (_options.MaxJsonBytes <= 0 || _options.MaxArchiveBytes <= 0 || _options.MaxManifestBytes <= 0
            || _options.RequestTimeout <= TimeSpan.Zero || _options.CacheLifetime < TimeSpan.Zero
            || _options.SearchCacheEntries < 0 || _options.PackageCacheEntries < 0
            || _options.ArchiveLimits.MaxTotalBytes <= 0 || _options.ArchiveLimits.MaxEntryBytes <= 0
            || _options.ArchiveLimits.MaxEntries <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), "Marketplace limits must be positive; cache sizes may be zero.");
        _ownsHttp = httpClient == null;
        _http = httpClient ?? new HttpClient(new HttpClientHandler { AllowAutoRedirect = false })
        {
            Timeout = Timeout.InfiniteTimeSpan,
        };
    }

    public async Task<IReadOnlyList<SkillSearchResult>> SearchAsync(
        string query, string? owner = null, CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        cancellationToken.ThrowIfCancellationRequested();
        query = (query ?? string.Empty).Trim();
        if (query.Length < 2 || query.Length > 256 || query.Any(char.IsControl))
            throw new SkillInstallException("A search query must contain 2–256 characters without control characters.");
        owner = string.IsNullOrWhiteSpace(owner) ? null : owner.Trim().ToLowerInvariant();
        // "owner/repository" (or a package) names its owner first: filter by that.
        if (owner != null && owner.IndexOfAny(new[] { '/', '@' }) > 0)
            owner = owner.Split('/', '@')[0];
        if (owner != null && !OwnerPattern.IsMatch(owner))
            throw new SkillInstallException("The owner filter must be a GitHub owner name.");
        var uri = new Uri("https://skills.sh/api/search?q=" + Uri.EscapeDataString(query) + "&limit=20"
            + (owner == null ? string.Empty : "&owner=" + Uri.EscapeDataString(owner)));
        Authorize(uri);
        string key = uri.AbsoluteUri;
        lock (_gate)
        {
            if (_searches.TryGetValue(key, out SearchCache? cached) && IsFresh(cached.Created))
                return cached.Results;
        }

        byte[] payload = await DownloadAsync(uri, _options.MaxJsonBytes, cancellationToken).ConfigureAwait(false);
        using JsonDocument json = ParseJson(payload, "skills.sh search");
        if (!json.RootElement.TryGetProperty("skills", out JsonElement items) || items.ValueKind != JsonValueKind.Array)
            throw new SkillInstallException("The skills.sh response has no skills array.");
        var results = new List<SkillSearchResult>();
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (JsonElement item in items.EnumerateArray())
        {
            if (item.ValueKind != JsonValueKind.Object) continue;
            string? name = StringProperty(item, "name");
            string? source = StringProperty(item, "source");
            if (!TryPackage(source + "@" + name, out PackageId? id)) continue;
            if (owner != null && !string.Equals(id!.Owner, owner, StringComparison.OrdinalIgnoreCase)) continue;
            if (!seen.Add(id!.Package)) continue;
            long installs = item.TryGetProperty("installs", out JsonElement count) && count.ValueKind == JsonValueKind.Number && count.TryGetInt64(out long n)
                ? Math.Max(0, n) : 0;
            results.Add(new SkillSearchResult(id.Name, id.Source, id.Package,
                "https://skills.sh/" + id.Source + "/" + id.Name, installs));
        }
        if (items.GetArrayLength() > 0 && results.Count == 0)
            throw new SkillInstallException("The search returned no supported public GitHub skill packages.");
        IReadOnlyList<SkillSearchResult> output = results.OrderByDescending(r => r.Installs)
            .ThenBy(r => r.Package, StringComparer.Ordinal).Take(20).ToList().AsReadOnly();
        lock (_gate)
        {
            if (_options.SearchCacheEntries > 0)
            {
                EvictOldest(_searches, _options.SearchCacheEntries, p => p.Created);
                _searches[key] = new SearchCache(DateTimeOffset.UtcNow, output);
            }
        }
        return output;
    }

    public async Task<SkillPackageInspection> InspectAsync(string package, CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        cancellationToken.ThrowIfCancellationRequested();
        PackageId id = ParsePackage(package);
        Authorize(CommitUri(id));
        Authorize(ArchiveUri(id, new string('0', 40)));
        lock (_gate)
        {
            if (_packages.TryGetValue(id.Package, out PackageCache? cached) && IsFresh(cached.Created))
                return cached.Inspection;
        }
        // Asked for the commit id alone (application/vnd.github.sha): the full commit JSON
        // carries the commit's whole file list and diff, which for a large commit in a skills
        // monorepo exceeded the JSON limit and made the package uninstallable.
        byte[] commit;
        try
        {
            commit = await DownloadAsync(CommitUri(id), _options.MaxJsonBytes, cancellationToken,
                accept: "application/vnd.github.sha").ConfigureAwait(false);
        }
        catch (SkillInstallException ex) when (ex.HttpStatus is 404 or 410 or 451)
        {
            // No such repository (or it is gone): no skill in it can be fetched.
            throw AsRepositoryWide(ex);
        }
        string? revision;
        string text = Encoding.UTF8.GetString(commit).Trim();
        if (text.StartsWith('{'))
        {
            using JsonDocument json = ParseJson(commit, "GitHub commit");
            revision = StringProperty(json.RootElement, "sha");
        }
        else
        {
            revision = text;
        }
        if (revision == null || !RevisionPattern.IsMatch(revision))
            throw new SkillInstallException("GitHub did not return an immutable 40-character commit revision.") { RepositoryWide = true };
        PackageCache prepared = await FetchPackageAsync(id, revision.ToLowerInvariant(), cancellationToken).ConfigureAwait(false);
        CachePackage(id.Package, prepared);
        return prepared.Inspection;
    }

    public async Task<SkillMarketplaceInstall> InstallAsync(
        string package, SkillRegistry registry, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(registry);
        if (!registry.CanInstall) throw new SkillInstallException("This host has no skill install directory.");
        SkillPackageInspection inspection = await InspectAsync(package, cancellationToken).ConfigureAwait(false);
        return await InstallAsync(inspection, registry, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>Install exactly the inspected revision; never overwrite an existing skill.</summary>
    public Task<SkillMarketplaceInstall> InstallAsync(
        SkillPackageInspection inspection, SkillRegistry registry, CancellationToken cancellationToken = default) =>
        InstallAsync(inspection, registry, installAs: null, beforeInstall: null, cancellationToken);

    /// <summary>
    /// Install exactly the inspected revision, never overwriting an existing skill, and
    /// record where it came from (<see cref="SkillSource"/>).
    /// </summary>
    /// <param name="installAs">
    /// Install under this name instead of the upstream one — the only edit ever made to a
    /// downloaded file is that SKILL.md's <c>name</c> line, recorded as
    /// <see cref="SkillSource.RenamedFrom"/>. Null keeps the upstream name.
    /// </param>
    /// <param name="beforeInstall">
    /// Called with the extracted skill directory and its manifest before anything is
    /// registered; a non-null answer refuses the install with that message.
    /// </param>
    public async Task<SkillMarketplaceInstall> InstallAsync(
        SkillPackageInspection inspection, SkillRegistry registry, string? installAs,
        Func<string, SkillManifest, string?>? beforeInstall, CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(inspection);
        ArgumentNullException.ThrowIfNull(registry);
        cancellationToken.ThrowIfCancellationRequested();
        if (!registry.CanInstall) throw new SkillInstallException("This host has no skill install directory.");
        PackageId id = ParsePackage(inspection.Package);
        if (!RevisionPattern.IsMatch(inspection.Revision ?? string.Empty)
            || !string.Equals(id.Source, inspection.Source, StringComparison.OrdinalIgnoreCase)
            || !string.Equals(id.Name, inspection.Manifest.Name, StringComparison.Ordinal))
            throw new SkillInstallException("The inspected package or revision is inconsistent.");
        string name = string.IsNullOrWhiteSpace(installAs) ? id.Name : installAs.Trim();
        if (!SkillPattern.IsMatch(name) || name.Length > SkillManifest.MaxNameLength)
            throw new SkillInstallException($"'{name}' is not a usable skill name.");
        if (registry.TryGet(name, out _))
            throw new SkillInstallException($"Skill '{name}' already exists; marketplace installation never overwrites it.");
        Authorize(ArchiveUri(id, inspection.Revision!));
        PackageCache? prepared;
        lock (_gate)
        {
            _packages.TryGetValue(id.Package, out prepared);
            if (prepared != null && !string.Equals(prepared.Inspection.Revision, inspection.Revision, StringComparison.OrdinalIgnoreCase))
                prepared = null;
        }
        prepared ??= await FetchPackageAsync(id, inspection.Revision!, cancellationToken).ConfigureAwait(false);
        string temporary = Path.Combine(_options.TemporaryDirectory ?? Path.GetTempPath(), "ts-marketplace-" + Guid.NewGuid().ToString("N"));
        try
        {
            using var memory = new MemoryStream(prepared.Archive, writable: false);
            using var cancellable = new CancellableStream(memory, cancellationToken);
            SkillArchive.Extract(cancellable, temporary, _options.ArchiveLimits);
            if (!SkillPathGuard.TryResolve(temporary, prepared.SkillDirectory, out string? skillRoot, out string? error))
                throw new SkillInstallException(error ?? "The selected skill directory is unsafe.");
            cancellationToken.ThrowIfCancellationRequested();
            if (!string.Equals(name, id.Name, StringComparison.Ordinal))
                RenameManifest(Path.Combine(skillRoot!, SkillManifestParser.SkillFileName), name);
            if (beforeInstall?.Invoke(skillRoot!, prepared.Inspection.Manifest) is { } refusal)
                throw new SkillInstallException(refusal);
            var source = new SkillSource(id.Package, prepared.Inspection.BrowseUrl, prepared.Inspection.Revision,
                DateTimeOffset.UtcNow, string.Equals(name, id.Name, StringComparison.Ordinal) ? null : id.Name);
            Skill installed = registry.InstallFromDirectory(skillRoot!, overwrite: false, source);
            return new SkillMarketplaceInstall(installed, id.Package, id.Source, prepared.Inspection.Revision);
        }
        finally
        {
            if (Directory.Exists(temporary)) Directory.Delete(temporary, recursive: true);
        }
    }

    private async Task<PackageCache> FetchPackageAsync(PackageId id, string revision, CancellationToken cancellationToken)
    {
        byte[] archive;
        try
        {
            archive = await DownloadAsync(ArchiveUri(id, revision), _options.MaxArchiveBytes, cancellationToken).ConfigureAwait(false);
        }
        catch (SkillInstallException ex) when (ex.HttpStatus is not (403 or 429))
        {
            // The archive is the whole repository's (missing, or over the download limit). A
            // refused request (rate limit) is about this host's requests, not the repository.
            throw AsRepositoryWide(ex);
        }
        try
        {
            using var memory = new MemoryStream(archive, writable: false);
            using var zip = new ZipArchive(memory, ZipArchiveMode.Read);
            // GitHub writes the archived commit's id as the ZIP comment. Requiring it to be
            // the commit asked for binds these bytes to the pinned revision: a cache or
            // proxy answering with another snapshot of the repository is refused.
            if (!string.Equals(zip.Comment?.Trim(), revision, StringComparison.OrdinalIgnoreCase))
                throw new SkillInstallException(
                    $"The archive GitHub returned is not commit {revision} (it is labelled '{Clip(zip.Comment, 64)}'); nothing was installed.")
                {
                    RepositoryWide = true,
                };
            try
            {
                ValidateEntries(zip);
            }
            catch (SkillInstallException ex)
            {
                // Every limit here is the repository archive's, whichever skill was asked for.
                throw AsRepositoryWide(ex);
            }
            var matches = new List<(string Directory, SkillManifest Manifest)>();
            foreach (ZipArchiveEntry entry in zip.Entries)
            {
                cancellationToken.ThrowIfCancellationRequested();
                string path = entry.FullName.Replace('\\', '/');
                if (!string.Equals(path.Split('/').Last(), SkillManifestParser.SkillFileName, StringComparison.Ordinal)) continue;
                if (IsSpecialFile(entry)) continue;
                // Nested example manifests are not independent skills. Match the same
                // boundary rule used by local discovery, even for hidden agent roots.
                if (HasParentManifest(path, zip)) continue;
                if (entry.Length > _options.MaxManifestBytes)
                    throw new SkillInstallException($"Manifest '{path}' exceeds the marketplace manifest byte limit.");
                using Stream source = entry.Open();
                string document = Encoding.UTF8.GetString(await ReadCappedAsync(source, _options.MaxManifestBytes, cancellationToken).ConfigureAwait(false));
                if (SkillManifestParser.TryParse(document, null, out SkillManifest? manifest, out _)
                    && string.Equals(manifest!.Name, id.Name, StringComparison.Ordinal))
                {
                    int slash = path.LastIndexOf('/');
                    matches.Add((slash < 0 ? string.Empty : path.Substring(0, slash), manifest));
                }
            }
            if (matches.Count == 0)
                throw new SkillInstallException($"No skill whose manifest name is '{id.Name}' exists in {id.Source} at {revision}.");
            if (matches.Count != 1)
                throw new SkillInstallException($"The repository contains multiple skills named '{id.Name}'; the package is ambiguous.");
            var selected = matches[0];
            string prefix = selected.Directory.Length == 0 ? string.Empty : selected.Directory + "/";
            ZipArchiveEntry[] files = zip.Entries.Where(e => !IsDirectory(e.FullName)
                && e.FullName.Replace('\\', '/').StartsWith(prefix, StringComparison.Ordinal)).ToArray();
            foreach (ZipArchiveEntry entry in zip.Entries.Where(e => e.FullName.Replace('\\', '/').StartsWith(prefix, StringComparison.Ordinal)))
            {
                if (IsSpecialFile(entry))
                    throw new SkillInstallException($"Selected skill entry '{entry.FullName}' is a symbolic link or special file.");
            }
            var paths = files.Select(e => e.FullName.Replace('\\', '/').Substring(prefix.Length))
                .OrderBy(p => p, StringComparer.Ordinal).ToList().AsReadOnly();
            // Keep only the selected skill. Repository-wide support files can contain
            // symlinks (for example Vercel's root CLAUDE.md), and unrelated files must
            // neither block a safe skill nor be extracted during its installation.
            // Repacking changes only ZIP names/metadata; every source byte is copied.
            using var skillArchive = new MemoryStream();
            long total = 0;
            using (var selectedZip = new ZipArchive(skillArchive, ZipArchiveMode.Create, leaveOpen: true))
            {
                byte[] buffer = new byte[81920];
                foreach (ZipArchiveEntry entry in files)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    string relative = entry.FullName.Replace('\\', '/').Substring(prefix.Length);
                    ZipArchiveEntry destination = selectedZip.CreateEntry(relative, CompressionLevel.Fastest);
                    using Stream source = entry.Open();
                    using Stream target = destination.Open();
                    long written = 0;
                    int read;
                    while ((read = await source.ReadAsync(buffer.AsMemory(), cancellationToken).ConfigureAwait(false)) > 0)
                    {
                        written += read;
                        total += read;
                        if (written > _options.ArchiveLimits.MaxEntryBytes || total > _options.ArchiveLimits.MaxTotalBytes)
                            throw new SkillInstallException("The selected skill exceeds the decompressed byte limit.");
                        await target.WriteAsync(buffer.AsMemory(0, read), cancellationToken).ConfigureAwait(false);
                        if (skillArchive.Length > _options.MaxArchiveBytes)
                            throw new SkillInstallException("The selected skill exceeds the archive byte limit.");
                    }
                }
            }
            if (skillArchive.Length > _options.MaxArchiveBytes)
                throw new SkillInstallException("The selected skill exceeds the archive byte limit.");
            return new PackageCache(DateTimeOffset.UtcNow, skillArchive.ToArray(), string.Empty,
                new SkillPackageInspection(id.Package, id.Source, revision, selected.Manifest, paths, total)
                {
                    // The archive's first segment is GitHub's "<repo>-<commit>" folder.
                    Directory = selected.Directory.Contains('/')
                        ? selected.Directory.Substring(selected.Directory.IndexOf('/') + 1)
                        : string.Empty,
                });
        }
        catch (InvalidDataException ex)
        {
            throw new SkillInstallException("The GitHub download is not a valid skill ZIP archive: " + ex.Message, ex)
            {
                RepositoryWide = true,
            };
        }
    }

    private static SkillInstallException AsRepositoryWide(SkillInstallException ex) => ex.RepositoryWide
        ? ex
        : new SkillInstallException(ex.Message, ex) { HttpStatus = ex.HttpStatus, RepositoryWide = true };

    private void ValidateEntries(ZipArchive zip)
    {
        if (zip.Entries.Count == 0) throw new SkillInstallException("The repository archive is empty.");
        if (zip.Entries.Count > _options.ArchiveLimits.MaxEntries)
            throw new SkillInstallException("The repository archive exceeds the entry limit.");
        long total = 0, compressed = 0;
        var paths = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        // This directory need not exist: the lexical guard resolves no outside files.
        string root = Path.Combine(_options.TemporaryDirectory ?? Path.GetTempPath(), "ts-marketplace-validation");
        foreach (ZipArchiveEntry entry in zip.Entries)
        {
            string path = entry.FullName.Replace('\\', '/').TrimEnd('/');
            if (path.Length == 0 || path.Split('/').Any(segment => !IsPortableSegment(segment))
                || !SkillPathGuard.TryResolve(root, path, out string? resolved, out string? error))
                throw new SkillInstallException($"Unsafe repository archive entry '{entry.FullName}'.");
            if (!paths.Add(Path.GetRelativePath(root, resolved!).Replace('\\', '/')))
                throw new SkillInstallException($"Duplicate repository archive entry '{entry.FullName}'.");
            if (entry.Length > _options.ArchiveLimits.MaxEntryBytes)
                throw new SkillInstallException($"Repository archive entry '{entry.FullName}' exceeds the per-file byte limit.");
            total = checked(total + entry.Length);
            compressed = checked(compressed + entry.CompressedLength);
            if (total > _options.ArchiveLimits.MaxTotalBytes)
                throw new SkillInstallException("The repository archive exceeds the decompressed byte limit.");
        }
        if (_options.ArchiveLimits.MaxCompressionRatio > 0
            && total / Math.Max(1, compressed) > _options.ArchiveLimits.MaxCompressionRatio)
            throw new SkillInstallException("The repository archive exceeds the compression ratio limit.");
    }

    private static bool HasParentManifest(string path, ZipArchive zip)
    {
        int slash = path.LastIndexOf('/');
        string directory = slash < 0 ? string.Empty : path.Substring(0, slash);
        while (directory.Contains('/'))
        {
            directory = directory.Substring(0, directory.LastIndexOf('/'));
            if (zip.GetEntry(directory + "/" + SkillManifestParser.SkillFileName) != null) return true;
        }
        return path.Contains('/') && zip.GetEntry(SkillManifestParser.SkillFileName) != null;
    }

    private async Task<byte[]> DownloadAsync(Uri uri, int limit, CancellationToken cancellationToken, string? accept = null)
    {
        Authorize(uri);
        using var timeout = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        timeout.CancelAfter(_options.RequestTimeout);
        using var request = new HttpRequestMessage(HttpMethod.Get, uri);
        request.Headers.UserAgent.ParseAdd("TensorSharp-Skills/1.0");
        request.Headers.Accept.ParseAdd(accept ?? (uri.Host == "codeload.github.com" ? "application/zip" : "application/json"));
        try
        {
            using HttpResponseMessage response = await _http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, timeout.Token).ConfigureAwait(false);
            if (response.RequestMessage?.RequestUri is { } final && final != uri)
                throw new SkillInstallException("Marketplace redirects are not supported; use a transport with redirects disabled.");
            if (!response.IsSuccessStatusCode)
                throw new SkillInstallException($"{uri.Host} returned HTTP {(int)response.StatusCode} ({response.ReasonPhrase}).")
                {
                    HttpStatus = (int)response.StatusCode,
                };
            if (response.Content.Headers.ContentLength is long length && length > limit)
                throw new SkillInstallException($"{uri.Host} returned more than the {limit}-byte download limit.");
            await using Stream body = await response.Content.ReadAsStreamAsync(timeout.Token).ConfigureAwait(false);
            return await ReadCappedAsync(body, limit, timeout.Token).ConfigureAwait(false);
        }
        catch (OperationCanceledException ex) when (!cancellationToken.IsCancellationRequested)
        {
            throw new SkillInstallException($"Marketplace request to {uri.Host} timed out.", ex);
        }
        catch (HttpRequestException ex)
        {
            throw new SkillInstallException($"Marketplace request to {uri.Host} failed: {ex.Message}", ex);
        }
    }

    private static async Task<byte[]> ReadCappedAsync(Stream source, int limit, CancellationToken cancellationToken)
    {
        using var memory = new MemoryStream();
        byte[] buffer = new byte[Math.Min(limit + 1L, 81920)];
        int read;
        while ((read = await source.ReadAsync(buffer.AsMemory(), cancellationToken).ConfigureAwait(false)) > 0)
        {
            if (memory.Length + read > limit)
                throw new SkillInstallException($"The marketplace response exceeds the {limit}-byte limit.");
            memory.Write(buffer, 0, read);
        }
        return memory.ToArray();
    }

    private void Authorize(Uri uri)
    {
        if (uri.Scheme != Uri.UriSchemeHttps || !uri.IsDefaultPort || uri.UserInfo.Length > 0
            || uri.Host is not ("skills.sh" or "api.github.com" or "codeload.github.com"))
            throw new SkillInstallException("Marketplace requests are restricted to skills.sh and public GitHub HTTPS endpoints.");
        if (_options.IsRequestAllowed != null && !_options.IsRequestAllowed(uri))
            throw new SkillInstallException($"The host network policy does not allow {uri.Host}.");
    }

    private static string Clip(string? value, int max) =>
        string.IsNullOrEmpty(value) ? string.Empty : value.Length <= max ? value : value.Substring(0, max);

    /// <summary>
    /// Give a downloaded SKILL.md another name: its frontmatter's <c>name</c> line, and
    /// nothing else, is rewritten.
    /// </summary>
    internal static void RenameManifest(string manifestPath, string name)
    {
        string text = File.ReadAllText(manifestPath);
        string newline = text.Contains("\r\n", StringComparison.Ordinal) ? "\r\n" : "\n";
        string[] lines = text.Split('\n');
        if (lines.Length < 2 || lines[0].TrimEnd('\r').Trim() != "---")
            throw new SkillInstallException("The skill's SKILL.md has no frontmatter to rename.");
        for (int i = 1; i < lines.Length; i++)
        {
            string line = lines[i].TrimEnd('\r');
            if (line.Trim() == "---")
            {
                var list = lines.ToList();
                list.Insert(i, "name: " + name + (newline == "\r\n" ? "\r" : string.Empty));
                File.WriteAllText(manifestPath, string.Join('\n', list));
                return;
            }
            if (Regex.IsMatch(line, "^name\\s*:"))
            {
                lines[i] = "name: " + name + (lines[i].EndsWith('\r') ? "\r" : string.Empty);
                File.WriteAllText(manifestPath, string.Join('\n', lines));
                return;
            }
        }
        throw new SkillInstallException("The skill's SKILL.md frontmatter is not closed.");
    }

    private static JsonDocument ParseJson(byte[] payload, string source)
    {
        try
        {
            JsonDocument result = JsonDocument.Parse(payload, new JsonDocumentOptions { MaxDepth = 32 });
            if (result.RootElement.ValueKind == JsonValueKind.Object) return result;
            result.Dispose();
            throw new SkillInstallException(source + " returned a JSON value that is not an object.");
        }
        catch (JsonException ex) { throw new SkillInstallException(source + " returned invalid JSON.", ex); }
    }

    private static string? StringProperty(JsonElement item, string key) =>
        item.ValueKind == JsonValueKind.Object && item.TryGetProperty(key, out JsonElement value)
        && value.ValueKind == JsonValueKind.String ? value.GetString() : null;

    private static PackageId ParsePackage(string package) => TryPackage(package, out PackageId? id) ? id!
        : throw new SkillInstallException("A package must be owner/repo@skill-name for a public GitHub repository; URLs and shell arguments are not accepted.");

    private static bool TryPackage(string? package, out PackageId? id)
    {
        id = null;
        if (package == null || package.Length > 210 || package.Any(char.IsWhiteSpace)) return false;
        string[] selector = package.Split('@');
        if (selector.Length != 2 || selector[1].Length > SkillManifest.MaxNameLength
            || !SkillPattern.IsMatch(selector[1]) || !IsPortableSegment(selector[1])) return false;
        string[] source = selector[0].Split('/');
        if (source.Length != 2 || !OwnerPattern.IsMatch(source[0]) || !RepoPattern.IsMatch(source[1])
            || source[1].EndsWith(".git", StringComparison.OrdinalIgnoreCase)) return false;
        id = new PackageId(source[0].ToLowerInvariant(), source[1].ToLowerInvariant(), selector[1]);
        return true;
    }

    private static Uri CommitUri(PackageId id) => new("https://api.github.com/repos/" + id.Source + "/commits/HEAD");
    private static Uri ArchiveUri(PackageId id, string revision) => new("https://codeload.github.com/" + id.Source + "/zip/" + revision);
    private static bool IsDirectory(string path) => path.EndsWith('/') || path.EndsWith('\\');
    private static bool IsPortableSegment(string segment) => !DevicePattern.IsMatch(segment)
        && !segment.Any(c => char.IsControl(c) || c is '<' or '>' or ':' or '"' or '|' or '?' or '*');
    private static bool IsSpecialFile(ZipArchiveEntry entry)
    {
        int unixKind = (entry.ExternalAttributes >> 16) & 0xF000;
        return (unixKind != 0 && unixKind != 0x8000 && unixKind != 0x4000)
            || (entry.ExternalAttributes & (int)FileAttributes.ReparsePoint) != 0;
    }
    private bool IsFresh(DateTimeOffset created) => DateTimeOffset.UtcNow - created < _options.CacheLifetime;
    private void CachePackage(string key, PackageCache value)
    {
        lock (_gate)
        {
            if (_options.PackageCacheEntries == 0) return;
            EvictOldest(_packages, _options.PackageCacheEntries, p => p.Created);
            _packages[key] = value;
        }
    }
    private static void EvictOldest<T>(Dictionary<string, T> cache, int cap, Func<T, DateTimeOffset> created)
    {
        while (cache.Count >= cap) cache.Remove(cache.OrderBy(p => created(p.Value)).First().Key);
    }
    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed, this);
    public void Dispose()
    {
        _disposed = true;
        lock (_gate) { _searches.Clear(); _packages.Clear(); }
        if (_ownsHttp) _http.Dispose();
    }

    private sealed record PackageId(string Owner, string Repository, string Name)
    {
        public string Source => Owner + "/" + Repository;
        public string Package => Source + "@" + Name;
    }
    private sealed record SearchCache(DateTimeOffset Created, IReadOnlyList<SkillSearchResult> Results);
    private sealed record PackageCache(DateTimeOffset Created, byte[] Archive, string SkillDirectory, SkillPackageInspection Inspection);

    private sealed class CancellableStream(Stream inner, CancellationToken cancellationToken) : Stream
    {
        public override bool CanRead => inner.CanRead;
        public override bool CanSeek => inner.CanSeek;
        public override bool CanWrite => false;
        public override long Length => inner.Length;
        public override long Position { get => inner.Position; set => inner.Position = value; }
        public override int Read(byte[] buffer, int offset, int count) { cancellationToken.ThrowIfCancellationRequested(); return inner.Read(buffer, offset, count); }
        public override int Read(Span<byte> buffer) { cancellationToken.ThrowIfCancellationRequested(); return inner.Read(buffer); }
        public override long Seek(long offset, SeekOrigin origin) { cancellationToken.ThrowIfCancellationRequested(); return inner.Seek(offset, origin); }
        public override void Flush() { }
        public override void SetLength(long value) => throw new NotSupportedException();
        public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
    }
}
