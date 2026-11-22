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
using System.Globalization;
using System.IO.Compression;
using System.Net;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using TensorAgent.Core.Sandbox;

namespace TensorAgent.Core.Python;

/// <summary>
/// One file PyPI publishes for one version of one package.
/// </summary>
/// <param name="FileName">The wheel or archive name, which is also where its tags are.</param>
/// <param name="Url">Where to fetch it.</param>
/// <param name="Sha256">The digest the index promises, hex, lower case.</param>
/// <param name="PackageType">PyPI's own word: <c>bdist_wheel</c>, <c>sdist</c>.</param>
/// <param name="Yanked">Whether the index has withdrawn it.</param>
/// <param name="RequiresPython">The file's optional interpreter-version constraint.</param>
internal sealed record PyPiFile(
    string FileName,
    string Url,
    string Sha256,
    string PackageType,
    bool Yanked,
    string RequiresPython)
{
    public IReadOnlyList<string> DeclaredDependencies { get; init; } = [];
    public bool ConditionalDependenciesOmitted { get; init; }
}

/// <summary>
/// The host's <c>pip install</c>: pure-Python wheels, fetched and unpacked here,
/// because there is no pip on this host and no process to run one in.
///
/// <para>
/// It installs less than pip does, on purpose. A wheel with compiled extensions
/// cannot be installed at all — iOS will not load a dynamic library that was not
/// signed into the app bundle, which is why the binary packages the app needs
/// (<c>numpy</c>, <c>Pillow</c>, and <c>lxml</c>, which the repository compiles
/// itself) are staged at build time by <c>prepare-python.sh</c> and why anything
/// else must be pure Python. A request for one of those never reaches this class:
/// <see cref="InstallHookPackageInstaller"/> answers for the bundle first. And
/// dependencies are not resolved: a name the model did not ask for is a name the
/// user did not agree to fetch, so a missing dependency is reported and left for
/// the model to ask for by name.
/// </para>
/// </summary>
public sealed class WheelInstaller : IInstallHook
{
    /// <summary>The index. Fixed here, not read from the command line: the shell already refuses <c>--index-url</c>.</summary>
    internal const string IndexHost = "pypi.org";

    /// <summary>PyPI's normal wheel payload host; it must be allowed as well as its JSON index.</summary>
    internal const string PayloadHost = "files.pythonhosted.org";

    private const string IndexRoot = "https://pypi.org/pypi";

    // These are deliberately below the practical memory and storage ceilings of a
    // phone. Pure-Python wheels are normally tiny; a package outside these bounds is
    // better staged into the app than allowed to consume an unbounded session quota.
    // PyPI's legacy JSON endpoint includes release history; popular pure packages
    // such as boto3 and awscli legitimately cross 3 MiB.
    internal const int MaxIndexBytes = 8 * 1024 * 1024;
    internal const long MaxWheelBytes = 64L * 1024 * 1024;
    internal const long MaxExpandedBytes = 256L * 1024 * 1024;
    internal const int MaxArchiveFiles = 10_000;
    private const int CopyBufferBytes = 64 * 1024;
    private const int EmbeddedPythonMajor = 3;
    private const int EmbeddedPythonMinor = 13;
    private const string EmbeddedPythonVersion = "3.13.14";
    private const int MaxRequiresPythonCharacters = 256;
    private const int MaxRequiresPythonClauses = 16;
    private const int MaxVersionComponents = 8;
    private const int MaxDependencyMetadataEntries = 64;
    private const int MaxDependencyMetadataCharacters = 512;
    private const int MaxDependencyNameCharacters = 128;
    private const int MaxDependenciesInNotice = 8;
    private const int MaxDiagnosticMetadataCharacters = 256;
    private const int MaxRemoteFileNameCharacters = 512;
    private const int MaxRemoteUrlCharacters = 4096;

    private static readonly HttpClient SharedHttp = new(new SocketsHttpHandler
    {
        AutomaticDecompression = System.Net.DecompressionMethods.All,
        // Redirects are followed below, one hop at a time, only after the next
        // destination has passed the session's host policy. Letting HttpClient do it
        // here would send the redirected request before the policy could inspect it.
        AllowAutoRedirect = false,
        ConnectTimeout = TimeSpan.FromSeconds(20),
    })
    {
        Timeout = TimeSpan.FromMinutes(2),
    };

    private readonly HttpClient _http;
    private readonly int[] _runtimeRelease;
    private readonly string _runtimeVersion;

    /// <summary>
    /// Bound to the session's policy, because <see cref="CanInstall"/> has no
    /// argument and the answer depends entirely on whether that session may
    /// reach the network.
    /// </summary>
    public WheelInstaller(
        ExecutionPolicy policy, HttpClient? http = null, string? pythonVersion = null)
    {
        Policy = policy ?? throw new ArgumentNullException(nameof(policy));
        _http = http ?? SharedHttp;
        string suppliedVersion = string.IsNullOrWhiteSpace(pythonVersion)
            ? EmbeddedPythonVersion
            : pythonVersion.Trim();
        string release = new(suppliedVersion
            .TakeWhile(character => char.IsAsciiDigit(character) || character == '.')
            .ToArray());
        release = release.TrimEnd('.');
        if (!TryParseRelease(release, out _runtimeRelease, out string? versionError)
            || _runtimeRelease.Length < 2
            || _runtimeRelease[0] != EmbeddedPythonMajor
            || _runtimeRelease[1] != EmbeddedPythonMinor)
        {
            throw new ArgumentException(
                $"the wheel installer requires a CPython {EmbeddedPythonMajor}.{EmbeddedPythonMinor} "
                + $"runtime version, but got '{suppliedVersion}'"
                + (versionError is null ? string.Empty : $": {versionError}"),
                nameof(pythonVersion));
        }
        _runtimeVersion = string.Join('.', _runtimeRelease);
    }

    /// <summary>
    /// The standing policy this installer answers <see cref="CanInstall"/> from.
    ///
    /// <para>
    /// Settable because on a phone the user owns it. "Allow network access" is a switch
    /// on a settings screen, and an installer built once at startup answers "the network
    /// is off" for the rest of the launch however many times the switch is flipped --
    /// which is a switch that does nothing, reported as one. An install already running
    /// keeps the policy it was launched with; only the standing answer moves.
    /// </para>
    /// </summary>
    public ExecutionPolicy Policy { get; set; }

    /// <inheritdoc />
    public bool CanInstall => Policy.AllowNetwork
        && Policy.IsHostAllowed(IndexHost)
        && Policy.IsHostAllowed(PayloadHost);

    /// <inheritdoc />
    public string? UnavailableReason
    {
        get
        {
            if (!Policy.AllowNetwork)
                return ExecutionPolicy.NetworkDisabledMessage;
            if (!Policy.IsHostAllowed(IndexHost))
                return $"{IndexHost} {ExecutionPolicy.HostNotAllowedSuffix}";
            if (!Policy.IsHostAllowed(PayloadHost))
                return $"{PayloadHost} {ExecutionPolicy.HostNotAllowedSuffix}";
            return null;
        }
    }

    /// <summary>
    /// Why <paramref name="name"/> can never be installed on this device -- every
    /// published file is compiled or a source build, or targets another Python -- from the
    /// same index entry and the same selection an install makes, without downloading
    /// anything else. Null when a compatible pure-Python wheel exists, and when the answer
    /// is unknown (the network is off, the index did not answer): only a definite "no" is
    /// a reason, because a skill must not be refused for a lookup that merely failed.
    /// </summary>
    public async Task<string?> ExplainUnavailableAsync(string name, CancellationToken cancellationToken)
    {
        if (!CanInstall || !TrySplit(name, out string package, out string? version, out _))
            return null;
        string url = version is null
            ? $"{IndexRoot}/{Uri.EscapeDataString(package)}/json"
            : $"{IndexRoot}/{Uri.EscapeDataString(package)}/{Uri.EscapeDataString(version)}/json";
        try
        {
            using HttpResponseMessage response = await GetAllowedAsync(new Uri(url), Policy, cancellationToken)
                .ConfigureAwait(false);
            if (!response.IsSuccessStatusCode)
                return null;
            string json = await ReadTextBoundedAsync(
                response.Content, MaxIndexBytes, "package metadata", cancellationToken).ConfigureAwait(false);
            return TrySelectWheel(json, _runtimeRelease, _runtimeVersion, out _, out string? why) ? null : why;
        }
        catch (Exception ex) when (ex is HttpRequestException or IOException or InvalidOperationException
            or TaskCanceledException or UnauthorizedAccessException)
        {
            return null;
        }
    }

    /// <inheritdoc />
    public async Task<ExecutionResult> InstallAsync(InstallRequest request, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(request);
        IReadOnlyDictionary<string, string> environment = new Dictionary<string, string>();
        string where = request.TargetDirectory;

        if (!string.Equals(request.Language, "python", StringComparison.OrdinalIgnoreCase))
            return ExecutionResult.Failed($"this installer only handles Python packages, not '{request.Language}'", where, environment);

        // The request's own policy decides, not the one this instance was built
        // with: the session may have changed since.
        if (!request.Policy.AllowNetwork)
            return ExecutionResult.Failed(ExecutionPolicy.NetworkDisabledMessage, where, environment);
        if (!request.Policy.IsHostAllowed(IndexHost))
            return ExecutionResult.Failed($"{IndexHost} {ExecutionPolicy.HostNotAllowedSuffix}", where, environment);
        if (!request.Policy.IsHostAllowed(PayloadHost))
            return ExecutionResult.Failed($"{PayloadHost} {ExecutionPolicy.HostNotAllowedSuffix}", where, environment);
        if (request.Packages.Count == 0)
            return ExecutionResult.Failed("no packages named", where, environment, 2);

        var log = new StringBuilder();
        void Say(string line)
        {
            log.Append(line).Append('\n');
            request.OnOutputLine?.Invoke(line);
        }

        ConfinedPaths confined;
        try
        {
            Directory.CreateDirectory(where);
            // The archive's members are confined to the directory the host chose
            // to install into, which is not the same tree a script may write to.
            confined = new ConfinedPaths(request.Policy with { WorkRoot = where });
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or ArgumentException)
        {
            return ExecutionResult.Failed($"{where} could not be prepared: {ex.Message}", where, environment);
        }

        var failures = new StringBuilder();
        int installed = 0;
        foreach (string spec in request.Packages)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (!TrySplit(spec, out string name, out string? version, out string? invalid))
            {
                failures.Append(invalid).Append('\n');
                continue;
            }

            string? failure;
            try
            {
                failure = await InstallOneAsync(
                    name, version, where, confined, request.Policy, Say, cancellationToken).ConfigureAwait(false);
            }
            catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
            {
                throw;
            }
            catch (Exception ex) when (ex is HttpRequestException or TaskCanceledException or IOException or InvalidDataException)
            {
                failure = $"{name}: {ex.Message}";
            }

            if (failure is null)
                installed++;
            else
                failures.Append(failure).Append('\n');
        }

        if (failures.Length > 0)
            return new ExecutionResult(1, log.ToString(), failures.ToString(), false, where, environment, TimeSpan.Zero);

        Say($"installed {installed} package{(installed == 1 ? string.Empty : "s")} into {where}");
        return new ExecutionResult(0, log.ToString(), string.Empty, false, where, environment, TimeSpan.Zero);
    }

    private async Task<string?> InstallOneAsync(
        string name,
        string? version,
        string target,
        ConfinedPaths confined,
        ExecutionPolicy policy,
        Action<string> say,
        CancellationToken cancellationToken)
    {
        string url = version is null
            ? $"{IndexRoot}/{Uri.EscapeDataString(name)}/json"
            : $"{IndexRoot}/{Uri.EscapeDataString(name)}/{Uri.EscapeDataString(version)}/json";

        say($"looking up {name}{(version is null ? string.Empty : "==" + version)}");
        using HttpResponseMessage response = await GetAllowedAsync(
            new Uri(url), policy, cancellationToken).ConfigureAwait(false);
        if (response.StatusCode == System.Net.HttpStatusCode.NotFound)
            return $"{name}: no such package{(version is null ? string.Empty : " or version " + version)} on {IndexHost}";
        if (!response.IsSuccessStatusCode)
            return $"{name}: {IndexHost} answered {(int)response.StatusCode} {response.ReasonPhrase}";

        string json = await ReadTextBoundedAsync(
            response.Content, MaxIndexBytes, "package metadata", cancellationToken).ConfigureAwait(false);
        if (!TrySelectWheel(
                json, _runtimeRelease, _runtimeVersion,
                out PyPiFile? wheel, out string? why)
            || wheel is null)
            return $"{name}: {why}";
        if (!Uri.TryCreate(wheel.Url, UriKind.Absolute, out Uri? wheelUri))
            return $"{name}: {IndexHost} published an invalid URL for {DisplayRemote(wheel.FileName)}";

        string transaction = Path.Combine(target, $".tensoragent-install-{Guid.NewGuid():N}");
        string archivePath = Path.Combine(transaction, "package.whl");
        string staged = Path.Combine(transaction, "contents");
        try
        {
            Directory.CreateDirectory(transaction);

            say($"downloading {DisplayRemote(wheel.FileName)}");
            using HttpResponseMessage download = await GetAllowedAsync(
                wheelUri, policy, cancellationToken).ConfigureAwait(false);
            download.EnsureSuccessStatusCode();
            string digest = await DownloadBoundedAsync(
                download.Content, archivePath, MaxWheelBytes, cancellationToken).ConfigureAwait(false);
            if (!string.Equals(digest, wheel.Sha256, StringComparison.OrdinalIgnoreCase))
            {
                return $"{name}: {DisplayRemote(wheel.FileName)} does not match the sha256 {IndexHost} published "
                    + $"(got {digest}, expected {wheel.Sha256})";
            }

            Directory.CreateDirectory(staged);
            await using (var buffer = new FileStream(
                archivePath, FileMode.Open, FileAccess.Read, FileShare.Read, CopyBufferBytes,
                FileOptions.Asynchronous | FileOptions.SequentialScan))
            using (var archive = new ZipArchive(buffer, ZipArchiveMode.Read))
            {
                (bool ok, int files, string? refusal) = await TryExtractAsync(
                    archive, staged,
                    new ConfinedPaths(policy with { WorkRoot = staged }),
                    MaxExpandedBytes, MaxArchiveFiles, cancellationToken).ConfigureAwait(false);
                if (!ok)
                    return $"{name}: {refusal}";

                string? commitFailure = await CommitStagedAsync(
                    staged, target, confined, name, cancellationToken).ConfigureAwait(false);
                if (commitFailure is not null)
                    return $"{name}: {commitFailure}";

                say($"installed {DisplayRemote(wheel.FileName)} ({files} files)");
                if (wheel.DeclaredDependencies.Count > 0 || wheel.ConditionalDependenciesOmitted)
                {
                    say(DependencyNotice(
                        name, wheel.DeclaredDependencies, wheel.ConditionalDependenciesOmitted));
                }
            }

            return null;
        }
        finally
        {
            // A stage is deliberately dot-prefixed and never on PYTHONPATH, so even a
            // cleanup failure cannot make a partial package importable.
            TryDeleteTree(transaction);
        }
    }

    private static async Task<string> ReadTextBoundedAsync(
        HttpContent content, int limit, string description, CancellationToken cancellationToken)
    {
        RejectKnownLength(content, limit, description);
        await using Stream input = await content.ReadAsStreamAsync(cancellationToken).ConfigureAwait(false);
        using var output = new MemoryStream(Math.Min((int)(content.Headers.ContentLength ?? 0), limit));
        await CopyBoundedAsync(input, output, limit, description, hash: null, cancellationToken).ConfigureAwait(false);
        return Encoding.UTF8.GetString(output.GetBuffer(), 0, checked((int)output.Length));
    }

    private static async Task<string> DownloadBoundedAsync(
        HttpContent content, string path, long limit, CancellationToken cancellationToken)
    {
        const string description = "wheel download";
        RejectKnownLength(content, limit, description);
        await using Stream input = await content.ReadAsStreamAsync(cancellationToken).ConfigureAwait(false);
        await using var output = new FileStream(
            path, FileMode.CreateNew, FileAccess.Write, FileShare.None, CopyBufferBytes,
            FileOptions.Asynchronous | FileOptions.SequentialScan);
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        await CopyBoundedAsync(input, output, limit, description, hash, cancellationToken).ConfigureAwait(false);
        await output.FlushAsync(cancellationToken).ConfigureAwait(false);
        return Convert.ToHexString(hash.GetHashAndReset()).ToLowerInvariant();
    }

    private static void RejectKnownLength(HttpContent content, long limit, string description)
    {
        if (content.Headers.ContentLength is long length && length > limit)
            throw new InvalidDataException($"{description} exceeds the {FormatLimit(limit)} limit");
    }

    private static async Task<long> CopyBoundedAsync(
        Stream input,
        Stream output,
        long limit,
        string description,
        IncrementalHash? hash,
        CancellationToken cancellationToken)
    {
        byte[] rented = ArrayPool<byte>.Shared.Rent(CopyBufferBytes);
        long total = 0;
        try
        {
            while (true)
            {
                int read = await input.ReadAsync(rented.AsMemory(0, rented.Length), cancellationToken).ConfigureAwait(false);
                if (read == 0)
                    return total;
                cancellationToken.ThrowIfCancellationRequested();
                if (read > limit - total)
                    throw new InvalidDataException($"{description} exceeds the {FormatLimit(limit)} limit");
                hash?.AppendData(rented, 0, read);
                await output.WriteAsync(rented.AsMemory(0, read), cancellationToken).ConfigureAwait(false);
                total += read;
            }
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(rented);
        }
    }

    private static string FormatLimit(long bytes)
        => bytes % (1024 * 1024) == 0
            ? $"{bytes / (1024 * 1024)} MiB"
            : $"{bytes} byte" + (bytes == 1 ? string.Empty : "s");

    /// <summary>
    /// Fetch one resource while applying the network allow-list before every request,
    /// including each redirect destination. Redirect handling cannot be delegated to
    /// <see cref="HttpClientHandler.AllowAutoRedirect"/>: by the time an automatically
    /// followed response is visible here, the request to the new host has already left
    /// the device.
    /// </summary>
    private async Task<HttpResponseMessage> GetAllowedAsync(
        Uri initial, ExecutionPolicy policy, CancellationToken cancellationToken)
    {
        const int maxRedirects = 10;
        Uri current = initial;
        int redirects = 0;

        while (true)
        {
            EnsureAllowed(current, policy);

            using var request = new HttpRequestMessage(HttpMethod.Get, current);
            HttpResponseMessage response = await _http.SendAsync(
                request, HttpCompletionOption.ResponseHeadersRead, cancellationToken).ConfigureAwait(false);

            // The app's client never follows automatically, but an injected client may.
            // Check its effective URI as a defence in depth. Such a client cannot provide
            // a pre-request guarantee, which is why production disables redirects above.
            Uri effective = response.RequestMessage?.RequestUri ?? current;
            try
            {
                EnsureAllowed(effective, policy);
            }
            catch
            {
                response.Dispose();
                throw;
            }

            if (!IsRedirect(response.StatusCode))
                return response;

            Uri? location = response.Headers.Location;
            if (location is null)
            {
                response.Dispose();
                throw new HttpRequestException(
                    $"{current.Host} returned a redirect without a Location header");
            }
            if (redirects >= maxRedirects)
            {
                response.Dispose();
                throw new HttpRequestException(
                    $"{initial.Host} redirected more than {maxRedirects} times");
            }

            Uri next;
            try
            {
                next = location.IsAbsoluteUri ? location : new Uri(effective, location);
            }
            catch (UriFormatException ex)
            {
                response.Dispose();
                throw new HttpRequestException(
                    $"{effective.Host} returned an invalid redirect destination", ex);
            }

            // Validate before the loop issues the request. Besides making the invariant
            // explicit, this ensures a refused redirect is reported against its target.
            try
            {
                EnsureAllowed(next, policy);
            }
            catch
            {
                response.Dispose();
                throw;
            }

            response.Dispose();
            current = next;
            redirects++;
        }
    }

    private static bool IsRedirect(HttpStatusCode status) => status is
        HttpStatusCode.MovedPermanently or
        HttpStatusCode.Found or
        HttpStatusCode.SeeOther or
        HttpStatusCode.TemporaryRedirect or
        HttpStatusCode.PermanentRedirect;

    private static void EnsureAllowed(Uri uri, ExecutionPolicy policy)
    {
        if (!uri.IsAbsoluteUri || !string.Equals(uri.Scheme, Uri.UriSchemeHttps, StringComparison.OrdinalIgnoreCase))
            throw new HttpRequestException($"'{uri}' is not an HTTPS package URL");
        if (!policy.IsHostAllowed(uri.Host))
            throw new HttpRequestException(
                ExecutionPolicy.HostNotAllowedMessage(uri.Host, policy.NetworkHosts));
    }

    /// <summary>
    /// Picks the one file that can be installed here, or says why none can.
    ///
    /// <para>
    /// <c>py3-none-any</c> (including <c>py2.py3-none-any</c>), versioned pure
    /// Python tags through <c>py313-none-any</c>, and the exact
    /// <c>cp313-none-any</c> tag qualify: the
    /// <c>none</c> ABI and the <c>any</c> platform together are what "no compiled
    /// code" means in a wheel name, while the Python tag must also match the
    /// embedded interpreter. Platform wheels and source distributions that would
    /// have to be built are refused in those words, because "install failed" would
    /// send a model looking for a fix that does not exist on this device.
    /// </para>
    /// </summary>
    internal static bool TrySelectWheel(string indexJson, out PyPiFile? wheel, out string? reason) =>
        TrySelectWheel(
            indexJson,
            ParseKnownEmbeddedPythonRelease(),
            EmbeddedPythonVersion,
            out wheel,
            out reason);

    private static bool TrySelectWheel(
        string indexJson,
        IReadOnlyList<int> runtimeRelease,
        string runtimeVersion,
        out PyPiFile? wheel,
        out string? reason)
    {
        wheel = null;
        reason = null;
        List<PyPiFile> files;
        string projectRequiresPython;
        IReadOnlyList<string> declaredDependencies;
        bool conditionalDependenciesOmitted = false;
        try
        {
            using JsonDocument document = JsonDocument.Parse(indexJson);
            JsonElement root = document.RootElement;
            projectRequiresPython = root.TryGetProperty("info", out JsonElement info)
                && info.ValueKind == JsonValueKind.Object
                    ? Text(info, "requires_python")
                    : string.Empty;
            declaredDependencies = info.ValueKind == JsonValueKind.Object
                ? ReadDeclaredDependencies(info, out conditionalDependenciesOmitted)
                : [];
            files = ReadFiles(root);
        }
        catch (JsonException ex)
        {
            reason = $"the index answered with something that is not JSON: {ex.Message}";
            return false;
        }

        if (!TryEvaluateRequiresPython(
                projectRequiresPython, runtimeRelease,
                out bool projectCompatible, out string? projectError))
        {
            reason = $"{IndexHost} published Requires-Python '{DisplayRemote(projectRequiresPython)}' that this host cannot safely "
                   + $"evaluate for CPython {runtimeVersion}: {projectError}";
            return false;
        }
        if (!projectCompatible)
        {
            reason = $"this release requires Python '{DisplayRemote(projectRequiresPython)}', but this host embeds CPython "
                   + runtimeVersion;
            return false;
        }

        if (files.Count == 0)
        {
            reason = $"{IndexHost} lists no files for it";
            return false;
        }

        string? filePythonRefusal = null;
        foreach (PyPiFile file in files)
        {
            if (file.Yanked || !string.Equals(file.PackageType, "bdist_wheel", StringComparison.Ordinal))
                continue;
            if (!IsPureWheel(file.FileName))
                continue;
            if (!TryEvaluateRequiresPython(
                    file.RequiresPython, runtimeRelease,
                    out bool fileCompatible, out string? fileError))
            {
                filePythonRefusal ??= $"{DisplayRemote(file.FileName)} has Requires-Python '{DisplayRemote(file.RequiresPython)}' that this host "
                    + $"cannot safely evaluate for CPython {runtimeVersion}: {fileError}";
                continue;
            }
            if (!fileCompatible)
            {
                filePythonRefusal ??= $"{DisplayRemote(file.FileName)} requires Python '{DisplayRemote(file.RequiresPython)}', but this host embeds "
                    + $"CPython {runtimeVersion}";
                continue;
            }
            if (file.Sha256.Length == 0)
            {
                reason = $"{DisplayRemote(file.FileName)} is published without a sha256, so it cannot be verified";
                return false;
            }
            if (!file.Url.StartsWith("https://", StringComparison.OrdinalIgnoreCase))
            {
                reason = $"{DisplayRemote(file.FileName)} is not published over https";
                return false;
            }
            wheel = file with
            {
                DeclaredDependencies = declaredDependencies,
                ConditionalDependenciesOmitted = conditionalDependenciesOmitted,
            };
            return true;
        }

        if (filePythonRefusal is not null)
        {
            reason = filePythonRefusal;
            return false;
        }

        bool sawPureForAnotherPython = files.Any(f => !f.Yanked
            && string.Equals(f.PackageType, "bdist_wheel", StringComparison.Ordinal)
            && HasNoAbiAnyTags(f.FileName));
        bool sawWheel = files.Any(f => !f.Yanked
            && string.Equals(f.PackageType, "bdist_wheel", StringComparison.Ordinal));
        reason = sawPureForAnotherPython
            ? $"every published pure-Python wheel targets another Python version. This host embeds CPython "
              + $"{runtimeVersion} and accepts py3-none-any, compatible py30..py313-none-any, or "
              + "cp313-none-any wheels"
            : sawWheel
            ? "every published wheel is built for a specific interpreter and platform, which means compiled code. "
              + "This host can only install compatible pure-Python (py3-none-any, py30..py313-none-any, or "
              + "cp313-none-any) wheels: "
              + "extension modules have to be signed "
              + "into the app at build time"
            : "only a source distribution is published, and building one needs a compiler this host does not have. "
              + "This host can only install compatible pure-Python (py3-none-any, py30..py313-none-any, or "
              + "cp313-none-any) wheels";
        return false;
    }

    private static IReadOnlyList<string> ReadDeclaredDependencies(
        JsonElement info, out bool conditionalDependenciesOmitted)
    {
        conditionalDependenciesOmitted = false;
        if (!info.TryGetProperty("requires_dist", out JsonElement requires)
            || requires.ValueKind != JsonValueKind.Array)
        {
            return [];
        }

        var names = new List<string>();
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        int inspected = 0;
        foreach (JsonElement item in requires.EnumerateArray())
        {
            if (inspected++ >= MaxDependencyMetadataEntries)
                break;
            if (item.ValueKind != JsonValueKind.String)
                continue;
            string requirement = item.GetString()?.Trim() ?? string.Empty;
            if (requirement.Length == 0 || requirement.Length > MaxDependencyMetadataCharacters)
                continue;
            if (requirement.Contains(';'))
            {
                // Markers require a PEP 508 evaluator. Only unconditional names are
                // included in this informational notice; installation never depends
                // on this deliberately small parser.
                conditionalDependenciesOmitted = true;
                continue;
            }

            int length = 0;
            while (length < requirement.Length)
            {
                char c = requirement[length];
                if (!char.IsAsciiLetterOrDigit(c) && c is not ('-' or '_' or '.'))
                    break;
                length++;
            }
            if (length == 0 || length > MaxDependencyNameCharacters)
                continue;
            string name = requirement[..length];
            if (!IsName(name) || !seen.Add(name))
                continue;
            names.Add(name);
        }
        return names;
    }

    private static string DependencyNotice(
        string package,
        IReadOnlyList<string> dependencies,
        bool conditionalDependenciesOmitted)
    {
        if (dependencies.Count == 0)
        {
            return $"note: {package} declares conditional dependencies; this host did not evaluate their "
                 + "markers or resolve them automatically. Install only an actually missing pure-Python dependency.";
        }
        string visible = string.Join(", ", dependencies.Take(MaxDependenciesInNotice));
        string remainder = dependencies.Count > MaxDependenciesInNotice
            ? $", and {dependencies.Count - MaxDependenciesInNotice} more"
            : string.Empty;
        string conditional = conditionalDependenciesOmitted
            ? " Conditional dependency markers were not evaluated or shown."
            : string.Empty;
        return $"note: {package} declares these unconditional dependencies ({visible}{remainder}); this host did not "
             + "resolve them automatically. Some may already be bundled or optional at runtime."
             + conditional + " Install only an actually missing pure-Python dependency.";
    }

    private static string DisplayRemote(string value)
    {
        string singleLine = value.Replace('\r', ' ').Replace('\n', ' ').Replace('\t', ' ');
        return singleLine.Length <= MaxDiagnosticMetadataCharacters
            ? singleLine
            : singleLine[..MaxDiagnosticMetadataCharacters] + "…";
    }

    /// <summary>
    /// True for a wheel whose tags promise both no compiled code and compatibility
    /// with the app's embedded CPython 3.13:
    /// <c>name-version[-build]-pythontag-abitag-platformtag.whl</c> with
    /// <c>none</c> and <c>any</c> in the last two, plus the generic <c>py3</c>, a
    /// versioned pure-Python tag from <c>py30</c> through <c>py313</c>, or the exact
    /// <c>cp313</c> interpreter tag. CPython 3.13 is compatible with older Python 3
    /// pure-wheel tags; a tag for a newer language version is not safe to load here.
    /// </summary>
    internal static bool IsPureWheel(string fileName)
    {
        if (!HasNoAbiAnyTags(fileName))
            return false;
        string[] parts = fileName[..^4].Split('-');
        string python = parts[^3];
        foreach (string tag in python.Split('.'))
        {
            if (string.Equals(tag, "py3", StringComparison.OrdinalIgnoreCase)
                || string.Equals(tag, $"cp{EmbeddedPythonMajor}{EmbeddedPythonMinor}", StringComparison.OrdinalIgnoreCase)
                || IsCompatibleVersionedPurePythonTag(tag))
            {
                return true;
            }
        }
        return false;
    }

    private static bool IsCompatibleVersionedPurePythonTag(string tag)
    {
        if (!tag.StartsWith("py3", StringComparison.OrdinalIgnoreCase)
            || tag.Length <= 3
            || !int.TryParse(tag.AsSpan(3), NumberStyles.None, CultureInfo.InvariantCulture, out int minor)
            || minor < 0
            || minor > EmbeddedPythonMinor)
        {
            return false;
        }

        // Reject non-canonical spellings such as py301 while accepting py30.
        return string.Equals(tag, $"py{EmbeddedPythonMajor}{minor}", StringComparison.OrdinalIgnoreCase);
    }

    private static bool HasNoAbiAnyTags(string fileName)
    {
        if (!fileName.EndsWith(".whl", StringComparison.OrdinalIgnoreCase))
            return false;
        string[] parts = fileName[..^4].Split('-');
        return parts.Length >= 5
            && string.Equals(parts[^2], "none", StringComparison.OrdinalIgnoreCase)
            && string.Equals(parts[^1], "any", StringComparison.OrdinalIgnoreCase);
    }

    /// <summary>
    /// Evaluates the bounded, release-number subset of PEP 440 used in normal
    /// <c>Requires-Python</c> metadata. Unsupported syntax is refused instead of
    /// guessed: accepting a wheel for the wrong interpreter produces a successful
    /// install followed by a much less useful syntax/import failure.
    /// </summary>
    internal static bool TryEvaluateRequiresPython(
        string? expression, out bool compatible, out string? error) =>
        TryEvaluateRequiresPython(
            expression, ParseKnownEmbeddedPythonRelease(), out compatible, out error);

    private static bool TryEvaluateRequiresPython(
        string? expression,
        IReadOnlyList<int> runtime,
        out bool compatible,
        out string? error)
    {
        compatible = true;
        error = null;
        if (string.IsNullOrWhiteSpace(expression))
            return true;

        string specifiers = expression.Trim();
        if (specifiers.Length > MaxRequiresPythonCharacters)
        {
            compatible = false;
            error = $"the expression exceeds {MaxRequiresPythonCharacters} characters";
            return false;
        }

        string[] clauses = specifiers.Split(',');
        if (clauses.Length > MaxRequiresPythonClauses)
        {
            compatible = false;
            error = $"the expression contains more than {MaxRequiresPythonClauses} clauses";
            return false;
        }

        foreach (string untrimmed in clauses)
        {
            string clause = untrimmed.Trim();
            if (clause.Length == 0)
            {
                compatible = false;
                error = "an empty version clause was published";
                return false;
            }

            string? op = null;
            foreach (string candidate in new[] { "===", "~=", "==", "!=", ">=", "<=", ">", "<" })
            {
                if (clause.StartsWith(candidate, StringComparison.Ordinal))
                {
                    op = candidate;
                    break;
                }
            }
            if (op is null)
            {
                compatible = false;
                error = $"'{clause}' has no supported comparison operator";
                return false;
            }
            if (op == "===")
            {
                compatible = false;
                error = "the arbitrary-equality operator === is not safe to evaluate here";
                return false;
            }

            string versionText = clause[op.Length..].Trim();
            bool wildcard = versionText.EndsWith(".*", StringComparison.Ordinal);
            if (wildcard)
            {
                if (op is not ("==" or "!="))
                {
                    compatible = false;
                    error = $"'{clause}' uses a wildcard with {op}";
                    return false;
                }
                versionText = versionText[..^2];
            }

            if (!TryParseRelease(versionText, out int[] expected, out string? parseError))
            {
                compatible = false;
                error = $"'{clause}' {parseError}";
                return false;
            }

            int comparison = CompareRelease(runtime, expected);
            bool clauseMatches;
            if (wildcard)
            {
                bool prefixMatches = ReleasePrefixMatches(runtime, expected);
                clauseMatches = op == "==" ? prefixMatches : !prefixMatches;
            }
            else
            {
                clauseMatches = op switch
                {
                    "==" => comparison == 0,
                    "!=" => comparison != 0,
                    ">=" => comparison >= 0,
                    "<=" => comparison <= 0,
                    ">" => comparison > 0,
                    "<" => comparison < 0,
                    "~=" => comparison >= 0 && CompatiblePrefixMatches(runtime, expected),
                    _ => false,
                };
            }

            if (op == "~=" && expected.Length < 2)
            {
                compatible = false;
                error = $"'{clause}' needs at least a major and minor release for ~= compatibility";
                return false;
            }
            if (!clauseMatches)
                compatible = false;
        }

        return true;
    }

    private static int[] ParseKnownEmbeddedPythonRelease()
    {
        if (!TryParseRelease(EmbeddedPythonVersion, out int[] release, out _))
            throw new InvalidOperationException("the compiled embedded Python version is invalid");
        return release;
    }

    private static bool TryParseRelease(string text, out int[] release, out string? error)
    {
        release = [];
        error = null;
        if (text.Length == 0)
        {
            error = "names no version";
            return false;
        }
        if (text[0] is 'v' or 'V')
            text = text[1..];

        string[] components = text.Split('.');
        if (components.Length == 0 || components.Length > MaxVersionComponents)
        {
            error = $"has more than {MaxVersionComponents} release components";
            return false;
        }

        release = new int[components.Length];
        for (int i = 0; i < components.Length; i++)
        {
            string component = components[i];
            if (component.Length == 0 || component.Length > 9
                || !component.All(char.IsAsciiDigit)
                || !int.TryParse(component, NumberStyles.None, CultureInfo.InvariantCulture, out release[i]))
            {
                release = [];
                error = $"contains unsupported release component '{component}'";
                return false;
            }
        }
        return true;
    }

    private static int CompareRelease(IReadOnlyList<int> left, IReadOnlyList<int> right)
    {
        int count = Math.Max(left.Count, right.Count);
        for (int i = 0; i < count; i++)
        {
            int leftPart = i < left.Count ? left[i] : 0;
            int rightPart = i < right.Count ? right[i] : 0;
            int comparison = leftPart.CompareTo(rightPart);
            if (comparison != 0)
                return comparison;
        }
        return 0;
    }

    private static bool ReleasePrefixMatches(IReadOnlyList<int> runtime, IReadOnlyList<int> prefix)
    {
        for (int i = 0; i < prefix.Count; i++)
        {
            int runtimePart = i < runtime.Count ? runtime[i] : 0;
            if (runtimePart != prefix[i])
                return false;
        }
        return true;
    }

    private static bool CompatiblePrefixMatches(IReadOnlyList<int> runtime, IReadOnlyList<int> expected)
    {
        // ~=3.8 means >=3.8,<4; ~=3.8.1 means >=3.8.1,<3.9.
        int prefixLength = expected.Count - 1;
        for (int i = 0; i < prefixLength; i++)
        {
            int runtimePart = i < runtime.Count ? runtime[i] : 0;
            if (runtimePart != expected[i])
                return false;
        }
        return true;
    }

    /// <summary>
    /// Unpacks a wheel, putting every member through the same confinement a
    /// script's own writes go through.
    ///
    /// <para>
    /// A member name comes from the archive, not from the user, so it gets both
    /// checks the shell's <c>unzip</c> applies: it must stay under the directory
    /// being extracted into (the zip-slip rule — the confinement alone would let
    /// <c>../../tmp/x</c> through, because the temp root is writable too), and it
    /// must satisfy the confinement, which is what stops a member from landing
    /// through a symlink somewhere else entirely.
    /// </para>
    /// <para>
    /// Every member is checked before any member is written. An archive that
    /// turns hostile halfway through would otherwise leave its first few files
    /// behind, and a refusal that says nothing was installed has to be true.
    /// </para>
    /// </summary>
    internal static bool TryExtract(ZipArchive archive, string target, ConfinedPaths confined, out int files, out string? refusal)
    {
        (bool ok, int extracted, string? why) = TryExtractAsync(
            archive, target, confined, MaxExpandedBytes, MaxArchiveFiles, CancellationToken.None)
            .GetAwaiter().GetResult();
        files = extracted;
        refusal = why;
        return ok;
    }

    /// <summary>
    /// Validates the complete archive before writing, then streams each member with
    /// both declared and observed expansion limits. Production always points this at
    /// an unimportable staging directory.
    /// </summary>
    internal static async Task<(bool Ok, int Files, string? Refusal)> TryExtractAsync(
        ZipArchive archive,
        string target,
        ConfinedPaths confined,
        long maxExpandedBytes,
        int maxFiles,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(archive);
        ArgumentNullException.ThrowIfNull(confined);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxExpandedBytes);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxFiles);
        int files = 0;
        string? refusal = null;
        string root = Path.TrimEndingDirectorySeparator(Path.GetFullPath(target));

        var planned = new List<(ZipArchiveEntry Entry, string Destination)>();
        var destinations = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        long declaredBytes = 0;
        int archiveMembers = 0;
        foreach (ZipArchiveEntry entry in archive.Entries)
        {
            cancellationToken.ThrowIfCancellationRequested();
            string member = entry.FullName.Replace('\\', '/');
            if (member.Length == 0)
                continue;

            if (archiveMembers >= maxFiles)
            {
                refusal = $"the wheel contains more than {maxFiles:N0} archive entries; nothing was installed";
                return (false, 0, refusal);
            }
            archiveMembers++;
            if (member.EndsWith('/'))
                continue;
            if (entry.Length > maxExpandedBytes - declaredBytes)
            {
                refusal = $"the expanded wheel exceeds the {FormatLimit(maxExpandedBytes)} limit; nothing was installed";
                return (false, 0, refusal);
            }

            if (Path.IsPathRooted(member) || member.StartsWith("../", StringComparison.Ordinal)
                || member.Contains("/../", StringComparison.Ordinal) || member == ".."
                || member.EndsWith("/..", StringComparison.Ordinal))
            {
                refusal = $"the archive member '{entry.FullName}' points outside {root}; nothing was installed";
                return (false, 0, refusal);
            }

            string combined = Path.GetFullPath(Path.Combine(root, member));
            if (!combined.StartsWith(root + Path.DirectorySeparatorChar, StringComparison.Ordinal))
            {
                refusal = $"the archive member '{entry.FullName}' points outside {root}; nothing was installed";
                return (false, 0, refusal);
            }

            if (!confined.TryResolve(combined, root, PathAccess.Write, out string resolved, out _))
            {
                refusal = $"the archive member '{entry.FullName}' would be written outside {root}; nothing was installed";
                return (false, 0, refusal);
            }
            if (!destinations.Add(resolved))
            {
                refusal = $"the archive writes '{entry.FullName}' more than once; nothing was installed";
                return (false, 0, refusal);
            }

            planned.Add((entry, resolved));
            declaredBytes += entry.Length;
        }

        long expandedBytes = 0;
        foreach ((ZipArchiveEntry entry, string destination) in planned)
        {
            cancellationToken.ThrowIfCancellationRequested();
            string? directory = Path.GetDirectoryName(destination);
            if (directory is { Length: > 0 })
                Directory.CreateDirectory(directory);
            await using Stream input = entry.Open();
            await using var output = new FileStream(
                destination, FileMode.Create, FileAccess.Write, FileShare.None, CopyBufferBytes,
                FileOptions.Asynchronous | FileOptions.SequentialScan);
            try
            {
                expandedBytes += await CopyBoundedAsync(
                    input, output, maxExpandedBytes - expandedBytes, "expanded wheel",
                    hash: null, cancellationToken).ConfigureAwait(false);
            }
            catch (InvalidDataException ex)
            {
                refusal = ex.Message + "; nothing was installed";
                return (false, 0, refusal);
            }
            files++;
        }

        return (true, files, null);
    }

    /// <summary>
    /// Builds complete replacements for each touched top-level entry before moving
    /// anything visible. Existing namespace-package files are copied into the staged
    /// tree, except files owned by an older version according to its RECORD. The short
    /// commit is rollback-capable and intentionally has no cancellation points.
    /// </summary>
    private static async Task<string?> CommitStagedAsync(
        string staged,
        string target,
        ConfinedPaths confined,
        string packageName,
        CancellationToken cancellationToken)
    {
        var removals = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        string? ownershipFailure = FindPreviouslyOwnedFiles(target, packageName, removals);
        if (ownershipFailure is not null)
            return ownershipFailure;

        var topLevels = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (string entry in Directory.EnumerateFileSystemEntries(staged))
            topLevels.Add(Path.GetFileName(entry));
        foreach (string relative in removals)
        {
            int slash = relative.IndexOf('/');
            topLevels.Add(slash < 0 ? relative : relative[..slash]);
        }

        foreach (string top in topLevels.OrderBy(value => value, StringComparer.Ordinal))
        {
            cancellationToken.ThrowIfCancellationRequested();
            string existing = Path.Combine(target, top);
            string candidate = Path.Combine(staged, top);
            string? mergeFailure = await MergeExistingAsync(
                existing, candidate, target, removals, cancellationToken).ConfigureAwait(false);
            if (mergeFailure is not null)
                return mergeFailure;
        }

        RemoveEmptyDirectories(staged, keepRoot: true);
        cancellationToken.ThrowIfCancellationRequested();
        return CommitPrepared(staged, target, confined, topLevels);
    }

    private static async Task<string?> MergeExistingAsync(
        string existing,
        string candidate,
        string target,
        HashSet<string> removals,
        CancellationToken cancellationToken)
    {
        if (!Path.Exists(existing))
            return null;

        FileAttributes attributes;
        try
        {
            attributes = File.GetAttributes(existing);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            return $"the existing package entry '{existing}' could not be inspected: {ex.Message}; nothing was installed";
        }
        if ((attributes & FileAttributes.ReparsePoint) != 0)
            return $"the existing package entry '{existing}' is a link; nothing was installed";

        string relative = Path.GetRelativePath(target, existing).Replace('\\', '/');
        bool isDirectory = (attributes & FileAttributes.Directory) != 0;
        if (!isDirectory)
        {
            if (removals.Contains(relative) || Path.Exists(candidate))
                return null;
            string? parent = Path.GetDirectoryName(candidate);
            if (parent is { Length: > 0 })
                Directory.CreateDirectory(parent);
            await CopyFileAsync(existing, candidate, cancellationToken).ConfigureAwait(false);
            return null;
        }

        // A new file intentionally replaces an old directory of the same name.
        if (File.Exists(candidate) && !Directory.Exists(candidate))
            return null;
        Directory.CreateDirectory(candidate);
        foreach (string child in Directory.EnumerateFileSystemEntries(existing))
        {
            cancellationToken.ThrowIfCancellationRequested();
            string? failure = await MergeExistingAsync(
                child, Path.Combine(candidate, Path.GetFileName(child)), target,
                removals, cancellationToken).ConfigureAwait(false);
            if (failure is not null)
                return failure;
        }
        return null;
    }

    private static async Task CopyFileAsync(string source, string destination, CancellationToken cancellationToken)
    {
        await using var input = new FileStream(
            source, FileMode.Open, FileAccess.Read, FileShare.Read, CopyBufferBytes,
            FileOptions.Asynchronous | FileOptions.SequentialScan);
        await using var output = new FileStream(
            destination, FileMode.CreateNew, FileAccess.Write, FileShare.None, CopyBufferBytes,
            FileOptions.Asynchronous | FileOptions.SequentialScan);
        await input.CopyToAsync(output, CopyBufferBytes, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Finds older installations by their METADATA Name and treats RECORD as the
    /// authority for files that may be removed. Bad or escaping RECORD paths are
    /// ignored rather than allowed to name arbitrary session files.
    /// </summary>
    private static string? FindPreviouslyOwnedFiles(
        string target, string packageName, HashSet<string> removals)
    {
        string canonicalName = CanonicalPackageName(packageName);
        try
        {
            foreach (string distInfo in Directory.EnumerateDirectories(target, "*.dist-info", SearchOption.TopDirectoryOnly))
            {
                string metadata = Path.Combine(distInfo, "METADATA");
                if (!File.Exists(metadata)
                    || !string.Equals(ReadMetadataName(metadata), canonicalName, StringComparison.Ordinal))
                {
                    continue;
                }

                string record = Path.Combine(distInfo, "RECORD");
                if (File.Exists(record))
                {
                    foreach (string line in File.ReadLines(record))
                    {
                        string path = FirstCsvField(line);
                        if (TryNormalizeOwnedPath(target, path, out string relative))
                            removals.Add(relative);
                    }
                }

                // RECORD should list these too, but removing the matching metadata
                // tree explicitly prevents two versions from appearing in pip list
                // when a publisher supplied an incomplete RECORD.
                foreach (string file in Directory.EnumerateFiles(distInfo, "*", SearchOption.AllDirectories))
                {
                    string relative = Path.GetRelativePath(target, file).Replace('\\', '/');
                    removals.Add(relative);
                }
            }
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or ArgumentException)
        {
            return $"the existing package metadata could not be inspected: {ex.Message}; nothing was installed";
        }
        return null;
    }

    private static string ReadMetadataName(string path)
    {
        const int maxMetadataCharacters = 64 * 1024;
        using var reader = new StreamReader(path, Encoding.UTF8, detectEncodingFromByteOrderMarks: true);
        int read = 0;
        while (read < maxMetadataCharacters && reader.ReadLine() is string line)
        {
            read += line.Length + 1;
            if (line.StartsWith("Name:", StringComparison.OrdinalIgnoreCase))
                return CanonicalPackageName(line[5..].Trim());
        }
        return string.Empty;
    }

    private static string CanonicalPackageName(string value)
    {
        var canonical = new StringBuilder(value.Length);
        bool separator = false;
        foreach (char c in value)
        {
            if (char.IsAsciiLetterOrDigit(c))
            {
                canonical.Append(char.ToLowerInvariant(c));
                separator = false;
            }
            else if (!separator && canonical.Length > 0)
            {
                canonical.Append('-');
                separator = true;
            }
        }
        if (canonical.Length > 0 && canonical[^1] == '-')
            canonical.Length--;
        return canonical.ToString();
    }

    private static string FirstCsvField(string line)
    {
        if (!line.StartsWith('"'))
        {
            int comma = line.IndexOf(',');
            return comma < 0 ? line : line[..comma];
        }

        var field = new StringBuilder();
        for (int i = 1; i < line.Length; i++)
        {
            if (line[i] != '"')
            {
                field.Append(line[i]);
                continue;
            }
            if (i + 1 < line.Length && line[i + 1] == '"')
            {
                field.Append('"');
                i++;
                continue;
            }
            break;
        }
        return field.ToString();
    }

    private static bool TryNormalizeOwnedPath(string target, string path, out string relative)
    {
        relative = string.Empty;
        if (string.IsNullOrWhiteSpace(path) || Path.IsPathRooted(path))
            return false;
        string portable = path.Replace('\\', '/');
        if (portable.Split('/').Any(part => part is ".." or "." or ""))
            return false;
        string root = Path.TrimEndingDirectorySeparator(Path.GetFullPath(target));
        string full = Path.GetFullPath(Path.Combine(root, portable));
        if (!full.StartsWith(root + Path.DirectorySeparatorChar, StringComparison.Ordinal))
            return false;
        relative = Path.GetRelativePath(root, full).Replace('\\', '/');
        return true;
    }

    private static string? CommitPrepared(
        string staged, string target, ConfinedPaths confined, IEnumerable<string> topLevels)
    {
        string backupRoot = Path.Combine(Path.GetDirectoryName(staged)!, "backup");
        Directory.CreateDirectory(backupRoot);
        var moves = new List<CommitMove>();
        try
        {
            int index = 0;
            foreach (string top in topLevels.OrderBy(value => value, StringComparer.Ordinal))
            {
                string source = Path.Combine(staged, top);
                string destination = Path.Combine(target, top);
                if (!confined.TryResolve(destination, target, PathAccess.Write, out _, out _))
                    throw new IOException($"'{top}' would be committed outside the package directory");
                if (Path.Exists(destination)
                    && (File.GetAttributes(destination) & FileAttributes.ReparsePoint) != 0)
                {
                    throw new IOException($"the existing package entry '{destination}' is a link");
                }

                string backup = Path.Combine(backupRoot, (index++).ToString(CultureInfo.InvariantCulture));
                var move = new CommitMove(destination, backup, Path.Exists(destination));
                moves.Add(move);
                if (move.HadExisting)
                    MoveEntry(destination, backup);
                if (Path.Exists(source))
                {
                    MoveEntry(source, destination);
                    move.Installed = true;
                }
            }
            return null;
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or ArgumentException)
        {
            string? rollbackFailure = RollBack(moves);
            return rollbackFailure is null
                ? $"the staged package could not be committed: {ex.Message}; existing files were restored"
                : $"the staged package could not be committed: {ex.Message}; rollback also failed: {rollbackFailure}";
        }
    }

    private sealed class CommitMove(string destination, string backup, bool hadExisting)
    {
        public string Destination { get; } = destination;
        public string Backup { get; } = backup;
        public bool HadExisting { get; } = hadExisting;
        public bool Installed { get; set; }
    }

    private static string? RollBack(List<CommitMove> moves)
    {
        string? failure = null;
        for (int i = moves.Count - 1; i >= 0; i--)
        {
            CommitMove move = moves[i];
            try
            {
                if (move.Installed && Path.Exists(move.Destination))
                    DeleteEntry(move.Destination);
                if (move.HadExisting && Path.Exists(move.Backup))
                    MoveEntry(move.Backup, move.Destination);
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or ArgumentException)
            {
                failure ??= ex.Message;
            }
        }
        return failure;
    }

    private static void MoveEntry(string source, string destination)
    {
        if (Directory.Exists(source))
            Directory.Move(source, destination);
        else
            File.Move(source, destination);
    }

    private static void DeleteEntry(string path)
    {
        if (Directory.Exists(path))
            Directory.Delete(path, recursive: true);
        else
            File.Delete(path);
    }

    private static void RemoveEmptyDirectories(string directory, bool keepRoot)
    {
        foreach (string child in Directory.EnumerateDirectories(directory).ToArray())
            RemoveEmptyDirectories(child, keepRoot: false);
        if (!keepRoot && !Directory.EnumerateFileSystemEntries(directory).Any())
            Directory.Delete(directory);
    }

    private static void TryDeleteTree(string path)
    {
        try
        {
            if (Directory.Exists(path))
                Directory.Delete(path, recursive: true);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            // The directory is hidden and never placed on an interpreter path. A later
            // workspace cleanup can reclaim it without changing install correctness.
        }
    }

    /// <summary>Splits <c>name</c> or <c>name==version</c>, refusing anything a name may not contain.</summary>
    internal static bool TrySplit(string spec, out string name, out string? version, out string? error)
    {
        name = string.Empty;
        version = null;
        error = null;

        string text = spec.Trim();
        int marker = text.IndexOf("==", StringComparison.Ordinal);
        if (marker >= 0)
        {
            version = text[(marker + 2)..].Trim();
            text = text[..marker].Trim();
        }

        if (text.Length == 0 || !IsName(text))
        {
            error = $"'{spec}' is not a package name this host will fetch; "
                + "only names, optionally with ==version, are accepted";
            return false;
        }
        if (version is { Length: 0 })
        {
            error = $"'{spec}' names no version after ==";
            return false;
        }
        if (version is not null && !IsVersion(version))
        {
            error = $"'{spec}' does not name a version this host will fetch";
            return false;
        }

        name = text;
        return true;
    }

    private static bool IsName(string value)
    {
        if (!char.IsAsciiLetterOrDigit(value[0]) || !char.IsAsciiLetterOrDigit(value[^1]))
            return false;
        foreach (char c in value)
        {
            if (!char.IsAsciiLetterOrDigit(c) && c is not ('-' or '_' or '.'))
                return false;
        }
        return true;
    }

    private static bool IsVersion(string value)
    {
        foreach (char c in value)
        {
            if (!char.IsAsciiLetterOrDigit(c) && c is not ('.' or '-' or '_' or '+' or '!'))
                return false;
        }
        return true;
    }

    private static List<PyPiFile> ReadFiles(JsonElement root)
    {
        var files = new List<PyPiFile>();
        if (!root.TryGetProperty("urls", out JsonElement urls) || urls.ValueKind != JsonValueKind.Array)
            return files;

        foreach (JsonElement entry in urls.EnumerateArray())
        {
            string fileName = Text(entry, "filename");
            string url = Text(entry, "url");
            string type = Text(entry, "packagetype");
            string sha = string.Empty;
            if (entry.TryGetProperty("digests", out JsonElement digests) && digests.ValueKind == JsonValueKind.Object)
                sha = Text(digests, "sha256");
            bool yanked = entry.TryGetProperty("yanked", out JsonElement yankedValue)
                && yankedValue.ValueKind == JsonValueKind.True;
            if (fileName.Length > 0 && fileName.Length <= MaxRemoteFileNameCharacters
                && url.Length > 0 && url.Length <= MaxRemoteUrlCharacters)
            {
                files.Add(new PyPiFile(
                    fileName,
                    url,
                    sha.ToLower(CultureInfo.InvariantCulture),
                    type,
                    yanked,
                    Text(entry, "requires_python")));
            }
        }
        return files;
    }

    private static string Text(JsonElement element, string name)
        => element.TryGetProperty(name, out JsonElement value) && value.ValueKind == JsonValueKind.String
            ? value.GetString() ?? string.Empty
            : string.Empty;
}
