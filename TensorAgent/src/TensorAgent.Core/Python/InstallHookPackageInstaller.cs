// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using TensorAgent.Core.Sandbox;
using TensorSharp.AgentHost.CodeExec;
using TensorSharp.AgentHost.Skills;

namespace TensorAgent.Core.Python;

/// <summary>
/// Adapts TensorAgent's in-process wheel hook to the package-installer contract used
/// by <see cref="ShellRunner"/>.
///
/// <para>
/// The distinction matters on iOS: <see cref="ShellRunner"/> reads a model-written
/// <c>pip install</c> command and calls <see cref="IPackageInstaller"/> before the
/// remaining shell line runs. Falling back to its desktop installer tries to start
/// <c>python -m pip</c>, but TensorAgent has neither child processes nor a bundled pip.
/// This bridge keeps the shared parser and ledger while handing the validated request
/// to the wheel unpacker that the app actually provides.
/// </para>
/// <para>
/// It also answers for the bundle before the hook is asked. A model types
/// <c>pip install lxml</c> out of habit, before <c>import lxml</c>; on this platform the
/// only lxml that can ever load is the one compiled into the app, and sending that
/// request to the index found nothing but compiled wheels and refused it — in words
/// that told the model lxml was unavailable on a phone where it had been importable
/// the whole time. Observed: a deck that became an HTML file. So a request for a
/// distribution the bundle already ships is reported as already there, and a request
/// to REPLACE a compiled one is refused by name, because no download can do that.
/// </para>
/// </summary>
internal sealed class InstallHookPackageInstaller : IPackageInstaller
{
    private const int MaxPackages = 16;

    /// <summary>
    /// Guidance for a turn that may reach the network, appended to the shell tool's
    /// description. Every sentence has to be TRUE OF EVERY TASK — including the turns
    /// where the answer IS an interpretation. It says what may not be INVENTED, never
    /// that a fetched number may not be explained or a new figure derived: an earlier
    /// draft kept the screener's own answer shape ("no cause or explanation for a
    /// number you fetched, and no figure you did not fetch"), which forbade the very
    /// analysis users ask for and contradicted this same tool description's own
    /// "running code is more reliable than doing arithmetic in your head".
    ///
    /// <para>
    /// This was, for a while, a complete working Yahoo Finance screener program —
    /// 3,204 characters of it, url and response fields and a ten-row markdown table —
    /// added to make one stock-gainers request come out right. It did. It also sat in
    /// the model's context on every networked turn, and a whole worked program in the
    /// tool description is not guidance, it is a demonstration of what code here looks
    /// like: the reported symptom was a model reaching for finance APIs on requests
    /// that had nothing to do with finance. A tool description is read on every turn,
    /// so anything in it that is true of only one task is a bias on all the others.
    /// Fix a specific task in a SKILL, which is injected only when it is selected.
    /// </para>
    /// <para>
    /// A constant rather than a description of the session, so the tool prefix is the
    /// same text on every launch and the prefix cache is worth something.
    /// </para>
    /// </summary>
    internal const string ModelExecutionInstructions =
        "Reach for the standard library first: `urllib.request` with `json` against a structured JSON or CSV "
        + "endpoint answers most lookups, and installing an HTTP or domain-specific client merely to make a GET "
        + "costs a round and often fails on this device. "
        + "Quote what a response actually contained rather than what you expected it to contain, and if a "
        + "request fails or arrives without what you needed, say so instead of inventing it — and change "
        + "something before running it again.";

    /// <summary>
    /// Shell guidance with nothing to do with the network, so it must not appear and
    /// disappear with the network switch. The quoting rule is about writing Python at
    /// all, and it used to be shown only to a model whose user had turned networking on.
    /// </summary>
    internal const string ModelShellInstructions =
        "This host uses embedded interpreters: `node` is a JavaScriptCore compatibility layer, not Node.js. "
        + "npm/npx, native executables and child-process APIs are unavailable, even with network access on. "
        + "A skill requiring those capabilities cannot run here; explain the platform limitation rather than claiming it succeeded. "
        + "Write anything longer than one line as a quoted heredoc — `python3 - <<'PY' ... PY` — rather than "
        + "`python3 -c`. Quoting is what breaks first in a `-c` one-liner once the program contains quotes of "
        + "its own, and the failure looks like a syntax error in code that is actually fine. "
        + "Do not narrate a command before running it, and when its output already answers the question, quote "
        + "what it printed rather than retyping it from memory.";

    /// <summary>
    /// The distributions <c>prepare-python.sh</c> stages into the bundle, as the model
    /// should think of them: the distribution name, with the import name after it where
    /// the two differ. Spelled out here rather than read off the bundle at startup
    /// because the tool prefix must be the same text on every launch for the prefix
    /// cache to be worth anything; <c>BundledPackagesTests</c> holds this list to the
    /// staging script so the two cannot drift apart.
    /// </summary>
    internal static readonly IReadOnlyList<string> BundledPackageNames = new[]
    {
        "numpy",
        "Pillow (import PIL)",
        "lxml",
        "python-pptx (import pptx)",
        "python-docx (import docx)",
        "openpyxl",
        "XlsxWriter (import xlsxwriter)",
        "reportlab",
        "pypdf",
        "defusedxml",
        "PyYAML (import yaml)",
        "certifi",
        "imageio",
        "charset-normalizer",
        "typing_extensions",
        "et_xmlfile",
    };

    /// <summary>The bundled distributions whose extension modules make them irreplaceable.</summary>
    internal static readonly IReadOnlyList<string> CompiledBundledPackageNames = new[] { "numpy", "Pillow", "lxml" };

    /// <summary>
    /// What the bundle ships, told to the model on EVERY launch -- with the network
    /// switch off as much as on, because the switch gates fetching and these need no
    /// fetch. Shown by the shell tool ahead of the install guidance.
    /// </summary>
    internal static readonly string ModelProvidedPackagesInstructions =
        "Python packages already built into this app and importable with no install at all: "
        + string.Join(", ", BundledPackageNames) + ". "
        + "Installing one of those is a no-op, and the compiled ones (" + string.Join(", ", CompiledBundledPackageNames)
        + ") cannot be replaced by another version.";

    internal static readonly string ModelInstallInstructions =
        "Python packages only. Use `pip install <name>` or `python3 -m pip install <name>`. "
        + "The host performs the install, so name packages plainly; options that change the source or target are refused. "
        + "Beyond the packages already built in, this device can install only pure-Python wheels tagged `none-any` for Python 3. "
        + "It does not resolve dependencies: if an import names another missing dependency, install that distribution "
        + "explicitly only if it is also pure Python. Other packages requiring compiled/native extensions or a source build, "
        + "npm/JavaScript packages, and native programs cannot be installed. "
        + "Do not retry a package the host reports as incompatible.";

    private readonly IInstallHook _hook;
    private readonly CodeExecOptions _options;
    private readonly Func<IReadOnlyList<string>> _networkHosts;
    private readonly Func<IReadOnlyList<BundledDistribution>> _bundled;

    /// <param name="bundled">
    /// What the app bundle already ships, asked on every install because the answer
    /// is cheap and the runtime may not have been discovered yet when this is built.
    /// Null means "nothing is known to be bundled", which is the desktop's situation.
    /// </param>
    public InstallHookPackageInstaller(
        IInstallHook hook,
        CodeExecOptions options,
        Func<IReadOnlyList<string>> networkHosts,
        Func<IReadOnlyList<BundledDistribution>>? bundled = null)
    {
        _hook = hook ?? throw new ArgumentNullException(nameof(hook));
        _options = options ?? throw new ArgumentNullException(nameof(options));
        _networkHosts = networkHosts ?? throw new ArgumentNullException(nameof(networkHosts));
        _bundled = bundled ?? (static () => Array.Empty<BundledDistribution>());
    }

    /// <inheritdoc />
    public bool CanInstall => _options.AllowInstall && _hook.CanInstall;

    /// <inheritdoc />
    public string? UnavailableReason => !_options.AllowInstall
        ? "installing packages is not enabled on this host"
        : _hook.UnavailableReason;

    /// <inheritdoc />
    public bool CanInstallLanguage(CodeLanguage language) =>
        language == CodeLanguage.Python && CanInstall;

    /// <inheritdoc />
    /// <remarks>
    /// True for a bundled distribution asked for by name or pinned to the bundled
    /// version, and for a COMPILED bundled distribution whatever the pin: no version of
    /// it can be fetched, so the request never needs the switch -- <see cref="Install"/>
    /// answers a mismatched pin by name. A pin to another version of a pure one is a
    /// real download and is not provided.
    /// </remarks>
    public bool IsProvided(CodeLanguage language, string package)
    {
        if (language != CodeLanguage.Python || string.IsNullOrWhiteSpace(package)
            || !WheelInstaller.TrySplit(package, out string name, out string? version, out _))
        {
            return false;
        }
        string canonical = CanonicalPackageName(name);
        BundledDistribution? shipped = _bundled().FirstOrDefault(
            d => string.Equals(d.CanonicalName, canonical, StringComparison.Ordinal));
        return shipped is not null
            && (shipped.Compiled
                || version is null
                || string.Equals(version, shipped.Version, StringComparison.OrdinalIgnoreCase));
    }

    /// <inheritdoc />
    /// <remarks>
    /// Built-in packages are always usable. Anything else is asked of the index the
    /// same way an install would ask (<see cref="WheelInstaller.ExplainUnavailableAsync"/>),
    /// so "pandas ships compiled code" is known before a skill that needs it is installed.
    /// </remarks>
    public string? ExplainUnavailable(CodeLanguage language, string package)
    {
        if (language != CodeLanguage.Python)
            return language == CodeLanguage.JavaScript
                ? "TensorAgent on this device installs Python packages only, not npm packages"
                : null;
        if (IsProvided(language, package) || _hook is not WheelInstaller wheels)
            return null;
        using var deadline = new CancellationTokenSource(TimeSpan.FromSeconds(20));
        try
        {
            string? why = wheels.ExplainUnavailableAsync(package, deadline.Token).GetAwaiter().GetResult();
            return why == null ? null : $"{package}: {why}";
        }
        catch (OperationCanceledException)
        {
            return null;
        }
    }

    /// <inheritdoc />
    public string? Install(
        SessionWorkspace workspace,
        CodeLanguage language,
        IReadOnlyList<string> packages,
        Action<string>? onOutput,
        out bool performed)
    {
        performed = false;
        ArgumentNullException.ThrowIfNull(workspace);
        packages ??= Array.Empty<string>();

        if (language != CodeLanguage.Python)
            return "TensorAgent installs Python packages only; npm packages are not available on this host";
        if (packages.Count == 0)
            return "no Python packages were named";
        if (packages.Count > MaxPackages)
            return $"too many packages requested ({packages.Count}); at most {MaxPackages} may be installed at once";

        var allowedNames = _options.AllowedPackages
            .Select(CanonicalPackageName)
            .ToHashSet(StringComparer.Ordinal);
        var validated = new List<(string Spec, string Name, string LedgerName, string? Version)>(packages.Count);
        foreach (string spec in packages)
        {
            if (!WheelInstaller.TrySplit(spec, out string name, out string? version, out string? invalid))
                return invalid;

            if (_options.AllowedPackages.Count > 0
                && !allowedNames.Contains(CanonicalPackageName(name)))
            {
                return $"'{name}' is not on this host's allowed-package list. Allowed: "
                     + string.Join(", ", _options.AllowedPackages) + ".";
            }

            validated.Add((spec.Trim(), name, CanonicalPackageName(name), version));
        }

        // The bundle is consulted BEFORE the switches. A request for something the app
        // already ships needs no network and no installer, and a model whose network is
        // off must still hear "lxml is built in" rather than "installing is not enabled"
        // -- the second sentence sends it looking for a setting it does not need.
        IReadOnlyList<BundledDistribution> bundled = _bundled();
        var pending = new List<(string Spec, string Name, string LedgerName, string? Version)>(validated.Count);
        var handledRequests = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var request in validated)
        {
            // PEP 503 treats runs of '-', '_' and '.' as equivalent. Use that
            // canonical spelling in the session ledger and when de-duplicating one
            // command, otherwise `zope.interface zope-interface` downloads the same
            // distribution twice. A pinned request still runs on a later call because
            // it may intentionally replace the installed version.
            string requestKey = request.LedgerName
                + (request.Version is null ? string.Empty : "==" + request.Version);
            if (!handledRequests.Add(requestKey))
                continue;

            BundledDistribution? shipped = bundled.FirstOrDefault(
                d => string.Equals(d.CanonicalName, request.LedgerName, StringComparison.Ordinal));
            if (shipped is null)
            {
                pending.Add(request);
                continue;
            }
            if (request.Version is null
                || string.Equals(request.Version, shipped.Version, StringComparison.OrdinalIgnoreCase))
            {
                onOutput?.Invoke($"{shipped.Name} {shipped.Version} is built into this app; nothing to install");
                continue;
            }
            if (shipped.Compiled)
            {
                return $"Could not install {request.Spec}: {shipped.Name} {shipped.Version} is compiled into this app "
                     + $"and cannot be replaced by version {request.Version}. Import the bundled one.";
            }
            // A pure-Python distribution CAN be shadowed: the session's package
            // directory precedes the bundle on sys.path, and the bootstrap evicts the
            // bundled copy from sys.modules when a session installs its own. So a
            // pinned request for another version of one goes through like any other.
            pending.Add(request);
        }
        if (pending.Count == 0)
            return null;

        if (!_options.AllowInstall)
        {
            return "installing packages is not enabled on this host "
                 + $"(an operator turns it on with {CodeExecOptions.AllowInstallFlag})";
        }
        if (!_hook.CanInstall)
            return _hook.UnavailableReason ?? "the in-process Python package installer is unavailable";

        TimeSpan timeout = _options.InstallTimeout > TimeSpan.Zero
            ? _options.InstallTimeout
            : TimeSpan.FromMilliseconds(1);
        var policy = new ExecutionPolicy(
            AllowScripts: _options.Enabled,
            AllowNetwork: _options.AllowNetwork,
            WorkRoot: workspace.WorkDirectory,
            ReadableRoots: new[] { workspace.Root },
            TempRoot: workspace.TempDirectory)
        {
            PackageRoot = workspace.EnvDirectory,
            NetworkHosts = _networkHosts(),
            DefaultTimeout = timeout,
        };

        var installedBeforeFailure = new List<string>();
        using (var deadline = new CancellationTokenSource())
        {
            deadline.CancelAfter(timeout);
            foreach (var (spec, name, ledgerName, version) in pending)
            {
                if (version is null && workspace.IsInstalled("python", ledgerName))
                    continue;

                ExecutionResult result;
                try
                {
                    result = _hook.InstallAsync(
                            new InstallRequest(
                                "python",
                                new[] { spec },
                                workspace.EnvDirectory,
                                policy)
                            {
                                OnOutputLine = onOutput,
                            },
                            deadline.Token)
                        .GetAwaiter().GetResult();
                }
                catch (OperationCanceledException) when (deadline.IsCancellationRequested)
                {
                    return Failure(
                        spec,
                        $"the install did not finish within {timeout.TotalSeconds:0.#}s",
                        installedBeforeFailure);
                }
                catch (Exception ex) when (ex is IOException or UnauthorizedAccessException
                                              or HttpRequestException or InvalidOperationException)
                {
                    return Failure(spec, ex.Message, installedBeforeFailure);
                }

                if (!result.Ok)
                {
                    string reason = result.TimedOut
                        ? $"the install did not finish within {timeout.TotalSeconds:0.#}s"
                        : FirstUsefulLine(result.Stderr) ?? FirstUsefulLine(result.Stdout)
                            ?? $"the installer exited with code {result.ExitCode}";
                    return Failure(spec, reason, installedBeforeFailure);
                }

                // WheelInstaller commits one distribution at a time. Record that fact
                // immediately so a later package failing in this same command cannot
                // make the next attempt download an already-visible wheel again.
                performed = true;
                workspace.MarkInstalled("python", new[] { ledgerName });
                installedBeforeFailure.Add(spec);
            }
        }

        return null;
    }

    private static string CanonicalPackageName(string name) => BundledPackages.Canonical(name);

    private static string Failure(
        string package, string reason, IReadOnlyList<string> installedBeforeFailure)
    {
        string prefix = installedBeforeFailure.Count == 0
            ? string.Empty
            : $"Installed {string.Join(", ", installedBeforeFailure)} before this failure. ";
        return prefix + $"Could not install {package}: {reason.Trim().TrimEnd('.')}.";
    }

    private static string? FirstUsefulLine(string? text)
    {
        if (string.IsNullOrWhiteSpace(text))
            return null;
        foreach (string line in text.Split('\n'))
        {
            string value = line.Trim();
            if (value.Length > 0)
                return value;
        }
        return null;
    }
}
