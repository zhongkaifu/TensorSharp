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
using System.Globalization;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp.AgentHost.Skills;
using TensorSharp.Runtime.Logging;

namespace TensorSharp.AgentHost.CodeExec
{
    /// <summary>
    /// What the host asks of an installer: whether it installs at all, and to put a
    /// named list of packages into a session's environment. <see cref="PackageInstaller"/>
    /// is the pip/npm implementation; a host with no package manager — or a wheel
    /// unpacker of its own — supplies another through <see cref="ShellRunner"/>.
    /// </summary>
    public interface IPackageInstaller
    {
        /// <summary>Whether this host installs anything at all.</summary>
        bool CanInstall { get; }

        /// <summary>Why installation is unavailable when <see cref="CanInstall"/> is false.</summary>
        string? UnavailableReason => null;

        /// <summary>
        /// Whether this host installs packages for one language. The default keeps
        /// existing pip/npm installers compatible; a narrower in-process installer can
        /// report its real surface without making every skill dependency look available.
        /// </summary>
        bool CanInstallLanguage(CodeLanguage language) => CanInstall;

        /// <summary>
        /// Whether <paramref name="package"/> (a name, or <c>name==version</c>) is
        /// already provided by the host itself — compiled into the app, in a host that
        /// ships its interpreter's packages — so that a request to install it fetches
        /// nothing and needs neither the network nor the install switch. The default is
        /// false: a desktop host provides nothing ahead of pip.
        /// </summary>
        bool IsProvided(CodeLanguage language, string package) => false;

        /// <summary>
        /// Install <paramref name="packages"/> into <paramref name="workspace"/>'s
        /// environment.
        /// </summary>
        /// <returns>Null on success or when there was nothing to do; otherwise why it failed, phrased for the model.</returns>
        /// <param name="performed">
        /// False when nothing was actually installed because every package was already in
        /// this session's ledger. <b>A null return does not mean an install happened.</b>
        /// </param>
        string? Install(
            SessionWorkspace workspace, CodeLanguage language,
            IReadOnlyList<string> packages, Action<string>? onOutput, out bool performed);

        /// <summary>Install, without asking whether anything actually happened.</summary>
        string? Install(
            SessionWorkspace workspace, CodeLanguage language,
            IReadOnlyList<string> packages, Action<string>? onOutput = null) =>
            Install(workspace, language, packages, onOutput, out _);
    }

    /// <summary>
    /// Installing packages the way the HOST asks for them: a validated name list, an
    /// argument vector the host builds, and a network hole that closes with the install.
    ///
    /// <para>
    /// Under the shell tool the model installs its own packages by typing
    /// <c>pip install pandas</c>, and this class is not on that path. It survives for the
    /// one case where the host is still the one asking: a skill SCRIPT whose import
    /// fails. There the package name comes from a traceback rather than from a command
    /// line, the host constructs the whole invocation, and every guarantee the old
    /// program runner had still holds — names are validated, wheels only, no build
    /// scripts, and the egress pinned to the registry allow-list.
    /// </para>
    /// <para>
    /// Kept separate from the shell runner precisely so those two paths cannot be
    /// confused for each other. What the host builds is checked to the letter; what the
    /// model types is confined by the sandbox and the proxy instead. Conflating them is
    /// how a control meant for one path quietly stops applying to the other.
    /// </para>
    /// </summary>
    public sealed class PackageInstaller : IPackageInstaller
    {
        private readonly CodeExecOptions _options;
        private readonly IShellBackend _backend;
        private readonly ILogger _logger;

        /// <param name="options">The host's terms: whether installing is allowed at all, the timeouts, the allow-list.</param>
        /// <param name="sandbox">The confinement to run the installer under, or null.</param>
        /// <param name="logger">Where denied egress is reported.</param>
        public PackageInstaller(CodeExecOptions options, ISkillSandbox? sandbox, ILogger? logger = null)
            : this(new ProcessShellBackend(sandbox, ModeOf(options), logger: logger), options, logger)
        {
        }

        /// <param name="backend">What launches the installer — a confined child process on a desktop.</param>
        /// <param name="options">The host's terms: whether installing is allowed at all, the timeouts, the allow-list.</param>
        /// <param name="logger">Where denied egress is reported.</param>
        public PackageInstaller(IShellBackend backend, CodeExecOptions options, ILogger? logger = null)
        {
            _backend = backend ?? throw new ArgumentNullException(nameof(backend));
            _options = options ?? throw new ArgumentNullException(nameof(options));
            _logger = logger ?? NullLogger.Instance;
        }

        /// <summary>The backend the installer launches through.</summary>
        public IShellBackend Backend => _backend;

        private static SkillSandboxMode ModeOf(CodeExecOptions options)
        {
            ArgumentNullException.ThrowIfNull(options);
            return options.Unconfined ? SkillSandboxMode.Preferred : options.Sandbox;
        }

        /// <summary>Whether this host installs anything at all.</summary>
        public bool CanInstall => _options.AllowInstall;

        /// <summary>
        /// Install <paramref name="packages"/> into <paramref name="workspace"/>'s
        /// environment.
        /// </summary>
        /// <returns>Null on success or when there was nothing to do; otherwise why it failed, phrased for the model.</returns>
        /// <param name="packages">
        /// The names to install. EMPTY means "whatever the manifest in the working
        /// directory says", which npm supports and pip does not — and which is a real
        /// install, not a no-op. Returning early on an empty list (as this did) reported
        /// "installed the dependencies named by the manifest" while installing nothing.
        /// </param>
        public string? Install(
            SessionWorkspace workspace, CodeLanguage language,
            IReadOnlyList<string> packages, Action<string>? onOutput = null) =>
            Install(workspace, language, packages, onOutput, out _);

        /// <param name="performed">
        /// False when nothing was actually installed because every package was already in
        /// this session's ledger. <b>A null return does not mean an install happened.</b>
        /// </param>
        /// <inheritdoc cref="Install(SessionWorkspace, CodeLanguage, IReadOnlyList{string}, Action{string})"/>
        public string? Install(
            SessionWorkspace workspace, CodeLanguage language,
            IReadOnlyList<string> packages, Action<string>? onOutput, out bool performed)
        {
            performed = false;
            ArgumentNullException.ThrowIfNull(workspace);
            packages ??= Array.Empty<string>();
            bool manifest = packages.Count == 0;
            if (manifest && language != CodeLanguage.JavaScript)
                return null;

            if (!_options.AllowInstall)
            {
                return "installing packages is not enabled on this host "
                     + $"(start it with {CodeExecOptions.AllowInstallFlag})";
            }
            if (!CodeExecOptions.SupportsInstall(language))
                return $"{CodeExecOptions.NameOf(language)} has no package installer here";
            if (!TryValidate(packages, out string? invalid, language))
                return invalid;
            if (!CodeEnvironment.TryResolveSessionInterpreter(workspace, language, out string? interpreter, out string? resolveError))
                return resolveError;

            string ledger = CodeExecOptions.NameOf(language);
            List<string> pending = packages
                .Where(p => BareName(p, language) != p || !workspace.IsInstalled(ledger, p))
                .ToList();
            // A manifest install always runs: its contents can change between calls, and
            // the ledger has no name to remember it by.
            //
            // Returning null here says "no error", which is NOT the same as "installed" —
            // and the two were being conflated one layer up. The auto-install loop asked
            // for a package that was already in the ledger, got null, told the model
            // "installed and the command was run again", and then, because the import
            // still failed, coached it to go looking for the package's real distribution
            // name. There was nothing to look for: the package was already there and the
            // problem was elsewhere. `performed` is the difference.
            if (pending.Count == 0 && !manifest)
                return null;

            string? failure = Run(language, interpreter!, workspace, pending, onOutput);
            if (failure != null)
                return failure;

            performed = true;
            workspace.MarkInstalled(ledger, pending.Select(p => BareName(p, language)));
            return null;
        }

        private string? Run(
            CodeLanguage language, string interpreter, SessionWorkspace workspace,
            IReadOnlyList<string> packages, Action<string>? onOutput)
        {
            string envDirectory = workspace.EnvDirectory;
            CodeInstallPlan plan;
            if (language == CodeLanguage.Python)
            {
                // A venv is not created: `pip install --target` populates a plain directory
                // that PYTHONPATH then points at, which needs no interpreter copy, no
                // activation, and no writable environment at run time.
                plan = CodeEnvironment.PythonInstall(
                    interpreter, envDirectory, packages, _options.InstallIndex);
            }
            else if (!CodeEnvironment.TryNpmInstall(interpreter, envDirectory, packages, out plan, out string? npmError))
            {
                return npmError;
            }

            using EgressProxy? proxy = ProxyOrNull();
            var environment = new Dictionary<string, string>(StringComparer.Ordinal);
            if (proxy != null)
            {
                string url = $"http://127.0.0.1:{proxy.Port.ToString(CultureInfo.InvariantCulture)}";
                environment["HTTPS_PROXY"] = url;
                environment["HTTP_PROXY"] = url;
                environment["https_proxy"] = url;
                environment["http_proxy"] = url;
                environment["NO_PROXY"] = string.Empty;
            }

            var argv = new List<string>(plan.Arguments.Count + 1) { plan.Interpreter };
            argv.AddRange(plan.Arguments);
            ConfinedResult result = _backend.Run(
                new ShellLaunch
                {
                    Argv = argv,
                    // A manifest install reads package.json from the read-only work tree
                    // and writes only into the session package environment.
                    WriteDirectory = workspace.EnvDirectory,
                    WorkingDirectory = workspace.WorkDirectory,
                    ReadOnlyDirectory = workspace.Root,
                    // Where the sandbox can pin one loopback port, the proxy is the
                    // installer's only route out. Where it cannot, it still gets the proxy
                    // through the environment — it simply is not prevented from ignoring it.
                    AllowNetwork = proxy == null || !ProxyIsEnforced,
                    AllowLoopbackPort = ProxyIsEnforced ? proxy?.Port : null,
                    Timeout = _options.InstallTimeout,
                    // The install is the longest silent stretch of all — a pip download is
                    // exactly what a user should be watching instead of a spinner.
                    OnOutputLine = onOutput,
                    Environment = environment,
                    Purpose = ShellLaunch.Purposes.Install,
                });

            if (result.Ok)
                return null;

            string why = result.Error
                ?? (result.TimedOut
                    ? $"it did not finish within {_options.InstallTimeout.TotalSeconds.ToString("0", CultureInfo.InvariantCulture)}s"
                    : $"the installer exited with code {result.ExitCode.ToString(CultureInfo.InvariantCulture)}");

            var sb = new StringBuilder();
            sb.Append("Could not install ").Append(string.Join(", ", packages)).Append(": ").Append(why).Append(".\n");

            // A host the egress proxy refused is THE reason the install failed, and pip
            // reports it only as an opaque connection error. Say it plainly, or the model
            // retries the same package against a wall it cannot see.
            IReadOnlyList<string> denied = proxy?.DrainDeniedHosts() ?? Array.Empty<string>();
            if (denied.Count > 0)
            {
                sb.Append("Installs may only reach this host's package registries (")
                  .Append(string.Join(", ", _options.EffectiveInstallDomains))
                  .Append("), and this one tried to reach: ")
                  .Append(string.Join(", ", denied.Take(6)))
                  .Append(". Those connections were refused.\n");
                _logger.LogInformation(LogEventIds.CodeExecEgressDenied,
                    "codeexec.egress.denied hosts={Hosts}", LogContent.Of(string.Join(",", denied.Take(6))));
            }

            // Not unconditional. This was appended after a TIMEOUT and after a proxy
            // denial as well, so a `pip install pandas` that merely ran out of time told
            // the model that pandas has no wheel for this platform — and a model that
            // believes that abandons a package that would have installed fine. Only said
            // when the installer actually ran and actually rejected the package.
            if (result.TimedOut)
            {
                sb.Append("The install did not finish in time — that is a deadline, not a verdict on the ")
                  .Append("package. Install fewer packages in one command, or name just the one you need ")
                  .Append("next, and try again.\n");
            }
            else if (denied.Count == 0 && result.Error == null && LooksLikeNetworkFailure(result.Stderr))
            {
                // A network failure is not a verdict on the package, and saying it was is
                // the worst answer available: the model abandons a package that installs
                // fine everywhere else, and the operator - the only one who can actually
                // fix it - is never told. Observed as a TLS handshake failure against
                // files.pythonhosted.org on a network that filters by SNI, reported to the
                // model as "no wheel for this platform".
                sb.Append("The installer could not REACH the package index. This is a network or TLS ")
                  .Append("failure on the host, not a problem with the package, and retrying will not ")
                  .Append("help - nothing you can write fixes it, so say so rather than trying another ")
                  .Append("package or another spelling. The host's operator can point installs at a ")
                  .Append("reachable mirror with ").Append(CodeExecOptions.InstallIndexFlag)
                  .Append(" (naming its download host in ").Append(CodeExecOptions.InstallDomainsFlag)
                  .Append(" too, if it serves files from a second hostname).\n");
            }
            else if (language == CodeLanguage.Python && denied.Count == 0 && result.Error == null)
            {
                sb.Append("Only prebuilt wheels are installed here (never source packages, which would run ")
                  .Append("their own build scripts), so a package with no wheel for this platform cannot be ")
                  .Append("used. Try a different package, or do without it.\n");
            }
            if (result.Stderr.Length > 0)
                sb.Append("\nInstaller output:\n").Append(Tail(result.Stderr, 1500));

            return sb.ToString();
        }

        /// <summary>
        /// A registry proxy for ONE install, disposed when that install returns.
        ///
        /// <para>
        /// It used to be created once and disposed with the runner, which left an
        /// unauthenticated loopback listener carrying the registry allowlist up for the
        /// process's whole life. That matters more than it looks: a sandbox is inherited
        /// and outlives its parent, so a child the installer backgrounded keeps the
        /// <c>localhost:&lt;port&gt;</c> allowance its profile granted. Ending the
        /// listener with the install is what makes that allowance worthless — there is
        /// nothing left on the port to talk to.
        /// </para>
        /// </summary>
        internal EgressProxy? ProxyOrNull()
        {
            if (_options.EffectiveInstallDomains.Count == 0)
                return null;
            return new EgressProxy(_options.EffectiveInstallDomains, _logger);
        }

        /// <summary>
        /// Whether the sandbox can make the proxy the installer's ONLY route out.
        ///
        /// <para>
        /// Seatbelt can admit exactly one loopback port, so there the allow-list is
        /// enforced: every other destination is unreachable at the OS level. Bubblewrap is
        /// all-or-nothing about the network and cannot, so there the installer is pointed
        /// at the proxy through <c>HTTPS_PROXY</c> and follows it because pip and npm
        /// honour that variable — a real narrowing, since the host built the argument
        /// vector and nothing in it says otherwise, but obedience rather than confinement.
        /// The two are reported differently because they ARE different, and a host that
        /// claims the stronger one while providing the weaker is the failure this whole
        /// area is written to avoid.
        /// </para>
        /// </summary>
        internal bool ProxyIsEnforced => _backend.Sandbox is { Name: "sandbox-exec" };

        /// <summary>
        /// Reject a package name that is not one.
        ///
        /// <para>
        /// These names come from a traceback and reach a command line and a package index.
        /// The argument vector already prevents shell interpretation, but
        /// <c>--index-url</c> arriving as a "package name" would redirect the whole
        /// install to a host of someone else's choosing.
        /// </para>
        /// </summary>
        internal bool TryValidate(IReadOnlyList<string> packages, out string? error,
            CodeLanguage language = CodeLanguage.Python)
        {
            error = null;

            if (packages.Count > MaxPackages)
            {
                error = $"too many packages requested ({packages.Count.ToString(CultureInfo.InvariantCulture)}); "
                      + $"at most {MaxPackages.ToString(CultureInfo.InvariantCulture)} may be installed at once";
                return false;
            }

            foreach (string package in packages)
            {
                bool npm = language == CodeLanguage.JavaScript;
                if (string.IsNullOrEmpty(package)
                    || !(npm ? NpmPackageName : PackageName).IsMatch(package)
                    || (npm && BareName(package, language).Length > 214))
                {
                    error = $"'{package}' is not a valid package name. " + (npm
                        ? "Use registry names, optionally scoped and versioned, such as '@scope/tool@1.2.3' or 'typescript@latest'. URLs, local paths and aliases are not accepted."
                        : "Use plain names, optionally with a version such as 'numpy==2.1.0'.");
                    return false;
                }

                if (_options.AllowedPackages.Count > 0)
                {
                    // Matched on the BARE name, so a version the model pins (numpy==2.1.0)
                    // still matches an operator's entry of "numpy".
                    string bare = BareName(package, language);
                    if (!_options.AllowedPackages.Any(a => string.Equals(a, bare, StringComparison.OrdinalIgnoreCase)))
                    {
                        error = $"'{bare}' is not on this host's allowed-package list. Allowed: "
                              + string.Join(", ", _options.AllowedPackages) + ".";
                        return false;
                    }
                }
            }

            return true;
        }

        private const int MaxPackages = 16;

        /// <summary>A name, optionally with extras and a version specifier. Nothing else.</summary>
        private static readonly Regex PackageName = new(
            @"\A[A-Za-z0-9][A-Za-z0-9._-]{0,63}(\[[A-Za-z0-9,._-]{1,64}\])?((==|>=|<=|~=|>|<)[A-Za-z0-9._-]{1,32})?\z",
            RegexOptions.Compiled | RegexOptions.CultureInvariant);

        // Registry package specs only. Slashes belong solely to @scope/name;
        // no git shorthand, file/URL requirements or npm aliases can change source.
        private static readonly Regex NpmPackageName = new(
            @"\A(@[A-Za-z0-9][A-Za-z0-9._-]{0,213}/)?[A-Za-z0-9][A-Za-z0-9._-]{0,213}(@[~^<>=]{0,2}[A-Za-z0-9*][A-Za-z0-9.*+_-]{0,127})?\z",
            RegexOptions.Compiled | RegexOptions.CultureInvariant);

        internal static string BareName(string package, CodeLanguage language = CodeLanguage.Python)
        {
            if (language == CodeLanguage.JavaScript)
            {
                int version = package.IndexOf('@', package.StartsWith('@') ? 1 : 0);
                return version < 0 ? package : package.Substring(0, version);
            }
            int cut = package.IndexOfAny(new[] { '[', '=', '>', '<', '~' });
            return cut < 0 ? package : package.Substring(0, cut);
        }

        /// <summary>
        /// True when the installer's own output says it could not REACH the index, rather
        /// than that it reached it and did not like what it found.
        ///
        /// <para>
        /// Every phrase here is one pip or npm prints only for a transport failure: a TLS
        /// handshake refused, a name that did not resolve, a connection reset or timed
        /// out, a proxy that would not tunnel. None can be produced by a package that
        /// merely has no wheel for this platform, which is what the host used to say.
        /// </para>
        /// </summary>
        internal static bool LooksLikeNetworkFailure(string? stderr)
        {
            if (string.IsNullOrEmpty(stderr))
                return false;
            foreach (string marker in NetworkFailureMarkers)
            {
                if (stderr!.Contains(marker, StringComparison.OrdinalIgnoreCase))
                    return true;
            }
            return false;
        }

        private static readonly string[] NetworkFailureMarkers =
        {
            "SSLError", "SSLV3_ALERT", "handshake failure", "CERTIFICATE_VERIFY_FAILED",
            "Max retries exceeded", "NewConnectionError", "ConnectionResetError",
            "Temporary failure in name resolution", "Name or service not known",
            "getaddrinfo", "ProxyError", "Tunnel connection failed",
            "Connection refused", "Read timed out", "network is unreachable",
            "ETIMEDOUT", "ECONNREFUSED", "ENOTFOUND", "ECONNRESET", "EAI_AGAIN",
        };

        private static string Tail(string text, int max) =>
            text.Length <= max ? text : "…" + text.Substring(text.Length - max);
    }
}
