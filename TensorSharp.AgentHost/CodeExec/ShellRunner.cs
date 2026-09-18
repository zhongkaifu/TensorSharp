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
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp.AgentHost.Skills;
using TensorSharp.Runtime.Logging;

namespace TensorSharp.AgentHost.CodeExec
{
    /// <summary>One <c>shell</c> call, after its arguments have been read.</summary>
    /// <param name="Command">The command line, exactly as the model wrote it.</param>
    public readonly record struct ShellRequest(string Command)
    {
        /// <summary>A directory inside the workspace to run in, or null to continue where the last command left off.</summary>
        public string? WorkDirectory { get; init; }

        /// <summary>What the call asked for, bounded by the host's maximum.</summary>
        public TimeSpan? Timeout { get; init; }

        /// <summary>Start it and return immediately, leaving it running.</summary>
        public bool Background { get; init; }

        /// <summary>Directories outside the workspace the command may read, e.g. selected skills.</summary>
        public IReadOnlyList<string> ReadablePaths { get; init; } = Array.Empty<string>();
    }

    /// <summary>
    /// What the host's install-and-retry loop did, by MODULE.
    ///
    /// <para>
    /// Two sets rather than one flag, because the two outcomes need opposite advice and
    /// telling them apart by re-reading the note the loop wrote would be guessing at own
    /// prose. "Installed it, re-ran, the import still fails" means the package is called
    /// something else; "the install itself was refused" means it is not coming, and the
    /// reason is already in the result. Saying the first when the second happened would
    /// be the host stating a constraint that is not the real one.
    /// </para>
    /// </summary>
    /// <param name="Installed">Modules a package was successfully installed for.</param>
    /// <param name="Failed">Modules whose install was refused or failed.</param>
    public readonly record struct AutoInstallOutcome(
        IReadOnlySet<string>? Installed, IReadOnlySet<string>? Failed)
    {
        /// <summary>True when a package was installed for <paramref name="module"/> and the command re-run.</summary>
        public bool WasInstalled(string module) => Installed?.Contains(module) == true;

        /// <summary>True when installing for <paramref name="module"/> was attempted and did not work.</summary>
        public bool InstallFailed(string module) => Failed?.Contains(module) == true;
    }

    /// <summary>
    /// Runs the model's shell commands and backs its read/edit/write/patch tools.
    ///
    /// <para>
    /// The surface separates execution from mutation: the shell runs and verifies code;
    /// <c>read_file</c> supplies bounded current context; <c>write_file</c> creates a
    /// new file; and <c>apply_patch</c> updates one or multiple files atomically. Keeping local
    /// fixes out of heredocs is what avoids re-emitting and re-rolling hundreds of lines
    /// that were already correct.
    /// </para>
    /// <para>
    /// Nothing about the CONFINEMENT changed, and that is deliberate: the sandbox, the
    /// path guard, the session workspace, the artifact store and the registry proxy are
    /// the hard-won parts and this is simply a new client of them. The one policy that
    /// had to be rebuilt is the network, because it used to be a property of the PHASE
    /// (install versus run) and a shell has no phases — see <see cref="ShellCommand"/>.
    /// </para>
    /// </summary>
    public sealed class ShellRunner : IDisposable
    {
        private readonly CodeExecOptions _options;
        private readonly ILogger _logger;
        private readonly IShellBackend _backend;
        private readonly ISkillSandbox? _sandbox;
        private readonly IApiProbe _apiProbe;
        private readonly ISyntaxVerifier _syntax;
        private readonly CodeArtifactStore? _artifacts;
        private readonly IPackageInstaller _installer;
        private readonly ShellProgram? _shell;
        private readonly ConcurrentDictionary<string, ShellSession> _sessions = new(StringComparer.Ordinal);
        private readonly ConcurrentDictionary<string, BackgroundJobs> _jobs = new(StringComparer.Ordinal);
        private readonly ConcurrentDictionary<string, ConcurrentDictionary<string, Lazy<string?>>>
            _sanitizedCertificateBundles = new(StringComparer.Ordinal);

        /// <param name="options">The host's terms.</param>
        /// <param name="logger">Where runs are recorded, as metadata only.</param>
        /// <param name="artifacts">Where produced files are kept, or null to keep none.</param>
        /// <param name="backend">
        /// What actually runs a launch. Null is today's desktop behaviour — a confined
        /// child process under the detected OS sandbox and the shell found on PATH
        /// (<see cref="ProcessShellBackend.Detect(CodeExecOptions, ILogger?, ISkillSandbox?, ShellProgram?)"/>).
        /// A host that cannot start processes supplies its own; its
        /// <see cref="IShellBackend.Sandbox"/> is then what every confinement question
        /// here is answered from, so an in-process backend with honest capabilities
        /// runs without <c>--code-exec-unconfined</c>.
        /// </param>
        /// <param name="sandbox">
        /// A sandbox for the default backend to use instead of detecting one. Ignored
        /// when <paramref name="backend"/> is supplied — a backend owns its own.
        /// </param>
        /// <param name="shell">
        /// A shell for the default backend to use instead of resolving one from PATH.
        /// Ignored when <paramref name="backend"/> is supplied.
        /// </param>
        /// <param name="syntax">Parses files after a write; null uses the interpreters the backend launches.</param>
        /// <param name="apiProbe">Reads a package's real API after a failed run; null uses the backend's interpreters.</param>
        /// <param name="installer">Performs installs; null is pip/npm through the backend.</param>
        public ShellRunner(
            CodeExecOptions? options = null,
            ILogger? logger = null,
            CodeArtifactStore? artifacts = null,
            IShellBackend? backend = null,
            ISkillSandbox? sandbox = null,
            ShellProgram? shell = null,
            ISyntaxVerifier? syntax = null,
            IApiProbe? apiProbe = null,
            IPackageInstaller? installer = null)
        {
            _options = options ?? new CodeExecOptions();
            _logger = logger ?? NullLogger.Instance;
            _backend = backend ?? ProcessShellBackend.Detect(_options, _logger, sandbox, shell);
            // Read off the backend rather than detected here, so that everything this
            // class says about confinement — CanRun, the declaration's network promise,
            // the result's gap list — is about what will actually run the command.
            _sandbox = _backend.Sandbox;
            _shell = _backend.Shell;
            _artifacts = artifacts;
            _installer = installer ?? new PackageInstaller(_backend, _options, _logger);
            _apiProbe = apiProbe ?? new ApiProbe(_backend);
            _syntax = syntax ?? new SyntaxCheck(_backend);

            // The watchdog's sink is static because the callers that trip it — a syntax
            // check, a violation monitor, an interpreter probe — hold no logger. First
            // runner to be built wins; they all log to the same place.
            ForkWatchdog.Observer ??= detail => _logger.LogWarning(
                LogEventIds.CodeExecForkWedged, "codeexec.forkwedged {Detail}", detail);
        }

        /// <summary>What runs each launch: a confined child process, or the host's own runtime.</summary>
        public IShellBackend Backend => _backend;

        /// <summary>The sandbox in use, or null.</summary>
        public ISkillSandbox? Sandbox => _sandbox;

        /// <summary>The resolved shell, or null when this host has none.</summary>
        public ShellProgram? Shell => _shell;

        /// <summary>The host's options, for a caller that has to describe them.</summary>
        public CodeExecOptions Options => _options;

        /// <summary>Whether files a command writes are kept for the user.</summary>
        public bool KeepsArtifacts => _artifacts != null;

        /// <summary>The installer skill scripts use for their own missing dependencies.</summary>
        public IPackageInstaller Installer => _installer;

        /// <summary>
        /// Whether commands may run here at all.
        ///
        /// <para>
        /// Asks whether the host actually confines every capability the operator did not
        /// deliberately open, rather than whether a sandbox object merely exists. A
        /// Windows job object bounds CPU and memory but cannot restrict a file or a socket,
        /// and an existence test made the default behave like "preferred" there, quietly
        /// running model-written commands with the filesystem open. Network confinement
        /// is not required after the operator explicitly enables network access; write
        /// confinement still is. The only way past a remaining "no" is
        /// <c>--code-exec-unconfined</c>, an explicit statement by an operator about their
        /// own machine.
        /// </para>
        /// </summary>
        public bool CanRun =>
            _options.Enabled
            && _shell != null
            && _backend.CanRun
            && (_options.Unconfined
                || _options.Sandbox != SkillSandboxMode.Required
                || Confines(_sandbox, requireNetworkConfinement: !_options.AllowNetwork));

        /// <summary>
        /// Whether the declaration can promise IP-network confinement before a command
        /// starts. Required mode fails closed if wrapping fails; preferred/unconfined
        /// modes may deliberately fall back to a raw process and therefore cannot make
        /// that promise even when a capable sandbox was detected at startup.
        /// </summary>
        internal bool NetworkConfinementGuaranteed =>
            !_options.AllowNetwork
            && !_options.Unconfined
            && _options.Sandbox == SkillSandboxMode.Required
            && _sandbox?.Capabilities.ConfinesNetwork == true;

        private static bool Confines(ISkillSandbox? sandbox, bool requireNetworkConfinement) =>
            sandbox is not null
            && sandbox.Capabilities.ConfinesWrites
            && (!requireNetworkConfinement || sandbox.Capabilities.ConfinesNetwork);

        /// <summary>Why <see cref="CanRun"/> is false, or null.</summary>
        public string? UnavailableReason
        {
            get
            {
                if (CanRun)
                    return null;
                if (!_options.Enabled)
                {
                    return "running model-written commands is not enabled on this host "
                         + $"(start it with {CodeExecOptions.EnabledFlag})";
                }
                if (_shell == null || !_backend.CanRun)
                    return _backend.UnavailableReason ?? "this host has no shell to run commands with";

                if (_sandbox is { } present)
                {
                    IReadOnlyList<string> gaps = CommandGaps(
                        present.Capabilities, includeNetwork: !_options.AllowNetwork);
                    return $"this host's sandbox ({present.Name}) cannot confine commands: "
                         + string.Join("; ", gaps)
                         + ". Running commands written during the request needs real isolation, so it is refused"
                         + $" — pass {CodeExecOptions.UnconfinedFlag} to accept the risk on a machine that is yours";
                }

                string sandboxDetail = OperatingSystem.IsLinux()
                    ? $" ({BubblewrapSandbox.AvailabilityError})"
                    : string.Empty;
                return "this host provides no OS sandbox" + sandboxDetail
                     + ", and running model-written commands without one is refused"
                     + $" — pass {CodeExecOptions.UnconfinedFlag} to accept the risk on a machine that is yours";
            }
        }

        /// <summary>The persisted shell state for one session, created on first use.</summary>
        public ShellSession SessionFor(SessionWorkspace workspace)
        {
            ArgumentNullException.ThrowIfNull(workspace);
            if (_shell == null)
                throw new InvalidOperationException("no shell is available on this host");

            string key = workspace.Root;
            if (_sessions.TryGetValue(key, out ShellSession? existing))
                return existing;

            var created = new ShellSession(workspace, _shell);
            if (_sessions.TryAdd(key, created))
            {
                // Forget it when the session ends. Without this the map grows one entry per
                // conversation for the life of the process — small each, unbounded together,
                // and on the CLI an ephemeral workspace per call makes it grow per CALL.
                workspace.RegisterCleanup(new Forget(
                    _sessions, _jobs, _sanitizedCertificateBundles, key));
                return created;
            }
            return _sessions[key];
        }

        /// <summary>Removes one workspace's entries when that workspace is released.</summary>
        private sealed class Forget : IDisposable
        {
            private readonly ConcurrentDictionary<string, ShellSession> _sessions;
            private readonly ConcurrentDictionary<string, BackgroundJobs> _jobs;
            private readonly ConcurrentDictionary<string, ConcurrentDictionary<string, Lazy<string?>>>
                _certificateBundles;
            private readonly string _key;

            public Forget(
                ConcurrentDictionary<string, ShellSession> sessions,
                ConcurrentDictionary<string, BackgroundJobs> jobs,
                ConcurrentDictionary<string, ConcurrentDictionary<string, Lazy<string?>>> certificateBundles,
                string key)
            {
                _sessions = sessions;
                _jobs = jobs;
                _certificateBundles = certificateBundles;
                _key = key;
            }

            public void Dispose()
            {
                _sessions.TryRemove(_key, out _);
                _jobs.TryRemove(_key, out _);
                _certificateBundles.TryRemove(_key, out _);
            }
        }

        // ---- running -------------------------------------------------------

        /// <summary>Run one command and return what it printed.</summary>
        /// <param name="request">What to run.</param>
        /// <param name="workspace">
        /// The session's workspace. Null gets a throwaway one that is deleted when the
        /// call returns — useful only to a direct one-shot caller. Interactive hosts and
        /// HTTP agent loops supply a session- or request-scoped workspace.
        /// </param>
        /// <param name="onOutput">
        /// Live tap, called per line while the command runs, so a host can show progress
        /// instead of a silent minute. Called from reader threads.
        /// </param>
        public CodeExecResult Run(
            ShellRequest request, SessionWorkspace? workspace = null, Action<string>? onOutput = null)
        {
            if (!CanRun)
                return CodeExecResult.Refused(UnavailableReason ?? "code execution is unavailable");

            string command = request.Command ?? string.Empty;
            if (string.IsNullOrWhiteSpace(command))
            {
                return CodeExecResult.Refused(
                    "the 'command' argument was empty. Send the command line to run, for example "
                    + Example("ls -la") + ".");
            }

            if (workspace == null && request.Background)
            {
                return CodeExecResult.Refused(
                    "this request has no working directory that outlives the call, so a background "
                    + "job would be killed the moment it started. Run the command in the foreground, "
                    + "raising timeout_ms if it needs longer.");
            }

            SessionWorkspace session = workspace ?? CreateEphemeral();
            bool ephemeral = workspace == null;
            try
            {
                return RunIn(request, command, session, onOutput, fileToolsAvailable: workspace != null);
            }
            finally
            {
                if (ephemeral)
                {
                    session.RunCleanups();
                    TryDeleteDirectory(session.Root);
                }
            }
        }

        private CodeExecResult RunIn(
            ShellRequest request, string command, SessionWorkspace workspace, Action<string>? onOutput,
            bool fileToolsAvailable)
        {
            // The command AS THE MODEL WROTE IT. `command` is rewritten further down —
            // TryPerformInstalls reads each install out of the line and substitutes it
            // with `true` or `false` — so anything that has to reason about what the model
            // SENT needs this copy. Keying the repeat ledger on the residual would mean
            // `pip install X && python y.py` hashes differently depending on whether the
            // install succeeded, so the one shape that most needs a repeat warning — a
            // retry after a failed install — would never match its own predecessor.
            string typed = command;

            // Whether every install this line asked for actually happened. Declared here
            // because the answer has to reach Describe, which reports success — a refused
            // install whose residual line exits 0 must not be reported as a success.
            bool installsOk = true;

            // apply_patch typed into the shell is answered HERE, by the host, and nothing
            // is executed. Codex does the same, and here it also removes a dependency the
            // sandbox cannot be assumed to have: there is no apply_patch binary to install
            // and no interpreter guaranteed to be present to implement one.
            if (ShellCommand.TryReadApplyPatch(command, out string inlinePatch))
                return ApplyPatch(inlinePatch, workspace);

            // There is no apply_patch BINARY in the sandbox — the host answers it. So a
            // command that invokes it in a shape the host cannot answer must be refused
            // outright, not passed to the shell: the shell would report "command not
            // found" for that one word, run everything after it, and exit with the LAST
            // command's status. The model then reads `exit 0` and the program's output and
            // concludes its edit landed, when the file was never touched.
            // A trailing `&` cannot do what the model means by it: every call is a fresh
            // confined process whose launched job is stopped when the call returns, so an
            // ordinary shell background job is dead before the next call can look at it.
            // Refused with the parameter
            // that DOES work — the shape of refusal this codebase uses everywhere, and the
            // shape Claude Code's own shims use ("Narrow the pattern, or target your own
            // children with `pkill -P $$ ...`").
            if (ShellCommand.EndsWithBackground(command))
            {
                return CodeExecResult.Refused(
                    "a trailing '&' does not work here. Every command is its own confined process and "
                    + "its launched job is stopped when the call returns, so a job put in the "
                    + "background with '&' is gone before you can read from it. Use the "
                    + "run_in_background argument instead: the host keeps that job alive for the rest of "
                    + "the conversation and gives you a log file to read with an ordinary command.");
            }

            if (ShellCommand.InvokesApplyPatch(command))
            {
                return CodeExecResult.Refused(
                    "this command calls apply_patch in a shape that cannot be answered here, and "
                    + "running it would silently skip the patch while the rest of the line succeeded. "
                    + "apply_patch must be the WHOLE command, with the envelope as one heredoc:\n"
                    + "  apply_patch <<'PATCH'\n  *** Begin Patch\n  ...\n  *** End Patch\n  PATCH\n"
                    + "Send anything else as a separate command — or call the apply_patch tool directly.");
            }

            // The workspace's directories are re-asserted here rather than trusted from
            // construction time. They sit in a scratch root beside the binary that every
            // host launched from that build shares, and things outside this conversation
            // remove them — another host's startup sweep, a cleanup, a rebuild into the
            // same output directory. When that happened every later command in the
            // conversation failed at the point where its wrapper script is written,
            // `echo hello` included, and nothing could recover. Repairing costs five
            // existence checks against a call that costs tens of milliseconds.
            //
            // OR, not assignment: whichever entry point noticed the damage first did the
            // repair, and the latch carries that fact to the result the model reads.
            bool workspaceRebuilt = workspace.TryEnsureDirectories() | workspace.ConsumeRebuiltNotice();

            if (!TryResolveWorkDirectory(request.WorkDirectory, workspace, out string? workDirectory, out string? workError))
                return CodeExecResult.Refused(workError!);

            // Installs are performed BY THE HOST and never widen the command's separately
            // configured network policy. See ShellInstall for why: granting a socket just
            // for an install would either let the model choose its index or share that
            // reach with everything else on the line. Reading the request and building
            // the argument vector ourselves closes both holes on every platform, while a
            // command gets general network access only from the explicit network switch.
            var notes = new List<string>();

            // Said in the tool RESULT, because the files this conversation wrote are gone
            // and nothing else will tell the model so. Without it the model reads a run of
            // unexplained "No such file" answers about files it watched itself create.
            if (workspaceRebuilt)
                notes.Add(SessionWorkspace.RebuiltNote);

            if (ShellCommand.ContainsInstall(command))
            {
                // The switch is about FETCHING. A request for something the host already
                // provides fetches nothing, so it goes through to the installer, which
                // answers for the bundle -- otherwise a model whose network is off and who
                // types `pip install lxml` out of habit is told installing is disabled, on
                // a device where lxml was importable the whole time.
                if (!_options.AllowInstall && !EveryPackageIsProvided(command, workspace))
                {
                    return CodeExecResult.Refused(
                        "this command installs packages, and installing is not enabled on this host "
                        + $"(an operator turns it on with {CodeExecOptions.AllowInstallFlag}). "
                        + "Nothing was run. Use what is already available, or say in your answer that "
                        + "this step needs a package the host does not provide.");
                }

                if (request.Background)
                {
                    // Checked before the install runs, so a refusal leaves nothing behind.
                    // Installs are performed by the host up front, so backgrounding such a
                    // line would report a job for work that is already finished — and the
                    // model would go looking in the log for install output that was never
                    // going to be there.
                    return CodeExecResult.Refused(
                        "this command installs packages, and installs are performed before anything is "
                        + "backgrounded — so there is nothing useful to background here. Install in the "
                        + "foreground, then background whatever needs the packages.");
                }

                if (!ShellInstall.TryRead(command, RelativeReader(workspace),
                        out IReadOnlyList<ShellInstallRequest> installs, out string? installError))
                {
                    return CodeExecResult.Refused(installError!);
                }

                installsOk = TryPerformInstalls(
                    command, installs, workspace, onOutput, notes, out command);

                // Every install in the line has been answered and substituted out. What is
                // left may be nothing at all — in which case the call's success IS the
                // installs' success.
                if (command.Trim().Length == 0)
                {
                    return new CodeExecResult(
                        installsOk, string.Join("\n", notes) + "\n",
                        Array.Empty<CodeArtifact>(), string.Empty);
                }
            }

            EnsureInterpreterAliases(workspace);
            EnsureNodeResolution(workspace);
            ShellSession session = SessionFor(workspace);
            var environment = BuildEnvironment(workspace, out IReadOnlyList<string> networkReadablePaths);

            // Everything the host has decided, handed across the seam. The backend
            // decides nothing about policy: what may be written, read or reached, for
            // how long, and in which directory are all settled here, and a backend that
            // is a child process and one that is an in-process interpreter receive the
            // same terms.
            var launch = new ShellLaunch
            {
                Command = command,
                Session = session,
                CallWorkDirectory = workDirectory,
                WorkingDirectory = workDirectory ?? session.CurrentDirectory,
                // Generated code writes its work tree plus narrowly separated temp and
                // shell-persistence roots. The package environment and host-authored
                // scripts/logs stay read-only, so a symlink planted in one command cannot
                // redirect a later host write outside the sandbox.
                WriteDirectory = workspace.WorkDirectory,
                WritablePaths = new[] { workspace.TempDirectory, workspace.ShellStateDirectory },
                ReadOnlyDirectory = workspace.Root,
                ReadablePaths = ReadablePathsFor(
                    workspace, request.ReadablePaths, networkReadablePaths),
                // Network access is a distinct operator decision. Package-install
                // permission does not imply it: installs are still performed by the host,
                // while this controls the generated command's own sockets.
                AllowNetwork = _options.AllowNetwork,
                Timeout = TimeoutFor(request.Timeout),
                MaxOutputBytes = _options.MaxOutputBytes,
                Environment = environment,
                OnOutputLine = onOutput,
                Purpose = ShellLaunch.Purposes.Shell,
            };

            if (request.Background)
            {
                return StartBackground(workspace, launch);
            }

            // Keyed on the command AS THE MODEL WROTE IT, captured before installs were
            // read out of it and substituted with `true`/`false`. Keying the residual
            // would mean `pip install X && python y.py` hashes differently depending on
            // whether the install succeeded — so the one shape that most needs a repeat
            // warning, a retry after a failed install, would never match its predecessor.
            (int repeats, bool failedBefore) = session.RecordAttempt(typed);
            RewriteWatch? rewrites = RewriteWatch.Before(typed, workspace, launch.WorkingDirectory);

            Dictionary<string, (long Length, DateTime WriteTime)> before = workspace.SnapshotWorkFiles();
            ConfinedResult result = RunWithAutoInstall(
                launch, workspace, notes, onOutput, out AutoInstallOutcome installedFor);
            session.RecordOutcome(typed, result.Ok);

            // Everything the command created or changed becomes content the model is
            // taken to have seen.
            //
            // The bytes are the model's own — it typed the heredoc — so treating them as
            // unread would gate the very first edit of a file on re-reading something it
            // had just written, which is a round spent proving something already true.
            // It is also what makes a heredoc-created file WATCHABLE: 9 of the 9
            // whole-file rewrites in the logs were over a file created earlier in the
            // same conversation, and without the previous content there is nothing to
            // compare a rewrite against. The snapshot this walks is already taken for
            // artifact capture, so this costs a read of the files that actually changed
            // and nothing at all for the common command that changes none.
            IngestWrittenFiles(workspace, before, typed);

            // A command that never STARTED is a host fault, not a failing command, and the
            // two are indistinguishable at Information: the 2026-09-10 incident logged
            // eight of these as ordinary runs, and the only clue that nothing had executed
            // was exit=-1 with ms=0. It is logged as a warning, with the reason, so that
            // "the shell keeps failing" is one grep rather than an inference.
            if (!result.Started)
            {
                _logger.LogWarning(LogEventIds.CodeExecRan,
                    "codeexec.not-started shell={Shell} reason={Reason}",
                    _shell!.Name, OutputPaths.Scrub(result.Error ?? "(none)", workspace, launch.WorkingDirectory));
            }

            _logger.LogInformation(LogEventIds.CodeExecRan,
                "codeexec.ran shell={Shell} sandbox={Sandbox} installs={Installs} exit={Exit} timedOut={TimedOut} ms={Ms} bytes={Bytes}",
                _shell!.Name, result.SandboxName, notes.Count,
                result.ExitCode, result.TimedOut, (long)result.Elapsed.TotalMilliseconds,
                result.Stdout.Length + result.Stderr.Length);

            return Describe(
                // The TYPED command, not the install-substituted residual: everything
                // Describe does with it — parsing imports for the API probe, finding the
                // files the command redirected into — is reasoning about what the model
                // wrote, not about the line the shell finally received.
                typed, Scrub(result, workspace, launch.WorkingDirectory),
                workspace, session, before, notes, launch.Timeout,
                repeats, failedBefore, rewrites, installedFor, installsOk, fileToolsAvailable);
        }

        /// <summary>
        /// Run the command; when it dies on a missing import and this host can install,
        /// install that module and run the command AGAIN — inside the same call.
        ///
        /// <para>
        /// <b>This is the largest single cost in this server's measured logs.</b> Across
        /// three days of real traffic, "the first run dies on a missing dependency" was
        /// 17 incidents, <b>68 rounds, 117,000 output tokens and 116 minutes of wall
        /// clock</b> — more than any other failure, and more than the API-guessing loop
        /// that <see cref="ApiProbe"/> exists for. The host already detected the module
        /// and already knew how to install it; it then told the model
        /// <i>"install it and run the command again"</i> and ended the round.
        /// </para>
        /// <para>
        /// The reason that advice is so expensive is not the round. It is that the
        /// command is <b>re-typed</b>: a shell call carries its whole program in a
        /// heredoc, so "run it again" means decoding sixteen thousand characters a second
        /// time to change nothing. Fourteen of the twenty-four byte-identical repeated
        /// commands in the corpus are exactly this, and twelve of them are the very first
        /// thing that happens in the turn. At a measured 53 ms per output token, a
        /// 4,000-token re-type is three and a half minutes to say what the host already
        /// knew.
        /// </para>
        /// <para>
        /// The same loop already exists one layer over for skill scripts
        /// (<c>SkillScriptRunner.RunWithAutoInstall</c>), written for the same reason and
        /// with the same bounds. This is that loop applied to the surface the model
        /// actually types into.
        /// </para>
        /// <para>
        /// <b>Bounds, all of which matter.</b> Only when installing is enabled and the
        /// package passes the host's allow-list — the install goes through
        /// <see cref="PackageInstaller"/>, so <c>--code-exec-packages</c> is enforced
        /// exactly as it is for an install the model asked for out loud. At most
        /// <see cref="CodeExecOptions.MaxAutoInstalls"/> distinct packages, and never the
        /// same one twice: a package that installs and does not fix the import is a
        /// different problem, and retrying it is the infinite loop this shape invites.
        /// Never for a background job, which has no output to diagnose yet.
        /// </para>
        /// <para>
        /// Re-running is not free of consequence — a command that wrote files before it
        /// failed writes them again. That is accepted deliberately, because it is
        /// precisely what the model does when it is told to run the command again, and
        /// doing it here costs one process instead of a round and a re-typed program.
        /// </para>
        /// </summary>
        private ConfinedResult RunWithAutoInstall(
            ShellLaunch launch, SessionWorkspace workspace, List<string> notes,
            Action<string>? onOutput, out AutoInstallOutcome installedFor)
        {
            // The whole CALL's deadline, not each process's. launch.Timeout bounds one
            // run; without a budget across the loop, a command asking for 20s could spend
            // six runs plus five installs — over two minutes — inside a single tool call
            // whose declaration promised 20 seconds. The caller's timeout_ms is a contract,
            // and a recovery loop is not a licence to ignore it.
            var budget = Stopwatch.StartNew();
            var attempted = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            // The MODULES an install actually succeeded for, so the coaching below can
            // tell "installed and still missing" from "the install itself failed" without
            // reading its own prose back.
            var installed = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            var failed = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            var outcome = new AutoInstallOutcome(installed, failed);
            installedFor = outcome;

            while (true)
            {
                ConfinedResult result = _backend.Run(launch);

                if (result.Ok || result.TimedOut || !result.Started
                    || !_installer.CanInstall
                    || attempted.Count >= Math.Max(1, _options.MaxAutoInstalls)
                    || budget.Elapsed >= launch.Timeout)
                {
                    return result;
                }

                string diagnosis = result.Stderr.Length > 0 ? result.Stderr : result.Stdout;
                if (!CodeDiagnostics.TryFindMissingModule(diagnosis, out CodeLanguage language, out string module))
                    return result;

                // Before installing a name the MODEL wrote, check it is not the model's own
                // file. `import helpers` failing when `helpers.py` is sitting in a
                // subdirectory is an import-PATH problem, and the host reaching out to a
                // package registry for a name that means something else entirely is both
                // the wrong answer and a way to install an arbitrary package chosen by
                // whatever the model happened to type. This is the check-the-environment-
                // first principle where it actually bites: look at what is on disk before
                // acting on a guess about what is missing.
                if (LocalModuleFor(workspace, module) is { } ownFile)
                {
                    notes.Add(
                        $"'{module}' is your own file at {ownFile}, not a package — it was not installed. "
                        + "It failed to import because its directory is not on the import path: run the "
                        + "program from that directory, or put the file beside the program that imports it.");
                    return result;
                }

                string package = CodeDiagnostics.InstallNameFor(language, module);
                if (!attempted.Add(package))
                    return result;

                onOutput?.Invoke($"[installing missing dependency: {package}]");
                string? error = _installer.Install(
                    workspace, language, new[] { package }, onOutput, out bool performed);

                // Nothing was installed because it was already in this session's ledger,
                // and the import STILL fails. Saying "installed and re-ran" here was a
                // false statement, and the advice that followed it — go and find the real
                // distribution name — sent the model hunting for a package that is already
                // on disk. What is actually wrong is downstream of the install.
                if (error == null && !performed)
                {
                    notes.Add(
                        $"'{module}' is missing, but {package} is already installed in this session — so "
                        + "nothing was installed and running the command again would fail the same way. "
                        + "Either the installed package does not provide that module (check its real "
                        + "import name), or the module lives somewhere the interpreter is not looking.");
                    return result;
                }

                if (error != null)
                {
                    // The install was refused or failed. Say why HERE rather than letting
                    // the bare ImportError stand: "numpy is not on this host's allow-list"
                    // is a fact the model can act on, and the traceback is not.
                    notes.Add($"'{module}' was missing, and installing {package} did not work: {error}");
                    failed.Add(module);
                    return result;
                }

                installed.Add(module);

                // The install itself takes seconds. If it consumed the call's deadline,
                // say so instead of re-running past it — and say it as the next action,
                // because the re-run is now cheap: the package is already there.
                if (budget.Elapsed >= launch.Timeout)
                {
                    notes.Add($"'{module}' was missing, so {package} was installed — but there was no time "
                            + "left in this call to run the command again. Run it again; the package is "
                            + "installed now.");
                    return result;
                }

                notes.Add($"'{module}' was missing, so {package} was installed and the command was run again.");
                onOutput?.Invoke("[re-running the command]");
            }
        }

        /// <summary>
        /// True when every install on the line names only packages the host itself
        /// provides (<see cref="IPackageInstaller.IsProvided"/>). A manifest install
        /// fetches by definition, and an unreadable line is not vouched for.
        /// </summary>
        private bool EveryPackageIsProvided(string command, SessionWorkspace workspace)
        {
            if (!ShellInstall.TryRead(command, RelativeReader(workspace),
                    out IReadOnlyList<ShellInstallRequest> installs, out _)
                || installs.Count == 0)
            {
                return false;
            }
            foreach (ShellInstallRequest install in installs)
            {
                if (install.Packages.Count == 0)
                    return false;
                foreach (string package in install.Packages)
                {
                    if (!_installer.IsProvided(install.Language, package))
                        return false;
                }
            }
            return true;
        }

        /// <summary>
        /// Rewrite every host path in the output into the model's frame of reference.
        ///
        /// <para>
        /// The model's own files, the package environment, the temp directory. See
        /// OutputPaths — 13.9% of the characters in this server's logged tool results
        /// were these prefixes, and one logged round was lost to a model splicing two of
        /// them together into a directory that never existed. The other half of this —
        /// the wrapper script's own path and line numbers — is the backend's, because
        /// only the backend that wrote the script knows where the model's text began in
        /// it; see <see cref="ProcessShellBackend.Rewrite"/>.
        /// </para>
        /// </summary>
        private static ConfinedResult Scrub(ConfinedResult result, SessionWorkspace workspace, string? ranIn)
        {
            string Fix(string text) =>
                text.Length == 0 ? text : OutputPaths.Scrub(text, workspace, ranIn);

            return result with { Stdout = Fix(result.Stdout), Stderr = Fix(result.Stderr) };
        }

        /// <summary>
        /// The model's own file that a failed import was probably reaching for, or null.
        ///
        /// <para>
        /// Bounded to the work directory and two levels down: a module the program imports
        /// by a bare name lives beside it or in a package directory next to it, and walking
        /// further would be walking the workspace on a path that must stay cheap.
        /// </para>
        /// </summary>
        private static string? LocalModuleFor(SessionWorkspace workspace, string module)
        {
            if (module.Length == 0 || module.IndexOfAny(new[] { '/', '\\', '.' }) >= 0)
                return null;

            try
            {
                foreach (string candidate in new[]
                {
                    module + ".py",
                    Path.Combine(module, "__init__.py"),
                })
                {
                    string direct = Path.Combine(workspace.WorkDirectory, candidate);
                    if (File.Exists(direct))
                        return candidate.Replace('\\', '/');
                }

                foreach (string sub in Directory.EnumerateDirectories(workspace.WorkDirectory))
                {
                    string name = Path.GetFileName(sub);
                    if (name.StartsWith('.') || CodeArtifactStore.IsRuntimeJunk(name))
                        continue;
                    foreach (string candidate in new[]
                    {
                        module + ".py",
                        Path.Combine(module, "__init__.py"),
                    })
                    {
                        if (File.Exists(Path.Combine(sub, candidate)))
                            return (name + "/" + candidate).Replace('\\', '/');
                    }
                }
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                // Cannot look, so cannot claim. The install goes ahead as before.
            }
            return null;
        }

        /// <summary>An example command in this host's dialect, so advice is never in the wrong shell.</summary>
        private string Example(string posix) =>
            _shell is { Kind: ShellKind.PowerShell }
                ? posix switch
                {
                    "ls -la" => "Get-ChildItem",
                    _ => posix,
                }
                : posix;

        private TimeSpan TimeoutFor(TimeSpan? requested)
        {
            if (requested is not { } asked || asked <= TimeSpan.Zero)
            {
                TimeSpan configured = _options.Timeout;
                TimeSpan ceiling = _options.EffectiveMaxTimeout;
                return configured > ceiling ? ceiling : configured;
            }
            TimeSpan max = _options.EffectiveMaxTimeout;
            return asked > max ? max : asked;
        }

        /// <summary>
        /// Perform every install the line asked for, then substitute each one out of the
        /// line so the install cannot alter the residual command's network policy.
        ///
        /// <para>
        /// Substitution rather than removal, because the operators around an install mean
        /// something: <c>pip install x &amp;&amp; python y.py</c> must not run <c>y.py</c>
        /// when the install failed. A successful install becomes <c>true</c> and a failed
        /// one becomes <c>false</c>, so <c>&amp;&amp;</c>, <c>||</c>, a pipeline and a loop
        /// body all keep exactly the meaning the model wrote.
        /// </para>
        /// </summary>
        /// <param name="command">
        /// The line to substitute into. A PARAMETER, not a field.
        ///
        /// <para>
        /// It used to be <c>_lastCommand</c>, an instance field on a runner that
        /// <see cref="CodeRunnerAdapter"/> holds ONE of and every conversation shares —
        /// while <c>_sessions</c> is a concurrent dictionary precisely because two
        /// conversations run at once. Session B could overwrite the field between session
        /// A writing it and A reading it, and A would then apply ITS install spans, which
        /// are offsets into A's command, to B's text. When B's line was shorter that threw
        /// out of the guarded region; when it was longer it did not throw at all, and A
        /// executed a mutilated splice of B's command inside A's workspace. Cross-session
        /// content, not merely corruption.
        /// </para>
        /// </param>
        private bool TryPerformInstalls(
            string command,
            IReadOnlyList<ShellInstallRequest> installs,
            SessionWorkspace workspace,
            Action<string>? onOutput,
            List<string> notes,
            out string remaining)
        {
            remaining = string.Empty;

            // Right to left, so an earlier install's span is still valid after a later
            // one has been replaced.
            var line = new StringBuilder(command);
            bool allOk = true;
            string? firstFailure = null;

            // TWO passes, and they go in OPPOSITE directions on purpose.
            //
            // Installs are performed in SOURCE order, because `pip install a && pip
            // install b` means a first and the `allOk` short-circuit only makes sense that
            // way. Running them right-to-left — which the single loop here used to do, to
            // keep the span offsets valid — meant the LAST install ran first, so a failure
            // in `b` marked `a` as "an earlier install in the same command failed" and `a`
            // was never attempted at all. That string was never surfaced either, so the
            // model read one error about `b`, concluded from `&&` that `a` had run, and was
            // wrong.
            //
            // Substitution then goes right-to-left, from the recorded outcomes, because an
            // earlier install's span is only still valid if the later ones have not moved.
            var outcomes = new List<(ShellInstallRequest Install, string? Error)>();
            foreach (ShellInstallRequest install in installs.OrderBy(i => i.Segment.Start))
            {
                bool performed = false;
                string? error = allOk
                    ? _installer.Install(
                        workspace, install.Language, install.Packages, onOutput, out performed)
                    : "an install earlier in the same command failed, so this one was not attempted";

                if (error == null)
                {
                    notes.Add(performed
                        ? install.Packages.Count > 0
                            ? "Installed: " + string.Join(", ", install.Packages)
                            : "Installed the dependencies named by the manifest."
                        : install.Packages.Count > 0
                            ? (install.Packages.All(p => _installer.IsProvided(install.Language, p))
                                ? "Built into this host, nothing to install: " + string.Join(", ", install.Packages)
                                : "Already installed this session: " + string.Join(", ", install.Packages))
                            : "The manifest named no new dependencies to install.");
                }
                else
                {
                    // Every skipped install is named. Silence here let the model believe a
                    // package it never got was present.
                    if (!allOk)
                    {
                        notes.Add(
                            (install.Packages.Count > 0
                                ? "NOT installed: " + string.Join(", ", install.Packages)
                                : "The manifest dependencies were NOT installed")
                            + " — an install earlier in this command failed, so nothing after it ran.");
                    }
                    allOk = false;
                    firstFailure ??= error;
                }

                outcomes.Add((install, error));
            }

            foreach ((ShellInstallRequest install, string? error) in
                outcomes.OrderByDescending(o => o.Install.Segment.Start))
            {
                line.Remove(install.Segment.Start, install.Segment.Length);
                line.Insert(install.Segment.Start, error == null ? SucceededNoOp : FailedNoOp);
            }

            // A failed install becomes `false` and the line RUNS. That is what the shell
            // itself would do, and the difference is visible: `pip install x && python y.py`
            // must not run y.py, while `pip install x || pip install y` must try the
            // fallback. Returning early on failure got the first right and the second
            // wrong, and silently — the model's own fallback never ran and nothing said so.
            if (!allOk)
                notes.Add(firstFailure!);

            remaining = StripNoOps(line.ToString());
            return allOk;
        }

        /// <summary>
        /// What a performed install is replaced by, in the dialect this host actually
        /// speaks.
        ///
        /// <para>
        /// It was <c>true</c> and <c>false</c> unconditionally, which are POSIX
        /// utilities and not PowerShell commands. On Windows every
        /// <c>pip install x; python y.py</c> therefore came back carrying
        /// "The term 'true' is not recognized as the name of a cmdlet" — an error about
        /// a command the MODEL never wrote, in the middle of a line that had otherwise
        /// worked, with nothing to connect it to the install the host had silently
        /// performed on its behalf.
        /// </para>
        /// <para>
        /// The replacements keep the properties the originals had: neither prints
        /// anything, both leave the exit status the wrapper reports, and the rest of the
        /// line still runs - which is the documented behaviour, so that
        /// <c>pip install x || pip install y</c> still reaches its fallback.
        /// </para>
        /// <para>
        /// Each is ONE statement with no <c>;</c> in it, and that is a requirement rather
        /// than a style: <see cref="ShellCommand.SplitSegments"/> splits on <c>;</c>, so a
        /// substitution containing one would be read as two segments - which would both
        /// defeat <see cref="StripNoOps"/> below and change how the rest of the model's
        /// line parses. An earlier draft used a <c>&amp; { ...; Write-Error ... }</c>
        /// block for exactly the reason given below, and had to be given up for this.
        /// </para>
        /// <para>
        /// The one thing not carried over: on PowerShell 7, <c>&amp;&amp;</c> and
        /// <c>||</c> branch on <c>$?</c>, and an assignment always succeeds, so a failed
        /// install no longer short-circuits a <c>&amp;&amp;</c> chain there. It cannot be
        /// fixed with a statement that has no <c>;</c>, it does not arise on Windows
        /// PowerShell 5.1 (which has no <c>&amp;&amp;</c> at all), and the model is told
        /// what did not install either way - the install notes name every package that
        /// failed and why, ahead of the command's own output.
        /// </para>
        /// </summary>
        private string SucceededNoOp =>
            _shell is { Kind: ShellKind.PowerShell } ? "$global:LASTEXITCODE = 0" : "true";

        private string FailedNoOp =>
            _shell is { Kind: ShellKind.PowerShell } ? "$global:LASTEXITCODE = 1" : "false";

        /// <summary>
        /// A line that is nothing but the no-ops left behind by substituted installs has
        /// nothing left to run.
        /// </summary>
        private string StripNoOps(string line)
        {
            string succeeded = SucceededNoOp;
            string failed = FailedNoOp;
            foreach (ShellSegment segment in ShellCommand.SplitSegments(line))
            {
                if (segment.Text != succeeded && segment.Text != failed)
                    return line;
            }
            return string.Empty;
        }

        /// <summary>
        /// Reads a file the model named, confined to the workspace — for a
        /// <c>-r requirements.txt</c>, whose contents become the package list the host
        /// installs. Null when it is not there or not readable, which the caller reports.
        /// </summary>
        private static Func<string, string?> RelativeReader(SessionWorkspace workspace) =>
            relative => workspace.TryReadFile(relative, out string content, out _) ? content : null;

        private bool TryResolveWorkDirectory(
            string? requested, SessionWorkspace workspace, out string? resolved, out string? error)
        {
            resolved = null;
            error = null;
            if (string.IsNullOrWhiteSpace(requested) || requested!.Trim() is "." or "./")
                return true;

            if (!workspace.TryResolve(requested, out string full, out error))
                return false;
            if (!Directory.Exists(full))
            {
                error = $"'{requested}' is not a directory in this conversation's working directory."
                      + workspace.DescribeWhatIsHere();
                return false;
            }
            resolved = full;
            return true;
        }

        /// <summary>
        /// Everything outside the workspace the command may read — and, just as
        /// importantly, EXECUTE.
        ///
        /// <para>
        /// The distinction cost a day once: an installed console script (pytest,
        /// markitdown, anything under <c>node_modules/.bin</c>) put on PATH but not on
        /// this list exits 127 with nothing in the output to explain it, because the
        /// sandbox must permit the binary to be executed and not merely named.
        /// </para>
        /// </summary>
        private static IReadOnlyList<string> ReadablePathsFor(
            SessionWorkspace workspace,
            IReadOnlyList<string>? extra,
            IReadOnlyList<string>? networkPaths = null)
        {
            var paths = new List<string> { workspace.EnvDirectory };
            if (extra != null)
                paths.AddRange(extra.Where(p => !string.IsNullOrWhiteSpace(p)));
            if (networkPaths != null)
                paths.AddRange(networkPaths.Where(p => !string.IsNullOrWhiteSpace(p)));
            return paths;
        }

        /// <summary>
        /// Make <c>python</c> mean this host's Python, once per workspace.
        ///
        /// <para>
        /// <b>9 rounds and 5 incidents in this server's measured logs</b>, all identical:
        /// <c>exit 127 … python: command not found</c>. The only Python on the machine is
        /// <c>python3</c> — on a Mac it may be the one inside Xcode — and models type
        /// <c>python</c>. The shell tool's description already names the interpreters this
        /// host has, with versions, and it did not help: a model that has written
        /// <c>python x.py</c> ten thousand times in training writes it here too.
        /// </para>
        /// <para>
        /// So the host makes the model right instead of telling it it is wrong. A one-line
        /// exec shim goes in the session's <c>env/bin</c>, which
        /// <see cref="BuildEnvironment"/> already puts at the FRONT of PATH and which is
        /// already on the sandbox's executable list. This is the transferable core of what
        /// Claude Code gets from its shell snapshots: the difference between "the model
        /// must learn this host" and "this host answers what the model says".
        /// </para>
        /// <para>
        /// <b>Only when a real interpreter resolves.</b> A shim over nothing would turn an
        /// honest <c>command not found</c> into a broken script, which is strictly worse:
        /// the model would then be debugging its own code instead of reading a fact about
        /// the host. And only <c>python</c> — never <c>pip</c>, because installs are read
        /// out of the command line and performed by the host before it runs, so a
        /// <c>pip</c> shim would let a line reach the sandbox that
        /// <see cref="ShellInstall"/> had deliberately refused.
        /// </para>
        /// </summary>
        private static void EnsureInterpreterAliases(SessionWorkspace workspace)
        {
            string bin = Path.Combine(workspace.EnvDirectory, "shim");

            foreach ((string typed, string? real) in InterpreterAliases())
            {
                if (real == null)
                    continue;
                string alias = Path.Combine(bin, OperatingSystem.IsWindows() ? typed + ".cmd" : typed);
                try
                {
                    if (File.Exists(alias))
                        continue;
                    Directory.CreateDirectory(bin);
                    if (OperatingSystem.IsWindows())
                    {
                        File.WriteAllText(alias, $"@echo off\r\n\"{real}\" %*\r\n");
                        continue;
                    }
                    File.WriteAllText(alias, $"#!/bin/sh\nexec \"{real}\" \"$@\"\n");
                    File.SetUnixFileMode(
                        alias,
                        UnixFileMode.UserRead | UnixFileMode.UserWrite | UnixFileMode.UserExecute
                        | UnixFileMode.GroupRead | UnixFileMode.GroupExecute
                        | UnixFileMode.OtherRead | UnixFileMode.OtherExecute);
                }
                catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
                {
                    // A shim that could not be written leaves the honest "command not
                    // found", which is where this started and is never a wrong answer.
                }
            }
        }

        /// <summary>
        /// Make <c>import</c> see the session's installed packages, not only
        /// <c>require</c>.
        ///
        /// <para>
        /// <b>A measured hole in the environment, not a theory.</b> The host points
        /// <c>NODE_PATH</c> at <c>env/node_modules</c>, and <c>NODE_PATH</c> is consulted
        /// by CommonJS resolution ONLY. Verified on Node v26.8.1 in a directory with no
        /// <c>node_modules</c> above it:
        /// </para>
        /// <code>
        /// NODE_PATH=…/env/node_modules  node a.cjs   ->  require: function
        /// NODE_PATH=…/env/node_modules  node b.mjs   ->  Error [ERR_MODULE_NOT_FOUND]:
        ///                                                Cannot find package 'jszip'
        /// </code>
        /// <para>
        /// So a program written with <c>import</c> — the syntax a current model writes by
        /// default — could not see a package the host had just installed for it, and the
        /// error it got back says "Cannot find package", which is indistinguishable from
        /// not being installed. It has not fired in production yet only because the old
        /// tool always wrote <c>.js</c>; <c>.mjs</c> already appears a dozen times in the
        /// logs, and the shell lets the model name its own files.
        /// </para>
        /// <para>
        /// The fix is the mechanism Node itself uses: a <c>node_modules</c> in the
        /// directory the program is in. Ordinary upward resolution then finds it, and BOTH
        /// forms work — verified, same probe:
        /// </para>
        /// <code>
        /// ln -s …/env/node_modules work/node_modules
        /// node b.mjs  ->  import: function        node a.cjs  ->  require: function
        /// </code>
        /// <para>
        /// A link rather than a copy, and safe to put in the work directory because both
        /// walkers already ignore it: <c>WorkspaceScan</c> prunes <c>node_modules</c> by
        /// name during enumeration, and artifact capture skips reparse points outright — so
        /// it is neither offered to the user as output nor counted as a produced file.
        /// </para>
        /// </summary>
        private static void EnsureNodeResolution(SessionWorkspace workspace)
        {
            string installed = Path.Combine(workspace.EnvDirectory, "node_modules");
            if (!Directory.Exists(installed))
                return;                                  // nothing installed yet

            string link = Path.Combine(workspace.WorkDirectory, "node_modules");
            try
            {
                // Never clobber: if something is already there — the model's own directory,
                // or a link from an earlier call — leave it alone.
                if (Directory.Exists(link) || File.Exists(link))
                    return;
                Directory.CreateSymbolicLink(link, installed);
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException
                                          or PlatformNotSupportedException)
            {
                // Windows needs a privilege for this, and without it the CommonJS path
                // still works exactly as before. A missing link is the status quo, never a
                // wrong answer — so it is not worth refusing a command over.
            }
        }

        /// <summary>
        /// The names a model types, paired with what this host should actually run for
        /// them — or null where nothing should be shimmed.
        ///
        /// <para>
        /// Two distinct cases, and the second is the one that cost real rounds.
        /// </para>
        /// <para>
        /// <c>python</c> may not exist at all. Five incidents and nine rounds of
        /// <c>exit 127 … python: command not found</c>, on a host whose shell description
        /// already names <c>python3</c> with its version — a model that has written
        /// <c>python x.py</c> ten thousand times in training writes it here too.
        /// </para>
        /// <para>
        /// <c>python3</c> may exist and be the WRONG ONE. Interpreter resolution prefers
        /// 3.14 down to 3.10 before a bare <c>python3</c>, so the host knows about the
        /// newer Python — but a command typing <c>python3</c> gets whatever PATH finds
        /// first, which on a Mac is Apple's frozen 3.9. That is not hypothetical: the
        /// bundled pptx skill's own <c>scripts/office/validate.py</c> fails to parse there,
        /// on a <c>match</c> statement, and the failure reads as a broken script rather
        /// than as an old interpreter. Shimmed only when the resolved interpreter is
        /// genuinely NEWER, so a host with one Python gets no pointless wrapper.
        /// </para>
        /// <para>
        /// Deliberately NOT <c>pip</c>: installs are read out of the command line and
        /// performed by the host before anything runs, so a <c>pip</c> shim would let a
        /// line reach the sandbox that ShellInstall had refused.
        /// </para>
        /// </summary>
        private static IEnumerable<(string Typed, string? Real)> InterpreterAliases()
        {
            // A host that described its own embedded runtime maps `python` and `python3`
            // itself; a shim over a name that is not a file on PATH would be a script
            // that execs nothing.
            if (CodeEnvironment.IsConfigured)
                yield break;

            if (!CodeEnvironment.TryResolveInterpreter(CodeLanguage.Python, out string? python, out _)
                || string.IsNullOrEmpty(python))
            {
                yield break;
            }

            yield return ("python", python);

            string? onPath = CodeEnvironment.Which("python3");
            yield return ("python3",
                onPath == null || IsNewer(python!, onPath) ? python : null);
        }

        /// <summary>
        /// True when <paramref name="candidate"/> is a strictly newer Python than
        /// <paramref name="incumbent"/>. Unknown versions answer false — shimming on a
        /// guess would replace an interpreter the model asked for by name.
        /// </summary>
        private static bool IsNewer(string candidate, string incumbent) =>
            CodeEnvironment.PythonVersionOf(candidate) is { } a
            && CodeEnvironment.PythonVersionOf(incumbent) is { } b
            && a > b;

        /// <summary>
        /// Host network settings that are useful without being general credential
        /// inheritance. They are copied only for the explicit network opt-in: many
        /// enterprise/CI hosts have no direct route and require an HTTP/SOCKS proxy or a
        /// custom trust bundle. Everything else still starts from ConfinedProcess's blank
        /// environment, especially cloud/API credentials.
        /// </summary>
        private static readonly string[] ProxyEnvironmentVariables =
        {
            "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
            "http_proxy", "https_proxy", "all_proxy", "no_proxy",
            "GRPC_PROXY", "grpc_proxy",
        };

        private static readonly string[] CertificateFileEnvironmentVariables =
        {
            "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "NODE_EXTRA_CA_CERTS",
            "GIT_SSL_CAINFO", "AWS_CA_BUNDLE", "PIP_CERT", "NPM_CONFIG_CAFILE",
            "CARGO_HTTP_CAINFO", "DENO_CERT", "CLOUDSDK_CORE_CUSTOM_CA_CERTS_FILE",
            "NIX_SSL_CERT_FILE", "GRPC_DEFAULT_SSL_ROOTS_FILE_PATH",
        };

        private const long MaxCertificateBundleBytes = 16L * 1024 * 1024;

        private Dictionary<string, string> BuildEnvironment(
            SessionWorkspace workspace, out IReadOnlyList<string> networkReadablePaths)
        {
            var readablePaths = new List<string>();
            var environment = new Dictionary<string, string>(StringComparer.Ordinal)
            {
                // ConfinedProcess points HOME and TMPDIR at the whole writable region,
                // which for a shell is the workspace root — so `ls ~` would show the
                // model the host's own bookkeeping and `~` would not mean "my files".
                // Both are narrowed here, where the layout is known.
                ["HOME"] = workspace.WorkDirectory,
                ["USERPROFILE"] = workspace.WorkDirectory,
                ["TMPDIR"] = workspace.TempDirectory,
                ["TEMP"] = workspace.TempDirectory,
                ["TMP"] = workspace.TempDirectory,
                // The session's package environment, reachable by whatever the model runs
                // without it having to know where the host put it.
                ["PYTHONPATH"] = workspace.EnvDirectory,
                ["NODE_PATH"] = Path.Combine(workspace.EnvDirectory, "node_modules"),
                // Or CPython writes .pyc files next to whatever it imported, which under
                // the sandbox fails in a way that reads as the script's own error.
                ["PYTHONDONTWRITEBYTECODE"] = "1",
                // Output the user is watching must not sit in a pipe buffer until exit.
                ["PYTHONUNBUFFERED"] = "1",
                ["PIP_DISABLE_PIP_VERSION_CHECK"] = "1",
                // pip's default target is the interpreter's own site-packages, which is
                // outside the workspace and not writable. Point it at the session's.
                ["PIP_TARGET"] = workspace.EnvDirectory,
                ["PIP_NO_INPUT"] = "1",
                // Never build from source: a source package runs its own setup script at
                // install time, which is arbitrary code from a name the model chose.
                ["PIP_ONLY_BINARY"] = ":all:",
                ["NPM_CONFIG_PREFIX"] = workspace.EnvDirectory,
                ["NPM_CONFIG_IGNORE_SCRIPTS"] = "true",
                ["NPM_CONFIG_UPDATE_NOTIFIER"] = "false",
                ["NPM_CONFIG_FUND"] = "false",
            };

            // PATH is set by ConfinedProcess from the host's own, then overwritten here —
            // the launch plan's variables are applied last, so this wins. The prefixed
            // directories are on the readable list above for the same reason.
            string hostPath = Environment.GetEnvironmentVariable("PATH") ?? string.Empty;
            string separator = OperatingSystem.IsWindows() ? ";" : ":";
            environment["PATH"] = string.Join(separator, new[]
            {
                // Ahead of everything: the host's own interpreter aliases, which must not
                // be shadowed by a package that happens to install a script of the same
                // name. See EnsureInterpreterAliases.
                Path.Combine(workspace.EnvDirectory, "shim"),
                CodeEnvironment.VenvBin(workspace.EnvDirectory),
                Path.Combine(workspace.EnvDirectory, "bin"),
                Path.Combine(workspace.EnvDirectory, "node_modules", ".bin"),
                hostPath,
            }.Where(p => p.Length > 0));

            if (_options.AllowNetwork)
                AddNetworkEnvironment(workspace, environment, readablePaths);

            networkReadablePaths = readablePaths;
            return environment;
        }

        private void AddNetworkEnvironment(
            SessionWorkspace workspace,
            Dictionary<string, string> environment,
            List<string> readablePaths)
        {
            foreach (string name in ProxyEnvironmentVariables)
            {
                if (Environment.GetEnvironmentVariable(name) is not { Length: > 0 } value)
                    continue;

                // Proxy URLs frequently carry user:password@host. The generated command
                // gets network access, not host credentials. Claude's proxy architecture
                // similarly exposes only an ephemeral local credential; until TensorSharp
                // has that forwarding layer, credential-bearing endpoints fail closed.
                if (ProxyValueHasNoCredentials(name, value))
                    environment[name] = value;
            }

            // A host CA path is never handed to generated code directly. Read it once,
            // extract and validate only X.509 certificate blocks, and write their
            // canonical public PEM form to random host-authored session state. That
            // closes three otherwise subtle holes at once: arbitrary text/private-key
            // material adjacent to a certificate is not disclosed, a writable source
            // cannot be swapped after validation but before sandbox launch, and a large
            // system bundle is not reparsed on every shell call. State is read-only to
            // the command (apart from the separate state/shell child), so the immutable
            // snapshot cannot be redirected by a model-created symlink.
            foreach (string name in CertificateFileEnvironmentVariables)
            {
                if (Environment.GetEnvironmentVariable(name) is not { Length: > 0 } source)
                    continue;
                try
                {
                    string full = Path.GetFullPath(source);
                    string? sanitized = SanitizedCertificateBundle(workspace, full);
                    if (sanitized == null)
                        continue;
                    environment[name] = sanitized;
                    if (!readablePaths.Contains(sanitized, StringComparer.Ordinal))
                        readablePaths.Add(sanitized);
                }
                catch (Exception ex) when (ex is IOException or UnauthorizedAccessException
                                               or ArgumentException or NotSupportedException)
                {
                    // A malformed, inaccessible or oversized host setting is omitted. A
                    // generated command must never inherit a path it cannot actually use.
                }
            }
        }

        private string? SanitizedCertificateBundle(SessionWorkspace workspace, string source)
        {
            ConcurrentDictionary<string, Lazy<string?>> bundles =
                _sanitizedCertificateBundles.GetOrAdd(
                    workspace.Root,
                    _ => new ConcurrentDictionary<string, Lazy<string?>>(StringComparer.Ordinal));
            return bundles.GetOrAdd(
                source,
                _ => new Lazy<string?>(
                    () => CreateSanitizedCertificateBundle(workspace, source),
                    System.Threading.LazyThreadSafetyMode.ExecutionAndPublication)).Value;
        }

        private static string? CreateSanitizedCertificateBundle(
            SessionWorkspace workspace, string source)
        {
            string? canonical = ReadAndNormalizeCertificateBundle(source);
            if (canonical == null)
                return null;

            for (int attempt = 0; attempt < 8; attempt++)
            {
                string destination = Path.Combine(
                    workspace.StateDirectory,
                    "ca-" + Path.GetRandomFileName() + ".pem");
                FileStream stream;
                try
                {
                    // CreateNew plus an unpredictable leaf is important even though the
                    // containing state directory is sandbox-read-only: it also avoids
                    // following stale files from a crashed or deliberately unconfined
                    // session.
                    stream = new FileStream(
                        destination, FileMode.CreateNew, FileAccess.Write, FileShare.Read,
                        bufferSize: 4096, FileOptions.WriteThrough);
                }
                catch (IOException) when (File.Exists(destination))
                {
                    // Vanishingly unlikely random collision. Choose a new leaf; never
                    // overwrite or follow what was already there.
                    continue;
                }

                using (stream)
                using (var writer = new StreamWriter(
                    stream, new UTF8Encoding(encoderShouldEmitUTF8Identifier: false)))
                {
                    writer.Write(canonical);
                }
                if (!OperatingSystem.IsWindows())
                {
                    try
                    {
                        File.SetUnixFileMode(
                            destination, UnixFileMode.UserRead | UnixFileMode.UserWrite);
                    }
                    catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { }
                }
                return destination;
            }

            return null;
        }

        private static string? ReadAndNormalizeCertificateBundle(string path)
        {
            try
            {
                // Hold one handle for length check and read. A rename then leaves this
                // handle on the original inode; an in-place race still has to produce a
                // valid public certificate before any byte is copied onward.
                using var stream = new FileStream(
                    path, FileMode.Open, FileAccess.Read, FileShare.Read,
                    bufferSize: 4096, FileOptions.SequentialScan);
                long length = stream.Length;
                if (length <= 0 || length > MaxCertificateBundleBytes)
                    return null;

                // One extra byte detects a source that grows after the length check.
                var bytes = new byte[checked((int)length) + 1];
                int read = 0;
                while (read < bytes.Length)
                {
                    int count = stream.Read(bytes, read, bytes.Length - read);
                    if (count == 0)
                        break;
                    read += count;
                }
                if (read != length)
                    return null;

                string text = new UTF8Encoding(
                    encoderShouldEmitUTF8Identifier: false,
                    throwOnInvalidBytes: true).GetString(bytes, 0, read);
                System.Text.RegularExpressions.MatchCollection certificates =
                    System.Text.RegularExpressions.Regex.Matches(
                        text,
                        @"-----BEGIN (?<label>CERTIFICATE|TRUSTED CERTIFICATE)-----\s*"
                        + @"(?<body>[A-Za-z0-9+/=\s]+?)\s*"
                        + @"-----END (?<end>CERTIFICATE|TRUSTED CERTIFICATE)-----",
                        System.Text.RegularExpressions.RegexOptions.CultureInvariant
                        | System.Text.RegularExpressions.RegexOptions.NonBacktracking);
                if (certificates.Count == 0)
                    return null;

                var normalized = new StringBuilder();
                foreach (System.Text.RegularExpressions.Match match in certificates)
                {
                    if (!string.Equals(
                        match.Groups["label"].Value,
                        match.Groups["end"].Value,
                        StringComparison.Ordinal))
                    {
                        continue;
                    }
                    byte[] der = Convert.FromBase64String(
                        System.Text.RegularExpressions.Regex.Replace(
                            match.Groups["body"].Value, @"\s+", string.Empty));
                    using System.Security.Cryptography.X509Certificates.X509Certificate2 certificate =
                        System.Security.Cryptography.X509Certificates.X509CertificateLoader
                            .LoadCertificate(der);
                    normalized.Append(certificate.ExportCertificatePem()).Append('\n');
                }
                return normalized.Length == 0 ? null : normalized.ToString();
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException
                                           or DecoderFallbackException or FormatException
                                           or System.Security.Cryptography.CryptographicException)
            {
                return null;
            }
        }

        internal static bool ProxyValueHasNoCredentials(string name, string value)
        {
            if (value.Length == 0 || value.Any(char.IsControl)
                || !string.Equals(value, value.Trim(), StringComparison.Ordinal))
                return false;
            if (name.Equals("NO_PROXY", StringComparison.OrdinalIgnoreCase))
                return true;
            if (value.Contains('@') || value.Contains('?') || value.Contains('#'))
                return false;

            // Some tools accept the conventional proxy.example:8080 spelling. Feeding
            // that directly to Uri.TryCreate is ambiguous (`localhost:` can be parsed as
            // a URI scheme), so validate the host/IPv4/bracketed-IPv6 + numeric-port form
            // explicitly before handling normal scheme:// URLs.
            if (!value.Contains("://", StringComparison.Ordinal))
            {
                System.Text.RegularExpressions.Match bare =
                    System.Text.RegularExpressions.Regex.Match(
                        value, @"^(?:\[[0-9A-Fa-f:.%]+\]|[A-Za-z0-9._-]+):(\d{1,5})$");
                return bare.Success
                    && int.TryParse(bare.Groups[1].Value, NumberStyles.None,
                        CultureInfo.InvariantCulture, out int port)
                    && port is > 0 and <= 65535;
            }

            if (!Uri.TryCreate(value, UriKind.Absolute, out Uri? uri)
                || uri.Host.Length == 0
                || uri.Scheme is not ("http" or "https" or "socks4" or "socks4a" or "socks5" or "socks5h"))
            {
                return false;
            }
            return string.IsNullOrEmpty(uri.UserInfo)
                && string.IsNullOrEmpty(uri.Query)
                && string.IsNullOrEmpty(uri.Fragment)
                && (uri.AbsolutePath.Length == 0 || uri.AbsolutePath == "/");
        }

        // ---- background jobs ------------------------------------------------

        /// <summary>
        /// The background jobs of one session, killed when the session ends.
        ///
        /// <para>
        /// Registered on the workspace rather than held here, because the workspace is
        /// what gets deleted — a job still writing into a directory being removed is how
        /// a "released" session leaves half a tree behind.
        /// </para>
        /// </summary>
        /// <summary>
        /// A background job's log file, written from the process's TWO reader threads.
        ///
        /// <para>
        /// The lock is not optional: stdout and stderr are drained by separate threads and
        /// a bare StreamWriter is not thread-safe, so a job that writes to both at once —
        /// which is every server and every build — interleaved bytes mid-line and
        /// occasionally threw. The model then reads a log with corrupted lines and
        /// concludes its program is broken.
        /// </para>
        /// </summary>
        private sealed class JobLog : IDisposable
        {
            private readonly StreamWriter _writer;
            private readonly object _gate = new();
            private bool _closed;

            public JobLog(string path) =>
                _writer = new StreamWriter(path, append: false) { AutoFlush = true };

            public void WriteLine(string line)
            {
                lock (_gate)
                {
                    if (_closed)
                        return;
                    try { _writer.WriteLine(line); }
                    catch (Exception ex) when (ex is IOException or ObjectDisposedException) { }
                }
            }

            public void Dispose()
            {
                lock (_gate)
                {
                    if (_closed)
                        return;
                    _closed = true;
                    try { _writer.Dispose(); }
                    catch (Exception ex) when (ex is IOException or ObjectDisposedException) { }
                }
            }
        }

        private sealed class BackgroundJobs : IDisposable
        {
            private readonly ConcurrentDictionary<string, (IShellJob Job, JobLog Log)> _running =
                new(StringComparer.Ordinal);
            private int _next;

            /// <summary>
            /// The next id. Handed out BEFORE the process starts, because the log file is
            /// named after it — computing the id twice let the name in the result and the
            /// name in the table disagree.
            /// </summary>
            public string NextId() =>
                "job-" + System.Threading.Interlocked.Increment(ref _next)
                    .ToString(CultureInfo.InvariantCulture);

            public void Add(string id, IShellJob job, JobLog log) => _running[id] = (job, log);

            public void Dispose()
            {
                foreach ((IShellJob job, JobLog log) in _running.Values)
                {
                    // The process first, then its log: closing the file while the reader
                    // threads are still delivering lines is what the writer's own lock
                    // guards, but killing first makes the window vanishingly small.
                    try { job.Dispose(); }
                    catch (Exception ex) when (ex is not (OutOfMemoryException or StackOverflowException)) { }
                    try { log.Dispose(); }
                    catch (Exception ex) when (ex is not (OutOfMemoryException or StackOverflowException)) { }
                }
                _running.Clear();
            }
        }

        private BackgroundJobs JobsFor(SessionWorkspace workspace) =>
            _jobs.GetOrAdd(workspace.Root, _ =>
            {
                var jobs = new BackgroundJobs();
                workspace.RegisterCleanup(jobs);
                return jobs;
            });

        private CodeExecResult StartBackground(
            SessionWorkspace workspace, ShellLaunch launch)
        {
            string logDirectory = Path.Combine(workspace.StateDirectory, ".jobs");
            Directory.CreateDirectory(logDirectory);

            BackgroundJobs jobs = JobsFor(workspace);

            // The host writes the log, not the shell. A `tee` in the wrapper would need a
            // tool that may not exist and a process substitution that plain `sh` does not
            // have, and PowerShell's transcript is a different shape again; the host
            // already has every line on its own tap.
            string id = jobs.NextId();
            string logPath = Path.Combine(logDirectory, id + ".log");
            JobLog log;
            try
            {
                log = new JobLog(logPath);
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                return CodeExecResult.Refused($"the job's log file could not be created: {ex.Message}");
            }

            Action<string>? tap = launch.OnOutputLine;
            ShellLaunch teed = launch with
            {
                // A background job has no deadline of its own: it ends when it ends, when
                // something kills it, or when the session does.
                Timeout = System.Threading.Timeout.InfiniteTimeSpan,
                MaxOutputBytes = _options.MaxOutputBytes,
                OnOutputLine = line =>
                {
                    log.WriteLine(line);
                    tap?.Invoke(line);
                },
            };

            if (!_backend.TryStart(teed, out IShellJob? job, out ConfinedResult failure))
            {
                log.Dispose();
                return CodeExecResult.Refused(failure.Error ?? "the command could not be started");
            }

            // The log is closed when the JOB is, not when this call returns: the process
            // outlives the call and keeps writing. Nothing closed it before, which on
            // Windows left the file open and made the workspace's recursive delete fail.
            jobs.Add(id, job!, log);
            _logger.LogInformation(LogEventIds.CodeExecBackgroundJob,
                "codeexec.job.started id={Id} pid={Pid} sandbox={Sandbox}",
                id, job!.ProcessId, job.SandboxName);

            // Relative to where the NEXT command will start, not to the work directory.
            // After a `cd src`, the suggested `tail -n 40 .jobs/job-1.log` resolved against
            // src/ and failed with "No such file" — the result's own instruction, wrong.
            string logFrom = launch.WorkingDirectory is { Length: > 0 } started
                ? started
                : workspace.WorkDirectory;
            string relativeLog = Path.GetRelativePath(logFrom, logPath).Replace('\\', '/');
            bool posix = _shell is { Kind: ShellKind.Posix };
            var sb = new StringBuilder();
            sb.Append("Started in the background as ").Append(id).Append(".\n");
            sb.Append("Its output is being written to ").Append(relativeLog)
              .Append(" — read it when you want to check on it, for example: ")
              .Append(posix ? $"tail -n 40 {relativeLog}" : $"Get-Content -Tail 40 {relativeLog}")
              .Append('\n');
            // Deliberately NOT "kill <pid>": each command runs in its own sandbox and cannot
            // signal a process started by another one, so telling the model to kill it would
            // send it round a loop it cannot win.
            sb.Append("It runs until it finishes or until this conversation ends; you cannot stop it "
                    + "from another command, so do not start something endless unless you mean to.\n");
            return new CodeExecResult(true, sb.ToString(), Array.Empty<CodeArtifact>(), string.Empty);
        }

        // ---- patching -------------------------------------------------------

        /// <summary>
        /// Apply a patch envelope to the session's files, all of it or none of it.
        ///
        /// <para>
        /// The order is the whole point: parse, then resolve EVERY hunk against the files
        /// as they stand, then check every destination path, then remember what each file
        /// held, and only then write. A failure at any earlier step has changed nothing; a
        /// failure while writing is rolled back. A patch half-applied across three files
        /// leaves a workspace neither the model nor the user can reason about, and the
        /// model's next move is to regenerate everything.
        /// </para>
        /// </summary>
        public CodeExecResult ApplyPatch(string? patch, SessionWorkspace? workspace)
        {
            if (workspace == null)
            {
                return CodeExecResult.Refused(
                    "there is no working directory in this conversation to patch.");
            }

            // See ShellRunner.RunIn: the layout is re-asserted, never assumed. A patch is
            // the one tool that can rebuild what a wiped workspace lost, so it must not be
            // the tool that fails because the workspace was wiped.
            workspace.TryEnsureDirectories();

            if (!CodePatch.TryParse(patch, out IReadOnlyList<CodePatch.FileSection> sections, out string? parseError))
                return CodeExecResult.Refused(parseError!);

            // Relative to where the SHELL is, not to the work directory: the two halves
            // of this tool surface have to agree about what "main.c" means after a `cd`.
            string from = _shell != null ? SessionFor(workspace).CurrentDirectory : workspace.WorkDirectory;

            // What each file will hold once the whole envelope has been resolved. Null
            // means "deleted". Every section reads from HERE first and only then from
            // disk, which is what makes an envelope internally consistent: two sections
            // naming the same file compose instead of the second silently discarding the
            // first, and a Delete followed by an Add of the same path is a rewrite rather
            // than a refusal. Resolving every section against the pre-patch disk state —
            // which is what this did — reported both edits as applied while keeping only
            // the last.
            var pending = new Dictionary<string, string?>(StringComparer.Ordinal);
            var order = new List<string>();
            var outcomes = new List<CodePatch.FileOutcome>();

            bool Exists(string path) =>
                pending.TryGetValue(path, out string? staged) ? staged != null : File.Exists(path);

            bool TryCurrent(string path, out string content, out string? error)
            {
                error = null;
                if (pending.TryGetValue(path, out string? staged))
                {
                    content = staged ?? string.Empty;
                    return staged != null;
                }
                try
                {
                    content = File.ReadAllText(path);
                    return true;
                }
                catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
                {
                    content = string.Empty;
                    error = ex.Message;
                    return false;
                }
            }

            void Stage(string path, string? content)
            {
                if (!pending.ContainsKey(path))
                    order.Add(path);
                pending[path] = content;
            }

            foreach (CodePatch.FileSection section in sections)
            {
                if (!workspace.TryResolveFrom(from, section.Path, out string full, out string? pathError))
                    return CodeExecResult.Refused(pathError!);

                string? moveFull = null;
                if (section.MoveTo != null
                    && !workspace.TryResolveFrom(from, section.MoveTo, out moveFull!, out pathError))
                {
                    return CodeExecResult.Refused(pathError!);
                }

                switch (section.Op)
                {
                    case CodePatch.FileOp.Add:
                        {
                            if (Exists(full))
                            {
                                return CodeExecResult.Refused(
                                    $"'{section.Path}' already exists, so it cannot be added. "
                                    + "Use '*** Update File:' to change it, or delete it first.");
                            }
                            V4ADiff.DiffResult created = V4ADiff.Create(section.Body, section.Newline);
                            if (!created.Ok)
                                return CodeExecResult.Refused($"in '{section.Path}': {created.Error}");
                            Stage(full, created.Text);
                            outcomes.Add(new CodePatch.FileOutcome(
                                section.Op, section.Path, null, created.LinesAdded, 0, 0));
                            break;
                        }

                    case CodePatch.FileOp.Delete:
                        {
                            if (!Exists(full))
                            {
                                return CodeExecResult.Refused(
                                    $"'{section.Path}' does not exist, so it cannot be deleted."
                                    + workspace.DescribeWhatIsHere());
                            }
                            Stage(full, null);
                            outcomes.Add(new CodePatch.FileOutcome(section.Op, section.Path, null, 0, 0, 0));
                            break;
                        }

                    default:
                        {
                            if (!Exists(full))
                            {
                                return CodeExecResult.Refused(
                                    $"'{section.Path}' does not exist, so it cannot be updated. "
                                    + "Use '*** Add File:' to create it."
                                    + workspace.DescribeWhatIsHere());
                            }
                            if (!TryCurrent(full, out string current, out string? readError))
                                return CodeExecResult.Refused($"'{section.Path}' could not be read: {readError}");

                            string updated = current;
                            int added = 0, removed = 0, fuzz = 0;
                            string? note = null;
                            if (section.Body.Count > 0)
                            {
                                V4ADiff.DiffResult result = V4ADiff.Update(
                                    current, section.Body, section.Newline, section.Path);
                                if (!result.Ok)
                                    return CodeExecResult.Refused($"in '{section.Path}': {result.Error}");
                                updated = result.Text;
                                added = result.LinesAdded;
                                removed = result.LinesRemoved;
                                fuzz = result.Fuzz;
                                note = result.Note;
                            }

                            // A '*** Move to:' naming the file's OWN path is a no-op rename,
                            // not a delete. Staging the new content and then unstaging the
                            // old one under the same key destroyed the file outright — and
                            // reported "updated a.txt -> a.txt" while doing it.
                            bool renamed = moveFull != null
                                && !PathsEqual(moveFull!, full);
                            Stage(renamed ? moveFull! : full, updated);
                            if (renamed)
                                Stage(full, null);

                            outcomes.Add(new CodePatch.FileOutcome(
                                section.Op, section.Path, renamed ? section.MoveTo : null, added, removed, fuzz)
                            { Note = note });
                            break;
                        }
                }
            }

            // Everything resolved. Remember what is there so a mid-commit I/O failure can
            // be undone, then write.
            var undo = new List<(string Path, byte[]? Content, UnixFileMode? Mode)>();
            try
            {
                foreach (string path in order)
                {
                    bool existed = File.Exists(path);
                    undo.Add((
                        path,
                        existed ? File.ReadAllBytes(path) : null,
                        existed && !OperatingSystem.IsWindows() ? File.GetUnixFileMode(path) : null));
                }

                foreach (string path in order)
                {
                    string? content = pending[path];
                    if (content == null)
                    {
                        if (File.Exists(path))
                            File.Delete(path);
                        continue;
                    }

                    // Write back in the encoding the file already had. Round-tripping
                    // through File.ReadAllText/WriteAllText silently strips a UTF-8 BOM and
                    // rewrites a UTF-16 file as UTF-8 — a whole-file change the model never
                    // asked for, and one that breaks anything reading it by byte.
                    string? parent = Path.GetDirectoryName(path);
                    if (!string.IsNullOrEmpty(parent))
                        Directory.CreateDirectory(parent);
                    Encoding encoding = EncodingOf(path);
                    File.WriteAllBytes(path, Concat(encoding.GetPreamble(), encoding.GetBytes(content)));
                }

                // A rename keeps the file's mode: a script that was executable before the
                // patch has to still be executable after it, or the next command that runs
                // it fails for a reason the model cannot see in its own diff.
                foreach (CodePatch.FileOutcome outcome in outcomes)
                {
                    if (outcome.MovedTo == null || OperatingSystem.IsWindows())
                        continue;
                    (string _, byte[]? _, UnixFileMode? mode) = undo.FirstOrDefault(
                        u => u.Mode != null && PathsEqual(u.Path, ResolveOr(workspace, from, outcome.Path)));
                    if (mode is { } m && workspace.TryResolveFrom(from, outcome.MovedTo, out string dest, out _))
                    {
                        try { File.SetUnixFileMode(dest, m); }
                        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { }
                    }
                }
            }
            // Widened from IOException/UnauthorizedAccessException to everything
            // recoverable: CreateDirectory and Delete can also throw NotSupportedException
            // or ArgumentException for a path the model chose, and those escaped to the
            // tool dispatcher's catch-all with NO rollback at all — while the patch tool's
            // declaration promises "nothing is written at all".
            catch (Exception ex) when (ex is not OutOfMemoryException and not StackOverflowException)
            {
                var lost = new List<string>();
                foreach ((string path, byte[]? content, UnixFileMode? mode) in Enumerable.Reverse(undo))
                {
                    try
                    {
                        if (content == null)
                        {
                            if (File.Exists(path)) File.Delete(path);
                        }
                        else
                        {
                            File.WriteAllBytes(path, content);
                            if (mode is { } m && !OperatingSystem.IsWindows())
                                File.SetUnixFileMode(path, m);
                        }
                    }
                    catch (Exception rollback) when (rollback is not OutOfMemoryException and not StackOverflowException)
                    {
                        // The condition that broke the write — a full disk, a locked file —
                        // is the same one that breaks the restore. Swallowing it and then
                        // asserting "every file was put back" was the worst available
                        // outcome: the model reasons from an unchanged workspace that is
                        // not unchanged.
                        lost.Add(Path.GetFileName(path));
                    }
                }

                string restored = lost.Count == 0
                    ? "Every file was put back as it was."
                    : "These file(s) could NOT be put back and are now in an unknown state: "
                      + string.Join(", ", lost)
                      + ". Read them before doing anything else; everything else was restored.";
                return CodeExecResult.Refused($"the patch could not be written ({ex.Message}). {restored}");
            }

            // A patch's writes are reads too, on the same "writing counts as reading"
            // rule the rest of this surface uses: the model composed those bytes, so an
            // edit_file that follows a patch must not be gated on re-reading them. Without
            // this the two halves of the surface disagreed about the same file.
            foreach (string path in order)
            {
                if (pending[path] is { } written)
                    workspace.Reads.Record(path, written.Replace("\r\n", "\n", StringComparison.Ordinal), 1, int.MaxValue, complete: true);
                else
                    workspace.Reads.Forget(path);
            }

            _logger.LogInformation(LogEventIds.CodeExecPatched,
                "codeexec.patched files={Files}", outcomes.Count);

            // A patch is how a file gets CREATED as often as how one gets changed, so its
            // result has to offer the same download links a shell command's does — the
            // declaration promises them for "files you write", and the model has no way to
            // know the promise was scoped to one of the two tools.
            var sb = new StringBuilder(CodePatch.Describe(outcomes));

            // Matching all-or-nothing is not the same as being right: every anchor can
            // resolve and the file can still stop compiling. The declaration tells the
            // model not to re-read a file after a patch reported success, which is correct
            // about placement and silent about syntax — so the host checks instead.
            if (_syntax.Verify(
                    outcomes.Select(o => o.MovedTo ?? o.Path).ToList(), workspace, from)
                    is { Length: > 0 } broken)
            {
                sb.Append('\n').Append(broken);
            }
            IReadOnlyList<CodeArtifact> artifacts = CaptureArtifacts(
                workspace, _artifacts != null ? PatchSnapshot(workspace, order) : new(), out IReadOnlyList<string> skipped);
            AppendArtifacts(sb, artifacts, skipped);

            return new CodeExecResult(
                true, sb.ToString(), artifacts,
                artifacts.Count > 0 ? artifacts[0].RunId : string.Empty);
        }

        // ---- the file tools ------------------------------------------------
        //
        // read_file supplies context and write_file creates new files. apply_patch is
        // the advertised editor for both one-line repairs and changes across files.
        // The exact-replacement and explicit-overwrite APIs remain for compatibility.

        /// <summary>
        /// Show the model a file's real current bytes, numbered, and remember that it has
        /// seen them.
        /// </summary>
        public CodeExecResult ReadFile(ShellTools.ReadRequest request, SessionWorkspace? workspace)
        {
            if (workspace == null)
                return CodeExecResult.NoChange("there is no working directory in this conversation to read from.");

            // See ShellRunner.RunIn: the layout is re-asserted, never assumed.
            workspace.TryEnsureDirectories();

            if (!TryResolveForFileTool(workspace, request.Path, out string full, out string from, out string? error))
                return CodeExecResult.NoChange(error!);

            if (Directory.Exists(full))
            {
                return CodeExecResult.NoChange(
                    $"'{request.Path}' is a directory, not a file. List what is in it with the "
                    + $"{ShellTools.ShellToolName} tool.");
            }
            if (!File.Exists(full))
                return CodeExecResult.NoChange(MissingFile(request.Path, workspace, "read"));
            if (TooBigForFileTool(full, request.Path, out string? tooBig))
                return CodeExecResult.NoChange(tooBig!);

            string text;
            try
            {
                text = File.ReadAllText(full);
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or NotSupportedException)
            {
                return CodeExecResult.NoChange(
                    $"'{request.Path}' could not be read: {OutputPaths.Scrub(ex.Message, workspace, from)}");
            }

            // The ledger works in LF space at every other site, and so does what the model
            // is actually SHOWN — SplitLines strips the CR before a line is numbered. Record
            // the same text, or a CRLF file's entry never matches the LF text an edit checks
            // against, and every first edit after a read reports a change nobody made.
            string seenText = text.Replace("\r\n", "\n", StringComparison.Ordinal);

            List<string> lines = NumberedListing.SplitLines(text);
            if (NumberedListing.LooksBinary(lines))
            {
                return CodeExecResult.NoChange(
                    $"'{request.Path}' is not a text file, so there is nothing to show as lines. "
                    + $"Inspect it with a program run through the {ShellTools.ShellToolName} tool.");
            }

            int total = NumberedListing.RealLineCount(lines);
            if (total == 0)
                return new CodeExecResult(true, $"{request.Path} is empty (0 lines).", Array.Empty<CodeArtifact>(), string.Empty);

            int first = request.Offset > 0 ? request.Offset : 1;
            if (first > total)
            {
                return CodeExecResult.NoChange(
                    $"'{request.Path}' has {total} line(s), so there is nothing at line {first}. "
                    + "Read from line 1, or from a line inside the file.");
            }
            int count = request.Limit > 0 ? request.Limit : ShellTools.DefaultReadLines;

            // In long, so an enormous 'limit' cannot wrap before the clamp. The argument
            // reader clamps any oversized JSON number to int.MaxValue, so "read the rest of
            // it" written as limit=99999999999 arrived here and overflowed to int.MinValue —
            // a header reading "line 2 to -2147483648 of 10", a footer offering offset
            // -2147483647, and a ledger window that poisoned the union arithmetic for every
            // later read of that path.
            int last = (int)Math.Min(total, (long)first + count - 1);
            bool complete = first == 1 && last == total;

            // Claude Code's anti-re-read: a file already read and unchanged is not
            // rendered again. The saving is the point but not the whole point — a model
            // that re-reads after every edit spends its context on bytes it already has,
            // and the reply is what tells it the earlier result is still authoritative.
            if (complete && workspace.Reads.Check(full, seenText).Freshness == ReadFreshness.Fresh)
            {
                return new CodeExecResult(
                    true,
                    $"{request.Path} is unchanged since you last read it ({total} line(s)) — that earlier "
                    + "result is still current, so use it rather than reading again.",
                    Array.Empty<CodeArtifact>(), string.Empty);
            }

            var sb = new StringBuilder();
            sb.Append(request.Path).Append(", line ").Append(first.ToString(CultureInfo.InvariantCulture));
            if (last != first)
                sb.Append(" to ").Append(last.ToString(CultureInfo.InvariantCulture));
            sb.Append(" of ").Append(total.ToString(CultureInfo.InvariantCulture)).Append(":\n");

            NumberedListing.Append(
                sb, lines, first - 1, last - 1,
                NumberedListing.MaxReadChars, NumberedListing.MaxReadLineChars, out int lastShownIndex);

            // What was RENDERED, which the character budget can cut short of what was asked
            // for. Recording the requested range instead told the ledger the model had seen
            // lines that were never printed — and then the re-read shortcut refused to show
            // them ("unchanged since you last read it") while the replace_all gate treated
            // the file as fully seen. Both of those invert the ledger's own purpose.
            int shown = lastShownIndex + 1;
            if (shown < last)
            {
                last = Math.Max(first, shown);
                complete = false;
            }

            if (last < total)
            {
                sb.Append("\n(").Append((total - last).ToString(CultureInfo.InvariantCulture))
                  .Append(" more line(s) below; read them with offset ")
                  .Append((last + 1).ToString(CultureInfo.InvariantCulture)).Append(".)\n");
            }

            workspace.Reads.Record(full, seenText, first, last, complete);
            _logger.LogInformation(LogEventIds.CodeExecRead,
                "codeexec.read path={Path} lines={Lines} complete={Complete}",
                request.Path, last - first + 1, complete);

            return new CodeExecResult(true, sb.ToString(), Array.Empty<CodeArtifact>(), string.Empty);
        }

        /// <summary>
        /// Replace one exact string in one file, leaving every other byte alone.
        /// </summary>
        public CodeExecResult EditFile(ShellTools.EditRequest request, SessionWorkspace? workspace)
        {
            if (workspace == null)
                return CodeExecResult.NoChange("there is no working directory in this conversation to edit in.");

            // See ShellRunner.RunIn: the layout is re-asserted, never assumed.
            workspace.TryEnsureDirectories();

            if (!TryResolveForFileTool(workspace, request.Path, out string full, out string from, out string? error))
                return CodeExecResult.NoChange(error!);

            if (Directory.Exists(full))
                return CodeExecResult.NoChange($"'{request.Path}' is a directory, not a file.");
            if (!File.Exists(full))
            {
                return CodeExecResult.NoChange(
                    $"'{request.Path}' does not exist, so there is nothing in it to change. "
                    + $"Create it with {ShellTools.WriteToolName}."
                    + NearbyHint(request.Path, workspace));
            }
            if (TooBigForFileTool(full, request.Path, out string? tooBig))
                return CodeExecResult.NoChange(tooBig!);

            string original;
            try
            {
                original = File.ReadAllText(full);
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or NotSupportedException)
            {
                return CodeExecResult.NoChange(
                    $"'{request.Path}' could not be read: {OutputPaths.Scrub(ex.Message, workspace, from)}");
            }

            // Everything below works in LF space and the file's own newline is put back at
            // the end, so a CRLF file is never silently converted to LF — a whole-file
            // change nobody asked for, and one that shows up in someone's diff the next
            // morning as every line modified.
            string newline = original.Contains("\r\n", StringComparison.Ordinal) ? "\r\n" : "\n";
            string content = original.Replace("\r\n", "\n", StringComparison.Ordinal);
            string oldString = request.OldString.Replace("\r\n", "\n", StringComparison.Ordinal);
            string newString = request.NewString.Replace("\r\n", "\n", StringComparison.Ordinal);

            List<string> lines = NumberedListing.SplitLines(content);
            FileLedger.ReadState seen = workspace.Reads.Check(full, content);

            if (string.Equals(oldString, newString, StringComparison.Ordinal))
            {
                return CodeExecResult.NoChange(
                    "old_string and new_string are identical, so this edit would change nothing. "
                    + "Send the text you want the file to hold instead.");
            }

            FileEdit.MatchResult match = FileEdit.Find(content, oldString);

            if (!match.Found)
            {
                // The whole reason the model reaches for a rewrite. A refusal that only
                // says "not found" leaves re-typing the file as the cheapest next move, so
                // this one hands over the file's real bytes and, when it can, says which
                // near-miss caused it.
                return CodeExecResult.NoChange(
                    DescribeEditMiss(request.Path, oldString, content, lines, seen, workspace, full));
            }

            if (match.Count > 1 && !request.ReplaceAll)
            {
                // Claude Code refuses here, and this deliberately differs from the patch
                // side of this same host, which applies at the first match and says so.
                // The reasoning does not transfer: a V4A hunk's context is a fixed few
                // lines the model cannot easily widen, so refusing taught models to stop
                // writing anchors at all — whereas old_string is unbounded and "include
                // more of the surrounding lines" is an instruction that can be acted on
                // immediately.
                var sb = new StringBuilder();
                sb.Append("that text appears ").Append(match.Count.ToString(CultureInfo.InvariantCulture))
                  .Append(" times in ").Append(request.Path)
                  .Append(", so it does not say which one to change. Include more of the lines around "
                        + "it until only the one you mean is left — or set replace_all to change every "
                        + "one, which is how you rename something throughout a file. Nothing was "
                        + "written.\nThe matches start at:\n");
                foreach (int offset in match.Offsets)
                {
                    int line = LineOf(content, offset);
                    sb.Append(NumberedListing.Prefix(line + 1))
                      .Append(Clip(lines.Count > line ? lines[line] : string.Empty)).Append('\n');
                }
                if (match.Count > match.Offsets.Count)
                    sb.Append("  … and ").Append((match.Count - match.Offsets.Count).ToString(CultureInfo.InvariantCulture))
                      .Append(" more.\n");
                return CodeExecResult.NoChange(sb.ToString());
            }

            // The read gate. A whole-file substitution whose extent the model cannot see
            // is the one case where "it matched, so apply it" is not good enough: it may
            // match in places the model has never looked at.
            if (request.ReplaceAll && match.Count > 1
                && !seen.Covers(1, NumberedListing.RealLineCount(lines)))
            {
                // Gated on having seen the WHOLE file, not merely on having seen some of
                // it. The hazard is specific: a substitution whose extent the model cannot
                // see may land in places it has never looked at, and "I read lines 1-400
                // of 900" is exactly as blind about lines 401-900 as having read nothing.
                // A successful edit records the file's new content, so gating on
                // "unread" alone would let one narrow edit unlock a whole-file rename.
                // Gated on COVERAGE, not on a single complete read. A model that pages
                // through a long file the way read_file's own footer tells it to — offset
                // 1, then 401, then 801 — has seen every line, and refusing it here would
                // be the host demanding something the model had already done and could not
                // do differently: a second plain read_file returns the same first page,
                // and nothing anywhere names the one call that would satisfy the check. A
                // small model in that loop reaches for write_file and rewrites the file,
                // which is the outcome this whole surface exists to prevent.
                int missing = NumberedListing.RealLineCount(lines);
                return CodeExecResult.NoChange(
                    $"replace_all would change {match.Count} places in {request.Path}, and you have not "
                    + $"seen all {missing} lines of it in this conversation, so you cannot tell what is "
                    + $"in every one of them. Read the whole file first — "
                    + $"{ShellTools.ReadToolName} with offset 1 and limit {missing} — then edit. "
                    + "Nothing was written.");
            }

            // The replacement goes through the same transform that made the search text
            // match, BEFORE it is restyled. Repairing one half of the model's input and
            // writing the other half through untouched is how '   42 | ' prefixes ended up
            // inside source files.
            string carried = FileEdit.ApplyRungTo(newString, match.Rung);
            string replacement = FileEdit.Restyle(carried, match.Matched, match.Search);
            bool restyled = !string.Equals(replacement, carried, StringComparison.Ordinal);
            string updated = request.ReplaceAll
                ? ReplaceAll(content, match, replacement)
                : content.Remove(match.Index, match.Search.Length).Insert(match.Index, replacement);

            int changedLine = LineOf(content, match.Index);

            if (!TryWriteFileBytes(full, updated, newline, overwrite: true, out string? writeError))
                return CodeExecResult.NoChange(
                    $"'{request.Path}' could not be written: {OutputPaths.Scrub(writeError!, workspace, from)}");

            // The new content, under the coverage the model ACTUALLY has — not 1..∞.
            //
            // Recording the whole file as seen would be the ledger telling itself a
            // convenient lie: an edit proves the model had the bytes it named, and says
            // nothing about the rest of the file. Claiming otherwise would let one narrow
            // edit silently authorise a whole-file replace_all over regions it has never
            // looked at, which is precisely the check three lines above.
            int editedFirst = changedLine + 1;
            int editedLast = editedFirst + replacement.Count(c => c == '\n');
            workspace.Reads.Record(full, updated, editedFirst, editedLast, complete: seen.Complete);

            _logger.LogInformation(LogEventIds.CodeExecEdited,
                "codeexec.edited path={Path} rung={Rung} freshness={Freshness} matches={Matches} all={All}",
                request.Path, match.Rung, seen.Freshness, match.Count, request.ReplaceAll);

            return CompleteFileMutation(
                DescribeEdit(request, match, seen, changedLine, restyled, workspace, from),
                workspace, full);
        }

        /// <summary>Create a file, or replace one whole.</summary>
        public CodeExecResult WriteFile(ShellTools.WriteRequest request, SessionWorkspace? workspace)
        {
            if (workspace == null)
                return CodeExecResult.NoChange("there is no working directory in this conversation to write to.");

            // See ShellRunner.RunIn: the layout is re-asserted, never assumed.
            workspace.TryEnsureDirectories();

            if (!TryResolveForFileTool(workspace, request.Path, out string full, out string from, out string? error))
                return CodeExecResult.NoChange(error!);

            if (Directory.Exists(full))
                return CodeExecResult.NoChange($"'{request.Path}' is a directory, so it cannot be written as a file.");

            bool existed = File.Exists(full);
            if (existed && TooBigForFileTool(full, request.Path, out string? tooBig))
                return CodeExecResult.NoChange(tooBig!);

            string? previous = null;
            FileLedger.ReadState seen = default;
            if (existed)
            {
                try
                {
                    previous = File.ReadAllText(full);
                }
                catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or NotSupportedException)
                {
                    return CodeExecResult.NoChange(
                        $"'{request.Path}' could not be read before being replaced: "
                        + OutputPaths.Scrub(ex.Message, workspace, from));
                }
                seen = workspace.Reads.Check(full, previous.Replace("\r\n", "\n", StringComparison.Ordinal));
            }

            string newline = previous is { } p && p.Contains("\r\n", StringComparison.Ordinal) ? "\r\n" : "\n";
            string content = request.Content.Replace("\r\n", "\n", StringComparison.Ordinal);

            // Creating and replacing are materially different operations. Treating an
            // omitted flag as permission to replace made write_file the easiest escape
            // hatch after a failed edit: the model re-emitted every correct line, the
            // host destroyed the old file, and only THEN explained that apply_patch would
            // have been cheaper. Refuse before touching the bytes. A genuine rewrite is
            // still available, but it has to be named explicitly so a local repair does
            // not silently become one.
            if (existed && !request.Overwrite)
            {
                string normalizedPrevious = previous!.Replace("\r\n", "\n", StringComparison.Ordinal);
                if (string.Equals(normalizedPrevious, content, StringComparison.Ordinal))
                {
                    // The model supplied every byte, so it has complete knowledge of the
                    // current file even though no disk write is necessary. Keep the old
                    // syntax-verification guarantee and refresh the ledger so a narrow
                    // edit can follow without an artificial read round.
                    workspace.Reads.Record(full, content, 1, int.MaxValue, complete: true);
                    string? broken = _syntax.Verify(new[] { request.Path }, workspace, from);
                    string message = $"'{request.Path}' already has exactly that content; nothing was written.";
                    if (!string.IsNullOrEmpty(broken))
                        message += "\n" + broken;
                    else
                        message += " Its syntax check passed.";
                    return new CodeExecResult(
                        true, message, Array.Empty<CodeArtifact>(), string.Empty);
                }

                return CodeExecResult.NoChange(
                    $"'{request.Path}' already exists, so write_file did not replace it. "
                    + $"Use {ShellTools.PatchToolName} to modify or replace existing content in one file or "
                    + "multiple files, with context matching the current text. Use write_file only to create "
                    + "a new file. Nothing was written.");
            }

            // Preserve the create-only promise across the small interval between the
            // File.Exists check above and opening the destination. A background job may
            // create the same path in that interval; CreateNew makes the kernel arbitrate
            // the race instead of silently truncating the other writer's file.
            if (!TryWriteFileBytes(
                    full,
                    content,
                    newline,
                    overwrite: existed || request.Overwrite,
                    out string? writeError))
                return CodeExecResult.NoChange(
                    $"'{request.Path}' could not be written: {OutputPaths.Scrub(writeError!, workspace, from)}");

            workspace.Reads.Record(full, content, 1, int.MaxValue, complete: true);

            var sb = new StringBuilder();
            int lineCount = NumberedListing.RealLineCount(NumberedListing.SplitLines(content));
            sb.Append(existed ? "Replaced " : "Created ").Append(request.Path)
              .Append(" (").Append(lineCount.ToString(CultureInfo.InvariantCulture)).Append(" line(s)).");

            // The one place a rewrite can be noticed BY CONSTRUCTION rather than by
            // scanning a command line for a redirect, which is why this tool exists at all
            // rather than being left to the shell.
            if (previous != null
                && RewriteWatch.DescribeRetyped(request.Path, previous, content, ShellTools.PatchToolName)
                    is { Length: > 0 } retyped)
            {
                sb.Append('\n').Append(retyped);
                _logger.LogInformation(LogEventIds.CodeExecRewrote,
                    "codeexec.rewrote path={Path} lines={Lines}", request.Path, lineCount);
            }
            else if (_syntax.Verify(new[] { request.Path }, workspace, from) is { Length: > 0 } broken)
            {
                sb.Append('\n').Append(broken);
            }
            else
            {
                sb.Append(" It now holds exactly what you sent — no need to read it back.");
            }

            if (existed && seen.Freshness == ReadFreshness.Unread)
            {
                sb.Append("\nYou had not read ").Append(request.Path)
                  .Append(" in this conversation, so whatever it held before is gone and is not "
                        + "recoverable from here.");
            }

            return CompleteFileMutation(sb.ToString(), workspace, full);
        }

        private CodeExecResult CompleteFileMutation(string content, SessionWorkspace workspace, string fullPath)
        {
            if (_artifacts == null)
                return new CodeExecResult(true, content, Array.Empty<CodeArtifact>(), string.Empty);

            // File tools must publish their own outputs. A later shell invocation
            // snapshots these files as existing inputs and will not capture them.
            // Reuse the patch path's filtering, limits, and retained copies, with
            // only this mutation's target eligible for publication.
            var sb = new StringBuilder(content);
            IReadOnlyList<CodeArtifact> artifacts = CaptureArtifacts(
                workspace, PatchSnapshot(workspace, new[] { fullPath }), out IReadOnlyList<string> skipped);
            AppendArtifacts(sb, artifacts, skipped);
            return new CodeExecResult(true, sb.ToString(), artifacts,
                artifacts.Count > 0 ? artifacts[0].RunId : string.Empty);
        }

        // ---- what the file tools say -----------------------------------------

        /// <summary>Describe a successful edit, including anything the host had to relax.</summary>
        private string DescribeEdit(
            ShellTools.EditRequest request, FileEdit.MatchResult match, FileLedger.ReadState seen,
            int changedLine, bool restyled, SessionWorkspace workspace, string from)
        {
            var sb = new StringBuilder();
            sb.Append("Edited ").Append(request.Path);
            if (request.ReplaceAll && match.Count > 1)
            {
                sb.Append(" in ").Append(match.Count.ToString(CultureInfo.InvariantCulture))
                  .Append(" places");
            }
            else
            {
                sb.Append(" at line ").Append((changedLine + 1).ToString(CultureInfo.InvariantCulture));
            }
            sb.Append('.');

            if (FileEdit.Describe(match.Rung, restyled) is { Length: > 0 } rung)
                sb.Append('\n').Append(rung);

            // Applied WITHOUT a prior read, which the reference gates on. It is applied
            // rather than refused because the anchor was unique — the model demonstrably
            // had the bytes — and a refusal here would cost a round to prove something
            // already proved. Saying so is what keeps the invariant meaningful.
            if (seen.Freshness == ReadFreshness.Unread)
            {
                sb.Append("\nYou had not read ").Append(request.Path)
                  .Append(" in this conversation, so this was checked against the file's current bytes "
                        + "before being applied; it matched, and nothing else was touched. Read a file "
                        + "before an edit that depends on lines you cannot see.");
            }
            else if (seen.Freshness == ReadFreshness.Stale)
            {
                sb.Append("\nNote that ").Append(request.Path)
                  .Append(" had changed since you last read it — probably by a command you ran, or by a "
                        + "program one of them started. The edit applied cleanly, but the file holds "
                        + "other changes that are not in your context.");
            }

            if (_syntax.Verify(new[] { request.Path }, workspace, from) is { Length: > 0 } broken)
                sb.Append('\n').Append(broken);
            else
                sb.Append(" The rest of the file is unchanged — no need to read it back.");

            return sb.ToString();
        }

        /// <summary>
        /// Say why an edit did not match, and show the bytes that are actually there.
        ///
        /// <para>
        /// The most important message on this surface. A model told only "not found" has
        /// re-typing the file as its cheapest next move, which is the behaviour the whole
        /// tool exists to remove — so this diagnoses the likely cause, then hands over the
        /// file's own numbered lines around the closest thing to what was asked for, and
        /// RECORDS that it showed them, so the next attempt is not gated on a read the
        /// model has effectively just had.
        /// </para>
        /// </summary>
        private static string DescribeEditMiss(
            string path, string oldString, string content, List<string> lines,
            FileLedger.ReadState seen, SessionWorkspace workspace, string full)
        {
            var sb = new StringBuilder();
            sb.Append("that text is not in ").Append(path)
              .Append(", so nothing was changed. You sent:\n");

            string[] wanted = oldString.Split('\n');
            for (int i = 0; i < wanted.Length && i < 6; i++)
                sb.Append("  | ").Append(Clip(wanted[i])).Append('\n');
            if (wanted.Length > 6)
                sb.Append("  | … and ").Append((wanted.Length - 6).ToString(CultureInfo.InvariantCulture))
                  .Append(" more line(s)\n");

            // The single most likely cause, named first. Indentation is the one that
            // actually happens and the one that is invisible in a summary.
            string firstWanted = wanted.Length > 0 ? wanted[0] : string.Empty;
            int anchor = -1;
            if (firstWanted.Trim().Length > 0)
            {
                for (int i = 0; i < lines.Count; i++)
                {
                    if (string.Equals(lines[i].Trim(), firstWanted.Trim(), StringComparison.Ordinal))
                    {
                        anchor = i;
                        break;
                    }
                }
                if (anchor >= 0)
                {
                    sb.Append("That line IS in the file, at line ")
                      .Append((anchor + 1).ToString(CultureInfo.InvariantCulture))
                      .Append(", but not with the spacing you wrote — the indentation has to match "
                            + "exactly. Here is what the file actually holds:\n");
                }
            }

            if (anchor < 0)
            {
                // Nothing resembles it. Do not invent a location: send the model to look.
                int shown = Math.Min(NumberedListing.RealLineCount(lines), 20);
                sb.Append("No line of ").Append(path).Append(" looks like the first line of that text. ")
                  .Append("Check you are editing the right file — here is how it starts:\n");
                NumberedListing.Append(sb, lines, 0, shown - 1);
                sb.Append("Read the part you meant to change with ").Append(ShellTools.ReadToolName)
                  .Append(", then copy the text out of that result.\n");
                RecordShown(workspace, full, content, 1, shown, seen);
                return sb.ToString();
            }

            int from = Math.Max(0, anchor - 2);
            int to = Math.Min(NumberedListing.LastRealLine(lines), anchor + wanted.Length + 2);
            NumberedListing.Append(sb, lines, from, to);
            sb.Append("Copy those lines exactly — without the '")
              .Append(NumberedListing.Prefix(anchor + 1).TrimStart())
              .Append("' prefix, which is not part of the file — and edit again.\n");

            RecordShown(workspace, full, content, from + 1, to + 1, seen);
            return sb.ToString();
        }

        /// <summary>
        /// Remember the region a refusal just showed.
        ///
        /// <para>
        /// There is no state in which the model is told "you have not read this" and then
        /// handed nothing to read. A refusal that quotes the file HAS shown it those
        /// lines, so the next attempt against them is authorised — otherwise the gate
        /// would spend a round demanding a read whose result the model is already holding.
        /// </para>
        /// </summary>
        private static void RecordShown(
            SessionWorkspace workspace, string full, string content, int first, int last,
            FileLedger.ReadState seen)
        {
            if (seen.Freshness is ReadFreshness.Fresh or ReadFreshness.Partial)
                return;
            workspace.Reads.Record(full, content, first, last, complete: false);
        }

        /// <summary>Replace every occurrence, walking the same rung that matched.</summary>
        private static string ReplaceAll(string content, FileEdit.MatchResult match, string replacement)
        {
            if (match.Rung == FileEdit.Rung.Exact)
                return content.Replace(match.Search, replacement, StringComparison.Ordinal);

            // A tolerant rung matched, so the occurrences are not all byte-identical and
            // string.Replace cannot find them. Walk them, restyling each against the bytes
            // that are actually there — the file may spell two of them differently.
            var sb = new StringBuilder(content.Length);
            int at = 0;
            while (at < content.Length)
            {
                FileEdit.MatchResult next = FileEdit.Find(content.Substring(at), match.Search);
                if (!next.Found)
                    break;
                sb.Append(content, at, next.Index);
                sb.Append(FileEdit.Restyle(replacement, next.Matched, next.Search));
                at += next.Index + next.Search.Length;
            }
            sb.Append(content, at, content.Length - at);
            return sb.ToString();
        }

        /// <summary>Write text back in the file's own newline style and encoding.</summary>
        internal static bool TryWriteFileBytes(
            string full,
            string lfContent,
            string newline,
            bool overwrite,
            out string? error)
        {
            error = null;
            string? temporary = null;
            try
            {
                string? parent = Path.GetDirectoryName(full);
                if (!string.IsNullOrEmpty(parent))
                    Directory.CreateDirectory(parent);

                string text = newline == "\n"
                    ? lfContent
                    : lfContent.Replace("\n", newline, StringComparison.Ordinal);
                Encoding encoding = EncodingOf(full);
                byte[] bytes = Concat(encoding.GetPreamble(), encoding.GetBytes(text));

                // Keep explicit replacement's established metadata behaviour: opening
                // the existing inode with Create preserves its permissions. The atomic
                // publication requirement below is specifically the create-only path,
                // whose destination must not become visible until every byte is ready.
                if (overwrite)
                {
                    using var replacement = new FileStream(
                        full, FileMode.Create, FileAccess.Write, FileShare.Read);
                    replacement.Write(bytes, 0, bytes.Length);
                    return true;
                }

                // Build the complete replacement beside its destination, then publish
                // it with one rename. In create-only mode, opening the destination with
                // CreateNew made the NAME visible before the bytes were complete. A
                // failed/short write consequently left a partial target behind, and the
                // catch block saw that target and falsely reported that another writer
                // had won the existence race. The temporary file keeps construction
                // failures private; only the move below can be a target collision.
                string temporaryDirectory = string.IsNullOrEmpty(parent)
                    ? Directory.GetCurrentDirectory()
                    : parent;
                temporary = Path.Combine(
                    temporaryDirectory,
                    ".tensorsharp-write-" + Guid.NewGuid().ToString("N") + ".tmp");
                using var stream = new FileStream(
                    temporary,
                    FileMode.CreateNew,
                    FileAccess.Write,
                    FileShare.None);
                stream.Write(bytes, 0, bytes.Length);
            }
            catch (Exception ex) when (ex is not OutOfMemoryException and not StackOverflowException)
            {
                TryDeleteUnpublishedWrite(temporary);
                error = ex.Message;
                return false;
            }

            try
            {
                // File.Move is the publication point. With overwrite=false the kernel,
                // not a racy File.Exists check, decides whether another writer won.
                File.Move(temporary!, full);
                temporary = null;
                return true;
            }
            catch (IOException) when (File.Exists(full) || Directory.Exists(full))
            {
                error = "the path appeared after it was checked; overwrite was not authorized, "
                    + "so the existing path was preserved and nothing was written";
                return false;
            }
            catch (Exception ex) when (ex is not OutOfMemoryException and not StackOverflowException)
            {
                error = ex.Message;
                return false;
            }
            finally
            {
                TryDeleteUnpublishedWrite(temporary);
            }
        }

        private static void TryDeleteUnpublishedWrite(string? path)
        {
            if (string.IsNullOrEmpty(path))
                return;
            try { File.Delete(path); }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { }
        }

        /// <summary>
        /// Resolve a model-supplied path for a file tool.
        ///
        /// <para>
        /// Through the workspace, from where the SHELL currently is, and never with
        /// <see cref="Path.Combine(string, string)"/>. Two reasons, both already paid for
        /// here: the halves of this surface have to agree what "main.c" means after a
        /// <c>cd</c>, and the workspace's resolution is what walks SYMLINKS. These reads
        /// and writes happen in the HOST process, which is not sandboxed, and
        /// <c>ln -s ~/.ssh/id_rsa notes.txt</c> is one ordinary permitted shell command —
        /// so a read tool that renders whatever file the model names is a strictly larger
        /// exposure than a patch that writes one.
        /// </para>
        /// </summary>
        private bool TryResolveForFileTool(
            SessionWorkspace workspace, string path, out string full, out string from, out string? error)
        {
            from = _shell != null ? SessionFor(workspace).CurrentDirectory : workspace.WorkDirectory;
            return workspace.TryResolveFrom(from, path, out full, out error);
        }

        /// <summary>What to say about a file that is not there, with the nearest name that is.</summary>
        private static string MissingFile(string path, SessionWorkspace workspace, string verb) =>
            $"'{path}' does not exist, so there is nothing to {verb}." + NearbyHint(path, workspace);

        /// <summary>
        /// "Did you mean …?" over what is actually in the workspace.
        ///
        /// <para>
        /// Half of the failures in the earlier editing era were about WHERE, not about
        /// what: a path spelled with a directory that is not there, or a name off by a
        /// character. A list of everything is a haystack; the nearest name is an answer.
        /// </para>
        /// </summary>
        private static string NearbyHint(string path, SessionWorkspace workspace)
        {
            string wanted = Path.GetFileName(path);
            if (wanted.Length == 0)
                return workspace.DescribeWhatIsHere();

            string? best = null;
            int bestScore = int.MaxValue;
            foreach ((string candidate, long _) in workspace.ListFiles())
            {
                string name = Path.GetFileName(candidate);
                if (string.Equals(name, wanted, StringComparison.OrdinalIgnoreCase))
                {
                    // The same name in a different directory: that is the answer, not a
                    // near miss, so say it and stop looking.
                    return $" There is a file with that name at '{candidate}' — did you mean that path?";
                }
                int score = Distance(name, wanted);
                if (score < bestScore)
                {
                    bestScore = score;
                    best = candidate;
                }
            }

            // Only a genuinely close name. Offering a distant one sends the model to edit
            // the wrong file, which is worse than offering nothing.
            if (best != null && bestScore <= Math.Max(2, wanted.Length / 3))
                return $" Did you mean '{best}'?" + workspace.DescribeWhatIsHere();

            return workspace.DescribeWhatIsHere();
        }

        /// <summary>Levenshtein distance, bounded by the short lengths this compares.</summary>
        private static int Distance(string a, string b)
        {
            if (a.Length == 0 || b.Length == 0)
                return Math.Max(a.Length, b.Length);
            if (Math.Abs(a.Length - b.Length) > 8)
                return int.MaxValue;

            var previous = new int[b.Length + 1];
            var current = new int[b.Length + 1];
            for (int j = 0; j <= b.Length; j++)
                previous[j] = j;

            for (int i = 1; i <= a.Length; i++)
            {
                current[0] = i;
                for (int j = 1; j <= b.Length; j++)
                {
                    int cost = char.ToLowerInvariant(a[i - 1]) == char.ToLowerInvariant(b[j - 1]) ? 0 : 1;
                    current[j] = Math.Min(Math.Min(current[j - 1] + 1, previous[j] + 1), previous[j - 1] + cost);
                }
                (previous, current) = (current, previous);
            }
            return previous[b.Length];
        }

        /// <summary>The 0-based line a character offset falls on.</summary>
        private static int LineOf(string content, int offset)
        {
            int line = 0;
            for (int i = 0; i < offset && i < content.Length; i++)
            {
                if (content[i] == '\n')
                    line++;
            }
            return line;
        }

        /// <summary>Longest line echoed back into a message.</summary>
        private const int MaxEchoChars = 200;

        private static string Clip(string line) =>
            line.Length <= MaxEchoChars ? line : line.Substring(0, MaxEchoChars) + " …";

        /// <summary>
        /// Largest file the host will read into its own memory for a file tool.
        ///
        /// <para>
        /// These run in the HOST process, unsandboxed and with no per-call memory limit of
        /// their own, and the path comes from the model. Reading a multi-gigabyte file to
        /// render 400 lines of it is not a hypothetical: a workspace accumulates whatever
        /// a command downloaded or generated, and .NET decodes UTF-8 to UTF-16, so the
        /// managed copy is up to twice the file plus the line list on top. Refused with a
        /// message that says what to do instead, rather than left to an
        /// OutOfMemoryException that takes the whole server with it.
        /// </para>
        /// </summary>
        private const long MaxFileToolBytes = 8L * 1024 * 1024;

        /// <summary>Whether this file is small enough to read into the host, or why not.</summary>
        private static bool TooBigForFileTool(string full, string path, out string? refusal)
        {
            refusal = null;
            try
            {
                var info = new FileInfo(full);
                if (!info.Exists || info.Length <= MaxFileToolBytes)
                    return false;

                refusal =
                    $"'{path}' is {info.Length / (1024 * 1024)} MB, which is too large to read or edit "
                    + $"here (the limit is {MaxFileToolBytes / (1024 * 1024)} MB). Work on it with a "
                    + $"program run through the {ShellTools.ShellToolName} tool instead.";
                return true;
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or ArgumentException)
            {
                return false;   // unknowable size is the read's problem to report, not this one
            }
        }

        /// <summary>Largest file worth taking into the ledger after a command.</summary>
        private const int MaxIngestedBytes = 512 * 1024;

        /// <summary>Files worth taking in from one command. A command writing more is not editing a program.</summary>
        private const int MaxIngestedFiles = 32;

        /// <summary>
        /// Record what a command created or changed, so the editor can work against it.
        ///
        /// <para>
        /// Deliberately NOT an attempt to parse <c>cat</c> or <c>sed -n</c> out of the
        /// command line to claim the model has READ something. The host would then be
        /// asserting the model attended to bytes that may have been cut off by the output
        /// cap or scrolled past unread, and a false "you have seen this" authorises
        /// exactly the blind edit the ledger exists to prevent. What a command WROTE is a
        /// different claim and a sound one: those bytes came out of the model.
        /// </para>
        /// </summary>
        private static void IngestWrittenFiles(
            SessionWorkspace workspace, Dictionary<string, (long Length, DateTime WriteTime)> before,
            string? command)
        {
            if (workspace == null)
                return;

            int taken = 0;
            try
            {
                foreach (KeyValuePair<string, (long Length, DateTime WriteTime)> now in workspace.SnapshotWorkFiles())
                {
                    if (taken >= MaxIngestedFiles)
                        break;
                    if (before.TryGetValue(now.Key, out (long Length, DateTime WriteTime) was)
                        && was.Length == now.Value.Length && was.WriteTime == now.Value.WriteTime)
                    {
                        continue;   // untouched
                    }
                    if (now.Value.Length == 0 || now.Value.Length > MaxIngestedBytes)
                        continue;

                    string full = Path.Combine(workspace.WorkDirectory, now.Key.Replace('/', Path.DirectorySeparatorChar));
                    try
                    {
                        string text = File.ReadAllText(full);
                        List<string> lines = NumberedListing.SplitLines(text);
                        if (NumberedListing.LooksBinary(lines))
                            continue;

                        string seen = text.Replace("\r\n", "\n", StringComparison.Ordinal);

                        // "Those bytes came out of the model" is true of a heredoc it
                        // typed and false of anything a PROGRAM produced. `unzip data.zip`
                        // and `python3 build.py` both drop files into the workspace that
                        // the model has never seen a line of, and marking those completely
                        // read hands replace_all a file nobody has looked at — the exact
                        // case the edit gate refuses. The test is whether the content is
                        // literally in the command, which is what a heredoc means.
                        bool typedItself = command != null
                            && seen.Length > 0
                            && command.Contains(seen.TrimEnd('\n'), StringComparison.Ordinal);

                        // Not recorded at all when a program produced it, rather than
                        // recorded with a narrow window. Recording the new content under
                        // ANY window would make the ledger's hash match again, so a file
                        // the model had read and a build step then regenerated would come
                        // back Fresh — the change hidden by the very mechanism that exists
                        // to notice it. Leaving it alone is what makes the next edit report
                        // "this had changed since you last read it", which is the truth.
                        if (!typedItself)
                            continue;

                        workspace.Reads.Record(full, seen, 1, int.MaxValue, complete: true);
                        taken++;
                    }
                    catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or NotSupportedException)
                    {
                        // A file that cannot be read cannot be remembered, and inventing
                        // an entry would authorise an edit against content nobody has.
                    }
                }
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                // Best effort throughout: this runs after a command that already
                // succeeded and must never be the reason its result is lost.
            }
        }

        /// <summary>
        /// The work directory as it was BEFORE this patch, for artifact capture: every
        /// file except the ones the patch just touched, so only those come back as output.
        /// </summary>
        private static Dictionary<string, (long Length, DateTime WriteTime)> PatchSnapshot(
            SessionWorkspace workspace, IReadOnlyList<string> touched)
        {
            Dictionary<string, (long Length, DateTime WriteTime)> snapshot = workspace.SnapshotWorkFiles();
            foreach (string path in touched)
            {
                string relative = Path.GetRelativePath(workspace.WorkDirectory, path).Replace('\\', '/');
                snapshot.Remove(relative);
            }
            return snapshot;
        }

        private static string ResolveOr(SessionWorkspace workspace, string from, string relative) =>
            workspace.TryResolveFrom(from, relative, out string full, out _) ? full : relative;

        private static bool PathsEqual(string a, string b) =>
            string.Equals(
                Path.GetFullPath(a), Path.GetFullPath(b),
                OperatingSystem.IsLinux() ? StringComparison.Ordinal : StringComparison.OrdinalIgnoreCase);

        /// <summary>The encoding a file already uses, so a patch does not silently convert it.</summary>
        private static Encoding EncodingOf(string path)
        {
            try
            {
                if (!File.Exists(path))
                    return new UTF8Encoding(encoderShouldEmitUTF8Identifier: false);

                var head = new byte[4];
                using FileStream stream = File.OpenRead(path);
                int read = stream.Read(head, 0, head.Length);
                if (read >= 3 && head[0] == 0xEF && head[1] == 0xBB && head[2] == 0xBF)
                    return new UTF8Encoding(encoderShouldEmitUTF8Identifier: true);
                if (read >= 2 && head[0] == 0xFF && head[1] == 0xFE)
                    return new UnicodeEncoding(bigEndian: false, byteOrderMark: true);
                if (read >= 2 && head[0] == 0xFE && head[1] == 0xFF)
                    return new UnicodeEncoding(bigEndian: true, byteOrderMark: true);
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { }
            return new UTF8Encoding(encoderShouldEmitUTF8Identifier: false);
        }

        private static byte[] Concat(byte[] a, byte[] b)
        {
            if (a.Length == 0)
                return b;
            var joined = new byte[a.Length + b.Length];
            Buffer.BlockCopy(a, 0, joined, 0, a.Length);
            Buffer.BlockCopy(b, 0, joined, a.Length, b.Length);
            return joined;
        }

        // ---- describing -----------------------------------------------------

        /// <summary>
        /// Render typed capability axes as command-facing prose. Security decisions must
        /// not be made by searching these sentences for a word: wording changes, while
        /// <see cref="SkillSandboxCapabilities.ConfinesNetwork"/> does not.
        /// </summary>
        private static IReadOnlyList<string> CommandGaps(
            SkillSandboxCapabilities? capabilities, bool includeNetwork)
        {
            SkillSandboxCapabilities actual = capabilities
                ?? new SkillSandboxCapabilities(false, false, false, false);
            var gaps = new List<string>();
            if (!actual.ConfinesWrites)
                gaps.Add("commands may write anywhere the host process can");
            if (includeNetwork && !actual.ConfinesNetwork)
                gaps.Add("commands may reach the network");
            if (!actual.ConfinesHomeReads)
                gaps.Add("commands may read the user's home directory");
            if (!actual.BoundsProcessTree)
                gaps.Add("a child process may outlive the request");
            return gaps;
        }

        private CodeExecResult Describe(
            string command,
            ConfinedResult run,
            SessionWorkspace workspace,
            ShellSession session,
            Dictionary<string, (long Length, DateTime WriteTime)> before,
            IReadOnlyList<string> notes,
            TimeSpan timeout,
            int repeats = 0,
            bool failedBefore = false,
            RewriteWatch? rewrites = null,
            AutoInstallOutcome installedFor = default,
            bool installsOk = true,
            bool fileToolsAvailable = false)
        {
            if (!run.Started)
            {
                // Scrubbed like every other path this tool hands back. A launch failure
                // reports an OS message about the HOST's own scratch — a file the model
                // never wrote and cannot reach — and the absolute path in it is the part a
                // model fixates on. The 2026-09-10 incident was six rounds of a model
                // reasoning about "the host's state directory" from exactly this string.
                string reason = OutputPaths.Scrub(
                    run.Error ?? "the shell could not be started", workspace, session.CurrentDirectory);

                // From the second in a row, say the thing the model cannot work out for
                // itself: the command is not what is wrong, so rewriting it is not the way
                // out. Named alternatives rather than advice, because a tool result that
                // only describes a problem produces another round of the same command.
                int inARow = session.RecordNotStarted();
                if (inARow >= 2)
                {
                    reason += $"\nThis is launch {inARow} in a row that failed before running anything, so the "
                        + "problem is this host's shell and not the command — simplifying it will not help. "
                        + "Use read_file for context, write_file to create new files, and apply_patch to modify "
                        + "one or multiple files; these tools do not need the shell. Use those tools "
                        + "or a skill's own script through skills_run; if the task cannot be done without a "
                        + "shell, say so in your answer instead of trying again.";
                }
                return CodeExecResult.Refused(reason);
            }

            session.RecordStarted();

            // A command whose exit 1 IS its answer. See ShellCommand.ExitCodeIsBenign: a
            // no-match grep, a clean-tree `git diff --quiet`, a false `test`, a `diff`
            // that found a difference — all correct commands, all reported as failures
            // until now, each one manufacturing a recovery round out of nothing.
            bool benign = !run.TimedOut && ShellCommand.ExitCodeIsBenign(command, run.ExitCode);

            var sb = new StringBuilder();

            foreach (string note in notes)
                sb.Append(note).Append('\n');

            // An install the command asked for did not happen. The line still RAN — the
            // install was substituted with `false`, which is what keeps `&&` and `||`
            // meaning what the model wrote — so the shell's exit code is reported as it
            // is. But the CALL did not do what was asked, and a result that opened with
            // "exit 0" was this codebase's cardinal defect in its purest form: the logged
            // case was `pip3 install python-pptx 2>&1 | tail -5`, where the install was
            // refused, `false | tail -5` ran, and the model was told exit 0.
            if (!installsOk)
            {
                sb.Append("The install above did not happen, so this command did NOT do what you asked. "
                        + "The exit status below is from the rest of the line, which ran without it.\n");
            }

            if (run.TimedOut)
            {
                sb.Append("The command did not finish within ")
                  .Append(timeout.TotalSeconds.ToString("0.#", CultureInfo.InvariantCulture))
                  .Append("s and was stopped. Its output up to that point is below.\n")
                  .Append("If it simply needs longer, run it again with a bigger timeout_ms (up to ")
                  .Append(((int)_options.EffectiveMaxTimeout.TotalMilliseconds).ToString(CultureInfo.InvariantCulture))
                  .Append("). If it is meant to keep running — a server, a watcher — start it with "
                        + "run_in_background instead and read its log.\n");
            }
            else
            {
                sb.Append("exit ").Append(run.ExitCode.ToString(CultureInfo.InvariantCulture))
                  .Append(" (").Append(run.Elapsed.TotalSeconds.ToString("0.##", CultureInfo.InvariantCulture))
                  .Append("s, sandbox: ").Append(run.SandboxName).Append(')');
                if (benign)
                {
                    // Said out loud, because "exit 1" on its own reads as broken to a model
                    // that has seen ten thousand shell transcripts. The exit code is still
                    // reported exactly as it was — this adds the reading, it does not hide
                    // the number.
                    sb.Append(" — which for this command means \"no match / no difference\", not an error");
                }
                sb.Append('\n');
            }

            string directory = session.CurrentDirectoryLabel;
            if (directory != ".")
                sb.Append("Working directory is now ").Append(directory).Append("\n");

            if (session.TakeEnvironmentWasReset())
            {
                sb.Append("Note: this session's saved environment could not be read back and was reset, "
                        + "so variables you exported earlier are gone. A value containing a newline does "
                        + "this; export those into a file instead. Re-export what you still need.\n");
            }

            // Say what was NOT confined. The model is about to act on this output, and
            // "the command could have reached the network" is a materially different
            // situation from "it could not".
            // What was ENFORCED for THIS run, not what this host is capable of.
            //
            // A sandbox whose TryWrap fails in any mode but Required leaves the process to
            // start RAW while `_sandbox` stays non-null — which is every run under
            // --code-exec-unconfined and every Windows run, since CanRun forces Preferred
            // there. Reading the capabilities off the object then reported an empty gap
            // list, so the "Not confined on this host" line was omitted ENTIRELY and the
            // model was told, by silence, that its command had been confined. That is the
            // opposite of the truth about whether the command could have reached the
            // network, and it is the one fact a model must be able to trust here.
            SkillSandboxCapabilities? enforcedCapabilities =
                string.Equals(run.SandboxName, "none", StringComparison.Ordinal)
                    ? null
                    : _sandbox?.Capabilities;
            IReadOnlyList<string> gaps = CommandGaps(
                enforcedCapabilities, includeNetwork: !_options.AllowNetwork);
            if (gaps.Count > 0)
                sb.Append("Not confined on this host: ").Append(string.Join("; ", gaps)).Append(".\n");
            if (_options.AllowNetwork)
            {
                sb.Append("Network access is intentionally ENABLED by ")
                  .Append(CodeExecOptions.AllowNetworkFlag)
                  .Append(": commands may reach the network without restriction, including host IP networks.\n");
            }

            string output = Merge(run.Stdout, run.Stderr);
            if (output.Length > 0)
                sb.Append('\n').Append(output);
            else if (!run.TimedOut)
                sb.Append("\n(no output)\n");

            // Nothing to coach and nothing to warn about when the command succeeded in
            // its own terms: a no-match grep is an answer, and telling the model it has
            // "already run this command and it failed each time" about a search that keeps
            // correctly finding nothing is the false-failure defect one layer along.
            if (!benign)
            {
                AppendCoaching(
                    sb, run, command, workspace, session.CurrentDirectory, installedFor,
                    fileToolsAvailable);
                AppendRepeatWarning(sb, run, repeats, failedBefore);
            }

            // A whole file re-typed to change two lines of it. Said here, right after the
            // act and with this file's real numbers, because a declaration saying the same
            // thing in general terms was measured to have no effect: apply_patch was used
            // zero times out of ten opportunities while it said exactly this.
            //
            // The note describes apply_patch's patch envelope and contextual hunks.
            // Its unordered line comparison is evidence of wasted work, not a ready-to-send patch.
            // Not gated on success. The whole file was re-typed either way, and a failing
            // run is exactly where the model is looping and re-emitting.
            if (rewrites?.Describe(workspace, ShellTools.PatchToolName) is { Length: > 0 } retyped)
            {
                sb.Append(retyped);
                _logger.LogInformation(LogEventIds.CodeExecRewrote,
                    "codeexec.rewrote via=shell");
            }

            // A command that exited 0 having written a file it never ran is the one case
            // where nothing in the result contradicts "done". See SyntaxCheck: a FAILED
            // command has already printed its own SyntaxError, so this is deliberately
            // gated on success, and the file list comes from the command's own redirects
            // rather than from a second walk of the work directory.
            IReadOnlyList<string> written = (run.Ok || benign)
                ? SyntaxCheck.RedirectTargets(command).Select(t => t.Path).ToList()
                : Array.Empty<string>();
            if (written.Count > 0)
            {
                if (_syntax.Verify(written, workspace, session.CurrentDirectory) is { Length: > 0 } broken)
                {
                    sb.Append('\n').Append(broken);
                }
                else
                {
                    // Only once the parse has PASSED, so this can never contradict the
                    // message above it. Claude Code carries the same clause on 98.4% of
                    // its successful edits, and its Read tool states the reason: a
                    // re-read to verify is waste, because the write would have errored.
                    sb.Append("\n").Append(written.Count == 1 ? written[0] : "Those files")
                      .Append(written.Count == 1 ? " now holds" : " now hold")
                      .Append(" exactly what you wrote, and it parses — no need to read it back.\n");
                }
            }

            IReadOnlyList<CodeArtifact> artifacts = CaptureArtifacts(workspace, before, out IReadOnlyList<string> skipped);
            AppendArtifacts(sb, artifacts, skipped);

            return new CodeExecResult(
                // Not Ok when an install was refused, whatever the residual line exited
                // with. `Ok` is what the caller renders as success and what the loop logs.
                (run.Ok || benign) && installsOk, sb.ToString(), artifacts,
                artifacts.Count > 0 ? artifacts[0].RunId : string.Empty);
        }

        /// <summary>
        /// stdout then stderr, in that order and unlabelled.
        ///
        /// <para>
        /// Unlabelled because that is what a terminal shows and what every example the
        /// model has ever seen looks like; a run whose useful output is a traceback should
        /// not read as though something separate went wrong. Codex merges them the same
        /// way, and appends a newline between only when stdout does not already end in one.
        /// </para>
        /// </summary>
        private static string Merge(string stdout, string stderr)
        {
            if (stderr.Length == 0)
                return stdout;
            if (stdout.Length == 0)
                return stderr;
            return stdout.EndsWith('\n') ? stdout + stderr : stdout + "\n" + stderr;
        }

        /// <summary>
        /// Turn a failure into the next command to type.
        ///
        /// <para>
        /// Every branch here exists because a model got stuck on exactly this and the
        /// bare output was not enough. A traceback names the module but says nothing about
        /// this host's empty environment, so the model runs the same code again; a
        /// missing program and a missing library both surface as a number; and an install
        /// blocked by the egress proxy arrives as an opaque connection error with no hint
        /// that a wall exists.
        /// </para>
        /// </summary>
        private void AppendCoaching(
            StringBuilder sb, ConfinedResult run, string command,
            SessionWorkspace workspace, string? ranIn, AutoInstallOutcome installedFor,
            bool fileToolsAvailable)
        {
            if (run.Ok || run.TimedOut)
                return;

            string diagnosis = run.Stderr.Length > 0 ? run.Stderr : run.Stdout;

            if (CodeDiagnostics.TryFindMissingModule(diagnosis, out CodeLanguage language, out string module))
            {
                // The host may already have installed a package for this very import and
                // re-run the command — see RunWithAutoInstall. If the import is STILL
                // failing after that, "install it and run the command again" is advice the
                // model would follow into the same wall, so it is not given. What is
                // actually wrong in that case is the NAME: the module and the package
                // that provides it differ, and only the model knows which one it wanted.
                //
                // Carried as a SET rather than read back out of the notes. Matching the
                // note's prose could not tell "installed, re-ran, still missing" from
                // "the install itself failed", and a message that says a package was
                // installed when none was is the constraint-stated-wrongly defect this
                // codebase has already been bitten by. A failed install has put its own
                // reason in the notes and needs nothing added here.
                bool installedAndStillMissing = installedFor.WasInstalled(module);
                bool installFailed = installedFor.InstallFailed(module);

                sb.Append("\n'").Append(module).Append("' is not installed. ");
                if (!_options.AllowInstall)
                {
                    sb.Append("Installing packages is not enabled on this host, so use only what is already "
                            + "available, or say in your answer that this step needs a package the host does "
                            + "not provide.\n");
                }
                else if (installedAndStillMissing)
                {
                    // "Called something else" is only true when the module was never
                    // found. When the failure came from INSIDE the installed package —
                    // an ABI mismatch, a missing native library, an ESM resolution gap —
                    // the name was right and this sentence sends the model to fix a
                    // spelling that is already correct.
                    sb.Append(CodeDiagnostics.ClassifyFailure(diagnosis).Source
                            == CodeDiagnostics.FailureSource.Environment
                        ? "A package of that name was installed and the command was run again, and it "
                          + "still fails — the package is present, so its NAME is not the problem. "
                          + "Something about the installed copy does not work on this host; use a "
                          + "different library, or say in your answer that this step cannot be done here.\n"
                        : "A package of that name was installed and the command was run again, and the "
                          + "import still failed — so the package that provides '" + module
                          + "' is called something else. Install it by its real distribution name and "
                          + "run the command again.\n");
                }
                else if (installFailed)
                {
                    // The reason is already in the notes above, verbatim from the
                    // installer. Repeating "install it and run again" here would be
                    // telling the model to redo the thing that just refused.
                    sb.Append("The host tried to install it for you and could not, for the reason given "
                            + "above. Use something already available, or say in your answer that this "
                            + "step needs a package the host cannot supply.\n");
                }
                else
                {
                    sb.Append("Install it and run the command again:\n  ")
                      .Append(CodeDiagnostics.InstallCommandFor(language, module)).Append('\n');
                }
            }

            if (CodeDiagnostics.MissingCommand(diagnosis) is { } absent)
            {
                if (CodeDiagnostics.IsPackageManager(absent))
                {
                    // Not "this host has no installer" — it has one, spelled differently.
                    // A model told to run `pip install` on a machine whose only Python is
                    // Apple's has followed its instructions exactly and has nowhere to go.
                    sb.Append("\n'").Append(absent).Append("' is not on this host's PATH under that name. ")
                      .Append("Install Python packages with:\n  ")
                      .Append(CodeDiagnostics.PythonInstallPrefix()).Append(" <package>\n");
                }
                else if (CodeDiagnostics.SpelledDifferentlyHere(absent) is { } spelling)
                {
                    // The host was actively steering the model away from a one-character
                    // fix. `python` is not a package manager, so it fell into the branch
                    // below and was answered "no package manager here can supply it — do
                    // the step another way" — on a host whose own shell description says
                    // `python3 (3.9.6)` two paragraphs earlier. Five incidents and nine
                    // rounds in the logs. There is also a shim now
                    // (EnsureInterpreterAliases), so this branch is the belt to its
                    // braces: it fires only where the shim could not be written.
                    sb.Append("\n'").Append(absent).Append("' is not on this host under that name — here it is ")
                      .Append("called ").Append(spelling).Append(". Run the same command with:\n  ")
                      .Append(spelling).Append('\n');
                }
                else
                {
                    sb.Append("\n'").Append(absent).Append("' is not on this host. If it is a PROGRAM rather than a "
                            + "library (ffmpeg, pandoc, pdftoppm), no package manager here can supply it — do the "
                            + "step another way, or say in your answer that it needs ").Append(absent)
                      .Append(" installed on the host. If you meant a library, install it first.\n");
                }
            }

            if (NetworkWasConfined(run) && LooksLikeNetworkFailure(diagnosis))
            {
                // The single most common wrong conclusion a model draws here is "the host
                // is offline, so I will implement the protocol by hand".
                sb.Append("\nCommands here have no Internet/IP network access. The only "
                        + "thing that reaches a registry is a package install, and the host performs those "
                        + "itself from the names you give. So bring what you need in by installing it; "
                        + "there is no way to fetch a URL or call an API from a command.\n");
            }

            // The API the model guessed at, read out of the package that is already
            // installed. Last, because it is the only branch that costs a process — and
            // it runs at most once, on a failure that named a member which does not
            // exist. See ApiProbe: this is the failure that dominates the round budget.
            if (CodeDiagnostics.TryFindApiMiss(diagnosis, out CodeDiagnostics.ApiMiss miss)
                && _apiProbe.Explain(miss, command, workspace, ranIn) is { Length: > 0 } api)
            {
                sb.Append('\n').Append(api);
                if (!api.EndsWith('\n'))
                    sb.Append('\n');
            }

            AppendWhoseFaultThisIs(sb, run, diagnosis);

            // Only name apply_patch when this request was actually offered
            // it. A null-workspace caller has shell alone and must never be instructed
            // to invoke a tool it cannot send.
            if (fileToolsAvailable
                && CodeRepairHint.Create(diagnosis, workspace, ranIn, NetworkWasConfined(run))
                    is { Length: > 0 } repair)
            {
                sb.Append(repair);
            }
        }

        /// <summary>
        /// Say when a failure came from the ENVIRONMENT, because the model's next move
        /// otherwise is to rewrite a program that was never the problem.
        ///
        /// <para>
        /// This is the reference implementation's own standing instruction — <i>"fix the
        /// problem at the root cause rather than applying surface-level patches"</i> — made
        /// answerable. In this server's logs the model could not tell what the root cause
        /// WAS: handed a host failure it re-emitted 15,000 characters of program and then
        /// switched language, and re-typing a program costs about 24 times what re-reading
        /// it from the prompt costs. So the most expensive wrong turn available to the loop
        /// is editing code over a failure the code did not cause.
        /// </para>
        /// <para>
        /// <b>It never says the code is correct.</b> A program can carry a bug and hit a
        /// missing library on the same run, and a host that says "your code is fine" when
        /// it is not has sent the model to re-run something broken — which is worse than
        /// silence, and is the defect class this codebase calls cardinal. What it says is
        /// only what the output actually establishes: this failure did not come from the
        /// code, so changing the code will not change it.
        /// </para>
        /// <para>
        /// Nothing is said for a case the HOST can fix. Those are already handled by
        /// doing it — installing the package and re-running, shimming the interpreter
        /// name — and a sentence about a problem that has already been dealt with is a
        /// sentence that makes the model look for something that is no longer there.
        /// </para>
        /// </summary>
        private void AppendWhoseFaultThisIs(StringBuilder sb, ConfinedResult run, string diagnosis)
        {
            bool networkConfined = NetworkWasConfined(run);

            CodeDiagnostics.FailureCause cause = CodeDiagnostics.ClassifyFailure(
                diagnosis, CodeLanguage.Unknown, networkConfined);
            if (cause.Source != CodeDiagnostics.FailureSource.Environment || cause.HostCanFix)
                return;

            sb.Append("\nThis is the ENVIRONMENT, not the code you wrote: ").Append(cause.Reason)
              .Append(". Editing or re-typing the program will not change it — and re-typing a "
                    + "working program is how a second bug appears in code that was already right. "
                    + "Either do this step a different way, or say in your answer that it needs "
                    + "something this host does not have.\n");
        }

        /// <summary>
        /// Say when a command has already been run, and already failed, in this session.
        ///
        /// <para>
        /// <b>A model will re-send a byte-identical failing command until the budget runs
        /// out.</b> The worst turn in this server's logs was asked to compute
        /// <c>17*23+5</c> and write it to a file; the model wrote one line of invalid
        /// Python and sent it NINE times — eight byte-identical, the ninth differing only
        /// in spaces around the operators — spending the entire round budget and making no
        /// progress at all on one line of code.
        /// </para>
        /// <para>
        /// The reason it could keep going is worth stating precisely, because it is a
        /// property of the RESULT and not of the model: every one of those nine results
        /// was identical except for the scratch filename in the traceback
        /// (<c>main-c50c0b90.py</c>), which changed each time. A single differing token in
        /// otherwise identical text reads as new information. Rewriting host paths out of
        /// the output (see <see cref="OutputPaths"/>) removes that false signal — and this
        /// note replaces it with a true one.
        /// </para>
        /// <para>
        /// Only for a command that failed before AND failed again. A command re-run after
        /// a fix, or one that is meant to be run repeatedly, says nothing.
        /// </para>
        /// </summary>
        private static void AppendRepeatWarning(
            StringBuilder sb, ConfinedResult run, int repeats, bool failedBefore)
        {
            if (run.Ok || repeats < 1 || !failedBefore)
                return;

            sb.Append("\nYou have already run this command ")
              .Append(repeats == 1
                  ? "once before in this conversation"
                  : repeats.ToString(CultureInfo.InvariantCulture) + " times before in this conversation")
              .Append(" and it failed each time. Sending it again will fail again — the command is the "
                    + "problem, not the run. Change what the command does, or say in your answer which "
                    + "part you could not do and why.\n");
        }

        /// <summary>
        /// Delegated rather than duplicated. This was a private copy of the same list the
        /// failure classifier needs, and two copies of a signature table is two copies
        /// that stop agreeing — the rule CodeDiagnostics already states about its regexes.
        /// </summary>
        /// <summary>
        /// Whether the sandbox that actually ran this command blocks its IP network.
        /// False for a raw/degraded run and on Windows, where a job object does not
        /// confine sockets. This must use the result's sandbox name rather than the
        /// detected object: preferred execution can fall back after declaration time.
        /// </summary>
        private bool NetworkWasConfined(ConfinedResult run) =>
            !_options.AllowNetwork
            && !string.Equals(run.SandboxName, "none", StringComparison.Ordinal)
            && _sandbox?.Capabilities.ConfinesNetwork == true;

        private static bool LooksLikeNetworkFailure(string text) =>
            CodeDiagnostics.LooksLikeNetworkAttempt(text);

        private IReadOnlyList<CodeArtifact> CaptureArtifacts(
            SessionWorkspace workspace,
            Dictionary<string, (long Length, DateTime WriteTime)> before,
            out IReadOnlyList<string> skipped)
        {
            skipped = Array.Empty<string>();
            if (_artifacts == null)
                return Array.Empty<CodeArtifact>();

            string runId = Guid.NewGuid().ToString("N").Substring(0, 16);
            IReadOnlyList<CodeArtifact> captured = _artifacts.Capture(
                runId,
                workspace.WorkDirectory,
                PointerFor,
                out skipped,
                relative => workspace.IsUnchangedSince(before, relative));

            if (captured.Count > 0)
            {
                _logger.LogInformation(LogEventIds.CodeExecArtifacts,
                    "codeexec.artifacts run={RunId} files={Files}", runId, captured.Count);
            }
            return captured;
        }

        /// <summary>
        /// The download URL handed to the model.
        ///
        /// <para>
        /// Each path SEGMENT is escaped, never the whole relative path:
        /// <see cref="Uri.EscapeDataString"/> escapes <c>/</c> itself, so a nested artifact
        /// came out as <c>out%2Freport.pdf</c> while the listing endpoint built the same URL
        /// with a raw separator. The result tells the model to copy these links into its
        /// answer verbatim, so whichever spelling was wrong is the one the user clicked.
        /// </para>
        /// </summary>
        private string PointerFor(string runId, string relative, string fullPath) =>
            string.IsNullOrEmpty(_options.ArtifactUriPrefix)
                ? fullPath
                : CodeArtifactStore.UrlFor(_options.ArtifactUriPrefix!, runId, relative);

        private void AppendArtifacts(
            StringBuilder sb, IReadOnlyList<CodeArtifact> artifacts, IReadOnlyList<string> skipped)
        {
            if (artifacts.Count > 0)
            {
                // The model is the one that tells the user their file is ready, so it needs
                // the address, not just the name. On a server the address is a URL and the
                // model gets it PRE-FORMATTED as a markdown link: told merely to "give the
                // link to the user", a small model wraps the URL in backticks, which the Web
                // UI renders as dead text the user cannot click.
                bool asUrl = !string.IsNullOrEmpty(_options.ArtifactUriPrefix);
                sb.Append(asUrl
                    ? "\nFiles produced. The user downloads them through these links - copy the markdown links below into your answer verbatim:\n"
                    : "\nFiles produced (tell the user where they are on disk):\n");
                foreach (CodeArtifact artifact in artifacts)
                {
                    if (asUrl)
                    {
                        sb.Append("- [").Append(artifact.Path).Append("](")
                          .Append(artifact.Pointer).Append(") (")
                          .Append(ShellCommand.FormatBytes(artifact.Bytes)).Append(")\n");
                    }
                    else
                    {
                        sb.Append("- ").Append(artifact.Path)
                          .Append(" (").Append(ShellCommand.FormatBytes(artifact.Bytes)).Append(") -> ")
                          .Append(artifact.Pointer).Append('\n');
                    }
                }
            }

            if (skipped.Count > 0)
            {
                sb.Append("\nNot kept:\n");
                foreach (string reason in skipped)
                    sb.Append("- ").Append(reason).Append('\n');
            }
        }

        private SessionWorkspace CreateEphemeral()
        {
            string root = Path.Combine(
                _options.ScratchDirectory ?? Path.GetTempPath(),
                "ts-shell-" + Guid.NewGuid().ToString("N").Substring(0, 12));
            return new SessionWorkspace(root);
        }

        private static void TryDeleteDirectory(string directory)
        {
            try
            {
                if (Directory.Exists(directory))
                    Directory.Delete(directory, recursive: true);
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { }
        }

        /// <summary>Stop background jobs and release per-session caches owned by this runner.</summary>
        public void Dispose()
        {
            foreach (BackgroundJobs jobs in _jobs.Values)
                jobs.Dispose();
            _jobs.Clear();
            _sessions.Clear();
            _sanitizedCertificateBundles.Clear();
        }
    }
}
