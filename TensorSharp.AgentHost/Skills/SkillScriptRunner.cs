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
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Threading;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp.AgentHost.CodeExec;
using TensorSharp.Runtime.Logging;

namespace TensorSharp.AgentHost.Skills
{
    /// <summary>
    /// Runs a skill's bundled script as a confined child process.
    ///
    /// <para>
    /// A skill is content somebody supplied — uploaded as a ZIP, or pulled off GitHub
    /// into a directory the operator pointed at — and the decision to run one of its
    /// scripts is made by a model reading that same person's Markdown. So the question
    /// is not whether to trust it but what it can reach when it runs.
    /// </para>
    /// <para>
    /// Two layers answer that. <b>In process, always:</b> the path resolves through
    /// <see cref="SkillPathGuard"/> so only files inside the skill can be named; the
    /// interpreter comes from an allow-list rather than from a shebang; no shell is
    /// involved, so arguments are data and never syntax; the environment is scrubbed of
    /// inherited credentials; the working directory is a fresh scratch directory rather
    /// than the skill; stdin is closed; and the run is bounded in time and in captured
    /// output. <b>In the OS, when it can be:</b> <see cref="ISkillSandbox"/> confines
    /// the child so it cannot reach the network, cannot read the user's home directory
    /// (credentials, SSH keys, every other installed skill), and cannot write anywhere
    /// but its scratch directory.
    /// </para>
    /// <para>
    /// With <see cref="SkillSandboxMode.Required"/> — the default — a host that has no
    /// sandbox refuses to run scripts at all and says so, rather than quietly running
    /// them unconfined. That is the whole point of the mode existing: "isolation was
    /// unavailable" must not degrade into "isolation was skipped".
    /// </para>
    /// </summary>
    public sealed class SkillScriptRunner : ISkillScriptRunner
    {
        private readonly SkillScriptRunnerOptions _options;
        private readonly ILogger _logger;
        private readonly IShellBackend _backend;
        private readonly ISkillSandbox? _sandbox;

        /// <summary>
        /// Latch for the once-per-process "running unconfined" warning below: the host
        /// state it reports cannot change while the process lives, so repeating it per
        /// script would only bury it.
        /// </summary>
        private static int s_unconfinedHostWarned;

        public SkillScriptRunner(SkillScriptRunnerOptions? options = null, ILogger? logger = null)
        {
            _options = options ?? new SkillScriptRunnerOptions();
            _logger = logger ?? NullLogger.Instance;
            // The same seam the shell tool launches through. The default is today's
            // confined child process under the detected OS sandbox; a host that cannot
            // start processes supplies its own backend, and the confinement questions
            // below are answered from THAT backend's sandbox — which is what lets an
            // in-process runtime with honest capabilities satisfy `required`.
            _backend = _options.Backend ?? ProcessShellBackend.Detect(_options.Sandbox, _logger);
            _sandbox = _options.Sandbox == SkillSandboxMode.Off ? null : _backend.Sandbox;
        }

        /// <summary>The sandbox in force, or null when running unconfined.</summary>
        public ISkillSandbox? Sandbox => _sandbox;

        /// <summary>What runs each script: a confined child process, or the host's own runtime.</summary>
        public IShellBackend Backend => _backend;

        /// <summary>
        /// True when this runner will actually run anything. False when the host
        /// demanded a sandbox and none is available — in which case
        /// <see cref="UnavailableReason"/> says so.
        /// </summary>
        /// <summary>
        /// Whether a script may run at all here.
        ///
        /// <para>
        /// Under <see cref="SkillSandboxMode.Required"/> this asks whether the host
        /// actually CONFINES a script, not merely whether an <see cref="ISkillSandbox"/>
        /// object exists. The two came apart on Windows:
        /// <see cref="SkillSandboxWindows"/>'s <c>IsAvailable</c> is unconditionally true
        /// because a job object can always be created, and a job object bounds CPU,
        /// memory and process count but cannot restrict a single file or socket — which
        /// its own <see cref="ISkillSandbox.Capabilities"/> says plainly. Testing only
        /// for existence therefore let <c>required</c> — the DEFAULT — behave exactly
        /// like <c>preferred</c> there, quietly running scripts with full filesystem and
        /// network access on the setting whose entire promise is "sandbox or refuse".
        /// </para>
        /// </summary>
        public bool CanRun =>
            _options.Sandbox != SkillSandboxMode.Required || Confines(_sandbox);

        /// <summary>
        /// The confinement <c>required</c> actually requires: keeping a script out of the
        /// rest of the filesystem and off the network. Resource caps are welcome but are
        /// not isolation, so a sandbox that only bounds them does not qualify.
        /// </summary>
        private static bool Confines(ISkillSandbox? sandbox) =>
            sandbox is not null
            && sandbox.Capabilities.ConfinesWrites
            && sandbox.Capabilities.ConfinesNetwork;

        /// <summary>Why <see cref="CanRun"/> is false, or null.</summary>
        public string? UnavailableReason
        {
            get
            {
                if (CanRun)
                    return null;

                // Name what is missing rather than claiming there is nothing at all: on
                // Windows there IS a sandbox, it just does not confine, and an operator
                // reading "no OS sandbox" would go looking for one that does not exist.
                if (_sandbox is { } present)
                {
                    SkillSandboxCapabilities caps = present.Capabilities;
                    var gaps = new List<string>();
                    if (!caps.ConfinesWrites) gaps.Add("filesystem writes");
                    if (!caps.ConfinesNetwork) gaps.Add("network access");
                    if (!caps.ConfinesHomeReads) gaps.Add("reads of your home directory");

                    return $"this host's sandbox ({present.Name}) cannot confine "
                        + string.Join(", ", gaps)
                        + ", and skill scripts are configured to run only when they can be confined"
                        + " - pass --skills-sandbox preferred to accept that and run them anyway";
                }

                // iOS has no OS sandbox to look for: code runs inside the app, and what
                // is missing is a backend presenting an in-process runtime that confines.
                if (OperatingSystem.IsIOS())
                {
                    return "this host runs skill scripts in an in-process runtime, and no backend presenting "
                        + "one that confines writes and the network was supplied, and skill scripts are "
                        + "configured to run only when they can be confined";
                }

                return "this host provides no OS sandbox (checked: "
                    + (OperatingSystem.IsMacOS() ? "sandbox-exec"
                       : OperatingSystem.IsLinux() ? "bubblewrap 0.12.0 or newer (install/update bwrap to enable it)"
                       : OperatingSystem.IsWindows() ? "windows job object"
                       : "none for this platform")
                    + "), and skill scripts are configured to run only when they can be confined";
            }
        }

        /// <inheritdoc />
        public SkillToolResult Run(
            Skill skill, string relativePath, IReadOnlyList<string> arguments,
            Action<string>? onOutput = null, IReadOnlyList<string>? packages = null)
        {
            ArgumentNullException.ThrowIfNull(skill);

            // The same workspace also backs model-written shell calls. Keep the whole
            // dependency/setup/run cycle atomic with respect to a replacement turn; the
            // lease is re-entrant when auto-install calls the code runner below.
            using IDisposable? execution = _options.Workspace?.EnterExecution();

            if (!CanRun)
                return SkillToolResult.Failure(UnavailableReason!);

            if (!SkillPathGuard.TryResolveExistingFile(skill.RootDirectory, relativePath, out string? scriptPath, out string? guardError))
                return SkillToolResult.Failure(ExplainUnresolvedScript(skill, relativePath, guardError));

            string normalized = SkillPathGuard.ToSkillRelative(skill.RootDirectory, scriptPath!);
            string extension = Path.GetExtension(normalized);

            if (!TryResolveInterpreter(extension, out string? interpreter, out string? interpreterError))
                return SkillToolResult.Failure($"Cannot run '{normalized}': {interpreterError}");

            // The session workspace, when the host supplies one, is the whole point of
            // running a skill's scripts at all: generate.py writes the pptx there,
            // validate.py finds it there a call later, and both import the packages the
            // session installed. Without one, the script gets the classic per-run
            // scratch that dies with the call.
            string workDirectory;
            bool perCallScratch = _options.Workspace == null;
            try
            {
                // The whole layout, not just the work directory: installs land in env/ and
                // repair overlays in state/, and a workspace removed from underneath a live
                // conversation takes all three. See SessionWorkspace.EnsureDirectories.
                _options.Workspace?.EnsureDirectories();
                workDirectory = _options.Workspace?.WorkDirectory
                    ?? Path.Combine(
                        _options.ScratchDirectory ?? Path.GetTempPath(),
                        "ts-skill-" + Guid.NewGuid().ToString("N"));
                Directory.CreateDirectory(workDirectory);
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                return SkillToolResult.Failure($"a scratch directory for '{normalized}' could not be created: {ex.Message}");
            }

            var setupNotes = new List<string>();

            // If something removed this conversation's working directory, the script is
            // about to look for inputs that are no longer there. Say so once, wherever the
            // model looks next — the latch means the shell tool and this one cannot both
            // claim it, and neither repeats it.
            if (_options.Workspace?.ConsumeRebuiltNotice() == true)
                setupNotes.Add(SessionWorkspace.RebuiltNote);

            string? installLanguage = InstallLanguageFor(extension);

            // Dependencies the model named up front, and any requirements.txt the skill
            // ships (root or the script's own directory, once per session): both go into
            // the session environment BEFORE the first attempt.
            if (installLanguage != null && CanAutoInstall(installLanguage))
            {
                if (packages is { Count: > 0 })
                    InstallInto(installLanguage, packages, setupNotes, onOutput, "requested");

                foreach ((string key, IReadOnlyList<string> required) in RequirementsFor(skill, normalized))
                {
                    if (!_options.Workspace!.TryMarkApplied(key))
                        continue;
                    InstallInto(installLanguage, required, setupNotes, onOutput, key);
                }
            }

            try
            {
                return RunWithAutoInstall(
                    skill, normalized, scriptPath!, interpreter!, arguments,
                    workDirectory, onOutput, installLanguage, setupNotes);
            }
            finally
            {
                if (perCallScratch && _options.DeleteScratchDirectory)
                    TryDeleteDirectory(workDirectory);
            }
        }

        private bool CanAutoInstall(string language) =>
            _options.Workspace != null
            && _options.PackageInstaller is { } installer
            && installer.CanInstallPackagesFor(language);

        /// <summary>The install language for a script extension, or null.</summary>
        private static string? InstallLanguageFor(string extension) =>
            extension.ToLowerInvariant() switch
            {
                ".py" => "python",
                ".js" or ".mjs" => "javascript",
                _ => null,
            };

        private void InstallInto(
            string language, IReadOnlyList<string> packages, List<string> notes,
            Action<string>? onOutput, string what)
        {
            string? error = _options.PackageInstaller!.InstallPackages(
                language, packages, _options.Workspace!, onOutput);
            notes.Add(error == null
                ? $"Set up dependencies ({what}): {string.Join(", ", packages)}"
                : $"Could not install {what} dependencies ({string.Join(", ", packages)}): {error}");
        }

        /// <summary>
        /// The dependency lists a skill declares: a <c>requirements.txt</c> at its root
        /// and one beside the script, keyed for once-per-session application. Lines are
        /// taken conservatively — bare names and version pins only; options, includes
        /// and URLs are for pip invocations this host deliberately does not make.
        /// </summary>
        private IEnumerable<(string Key, IReadOnlyList<string> Packages)> RequirementsFor(Skill skill, string normalized)
        {
            string? scriptDir = Path.GetDirectoryName(normalized)?.Replace('\\', '/');
            var candidates = new List<string> { "requirements.txt" };
            if (!string.IsNullOrEmpty(scriptDir))
                candidates.Add(scriptDir + "/requirements.txt");

            foreach (string candidate in candidates.Distinct())
            {
                if (!SkillPathGuard.TryResolveExistingFile(skill.RootDirectory, candidate, out string? path, out _))
                    continue;

                List<string> packages;
                try
                {
                    packages = File.ReadLines(path!)
                        .Select(line => line.Split('#')[0].Trim())
                        .Where(line => line.Length > 0 && !line.StartsWith('-') && !line.Contains("://"))
                        .Take(16)
                        .ToList();
                }
                catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
                {
                    continue;
                }

                if (packages.Count > 0)
                    yield return ($"requirements:{skill.Id}:{candidate}", packages);
            }
        }

        /// <summary>
        /// Run the script; when it dies on a missing import and this host can install,
        /// install the module and run it again. This is what "the skill's scripts just
        /// work" means in practice: the pptx validator needs defusedxml and lxml, no
        /// interpreter ships them, and without this loop the model spends whole rounds
        /// discovering that one import at a time.
        /// </summary>
        private SkillToolResult RunWithAutoInstall(
            Skill skill, string normalized, string scriptPath, string interpreter,
            IReadOnlyList<string> arguments, string workDirectory, Action<string>? onOutput,
            string? installLanguage, List<string> setupNotes)
        {
            var attempted = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            SkillToolResult result;

            while (true)
            {
                result = RunConfined(skill, normalized, scriptPath, interpreter, arguments,
                    workDirectory, onOutput, out string stderrText);

                if (result.Ok || installLanguage == null || !CanAutoInstall(installLanguage)
                    || attempted.Count >= Math.Max(1, _options.MaxAutoInstallAttempts))
                    break;

                CodeExec.CodeLanguage language = CodeExec.CodeExecOptions.ParseLanguage(installLanguage);
                string? missing = CodeExec.CodeDiagnostics.MissingModule(language, stderrText);
                if (missing == null)
                    break;

                string package = CodeExec.CodeDiagnostics.InstallNameFor(language, missing);
                if (!attempted.Add(package))
                    break;      // installing it did not make the import work; stop

                Tap(onOutput, $"[installing missing dependency: {package}]");
                string? error = _options.PackageInstaller!.InstallPackages(
                    installLanguage, new[] { package }, _options.Workspace!, onOutput);
                if (error != null)
                {
                    setupNotes.Add($"Could not auto-install '{package}': {error}");
                    break;
                }

                setupNotes.Add($"Auto-installed missing dependency: {package}");
                Tap(onOutput, $"[re-running {normalized}]");
            }

            if (setupNotes.Count == 0)
                return result;

            // The model reads what happened in order: setup first, then the run.
            return result with { Content = string.Join("\n", setupNotes) + "\n\n" + result.Content };
        }

        private SkillToolResult RunConfined(
            Skill skill,
            string normalized,
            string scriptPath,
            string interpreter,
            IReadOnlyList<string> arguments,
            string workDirectory,
            Action<string>? onOutput,
            out string stderrText)
        {
            stderrText = string.Empty;

            // No shell: the interpreter and the script path lead the argument vector, so
            // a path or an argument containing ; | > $ ` is data, not syntax.
            var argv = new List<string> { interpreter, scriptPath };
            if (arguments != null)
                argv.AddRange(arguments);

            // The session's package environment must be readable inside the sandbox, or
            // PYTHONPATH points at a directory the seatbelt profile denies.
            IReadOnlyList<string> readablePaths = _options.ReadablePaths;
            if (_options.Workspace is { } workspace)
            {
                var extended = new List<string>(readablePaths) { workspace.EnvDirectory };
                readablePaths = extended;
            }

            // What was already in the shared directory before this script ran, so the
            // result reports what THIS run produced rather than the session's history.
            Dictionary<string, (long Length, DateTime WriteTime)>? preRun =
                _options.Workspace?.SnapshotWorkFiles();

            if (_sandbox == null && _options.Sandbox == SkillSandboxMode.Preferred
                && Interlocked.Exchange(ref s_unconfinedHostWarned, 1) == 0)
            {
                // `preferred` quietly degrades to no confinement when the host has no
                // sandbox at all — the one case the degraded-run warning below never
                // sees, because there is nothing to wrap with.
                _logger.LogWarning(LogEventIds.SkillScriptExecuted,
                    "skills.script.unconfined host={Host}: --skills-sandbox preferred found no OS sandbox, so skill scripts " +
                    "run UNCONFINED — full filesystem and network access with this process's privileges. Use " +
                    "--skills-sandbox required to refuse instead. Reported once.",
                    SkillSandboxFactory.DescribeHost());
            }

            var launch = new ShellLaunch
            {
                Argv = argv,
                WorkingDirectory = workDirectory,
                WriteDirectory = workDirectory,
                ReadOnlyDirectory = skill.RootDirectory,
                ReadablePaths = readablePaths,
                AllowNetwork = _options.AllowNetwork,
                Timeout = _options.Timeout,
                MaxOutputBytes = _options.MaxOutputBytes,
                Environment = BuildEnvironment(workDirectory, skill.RootDirectory),
                OnOutputLine = line => Tap(onOutput, line),
                Purpose = ShellLaunch.Purposes.Script,
            };

            ConfinedResult result;
            try
            {
                if (!_backend.TryStart(launch, out IShellJob? job, out ConfinedResult failure))
                {
                    string why = failure.Error ?? $"'{interpreter}' could not be started";
                    // A confinement that could not be set up in `required` mode is the
                    // refusal the mode exists for, and is phrased as one; anything else
                    // is the interpreter not starting.
                    if (why.StartsWith("the sandbox could not be applied", StringComparison.Ordinal))
                    {
                        return SkillToolResult.Failure(
                            $"'{normalized}' was stopped: {why}, and skill scripts are configured to run "
                            + "only when they can be confined.");
                    }
                    if (why.StartsWith("the sandbox could not be", StringComparison.Ordinal))
                    {
                        return SkillToolResult.Failure(
                            $"'{normalized}' was not run: {why}, and skill scripts are configured to run "
                            + "only when they can be confined.");
                    }
                    return SkillToolResult.Failure($"'{normalized}' could not be started: {why}");
                }

                using (job)
                {
                    // Said, not swallowed: a run that went ahead without the confinement
                    // that was detected is a materially different run, and the operator
                    // reading the log should see why it happened.
                    if (_sandbox != null && job!.DegradedReason is { } degraded)
                    {
                        _logger.LogWarning(LogEventIds.SkillScriptExecuted,
                            "skills.script.unconfined skill={SkillId} script={Script} reason={Reason}",
                            skill.Id, normalized, degraded);
                    }
                    result = job!.WaitForExit(_options.Timeout);
                }
            }
            catch (Exception ex) when (ex is System.ComponentModel.Win32Exception or InvalidOperationException or IOException)
            {
                return SkillToolResult.Failure($"'{normalized}' could not be run: {ex.Message}");
            }

            if (!result.Started)
                return SkillToolResult.Failure($"'{normalized}' could not be run: {result.Error ?? "it did not start"}");

            string sandboxName = result.SandboxName;
            if (result.TimedOut)
            {
                _logger.LogWarning(LogEventIds.SkillScriptExecuted,
                    "skills.script.timeout skill={SkillId} script={Script} sandbox={Sandbox} timeoutMs={TimeoutMs}",
                    skill.Id, normalized, sandboxName, (int)_options.Timeout.TotalMilliseconds);
                return SkillToolResult.Failure(
                    $"'{normalized}' did not finish within "
                    + $"{_options.Timeout.TotalSeconds.ToString("0.#", CultureInfo.InvariantCulture)}s and was stopped."
                    + Describe(result.Stdout, result.Stderr, workDirectory, preRun, _options.Workspace));
            }

            stderrText = result.Stderr;

            _logger.LogInformation(LogEventIds.SkillScriptExecuted,
                "skills.script.ran skill={SkillId} script={Script} sandbox={Sandbox} exit={ExitCode} ms={ElapsedMs} stdout={StdoutBytes}",
                skill.Id, normalized, sandboxName, result.ExitCode, (long)result.Elapsed.TotalMilliseconds, result.Stdout.Length);

            var sb = new StringBuilder();
            sb.Append("Ran ").Append(normalized).Append(" (exit code ")
              .Append(result.ExitCode.ToString(CultureInfo.InvariantCulture)).Append(", sandbox: ")
              .Append(sandboxName).Append(")\n");

            // Say what the sandbox did NOT confine. The model is deciding what to do
            // with this script's output, and on a platform where the script could
            // have reached the network or the wider filesystem that is a materially
            // different situation from one where it could not.
            IReadOnlyList<string> gaps = _sandbox?.Capabilities.Gaps() ?? AllGaps;
            if (gaps.Count > 0)
                sb.Append("Not confined on this host: ").Append(string.Join("; ", gaps)).Append(".\n");

            sb.Append(Describe(result.Stdout, result.Stderr, workDirectory, preRun, _options.Workspace));

            // A script that died on a missing import is the single most common way a
            // skill's tooling fails on a fresh host, and the fix is one call away:
            // the session's environment is shared, so the shell can install what the
            // script needs. Without this the model re-runs the script unchanged.
            // A `match` statement on Apple's frozen python3 (3.9) dies as a bare
            // "SyntaxError: invalid syntax" — which reads as a broken script when
            // the actual problem is the host's interpreter. Name it.
            if (result.ExitCode != 0
                && interpreter.Contains("python", StringComparison.OrdinalIgnoreCase)
                && CodeExec.CodeDiagnostics.LooksLikeOldInterpreter(
                    result.Stderr,
                    CodeExec.CodeLanguage.Python,
                    interpreter)
                && CodeExec.CodeEnvironment.PythonVersionOf(interpreter) is { } version
                && version < new Version(3, 10))
            {
                sb.Append("\nNote: this host's Python is ").Append(version)
                  .Append(", and skill scripts commonly need 3.10+ (the 'match' statement). If the ")
                  .Append("script looks correct, the fix is on the host: install a newer Python ")
                  .Append("(e.g. `brew install python@3.12`) and restart the server — it is picked up automatically.\n");
            }

            if (result.ExitCode != 0
                && CodeExec.CodeDiagnostics.MissingModule(CodeExec.CodeLanguage.Python, result.Stderr) is { } missing)
            {
                // A module that is a DIRECTORY OF THIS SKILL is never a package to
                // install, and saying so was actively dangerous. skill-creator's entry
                // points import `scripts.quick_validate`; the advice this produced was
                // "pip install scripts", and `scripts` is a real name on PyPI owned by
                // nobody in particular - the host was telling the model to pull a
                // stranger's package to satisfy an import of a file sitting beside the
                // script. The import itself is now made to work (the skill root goes on
                // PYTHONPATH in BuildEnvironment); this is the second half, so that if
                // one ever fails again it fails honestly.
                string top = missing.Split('.')[0];
                bool insideSkill =
                    File.Exists(Path.Combine(skill.RootDirectory, top + ".py"))
                    || Directory.Exists(Path.Combine(skill.RootDirectory, top));

                if (insideSkill)
                {
                    sb.Append("\n'").Append(missing)
                      .Append("' is part of this skill, not a package to install - do NOT try to ")
                      .Append("install it. It failed to import because of how the script was ")
                      .Append("invoked, not because anything is missing. Run it from the shell ")
                      .Append("instead, with the skill's own directory on PYTHONPATH.\n");
                }
                else
                {
                    string install = CodeExec.CodeDiagnostics.InstallNameFor(CodeExec.CodeLanguage.Python, missing);
                    sb.Append("\nThe module '").Append(missing)
                      .Append("' is not installed in this session's environment. Install it from the shell ")
                      .Append("and run this script again:\n  pip install ").Append(install)
                      .Append("\nThe shell and skill scripts share one environment, so what you install ")
                      .Append("there is visible here.\n");
                }
            }

            // A document writer's program and its INPUT are two different repair
            // targets.  The bundled writers deliberately turn malformed JSON into a
            // short "the spec is not valid JSON" diagnostic, without a traceback.  A
            // blanket script-overlay response therefore points at make_pptx.py even
            // though the only broken bytes are in the workspace spec the model just
            // wrote.  Resolve that argument under the workspace, parse it ourselves to
            // recover the exact line, and show only the nearby editable text.
            string? inputRepair = result.ExitCode != 0 && _options.Workspace is { } inputWorkspace
                ? InvalidJsonSpecRepairHint(arguments, result.Stderr, inputWorkspace, workDirectory)
                : null;
            if (inputRepair != null)
                sb.Append(inputRepair);

            // A failure while running a skill is not proof that the skill SCRIPT is
            // wrong. Argument errors, missing inputs and missing packages are much more
            // common, and staging the entry point for all of them points the model at
            // trusted code that cannot fix the problem. Only a traceback/error location
            // whose deepest relevant frame is this exact entry script earns an overlay.
            //
            // That overlay is session-local. It becomes an ordinary file the shell can
            // edit and run while the skill on disk remains untouched.
            if (result.ExitCode != 0 && inputRepair == null
                && _options.Workspace is { } fixWorkspace
                && CanStageRepairOverlay(normalized)
                && FailurePointsToBundledScript(result.Stderr, scriptPath, interpreter))
            {
                try
                {
                    byte[] sourceBytes = File.ReadAllBytes(scriptPath);
                    if (TryGetOrStageRepairOverlay(
                        fixWorkspace,
                        skill,
                        normalized,
                        scriptPath,
                        sourceBytes,
                        out string? overlay,
                        out string? repairLauncher,
                        out bool overlayAlreadyExists))
                    {
                        string runAdvice = repairLauncher == null
                            ? "run it from the shell. "
                            : "run the same arguments through its compatibility launcher '"
                              + repairLauncher
                              + "' from the shell. Do not edit the launcher: it makes sibling imports and "
                              + "resources resolved through __file__ behave exactly as they did in the skill. ";
                        if (overlayAlreadyExists)
                        {
                            sb.Append("\nThe traceback points to this bundled script. Its editable repair copy already exists in your working directory as '")
                              .Append(overlay)
                              .Append("'. It was not overwritten, so any repair already made there is intact. "
                                      + "Continue with that copy: use read_file for only the relevant region, "
                                      + "change the broken region with apply_patch, and ")
                              .Append(runAdvice)
                              .Append("Do not rewrite the complete file. The skill's own copy is read-only "
                                      + "and unchanged.\n");
                        }
                        else
                        {
                            sb.Append("\nThe traceback points to this bundled script. A copy of this script is now in your working directory as '")
                              .Append(overlay)
                              .Append("'. Fix THAT copy: use read_file for only the relevant region, "
                                      + "change the broken region with apply_patch, and ")
                              .Append(runAdvice)
                              .Append("Do not rewrite the complete file. "
                                      + "The skill's own copy is read-only and unchanged.\n");
                        }
                    }
                }
                catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
                {
                    // Staging is a convenience; a failure to stage must not replace
                    // the script's own error with a filesystem one.
                }
            }

            // The files this script produced, kept for the user to download. Same
            // contract as the shell tool: the model gets ready-made markdown links.
            IReadOnlyList<SkillProducedFile> files = Array.Empty<SkillProducedFile>();
            if (_options.CaptureProducedFiles != null && _options.Workspace is { } ws)
            {
                files = _options.CaptureProducedFiles(
                    workDirectory,
                    relative => ws.IsHostRepairArtifact(relative)
                        || (preRun != null && ws.IsUnchangedSince(preRun, relative)));
                if (files.Count > 0)
                {
                    sb.Append("\nFiles produced. The user downloads them through these links - copy the ")
                      .Append("markdown links below into your answer verbatim when the user asked for the file:\n");
                    foreach (SkillProducedFile file in files)
                        sb.Append("- [").Append(file.Name).Append("](").Append(file.Url).Append(")\n");
                }
            }

            return new SkillToolResult(result.ExitCode == 0, sb.ToString(), skill.Id, normalized)
            { Files = files };
        }

        private const int MaxRepairInputBytes = 1024 * 1024;

        private static readonly Regex PythonFrame = new(
            "^\\s*File \\\"(?<path>[^\\\"]+)\\\", line [0-9]+(?:, in .*)?\\s*$",
            RegexOptions.Compiled | RegexOptions.CultureInvariant);

        private const string RepairOverlayDirectory = "skill-repairs";

        private const string RepairLauncherSuffix = ".tensorsharp-runner";

        /// <summary>
        /// Moving an entry point changes language-specific resolution rules. The Python
        /// launcher below restores its original sibling imports, argv[0] and __file__.
        /// We do not have an equivalent proven wrapper for CommonJS, ESM or shell
        /// scripts (where __dirname/import.meta.url/$0 all matter), so do not offer a
        /// repair copy that would behave differently from the bundled program.
        /// </summary>
        private static bool CanStageRepairOverlay(string normalized) =>
            string.Equals(Path.GetExtension(normalized), ".py", StringComparison.OrdinalIgnoreCase);

        /// <summary>
        /// Reuse an edited repair copy only when host-owned state proves which bundled
        /// script it came from. The visible path retains the complete skill-relative
        /// path, so two scripts called <c>tool.py</c> cannot silently share a copy.
        /// </summary>
        private static bool TryGetOrStageRepairOverlay(
            SessionWorkspace workspace,
            Skill skill,
            string normalized,
            string scriptPath,
            byte[] sourceBytes,
            out string? overlay,
            out string? launcher,
            out bool alreadyStaged)
        {
            overlay = null;
            launcher = null;
            alreadyStaged = false;

            string repairRoot = RepairOverlayDirectory + "/" + skill.Id;
            string identity = RepairOverlayIdentity(scriptPath, sourceBytes);
            string markerDirectory = Path.Combine(workspace.StateDirectory, "skill-repair-overlays");
            string markerPath = Path.Combine(markerDirectory, identity + ".path");

            if (TryReadRepairOverlayMarker(
                workspace, markerPath, repairRoot, out string? priorOverlay))
            {
                if (!TryGetOrStageRepairLauncher(
                    workspace, skill, normalized, scriptPath, priorOverlay!,
                    acceptExisting: true, out launcher, out _))
                {
                    return false;
                }
                overlay = priorOverlay;
                alreadyStaged = true;
                workspace.MarkHostRepairArtifacts(overlay!, launcher!);
                return true;
            }

            string baseOverlay = repairRoot + "/" + normalized;
            for (int attempt = 0; attempt < 8; attempt++)
            {
                string candidate = attempt == 0
                    ? baseOverlay
                    : RepairOverlayWithIdentitySuffix(baseOverlay, identity, attempt);
                if (workspace.TryCreateFile(
                    candidate, sourceBytes, out bool collision, out _))
                {
                    if (!TryGetOrStageRepairLauncher(
                        workspace, skill, normalized, scriptPath, candidate,
                        acceptExisting: false, out launcher, out bool launcherCollision))
                    {
                        // candidate is known host-created even when a pre-existing
                        // launcher prevents this pair from being usable.
                        workspace.MarkHostRepairArtifacts(candidate);
                        if (launcherCollision)
                            continue;
                        return false;
                    }

                    if (!TryPublishRepairOverlayMarker(
                        workspace,
                        markerDirectory,
                        markerPath,
                        repairRoot,
                        candidate,
                        out string? canonical))
                    {
                        workspace.MarkHostRepairArtifacts(candidate, launcher!);
                        return false;
                    }

                    if (!string.Equals(canonical, candidate, SkillPathGuard.PathComparison))
                    {
                        // Another host publisher won. Keep its immutable provenance. The
                        // losing pair is still proven host-created and is hidden exactly,
                        // without a racy delete that could remove a path replaced between
                        // verification and unlinking.
                        workspace.MarkHostRepairArtifacts(candidate, launcher!);
                        if (!TryGetOrStageRepairLauncher(
                            workspace, skill, normalized, scriptPath, canonical!,
                            acceptExisting: true, out launcher, out _))
                        {
                            return false;
                        }
                        alreadyStaged = true;
                    }

                    overlay = canonical;
                    workspace.MarkHostRepairArtifacts(overlay!, launcher!);
                    return true;
                }
                if (!collision)
                    return false;
            }

            return false;
        }

        /// <summary>
        /// A copied Python entry point no longer sits beside the modules and resources it
        /// was written against. This tiny launcher executes the edited bytes but gives
        /// them the original entry point's <c>sys.path</c>, <c>sys.argv[0]</c> and
        /// <c>__file__</c>. Thus <c>import specs</c> and
        /// <c>Path(__file__).with_name(...)</c> keep working without copying a skill's
        /// whole resource tree into the writable workspace.
        /// </summary>
        private static bool TryGetOrStageRepairLauncher(
            SessionWorkspace workspace,
            Skill skill,
            string normalized,
            string scriptPath,
            string overlay,
            bool acceptExisting,
            out string? launcher,
            out bool collision)
        {
            launcher = null;
            collision = false;
            if (!string.Equals(Path.GetExtension(normalized), ".py", StringComparison.OrdinalIgnoreCase))
                return true;

            launcher = overlay + RepairLauncherSuffix;
            if (!workspace.TryResolve(overlay, out string overlayPath, out _))
                return false;

            string source = PythonRepairLauncher(
                overlayPath, scriptPath, skill.RootDirectory, workspace.EnvDirectory);
            if (workspace.TryCreateFile(launcher, source, out collision, out _))
                return true;

            if (!collision || !acceptExisting
                || !workspace.TryResolve(launcher, out string existing, out _)
                || !File.Exists(existing))
            {
                return false;
            }

            // A provenance marker is written only after this launcher and the repair
            // copy were staged together. Still verify the host-authored launcher bytes:
            // the repair copy is intentionally editable, while this compatibility
            // wrapper is not, and running a modified wrapper would no longer preserve
            // the bundled script's semantics.
            try
            {
                return string.Equals(
                    File.ReadAllText(existing), source, StringComparison.Ordinal);
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                return false;
            }
        }

        private static string PythonRepairLauncher(
            string overlayPath, string scriptPath, string skillRoot, string environmentRoot)
        {
            string overlay = JsonSerializer.Serialize(Path.GetFullPath(overlayPath));
            string original = JsonSerializer.Serialize(Path.GetFullPath(scriptPath));
            string root = JsonSerializer.Serialize(Path.GetFullPath(skillRoot));
            string environment = JsonSerializer.Serialize(Path.GetFullPath(environmentRoot));
            return "# Host-created launcher for an editable skill repair; do not edit.\n"
                 + "import os as _os\n"
                 + "import sys as _sys\n"
                 + "_overlay = " + overlay + "\n"
                 + "_original = " + original + "\n"
                 + "_skill_root = " + root + "\n"
                 + "_environment_root = " + environment + "\n"
                 + "_script_dir = _os.path.dirname(_original)\n"
                 + "if _sys.path:\n"
                 + "    _sys.path[0] = _script_dir\n"
                 + "else:\n"
                 + "    _sys.path.append(_script_dir)\n"
                 + "try:\n"
                 + "    _root_index = next(i for i, p in enumerate(_sys.path) "
                 + "if _os.path.abspath(p) == _os.path.abspath(_environment_root)) + 1\n"
                 + "except StopIteration:\n"
                 + "    _root_index = 1\n"
                 + "if _skill_root not in _sys.path:\n"
                 + "    _sys.path.insert(_root_index, _skill_root)\n"
                 + "_sys.argv[0] = _original\n"
                 + "_globals = {'__name__': '__main__', '__file__': _original, "
                 + "'__package__': None, '__cached__': None, '__spec__': None}\n"
                 + "with open(_overlay, 'rb') as _handle:\n"
                 + "    _source = _handle.read()\n"
                 + "exec(compile(_source, _overlay, 'exec'), _globals, _globals)\n";
        }

        private static string RepairOverlayIdentity(string scriptPath, byte[] sourceBytes)
        {
            string sourceHash = Convert.ToHexString(SHA256.HashData(sourceBytes));
            string sourceIdentity = Path.GetFullPath(scriptPath) + "\n" + sourceHash;
            return Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(sourceIdentity)))
                .ToLowerInvariant();
        }

        private static string RepairOverlayWithIdentitySuffix(
            string baseOverlay, string identity, int attempt)
        {
            string extension = Path.GetExtension(baseOverlay);
            string withoutExtension = extension.Length == 0
                ? baseOverlay
                : baseOverlay.Substring(0, baseOverlay.Length - extension.Length);
            string ordinal = attempt == 1
                ? string.Empty
                : "-" + attempt.ToString(CultureInfo.InvariantCulture);
            return withoutExtension + ".repair-" + identity.Substring(0, 12) + ordinal + extension;
        }

        private static bool TryReadRepairOverlayMarker(
            SessionWorkspace workspace,
            string markerPath,
            string repairRoot,
            out string? overlay)
        {
            overlay = null;
            try
            {
                if (!File.Exists(markerPath))
                    return false;

                string candidate = File.ReadAllText(markerPath).Trim().Replace('\\', '/');
                if (candidate.IndexOfAny(new[] { '\r', '\n' }) >= 0
                    || !candidate.StartsWith(repairRoot + "/", StringComparison.Ordinal)
                    || !workspace.TryResolve(candidate, out string full, out _)
                    || !File.Exists(full))
                {
                    return false;
                }

                overlay = candidate;
                return true;
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                return false;
            }
        }

        private static bool TryPublishRepairOverlayMarker(
            SessionWorkspace workspace,
            string markerDirectory,
            string markerPath,
            string repairRoot,
            string overlay,
            out string? canonical)
        {
            canonical = null;
            string? temporary = null;
            try
            {
                Directory.CreateDirectory(markerDirectory);
                temporary = Path.Combine(
                    markerDirectory,
                    ".overlay-" + Guid.NewGuid().ToString("N") + ".tmp");
                File.WriteAllText(
                    temporary,
                    overlay,
                    new UTF8Encoding(encoderShouldEmitUTF8Identifier: false));
                File.Move(temporary, markerPath);
                temporary = null;
                canonical = overlay;
                return true;
            }
            catch (IOException) when (File.Exists(markerPath))
            {
                // Immutable create-only publication makes concurrent provenance writers
                // converge on the winner instead of replacing one another's mapping.
                return TryReadRepairOverlayMarker(
                    workspace, markerPath, repairRoot, out canonical);
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                return false;
            }
            finally
            {
                if (!string.IsNullOrEmpty(temporary))
                {
                    try { File.Delete(temporary); }
                    catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { }
                }
            }
        }

        /// <summary>
        /// True only when a code-shaped failure locates its deepest relevant frame in
        /// the exact bundled entry point. Merely failing while that script was running
        /// is deliberately insufficient: package, CLI and input failures belong to the
        /// environment or arguments and cannot be repaired by copying the script.
        /// </summary>
        private static bool FailurePointsToBundledScript(
            string? stderr,
            string scriptPath,
            string interpreter)
        {
            if (string.IsNullOrWhiteSpace(stderr) || string.IsNullOrWhiteSpace(scriptPath))
                return false;

            string extension = Path.GetExtension(scriptPath);
            if (!string.Equals(extension, ".py", StringComparison.OrdinalIgnoreCase))
                return false;
            CodeExec.CodeLanguage language = CodeExec.CodeLanguage.Python;

            // Classify with network confinement enabled because this method emits no
            // policy claim; it only refuses to blame code for DNS/socket failures.
            if (CodeExec.CodeDiagnostics.ClassifyFailure(
                    stderr,
                    language,
                    networkConfined: true,
                    pythonInterpreter: interpreter).Source
                == CodeExec.CodeDiagnostics.FailureSource.Environment
                || LooksLikeArgumentOrInputFailure(stderr))
            {
                return false;
            }

            bool traceback = stderr.IndexOf(
                "Traceback (most recent call last):", StringComparison.Ordinal) >= 0;
            bool syntaxDiagnostic = stderr.IndexOf("SyntaxError", StringComparison.Ordinal) >= 0
                || stderr.IndexOf("IndentationError", StringComparison.Ordinal) >= 0
                || stderr.IndexOf("TabError", StringComparison.Ordinal) >= 0;
            if (!traceback && !syntaxDiagnostic)
                return false;

            // Locating the deepest frame in the entry point proves where an
            // exception surfaced, not why it surfaced. Validation/data exceptions
            // such as KeyError, ValueError and Pillow's UnidentifiedImageError are
            // normally repaired in the workspace input. Stage trusted code only for
            // syntax faults and a deliberately small set of programmer-fault shapes.
            if (!syntaxDiagnostic && !HasRepairablePythonTerminalException(stderr))
                return false;

            return TryGetAttributedPythonFrame(stderr, syntaxDiagnostic, out string? frame)
                && SameDiagnosticPath(frame, scriptPath);
        }

        private static bool HasRepairablePythonTerminalException(string stderr)
        {
            string terminal = stderr.Replace("\r\n", "\n", StringComparison.Ordinal)
                .Split('\n', StringSplitOptions.RemoveEmptyEntries)
                .LastOrDefault()?.Trim() ?? string.Empty;
            foreach (string exception in new[]
                     {
                         "RuntimeError", "NameError", "UnboundLocalError", "NotImplementedError",
                     })
            {
                if (terminal.StartsWith(exception + ":", StringComparison.Ordinal)
                    || string.Equals(terminal, exception, StringComparison.Ordinal))
                {
                    return true;
                }
            }
            return false;
        }

        /// <summary>
        /// Extract a location only from Python's structured diagnostic block. Searching
        /// every frame-shaped substring in stderr lets an exception message or echoed
        /// workspace input append <c>File "...", line 1</c> after the real traceback and
        /// forge attribution to the bundled entry point.
        /// </summary>
        private static bool TryGetAttributedPythonFrame(
            string stderr,
            bool syntaxDiagnostic,
            out string? path)
        {
            path = null;
            string[] lines = stderr.Replace("\r\n", "\n", StringComparison.Ordinal).Split('\n');

            if (syntaxDiagnostic)
            {
                int terminal = -1;
                for (int index = lines.Length - 1; index >= 0; index--)
                {
                    string value = lines[index].TrimStart();
                    if (value.StartsWith("SyntaxError", StringComparison.Ordinal)
                        || value.StartsWith("IndentationError", StringComparison.Ordinal)
                        || value.StartsWith("TabError", StringComparison.Ordinal))
                    {
                        terminal = index;
                        break;
                    }
                }
                if (terminal < 0)
                    return false;

                // A compile diagnostic puts the File line directly above its source
                // excerpt and caret. Do not borrow a frame from an earlier traceback.
                for (int index = terminal - 1; index >= 0; index--)
                {
                    string trimmed = lines[index].Trim();
                    if (string.Equals(trimmed, "Traceback (most recent call last):", StringComparison.Ordinal)
                        || IsPythonTerminalLine(trimmed))
                    {
                        break;
                    }

                    Match frame = PythonFrame.Match(lines[index]);
                    if (frame.Success)
                    {
                        path = frame.Groups["path"].Value;
                        return true;
                    }
                }
                return false;
            }

            int header = -1;
            for (int index = lines.Length - 1; index >= 0; index--)
            {
                if (string.Equals(
                    lines[index].Trim(),
                    "Traceback (most recent call last):",
                    StringComparison.Ordinal))
                {
                    header = index;
                    break;
                }
            }
            if (header < 0)
                return false;

            string? deepest = null;
            for (int index = header + 1; index < lines.Length; index++)
            {
                string trimmed = lines[index].Trim();
                if (IsPythonTerminalLine(trimmed))
                    break;

                Match frame = PythonFrame.Match(lines[index]);
                if (frame.Success)
                    deepest = frame.Groups["path"].Value;
            }

            path = deepest;
            return path != null;
        }

        private static bool IsPythonTerminalLine(string value)
        {
            int colon = value.IndexOf(':');
            string name = colon >= 0 ? value.Substring(0, colon) : value;
            return name.EndsWith("Error", StringComparison.Ordinal)
                || name.EndsWith("Exception", StringComparison.Ordinal);
        }

        private static bool LooksLikeArgumentOrInputFailure(string stderr)
        {
            bool argparse = stderr.IndexOf("usage:", StringComparison.OrdinalIgnoreCase) >= 0
                && (stderr.IndexOf(": error:", StringComparison.OrdinalIgnoreCase) >= 0
                    || stderr.IndexOf("unrecognized arguments", StringComparison.OrdinalIgnoreCase) >= 0
                    || stderr.IndexOf("the following arguments are required", StringComparison.OrdinalIgnoreCase) >= 0
                    || stderr.IndexOf("expected one argument", StringComparison.OrdinalIgnoreCase) >= 0
                    || stderr.IndexOf("invalid choice", StringComparison.OrdinalIgnoreCase) >= 0);
            if (argparse)
                return true;

            foreach (string marker in new[]
                     {
                         "FileNotFoundError", "NotADirectoryError", "IsADirectoryError", "PermissionError",
                         "UnicodeDecodeError", "JSONDecodeError", "EOFError", "SystemExit",
                         "No such file or directory", "Permission denied", "not valid JSON",
                     })
            {
                if (stderr.IndexOf(marker, StringComparison.OrdinalIgnoreCase) >= 0)
                    return true;
            }
            return false;
        }

        private static bool SameDiagnosticPath(string? candidate, string actual)
        {
            if (!TryNormalizeDiagnosticPath(candidate, out string? normalized))
                return false;
            try
            {
                return string.Equals(
                    Path.GetFullPath(normalized!), Path.GetFullPath(actual), SkillPathGuard.PathComparison);
            }
            catch (Exception ex) when (ex is ArgumentException or NotSupportedException or PathTooLongException)
            {
                return false;
            }
        }

        private static bool TryNormalizeDiagnosticPath(string? value, out string? normalized)
        {
            normalized = value?.Trim();
            if (string.IsNullOrEmpty(normalized))
                return false;

            // Do not peel text after the last '('. Python's quoted frame already
            // supplies the literal filename, and directories such as "TensorAgent
            // (dev)" are valid.
            if (normalized.StartsWith("at ", StringComparison.Ordinal))
                normalized = normalized.Substring(3).TrimStart();

            if (normalized.StartsWith("file://", StringComparison.OrdinalIgnoreCase))
            {
                if (!Uri.TryCreate(normalized, UriKind.Absolute, out Uri? uri) || !uri.IsFile)
                    return false;
                normalized = uri.LocalPath;
            }

            return Path.IsPathFullyQualified(normalized);
        }

        /// <summary>
        /// Describe a malformed workspace JSON file passed through <c>--spec</c>, or
        /// return null when this failure does not prove that that input is the problem.
        /// The returned excerpt is also entered in the file ledger: an immediately
        /// following <c>apply_patch</c> call may use the shown text without spending a
        /// separate read round.
        /// </summary>
        private static string? InvalidJsonSpecRepairHint(
            IReadOnlyList<string>? arguments,
            string? stderr,
            SessionWorkspace workspace,
            string currentDirectory)
        {
            if (!LooksLikeMalformedJson(stderr))
            {
                return null;
            }

            string? relative = SpecArgument(arguments);
            if (string.IsNullOrWhiteSpace(relative) || relative == "-"
                || !relative.EndsWith(".json", StringComparison.OrdinalIgnoreCase)
                || !workspace.TryResolveFrom(currentDirectory, relative, out string full, out _)
                || !ShellSession.TryReadBoundedRegularTextUnderRoot(
                    workspace.WorkDirectory, full, MaxRepairInputBytes, out string source))
            {
                return null;
            }

            int line;
            long byteInLine;
            try
            {
                using JsonDocument _ = JsonDocument.Parse(source);
                return null; // The stderr named JSON, but this workspace input is valid.
            }
            catch (JsonException ex)
            {
                long zeroBasedLine = Math.Max(0, ex.LineNumber ?? 0);
                line = (int)Math.Min(int.MaxValue, zeroBasedLine + 1);
                byteInLine = Math.Max(0, ex.BytePositionInLine ?? 0) + 1;
            }

            IReadOnlyList<string> lines = NumberedListing.SplitLines(source);
            int total = NumberedListing.RealLineCount(lines);
            if (total < 1 || NumberedListing.LooksBinary(lines))
                return null;

            line = Math.Clamp(line, 1, total);
            int first = Math.Max(1, line - 3);
            int last = Math.Min(total, line + 3);
            var excerpt = new StringBuilder();
            NumberedListing.Append(
                excerpt,
                lines,
                first - 1,
                last - 1,
                NumberedListing.MaxExcerptChars,
                NumberedListing.MaxExcerptLineChars,
                out int lastShownIndex);

            // Only authorize bytes that were shown completely.  A clipped minified
            // JSON line must be read explicitly before it can be used as an exact edit
            // anchor; claiming otherwise would defeat the ledger's safety contract.
            bool exact = lastShownIndex >= first - 1;
            for (int index = first - 1; exact && index <= lastShownIndex; index++)
                exact = lines[index].Length <= NumberedListing.MaxExcerptLineChars;
            if (exact)
            {
                workspace.Reads.Record(
                    full,
                    source.Replace("\r\n", "\n", StringComparison.Ordinal),
                    first,
                    lastShownIndex + 1,
                    complete: first == 1 && lastShownIndex + 1 == total);
            }

            string display = Path.GetRelativePath(workspace.WorkDirectory, full)
                .Replace(Path.DirectorySeparatorChar, '/');
            if (Path.AltDirectorySeparatorChar != Path.DirectorySeparatorChar)
                display = display.Replace(Path.AltDirectorySeparatorChar, '/');

            var sb = new StringBuilder();
            sb.Append("\nThis failure came from the input file '").Append(display)
              .Append("', not from the bundled skill script. Its invalid JSON around line ")
              .Append(line.ToString(CultureInfo.InvariantCulture)).Append(", UTF-8 byte offset ")
              .Append(byteInLine.ToString(CultureInfo.InvariantCulture)).Append(" is:\n")
              .Append(excerpt)
              .Append("Fix the smallest incorrect region in '").Append(display).Append("' with `")
              .Append(ShellTools.PatchToolName)
              .Append("`, then run the same skill call again. Do not use `")
              .Append(ShellTools.WriteToolName)
              .Append("` or re-type the whole spec, and do not edit or copy the bundled skill script. ")
              .Append("If the patch context no longer matches, read that region and retry against its current text.\n");
            return sb.ToString();
        }

        private static bool LooksLikeMalformedJson(string? stderr)
        {
            if (string.IsNullOrEmpty(stderr))
                return false;
            if (stderr.IndexOf("not valid JSON", StringComparison.OrdinalIgnoreCase) >= 0
                || stderr.IndexOf("JSONDecodeError", StringComparison.OrdinalIgnoreCase) >= 0)
            {
                return true;
            }

            // Node/V8 uses several version-dependent spellings, for example
            // "Unexpected token } in JSON at position 42" and "Unexpected end of
            // JSON input" followed by an `at JSON.parse` frame. Requiring both a
            // SyntaxError and JSON context avoids treating an ordinary JavaScript
            // syntax defect as a malformed workspace spec.
            return stderr.IndexOf("SyntaxError", StringComparison.OrdinalIgnoreCase) >= 0
                && (stderr.IndexOf(" in JSON at position", StringComparison.OrdinalIgnoreCase) >= 0
                    || stderr.IndexOf("JSON input", StringComparison.OrdinalIgnoreCase) >= 0
                    || stderr.IndexOf("JSON.parse", StringComparison.OrdinalIgnoreCase) >= 0
                    || stderr.IndexOf("JSON Parse error", StringComparison.OrdinalIgnoreCase) >= 0);
        }

        private static string? SpecArgument(IReadOnlyList<string>? arguments)
        {
            if (arguments == null)
                return null;

            for (int index = 0; index < arguments.Count; index++)
            {
                string? argument = arguments[index];
                if (string.Equals(argument, "--spec", StringComparison.Ordinal))
                    return index + 1 < arguments.Count ? arguments[index + 1] : null;
                if (argument != null && argument.StartsWith("--spec=", StringComparison.Ordinal))
                    return argument.Substring("--spec=".Length);
            }
            return null;
        }

        /// <summary>
        /// Give the child a minimal environment.
        ///
        /// <para>
        /// The host process's environment is where credentials live —
        /// <c>AWS_SECRET_ACCESS_KEY</c>, <c>OPENAI_API_KEY</c>, <c>GITHUB_TOKEN</c>, a
        /// database URL. Inheriting it wholesale would hand every one of them to an
        /// uploaded script, and the sandbox cannot help: the values are already in the
        /// process image. So the child starts from nothing and is given back only what
        /// an interpreter needs to run.
        /// </para>
        /// </summary>
        /// <param name="skillRoot">
        /// The directory of the skill whose script this is, so its own modules are
        /// importable. See the PYTHONPATH note below.
        /// </param>
        private Dictionary<string, string> BuildEnvironment(string workDirectory, string? skillRoot = null)
        {
            var startInfo = new EnvironmentBag();

            foreach (string name in _options.PassThroughEnvironmentVariables)
            {
                string? value = Environment.GetEnvironmentVariable(name);
                if (!string.IsNullOrEmpty(value))
                    startInfo.Environment[name] = value;
            }

            startInfo.Environment["PWD"] = workDirectory;
            startInfo.Environment["TMPDIR"] = workDirectory;
            startInfo.Environment["HOME"] = workDirectory;
            // HOME is not what Windows reads. CPython's ntpath.expanduser resolves `~`
            // from USERPROFILE and never looks at HOME, so on Windows the redirection
            // above was invisible to the one call that asks for the home directory by
            // name; the same list also supplies TEMP/TMP, LOCALAPPDATA/APPDATA and the
            // machine variables a Windows child cannot start without.
            CodeExec.CodeEnvironment.ApplyWindowsBaseline(startInfo.Environment, workDirectory);
            // Keeps CPython from writing .pyc files into the read-only skill directory,
            // which fails under the sandbox and produces a confusing error.
            startInfo.Environment["PYTHONDONTWRITEBYTECODE"] = "1";
            startInfo.Environment["PYTHONUNBUFFERED"] = "1";

            // The session's package environment: what the shell installed, the script can
            // import. This single line is what makes a skill's bundled tooling actually
            // RUNNABLE — validate.py needs defusedxml, thumbnail.py needs Pillow, and
            // none of them ship with the interpreter.
            if (_options.Workspace is { } workspace)
            {
                startInfo.Environment["PYTHONPATH"] = workspace.EnvDirectory;
                startInfo.Environment["NODE_PATH"] = Path.Combine(workspace.EnvDirectory, "node_modules");
            }

            // The skill's OWN root, so a script can import its siblings the way its
            // author wrote them.
            //
            // Python puts the SCRIPT's directory on sys.path, never the skill root, so a
            // script at <skill>/scripts/package_skill.py that says
            // `from scripts.quick_validate import validate_skill` — a package-qualified
            // import of the file sitting right beside it — cannot resolve it. Both of
            // skill-creator's entry points are written that way and both failed with
            // ModuleNotFoundError before this line existed. It is not a Windows problem;
            // it simply had never been exercised.
            //
            // Appended AFTER the session environment rather than prepended, so a skill
            // cannot shadow an installed package with a directory of the same name —
            // `scripts`, `utils` and `core` are exactly the names at stake here.
            if (!string.IsNullOrEmpty(skillRoot))
            {
                startInfo.Environment["PYTHONPATH"] =
                    startInfo.Environment.TryGetValue("PYTHONPATH", out string? existing)
                    && !string.IsNullOrEmpty(existing)
                        ? existing + Path.PathSeparator + skillRoot
                        : skillRoot;
            }

            foreach (KeyValuePair<string, string> entry in _options.EnvironmentVariables)
                startInfo.Environment[entry.Key] = entry.Value;

            return startInfo.Environment;
        }

        /// <summary>
        /// A stand-in for the <see cref="ProcessStartInfo"/> this used to fill in, so the
        /// assignments below read exactly as they did when the environment was applied to a
        /// start info rather than handed over as a complete set.
        /// </summary>
        private sealed class EnvironmentBag
        {
            public Dictionary<string, string> Environment { get; } = new(StringComparer.Ordinal);
        }

        /// <summary>
        /// Render the run's output, plus anything it left in the scratch directory.
        /// Listing the produced files is what makes a script that writes a report
        /// useful: the model can name the file in its answer, and the caller can find it.
        /// </summary>
        private static string Describe(
            string stdout, string stderr, string workDirectory,
            Dictionary<string, (long Length, DateTime WriteTime)>? preRun = null,
            SessionWorkspace? workspace = null)
        {
            var sb = new StringBuilder();
            if (stdout.Length > 0)
                sb.Append("\nstdout:\n").Append(stdout);
            if (stderr.Length > 0)
                sb.Append("\nstderr:\n").Append(stderr);

            string[] produced = ListProducedFiles(workDirectory, preRun, workspace);
            if (produced.Length > 0)
            {
                sb.Append("\nFiles this run wrote to the working directory:\n");
                foreach (string file in produced.Take(40))
                    sb.Append("- ").Append(file).Append('\n');
                if (produced.Length > 40)
                {
                    sb.Append("- (").Append((produced.Length - 40).ToString(CultureInfo.InvariantCulture))
                      .Append(" more)\n");
                }
            }

            if (sb.Length == 0)
                sb.Append("\n(no output)");
            return sb.ToString();
        }

        private static string[] ListProducedFiles(
            string workDirectory,
            Dictionary<string, (long Length, DateTime WriteTime)>? preRun,
            SessionWorkspace? workspace)
        {
            try
            {
                return Directory
                    .EnumerateFiles(workDirectory, "*", SearchOption.AllDirectories)
                    .Where(f => !Path.GetFileName(f).StartsWith(".tensorsharp-", StringComparison.Ordinal))
                    .Select(f => Path.GetRelativePath(workDirectory, f).Replace(Path.DirectorySeparatorChar, '/'))
                    // Bytecode caches and HOME-redirect fallout are a runtime's mess,
                    // not this script's report.
                    .Where(f => !CodeExec.CodeArtifactStore.IsRuntimeJunk(f))
                    // In a shared session workspace, "produced" means what THIS run
                    // added or changed, not the whole conversation's accumulation.
                    .Where(f => preRun == null || workspace == null || !workspace.IsUnchangedSince(preRun, f))
                    .OrderBy(f => f, StringComparer.Ordinal)
                    .ToArray();
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                return Array.Empty<string>();
            }
        }

        /// <summary>A live-output tap must never be able to kill the reader thread.</summary>
        private static void Tap(Action<string>? tap, string line)
        {
            if (tap == null) return;
            try { tap(line); }
            catch (Exception) { /* the tap is best-effort observability */ }
        }

        private static void TryDeleteDirectory(string directory)
        {
            try
            {
                if (Directory.Exists(directory))
                    Directory.Delete(directory, recursive: true);
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { /* best effort */ }
        }

        /// <summary>
        /// Map a file extension to the interpreter that runs it.
        ///
        /// <para>
        /// An allow-list rather than "make it executable and exec it": handing an
        /// arbitrary file to the OS loader would run a shipped binary, and a shebang
        /// line inside an uploaded script would choose the interpreter rather than this
        /// table.
        /// </para>
        /// </summary>
        /// <summary>
        /// Why the script path did not resolve, phrased so the model can fix it itself.
        ///
        /// <para>
        /// The common failure is not a typo. It is a model that puts the whole command
        /// line into <c>path</c> - <c>skills_run(path: "scripts/budget.py 2400")</c> -
        /// because that is how the skill's own SKILL.md writes the invocation
        /// ("RUN <c>scripts/budget.py &lt;payload_kg&gt;</c>"). Answering that with
        /// "'scripts/budget.py 2400' does not exist in this skill" is true and useless:
        /// the model reads it as a missing file, goes and reads the script to check the
        /// name, and retries the identical call. Measured on gemma-4-E4B, that cost three
        /// rounds and the whole round budget, and the request returned nothing.
        /// </para>
        /// <para>
        /// So when the path does not resolve but its leading token does, say which
        /// mistake was made and name the parameter that takes the rest. One round instead
        /// of a dead end.
        /// </para>
        /// </summary>
        private static string ExplainUnresolvedScript(Skill skill, string relativePath, string? guardError)
        {
            string message = $"Cannot run '{relativePath}' from skill '{skill.Id}': {guardError}";

            if (string.IsNullOrEmpty(relativePath))
                return message;

            int split = relativePath.IndexOfAny(ArgumentSeparators);
            if (split <= 0)
                return message;

            string head = relativePath.Substring(0, split);
            if (!SkillPathGuard.TryResolveExistingFile(skill.RootDirectory, head, out _, out _))
                return message;

            string tail = relativePath.Substring(split).Trim();
            return message
                + $". It looks like the arguments were included in 'path': '{head}' does exist."
                + $" Call skills_run again with path=\"{head}\" and args=\"{tail}\".";
        }

        private static readonly char[] ArgumentSeparators = { ' ', '\t' };

        private bool TryResolveInterpreter(string extension, out string? interpreter, out string? error)
        {
            interpreter = null;
            error = null;

            if (_options.Interpreters.TryGetValue(extension, out string? configured))
            {
                interpreter = configured;
                return true;
            }

            error = extension.Length == 0
                ? "it has no file extension, so there is no interpreter for it. Only "
                  + string.Join(", ", _options.Interpreters.Keys) + " files can be run."
                : $"'{extension}' files cannot be run here. Only "
                  + string.Join(", ", _options.Interpreters.Keys) + " files can be.";
            return false;
        }

        /// <summary>
        /// Split a command line into an argument vector the way a POSIX shell would —
        /// honouring single quotes, double quotes and backslash escapes — WITHOUT
        /// invoking a shell.
        ///
        /// <para>
        /// The model writes arguments as one string because a tool parameter cannot be
        /// an array here. Passing that string to a shell would make every metacharacter
        /// in it executable, so it is split in process and handed over as separate
        /// arguments instead: <c>--out "my file.pdf"; rm -rf ~</c> becomes four literal
        /// arguments, one of which is the harmless text <c>rm</c>.
        /// </para>
        /// </summary>
        internal static List<string> SplitArguments(string? commandLine)
        {
            var arguments = new List<string>();
            if (string.IsNullOrWhiteSpace(commandLine))
                return arguments;

            var current = new StringBuilder();
            bool started = false;
            char quote = '\0';

            for (int i = 0; i < commandLine.Length; i++)
            {
                char c = commandLine[i];

                if (quote != '\0')
                {
                    if (c == quote)
                    {
                        quote = '\0';
                        continue;
                    }
                    if (c == '\\' && quote == '"' && i + 1 < commandLine.Length)
                    {
                        char next = commandLine[i + 1];
                        if (next is '"' or '\\')
                        {
                            current.Append(next);
                            i++;
                            continue;
                        }
                    }
                    current.Append(c);
                    continue;
                }

                if (c is '"' or '\'')
                {
                    quote = c;
                    started = true;
                    continue;
                }

                if (char.IsWhiteSpace(c))
                {
                    if (started)
                    {
                        arguments.Add(current.ToString());
                        current.Clear();
                        started = false;
                    }
                    continue;
                }

                if (c == '\\' && i + 1 < commandLine.Length)
                {
                    current.Append(commandLine[++i]);
                    started = true;
                    continue;
                }

                current.Append(c);
                started = true;
            }

            if (started)
                arguments.Add(current.ToString());
            return arguments;
        }

        /// <summary>What is unconfined when there is no sandbox at all.</summary>
        private static readonly IReadOnlyList<string> AllGaps =
            new SkillSandboxCapabilities(false, false, false, false).Gaps();

    }

    /// <summary>Bounds, isolation policy and interpreter mapping for <see cref="SkillScriptRunner"/>.</summary>
    public sealed class SkillScriptRunnerOptions
    {
        /// <summary>
        /// How hard to insist on OS isolation. <see cref="SkillSandboxMode.Required"/>
        /// by default: a host that cannot confine a script should refuse to run it
        /// rather than run it unconfined.
        /// </summary>
        public SkillSandboxMode Sandbox { get; init; } = SkillSandboxMode.Required;

        /// <summary>
        /// What runs each script. Null is today's desktop behaviour: a confined child
        /// process under the OS sandbox detected for <see cref="Sandbox"/>. A host that
        /// cannot start processes supplies an in-process backend here; its
        /// <see cref="IShellBackend.Sandbox"/> is then what <c>required</c> is judged
        /// against.
        /// </summary>
        public IShellBackend? Backend { get; init; }

        /// <summary>Let the script reach the network. Off by default — the sandbox blocks it.</summary>
        public bool AllowNetwork { get; init; }

        /// <summary>How long a script may run before it is killed.</summary>
        public TimeSpan Timeout { get; init; } = TimeSpan.FromSeconds(60);

        /// <summary>Ceiling on captured stdout and on captured stderr, each.</summary>
        public int MaxOutputBytes { get; init; } = 32 * 1024;

        /// <summary>Where per-run scratch directories are created. Null uses the system temp directory.</summary>
        public string? ScratchDirectory { get; init; }

        /// <summary>
        /// Delete the scratch directory after the run. On by default; turn it off to
        /// keep whatever a script produced.
        /// </summary>
        public bool DeleteScratchDirectory { get; init; } = true;

        /// <summary>Extra paths the script may read, beyond the system and its own skill.</summary>
        public IReadOnlyList<string> ReadablePaths { get; init; } = Array.Empty<string>();

        /// <summary>
        /// The session's persistent workspace. When set, scripts run in its shared
        /// working directory (which survives the call, so one script's output is the
        /// next one's input), can import the packages the session installed via
        /// the shell tool, and per-run scratch handling is bypassed entirely.
        /// </summary>
        public SessionWorkspace? Workspace { get; init; }

        /// <summary>
        /// Keeps the files a script produced for the user to download, returning where
        /// each is fetched from. Supplied by the host (a server points it at its
        /// artifact store); only consulted in workspace mode. Null keeps nothing.
        /// </summary>
        public WorkspaceFileCapture? CaptureProducedFiles { get; init; }

        /// <summary>
        /// Installs packages into the session environment, so a script's missing
        /// dependencies are set up automatically instead of costing the model a
        /// round per import. Wired to the same installer the host uses (wheels
        /// only, allow-list honored). Null disables every install path here.
        /// </summary>
        public ICodeRunner? PackageInstaller { get; init; }

        /// <summary>
        /// Most distinct dependencies one script run may auto-install before it stops
        /// retrying. A bound, not a budget: chains longer than this smell like a
        /// script that is not going to work.
        /// </summary>
        public int MaxAutoInstallAttempts { get; init; } = 5;

        /// <summary>
        /// Extension to interpreter. Replace an entry to point at a virtualenv's
        /// <c>python</c>, or shorten the map to forbid a language outright.
        /// </summary>
        public Dictionary<string, string> Interpreters { get; init; } = new(StringComparer.OrdinalIgnoreCase)
        {
            // The SAME resolution the shell uses (newest Python first), so a package the
            // session installed with one interpreter is importable by the script — a
            // 3.13 pip's wheels under a 3.9 script would not be.
            [".py"] = OperatingSystem.IsWindows()
                ? "python"
                : CodeExec.CodeEnvironment.TryResolveInterpreter(CodeExec.CodeLanguage.Python, out string? python, out _)
                    ? python!
                    : "python3",
            [".js"] = "node",
            [".mjs"] = "node",
            [".sh"] = OperatingSystem.IsWindows() ? "bash" : "/bin/sh",
            [".bash"] = "bash",
        };

        /// <summary>
        /// Environment variables copied from the host into the child. Deliberately
        /// short: everything not listed here is dropped, so a credential the host
        /// happens to carry cannot leak into an uploaded script's stdout.
        /// </summary>
        public IReadOnlyList<string> PassThroughEnvironmentVariables { get; init; } = new[]
        {
            "PATH", "LANG", "LC_ALL", "TZ", "SystemRoot", "COMSPEC", "PATHEXT",
        };

        /// <summary>Extra environment variables for the child process.</summary>
        public Dictionary<string, string> EnvironmentVariables { get; init; } = new(StringComparer.Ordinal);
    }
}
