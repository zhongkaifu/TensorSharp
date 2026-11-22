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
using TensorSharp.AgentHost.Skills;

namespace TensorSharp.AgentHost.CodeExec
{
    /// <summary>How to build, populate and launch one language's environment.</summary>
    /// <param name="Interpreter">Absolute path to the interpreter that runs the snippet.</param>
    /// <param name="Arguments">Extra arguments before the script path, if any.</param>
    /// <param name="ReadablePaths">Paths the run phase must be able to read, e.g. the environment.</param>
    /// <param name="EnvironmentVariables">Variables the run phase needs, e.g. NODE_PATH.</param>
    public readonly record struct CodeLaunchPlan(
        string Interpreter,
        IReadOnlyList<string> Arguments,
        IReadOnlyList<string> ReadablePaths,
        IReadOnlyDictionary<string, string> EnvironmentVariables);

    /// <summary>An install step to run with the network open and nothing else reachable.</summary>
    /// <param name="Interpreter">The installer executable.</param>
    /// <param name="Arguments">Its full argument vector.</param>
    /// <param name="WriteDirectory">The only directory it may write to.</param>
    public readonly record struct CodeInstallPlan(
        string Interpreter,
        IReadOnlyList<string> Arguments,
        string WriteDirectory);

    /// <summary>
    /// Locates interpreters and works out how to build an environment for each language.
    ///
    /// <para>
    /// Everything here is about the gap between "the interpreter is installed" and "we can
    /// launch it under a sandbox". Two of those gaps are Windows-only and both would
    /// quietly defeat the confinement rather than fail loudly, which is why they are
    /// refused by name rather than worked around.
    /// </para>
    /// </summary>
    public static class CodeEnvironment
    {
        /// <summary>The shared executable search path for shell commands and skill scripts.</summary>
        public static string SessionExecutablePath(SessionWorkspace workspace, string? hostPath) =>
            string.Join(Path.PathSeparator, new[]
            {
                Path.Combine(workspace.EnvDirectory, "shim"),
                VenvBin(workspace.EnvDirectory),
                Path.Combine(workspace.EnvDirectory, "bin"),
                Path.Combine(workspace.EnvDirectory, "node_modules", ".bin"),
                Path.Combine(workspace.WorkDirectory, ".local", "bin"),
                OperatingSystem.IsWindows()
                    ? Path.Combine(workspace.WorkDirectory, ".home", ".local", "bin") : string.Empty,
                hostPath ?? string.Empty,
            }.Where(path => path.Length > 0));

        // ---- a host with its own runtime ------------------------------------

        /// <summary>What a host told <see cref="Configure"/> about the runtime it embeds.</summary>
        private sealed record HostRuntime(
            IReadOnlyList<string> AvailableTools,
            Func<CodeLanguage, string?> ResolveInterpreter,
            Func<string, Version?> PythonVersionOf);

        private static volatile HostRuntime? _configured;
        private static readonly object ConfigureGate = new();

        /// <summary>
        /// Describe an embedded runtime instead of probing PATH.
        ///
        /// <para>
        /// Everything in this class answers "what can this host run" by looking at PATH,
        /// and every answer is cached for the life of the process because PATH does not
        /// change. A host that runs code in-process — an iOS app with CPython and
        /// JavaScriptCore linked in — has no PATH entry for either, so without this the
        /// shell declaration's "On this host:" line would be empty, the interpreter
        /// shims would never be written, and the coaching that names the real
        /// interpreter version would say nothing. The three answers a host has to give
        /// are exactly the three this takes; the caches are dropped so nothing probed
        /// before the call survives it.
        /// </para>
        /// </summary>
        /// <param name="availableTools">What the declaration lists, e.g. <c>python3 3.13</c>, <c>node (JavaScriptCore)</c>.</param>
        /// <param name="resolveInterpreter">The name the backend answers to for a language, or null when it has none.</param>
        /// <param name="pythonVersionOf">The version behind an interpreter name, or null when unknown.</param>
        public static void Configure(
            IReadOnlyList<string> availableTools,
            Func<CodeLanguage, string?> resolveInterpreter,
            Func<string, Version?> pythonVersionOf)
        {
            ArgumentNullException.ThrowIfNull(availableTools);
            ArgumentNullException.ThrowIfNull(resolveInterpreter);
            ArgumentNullException.ThrowIfNull(pythonVersionOf);

            lock (ConfigureGate)
            {
                _configured = new HostRuntime(availableTools.ToArray(), resolveInterpreter, pythonVersionOf);
                ClearCaches();
            }
        }

        /// <summary>Back to probing PATH, with every cache dropped. For tests.</summary>
        public static void Reset()
        {
            lock (ConfigureGate)
            {
                _configured = null;
                ClearCaches();
            }
        }

        /// <summary>True between <see cref="Configure"/> and <see cref="Reset"/>.</summary>
        public static bool IsConfigured => _configured != null;

        private static void ClearCaches()
        {
            WhichCache.Clear();
            InterpreterVersions.Clear();
            _available = null;
            CodeDiagnostics.ResetCaches();
        }

        /// <summary>Names to try for each language's interpreter, in order.</summary>
        private static readonly Dictionary<CodeLanguage, string[]> Candidates = new()
        {
            // Newest first: skill scripts routinely use post-3.9 syntax (`match`,
            // PEP 604 unions), and on macOS the bare `python3` is Apple's frozen 3.9.
            // A host with a modern interpreter installed should never fail on syntax
            // the script's own authors consider baseline.
            [CodeLanguage.Python] = OperatingSystem.IsWindows()
                ? new[] { "python.exe", "python3.exe" }
                : new[] { "python3.14", "python3.13", "python3.12", "python3.11", "python3.10", "python3", "python" },
            [CodeLanguage.JavaScript] = OperatingSystem.IsWindows()
                ? new[] { "node.exe" }
                : new[] { "node" },
        };

        /// <summary>
        /// Prefer a runtime installed in this session's conventional .local prefix.
        /// Resolve every link inside the work tree before selecting it; an executable
        /// link must never authorize reading a host runtime or the user's home.
        /// Embedded hosts keep their configured interpreter resolution unchanged.
        /// </summary>
        public static bool TryResolveSessionInterpreter(
            SessionWorkspace workspace, CodeLanguage language, out string? path, out string? error)
        {
            if (!IsConfigured && Candidates.TryGetValue(language, out string[]? names))
            {
                string root = Path.GetFullPath(workspace.WorkDirectory);
                foreach (string name in names)
                {
                    // Preserve the work-relative prefix, and also honor $HOME/.local
                    // when Windows HOME is the shared private .home profile.
                    string[] prefixes = OperatingSystem.IsWindows()
                        ? new[] { root, Path.Combine(root, ".home") } : new[] { root };
                    foreach (string prefix in prefixes)
                    {
                        string candidate = Path.Combine(prefix, ".local", "bin", name);
                        if (SkillPathGuard.TryResolveSymlinks(root, candidate, out string? resolved, out _)
                            && resolved != null && IsUsableExecutable(resolved))
                        {
                            path = resolved;
                            error = null;
                            return true;
                        }
                    }
                }
            }
            return TryResolveInterpreter(language, out path, out error);
        }

        /// <summary>
        /// Find the interpreter for <paramref name="language"/>, or explain why it cannot
        /// be used here.
        /// </summary>
        public static bool TryResolveInterpreter(
            CodeLanguage language, out string? path, out string? error)
        {
            path = null;
            error = null;

            if (_configured is { } configured)
            {
                path = configured.ResolveInterpreter(language);
                if (!string.IsNullOrEmpty(path))
                    return true;
                path = null;
                error = $"this host's runtime has no interpreter for {CodeExecOptions.NameOf(language)}";
                return false;
            }

            if (!Candidates.TryGetValue(language, out string[]? names))
            {
                error = $"'{CodeExecOptions.NameOf(language)}' is not a language this host can run";
                return false;
            }

            foreach (string name in names)
            {
                string? found = Which(name);
                if (found == null)
                    continue;

                path = found;
                return true;
            }

            error = $"no interpreter for {CodeExecOptions.NameOf(language)} was found on PATH "
                  + $"(looked for: {string.Join(", ", names)})";
            return false;
        }

        /// <summary>
        /// The major.minor of a Python interpreter, probed once per path and cached.
        /// Null when the probe fails. Used to turn "SyntaxError: invalid syntax" on a
        /// 3.9 host into "your Python is too old", which is actionable.
        /// </summary>
        public static Version? PythonVersionOf(string interpreter)
        {
            if (_configured is { } configured)
                return InterpreterVersions.GetOrAdd(interpreter, configured.PythonVersionOf);

            return InterpreterVersions.GetOrAdd(interpreter, path =>
            {
                try
                {
                    // Old Pythons print the version to stderr and new ones to stdout, so
                    // both are collected and searched together.
                    var output = new System.Text.StringBuilder();
                    void Collect(string line) { lock (output) output.Append(line).Append('\n'); }

                    var request = new SpawnRequest
                    {
                        FileName = path,
                        Arguments = new[] { "--version" },
                        Environment = new Dictionary<string, string>(StringComparer.Ordinal)
                        {
                            ["PATH"] = Environment.GetEnvironmentVariable("PATH") ?? string.Empty,
                            ["LANG"] = "C.UTF-8",
                        },
                        OnStdoutLine = Collect,
                        OnStderrLine = Collect,
                    };

                    // The caching is why a hang here would matter more than a one-line probe
                    // suggests: whatever the first caller concluded is memoised for every
                    // caller after it.
                    if (!SpawnedProcess.TryStart(request, out SpawnedProcess? started, out _)
                        || started == null)
                    {
                        return null;
                    }

                    using var process = started;
                    process.WaitForExit(5000);
                    process.WaitForDrain(1000);

                    string text;
                    lock (output) text = output.ToString();
                    var match = System.Text.RegularExpressions.Regex.Match(text, @"Python (\d+)\.(\d+)");
                    return match.Success
                        ? new Version(int.Parse(match.Groups[1].Value), int.Parse(match.Groups[2].Value))
                        : null;
                }
                catch (Exception)
                {
                    return null;
                }
            });
        }

        private static readonly System.Collections.Concurrent.ConcurrentDictionary<string, Version?> InterpreterVersions = new();

        /// <summary>
        /// The interpreters and tools this host actually has, named for the model.
        ///
        /// <para>
        /// This is what replaced <c>--code-exec-languages</c>. A shell reaches every
        /// interpreter on PATH and PATH has to contain <c>/bin</c> and <c>/usr/bin</c> for
        /// the shell to work at all, so a flag claiming to gate languages could never have
        /// been honoured. Reporting is the honest version of the same information — and it
        /// is worth more to the model than the flag ever was, because a model that is told
        /// what exists does not spend a round running <c>which python3</c> to find out, and
        /// does not write a program for an interpreter this host has never had.
        /// </para>
        /// <para>
        /// Probed once and cached: the answer cannot change while the process runs, and the
        /// result goes into a tool description that sits in prefix-cache block zero, where
        /// anything that varies between turns costs the whole conversation its prefix reuse.
        /// </para>
        /// </summary>
        public static IReadOnlyList<string> AvailableTools
        {
            get
            {
                if (_configured is { } configured)
                    return configured.AvailableTools;
                IReadOnlyList<string>? probed = _available;
                if (probed != null)
                    return probed;
                lock (ConfigureGate)
                    return _available ??= ProbeAvailable();
            }
        }

        // A field rather than a Lazy, so Configure/Reset can drop it: a Lazy holds its
        // first answer for the life of the process, and that answer was read off a PATH
        // the host has just replaced.
        private static volatile IReadOnlyList<string>? _available;

        private static IReadOnlyList<string> ProbeAvailable()
        {
            var found = new List<string>();

            if (TryResolveInterpreter(CodeLanguage.Python, out string? python, out _) && python != null)
            {
                // Only when the NAME does not already carry it: "python3.12 3.12" is noise,
                // while a bare "python3" that is really 3.9 is exactly what a model needs to
                // know before it writes a match statement.
                string name = System.IO.Path.GetFileName(python);
                Version? version = PythonVersionOf(python);
                found.Add(version != null && !name.Any(char.IsAsciiDigit)
                    ? name + " " + version
                    : version != null && !name.Contains(version.ToString(), StringComparison.Ordinal)
                        ? name + " (" + version + ")"
                        : name);
            }
            if (TryResolveInterpreter(CodeLanguage.JavaScript, out string? node, out _) && node != null)
                found.Add(System.IO.Path.GetFileName(node));

            // Tools a model reaches for constantly and cannot install: naming the ones that
            // are here is cheaper than a failed command, and NOT naming the rest is what
            // stops it planning around ffmpeg on a host that has none.
            // Deliberately no curl or wget: they exist on nearly every host and nothing
            // here can reach the network, so naming them would invite exactly the plan
            // ("I will just fetch it") that the network rule is there to prevent.
            // pdftoppm is here because its ABSENCE cost two logged rounds: the pdf and
            // pptx skills tell the model to render a page to an image, and a program
            // cannot be installed by any package manager here — so discovering it is
            // missing three commands into a plan is a wasted plan, not a fixable error.
            foreach (string name in new[]
            {
                "git", "rg", "jq", "ffmpeg", "pandoc", "soffice", "pdftoppm",
                "dotnet", "go", "cargo", "make",
            })
            {
                if (Which(name) != null)
                    found.Add(name);
            }

            return found;
        }

        /// <summary>Resolve a bare executable name against PATH.</summary>
        public static string? Which(string name) =>
            Path.IsPathRooted(name)
                ? (IsUsableExecutable(name) ? name : ResolveRootedWithExtension(name))
                : WhichCache.GetOrAdd(name, ProbePath);

        /// <summary>
        /// A rooted name that exists as given, or with one of Windows' executable
        /// extensions appended. <c>--code-exec-shell C:	ools\pwsh</c> is a path a
        /// person would write and a path that does not exist.
        /// </summary>
        private static string? ResolveRootedWithExtension(string name)
        {
            if (!OperatingSystem.IsWindows() || Path.HasExtension(name))
                return null;
            foreach (string extension in WindowsExecutableExtensions)
            {
                if (IsUsableExecutable(name + extension))
                    return name + extension;
            }
            return null;
        }

        /// <summary>
        /// The extensions a bare name may be carrying on Windows, in PATHEXT order.
        ///
        /// <para>
        /// Without this, <see cref="Which"/> looked for a file literally named
        /// <c>git</c> beside a <c>git.exe</c> and concluded the host had no git. That
        /// is not a cosmetic miss: <see cref="AvailableTools"/> is the list this host
        /// puts in the model's tool description to say what it can reach, and on
        /// Windows it came back with nothing but the interpreters, which are the only
        /// two candidates spelled with <c>.exe</c> in the table above. A model told the
        /// host has no <c>git</c>, no <c>ffmpeg</c>, no <c>soffice</c> and no
        /// <c>pdftoppm</c> plans around tools that are installed and on PATH.
        /// </para>
        /// <para>
        /// PATHEXT is read rather than assumed, because an operator may have added to
        /// it, but <c>.EXE</c>, <c>.CMD</c> and <c>.BAT</c> are forced in: a stripped
        /// service environment can arrive without PATHEXT at all, and a host that then
        /// finds no executables would be back where it started. <c>.COM</c> is left to
        /// PATHEXT; nothing here ships one.
        /// </para>
        /// </summary>
        private static readonly string[] WindowsExecutableExtensions = BuildExecutableExtensions();

        private static string[] BuildExecutableExtensions()
        {
            if (!OperatingSystem.IsWindows())
                return Array.Empty<string>();

            var extensions = new List<string>();
            void Add(string extension)
            {
                if (extension.Length > 1
                    && extension[0] == '.'
                    && !extensions.Contains(extension, StringComparer.OrdinalIgnoreCase))
                {
                    extensions.Add(extension);
                }
            }

            foreach (string entry in (Environment.GetEnvironmentVariable("PATHEXT") ?? string.Empty)
                         .Split(';', StringSplitOptions.RemoveEmptyEntries))
            {
                Add(entry.Trim());
            }

            // Order matters, and this is the order CreateProcess itself uses: an
            // interpreter shipped as both a .exe and a .cmd shim must resolve to the
            // .exe, because a .cmd is a batch file that ProcessStartInfo with
            // UseShellExecute=false cannot start at all.
            foreach (string forced in new[] { ".EXE", ".CMD", ".BAT" })
                Add(forced);

            return extensions.ToArray();
        }

        /// <summary>
        /// Answers remembered for the life of the process.
        ///
        /// <para>
        /// A lookup walks every PATH entry plus the two Homebrew directories — a filesystem
        /// probe per candidate — and it is on paths that run per call: the syntax check
        /// resolves two interpreters every time it verifies a file, and the
        /// spelled-differently diagnosis probes on every command-not-found. What is on
        /// PATH does not change while the server runs, and the interpreter versions beside
        /// this are cached for exactly the same reason.
        /// </para>
        /// </summary>
        private static readonly System.Collections.Concurrent.ConcurrentDictionary<string, string?> WhichCache =
            new(StringComparer.Ordinal);

        /// <summary>
        /// A file that exists AND is not one of Windows' App Execution Aliases.
        ///
        /// <para>
        /// <c>%LOCALAPPDATA%\Microsoft\WindowsApps</c> is on every Windows user's PATH
        /// and ships zero-byte reparse stubs named <c>python.exe</c> and
        /// <c>python3.exe</c>. Running one does not run Python: it prints "Python was not
        /// found; run without arguments to install from the Microsoft Store" and exits
        /// 9009. A plain <c>File.Exists</c> cannot tell it apart from an interpreter, and
        /// the consequences ran in both directions.
        /// </para>
        /// <para>
        /// It made <c>python3</c> fail outright. <see cref="ShellRunner"/> writes a
        /// <c>python3</c> shim precisely so a model that types the name every SKILL.md
        /// uses gets this host's interpreter — but only when no <c>python3</c> already
        /// resolves. The stub resolved, so no shim was written, and every
        /// <c>python3 scripts/thumbnail.py</c> came back as exit 9009 and an advertisement
        /// for the Store.
        /// </para>
        /// <para>
        /// And on a machine where WindowsApps precedes the real install on PATH it would
        /// have been worse: <c>python</c> ITSELF would resolve to the stub, and the host
        /// would report an interpreter it does not have.
        /// </para>
        /// <para>
        /// Zero length is the test because that is what the stubs are — the reparse point
        /// carries the target and the file has no data. It costs one stat on a path that
        /// is cached for the life of the process.
        /// </para>
        /// </summary>
        private static bool IsUsableExecutable(string path)
        {
            if (!File.Exists(path))
                return false;
            if (!OperatingSystem.IsWindows())
                return true;
            try { return new FileInfo(path).Length > 0; }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                return true;         // cannot tell, so do not remove a real candidate
            }
        }

        private static string? ProbePath(string name)
        {

            string? pathVar = Environment.GetEnvironmentVariable("PATH");
            if (!string.IsNullOrEmpty(pathVar))
            {
                foreach (string dir in pathVar.Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries))
                {
                    string candidate;
                    try { candidate = Path.Combine(dir.Trim(), name); }
                    catch (ArgumentException) { continue; }      // a malformed PATH entry

                    if (IsUsableExecutable(candidate))
                        return candidate;

                    // Windows: a bare name on PATH is spelled without its extension.
                    // Skipped when the caller already gave one, so "python.exe" is not
                    // probed as "python.exe.exe".
                    if (!Path.HasExtension(name))
                    {
                        foreach (string extension in WindowsExecutableExtensions)
                        {
                            if (IsUsableExecutable(candidate + extension))
                                return candidate + extension;
                        }
                    }
                }
            }

            // A server launched outside a login shell (launchd, a service manager, a
            // stripped environment) misses Homebrew's directory even though the
            // interpreter is right there. These two are where user-installed tools
            // land on macOS; checking them beats "no interpreter found" on a machine
            // that plainly has one.
            if (!OperatingSystem.IsWindows())
            {
                foreach (string dir in new[] { "/opt/homebrew/bin", "/usr/local/bin" })
                {
                    string candidate = Path.Combine(dir, name);
                    if (File.Exists(candidate))
                        return candidate;
                }
            }

            return null;
        }

        // ---- windows --------------------------------------------------------

        /// <summary>
        /// Add the variables a Windows child needs, and redirect the per-user ones into
        /// <paramref name="workDirectory"/>.
        ///
        /// <para>
        /// One copy, called from both confined-launch paths, because there ARE two of
        /// them (<c>ConfinedProcess</c> and <c>SkillScriptRunner.RunConfined</c>) and
        /// they had already drifted: between them they passed <c>SYSTEMROOT</c>,
        /// <c>COMSPEC</c> and <c>PATHEXT</c> and nothing else, which is short of what
        /// Windows actually needs by a long way.
        /// </para>
        /// <para>
        /// What was missing and what it cost:
        /// <list type="bullet">
        /// <item><c>USERPROFILE</c> — CPython's <c>ntpath.expanduser</c> reads
        /// USERPROFILE and does NOT read HOME, so <c>~</c> in a skill script resolved to
        /// the REAL user profile even though HOME had been redirected. Every scrubbing
        /// guarantee about the home directory was a no-op for the one call that asks for
        /// it by name.</item>
        /// <item><c>TEMP</c> / <c>TMP</c> — set here to the work directory. Python reads
        /// TMPDIR first so it coped, but nothing else on Windows does: a tool writing a
        /// temporary file found neither and fell back to the current directory or to a
        /// path it had no business using.</item>
        /// <item><c>LOCALAPPDATA</c> / <c>APPDATA</c> — pip's cache, npm's cache and
        /// prefix, and a long tail of tools. Redirected into the work directory rather
        /// than passed through, for the same reason HOME is: a session's caches belong
        /// in that session's workspace. This is also the variable an AppContainer launch
        /// REQUIRES — CreateProcess fails with ERROR_ENVVAR_NOT_FOUND (203) without
        /// it — which is how its absence was found.</item>
        /// <item><c>PSMODULEPATH</c> — PowerShell is the shell on Windows, and without
        /// it a script that calls anything outside the core snap-ins fails to bind.</item>
        /// <item><c>SYSTEMDRIVE</c>, <c>WINDIR</c>, <c>PROGRAMDATA</c>,
        /// <c>PROGRAMFILES</c>, <c>PROGRAMFILES(X86)</c>, <c>PUBLIC</c>,
        /// <c>NUMBER_OF_PROCESSORS</c>, <c>PROCESSOR_ARCHITECTURE</c>, <c>OS</c> — they
        /// name the machine and the OS install, not the user, so they carry nothing
        /// sensitive, and a surprising amount of tooling reads them.</item>
        /// </list>
        /// </para>
        /// <para>
        /// Deliberately still absent: <c>USERNAME</c>, <c>USERDOMAIN</c>,
        /// <c>COMPUTERNAME</c>, <c>HOMEDRIVE</c>, <c>HOMEPATH</c>, <c>SESSIONNAME</c>,
        /// and everything else that identifies the person rather than the machine.
        /// </para>
        /// </summary>
        public static void ApplyWindowsBaseline(
            IDictionary<string, string> environment, string workDirectory)
        {
            ArgumentNullException.ThrowIfNull(environment);
            if (!OperatingSystem.IsWindows())
                return;

            foreach (string name in WindowsPassThrough)
            {
                if (Environment.GetEnvironmentVariable(name) is { Length: > 0 } value)
                    environment[name] = value;
            }

            // Per-user state, pointed into ONE hidden directory rather than at the
            // working directory itself.
            //
            // Pointing them straight at the work directory is what the POSIX side does
            // with HOME, and on Windows it does not stay invisible: PowerShell creates
            // %USERPROFILE%\AppData\Roaming on startup and pip creates a cache under
            // TEMP, so after two commands the directory the model lists with
            // Get-ChildItem contained AppData/ and pip/ beside its own files. That is
            // not cosmetic - it is what the model reads to decide what it has produced,
            // and it is what artifact capture walks to decide what to offer the user.
            //
            // A single dot-directory keeps all of it in one place that both walkers
            // already skip by name, and the hidden attribute keeps it out of a bare
            // Get-ChildItem too, so the model never has to be told to ignore it.
            string home = EnsureSubdirectory(workDirectory, ".home", hidden: true);
            environment["HOME"] = home;
            environment["USERPROFILE"] = home;
            environment["TEMP"] = EnsureSubdirectory(home, "Temp");
            environment["TMP"] = environment["TEMP"];
            // CPython reads TMPDIR BEFORE TEMP, on Windows as well as on Unix, so leaving
            // the caller's value in place would send every tempfile.* call back into the
            // working directory and undo the redirection above for the one language every
            // skill script here is written in.
            environment["TMPDIR"] = environment["TEMP"];
            // Native Windows applications also resolve these through SHGetFolderPath,
            // which expands USERPROFILE\AppData rather than reading LOCALAPPDATA.
            // Keep the environment and the native folder APIs in agreement, and create
            // the directories: Chrome refuses its automation connection when the
            // non-creating native lookup cannot determine the default profile location.
            string appData = EnsureSubdirectory(home, "AppData");
            environment["LOCALAPPDATA"] = EnsureSubdirectory(appData, "Local");
            environment["APPDATA"] = EnsureSubdirectory(appData, "Roaming");

            // CPython on Windows picks its stdio codec from the console code page, which
            // for a redirected pipe means cp1252. A script that prints one character
            // outside it - an arrow, an em dash, a CJK title, anything a slide deck or a
            // document skill routinely handles - dies with UnicodeEncodeError partway
            // through, and the model reads a traceback about encoding rather than the
            // output it asked for. PYTHONUTF8 also makes open() default to UTF-8, which
            // is what every skill script here already assumes when it writes a file.
            // Both are ignored by a non-Python child.
            environment["PYTHONUTF8"] = "1";
            environment["PYTHONIOENCODING"] = "utf-8";
        }

        private static string EnsureSubdirectory(string parent, string name, bool hidden = false)
        {
            string path = Path.Combine(parent, name);
            try
            {
                Directory.CreateDirectory(path);
                if (hidden && OperatingSystem.IsWindows())
                    File.SetAttributes(path, File.GetAttributes(path) | FileAttributes.Hidden);
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                // The work directory itself is always writable, and is a correct answer
                // for every consumer of these variables. Naming a directory that could
                // not be created would not be.
                return parent;
            }
            return path;
        }

        /// <summary>Machine-describing variables copied through unchanged. See <see cref="ApplyWindowsBaseline"/>.</summary>
        private static readonly string[] WindowsPassThrough =
        {
            "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC", "PATHEXT",
            "PROGRAMDATA", "PROGRAMFILES", "PROGRAMFILES(X86)", "PROGRAMW6432", "PUBLIC",
            "PSMODULEPATH", "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE", "OS",
        };

        // ---- python --------------------------------------------------------

        /// <summary>Where a venv puts its executables — <c>Scripts</c> on Windows, <c>bin</c> elsewhere.</summary>
        public static string VenvBin(string envDirectory) =>
            Path.Combine(envDirectory, OperatingSystem.IsWindows() ? "Scripts" : "bin");

        /// <summary>
        /// Install packages into <paramref name="envDirectory"/> using the HOST interpreter's
        /// pip with an explicit target, rather than a pip inside the venv.
        ///
        /// <para>
        /// <c>--only-binary=:all:</c> is the load-bearing argument and the reason this is
        /// safe enough to offer at all. Without it pip will happily fetch a source
        /// distribution and execute its <c>setup.py</c> — arbitrary code, running as the
        /// host user, before a single line of the model's own snippet is reached, and
        /// before any sandbox that confines the RUN phase applies. With it, only prebuilt
        /// wheels are accepted: they are unpacked, never executed. A package with no wheel
        /// for this platform fails to install, and that is the correct trade.
        /// </para>
        /// <para>
        /// <c>--no-input</c> and <c>--disable-pip-version-check</c> keep it from ever
        /// blocking on a prompt, which would otherwise consume the whole install timeout.
        /// </para>
        /// </summary>
        /// <param name="allowSourceBuilds">
        /// Let pip build a package that publishes no wheel from its source distribution,
        /// preferring a wheel whenever one exists. Only for a launch whose sandbox confines
        /// writes and pins the network to the package index (see
        /// <see cref="PackageInstaller"/>): a source build runs the package's own setup
        /// code. False keeps the default of prebuilt wheels only.
        /// </param>
        public static CodeInstallPlan PythonInstall(
            string interpreter, string envDirectory, IReadOnlyList<string> packages,
            string? indexUrl = null, bool allowSourceBuilds = false)
        {
            var args = new List<string>
            {
                "-m", "pip", "install",
                allowSourceBuilds ? "--prefer-binary" : "--only-binary=:all:",
                "--no-input",
                "--disable-pip-version-check",
                "--no-warn-script-location",
                "--target", envDirectory,
            };
            // The OPERATOR's index, never the model's. A --index-url the model wrote is
            // refused by PackageInstaller and must stay refused; this one comes from
            // --code-exec-install-index, and it is the only way a host that cannot reach
            // pypi.org can install anything at all.
            if (!string.IsNullOrWhiteSpace(indexUrl))
            {
                args.Add("--index-url");
                args.Add(indexUrl!);
            }
            args.AddRange(packages);
            return new CodeInstallPlan(interpreter, args, envDirectory);
        }

        // ---- javascript ----------------------------------------------------

        /// <summary>
        /// The npm install step, launched as <c>node npm-cli.js</c> rather than <c>npm</c>.
        ///
        /// <para>
        /// Windows, problem two: <c>npm</c> is <c>npm.cmd</c>, a batch file, and
        /// <see cref="System.Diagnostics.ProcessStartInfo"/> with
        /// <c>UseShellExecute = false</c> — which every confined launch here uses — cannot
        /// start one. The usual workaround is to launch <c>cmd.exe /c npm</c>, which puts a
        /// command interpreter inside the sandbox with the model's package names as its
        /// arguments; that is a shell injection surface introduced purely to run an
        /// installer. Invoking npm's own entry script through <c>node</c> avoids the batch
        /// file, the shell, and the difference between platforms all at once.
        /// </para>
        /// </summary>
        public static bool TryNpmInstall(
            string nodeInterpreter,
            string envDirectory,
            IReadOnlyList<string> packages,
            out CodeInstallPlan plan,
            out string? error)
        {
            plan = default;
            string? cli = FindNpmCli(nodeInterpreter);
            if (cli == null)
            {
                error = "npm's entry script (npm-cli.js) was not found next to node, so packages "
                      + "cannot be installed for javascript on this host";
                return false;
            }

            var args = new List<string>
            {
                cli, "install",
                "--prefix", envDirectory,
                "--no-audit", "--no-fund", "--no-package-lock",
                // Node's equivalent of the wheels-only rule: a package's install scripts
                // are arbitrary code running as the host user, before the sandbox that
                // confines the run phase ever applies.
                "--ignore-scripts",
            };
            args.AddRange(packages);

            plan = new CodeInstallPlan(nodeInterpreter, args, envDirectory);
            error = null;
            return true;
        }

        /// <summary>Locate <c>npm-cli.js</c> relative to the node executable.</summary>
        private static string? FindNpmCli(string nodeInterpreter)
        {
            string? dir = Path.GetDirectoryName(nodeInterpreter);
            if (string.IsNullOrEmpty(dir))
                return null;

            // Windows keeps npm beside node.exe; Unix installs put it one level up in lib/.
            string[] candidates =
            {
                Path.Combine(dir, "node_modules", "npm", "bin", "npm-cli.js"),
                Path.Combine(dir, "..", "lib", "node_modules", "npm", "bin", "npm-cli.js"),
                Path.Combine(dir, "..", "node_modules", "npm", "bin", "npm-cli.js"),
            };

            return candidates.Select(c => Path.GetFullPath(c)).FirstOrDefault(File.Exists);
        }

        // ---- shell ---------------------------------------------------------

    }
}
