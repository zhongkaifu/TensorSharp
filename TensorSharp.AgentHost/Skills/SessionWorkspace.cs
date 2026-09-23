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
using System.IO;
using System.IO.Enumeration;
using System.Linq;
using System.Text;
using System.Threading;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp.Runtime.Logging;

namespace TensorSharp.AgentHost.Skills
{
    /// <summary>
    /// The one walk of a session's working directory, and the one list of directories
    /// it does not descend into.
    ///
    /// <para>
    /// Every shell command walks this tree twice — once before, for the "what was here
    /// already" snapshot, and once after, to capture what the command produced — so the
    /// cost of the walk is paid by <c>echo hi</c> exactly as much as by a build. A real
    /// session installs packages, and then the working directory holds a
    /// <c>node_modules</c> or a <c>.venv</c>: tens of thousands of files that no filter
    /// applied AFTER enumeration can save, because the enumeration has already stat'd
    /// every one of them. Measured on a 20k-file <c>node_modules</c>, the two walks cost
    /// 182 ms per command against 0.02 ms on an empty workspace — a trivial command
    /// spending a fifth of a second looking at files nobody asked about.
    /// </para>
    /// <para>
    /// So the junk is skipped at the only place where skipping is free: the recursion
    /// itself. <see cref="Files"/> never opens a pruned directory, and the two call
    /// sites share this list rather than each keeping their own — a snapshot that
    /// descends where the capture does not would report every file under it as newly
    /// produced.
    /// </para>
    /// </summary>
    internal static class WorkspaceScan
    {
        /// <summary>
        /// Directory names that are a tool's own storage wherever they appear.
        ///
        /// <para>
        /// At ANY depth, because that is where they actually occur: npm nests a
        /// <c>node_modules</c> inside a package whose dependency versions conflict, a
        /// virtualenv lives at <c>myproject/.venv</c> rather than at the workspace root,
        /// and CPython drops a <c>__pycache__</c> beside every package it imports.
        /// </para>
        /// </summary>
        private static readonly HashSet<string> PrunedAnywhere = new(StringComparer.Ordinal)
        {
            "node_modules", "__pycache__", ".venv", ".git",
            ".npm", ".cargo", ".gradle", ".m2",
            ".config", ".cache", ".local", ".fontconfig",
            ".pytest_cache", ".ruff_cache", ".mypy_cache",
        };

        /// <summary>
        /// Directory names pruned only directly under the working directory.
        ///
        /// <para>
        /// These are fallout from HOME pointing at the working directory, so the root is
        /// the only place they can be that. <c>Library</c> in particular is an ordinary
        /// English word — a model asked to organise documents may well create one — and
        /// pruning it at depth would silently swallow the user's own output.
        /// </para>
        /// <para>
        /// <c>AppData</c> and <c>pip</c> are the Windows half of the same list, and it was
        /// missing: PowerShell creates <c>AppData\Roaming</c> the moment it starts, and
        /// pip writes a cache tree, both directly in the working directory. Two commands
        /// into a session the model's own <c>Get-ChildItem</c> showed them beside its
        /// files, and artifact capture counted them as things the run had produced. Both
        /// are ordinary-looking names, so like <c>Library</c> they are pruned at the root
        /// only — a <c>pip</c> directory the model deliberately creates inside a project it
        /// is building is its own output and must survive.
        /// </para>
        /// </summary>
        private static readonly HashSet<string> PrunedAtRoot = new(StringComparer.Ordinal)
        {
            "Library", ".jobs", "AppData", "pip",
        };

        private static readonly HashSet<string>.AlternateLookup<ReadOnlySpan<char>> AnywhereLookup =
            PrunedAnywhere.GetAlternateLookup<ReadOnlySpan<char>>();

        private static readonly HashSet<string>.AlternateLookup<ReadOnlySpan<char>> AtRootLookup =
            PrunedAtRoot.GetAlternateLookup<ReadOnlySpan<char>>();

        /// <summary>
        /// The enumeration terms of the "what was here before" snapshot: everything,
        /// hidden files included, but never through a symbolic link.
        ///
        /// <para>
        /// The link rule is not tidiness. <c>ln -s . loop</c> is one ordinary command a
        /// model may run inside its own workspace, and a walk that follows it re-walks
        /// the workspace once per level until the kernel's symlink limit stops it —
        /// measured at 3,747 entries and 55 ms from a working directory holding a single
        /// file, and multiplying with every file added after. <c>ln -s / x</c> is worse.
        /// Capture already skips reparse points for a different reason (it copies as the
        /// unsandboxed host, so following a link out of the workspace would hand the user
        /// a download link to the host's home), and the two walks must agree.
        /// </para>
        /// <para>
        /// A directory that cannot be read is stepped over rather than ending the walk,
        /// which is what the capture side already does. The old snapshot let the
        /// exception out and kept whatever it had gathered so far, so one unreadable
        /// directory silently truncated the "what was here before" picture — and every
        /// file the walk never reached then looked newly produced.
        /// </para>
        /// </summary>
        internal static readonly EnumerationOptions SnapshotOptions = new()
        {
            RecurseSubdirectories = true,
            AttributesToSkip = FileAttributes.ReparsePoint,
            IgnoreInaccessible = true,
        };

        /// <summary>
        /// Every pruned name and whether it counts only at the root — so a test can pin
        /// the two walks to this one list rather than to a copy of it that drifts.
        /// </summary>
        internal static IEnumerable<(string Name, bool RootOnly)> PrunedNames()
        {
            foreach (string name in PrunedAnywhere)
                yield return (name, false);
            foreach (string name in PrunedAtRoot)
                yield return (name, true);
        }

        /// <summary>True for a directory the walk does not descend into.</summary>
        /// <param name="atRoot">Whether it sits directly in the working directory.</param>
        internal static bool IsPrunedDirectory(ReadOnlySpan<char> name, bool atRoot) =>
            AnywhereLookup.Contains(name) || (atRoot && AtRootLookup.Contains(name));

        /// <summary>The same question for a caller that has a string and a relative path.</summary>
        internal static bool IsPrunedDirectory(string name, bool atRoot) =>
            IsPrunedDirectory(name.AsSpan(), atRoot);

        /// <summary>
        /// Every file under <paramref name="root"/> that is not inside a pruned
        /// directory, as full paths.
        ///
        /// <para>
        /// Identical in what it yields to
        /// <c>Directory.EnumerateFiles(root, "*", options)</c> for everything outside the
        /// pruned directories — same options, same order, same treatment of hidden files
        /// and links — and it simply never opens the ones inside them.
        /// </para>
        /// </summary>
        internal static IEnumerable<string> Files(string root, EnumerationOptions options)
        {
            string normalizedRoot = Path.TrimEndingDirectorySeparator(Path.GetFullPath(root));
            return new FileSystemEnumerable<string>(
                root,
                static (ref FileSystemEntry entry) => entry.ToFullPath(),
                options)
            {
                ShouldIncludePredicate = static (ref FileSystemEntry entry) => !entry.IsDirectory,
                ShouldRecursePredicate = (ref FileSystemEntry entry) =>
                    !IsPrunedDirectory(entry.FileName, entry.Directory.SequenceEqual(normalizedRoot)),
            };
        }
    }

    /// <summary>
    /// One chat session's persistent execution workspace: the working directory every
    /// <c>shell</c> command and every skill script of that session runs in, and the
    /// package environment their installs accumulate into.
    ///
    /// <para>
    /// The per-call scratch that preceded this was correct for isolation but wrong for
    /// WORK: a pptx generated by one script was deleted before <c>validate.py</c> could
    /// check it, and every call re-installed its packages from scratch. Real tasks are
    /// pipelines — generate, then validate, then convert — so the pipeline's files must
    /// outlive each step. The unit of trust is the SESSION: everything in one
    /// conversation already shares a context, so its steps sharing a disk is no new
    /// exposure, while two different sessions never share a workspace.
    /// </para>
    /// <para>
    /// Lifecycle: created on first use, deleted when the session is disposed or reset
    /// (a new chat starts clean), and swept at server startup — a server restart orphans
    /// every session, so anything left under the root is finished business.
    /// </para>
    /// </summary>
    public sealed class SessionWorkspace
    {
        internal const string DirectoryPrefix = "ts-session-";

        private readonly HashSet<string> _installedPackages = new(StringComparer.OrdinalIgnoreCase);
        private readonly HashSet<string> _hostRepairArtifacts = new(
            SkillPathGuard.PathComparison == StringComparison.Ordinal
                ? StringComparer.Ordinal
                : StringComparer.OrdinalIgnoreCase);
        private readonly object _gate = new();
        private readonly object _executionGate = new();
        private int _activeOperations;

        /// <summary>Set once the constructor's own layout pass is done, so that pass is
        /// never mistaken for a repair.</summary>
        private readonly bool _constructed;

        /// <summary>1 when a repair has happened that no tool result has reported yet.</summary>
        private int _rebuiltNoticePending;
        private bool _releaseRequested;
        private Action? _releaseWhenIdle;
        private bool _cleanupRegistrationClosed;

        /// <summary>
        /// The workspace this is a lane of (see <see cref="ForAgent"/>), or null when this
        /// IS the workspace. Every conversation-wide member forwards to it.
        /// </summary>
        private readonly SessionWorkspace? _owner;

        /// <summary>
        /// The lanes handed out so far, by sanitized id. Owner only, under
        /// <see cref="_gate"/>, and allocated on the first <see cref="ForAgent"/> — a
        /// conversation with no sub-agents never has one.
        /// </summary>
        private Dictionary<string, SessionWorkspace>? _lanes;

        /// <summary>The directory under <see cref="StateDirectory"/> that holds every lane's own state.</summary>
        internal const string LanesDirectoryName = "lanes";

        /// <summary>The longest lane id kept; a longer one is truncated to this.</summary>
        internal const int MaxLaneIdLength = 64;

        internal SessionWorkspace(string root, string sessionId = "")
        {
            Id = sessionId;
            Root = root;
            LaneId = string.Empty;
            ShellKey = root;
            WorkDirectory = Path.Combine(root, "work");
            EnvDirectory = Path.Combine(root, "env");
            StateDirectory = Path.Combine(root, "state");
            ShellScriptDirectory = StateDirectory;
            ShellStateDirectory = Path.Combine(StateDirectory, "shell");
            TempDirectory = Path.Combine(root, "tmp");
            EnsureDirectories();
            _constructed = true;
        }

        /// <summary>A lane of <paramref name="owner"/>; see <see cref="ForAgent"/>.</summary>
        /// <param name="laneId">Already sanitized.</param>
        private SessionWorkspace(SessionWorkspace owner, string laneId)
        {
            _owner = owner;

            // Immutable, so copying them IS sharing them: the lane names exactly the
            // directories its owner does, and there is nothing to keep in step.
            Id = owner.Id;
            Root = owner.Root;
            WorkDirectory = owner.WorkDirectory;
            EnvDirectory = owner.EnvDirectory;
            StateDirectory = owner.StateDirectory;
            TempDirectory = owner.TempDirectory;

            // The lane's own. The layout under state/lanes/<id> mirrors the owner's under
            // state/: host-written wrapper scripts at the top (read-only to the sandbox),
            // and the one writable child the wrapper saves cwd and exports into.
            LaneId = laneId;
            ShellKey = owner.Root + "#" + laneId;
            ShellScriptDirectory = Path.Combine(StateDirectory, LanesDirectoryName, laneId);
            ShellStateDirectory = Path.Combine(ShellScriptDirectory, "shell");
            EnsureDirectories();
            _constructed = true;
        }

        /// <summary>
        /// Re-assert the directory layout, and report whether any of it was missing.
        /// </summary>
        /// <remarks>
        /// <para>
        /// Creating these in the constructor alone was an assumption that nothing outside
        /// the process ever removes them, and that assumption does not hold: the workspace
        /// root lives beside the running binary, where a rebuild into that output
        /// directory, a temp reaper, or an ordinary cleanup can take it away while a
        /// conversation is still using it. Recorded 2026-09-10 — a rebuild landed in the
        /// host's output directory mid-conversation and every later <c>shell</c> call for
        /// twenty minutes answered "the command could not be prepared: Could not find a
        /// part of the path '.../state/cmd-1.sh'", <c>echo hello</c> included. Nothing in
        /// the workspace could recover, because the only code that ever created these
        /// directories had already run.
        /// </para>
        /// <para>
        /// So every entry point re-asserts the layout instead of assuming it. Five
        /// existence checks against a command that costs tens of milliseconds is not a
        /// cost worth optimising, and the alternative is a conversation that cannot run
        /// anything until it is restarted.
        /// </para>
        /// <para>
        /// The return value matters as much as the repair: the files that were in there
        /// are gone, so a caller has to be able to TELL the model that rather than let it
        /// discover it as a series of unexplained missing files.
        /// </para>
        /// </remarks>
        public bool EnsureDirectories()
        {
            if (_owner != null)
                return EnsureLaneDirectories(_owner);

            EnsureRealDirectory(Root);
            bool rebuilt = false;
            rebuilt |= CreateIfMissing(WorkDirectory);
            rebuilt |= CreateIfMissing(EnvDirectory);
            rebuilt |= CreateIfMissing(StateDirectory);
            rebuilt |= CreateIfMissing(ShellStateDirectory);
            rebuilt |= CreateIfMissing(TempDirectory);
            if (rebuilt)
                WorkspaceOwner.Stamp(Root);
            if (rebuilt && _constructed)
            {
                Volatile.Write(ref _rebuiltNoticePending, 1);
                NotifyLanesOfRebuild();
            }
            return rebuilt;
        }

        /// <summary>
        /// A lane's form: the owner's layout first — the files are shared, so a lane that
        /// finds them gone repairs them for everyone — then the lane's own shell state,
        /// which the owner's pass does not know about.
        /// </summary>
        private bool EnsureLaneDirectories(SessionWorkspace owner)
        {
            bool rebuilt = owner.EnsureDirectories();
            bool laneRebuilt = false;
            laneRebuilt |= CreateIfMissing(Path.Combine(StateDirectory, LanesDirectoryName));
            laneRebuilt |= CreateIfMissing(ShellScriptDirectory);
            laneRebuilt |= CreateIfMissing(ShellStateDirectory);
            if (laneRebuilt && _constructed)
                Volatile.Write(ref _rebuiltNoticePending, 1);
            return rebuilt | laneRebuilt;
        }

        /// <summary>
        /// Carry a rebuild to every lane's latch as well as the owner's.
        ///
        /// <para>
        /// The latch is consumed by whichever tool result reports it first, and with
        /// sub-agents that is a race between agents. One latch would tell whichever agent
        /// happened to run next and leave the others — the parent, typically, which wrote
        /// most of the files and outlives every sub-agent — to rediscover the loss one
        /// missing file at a time. Each agent is told once, in its own next result.
        /// </para>
        /// </summary>
        private void NotifyLanesOfRebuild()
        {
            lock (_gate)
            {
                if (_lanes == null)
                    return;
                foreach (SessionWorkspace lane in _lanes.Values)
                    Volatile.Write(ref lane._rebuiltNoticePending, 1);
            }
        }

        private static bool CreateIfMissing(string path)
        {
            EnsureRealDirectory(path);
            if (Directory.Exists(path))
                return false;
            Directory.CreateDirectory(path);
            return true;
        }

        private static void EnsureRealDirectory(string path)
        {
            if (new DirectoryInfo(path).LinkTarget != null)
                throw new IOException("The session directory was replaced by a symbolic link: " + path);
        }

        /// <summary>
        /// <see cref="EnsureDirectories"/> where a failure to repair must not become the
        /// caller's exception — the tool it precedes has its own refusal to report, and a
        /// raw <c>IOException</c> from a repair attempt would replace a sentence the model
        /// can act on with one it cannot.
        /// </summary>
        public bool TryEnsureDirectories()
        {
            try { return EnsureDirectories(); }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { return false; }
        }

        /// <summary>
        /// Whether a rebuild has happened since this was last asked, clearing the flag.
        /// </summary>
        /// <remarks>
        /// A latch rather than the return of <see cref="EnsureDirectories"/> because the
        /// repair and the telling happen in different places: whichever entry point runs
        /// first does the repair, and the next tool RESULT is what actually reaches the
        /// model. Reporting it once is the point — the workspace is whole again after the
        /// first repair, and repeating the notice on every later command would tell a
        /// model its files had just vanished when nothing had happened at all.
        /// <para>
        /// Per lane: a rebuild is carried to every lane's latch (see
        /// <see cref="NotifyLanesOfRebuild"/>), so each agent is told once.
        /// </para>
        /// </remarks>
        public bool ConsumeRebuiltNotice() =>
            Interlocked.Exchange(ref _rebuiltNoticePending, 0) == 1;

        /// <summary>
        /// What to tell the model when the layout had to be rebuilt: the files it wrote
        /// earlier in this conversation are not there any more, and it will otherwise
        /// spend rounds re-discovering that one missing file at a time.
        /// </summary>
        public const string RebuiltNote =
            "note: this conversation's working directory was missing and has been recreated - "
            + "something outside this session removed it. Files written by earlier steps are gone; "
            + "installed packages are gone too. Recreate what you still need before using it.";

        /// <summary>
        /// This workspace as one agent sees it, when several agents — a parent and the
        /// sub-agents it spawned — work in it at once.
        /// </summary>
        /// <remarks>
        /// <para>
        /// The agents share the FILES: that is the point of spawning a helper into the
        /// same conversation, and it is what Codex does. What they must not share is the
        /// state that belongs to one context window. Before lanes, a sub-agent's
        /// <c>cd sub</c> silently moved the parent's next command into <c>sub</c>, and a
        /// file the sub-agent read counted as read by the parent. A lane is a view that
        /// forwards everything conversation-wide to this workspace and keeps its own
        /// copy of the rest:
        /// </para>
        /// <list type="bullet">
        /// <item><b>Shared.</b> <see cref="Id"/>, <see cref="Root"/>,
        /// <see cref="WorkDirectory"/>, <see cref="EnvDirectory"/>,
        /// <see cref="StateDirectory"/>, <see cref="TempDirectory"/> (the same
        /// directories); <see cref="RuntimeTempDirectory"/> (one alias of the one temp
        /// directory); <see cref="EnterExecution"/> (the OWNER's gate, so every agent's
        /// code tools still run one at a time); <see cref="BeginOperation"/>,
        /// <see cref="RegisterCleanup"/> and release (one lifetime — a lane is unusable
        /// once the owner is released); <see cref="IsInstalled"/>,
        /// <see cref="MarkInstalled"/> and <see cref="TryMarkApplied"/> (one package
        /// tree); the host-repair-artifact set (one work directory). The file helpers —
        /// <see cref="TryResolve"/>, <see cref="TryReadFile"/>, <see cref="ListFiles"/>,
        /// <see cref="SnapshotWorkFiles"/> and the rest — hold no state beyond
        /// <see cref="WorkDirectory"/> and so answer identically through any lane.</item>
        /// <item><b>Per lane.</b> <see cref="ShellStateDirectory"/> (the shell's saved
        /// directory and exports), <see cref="ShellScriptDirectory"/> (its wrapper
        /// scripts), <see cref="ShellKey"/> (the host's map key for its shell session),
        /// <see cref="Reads"/> (what this agent has been shown), and the rebuilt-notice
        /// latch (each agent is told once). <see cref="EnsureDirectories"/> repairs the
        /// shared layout and then the lane's own.</item>
        /// </list>
        /// <para>
        /// The same id always returns the same lane, so an agent's shell keeps its
        /// directory from one call to the next. Lanes are flat — asking a lane for a lane
        /// asks its owner — and live as long as the owner: a host that reuses an id for a
        /// DIFFERENT agent inherits that agent's shell state and reads, so an id must name
        /// one agent for the whole conversation. The id is reduced to
        /// <c>[A-Za-z0-9_-]</c> and 64 characters because it becomes a directory name, and
        /// compared without regard to case because on macOS and Windows two ids that
        /// differ only in case would be one directory — ids that reduce to the same thing
        /// are the same lane rather than two lanes silently sharing a shell.
        /// </para>
        /// </remarks>
        /// <param name="laneId">The agent's id, for example <c>agent_1</c>. The owner itself is the unnamed lane and is not asked for.</param>
        /// <exception cref="ArgumentException">The id is empty, or has no usable character.</exception>
        /// <exception cref="ObjectDisposedException">This workspace has been released.</exception>
        public SessionWorkspace ForAgent(string laneId)
        {
            if (_owner != null)
                return _owner.ForAgent(laneId);

            string sanitized = SanitizeLaneId(laneId);
            lock (_gate)
            {
                // Checked under the same gate release takes to set the flag, and the lane
                // is built inside it: a lane created after release would recreate
                // directories under a root that is about to be deleted — or already has
                // been — and the host would own a stray directory until its next start.
                if (_releaseRequested)
                    throw new ObjectDisposedException(nameof(SessionWorkspace));
                _lanes ??= new Dictionary<string, SessionWorkspace>(StringComparer.OrdinalIgnoreCase);
                if (!_lanes.TryGetValue(sanitized, out SessionWorkspace? lane))
                {
                    lane = new SessionWorkspace(this, sanitized);
                    _lanes.Add(sanitized, lane);
                }
                return lane;
            }
        }

        private static string SanitizeLaneId(string laneId)
        {
            if (string.IsNullOrWhiteSpace(laneId))
            {
                throw new ArgumentException(
                    "a lane id is required; the workspace itself is the unnamed lane.", nameof(laneId));
            }
            string sanitized = new(laneId
                .Where(c => char.IsAsciiLetterOrDigit(c) || c is '_' or '-')
                .Take(MaxLaneIdLength)
                .ToArray());
            if (sanitized.Length == 0)
            {
                throw new ArgumentException(
                    $"the lane id '{laneId}' has no letter, digit, '_' or '-' to name a directory with.",
                    nameof(laneId));
            }
            return sanitized;
        }

        /// <summary>
        /// The chat session this workspace belongs to — the key
        /// <see cref="SessionWorkspaceManager.Release"/> takes, so a holder of the
        /// workspace can release it without separately remembering the id.
        /// </summary>
        public string Id { get; }

        /// <summary>The workspace's own directory, holding the three below.</summary>
        public string Root { get; }

        /// <summary>
        /// What the model has actually been shown of each file in this workspace.
        ///
        /// <para>
        /// It hangs off the WORKSPACE rather than off the shell session because the
        /// workspace is what every entry point already receives — the tool dispatch, the
        /// patcher and the editor all take one — while a shell session is reachable only
        /// through a resolved shell that a host without one does not have. It also has
        /// exactly the right lifetime: a conversation's reads are worth nothing to the
        /// next conversation, and the workspace is released with the session.
        /// </para>
        /// <para>
        /// Per LANE (see <see cref="ForAgent"/>), not per directory. The ledger records
        /// what one context window has been shown, and a sub-agent's context is not its
        /// parent's: a file the sub-agent read must not authorize an edit the parent makes
        /// blind. The ledger compares content, so a file another agent changed after this
        /// one read it is caught as stale here exactly as a shell rewrite would be.
        /// </para>
        /// </summary>
        public FileLedger Reads { get; } = new();

        /// <summary>
        /// Which agent's view of the workspace this is: empty for the workspace itself
        /// (the root agent), otherwise the sanitized id <see cref="ForAgent"/> was given.
        /// </summary>
        public string LaneId { get; }

        /// <summary>
        /// The key a host uses for per-AGENT shell state it keeps outside the workspace:
        /// the persisted shell session and anything that must follow it.
        ///
        /// <para>
        /// <see cref="Root"/> for the workspace itself — the value such maps were keyed on
        /// before lanes existed, so a host that never calls <see cref="ForAgent"/> keys
        /// exactly as it did — and <c>Root#laneId</c> for a lane. State that is
        /// conversation-wide (background jobs, the sanitized CA bundle) stays keyed on
        /// <see cref="Root"/>.
        /// </para>
        /// </summary>
        public string ShellKey { get; }

        /// <summary>
        /// Where the shell writes each command's wrapper script: <see cref="StateDirectory"/>
        /// for the workspace itself, and the lane's own directory under it for a lane.
        ///
        /// <para>
        /// Per lane because the script names are a per-SESSION sequence
        /// (<c>cmd-1.sh</c>, <c>cmd-2.sh</c>, …). Two shell sessions writing into one
        /// directory would hand out the same names, and a background job is still
        /// executing its script after the call that started it returned — so another
        /// agent's next command would rewrite the file under a running shell. Read-only to
        /// the sandbox, as <see cref="StateDirectory"/> is: only
        /// <see cref="ShellStateDirectory"/> beneath it is writable.
        /// </para>
        /// </summary>
        internal string ShellScriptDirectory { get; }

        /// <summary>
        /// Where everything runs and every file lives: the shell's starting directory,
        /// shared by <c>shell</c> commands and skill scripts alike so one step's output
        /// is the next step's input.
        /// </summary>
        public string WorkDirectory { get; }

        /// <summary>
        /// The session's package environment (<c>pip install --target</c>,
        /// <c>npm --prefix</c>). Reached read-only at run time via PYTHONPATH /
        /// NODE_PATH — by model-written code and by skill scripts equally, which is what
        /// lets a script's dependencies be installed once from the shell and then found
        /// by every later step.
        /// </summary>
        public string EnvDirectory { get; }

        /// <summary>
        /// The host's own bookkeeping for this session: the shell's persisted working
        /// directory and exported environment, the script each command is handed to the
        /// shell as, and the log of every background job.
        ///
        /// <para>
        /// A sibling of <see cref="WorkDirectory"/> rather than a hidden folder inside it,
        /// so that <c>ls</c> shows the model its own files and nothing else, and so that
        /// artifact capture — which scans only the work directory — can never hand the
        /// user a download link to the host's scratch.
        /// </para>
        /// </summary>
        public string StateDirectory { get; }

        /// <summary>
        /// The small part of state a shell wrapper may update: its saved working
        /// directory and exported environment. Sandboxes mount this child writable while
        /// keeping the parent <see cref="StateDirectory"/>, which holds host-authored
        /// scripts and logs, read-only.
        ///
        /// <para>
        /// Per lane: <c>state/shell</c> for the workspace itself,
        /// <c>state/lanes/&lt;id&gt;/shell</c> for a lane. This is what makes a sub-agent's
        /// <c>cd sub</c> move only that sub-agent — the directory is the shell's memory,
        /// and one shared between agents is one shell being driven by all of them. It is
        /// also exactly the directory a lane's sandbox is granted, so one agent's command
        /// cannot rewrite another's saved directory or environment.
        /// </para>
        /// </summary>
        public string ShellStateDirectory { get; }

        /// <summary>
        /// Where <c>TMPDIR</c> points, so a tool's scratch file is not mistaken for the
        /// user's output.
        ///
        /// <para>
        /// Temp used to be the work directory, which is change-based capture's worst
        /// case: every intermediate a converter writes and deletes looks exactly like a
        /// file the user asked for. A sibling directory is writable, is still inside the
        /// sandbox, and is not scanned.
        /// </para>
        /// </summary>
        public string TempDirectory { get; }

        private string? _runtimeTempDirectory;

        /// <summary>
        /// A short spelling of the temporary directory for runtimes which create Unix
        /// sockets. Unix socket addresses have a roughly 100-byte limit, independent
        /// of the filesystem's path limit. Long deployment/session paths must not
        /// prevent an otherwise valid local IPC endpoint from being created.
        /// </summary>
        public string RuntimeTempDirectory
        {
            get
            {
                // Shared: an alias of the shared temp directory, created once and
                // cleaned up with the owner.
                if (_owner != null)
                    return _owner.RuntimeTempDirectory;
                EnsureRealDirectory(Root);
                EnsureRealDirectory(TempDirectory);
                if (!OperatingSystem.IsMacOS() || Encoding.UTF8.GetByteCount(TempDirectory) <= 48)
                    return TempDirectory;
                lock (_gate)
                {
                    if (_runtimeTempDirectory != null
                        && string.Equals(new DirectoryInfo(_runtimeTempDirectory).LinkTarget, TempDirectory, StringComparison.Ordinal))
                        return _runtimeTempDirectory;
                    // Shared /tmp lets an earlier command replace the alias. Recreate a
                    // fresh spelling if that happened, and never derive sandbox grants
                    // from this model-mutable symlink; only TempDirectory is trusted.
                    string alias = Path.Combine("/tmp", "tsh-" + Guid.NewGuid().ToString("N"));
                    Directory.CreateSymbolicLink(alias, TempDirectory);
                    _runtimeTempDirectory = alias;
                    RegisterCleanup(new TemporaryAlias(alias));
                    return alias;
                }
            }
        }

        private sealed class TemporaryAlias(string path) : IDisposable
        {
            public void Dispose()
            {
                // Delete only the link, never recursively traverse model-owned scratch.
                try { Directory.Delete(path); }
                catch (IOException) { }
                catch (UnauthorizedAccessException) { }
            }
        }

        /// <summary>
        /// Keep this workspace alive while one host tool is using it.
        /// </summary>
        /// <remarks>
        /// HTTP cancellation stops an async iterator immediately, but the synchronous
        /// tool it started on a worker may still be winding down. The operation token
        /// lets the manager detach the request immediately while deferring cleanup and
        /// directory deletion until that worker has finished touching the workspace.
        /// </remarks>
        public IDisposable BeginOperation()
        {
            // Shared: a lane's operation keeps the OWNER alive, and a released owner
            // refuses its lanes — there is one directory, deleted once.
            if (_owner != null)
                return _owner.BeginOperation();
            lock (_gate)
            {
                if (_releaseRequested)
                    throw new ObjectDisposedException(nameof(SessionWorkspace));
                _activeOperations++;
            }
            return new WorkspaceOperation(this);
        }

        /// <summary>
        /// Serialize synchronous tools that touch this session's files or package
        /// environment. A replacement chat turn can begin while the cancelled turn's
        /// synchronous tool is still unwinding; without this gate it can import a wheel
        /// while the first turn is only halfway through extracting it.
        /// </summary>
        /// <remarks>
        /// Monitor ownership is deliberately re-entrant. A skill script holds the lease
        /// while it runs and may synchronously call back into the same code runner to
        /// install a missing dependency on that same thread.
        /// <para>
        /// A lane takes its OWNER's gate. Lanes share the files and the package tree, so
        /// the hazard this exists for — one tool observing another's half-finished write
        /// or install — is exactly as real between two agents as between two turns.
        /// </para>
        /// </remarks>
        internal IDisposable EnterExecution()
        {
            if (_owner != null)
                return _owner.EnterExecution();
            Monitor.Enter(_executionGate);
            return new ExecutionLease(_executionGate);
        }

        /// <summary>
        /// Record exact workspace paths that this host created solely to repair a bundled
        /// skill script. Artifact capture excludes these paths, but must not exclude a
        /// directory merely because a user happened to call it <c>skill-repairs</c>.
        /// </summary>
        internal void MarkHostRepairArtifacts(params string[] relativePaths)
        {
            if (relativePaths == null)
                return;

            // Shared: the paths name files in the shared work directory.
            if (_owner != null)
            {
                _owner.MarkHostRepairArtifacts(relativePaths);
                return;
            }

            lock (_gate)
            {
                foreach (string relativePath in relativePaths)
                {
                    if (!TryResolve(relativePath, out string full, out _)
                        || !File.Exists(full))
                    {
                        continue;
                    }

                    string normalized = Path.GetRelativePath(WorkDirectory, full).Replace('\\', '/');
                    _hostRepairArtifacts.Add(normalized);
                }
            }
        }

        /// <summary>True only for an exact path proven host-created by the method above.</summary>
        internal bool IsHostRepairArtifact(string? relativePath)
        {
            if (_owner != null)
                return _owner.IsHostRepairArtifact(relativePath);
            if (string.IsNullOrWhiteSpace(relativePath))
                return false;
            string normalized = relativePath.Replace('\\', '/');
            lock (_gate)
                return _hostRepairArtifacts.Contains(normalized);
        }

        private sealed class ExecutionLease : IDisposable
        {
            private object? _gate;

            public ExecutionLease(object gate) => _gate = gate;

            public void Dispose()
            {
                object? gate = Interlocked.Exchange(ref _gate, null);
                if (gate != null)
                    Monitor.Exit(gate);
            }
        }

        private void EndOperation()
        {
            Action? release = null;
            lock (_gate)
            {
                if (_activeOperations > 0)
                    _activeOperations--;
                if (_activeOperations == 0 && _releaseWhenIdle != null)
                {
                    release = _releaseWhenIdle;
                    _releaseWhenIdle = null;
                }
            }
            release?.Invoke();
        }

        /// <summary>
        /// Mark the workspace unavailable to new operations and run its final cleanup
        /// now, or after the last operation completes.
        /// </summary>
        internal void ReleaseWhenIdle(Action release)
        {
            ArgumentNullException.ThrowIfNull(release);
            if (_owner != null)
            {
                _owner.ReleaseWhenIdle(release);
                return;
            }
            bool releaseNow;
            lock (_gate)
            {
                if (_releaseRequested)
                    return;
                _releaseRequested = true;
                releaseNow = _activeOperations == 0;
                if (!releaseNow)
                    _releaseWhenIdle = release;
            }
            if (releaseNow)
                release();
        }

        private sealed class WorkspaceOperation : IDisposable
        {
            private SessionWorkspace? _workspace;

            public WorkspaceOperation(SessionWorkspace workspace) => _workspace = workspace;

            public void Dispose() =>
                Interlocked.Exchange(ref _workspace, null)?.EndOperation();
        }

        /// <summary>
        /// The packages already installed into <see cref="EnvDirectory"/> this session,
        /// so a repeated request skips pip entirely. Names only — a request that pins a
        /// different version is not filtered and pip resolves it.
        ///
        /// <para>
        /// Keyed by LANGUAGE, because one workspace holds a pip environment and an npm
        /// one and plenty of names exist in both registries. A flat ledger made an npm
        /// install of <c>markitdown</c> suppress the later pip install of the Python
        /// package with that name, and the model was told "Already installed this
        /// session: markitdown" while <c>import markitdown</c> kept failing — a false
        /// statement from the host, which is the one kind of tool result a model has no
        /// way to recover from. Observed on exactly this path: the pptx skill asks for
        /// <c>pptxgenjs</c> (npm) and <c>markitdown</c> (pip) in the same task.
        /// </para>
        /// </summary>
        /// <param name="language">The installer's language, e.g. "python" or "javascript".</param>
        public bool IsInstalled(string language, string package)
        {
            // Shared, as the package tree it describes is.
            if (_owner != null)
                return _owner.IsInstalled(language, package);
            lock (_gate)
                return _installedPackages.Contains(InstallKey(language, package));
        }

        /// <summary>Record a successful install, against the language that performed it.</summary>
        public void MarkInstalled(string language, IEnumerable<string> packages)
        {
            if (_owner != null)
            {
                _owner.MarkInstalled(language, packages);
                return;
            }
            lock (_gate)
            {
                foreach (string package in packages)
                    _installedPackages.Add(InstallKey(language, package));
            }
        }

        private static string InstallKey(string language, string package) =>
            (language ?? string.Empty) + "\u0000" + package;

        /// <summary>
        /// Read a file from the session's working directory, for the tools that address
        /// code BY PATH rather than as "the program I just ran".
        ///
        /// <para>
        /// Confined to <see cref="WorkDirectory"/>: the path comes from the model, so it
        /// is resolved and checked to be inside, and a traversal out of the workspace is
        /// refused rather than followed.
        /// </para>
        /// </summary>
        public bool TryReadFile(string relativePath, out string content, out string? error)
        {
            content = string.Empty;
            error = null;
            if (!TryResolve(relativePath, out string full, out error))
                return false;
            if (!File.Exists(full))
            {
                error = $"'{relativePath}' does not exist in this conversation's working directory."
                        + DescribeWhatIsHere();
                return false;
            }
            try
            {
                content = File.ReadAllText(full);
                return true;
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                error = $"'{relativePath}' could not be read: {ex.Message}";
                return false;
            }
        }

        /// <summary>Write a file into the session's working directory, creating directories as needed.</summary>
        public bool TryWriteFile(string relativePath, string content, out string? error)
        {
            error = null;
            if (!TryResolve(relativePath, out string full, out error))
                return false;
            try
            {
                string? parent = Path.GetDirectoryName(full);
                if (!string.IsNullOrEmpty(parent))
                    Directory.CreateDirectory(parent);
                File.WriteAllText(full, content);
                return true;
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                error = $"'{relativePath}' could not be written: {ex.Message}";
                return false;
            }
        }

        /// <summary>
        /// Create a workspace file without replacing an existing path. Used for
        /// host-staged repair copies whose prior, model-edited contents must survive a
        /// repeated failure. <paramref name="alreadyExists"/> distinguishes that safe
        /// collision from an unrelated filesystem error.
        /// </summary>
        public bool TryCreateFile(
            string relativePath,
            string content,
            out bool alreadyExists,
            out string? error)
        {
            return TryCreateFile(
                relativePath,
                new UTF8Encoding(encoderShouldEmitUTF8Identifier: false).GetBytes(content),
                out alreadyExists,
                out error);
        }

        /// <summary>
        /// Byte-preserving form used when the host copies an existing file into the
        /// workspace. Kept internal because model-facing text writes should continue
        /// through the string overload above.
        /// </summary>
        internal bool TryCreateFile(
            string relativePath,
            byte[] content,
            out bool alreadyExists,
            out string? error)
        {
            alreadyExists = false;
            error = null;
            if (!TryResolve(relativePath, out string full, out error))
                return false;

            string? temporary = null;
            try
            {
                string? parent = Path.GetDirectoryName(full);
                if (!string.IsNullOrEmpty(parent))
                    Directory.CreateDirectory(parent);

                // Finish the potentially fallible write before publishing the target.
                // If construction, writing or disposal fails, no partial destination
                // exists that a later call could mistake for a pre-existing repair.
                temporary = Path.Combine(
                    parent!, ".tensorsharp-create-" + Guid.NewGuid().ToString("N") + ".tmp");
                using var stream = new FileStream(
                    temporary, FileMode.CreateNew, FileAccess.Write, FileShare.None);
                stream.Write(content, 0, content.Length);
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                TryDeleteTemporary(temporary);
                error = $"'{relativePath}' could not be created: {ex.Message}";
                return false;
            }

            try
            {
                // File.Move without overwrite is the atomic publication point. Only an
                // error here can truthfully mean that the requested target collided.
                File.Move(temporary!, full);
                temporary = null;
                return true;
            }
            catch (IOException) when (File.Exists(full) || Directory.Exists(full))
            {
                alreadyExists = true;
                error = $"'{relativePath}' already exists in this conversation's working directory.";
                return false;
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                error = $"'{relativePath}' could not be created: {ex.Message}";
                return false;
            }
            finally
            {
                TryDeleteTemporary(temporary);
            }
        }

        private static void TryDeleteTemporary(string? path)
        {
            if (string.IsNullOrEmpty(path))
                return;
            try { File.Delete(path); }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { }
        }

        /// <summary>
        /// The files in the working directory, relative and sorted, excluding the
        /// runtime fallout no one asked for (bytecode caches and the like).
        /// </summary>
        public IReadOnlyList<(string Path, long Bytes)> ListFiles(int max = 200)
        {
            var found = new List<(string, long)>();
            try
            {
                foreach (string file in WorkspaceScan.Files(WorkDirectory, WorkspaceScan.SnapshotOptions))
                {
                    string relative = Path.GetRelativePath(WorkDirectory, file).Replace('\\', '/');
                    if (relative.StartsWith(".tensorsharp-", StringComparison.Ordinal)
                        || relative.Contains("__pycache__/", StringComparison.Ordinal)
                        || relative.EndsWith(".pyc", StringComparison.Ordinal)
                        || relative.StartsWith("Library/", StringComparison.Ordinal)
                        || relative.StartsWith("node_modules/", StringComparison.Ordinal)
                        || relative.StartsWith(".", StringComparison.Ordinal))
                    {
                        continue;
                    }
                    try { found.Add((relative, new FileInfo(file).Length)); }
                    catch (IOException) { /* vanished mid-scan */ }
                    if (found.Count >= max)
                        break;
                }
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { /* partial is fine */ }
            found.Sort((a, b) => string.CompareOrdinal(a.Item1, b.Item1));
            return found;
        }

        /// <summary>
        /// Resolve a model-supplied relative path inside <see cref="WorkDirectory"/>.
        /// The containment check is the point: "../../.ssh/id_rsa" is a path the model
        /// can write, and only the resolved comparison catches it.
        ///
        /// <para>
        /// Public because the deterministic editor is what needs it most: every path in a
        /// patch comes from the model, and every one of them is resolved through here
        /// before a single byte is written.
        /// </para>
        /// </summary>
        public bool TryResolve(string relativePath, out string fullPath, out string? error) =>
            TryResolveFrom(WorkDirectory, relativePath, out fullPath, out error);

        /// <summary>
        /// The same resolution, but relative to <paramref name="baseDirectory"/> —
        /// wherever the shell currently is.
        ///
        /// <para>
        /// Needed because the two halves of the tool surface have to agree about what a
        /// relative path means. A model that <c>cd</c>s into <c>build/</c> and then
        /// patches <c>main.c</c> means <c>build/main.c</c>; resolving from the work
        /// directory instead would silently patch a different file, or refuse a file that
        /// is plainly right there. Containment is still checked against
        /// <see cref="WorkDirectory"/>, so a base of the model's choosing cannot widen it.
        /// </para>
        /// </summary>
        public bool TryResolveFrom(
            string baseDirectory, string relativePath, out string fullPath, out string? error)
        {
            fullPath = string.Empty;
            error = null;
            if (string.IsNullOrWhiteSpace(relativePath))
            {
                error = "a file path is required.";
                return false;
            }
            try
            {
                string root = Path.GetFullPath(WorkDirectory);
                string from = string.IsNullOrWhiteSpace(baseDirectory)
                    ? root
                    : Path.GetFullPath(baseDirectory);

                // An absolute path is followed when it lands INSIDE the working
                // directory and refused when it does not. A model reading a traceback
                // sees absolute paths and repeats them, and the containment check below
                // is what makes the path safe — not its spelling. Refusing an absolute
                // path that names a file the model may already open relatively taught it
                // nothing except to go looking, which is how a fixable typo became a
                // hunt through tools.
                string candidate = Path.IsPathRooted(relativePath)
                    ? Path.GetFullPath(relativePath)
                    : Path.GetFullPath(Path.Combine(from, relativePath));
                string rootWithSeparator = root.EndsWith(Path.DirectorySeparatorChar)
                    ? root
                    : root + Path.DirectorySeparatorChar;
                if (!candidate.StartsWith(rootWithSeparator, StringComparison.Ordinal))
                {
                    error = $"'{relativePath}' is outside this conversation's working directory."
                            + DescribeWhatIsHere();
                    return false;
                }

                // The lexical check above collapses "..", and that is all it does. It does
                // NOT follow symbolic links, and the caller is the HOST process, which is
                // not sandboxed — so `ln -s ~/.ssh/id_rsa notes.txt` (one ordinary shell
                // command, entirely permitted inside the workspace) would otherwise turn a
                // patch of "notes.txt" into a write through the link, and a read of it into
                // a read of the target. The component-by-component walk is the primitive
                // this codebase already uses for exactly this, and it is what makes the
                // sandbox's two central promises — home unreadable, writes confined — hold
                // for the file operations the host performs on the model's behalf.
                if (!SkillPathGuard.TryResolveSymlinks(root, candidate, out string? _, out string? linkError))
                {
                    error = $"'{relativePath}' {linkError}. Paths must stay inside this "
                            + "conversation's working directory.";
                    return false;
                }

                fullPath = candidate;
                return true;
            }
            catch (Exception ex) when (ex is ArgumentException or NotSupportedException or PathTooLongException)
            {
                error = $"'{relativePath}' is not a usable file name: {ex.Message}";
                return false;
            }
        }

        /// <summary>
        /// The tail of a path error: what the working directory actually holds, and the
        /// fact that the program just run needs no path at all.
        ///
        /// <para>
        /// A path error that only says the path is wrong leaves the model guessing, and
        /// it guesses the same wrong path again — observed exactly: an edit addressed at
        /// an invented <c>/tmp/Untitled.mjs</c> was refused, and the next round sent the
        /// identical path. Naming the real files, in the result, is what turns a refusal
        /// into a correction; small models act on tool results far more reliably than on
        /// parameter documentation they read thousands of tokens earlier.
        /// </para>
        /// </summary>
        public string DescribeWhatIsHere(int max = 20)
        {
            var sb = new StringBuilder();

            IReadOnlyList<(string Path, long Bytes)> files = ListFiles(max + 1);
            if (files.Count == 0)
            {
                sb.Append(" The working directory is empty.");
            }
            else
            {
                sb.Append(" It holds: ");
                for (int i = 0; i < files.Count && i < max; i++)
                {
                    if (i > 0)
                        sb.Append(", ");
                    sb.Append(files[i].Path);
                }
                if (files.Count > max)
                    sb.Append(", and more");
                sb.Append('.');
            }

            return sb.ToString();
        }

        private readonly List<IDisposable> _cleanups = new();

        /// <summary>
        /// Register something to shut down when this session ends.
        ///
        /// <para>
        /// A background job outlives the call that started it — that is the point of one —
        /// so something has to own the moment it stops, and the only honest owner is the
        /// session whose files it is writing into. Without this the workspace directory is
        /// deleted out from under a process that is still running in it.
        /// </para>
        /// </summary>
        public void RegisterCleanup(IDisposable cleanup)
        {
            ArgumentNullException.ThrowIfNull(cleanup);

            // Shared: a lane has no end of its own. What it starts — a background job, a
            // shell-session map entry — lives until the conversation ends and is stopped
            // with everything else, before the one directory is deleted.
            if (_owner != null)
            {
                _owner.RegisterCleanup(cleanup);
                return;
            }
            bool disposeNow;
            lock (_gate)
            {
                disposeNow = _cleanupRegistrationClosed;
                if (!disposeNow)
                    _cleanups.Add(cleanup);
            }

            // Run outside the gate: cleanup may wait for a worker that needs another
            // workspace lock. Once final cleanup starts, RunCleanups may already have
            // copied and cleared the list, so appending here would otherwise leak the
            // resource forever.
            if (disposeNow)
                DisposeCleanup(cleanup);
        }

        /// <summary>Run every registered cleanup. Called once, before the directory is deleted.</summary>
        internal void RunCleanups()
        {
            if (_owner != null)
            {
                _owner.RunCleanups();
                return;
            }
            IDisposable[] pending;
            lock (_gate)
            {
                _cleanupRegistrationClosed = true;
                pending = _cleanups.ToArray();
                _cleanups.Clear();
            }
            foreach (IDisposable cleanup in pending)
                DisposeCleanup(cleanup);
        }

        private static void DisposeCleanup(IDisposable cleanup)
        {
            // One misbehaving cleanup must not strand the rest, and must not leave the
            // workspace undeleted — the disk is the resource that actually accumulates.
            try { cleanup.Dispose(); }
            catch (Exception ex) when (ex is not (OutOfMemoryException or StackOverflowException)) { }
        }

        private readonly HashSet<string> _appliedSetups = new(StringComparer.Ordinal);

        /// <summary>
        /// True exactly once per <paramref name="key"/> for this workspace's lifetime.
        /// Used for one-time setup steps — a skill's <c>requirements.txt</c> is applied
        /// on the session's first script run, not on every one.
        /// </summary>
        public bool TryMarkApplied(string key)
        {
            // Shared: what was applied went into the shared package tree.
            if (_owner != null)
                return _owner.TryMarkApplied(key);
            lock (_gate)
                return _appliedSetups.Add(key);
        }

        /// <summary>
        /// The state of every file under <see cref="WorkDirectory"/>, for telling what a
        /// run changed: capture-worthy output is what is NEW or MODIFIED relative to the
        /// snapshot taken before the run, never the accumulated history of the session.
        /// </summary>
        public Dictionary<string, (long Length, DateTime WriteTime)> SnapshotWorkFiles()
        {
            var snapshot = new Dictionary<string, (long, DateTime)>(StringComparer.Ordinal);
            try
            {
                foreach (string file in WorkspaceScan.Files(WorkDirectory, WorkspaceScan.SnapshotOptions))
                {
                    var info = new FileInfo(file);
                    string relative = Path.GetRelativePath(WorkDirectory, file).Replace('\\', '/');
                    snapshot[relative] = (info.Length, info.LastWriteTimeUtc);
                }
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { /* partial is fine */ }
            return snapshot;
        }

        /// <summary>True when <paramref name="relative"/> is unchanged from <paramref name="snapshot"/>.</summary>
        public bool IsUnchangedSince(
            Dictionary<string, (long Length, DateTime WriteTime)> snapshot, string relative)
        {
            if (!snapshot.TryGetValue(relative.Replace('\\', '/'), out (long Length, DateTime WriteTime) before))
                return false;
            try
            {
                var now = new FileInfo(Path.Combine(WorkDirectory, relative));
                return now.Exists && now.Length == before.Length && now.LastWriteTimeUtc == before.WriteTime;
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                return false;
            }
        }
    }

    /// <summary>
    /// Owns the map from session id to <see cref="SessionWorkspace"/> and their disk
    /// lifecycle. One per host; sessions get workspaces lazily and lose them exactly
    /// when the session itself ends.
    /// </summary>
    public sealed class SessionWorkspaceManager
    {
        private readonly string _root;
        private readonly ILogger _logger;
        private readonly ConcurrentDictionary<string, SessionWorkspace> _workspaces = new(StringComparer.Ordinal);
        private readonly object _mapGate = new();

        /// <param name="root">Parent directory the workspaces live under.</param>
        public SessionWorkspaceManager(string root, ILogger? logger = null)
        {
            _root = root ?? throw new ArgumentNullException(nameof(root));
            _logger = logger ?? NullLogger.Instance;
        }

        /// <summary>
        /// Delete every workspace left behind by a host that is no longer running, and
        /// leave alone any that a live one still owns.
        /// </summary>
        /// <remarks>
        /// <para>
        /// This used to delete everything under the root, on the reasoning that a restart
        /// orphans every session. That reasoning holds for ONE host and the root is not
        /// one host's: the scratch directory defaults to a folder beside the binary, so
        /// every host launched from that build shares it. A second server started on
        /// another port — four such launches appear in the log of the 2026-09-10 incident
        /// — therefore deleted the working directory of a conversation that was still
        /// running in the first, and every later command in it failed at the point where
        /// its wrapper script is written.
        /// </para>
        /// <para>
        /// So a workspace records the process that owns it and the sweep skips one whose
        /// owner is still alive. A stamp that cannot be read is treated as an orphan,
        /// which keeps the old behaviour for anything a previous version left behind.
        /// </para>
        /// </remarks>
        public void SweepOrphans()
        {
            try
            {
                if (!Directory.Exists(_root))
                    return;
                foreach (string dir in Directory.EnumerateDirectories(_root, SessionWorkspace.DirectoryPrefix + "*"))
                {
                    if (WorkspaceOwner.IsHeldByALiveProcess(dir))
                    {
                        _logger.LogInformation(LogEventIds.SkillScriptExecuted,
                            "workspace.sweep-skipped dir={Dir} reason=owned-by-a-running-host", dir);
                        continue;
                    }
                    TryDelete(dir);
                }
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                _logger.LogWarning(LogEventIds.SkillScriptExecuted,
                    "workspace.sweep-failed root={Root} reason={Reason}", _root, ex.Message);
            }
        }

        /// <summary>The session's workspace, created on first use.</summary>
        public SessionWorkspace GetOrCreate(string sessionId)
        {
            ArgumentException.ThrowIfNullOrEmpty(sessionId);
            lock (_mapGate)
            {
                if (_workspaces.TryGetValue(sessionId, out SessionWorkspace? existing))
                {
                    // The single choke point every consumer already passes through, so the
                    // invariant it establishes — "the directories of the workspace you were
                    // handed exist" — holds for call sites that have not been written yet.
                    // A workspace is created once and used for a whole conversation, and
                    // nothing kept its directories alive over that span.
                    existing.TryEnsureDirectories();
                    return existing;
                }

                // The suffix matters when an id is reused while its previous workspace
                // is retiring: Release removes the mapping immediately, but an active
                // worker may keep the old directory alive for a little longer.
                var workspace = new SessionWorkspace(
                    Path.Combine(_root, SessionWorkspace.DirectoryPrefix + Sanitize(sessionId)
                        + "-" + Guid.NewGuid().ToString("N")), sessionId);
                _workspaces[sessionId] = workspace;
                _logger.LogInformation(LogEventIds.SkillScriptExecuted,
                    "workspace.created session={SessionId} root={Root}", sessionId, workspace.Root);
                return workspace;
            }
        }

        /// <summary>Delete the session's workspace and everything in it, if one exists.</summary>
        public void Release(string sessionId)
        {
            if (string.IsNullOrEmpty(sessionId))
                return;

            SessionWorkspace? workspace;
            lock (_mapGate)
                _workspaces.TryRemove(sessionId, out workspace);

            if (workspace != null)
            {
                _logger.LogInformation(LogEventIds.SkillScriptExecuted,
                    "workspace.released session={SessionId} root={Root}", sessionId, workspace.Root);
                workspace.ReleaseWhenIdle(() =>
                {
                    // Stop anything still running in the directory BEFORE deleting it: a
                    // background job holding a file open turns the delete into a partial one,
                    // and the leftovers are what SweepOrphans has to clean up next boot.
                    workspace.RunCleanups();
                    TryDelete(workspace.Root);
                });
            }
        }

        /// <summary>Workspace directory names come from session ids we generate, but a
        /// path separator in one must never choose where the delete lands.</summary>
        private static string Sanitize(string sessionId) =>
            new(sessionId.Where(char.IsAsciiLetterOrDigit).Take(64).ToArray());

        private void TryDelete(string directory)
        {
            try
            {
                if (Directory.Exists(directory))
                    Directory.Delete(directory, recursive: true);
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                _logger.LogWarning(LogEventIds.SkillScriptExecuted,
                    "workspace.delete-failed dir={Dir} reason={Reason}", directory, ex.Message);
            }
        }
    }
}
