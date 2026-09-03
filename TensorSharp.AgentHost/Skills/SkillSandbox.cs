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
using System.Linq;
using System.Text;

using TensorSharp.AgentHost.CodeExec;

namespace TensorSharp.AgentHost.Skills
{
    /// <summary>How hard a host insists on OS-level isolation for a skill's scripts.</summary>
    public enum SkillSandboxMode
    {
        /// <summary>
        /// Run the script directly, with only the in-process limits
        /// (<see cref="SkillScriptRunner"/>'s working directory, argument vector,
        /// timeout and output cap). No filesystem or network confinement.
        /// </summary>
        Off,

        /// <summary>
        /// Sandbox when the platform provides one, run unsandboxed when it does not.
        /// A developer's own machine is the case this is for; it is the wrong choice
        /// for anything that accepts skill uploads, because "no sandbox available"
        /// silently becomes "no sandbox".
        /// </summary>
        Preferred,

        /// <summary>
        /// Sandbox or refuse. The default whenever script execution is enabled: a
        /// host that cannot isolate a script should say so rather than run it, and the
        /// model is told plainly that the tool is unavailable here.
        /// </summary>
        Required,
    }

    /// <summary>What a sandbox is asked to confine one script run to.</summary>
    /// <param name="Interpreter">Absolute path or PATH name of the interpreter to launch.</param>
    /// <param name="Arguments">The interpreter's argument vector, script path first.</param>
    /// <param name="SkillDirectory">The skill's own directory. Readable, never writable.</param>
    /// <param name="WorkDirectory">
    /// A scratch directory that becomes the process's working directory and is the only
    /// place it may write. Anything the script produces lands here, where the caller can
    /// find it — and a script that tries to edit its own skill, the model's uploads, or
    /// anything else on the host fails instead.
    /// </param>
    /// <param name="AllowNetwork">Let the script reach the network. Off by default.</param>
    /// <param name="ReadablePaths">Extra paths the script may read, beyond the system and its own skill.</param>
    public readonly record struct SkillSandboxRequest(
        string Interpreter,
        IReadOnlyList<string> Arguments,
        string SkillDirectory,
        string WorkDirectory,
        bool AllowNetwork,
        IReadOnlyList<string> ReadablePaths)
    {
        /// <summary>Additional exact paths writable beside <see cref="WorkDirectory"/>.</summary>
        public IReadOnlyList<string> WritablePaths { get; init; } = Array.Empty<string>();

        /// <summary>Directory the process starts in when it differs from the writable root.</summary>
        public string? StartDirectory { get; init; }

        /// <summary>
        /// When set (and <see cref="AllowNetwork"/> is false), the one TCP port on
        /// localhost the process may connect to. This is how an install phase is
        /// given a package registry without the whole internet: the host runs an
        /// egress proxy with a domain allowlist on this port, HTTPS_PROXY points the
        /// installer at it, and the sandbox admits exactly that loopback port —
        /// every other destination stays denied at the OS level.
        /// </summary>
        public int? AllowLoopbackPort { get; init; }
    }

    /// <summary>
    /// What a sandbox actually enforces.
    ///
    /// <para>
    /// The three platforms do not offer the same primitives, and pretending otherwise
    /// would be the worst outcome: an operator who reads "sandboxed" and gets only
    /// process-lifetime bounds has been misled. Every sandbox states its guarantees
    /// here, the runner reports the ones that are MISSING in the result the model sees
    /// and in the startup log, and the documentation is generated from the same record
    /// so it cannot drift.
    /// </para>
    /// </summary>
    /// <param name="ConfinesWrites">The script can only write to its scratch directory.</param>
    /// <param name="ConfinesNetwork">The script cannot use the external/IP network. Sandboxes may still admit narrowly scoped local Unix IPC.</param>
    /// <param name="ConfinesHomeReads">The script cannot read the user's home directory — credentials, keys, other skills.</param>
    /// <param name="BoundsProcessTree">Children are killed with the parent, so nothing outlives the request.</param>
    public readonly record struct SkillSandboxCapabilities(
        bool ConfinesWrites,
        bool ConfinesNetwork,
        bool ConfinesHomeReads,
        bool BoundsProcessTree)
    {
        /// <summary>The properties this sandbox does NOT provide, phrased for a human.</summary>
        public IReadOnlyList<string> Gaps()
        {
            var gaps = new List<string>();
            if (!ConfinesWrites) gaps.Add("the script may write anywhere the host process can");
            if (!ConfinesNetwork) gaps.Add("the script may reach the network");
            if (!ConfinesHomeReads) gaps.Add("the script may read the user's home directory");
            if (!BoundsProcessTree) gaps.Add("a child process may outlive the request");
            return gaps;
        }
    }

    /// <summary>An OS mechanism that can confine a child process.</summary>
    public interface ISkillSandbox
    {
        /// <summary>Short name for logs and for the result the model sees (<c>sandbox-exec</c>, <c>bubblewrap</c>).</summary>
        string Name { get; }

        /// <summary>False when the mechanism is not present or not usable on this host.</summary>
        bool IsAvailable { get; }

        /// <summary>What this sandbox enforces. See <see cref="SkillSandboxCapabilities"/>.</summary>
        SkillSandboxCapabilities Capabilities { get; }

        /// <summary>
        /// Called after the child has started, for mechanisms that attach to a running
        /// process rather than wrapping its command line (a Windows job object). A
        /// wrapper-style sandbox does nothing here.
        /// </summary>
        /// <returns>False when the child could not be confined and must be killed.</returns>
        bool TryAttach(SpawnedProcess process, out string error) { error = null!; return true; }

        /// <summary>
        /// One line describing what this sandbox actually enforces, so a host can log
        /// it and the docs cannot drift from the implementation.
        /// </summary>
        string Describe();

        /// <summary>
        /// Rewrite <paramref name="request"/> into the command that runs it confined.
        /// </summary>
        /// <param name="fileName">The executable to launch (the sandbox helper, not the interpreter).</param>
        /// <param name="arguments">Its full argument vector.</param>
        /// <param name="cleanup">
        /// Disposed after the run — a generated profile file, a temporary mount point.
        /// Null when the sandbox needs no scratch state.
        /// </param>
        /// <param name="error">Why the request could not be wrapped, or null.</param>
        bool TryWrap(
            SkillSandboxRequest request,
            out string fileName,
            out IReadOnlyList<string> arguments,
            out IDisposable cleanup,
            out string error);
    }

    /// <summary>
    /// Picks the strongest sandbox this host actually provides.
    ///
    /// <para>
    /// Detection is by probing, not by guessing from the OS: <c>sandbox-exec</c> is on
    /// every macOS but has been deprecated for years and could be removed, and
    /// <c>bwrap</c> is on most Linux desktops but almost no containers. A host that
    /// reports which sandbox is in force — and refuses to run scripts when the answer
    /// is "none" — is the difference between a security property and a hope.
    /// </para>
    /// </summary>
    public static class SkillSandboxFactory
    {
        private static readonly object Gate = new();
        private static ISkillSandbox? _detected;
        private static bool _probed;

        /// <summary>The available sandbox, or null when this host provides none.</summary>
        public static ISkillSandbox? Detect()
        {
            lock (Gate)
            {
                if (_probed)
                    return _detected;

                _probed = true;
                foreach (ISkillSandbox candidate in Candidates())
                {
                    if (!candidate.IsAvailable)
                        continue;
                    _detected = candidate;
                    break;
                }
                return _detected;
            }
        }

        private static IEnumerable<ISkillSandbox> Candidates()
        {
            if (OperatingSystem.IsMacOS())
                yield return new SeatbeltSandbox();
            if (OperatingSystem.IsLinux())
                yield return new BubblewrapSandbox();
            if (OperatingSystem.IsWindows())
                yield return new WindowsJobObjectSandbox();
        }

        /// <summary>
        /// A one-line summary of this host's isolation, for the startup banner and the
        /// <c>--list-skills</c> footer.
        /// </summary>
        public static string DescribeHost()
        {
            ISkillSandbox? sandbox = Detect();
            if (sandbox == null)
                return NoSandboxSummary(
                    OperatingSystem.IsLinux() ? BubblewrapSandbox.AvailabilityError : string.Empty);

            IReadOnlyList<string> gaps = sandbox.Capabilities.Gaps();
            return gaps.Count == 0
                ? $"{sandbox.Name}: {sandbox.Describe()}"
                : $"{sandbox.Name}: {sandbox.Describe()}. NOT confined: {string.Join("; ", gaps)}";
        }

        /// <summary>
        /// The "nothing on this host can confine a child process" line. It always
        /// carries the literal phrase <c>no OS sandbox</c> — the phrase the shell tool
        /// and the skill runner also use, and the one an operator greps their logs for —
        /// and names <paramref name="reason"/> whenever the host can say why.
        ///
        /// <para>
        /// Split out of <see cref="DescribeHost"/> because each branch is reachable only
        /// on one shape of host: the Linux wording can be edited on Windows or macOS
        /// with nothing there to run it, which is how <c>"no safe OS sandbox available"</c>
        /// shipped and broke the test that pins this phrase on a Linux box with no
        /// bubblewrap. Here it is pinned by value on every platform. The distinction the
        /// dropped adjective was reaching for survives in the reason: an unusably old
        /// bubblewrap names its version, an absent one says it is not installed, and
        /// those are different jobs to fix.
        /// </para>
        /// </summary>
        /// <param name="reason">Why nothing is confining this host, or empty when there
        /// is nothing to add beyond the platform having no mechanism at all.</param>
        internal static string NoSandboxSummary(string reason) =>
            string.IsNullOrWhiteSpace(reason)
                ? "no OS sandbox available on this platform"
                : "no OS sandbox available: " + reason;
    }

    /// <summary>
    /// macOS Seatbelt, driven through <c>/usr/bin/sandbox-exec</c> and a generated
    /// SBPL profile.
    ///
    /// <para>
    /// The profile denies everything, then re-allows the narrowest set a scripting
    /// interpreter actually needs. Verified against a probe script that tries each
    /// escape: reading <c>~/.ssh</c> fails, writing anywhere but the scratch directory
    /// fails, and opening a socket fails, while <c>python3</c> still starts and imports
    /// its standard library.
    /// </para>
    /// <para>
    /// Reads of system paths stay allowed, because narrowing them further stops the
    /// interpreter from loading at all (an earlier profile that whitelisted
    /// <c>/usr</c>, <c>/System</c> and <c>/Library</c> by subpath aborted CPython with
    /// SIGABRT before it reached <c>main</c>). What matters is closed: the user's home
    /// directory — where credentials, SSH keys and every other skill live — is denied
    /// wholesale, and only the running skill's own directory is punched back through.
    /// </para>
    /// </summary>
    internal sealed class SeatbeltSandbox : ISkillSandbox
    {
        private const string Helper = "/usr/bin/sandbox-exec";

        public string Name => "sandbox-exec";

        public bool IsAvailable => OperatingSystem.IsMacOS() && File.Exists(Helper);

        public SkillSandboxCapabilities Capabilities => new(
            ConfinesWrites: true,
            ConfinesNetwork: true,
            ConfinesHomeReads: true,
            // Seatbelt policy is inherited across fork/exec, but macOS has no cgroup,
            // PID namespace, job object, or supported kernel descendant-tracking API.
            // The launcher kills the process group even after its leader exits, which
            // contains normal background children, but generated code can call setsid()
            // and leave that group. Claiming that nothing can outlive the request would
            // therefore turn a best-effort cleanup into a false security guarantee.
            BoundsProcessTree: false);

        public string Describe() =>
            "can deny IP network access (pathname Unix sockets remain scoped to shared temporary/scratch paths " +
            "plus the exact mDNSResponder endpoint when networking is enabled), " +
            "denies reads and file metadata of the user's home directory, and confines writes to the run's scratch directory";

        public bool TryWrap(
            SkillSandboxRequest request,
            out string fileName,
            out IReadOnlyList<string> arguments,
            out IDisposable cleanup,
            out string error)
        {
            fileName = null!;
            arguments = Array.Empty<string>();
            cleanup = null!;
            error = null!;

            if (!IsAvailable)
            {
                error = "sandbox-exec is not present on this host";
                return false;
            }

            // Pass the policy as an argument. A profile file creates a read-after-write
            // race with an earlier background command whenever TMPDIR resolves to a
            // shared writable directory; sandbox-exec supports the in-memory -p form, so
            // there is no file for generated code to replace before exec reads it.
            var argv = new List<string> { "-p", BuildProfile(request), request.Interpreter };
            argv.AddRange(request.Arguments);

            fileName = Helper;
            arguments = argv;
            cleanup = NullCleanup.Instance;
            return true;
        }

        /// <summary>
        /// Build the SBPL profile. Rules are evaluated in order with the LAST match
        /// winning, which is what lets a broad allow be carved back by a narrower deny
        /// and then punched through again for one directory.
        /// </summary>
        internal static string BuildProfile(SkillSandboxRequest request)
        {
            var sb = new StringBuilder();
            sb.AppendLine("(version 1)");
            sb.AppendLine("(deny default)");
            // Direct network opt-in opens the host's IP network. Keep pathname AF_UNIX
            // sockets scoped to scratch/shared-temp paths in both modes, and explicitly
            // deny launchd's shared-temp endpoints, while allowing Internet/LAN/loopback
            // TCP and UDP only in the opted-in mode. This reduces exposure to common
            // local control endpoints but is not a complete Unix-IPC boundary: tools
            // such as LibreOffice retain shared-temp singleton sockets for compatibility.
            sb.AppendLine(request.AllowNetwork ? "(allow network*)" : "(deny network*)");
            if (request.AllowNetwork)
            {
                sb.AppendLine("(deny network-bind (local unix-socket))");
                sb.AppendLine("(deny network-outbound (remote unix-socket))");
                // macOS getaddrinfo(3) talks to mDNSResponder over this exact local
                // pathname socket. Opening IP operations without it looks like network
                // access but cannot resolve a host name, so web search still fails. Keep
                // the exception literal and network-opt-in-only; no other /var/run
                // service endpoint is exposed by this rule.
                sb.AppendLine("(allow network-outbound (remote unix-socket (literal \"/private/var/run/mDNSResponder\")))");
                sb.AppendLine("(allow network-outbound (remote unix-socket (literal \"/var/run/mDNSResponder\")))");
            }
            sb.AppendLine("(allow system-socket (socket-domain AF_UNIX))");
            var socketRoots = new List<string> { "/private/tmp" };
            socketRoots.AddRange(Forms(request.WorkDirectory));
            foreach (string writable in request.WritablePaths ?? Array.Empty<string>())
            {
                if (!string.IsNullOrWhiteSpace(writable))
                    socketRoots.AddRange(Forms(writable));
            }
            foreach (string root in socketRoots.Distinct(StringComparer.Ordinal))
            {
                sb.Append("(allow network-bind (local unix-socket (subpath ").Append(Quote(root)).AppendLine(")))");
                sb.Append("(allow network-outbound (remote unix-socket (subpath ").Append(Quote(root)).AppendLine(")))");
            }
            sb.AppendLine("(deny network-outbound (remote unix-socket (regex #\"^/private/tmp/com\\.apple\\.launchd\")))");

            // The install phase's registry proxy: one loopback TCP port, and nothing
            // else. The proxy on the host side enforces the domain allowlist.
            if (!request.AllowNetwork && request.AllowLoopbackPort is int port and > 0 and <= 65535)
            {
                sb.Append("(allow network-outbound (remote ip \"localhost:")
                  .Append(port).AppendLine("\"))");
            }

            // The interpreter needs to fork/exec, read sysctls, talk to the bootstrap
            // server for dyld, and signal itself. Without these CPython aborts before
            // running a single line of the script.
            sb.AppendLine("(allow process-fork process-exec)");
            sb.AppendLine("(allow sysctl-read mach-lookup ipc-posix-shm)");
            sb.AppendLine("(allow signal (target self))");

            // Reads: broad, then the user's home carved out. Narrowing reads to a
            // system whitelist kills the interpreter (see the class remarks); what has
            // to be closed is the home directory, and it is.
            sb.AppendLine("(allow file-read*)");
            string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
            if (!string.IsNullOrEmpty(home))
            {
                foreach (string form in Forms(home))
                    sb.Append("(deny file-read* (subpath ").Append(Quote(form)).AppendLine("))");
            }

            // Metadata: readable everywhere OUTSIDE the home directory (a denial on
            // system paths surfaces as a confusing "file not found" from deep inside
            // a library rather than a permission error) — but hidden for FILES under
            // home. Existence is information: a script that can os.path.exists() its
            // way through ~/.ssh and ~/.aws learns which credentials this machine
            // holds even though every open() is denied. Directories stay visible
            // (vnode-type DIRECTORY) because the kernel checks each ancestor on the
            // way into the re-allowed subtrees below, and Seatbelt gives the more
            // specific file-read-metadata operation precedence over the general
            // file-read* allows — which is also why each re-allowed subtree needs
            // its own explicit metadata rule.
            sb.AppendLine("(allow file-read-metadata)");
            if (!string.IsNullOrEmpty(home))
            {
                foreach (string form in Forms(home))
                    sb.Append("(deny file-read-metadata (subpath ").Append(Quote(form)).AppendLine("))");
                sb.AppendLine("(allow file-read-metadata (vnode-type DIRECTORY))");
            }

            var readableRoots = new List<string>();
            readableRoots.AddRange(Forms(request.SkillDirectory));
            foreach (string extra in request.ReadablePaths ?? Array.Empty<string>())
            {
                if (string.IsNullOrWhiteSpace(extra))
                    continue;
                readableRoots.AddRange(Forms(extra));
            }
            foreach (string form in readableRoots)
            {
                string filter = File.Exists(form) ? "literal" : "subpath";
                sb.Append("(allow file-read* (").Append(filter).Append(' ')
                  .Append(Quote(form)).AppendLine("))");
                sb.Append("(allow file-read-metadata (").Append(filter).Append(' ')
                  .Append(Quote(form)).AppendLine("))");
            }

            sb.AppendLine("(allow file-write-data (literal \"/dev/null\"))");
            sb.AppendLine("(allow file-ioctl (literal \"/dev/null\") (literal \"/dev/urandom\"))");

            // Apple's /usr/bin/python3 is an xcrun shim that writes a cache database
            // into the per-user Darwin temp directory, which it locates through
            // confstr() rather than TMPDIR, so no environment scrubbing can redirect
            // it. Denying it costs nothing functionally — the interpreter still runs —
            // but it prints a permission error to stderr on EVERY run, and that stderr
            // goes to the model as part of the tool result, where it reads as the
            // script having failed. Allowing exactly that file name keeps the result
            // honest without opening the directory.
            sb.AppendLine("(allow file-read* file-write* (regex #\"^/private/var/folders/[^/]+/[^/]+/T/xcrun_db\"))");

            // The shared system temp, /tmp. A whole class of tools a skill invokes
            // keeps a FIXED-path scratch or singleton-IPC node here, independent of
            // TMPDIR and HOME (both already redirected into the workdir): LibreOffice's
            // headless converter opens /private/tmp/OSL_PIPE_<uid>_SingleOfficeIPC_<hash>
            // and, denied it, exits without building its profile — which is why the xlsx
            // skill's recalc.py could never recalculate a sheet under the sandbox. This
            // is an explicit compatibility exception: /private/tmp is world-writable
            // already (mode 1777 — every process on the host shares it), so generated
            // code can exchange data/IPC there. The user's home stays unreadable and the
            // session's own files live under the server's scratch root, never here. The
            // per-user Darwin temp (/var/folders/.../T) is left denied; nothing needed it
            // once /private/tmp was open.
            sb.AppendLine("(allow file-read* file-write* (subpath \"/private/tmp\"))");

            // Reassert the caller's boundary after the compatibility carve-out above.
            // This matters when a test or deployment itself lives under /private/tmp:
            // host-authored state stays read-only, then the work root and explicitly
            // named runtime-state paths are punched back through. Exact readable files
            // (notably sanitized per-session CA snapshots) stay read-only too.
            foreach (string form in Forms(request.SkillDirectory))
                sb.Append("(deny file-write* (subpath ").Append(Quote(form)).AppendLine("))");
            foreach (string extra in request.ReadablePaths ?? Array.Empty<string>())
            {
                if (string.IsNullOrWhiteSpace(extra))
                    continue;
                foreach (string form in Forms(extra))
                {
                    string filter = File.Exists(form) ? "literal" : "subpath";
                    sb.Append("(deny file-write* (").Append(filter).Append(' ')
                      .Append(Quote(form)).AppendLine("))");
                }
            }

            var writableRoots = new List<string>();
            writableRoots.AddRange(Forms(request.WorkDirectory));
            foreach (string extra in request.WritablePaths ?? Array.Empty<string>())
            {
                if (!string.IsNullOrWhiteSpace(extra))
                    writableRoots.AddRange(Forms(extra));
            }
            foreach (string form in writableRoots)
            {
                string filter = File.Exists(form) ? "literal" : "subpath";
                sb.Append("(allow file-read* file-write* (").Append(filter).Append(' ')
                  .Append(Quote(form)).AppendLine("))");
                sb.Append("(allow file-read-metadata (").Append(filter).Append(' ')
                  .Append(Quote(form)).AppendLine("))");
            }

            return sb.ToString();
        }

        /// <summary>
        /// Every spelling of <paramref name="path"/> a <c>subpath</c> rule might need to
        /// match: the path as given and, when they differ, the path with every symlinked
        /// ancestor resolved.
        ///
        /// <para>
        /// Resolving only the leaf is not enough, and getting this wrong silently
        /// TIGHTENS the sandbox rather than loosening it — which is how it was found.
        /// On macOS the system temp directory is <c>/var/folders/...</c> and <c>/var</c>
        /// is a symlink to <c>/private/var</c>; the scratch directory itself is not a
        /// link, so a leaf-only resolve returns the <c>/var</c> spelling, the kernel
        /// checks the <c>/private/var</c> one, no rule matches, and the script cannot
        /// write to its own working directory.
        /// </para>
        /// </summary>
        private static IReadOnlyList<string> Forms(string path)
        {
            string full;
            try
            {
                full = Path.GetFullPath(path);
            }
            catch (Exception ex) when (ex is ArgumentException or NotSupportedException or PathTooLongException)
            {
                return new[] { path };
            }

            string resolved = ResolveThroughAncestors(full);
            return string.Equals(resolved, full, StringComparison.Ordinal)
                ? new[] { full }
                : new[] { full, resolved };
        }

        /// <summary>
        /// Walk from the filesystem root, following each component that is a symlink, so
        /// the result is the path the kernel sees.
        /// </summary>
        private static string ResolveThroughAncestors(string fullPath)
        {
            try
            {
                string current = Path.DirectorySeparatorChar.ToString();
                foreach (string part in fullPath.Split(Path.DirectorySeparatorChar, StringSplitOptions.RemoveEmptyEntries))
                {
                    current = Path.Combine(current, part);
                    FileSystemInfo info = Directory.Exists(current) ? new DirectoryInfo(current) : new FileInfo(current);
                    if (!info.Exists)
                        continue;
                    FileSystemInfo? target = info.ResolveLinkTarget(returnFinalTarget: true);
                    if (target != null)
                        current = target.FullName;
                }
                return current;
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException
                                          or ArgumentException or NotSupportedException or PathTooLongException)
            {
                return fullPath;
            }
        }

        /// <summary>
        /// Quote a path for SBPL. A path is host-controlled rather than model-controlled
        /// (it is the skill root the operator configured plus a GUID), but a stray quote
        /// would silently truncate the profile and widen the sandbox, so it is escaped
        /// rather than trusted.
        ///
        /// <para>
        /// Internal rather than private so a test can build the rule it expects with the
        /// SAME escaping the profile was written with. A test that spelled the path out
        /// raw matched on every platform whose separator is not itself the escape
        /// character, and only on the one where it is did the expectation and the
        /// profile disagree.
        /// </para>
        /// </summary>
        internal static string Quote(string path) =>
            "\"" + path.Replace("\\", "\\\\").Replace("\"", "\\\"") + "\"";

        private sealed class NullCleanup : IDisposable
        {
            public static readonly NullCleanup Instance = new();

            public void Dispose() { }
        }
    }

    /// <summary>
    /// Linux namespaces, driven through <c>bwrap</c> (bubblewrap).
    ///
    /// <para>
    /// Bubblewrap is the unprivileged sandbox Flatpak is built on, and it is the only
    /// widely deployed mechanism a .NET process can drive without root or a helper
    /// daemon. It is present on most desktop distributions and almost no container
    /// images, which is exactly why <see cref="SkillSandboxMode.Required"/> is the
    /// default: a server that cannot isolate a script refuses instead of running it.
    /// </para>
    /// </summary>
    internal sealed class BubblewrapSandbox : ISkillSandbox
    {
        internal static readonly Version MinimumSafeVersion = new(0, 12, 0);
        private static readonly Lazy<(string? Path, string Error)> Located = new(Locate);

        internal static string AvailabilityError => Located.Value.Error;

        public string Name => "bubblewrap";

        public bool IsAvailable => OperatingSystem.IsLinux() && Located.Value.Path != null;

        public SkillSandboxCapabilities Capabilities => new(
            ConfinesWrites: true,
            ConfinesNetwork: true,
            ConfinesHomeReads: true,
            BoundsProcessTree: true);

        public string Describe() =>
            "can unshare the IP-network namespace, always unshares PID/IPC/UTS namespaces, mounts the filesystem read-only, " +
            "and confines writes to the run's scratch directory";

        public bool TryWrap(
            SkillSandboxRequest request,
            out string fileName,
            out IReadOnlyList<string> arguments,
            out IDisposable cleanup,
            out string error)
        {
            fileName = null!;
            arguments = Array.Empty<string>();
            cleanup = null!;
            error = null!;

            (string? bwrap, string reason) = Located.Value;
            if (bwrap == null)
            {
                error = reason;
                return false;
            }

            List<string> argv = BuildArguments(request);

            fileName = bwrap;
            arguments = argv;
            cleanup = NullCleanup.Instance;
            return true;
        }

        /// <summary>
        /// Build the platform-independent bubblewrap argument vector. Kept separate from
        /// locating/executing bwrap so policy construction is testable on macOS and
        /// Windows too; a cross-platform test can pin that only the network opt-in removes
        /// <c>--unshare-net</c> while all filesystem/process restrictions remain.
        /// </summary>
        internal static List<string> BuildArguments(SkillSandboxRequest request)
        {
            var argv = new List<string>
            {
                // Everything read-only, then the pieces that must be writable bound
                // back over it. --die-with-parent is what stops an abandoned script
                // outliving the request that started it.
                "--ro-bind", "/", "/",
                "--dev", "/dev",
                "--proc", "/proc",
                "--tmpfs", "/tmp",
                "--die-with-parent",
                // Start without the caller's controlling terminal. Bubblewrap's own
                // security guidance requires this unless a TIOCSTI-blocking seccomp
                // policy is installed; redirected stdio alone does not prove that the
                // child cannot still address the inherited controlling TTY.
                "--new-session",
                "--unshare-pid",
                "--unshare-ipc",
                "--unshare-uts",
            };

            // Hide host service sockets in both modes. A network-enabled command needs
            // DNS configuration, not Docker/containerd/D-Bus authority; the resolved
            // resolv.conf target is rebound read-only below when systemd stores it here.
            argv.Add("--tmpfs");
            argv.Add("/run");

            if (!request.AllowNetwork)
            {
                argv.Add("--unshare-net");
            }
            else if (ResolverRuntimeFile() is { } resolver)
            {
                argv.Add("--ro-bind");
                argv.Add(resolver);
                argv.Add(resolver);
            }

            // The user's home is replaced by an empty tmpfs rather than merely made
            // read-only: credentials and every other installed skill live there, and a
            // script that can read them can put them in its stdout and thus in the
            // model's context.
            string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
            if (!string.IsNullOrEmpty(home) && Directory.Exists(home))
            {
                argv.Add("--tmpfs");
                argv.Add(home);
            }

            argv.Add("--ro-bind");
            argv.Add(request.SkillDirectory);
            argv.Add(request.SkillDirectory);

            foreach (string extra in request.ReadablePaths ?? Array.Empty<string>())
            {
                if (string.IsNullOrWhiteSpace(extra)
                    || (!Directory.Exists(extra) && !File.Exists(extra)))
                    continue;
                argv.Add("--ro-bind");
                argv.Add(extra);
                argv.Add(extra);
            }

            argv.Add("--bind");
            argv.Add(request.WorkDirectory);
            argv.Add(request.WorkDirectory);
            foreach (string extra in request.WritablePaths ?? Array.Empty<string>())
            {
                if (string.IsNullOrWhiteSpace(extra)
                    || (!Directory.Exists(extra) && !File.Exists(extra)))
                    continue;
                argv.Add("--bind");
                argv.Add(extra);
                argv.Add(extra);
            }
            argv.Add("--chdir");
            argv.Add(request.StartDirectory ?? request.WorkDirectory);

            argv.Add("--");
            argv.Add(request.Interpreter);
            argv.AddRange(request.Arguments);
            return argv;
        }

        private static string? ResolverRuntimeFile()
        {
            if (!OperatingSystem.IsLinux())
                return null;
            try
            {
                FileSystemInfo? target = new FileInfo("/etc/resolv.conf")
                    .ResolveLinkTarget(returnFinalTarget: true);
                if (target == null || !File.Exists(target.FullName))
                    return null;
                string full = Path.GetFullPath(target.FullName);
                return full.Equals("/run", StringComparison.Ordinal)
                    || full.StartsWith("/run/", StringComparison.Ordinal)
                        ? full
                        : null;
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException
                                          or ArgumentException or NotSupportedException)
            {
                return null;
            }
        }

        private static (string? Path, string Error) Locate()
        {
            if (!OperatingSystem.IsLinux())
                return (null, "bubblewrap is available only on Linux");
            string? rejected = null;
            foreach (string candidate in new[] { "/usr/bin/bwrap", "/bin/bwrap", "/usr/local/bin/bwrap" })
            {
                if (!File.Exists(candidate))
                    continue;
                if (TryReadVersion(candidate, out Version? version)
                    && version >= MinimumSafeVersion)
                {
                    return (candidate, string.Empty);
                }
                rejected = version == null
                    ? $"{candidate} did not report a usable version"
                    : $"{candidate} {version} is unsafe; TensorSharp requires bubblewrap {MinimumSafeVersion} or newer";
            }
            return (null, rejected ?? "bwrap (bubblewrap) is not installed on this host");
        }

        internal static bool TryReadVersion(
            string path, out Version? version, int timeoutMilliseconds = 3000)
        {
            version = null;
            try
            {
                var output = new ConcurrentQueue<string>();
                var request = new SpawnRequest
                {
                    FileName = path,
                    Arguments = new[] { "--version" },
                    WorkingDirectory = Path.GetDirectoryName(path),
                    Environment = new Dictionary<string, string>(StringComparer.Ordinal)
                    {
                        ["PATH"] = Environment.GetEnvironmentVariable("PATH") ?? string.Empty,
                        ["LANG"] = "C.UTF-8",
                    },
                    OnStdoutLine = output.Enqueue,
                    OnStderrLine = output.Enqueue,
                };
                if (!SpawnedProcess.TryStart(request, out SpawnedProcess? process, out _)
                    || process == null)
                {
                    return false;
                }

                using (process)
                {
                    int timeout = Math.Clamp(timeoutMilliseconds, 1, 30_000);
                    if (!process.WaitForExit(timeout))
                    {
                        process.Kill();
                        process.WaitForExit(1000);
                        return false;
                    }

                    // SpawnedProcess drains stdout and stderr concurrently, avoiding the
                    // classic full-pipe deadlock. Keep the drain bounded too: a malicious
                    // wrapper can leave a descendant holding either descriptor open.
                    if (!process.WaitForDrain(Math.Min(timeout, 1000)))
                        return false;
                    return TryParseVersion(string.Join(" ", output), out version);
                }
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException
                                          or InvalidOperationException or System.ComponentModel.Win32Exception)
            {
                return false;
            }
        }

        internal static bool TryParseVersion(string? output, out Version? version)
        {
            version = null;
            if (string.IsNullOrWhiteSpace(output))
                return false;
            System.Text.RegularExpressions.Match match =
                System.Text.RegularExpressions.Regex.Match(output, @"(?:bubblewrap|bwrap)\s+(\d+(?:\.\d+){1,3})");
            return match.Success && Version.TryParse(match.Groups[1].Value, out version);
        }

        private sealed class NullCleanup : IDisposable
        {
            public static readonly NullCleanup Instance = new();

            public void Dispose() { }
        }
    }
}
