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
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using Microsoft.Win32.SafeHandles;
using TensorSharp.AgentHost.Skills;

namespace TensorSharp.AgentHost.CodeExec
{
    /// <summary>
    /// The shell state that survives from one call to the next — the working directory
    /// and the exported environment — and the script each command is wrapped in to
    /// restore and re-save it.
    ///
    /// <para>
    /// Every call is a FRESH confined process; there is no long-lived shell. That is
    /// deliberate, and it is the same choice Claude Code makes: a persistent shell
    /// process is a resource that leaks, a state that corrupts, and — here especially —
    /// a policy that cannot change, because a Seatbelt profile is fixed at exec time and
    /// the network decision has to be made per command. A hung command would poison the
    /// session rather than costing one call.
    /// </para>
    /// <para>
    /// So persistence is done with files instead. The wrapper sources the saved
    /// environment, changes to the saved directory, runs the model's command, and on the
    /// way out — through an EXIT trap, so an explicit <c>exit</c> inside the command does
    /// not skip it — writes both back. <c>cd</c>, <c>export</c> and
    /// <c>source .venv/bin/activate</c> then all persist, and nothing is left running.
    /// </para>
    /// </summary>
    public sealed class ShellSession
    {
        private readonly SessionWorkspace _workspace;
        private readonly ShellProgram _shell;
        private int _sequence;

        /// <param name="workspace">The session whose directories this drives.</param>
        /// <param name="shell">The resolved shell, which decides the dialect written.</param>
        public ShellSession(SessionWorkspace workspace, ShellProgram shell)
        {
            _workspace = workspace ?? throw new ArgumentNullException(nameof(workspace));
            _shell = shell ?? throw new ArgumentNullException(nameof(shell));
        }

        /// <summary>Where the persisted working directory is kept.</summary>
        private string CwdFile => Path.Combine(_workspace.ShellStateDirectory, "cwd");

        /// <summary>Where the persisted exported environment is kept.</summary>
        private string EnvFile => Path.Combine(_workspace.ShellStateDirectory,
            _shell.Kind == ShellKind.PowerShell ? "env.txt" : "env.sh");

        /// <summary>
        /// The same path, for a test that has to corrupt the saved environment on
        /// purpose. Exposed rather than letting a test re-spell the leaf name, which
        /// would quietly stop testing anything the day the name changed.
        /// </summary>
        internal string EnvironmentFilePath => EnvFile;

        /// <summary>
        /// The directory the next command will start in — the work directory until
        /// something <c>cd</c>s, and whatever it left behind after that.
        /// </summary>
        public string CurrentDirectory
        {
            get
            {
                try
                {
                    if (TryReadStateText(CwdFile, out string saved))
                    {
                        saved = saved.Trim();
                        if (saved.Length > 0 && Directory.Exists(saved) && IsInsideRoot(saved)
                            && SkillPathGuard.TryResolveSymlinks(
                                Path.GetFullPath(_workspace.Root), Path.GetFullPath(saved),
                                out string? resolved, out _)
                            && resolved != null && Directory.Exists(resolved))
                        {
                            // Use the resolved spelling as well as validating it. That
                            // keeps a persisted in-workspace directory symlink from
                            // becoming an escape if its target is changed between calls.
                            return resolved;
                        }
                    }
                }
                catch (Exception ex) when (ex is IOException or UnauthorizedAccessException
                                              or ArgumentException or NotSupportedException
                                              or PathTooLongException) { }
                return _workspace.WorkDirectory;
            }
        }

        /// <summary>
        /// Read one of the small files written by the model-owned shell wrapper without
        /// letting the model turn a host-side status lookup into an unbounded read.
        ///
        /// <para>
        /// The leaf is attacker-controlled: a command can replace <c>cwd</c> with a
        /// symlink or FIFO before the unsandboxed host asks for the next prompt label.
        /// <see cref="File.ReadAllText(string)"/> follows the former and blocks on the
        /// latter; a link to <c>/dev/zero</c> additionally grows memory until the host is
        /// killed. Open the leaf without following it, inspect that exact handle, accept
        /// only a regular file, and never allocate or read beyond a path-sized cap.
        /// </para>
        /// </summary>
        internal static bool TryReadStateText(string path, out string text)
            => TryReadBoundedRegularText(path, MaxStateFileBytes, out text);

        /// <summary>
        /// Read a bounded regular file through the exact no-follow handle that was
        /// inspected. This is also used for model-authored source locations: those paths
        /// can be replaced with a link, FIFO, device or socket after a command exits, so
        /// checking a <see cref="FileInfo"/> and then reopening by name is not safe.
        /// </summary>
        internal static bool TryReadBoundedRegularText(string path, int maxBytes, out string text)
        {
            text = string.Empty;
            if (maxBytes < 0)
                return false;
            try
            {
                using SafeFileHandle? handle = OpenStateFileNoFollow(path);
                return handle != null && !handle.IsInvalid
                    && TryReadBoundedRegularText(handle, maxBytes, out text);
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException
                                          or ArgumentException or NotSupportedException
                                          or DllNotFoundException or EntryPointNotFoundException)
            {
                text = string.Empty;
                return false;
            }
        }

        /// <summary>
        /// The workspace form additionally anchors every path component to an already
        /// open root. A background process can rename a parent directory between path
        /// validation and the read; opening each Unix component with openat/no-follow,
        /// or validating the Windows file handle's final path against the root handle,
        /// prevents that race from redirecting a diagnostic excerpt outside the worktree.
        /// </summary>
        internal static bool TryReadBoundedRegularTextUnderRoot(
            string root, string path, int maxBytes, out string text)
        {
            text = string.Empty;
            if (maxBytes < 0)
                return false;
            try
            {
                using SafeFileHandle? handle = OpenFileUnderRootNoFollow(root, path);
                return handle != null && !handle.IsInvalid
                    && TryReadBoundedRegularText(handle, maxBytes, out text);
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException
                                          or ArgumentException or NotSupportedException
                                          or DllNotFoundException or EntryPointNotFoundException)
            {
                text = string.Empty;
                return false;
            }
        }

        private static bool TryReadBoundedRegularText(
            SafeFileHandle handle, int maxBytes, out string text)
        {
            text = string.Empty;
            if (!TryGetRegularFileSnapshot(handle, out RegularFileSnapshot before)
                || before.Length < 0 || before.Length > maxBytes || before.Length >= int.MaxValue)
            {
                return false;
            }
            long length = before.Length;

            // One extra byte detects a file extended after fstat/GetFileInformation.
            // A shrink is rejected by the exact-length check below. Both make a
            // concurrently rewritten file cost one fallback, not a torn excerpt.
            byte[] bytes = new byte[checked((int)length + 1)];
            int total = 0;
            bool reachedEnd = false;
            while (total < bytes.Length)
            {
                int read = RandomAccess.Read(handle, bytes.AsSpan(total), total);
                if (read == 0)
                {
                    reachedEnd = true;
                    break;
                }
                total += read;
            }
            if (!reachedEnd || total != (int)length
                || !TryGetRegularFileSnapshot(handle, out RegularFileSnapshot after)
                || after != before)
                return false;

            // Windows PowerShell 5.1 writes UTF-16 with a BOM; PowerShell 7 and the
            // POSIX wrappers write UTF-8. StreamReader retains File.ReadAllText's BOM
            // detection while operating only on the bounded bytes above.
            using var memory = new MemoryStream(bytes, 0, total, writable: false, publiclyVisible: false);
            using var reader = new StreamReader(memory, Encoding.UTF8,
                detectEncodingFromByteOrderMarks: true, bufferSize: 1024, leaveOpen: false);
            text = reader.ReadToEnd();
            return true;
        }

        /// <summary>
        /// Paths can exceed the traditional POSIX PATH_MAX, while an extended Windows
        /// path can contain 32,767 UTF-16 code units (up to four UTF-8 bytes each).
        /// This covers both plus a line terminator without making attacker-controlled
        /// state an appreciable allocation.
        /// </summary>
        private const int MaxStateFileBytes = 128 * 1024;

        private static SafeFileHandle? OpenStateFileNoFollow(string path)
        {
            if (OperatingSystem.IsWindows())
            {
                SafeFileHandle handle = CreateFileW(path, GenericRead,
                    FileShareRead | FileShareWrite | FileShareDelete, IntPtr.Zero,
                    OpenExisting, FileFlagOpenReparsePoint | SecuritySqosPresent, IntPtr.Zero);
                if (!handle.IsInvalid)
                    return handle;
                handle.Dispose();
                return null;
            }

            int flags;
            if (OperatingSystem.IsLinux())
                flags = LinuxONonBlock | LinuxONoFollow | LinuxOCloseExec;
            else if (OperatingSystem.IsMacOS())
                flags = MacONonBlock | MacONoFollow | MacOCloseExec;
            else
                return null; // Safe fallback for a Unix whose open(2) values we do not know.

            int fd;
            do
            {
                fd = OpenUnix(path, flags);
            }
            while (fd < 0 && Marshal.GetLastWin32Error() == InterruptedSystemCall);
            return fd < 0 ? null : new SafeFileHandle((IntPtr)fd, ownsHandle: true);
        }

        private static SafeFileHandle? OpenFileUnderRootNoFollow(string root, string path)
        {
            string fullRoot = Path.GetFullPath(root);
            string fullPath = Path.GetFullPath(path);
            string relative = Path.GetRelativePath(fullRoot, fullPath);
            string[] segments = relative.Split(
                new[] { Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar },
                StringSplitOptions.RemoveEmptyEntries);
            if (segments.Length == 0 || Path.IsPathRooted(relative)
                || segments.Any(segment => segment is "." or ".."))
            {
                return null;
            }

            if (OperatingSystem.IsWindows())
                return OpenWindowsFileUnderRoot(fullRoot, fullPath);
            if (!OperatingSystem.IsLinux() && !OperatingSystem.IsMacOS())
                return null;

            int noFollow = OperatingSystem.IsMacOS() ? MacONoFollow : LinuxONoFollow;
            int closeExec = OperatingSystem.IsMacOS() ? MacOCloseExec : LinuxOCloseExec;
            int directoryFlag = OperatingSystem.IsMacOS() ? MacODirectory : LinuxODirectory;
            int nonBlock = OperatingSystem.IsMacOS() ? MacONonBlock : LinuxONonBlock;

            int rootFd;
            do
            {
                rootFd = OpenUnix(fullRoot, noFollow | closeExec | directoryFlag);
            }
            while (rootFd < 0 && Marshal.GetLastWin32Error() == InterruptedSystemCall);
            if (rootFd < 0)
                return null;

            SafeFileHandle? directory = new((IntPtr)rootFd, ownsHandle: true);
            try
            {
                for (int index = 0; index < segments.Length; index++)
                {
                    bool leaf = index == segments.Length - 1;
                    int flags = noFollow | closeExec
                              | (leaf ? nonBlock : directoryFlag);
                    int fd;
                    do
                    {
                        fd = OpenAtUnix(
                            directory.DangerousGetHandle().ToInt32(), segments[index], flags);
                    }
                    while (fd < 0 && Marshal.GetLastWin32Error() == InterruptedSystemCall);
                    if (fd < 0)
                        return null;

                    var next = new SafeFileHandle((IntPtr)fd, ownsHandle: true);
                    if (leaf)
                        return next;

                    directory.Dispose();
                    directory = next;
                }
                return null;
            }
            finally
            {
                directory.Dispose();
            }
        }

        private static SafeFileHandle? OpenWindowsFileUnderRoot(string root, string path)
        {
            // Final-handle comparison rejects ordinary intermediate junctions and swaps.
            // Windows code execution is available only through the explicit unconfined
            // mode, so this is defense in depth rather than a filesystem-isolation claim:
            // an adversarial host-user process can also rename the root itself.
            using SafeFileHandle rootHandle = CreateFileW(
                root, 0, FileShareRead | FileShareWrite | FileShareDelete, IntPtr.Zero,
                OpenExisting,
                FileFlagOpenReparsePoint | FileFlagBackupSemantics | SecuritySqosPresent,
                IntPtr.Zero);
            if (rootHandle.IsInvalid
                || !GetFileInformationByHandle(rootHandle, out ByHandleFileInformation rootInfo)
                || (rootInfo.FileAttributes & FileAttributeDirectory) == 0
                || (rootInfo.FileAttributes & FileAttributeReparsePoint) != 0
                || !TryGetFinalPath(rootHandle, out string finalRoot))
            {
                return null;
            }

            SafeFileHandle fileHandle = CreateFileW(
                path, GenericRead, FileShareRead | FileShareDelete, IntPtr.Zero,
                OpenExisting, FileFlagOpenReparsePoint | SecuritySqosPresent, IntPtr.Zero);
            if (fileHandle.IsInvalid || !TryGetFinalPath(fileHandle, out string finalFile))
            {
                fileHandle.Dispose();
                return null;
            }

            string rootPrefix = finalRoot.TrimEnd('\\', '/') + "\\";
            string normalizedFile = finalFile.Replace('/', '\\');
            if (!normalizedFile.StartsWith(rootPrefix.Replace('/', '\\'), StringComparison.OrdinalIgnoreCase))
            {
                fileHandle.Dispose();
                return null;
            }
            return fileHandle;
        }

        private static bool TryGetFinalPath(SafeFileHandle handle, out string path)
        {
            path = string.Empty;
            var buffer = new StringBuilder(512);
            uint length = GetFinalPathNameByHandleW(handle, buffer, (uint)buffer.Capacity, 0);
            if (length == 0)
                return false;
            if (length >= (uint)buffer.Capacity)
            {
                if (length >= 32768)
                    return false;
                buffer = new StringBuilder(checked((int)length + 1));
                length = GetFinalPathNameByHandleW(handle, buffer, (uint)buffer.Capacity, 0);
                if (length == 0 || length >= (uint)buffer.Capacity)
                    return false;
            }
            path = buffer.ToString();
            return path.Length > 0;
        }

        private static bool TryGetRegularFileSnapshot(
            SafeFileHandle handle, out RegularFileSnapshot snapshot)
        {
            snapshot = default;
            if (OperatingSystem.IsWindows())
            {
                if (GetFileType(handle) != FileTypeDisk
                    || !GetFileInformationByHandle(handle, out ByHandleFileInformation info)
                    || (info.FileAttributes & (FileAttributeDirectory | FileAttributeReparsePoint)) != 0)
                {
                    return false;
                }

                ulong unsignedLength = ((ulong)info.FileSizeHigh << 32) | info.FileSizeLow;
                if (unsignedLength > long.MaxValue)
                    return false;
                snapshot = new RegularFileSnapshot(
                    (long)unsignedLength,
                    info.VolumeSerialNumber,
                    ((ulong)info.FileIndexHigh << 32) | info.FileIndexLow,
                    ((ulong)info.LastWriteTime.HighDateTime << 32) | info.LastWriteTime.LowDateTime,
                    0);
                return true;
            }

            if (OperatingSystem.IsLinux() || OperatingSystem.IsMacOS())
            {
                int result;
                UnixFileStatus status;
                do
                {
                    result = FStatUnix(handle, out status);
                }
                while (result != 0 && Marshal.GetLastWin32Error() == InterruptedSystemCall);

                if (result == 0 && (status.Mode & UnixFileTypeMask) == UnixRegularFile)
                {
                    snapshot = new RegularFileSnapshot(
                        status.Size,
                        unchecked((ulong)status.Dev),
                        unchecked((ulong)status.Ino),
                        CombineUnixTime(status.MTime, status.MTimeNsec),
                        CombineUnixTime(status.CTime, status.CTimeNsec));
                    return true;
                }
            }
            return false;
        }

        private static ulong CombineUnixTime(long seconds, long nanoseconds) =>
            unchecked((ulong)seconds * 1_000_000_000UL + (ulong)nanoseconds);

        private readonly record struct RegularFileSnapshot(
            long Length, ulong Device, ulong Identity, ulong Modified, ulong Changed);

        // open(2) flag values are ABI values, not POSIX constants, and differ between
        // Linux and Darwin. O_RDONLY is zero on both platforms.
        private const int LinuxONonBlock = 0x00000800;
        private const int LinuxODirectory = 0x00010000;
        private const int LinuxONoFollow = 0x00020000;
        private const int LinuxOCloseExec = 0x00080000;
        private const int MacONonBlock = 0x00000004;
        private const int MacONoFollow = 0x00000100;
        private const int MacODirectory = 0x00100000;
        private const int MacOCloseExec = 0x01000000;
        private const int InterruptedSystemCall = 4;

        private const int UnixFileTypeMask = 0xF000;
        private const int UnixRegularFile = 0x8000;

        private const uint GenericRead = 0x80000000;
        private const uint FileShareRead = 0x00000001;
        private const uint FileShareWrite = 0x00000002;
        private const uint FileShareDelete = 0x00000004;
        private const uint OpenExisting = 3;
        private const uint SecuritySqosPresent = 0x00100000;
        private const uint FileFlagBackupSemantics = 0x02000000;
        private const uint FileFlagOpenReparsePoint = 0x00200000;
        private const uint FileTypeDisk = 0x0001;
        private const uint FileAttributeDirectory = 0x00000010;
        private const uint FileAttributeReparsePoint = 0x00000400;

        [DllImport("libc", EntryPoint = "open", SetLastError = true)]
        private static extern int OpenUnix(
            [MarshalAs(UnmanagedType.LPUTF8Str)] string path, int flags);

        [DllImport("libc", EntryPoint = "openat", SetLastError = true)]
        private static extern int OpenAtUnix(
            int directoryFd,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string path,
            int flags);

        [DllImport("libSystem.Native", EntryPoint = "SystemNative_FStat", SetLastError = true)]
        private static extern int FStatUnix(SafeFileHandle handle, out UnixFileStatus status);

        [DllImport("kernel32.dll", CharSet = CharSet.Unicode,
            ExactSpelling = true, SetLastError = true)]
        private static extern SafeFileHandle CreateFileW(
            string fileName, uint desiredAccess, uint shareMode, IntPtr securityAttributes,
            uint creationDisposition, uint flagsAndAttributes, IntPtr templateFile);

        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern uint GetFileType(SafeFileHandle handle);

        [DllImport("kernel32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool GetFileInformationByHandle(
            SafeFileHandle handle, out ByHandleFileInformation information);

        [DllImport("kernel32.dll", CharSet = CharSet.Unicode,
            ExactSpelling = true, SetLastError = true)]
        private static extern uint GetFinalPathNameByHandleW(
            SafeFileHandle handle, StringBuilder path, uint pathLength, uint flags);

        /// <summary>
        /// Stable layout exported by the .NET runtime's System.Native shim. Using its
        /// normalized status record avoids hard-coding the incompatible Linux/Darwin
        /// struct stat layouts while still checking the already-open descriptor.
        /// </summary>
        [StructLayout(LayoutKind.Sequential)]
        private struct UnixFileStatus
        {
            public int Flags;
            public int Mode;
            public uint Uid;
            public uint Gid;
            public long Size;
            public long ATime;
            public long ATimeNsec;
            public long MTime;
            public long MTimeNsec;
            public long CTime;
            public long CTimeNsec;
            public long BirthTime;
            public long BirthTimeNsec;
            public long Dev;
            public long RDev;
            public long Ino;
            public uint UserFlags;
            // Appended in newer System.Native versions without changing the 64-bit
            // record size (it occupied prior tail padding). Keeping it explicit is also
            // safe with older runtimes, which simply leave it zero.
            public uint HardLinkCount;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct FileTime
        {
            public uint LowDateTime;
            public uint HighDateTime;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct ByHandleFileInformation
        {
            public uint FileAttributes;
            public FileTime CreationTime;
            public FileTime LastAccessTime;
            public FileTime LastWriteTime;
            public uint VolumeSerialNumber;
            public uint FileSizeHigh;
            public uint FileSizeLow;
            public uint NumberOfLinks;
            public uint FileIndexHigh;
            public uint FileIndexLow;
        }

        /// <summary>
        /// <see cref="CurrentDirectory"/> written the way the model addressed it: relative
        /// to the work directory, or <c>.</c> when it is the work directory itself.
        ///
        /// <para>
        /// Never the absolute path. It is host-specific noise in a tool result the model
        /// will quote back, and — because the injected prompt sits in prefix-cache block
        /// zero — an absolute path that varies between hosts is exactly the kind of text
        /// that must not leak into anything cached.
        /// </para>
        /// </summary>
        public string CurrentDirectoryLabel
        {
            get
            {
                string relative = Path.GetRelativePath(_workspace.WorkDirectory, CurrentDirectory)
                    .Replace('\\', '/');
                return relative is "." or "" ? "." : relative;
            }
        }

        private bool IsInsideRoot(string path)
        {
            string root = Path.GetFullPath(_workspace.Root);
            string full = Path.GetFullPath(path);
            StringComparison comparison = OperatingSystem.IsLinux()
                ? StringComparison.Ordinal
                : StringComparison.OrdinalIgnoreCase;
            return full.Equals(root, comparison)
                || full.StartsWith(root + Path.DirectorySeparatorChar, comparison);
        }

        /// <summary>
        /// True when the last command found the saved environment unusable and started
        /// from a clean one — so the result can SAY so.
        ///
        /// <para>
        /// A value containing a newline makes <c>export -p</c> emit a multi-line record
        /// that the save-side filter can cut in half, and the file that leaves is not
        /// valid shell. Resetting it is the safe response; doing that silently means a
        /// variable the model set two calls ago is simply gone, and it has no way to tell
        /// that from having mistyped the name.
        /// </para>
        /// </summary>
        public bool TakeEnvironmentWasReset()
        {
            string marker = EnvFile + ".reset";
            try
            {
                if (!File.Exists(marker))
                    return false;
                File.Delete(marker);
                return true;
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                return false;
            }
        }

        /// <summary>Forget the working directory and the environment. Used by tests and by a reset.</summary>
        public void Reset()
        {
            // The reset marker goes with the file it describes. Left behind, a later
            // command reads it as "the environment you exported was thrown away" when
            // nothing of the sort happened on that command. The Exists probe stays: on
            // Windows a File.Delete of a path that is not there costs more than asking
            // whether it is, and the marker is normally not there.
            foreach (string file in new[] { CwdFile, EnvFile, EnvFile + ".reset" })
            {
                try { if (File.Exists(file)) File.Delete(file); }
                catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { }
            }
        }

        /// <summary>
        /// A written wrapper script: where it is, and how many of its lines come before
        /// the model's own command.
        ///
        /// <para>
        /// The offset is what lets a shell's own diagnostics be rewritten into the
        /// model's frame of reference. Bash reports an error as
        /// "<c>/…/state/cmd-7.sh: line 24: foo: command not found</c>" — a path the model
        /// has never seen and a line number twenty lines past anything it wrote. Left
        /// alone it goes looking for line 24 of a file it did not write.
        /// </para>
        /// </summary>
        /// <param name="Path">The script on disk.</param>
        /// <param name="CommandOffset">Lines of wrapper before the command's first line.</param>
        public readonly record struct ShellScript(string Path, int CommandOffset);

        // ---- what has already been tried -----------------------------------

        /// <summary>
        /// How many times this exact command has already been run in this session, and
        /// whether it failed. Records this attempt.
        /// </summary>
        /// <param name="command">The command as the model typed it.</param>
        /// <returns>
        /// Attempts BEFORE this one, and whether any of them failed. Zero and false for a
        /// command that has not been seen.
        /// </returns>
        public (int Before, bool FailedBefore) RecordAttempt(string command)
        {
            string key = Key(command ?? string.Empty);
            lock (_attempts)
            {
                if (_attempts.Count >= MaxRemembered && !_attempts.ContainsKey(key))
                    _attempts.Clear();
                _attempts.TryGetValue(key, out (int Count, bool Failed) seen);
                _attempts[key] = (seen.Count + 1, seen.Failed);
                return (seen.Count, seen.Failed);
            }
        }

        /// <summary>Record how the attempt just made turned out.</summary>
        public void RecordOutcome(string command, bool ok)
        {
            string key = Key(command ?? string.Empty);
            lock (_attempts)
            {
                if (_attempts.TryGetValue(key, out (int Count, bool Failed) seen))
                    _attempts[key] = (seen.Count, seen.Failed || !ok);
            }
        }

        /// <summary>
        /// Commands are compared with all whitespace removed, so a re-send that differs
        /// only in formatting still counts as the same attempt.
        ///
        /// <para>
        /// This is not tidiness. In the worst loop in this server's logs the model sent
        /// one broken line of Python nine times; eight were byte-identical and the ninth
        /// differed only by spaces around <c>*</c> and <c>+</c>. A comparison that missed
        /// that one would have reported two short streaks instead of one long one, which
        /// is the difference between a warning that lands and one that does not.
        /// </para>
        /// </summary>
        private static string Key(string command)
        {
            var sb = new System.Text.StringBuilder(command.Length);
            foreach (char c in command)
            {
                if (!char.IsWhiteSpace(c))
                    sb.Append(c);
            }
            return sb.ToString();
        }

        /// <summary>
        /// Bounded, because a long conversation must not accumulate every command it ever
        /// ran. Beyond the cap the ledger is cleared rather than trimmed: the point is to
        /// catch a loop, which is always recent, and a half-remembered history would tell
        /// the model "you have run this once before" about a command it ran five times.
        /// </summary>
        private const int MaxRemembered = 200;

        private readonly Dictionary<string, (int Count, bool Failed)> _attempts =
            new(StringComparer.Ordinal);

        /// <summary>
        /// Write the script for one command and return where it is.
        /// </summary>
        /// <param name="command">The model's command line, spliced in verbatim.</param>
        /// <param name="workDirectory">
        /// An absolute directory the call asked to run in, already validated to be inside
        /// the workspace, or null to continue where the last command left off.
        /// </param>
        public ShellScript WriteScript(string command, string? workDirectory)
        {
            int n = Interlocked.Increment(ref _sequence);
            string path = Path.Combine(
                _workspace.StateDirectory,
                "cmd-" + n.ToString(CultureInfo.InvariantCulture) + _shell.ScriptExtension);

            string text = _shell.Kind == ShellKind.PowerShell
                ? PowerShellScript(command, workDirectory)
                : PosixScript(command, workDirectory);

            // Count the newlines up to and INCLUDING the marker's own line terminator:
            // that many lines precede the model's first line. Counting to the end of the
            // marker's TEXT stops one newline short, and every diagnostic then pointed the
            // model at the line after the one it wrote.
            int offset = 0;
            int marker = text.IndexOf(CommandMarker, StringComparison.Ordinal);
            if (marker >= 0)
            {
                int endOfMarkerLine = text.IndexOf('\n', marker);
                offset = endOfMarkerLine < 0
                    ? text.Count(c => c == '\n')
                    : text.Take(endOfMarkerLine + 1).Count(c => c == '\n');
            }

            // The encoding is not a detail on Windows. Windows PowerShell 5.1 decodes a
            // -File script as the system's ANSI code page unless it starts with a UTF-8
            // BOM, and File.WriteAllText writes UTF-8 WITHOUT one. Any non-ASCII byte in
            // the model's command — an em dash in a slide title, a CJK string, an arrow
            // in a comment — therefore reached the shell as mojibake, and the file the
            // command went on to write carried the damage. A BOM costs three bytes and
            // PowerShell 7 reads it identically; a POSIX shell must NOT get one, because
            // `#!` has to be the first two bytes.
            File.WriteAllText(path, text, ScriptEncoding);
            PruneOldScripts(n);
            if (!OperatingSystem.IsWindows())
            {
                try { File.SetUnixFileMode(path, UnixFileMode.UserRead | UnixFileMode.UserWrite | UnixFileMode.UserExecute); }
                catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { }
            }
            return new ShellScript(path, offset);
        }

        /// <summary>The line the model's own command starts after, in both dialects.</summary>
        internal const string CommandMarker = "# ---- command ----";

        /// <summary>
        /// How a command script is encoded: UTF-8, with a BOM only for PowerShell.
        /// See the note at the <c>WriteAllText</c> call above.
        /// </summary>
        private Encoding ScriptEncoding =>
            new UTF8Encoding(encoderShouldEmitUTF8Identifier: _shell.Kind == ShellKind.PowerShell);

        /// <summary>How many recent command scripts are kept on disk.</summary>
        /// <remarks>
        /// A few, not none: a background job is still executing its script long after the
        /// call that started it returned, so deleting immediately would pull the file out
        /// from under a running shell. A few, not all: a long conversation otherwise
        /// leaves hundreds of small files in a directory the model can see.
        /// </remarks>
        private const int ScriptsKept = 16;

        private void PruneOldScripts(int current)
        {
            int stale = current - ScriptsKept;
            if (stale < 1)
                return;
            string path = Path.Combine(
                _workspace.StateDirectory,
                "cmd-" + stale.ToString(CultureInfo.InvariantCulture) + _shell.ScriptExtension);
            try { if (File.Exists(path)) File.Delete(path); }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { }
        }

        /// <summary>
        /// Variables the wrapper must never carry from one call to the next.
        ///
        /// <para>
        /// Two different reasons, both of which have to hold. <c>PATH</c>, <c>HOME</c>,
        /// <c>TMPDIR</c> and the proxy variables are set by the host on every launch and
        /// carry the sandbox's own decisions; letting a saved copy win would mean a
        /// command from ten calls ago quietly deciding where this one looks for its
        /// interpreter, or which proxy an installer uses. <c>LD_PRELOAD</c> and
        /// <c>DYLD_*</c> are how a process is made to load something before its own
        /// <c>main</c>, and nothing should acquire that by being exported once.
        /// </para>
        /// </summary>
        private const string PosixEnvFilter =
            // The option letters are a CHARACTER CLASS, not the single "-x" the first
            // version matched: bash writes `declare -rx NAME=` for a readonly export and
            // `declare -ax` for an array one, and both slipped past a filter looking for
            // "declare -x ". That let a command make PATH or LD_PRELOAD persist by
            // declaring it readonly — the exact two variables this filter exists for.
            "^(declare -[-aAilnrtux]+ |export |typeset -[-aAilnrtux]+ |readonly )?"
            + "(PATH|HOME|TMPDIR|TEMP|TMP|PWD|OLDPWD|SHLVL|IFS|_|"
            + "LD_PRELOAD|LD_LIBRARY_PATH|DYLD_[A-Za-z_]*|"
            + "HTTP_PROXY|HTTPS_PROXY|ALL_PROXY|NO_PROXY|"
            + "http_proxy|https_proxy|all_proxy|no_proxy|GRPC_PROXY|grpc_proxy|"
            + "SSL_CERT_FILE|SSL_CERT_DIR|REQUESTS_CA_BUNDLE|CURL_CA_BUNDLE|"
            + "NODE_EXTRA_CA_CERTS|GIT_SSL_CAINFO|AWS_CA_BUNDLE|PIP_CERT|NPM_CONFIG_CAFILE|"
            + "CARGO_HTTP_CAINFO|DENO_CERT|CLOUDSDK_CORE_CUSTOM_CA_CERTS_FILE|"
            + "NIX_SSL_CERT_FILE|GRPC_DEFAULT_SSL_ROOTS_FILE_PATH|"
            + "TS_[A-Za-z_]*)=";

        private static string Sq(string value) => ShellCommand.QuotePosix(value);

        private string PosixScript(string command, string? workDirectory)
        {
            var sb = new StringBuilder();
            sb.Append("# TensorSharp shell wrapper. Everything between the markers is the command\n");
            sb.Append("# as the model wrote it; the rest restores and re-saves the session's state.\n");
            sb.Append("__ts_state=").Append(Sq(_workspace.ShellStateDirectory)).Append('\n');
            sb.Append("__ts_work=").Append(Sq(_workspace.WorkDirectory)).Append('\n');
            sb.Append("__ts_root=").Append(Sq(_workspace.Root)).Append('\n');
            sb.Append("__ts_env=").Append(Sq(EnvFile)).Append('\n');
            sb.Append("__ts_cwdfile=").Append(Sq(CwdFile)).Append('\n');

            // Restore. Failures are swallowed: a corrupt state file must cost this command
            // its inherited environment, never the command itself.
            // Validate before sourcing, in a SUBSHELL. `export -p` emits a multi-line
            // record for a value containing a newline, the save-side filter drops only the
            // line that matched, and what is left has a dangling quote — sourcing that in
            // the real shell aborts it under `sh` and silently swallows every later
            // variable under bash. Sourcing a copy first turns a corrupt file into one
            // reset of the session's environment, which the next command can rebuild.
            sb.Append("if [ -r \"$__ts_env\" ]; then\n");
            sb.Append("  if ( . \"$__ts_env\" ) >/dev/null 2>&1; then . \"$__ts_env\" 2>/dev/null || true;\n");
            sb.Append("  else : > \"$__ts_env\" 2>/dev/null || true; : > \"$__ts_env.reset\" 2>/dev/null || true; fi\n");
            sb.Append("fi\n");
            // The saved directory is accepted only if it is still inside the workspace —
            // the same test the host applies when it reads the file back. Without it a
            // `cd /usr/lib` persisted for the shell while the host still believed the
            // session was in the work directory, so the two halves of this tool surface
            // disagreed about what a relative path meant.
            sb.Append("__ts_cwd=$__ts_work\n");
            sb.Append("if [ -r \"$__ts_cwdfile\" ]; then __ts_cwd=$(cat \"$__ts_cwdfile\" 2>/dev/null) || __ts_cwd=$__ts_work; fi\n");
            sb.Append("case \"$__ts_cwd\" in \"$__ts_root\"|\"$__ts_root\"/*) ;; *) __ts_cwd=$__ts_work ;; esac\n");
            sb.Append("[ -d \"$__ts_cwd\" ] || __ts_cwd=$__ts_work\n");
            sb.Append("cd \"$__ts_cwd\" 2>/dev/null || cd \"$__ts_work\" || exit 1\n");

            // A per-call workdir is for THIS command only, so the directory the session
            // resumes from is captured before moving. Without this, one `workdir` moved
            // the conversation permanently and the next command — and every relative path
            // in a patch — quietly meant somewhere else.
            sb.Append("__ts_resume=$(pwd)\n");
            if (workDirectory != null)
            {
                sb.Append("cd ").Append(Sq(workDirectory))
                  .Append(" || { echo \"workdir does not exist: ")
                  .Append(Path.GetRelativePath(_workspace.WorkDirectory, workDirectory).Replace('\\', '/'))
                  .Append("\" >&2; exit 1; }\n");
            }

            // Saving through a trap rather than after the command, so that a command
            // ending in `exit 3` still records where it left the session.
            sb.Append("__ts_save() {\n");
            sb.Append("  __ts_rc=$?\n");
            sb.Append(workDirectory != null
                ? "  printf '%s' \"$__ts_resume\" > \"$__ts_cwdfile\" 2>/dev/null || true\n"
                : "  pwd > \"$__ts_cwdfile\" 2>/dev/null || true\n");
            sb.Append("  { export -p 2>/dev/null | grep -Ev ").Append(Sq(PosixEnvFilter))
              .Append(" | head -c 262144 > \"$__ts_env.tmp\" && mv \"$__ts_env.tmp\" \"$__ts_env\"; } 2>/dev/null || true\n");
            sb.Append("  return $__ts_rc\n");
            sb.Append("}\n");
            sb.Append("trap __ts_save EXIT\n");

            sb.Append(CommandMarker).Append('\n');
            sb.Append(command);
            if (!command.EndsWith('\n'))
                sb.Append('\n');
            sb.Append("# ---- end command ----\n");
            return sb.ToString();
        }

        private static string Pq(string value) => ShellCommand.QuotePowerShell(value);

        private string PowerShellScript(string command, string? workDirectory)
        {
            var sb = new StringBuilder();
            sb.Append("# TensorSharp shell wrapper (PowerShell).\n");
            sb.Append("$ErrorActionPreference = 'Continue'\n");
            // UTF-8 in both directions, so nothing above 0x7F is lost on the way out.
            //
            // The decode that mattered most was NOT here - it was the host reading the
            // pipe with the parent's OEM console encoding, and it is fixed in
            // SpawnedProcess. A native command's bytes reach the host untouched:
            // PowerShell passes its inherited handle straight through, verified with a
            // script that writes the three raw bytes of an arrow.
            //
            // What is left for this prologue is PowerShell's OWN output - Write-Output,
            // Get-ChildItem, an error record. Those it encodes with
            // [Console]::OutputEncoding, which defaults to the OEM code page, so without
            // the assignment below a cmdlet printing an accented filename would still
            // arrive mangled even though `python x.py` no longer does.
            //
            // chcp runs first because the assignment alone does not move the console code
            // page when stdout is a pipe (.NET calls SetConsoleOutputCP only for a real
            // console handle), and a native tool that asks the console which code page to
            // write reads THAT. It is best-effort: 5.1 snapshots its own native-command
            // decoder before this line can run, which is precisely why the host-side fix
            // is the one carrying the weight. $OutputEncoding covers what PowerShell
            // writes INTO a native command's stdin. All of it is a no-op on PowerShell 7.
            sb.Append("try { chcp 65001 > $null 2>&1 } catch { }\n");
            sb.Append("try { [Console]::OutputEncoding = New-Object Text.UTF8Encoding $false } catch { }\n");
            sb.Append("try { [Console]::InputEncoding = New-Object Text.UTF8Encoding $false } catch { }\n");
            sb.Append("try { $OutputEncoding = [Console]::OutputEncoding } catch { }\n");
            sb.Append("$__ts_state = ").Append(Pq(_workspace.ShellStateDirectory)).Append('\n');
            sb.Append("$__ts_work  = ").Append(Pq(_workspace.WorkDirectory)).Append('\n');
            sb.Append("$__ts_root  = ").Append(Pq(_workspace.Root)).Append('\n');
            sb.Append("$__ts_env   = ").Append(Pq(EnvFile)).Append('\n');
            sb.Append("$__ts_cwdf  = ").Append(Pq(CwdFile)).Append('\n');

            sb.Append("if (Test-Path -LiteralPath $__ts_env) {\n");
            sb.Append("  foreach ($__ts_line in (Get-Content -LiteralPath $__ts_env -ErrorAction SilentlyContinue)) {\n");
            sb.Append("    $__ts_i = $__ts_line.IndexOf('=')\n");
            sb.Append("    if ($__ts_i -gt 0) {\n");
            sb.Append("      $__ts_n = $__ts_line.Substring(0, $__ts_i)\n");
            sb.Append("      if ($__ts_n -notmatch ").Append(Pq(PowerShellEnvFilter)).Append(") {\n");
            sb.Append("        Set-Item -LiteralPath \"Env:$__ts_n\" -Value $__ts_line.Substring($__ts_i + 1) -ErrorAction SilentlyContinue\n");
            sb.Append("      }\n    }\n  }\n}\n");

            sb.Append("$__ts_cwd = $__ts_work\n");
            sb.Append("if (Test-Path -LiteralPath $__ts_cwdf) {\n");
            sb.Append("  $__ts_saved = (Get-Content -LiteralPath $__ts_cwdf -Raw -ErrorAction SilentlyContinue)\n");
            sb.Append("  if ($__ts_saved) { $__ts_saved = $__ts_saved.Trim() }\n");
            sb.Append("  if ($__ts_saved -and (Test-Path -LiteralPath $__ts_saved) -and\n");
            sb.Append("      ($__ts_saved -eq $__ts_root -or $__ts_saved.StartsWith($__ts_root + [IO.Path]::DirectorySeparatorChar))) {\n");
            sb.Append("    $__ts_cwd = $__ts_saved }\n");
            sb.Append("}\n");
            sb.Append("Set-Location -LiteralPath $__ts_cwd -ErrorAction SilentlyContinue\n");

            // Same rule as the POSIX wrapper: a per-call workdir moves this command only.
            sb.Append("$__ts_resume = (Get-Location).Path\n");
            if (workDirectory != null)
            {
                sb.Append("if (-not (Test-Path -LiteralPath ").Append(Pq(workDirectory)).Append(")) {\n");
                sb.Append("  Write-Error ").Append(Pq("workdir does not exist: "
                    + Path.GetRelativePath(_workspace.WorkDirectory, workDirectory).Replace('\\', '/'))).Append('\n');
                sb.Append("  exit 1\n}\n");
                sb.Append("Set-Location -LiteralPath ").Append(Pq(workDirectory)).Append('\n');
            }

            // PowerShell reports a native command's status in $LASTEXITCODE and a cmdlet's
            // failure as a terminating error only when asked. Both are folded into one exit
            // code here so the model reads the same "exit N" it would from any other shell.
            sb.Append("$global:LASTEXITCODE = 0\n");
            // The save lives in a `finally`, which is the only construct PowerShell runs
            // on an `exit` from inside the command — the POSIX wrapper uses an EXIT trap
            // for exactly the same reason. Without it, `Set-Location src; exit 0` reported
            // success and silently threw away the directory and environment it had just
            // been told would persist.
            sb.Append("try {\n");
            sb.Append("try {\n");
            sb.Append(CommandMarker).Append('\n');
            sb.Append(command);
            if (!command.EndsWith('\n'))
                sb.Append('\n');
            sb.Append("# ---- end command ----\n");
            sb.Append("} catch {\n");
            sb.Append("  Write-Error $_\n");
            sb.Append("  if (-not $LASTEXITCODE) { $global:LASTEXITCODE = 1 }\n");
            sb.Append("}\n");
            sb.Append("} finally {\n");

            sb.Append("$__ts_rc = $LASTEXITCODE\n");
            sb.Append("if ($null -eq $__ts_rc) { $__ts_rc = 0 }\n");
            sb.Append(workDirectory != null
                ? "try { $__ts_resume | Set-Content -LiteralPath $__ts_cwdf -NoNewline -ErrorAction SilentlyContinue } catch { }\n"
                : "try { $__ts_end = (Get-Location).Path\n"
                  + "  if (-not ($__ts_end -eq $__ts_root -or $__ts_end.StartsWith($__ts_root + [IO.Path]::DirectorySeparatorChar))) { $__ts_end = $__ts_work }\n"
                  + "  $__ts_end | Set-Content -LiteralPath $__ts_cwdf -NoNewline -ErrorAction SilentlyContinue } catch { }\n");
            // Values containing a newline cannot survive a line-per-variable file, and a
            // half-parsed value restored into the next call is worse than not restoring it.
            sb.Append("try { Get-ChildItem Env: -ErrorAction SilentlyContinue |\n");
            sb.Append("  Where-Object { $_.Value -notmatch \"[`r`n]\" -and $_.Name -notmatch ").Append(Pq(PowerShellEnvFilter)).Append(" } |\n");
            sb.Append("  ForEach-Object { \"$($_.Name)=$($_.Value)\" } |\n");
            sb.Append("  Set-Content -LiteralPath $__ts_env -ErrorAction SilentlyContinue } catch { }\n");
            sb.Append("}\n");
            sb.Append("exit $__ts_rc\n");
            return sb.ToString();
        }

        /// <summary>The PowerShell spelling of <see cref="PosixEnvFilter"/>. Same reasons.</summary>
        private const string PowerShellEnvFilter =
            "^(Path|PATHEXT|HOME|USERPROFILE|TEMP|TMP|PWD|PSModulePath|"
            + "COMSPEC|SYSTEMROOT|SYSTEMDRIVE|WINDIR|"
            + "HTTP_PROXY|HTTPS_PROXY|ALL_PROXY|NO_PROXY|"
            + "SSL_CERT_FILE|SSL_CERT_DIR|REQUESTS_CA_BUNDLE|CURL_CA_BUNDLE|"
            + "NODE_EXTRA_CA_CERTS|GIT_SSL_CAINFO|AWS_CA_BUNDLE|PIP_CERT|NPM_CONFIG_CAFILE|"
            + "CARGO_HTTP_CAINFO|DENO_CERT|CLOUDSDK_CORE_CUSTOM_CA_CERTS_FILE|"
            + "NIX_SSL_CERT_FILE|GRPC_DEFAULT_SSL_ROOTS_FILE_PATH|GRPC_PROXY|TS_.*)$";
    }
}
