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
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp.Runtime.Logging;
using TensorSharp.AgentHost.Skills;

namespace TensorSharp.AgentHost.CodeExec
{
    /// <summary>One file a run produced and kept.</summary>
    /// <param name="RunId">The run that made it. Also the directory it is kept in.</param>
    /// <param name="Path">Its path relative to the run's working directory.</param>
    /// <param name="Bytes">Its size.</param>
    /// <param name="Pointer">Where a user can get it: a URL on a server, a file path on the CLI.</param>
    public readonly record struct CodeArtifact(string RunId, string Path, long Bytes, string Pointer);

    /// <summary>
    /// Keeps the files a run produced after the run's environment is destroyed.
    ///
    /// <para>
    /// A program that writes a PDF has produced the answer, not a side effect — deleting
    /// it with the scratch directory would throw away the very thing the user asked for.
    /// So the two lifetimes are separated: the <b>environment</b> (interpreter, installed
    /// packages, the source file) dies with the call as it must, and the <b>artifacts</b>
    /// are copied out first into a store that outlives it.
    /// </para>
    /// <para>
    /// The store is bounded in every direction, because it is filled by programs a model
    /// wrote: a per-file cap, a per-run cap, a file-count cap, and a total budget with
    /// oldest-first eviction. Without those, one loop writing zeros fills the disk. It is
    /// also flat by construction — every file is re-rooted under a per-run directory and
    /// its name is sanitised — so nothing a program chooses to call its output can place a
    /// byte outside the store.
    /// </para>
    /// </summary>
    public sealed class CodeArtifactStore
    {
        private readonly string _root;
        private readonly CodeArtifactLimits _limits;
        private readonly ILogger _logger;
        private readonly object _gate = new();

        public CodeArtifactStore(string root, CodeArtifactLimits? limits = null, ILogger? logger = null)
        {
            _root = root ?? throw new ArgumentNullException(nameof(root));
            _limits = limits ?? new CodeArtifactLimits();
            _logger = logger ?? NullLogger.Instance;
        }

        /// <summary>Where artifacts are kept.</summary>
        public string Root => _root;

        /// <summary>
        /// Copy everything <paramref name="workDirectory"/> holds into the store under
        /// <paramref name="runId"/>, and say what was kept and what was not.
        /// </summary>
        /// <param name="pointerFor">
        /// Turns a run id and a relative path into the thing a user acts on. A server
        /// hands back a URL; the CLI hands back the absolute path on disk.
        /// </param>
        /// <param name="exclude">
        /// Files not to keep, by working-directory-relative path — the caller's staged
        /// input files, which are the user's own uploads rather than anything the run
        /// produced. Null keeps everything.
        /// </param>
        public IReadOnlyList<CodeArtifact> Capture(
            string runId,
            string workDirectory,
            Func<string, string, string, string> pointerFor,
            out IReadOnlyList<string> skipped,
            Func<string, bool>? exclude = null)
        {
            var kept = new List<CodeArtifact>();
            var rejected = new List<string>();
            skipped = rejected;

            string[] files;
            try
            {
                // ReparsePoint is skipped, not merely unfollowed: enumeration would
                // otherwise walk THROUGH a directory symlink and File.Copy would read
                // through a file one, so `ln -s ~ h` inside the workspace would hand the
                // user download links to the host's home directory. The copy is done by
                // the host process, which no sandbox confines.
                var options = new EnumerationOptions
                {
                    RecurseSubdirectories = true,
                    AttributesToSkip = FileAttributes.Hidden | FileAttributes.System | FileAttributes.ReparsePoint,
                    IgnoreInaccessible = true,
                };

                // Pruned DURING the walk, not filtered after it. IsRuntimeJunk below still
                // rejects the same paths, but by then every file under node_modules has
                // already been stat'd and sorted — 106 ms per command on a 20k-file
                // install, paid by `echo hi` as much as by a build. WorkspaceScan owns the
                // list of directories not to open, and the pre-command snapshot in
                // SessionWorkspace prunes by the same one: a snapshot that descended where
                // this does not would mark everything under it as newly produced.
                files = WorkspaceScan.Files(workDirectory, options)
                    .Where(NotALink)
                    .OrderBy(f => f, StringComparer.Ordinal)
                    .ToArray();
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                // Said out loud. Returning an empty list here was indistinguishable from
                // "this command produced no files", so a model that had just written the
                // PDF told the user it was ready and there was no link in the result and no
                // file card in the UI — a missing deliverable with nothing anywhere to
                // explain it.
                rejected.Add(
                    $"the working directory could not be scanned ({ex.Message}), so no files were kept "
                    + "from this command even if it wrote some");
                return kept;
            }

            if (files.Length == 0)
                return kept;

            string runRoot = Path.Combine(_root, runId);
            long runTotal = 0;

            int index = -1;
            foreach (string file in files)
            {
                index++;
                if (kept.Count >= _limits.MaxFilesPerRun)
                {
                    // Counted over what would ACTUALLY have been kept, not over the whole
                    // walk. `files` still holds every pre-existing and every junk path, so
                    // subtracting was reporting "201 more file(s)" when one new file had
                    // been dropped — a number that sends the model looking for two hundred
                    // outputs it never made.
                    int remaining = 0;
                    foreach (string later in files.Skip(index))
                    {
                        string laterRelative = Path.GetRelativePath(workDirectory, later).Replace('\\', '/');
                        if (!IsRuntimeJunk(laterRelative) && exclude?.Invoke(laterRelative) != true)
                            remaining++;
                    }
                    if (remaining > 0)
                    {
                        rejected.Add(
                            $"{remaining} more produced file(s) (at most {_limits.MaxFilesPerRun} are kept per run)");
                    }
                    break;
                }

                FileInfo info;
                try { info = new FileInfo(file); }
                catch (Exception) { continue; }

                string relative = Path.GetRelativePath(workDirectory, file);

                // The runtime's own droppings — the seatbelt profile a wrapper writes
                // into the work directory — are never anyone's output.
                if (Path.GetFileName(file).StartsWith(".tensorsharp-", StringComparison.Ordinal))
                    continue;

                // Interpreter and desktop-app fallout is not output either. HOME points
                // at the work directory (deliberately), so Apple's Python drops its
                // bytecode cache under Library/Caches and LibreOffice writes .config —
                // and change-based capture would faithfully present sixteen .pyc files
                // as "files produced". Filter the well-known junk by shape.
                if (IsRuntimeJunk(relative))
                    continue;

                if (exclude != null && exclude(relative))
                    continue;

                if (info.Length > _limits.MaxFileBytes)
                {
                    rejected.Add($"{relative} ({Format(info.Length)}, over the {Format(_limits.MaxFileBytes)} per-file limit)");
                    continue;
                }
                if (runTotal + info.Length > _limits.MaxRunBytes)
                {
                    rejected.Add($"{relative} (this run's output passed the {Format(_limits.MaxRunBytes)} limit)");
                    continue;
                }

                // Re-root under the run directory and re-check: a name is chosen by the
                // program, and the point of a confinement is that it does not depend on
                // the confined thing behaving.
                string destination = Path.Combine(runRoot, relative);
                if (!SkillPathGuard.IsUnder(runRoot, destination))
                {
                    rejected.Add($"{relative} (its name would place it outside the artifact store)");
                    continue;
                }

                try
                {
                    Directory.CreateDirectory(Path.GetDirectoryName(destination)!);
                    File.Copy(file, destination, overwrite: true);
                }
                catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
                {
                    rejected.Add($"{relative} ({ex.Message})");
                    continue;
                }

                runTotal += info.Length;
                kept.Add(new CodeArtifact(runId, relative.Replace('\\', '/'), info.Length,
                    pointerFor(runId, relative.Replace('\\', '/'), destination)));
            }

            if (kept.Count > 0)
            {
                _logger.LogInformation(LogEventIds.SkillScriptExecuted,
                    "codeexec.artifacts run={RunId} files={Count} bytes={Bytes}", LogContent.Of(runId), kept.Count, runTotal);
                EvictIfOverBudget(runTotal);
            }

            return kept;
        }

        /// <summary>
        /// Belt and braces on the enumeration's own filter: a broken link, or one the
        /// enumerator's attribute check treats differently on this platform, must not
        /// reach File.Copy. A path that cannot be inspected is not copied either.
        /// </summary>
        private static bool NotALink(string path)
        {
            try { return (File.GetAttributes(path) & FileAttributes.ReparsePoint) == 0; }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { return false; }
        }

        /// <summary>
        /// True for files a RUNTIME left in the working directory rather than the
        /// program's own output: bytecode caches, and the config/cache trees that
        /// HOME-redirected tools (LibreOffice, fontconfig, pip) grow on first use.
        /// </summary>
        internal static bool IsRuntimeJunk(string relative)
        {
            string normalized = relative.Replace('\\', '/');

            if (normalized.EndsWith(".pyc", StringComparison.OrdinalIgnoreCase))
                return true;

            // LibreOffice writes its session marker and a MAT debug log into the working
            // directory on every headless conversion, so a run that produced one deck
            // offered the user four downloads. Same lesson as the .pyc filter above: a
            // tool's own scratch is not the user's output.
            string name = normalized.Substring(normalized.LastIndexOf('/') + 1);
            if (string.Equals(name, ".ses", StringComparison.Ordinal)
                || (name.StartsWith("mat-debug-", StringComparison.Ordinal)
                    && name.EndsWith(".log", StringComparison.OrdinalIgnoreCase)))
            {
                return true;
            }

            // Every directory the path passes through, against the one list
            // WorkspaceScan prunes the walk by — so a caller that reaches this with a
            // path the pruned walk would never have produced (a skill script listing what
            // it made) answers the same question the same way.
            //
            // A shell reaches far more tools than a single interpreter did, so that list
            // is expected to grow rather than shrink: everything on it is a directory some
            // program creates on first use, in $HOME (which points at the work directory
            // on purpose) or beside the code it is building. Plus `.jobs`, which is this
            // host's own — a background job's log lives in the work directory so the model
            // can `cat` it, and a log the model is reading is not a file the user asked for.
            int start = 0;
            int slash;
            while ((slash = normalized.IndexOf('/', start)) > 0)
            {
                if (WorkspaceScan.IsPrunedDirectory(
                        normalized.AsSpan(start, slash - start), atRoot: start == 0))
                {
                    return true;
                }
                start = slash + 1;
            }

            return false;
        }

        /// <summary>
        /// The download URL for one artifact, built the SAME way wherever a pointer is
        /// made.
        ///
        /// <para>
        /// Each path SEGMENT is escaped and the separators are not, because
        /// <see cref="Uri.EscapeDataString"/> escapes <c>/</c> itself: escaping the whole
        /// relative path turned <c>out/report.pdf</c> into <c>out%2Freport.pdf</c>, which
        /// this store's own route then refuses. The two producers -- a shell command's
        /// captured output and a skill script's -- built this string separately and only
        /// one of them escaped anything, so a file with a space, a <c>#</c> or a <c>%</c>
        /// in its name was reachable from one of them and a 404 from the other.
        /// </para>
        /// </summary>
        public static string UrlFor(string uriPrefix, string runId, string relativePath)
        {
            ArgumentNullException.ThrowIfNull(uriPrefix);
            string relative = (relativePath ?? string.Empty).Replace('\\', '/');
            var sb = new System.Text.StringBuilder(uriPrefix.TrimEnd('/'));
            sb.Append('/').Append(Uri.EscapeDataString(runId ?? string.Empty));
            foreach (string segment in relative.Split('/'))
            {
                if (segment.Length == 0)
                    continue;
                sb.Append('/').Append(Uri.EscapeDataString(segment));
            }
            return sb.ToString();
        }

        /// <summary>Resolve one artifact for download, refusing anything outside the store.</summary>
        public bool TryResolve(string runId, string relativePath, out string? fullPath, out string? error)
        {
            fullPath = null;
            error = null;

            if (string.IsNullOrWhiteSpace(runId) || !IsSafeRunId(runId))
            {
                error = "unknown run";
                return false;
            }

            string runRoot = Path.Combine(_root, runId);
            if (!SkillPathGuard.TryResolveExistingFile(runRoot, relativePath, out string? resolved, out string? guardError))
            {
                error = guardError ?? "not found";
                return false;
            }

            fullPath = resolved;
            return true;
        }

        /// <summary>Everything currently held for one run.</summary>
        public IReadOnlyList<CodeArtifact> List(string runId, Func<string, string, string, string> pointerFor)
        {
            if (!IsSafeRunId(runId))
                return Array.Empty<CodeArtifact>();

            string runRoot = Path.Combine(_root, runId);
            if (!Directory.Exists(runRoot))
                return Array.Empty<CodeArtifact>();

            try
            {
                return Directory.EnumerateFiles(runRoot, "*", SearchOption.AllDirectories)
                    .Select(f =>
                    {
                        string rel = Path.GetRelativePath(runRoot, f).Replace('\\', '/');
                        return new CodeArtifact(runId, rel, new FileInfo(f).Length, pointerFor(runId, rel, f));
                    })
                    .OrderBy(a => a.Path, StringComparer.Ordinal)
                    .ToList();
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                return Array.Empty<CodeArtifact>();
            }
        }

        /// <summary>A run id we generated: hex, fixed length, no separators.</summary>
        private static bool IsSafeRunId(string runId) =>
            runId.Length is > 0 and <= 64 && runId.All(c => char.IsAsciiLetterOrDigit(c));

        /// <summary>
        /// What the store held after the last time it was counted, or -1 when that is
        /// not known — at startup, and after anything deleted a run behind our back.
        ///
        /// <para>
        /// Kept because the count is not cheap: it walks every run directory, and it was
        /// being done on EVERY command that produced a file. Measured against a store
        /// holding 500 runs of 20 files, capturing one output file cost 27.5 ms, of which
        /// 26.7 ms was re-measuring artifacts nobody had touched — and it grows with the
        /// store, which is bounded in bytes rather than in files.
        /// </para>
        /// </summary>
        private long _totalBytes = -1;

        /// <summary>
        /// Keep the store under its total budget by deleting the oldest runs first.
        /// </summary>
        /// <param name="added">Bytes this run just wrote, to carry the running total forward.</param>
        private void EvictIfOverBudget(long added)
        {
            lock (_gate)
            {
                // The common case is a store well under budget, and that case is answered
                // by arithmetic. Only crossing the line — or not knowing where the line is
                // yet — buys a walk, and the walk leaves the total known again. The
                // running total can only ever be an OVER-estimate (a run deleted elsewhere
                // is not subtracted), which costs one unnecessary recount and never a
                // missed eviction.
                if (_totalBytes >= 0)
                {
                    _totalBytes += added;
                    if (_totalBytes <= _limits.MaxTotalBytes)
                        return;
                }

                List<DirectoryInfo> runs = EnumerateRuns().OrderBy(d => d.LastWriteTimeUtc).ToList();
                long total = runs.Sum(SizeOf);

                foreach (DirectoryInfo dir in runs)
                {
                    if (total <= _limits.MaxTotalBytes)
                        break;
                    total -= SizeOf(dir);
                    TryDelete(dir);
                }

                _totalBytes = total;
            }
        }

        private IEnumerable<DirectoryInfo> EnumerateRuns()
        {
            DirectoryInfo root;
            try
            {
                root = new DirectoryInfo(_root);
                if (!root.Exists)
                    return Array.Empty<DirectoryInfo>();
                return root.EnumerateDirectories().ToList();
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                return Array.Empty<DirectoryInfo>();
            }
        }

        private static long SizeOf(DirectoryInfo dir)
        {
            try { return dir.EnumerateFiles("*", SearchOption.AllDirectories).Sum(f => f.Length); }
            catch (Exception) { return 0; }
        }

        private void TryDelete(DirectoryInfo dir)
        {
            try { dir.Delete(recursive: true); }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                _logger.LogWarning(LogEventIds.SkillScriptExecuted,
                    "codeexec.artifact-evict-failed dir={Dir} reason={Reason}", LogContent.Of(dir.FullName), LogContent.Of(ex.Message));
            }
        }

        private static string Format(long bytes) => bytes switch
        {
            < 1024 => bytes.ToString(CultureInfo.InvariantCulture) + " B",
            < 1024 * 1024 => (bytes / 1024.0).ToString("0.#", CultureInfo.InvariantCulture) + " KB",
            _ => (bytes / (1024.0 * 1024)).ToString("0.#", CultureInfo.InvariantCulture) + " MB",
        };
    }

    /// <summary>How much a model's programs may leave behind.</summary>
    public sealed class CodeArtifactLimits
    {
        /// <summary>Largest single file kept.</summary>
        public long MaxFileBytes { get; init; } = 32L * 1024 * 1024;

        /// <summary>Most one run may leave behind in total.</summary>
        public long MaxRunBytes { get; init; } = 64L * 1024 * 1024;

        /// <summary>Most files one run may leave behind.</summary>
        public int MaxFilesPerRun { get; init; } = 32;

        /// <summary>Budget for the whole store; the oldest runs are evicted past it.</summary>
        public long MaxTotalBytes { get; init; } = 512L * 1024 * 1024;

        /// <summary>How long an artifact stays downloadable. Zero disables expiry.</summary>
        public TimeSpan Retention { get; init; } = TimeSpan.FromHours(6);
    }
}
