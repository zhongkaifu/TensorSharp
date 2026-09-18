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
using System.Text;
using TensorSharp.AgentHost.Skills;

namespace TensorSharp.AgentHost.CodeExec
{
    /// <summary>
    /// Notice when a command re-typed a whole file to change a couple of lines of it, and
    /// say so with the numbers.
    ///
    /// <para>
    /// <b>The second-largest measured waste in this server's logs, after missing
    /// dependencies.</b> Across three days: 38 whole-program re-emissions in 26 clusters,
    /// <b>207,156 redundant characters — about 51,776 output tokens, 11% of everything
    /// the models produced, 46 minutes of pure decode</b>. One turn wrote the same
    /// JavaScript program seven times. Another rewrote a 190-line, 6,902-character
    /// heredoc whose entire difference from the previous version was ONE line:
    /// <c>pres.layout = {width: 10, height: 5.625}</c> became
    /// <c>pres.layout = 'LAYOUT_16x9'</c>. That is 1,725 output tokens and about 92
    /// seconds of decoding to change 39 characters.
    /// </para>
    /// <para>
    /// <b>And the patch tool was declared the whole time.</b> On the runs where both
    /// <c>shell</c> and <c>apply_patch</c> were offered, the model used
    /// <c>apply_patch</c> <b>zero</b> times and heredoc'd whole files ten times, six of
    /// them over a file it had written earlier in the same turn. So the tool description
    /// telling it to prefer a patch — which is there, and is emphatic — did not work.
    /// This codebase has already learned why: guidance lands when it is attached to the
    /// failing RESULT, not when it sits in a declaration the model read once.
    /// </para>
    /// <para>
    /// What makes this message different from the declaration's advice is that it is
    /// <b>specific and true about this file</b>: it names the file, counts the lines that
    /// were re-typed, counts the lines that actually differ, and does so immediately after
    /// the act. "You rewrote all 190 lines of deck.py and 1 line differs" is an argument;
    /// "prefer apply_patch" is a preference.
    /// </para>
    /// <para>
    /// <b>Cost.</b> The pre-run read happens only for a command that redirects into a
    /// file which already exists — no walk, no extra process, and nothing at all for the
    /// overwhelmingly common command that redirects nowhere. Bounded to a few files of
    /// modest size, because the point is to catch a re-typed program and a program is not
    /// megabytes.
    /// </para>
    /// </summary>
    public sealed class RewriteWatch
    {
        /// <summary>Files watched in one command. A command writing more is not re-typing one program.</summary>
        private const int MaxFiles = 8;

        /// <summary>Largest file worth remembering. A program the model typed is far under this.</summary>
        private const int MaxBytes = 512 * 1024;

        /// <summary>
        /// Shortest file worth mentioning. Re-typing twenty lines is not the waste this
        /// exists to name, and saying so would be nagging.
        /// </summary>
        private const int MinLines = 30;

        /// <summary>
        /// Most of the file that may differ before this is a genuine rewrite rather than a
        /// small edit expressed as one. A model that really did rewrite the program should
        /// not be told it should have patched it.
        ///
        /// <para>
        /// Raised from 0.25, because a RATIO was the wrong gate and let the measured waste
        /// through. A 40-line file re-typed to change 11 lines is 27.5% different and was
        /// therefore exempt — while 29 lines that were already correct had just been
        /// re-emitted and re-rolled. What is being counted is an absolute number of wasted
        /// lines, so <see cref="MinRetypedIdentical"/> is the trigger and this is only the
        /// suppressor: it exists to stay silent about a file that really was rewritten.
        /// </para>
        /// </summary>
        private const double MaxChangedFraction = 0.5;

        /// <summary>
        /// Lines re-typed byte-identically before this is worth saying anything about.
        ///
        /// <para>
        /// The honest unit. Every one of these is a line the model paid to produce, paid
        /// to have re-sampled, and could have left alone — the worst case in the logs
        /// re-typed 187 of 188 lines to change one, at 2,213 output tokens and 37.6
        /// seconds. Twenty-five is roughly where the note stops being nagging and starts
        /// being an argument.
        /// </para>
        /// </summary>
        private const int MinRetypedIdentical = 25;

        /// <summary>
        /// The RESOLVED absolute path is what is remembered, alongside the relative name to
        /// show the model. Holding only the relative name meant "before" and "after" could
        /// resolve against different directories — `cd sub &amp;&amp; cat > deck.py …` read
        /// `work/deck.py` going in and `work/sub/deck.py` coming out, then reported that
        /// one line of "it" had changed while describing two different files.
        /// </summary>
        private readonly List<(string Relative, string Full, string[] Lines)> _before;

        private RewriteWatch(List<(string, string, string[])> before) => _before = before;

        /// <summary>
        /// Remember the files <paramref name="command"/> is about to redirect into, or
        /// return null when it redirects into nothing that already exists.
        /// </summary>
        public static RewriteWatch? Before(string? command, SessionWorkspace workspace, string? from)
        {
            if (workspace == null)
                return null;

            string origin = from is { Length: > 0 } ? from : workspace.WorkDirectory;
            var kept = new List<(string, string, string[])>();
            foreach (SyntaxCheck.RedirectTarget target in SyntaxCheck.RedirectTargets(command))
            {
                if (kept.Count >= MaxFiles)
                    break;

                // An APPEND is not a rewrite, and calling one a rewrite is the host
                // stating a falsehood about what the model just did. `cat >> notes.py`
                // adding five lines to a 190-line file reads, to a line-count comparison,
                // exactly like a 195-line file with five lines changed — and the model
                // would be told its correct action had been wasteful.
                if (target.Appends)
                    continue;

                // Model-supplied path, resolved by the workspace: this read runs in the
                // HOST, which is not sandboxed, so a symlink planted in the workspace must
                // not be followed out of it.
                if (!workspace.TryResolveFrom(origin, target.Path, out string full, out _))
                    continue;
                try
                {
                    var info = new FileInfo(full);
                    if (!info.Exists || info.Length == 0 || info.Length > MaxBytes)
                        continue;
                    string[] lines = File.ReadAllLines(full);
                    if (lines.Length >= MinLines && !LooksBinary(lines))
                        kept.Add((target.Path, full, lines));
                }
                catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or ArgumentException)
                {
                    // A file that cannot be read cannot be compared, and inventing a
                    // comparison would be a false statement about the model's own file.
                }
            }

            return kept.Count == 0 ? null : new RewriteWatch(kept);
        }

        /// <summary>
        /// Describe any watched file that was re-typed in full to change a little of it,
        /// or return null.
        /// </summary>
        public string? Describe(SessionWorkspace workspace, string patchToolName)
        {
            ArgumentNullException.ThrowIfNull(workspace);

            var sb = new StringBuilder();
            foreach ((string relative, string full, string[] before) in _before)
            {
                string[] after;
                try
                {
                    if (!File.Exists(full))
                        continue;
                    after = File.ReadAllLines(full);
                }
                catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
                {
                    continue;
                }

                if (after.Length < MinLines || LooksBinary(after))
                    continue;

                if (Compare(relative, before, after, patchToolName) is { Length: > 0 } note)
                    sb.Append('\n').Append(note);
            }

            return sb.Length == 0 ? null : sb.ToString();
        }

        /// <summary>
        /// Compare two versions of a file's text and say whether one was re-typed to
        /// change a little of the other — the whole comparison, with no filesystem and no
        /// command line anywhere near it.
        ///
        /// <para>
        /// Public and static because the command-line scan this class was built around
        /// cannot see most of the ways a file gets rewritten. <c>cp</c>, <c>mv</c>,
        /// <c>sed -i</c>, a Python <c>open().write()</c>, a redirect through a shell
        /// variable, a background job, and <c>apply_patch</c>'s own whole-file
        /// <c>*** Update File:</c> section were all invisible to it — the last one being
        /// the embarrassing case, where the mechanism built to discourage rewriting gave
        /// no pushback at all to a rewrite performed through the patch tool. Anything
        /// holding a before and an after can ask this directly.
        /// </para>
        /// </summary>
        /// <param name="path">The file's name, as the model spelled it.</param>
        /// <param name="before">What it held.</param>
        /// <param name="after">What it holds now.</param>
        /// <param name="editToolName">The patch tool to point at instead.</param>
        /// <returns>The note, or null when this was not a re-typing.</returns>
        public static string? DescribeRetyped(string path, string? before, string? after, string editToolName)
        {
            if (string.IsNullOrEmpty(before) || string.IsNullOrEmpty(after))
                return null;

            string[] beforeLines = SplitLines(before!);
            string[] afterLines = SplitLines(after!);
            return Compare(path, beforeLines, afterLines, editToolName);
        }

        /// <summary>
        /// Split file text the way <see cref="File.ReadAllLines(string)"/> does.
        ///
        /// <para>
        /// The trailing element from splitting <c>"a\nb\n"</c> is a POSITION, not a line,
        /// and it is not a cosmetic difference here: counting it reports a 60-line file as
        /// 61 lines to the model, and — because the empty string is present in both
        /// versions — adds a phantom identical line to the count this whole note is an
        /// argument about. The other entry point reads through
        /// <see cref="File.ReadAllLines(string)"/> and never saw it.
        /// </para>
        /// </summary>
        private static string[] SplitLines(string text)
        {
            string[] lines = text.Replace("\r\n", "\n", StringComparison.Ordinal).Split('\n');
            return lines.Length > 0 && lines[^1].Length == 0
                ? lines[..^1]
                : lines;
        }

        /// <summary>
        /// The judgement and the message, shared by both entry points.
        ///
        /// <para>
        /// The gate is an ABSOLUTE count of lines re-typed byte-identically, not a ratio
        /// of lines changed. A ratio exempted exactly the cases worth naming: a short file
        /// re-typed in full sails under a line-count floor, and a 40-line file re-typed to
        /// change 11 sails under a 25%-changed ceiling while 29 correct lines are thrown
        /// away and re-sampled. What costs tokens is the identical lines, so that is what
        /// is counted, and the ratio survives only as a SUPPRESSOR — a file that really
        /// was rewritten must not be told it should have edited.
        /// </para>
        /// </summary>
        private static string? Compare(string path, string[] before, string[] after, string patchToolName)
        {
            if (before.Length < MinLines && after.Length < MinLines)
                return null;
            if (LooksBinary(before) || LooksBinary(after))
                return null;

            (int changed, IReadOnlyList<string> removed, IReadOnlyList<string> added, int identical) =
                Difference(before, after);

            if (changed == 0)
                return null;                                        // nothing was changed at all
            if (identical < MinRetypedIdentical)
                return null;                                        // too little re-typed to be worth saying
            if (changed > after.Length * MaxChangedFraction)
                return null;                                        // a real rewrite, not a small edit typed long

            var sb = new StringBuilder();
            sb.Append(path).Append(" already existed, and this replaced all ")
              .Append(after.Length.ToString(CultureInfo.InvariantCulture))
              .Append(" lines of it — but only ")
              .Append(changed == 1
                  ? "1 line is different"
                  : changed.ToString(CultureInfo.InvariantCulture) + " lines are different")
              .Append(" from what was there. ")
              .Append(identical.ToString(CultureInfo.InvariantCulture))
              .Append(" lines came back exactly as they already were:\n");

            // The lines themselves, not just the count. Telling a model "send those 3
            // lines to the editor" without saying WHICH asks it to reconstruct them from
            // memory — the failure the editor exists to avoid, so the advice would cause
            // the thing it warns about.
            foreach (string line in removed.Take(MaxShownLines))
                sb.Append("  - ").Append(Clip(line)).Append('\n');
            foreach (string line in added.Take(MaxShownLines))
                sb.Append("  + ").Append(Clip(line)).Append('\n');
            if (removed.Count > MaxShownLines || added.Count > MaxShownLines)
                sb.Append("  … and more\n");

            sb.Append("Re-typing a file costs you every line that was already correct, and re-rolls "
                    + "each of them, which is how a second bug appears in code that worked. ");

            // Difference is an unordered, potentially clipped summary. It cannot supply
            // the ordered context required by a patch, and the change has already happened.
            sb.Append("For the next modification, use ").Append(patchToolName)
              .Append(" for either a single-file or multi-file change. Set its patch argument to a "
                    + "*** Begin Patch / *** End Patch envelope with a *** Update File: <path> section "
                    + "per existing file and @@ hunks containing exact current context (space prefix), "
                    + "removed lines (-), and replacement lines (+). Read the relevant current region "
                    + "if needed. Multiple hunks and file sections can share one patch. The excerpts "
                    + "above summarize the completed change; they are not a patch to apply now.\n");

            return sb.ToString();
        }

        /// <summary>
        /// How many lines differ, by multiset: lines present in one version and not the
        /// other, counted once each.
        ///
        /// <para>
        /// Not an alignment diff, and deliberately so. The question here is "was this a
        /// small change typed the long way", which a multiset answers exactly as well as
        /// an LCS for a file the model just re-emitted, at a fraction of the work — and
        /// this runs after a command that already succeeded, so it must not be the
        /// expensive part of the call. It is honest in both directions: a pure reordering
        /// reports 0 differing lines, which is the truth about the LINES, and the message
        /// only ever claims that.
        /// </para>
        /// </summary>
        private static (int Changed, IReadOnlyList<string> Removed, IReadOnlyList<string> Added, int Identical) Difference(
            string[] before, string[] after)
        {
            var counts = new Dictionary<string, int>(before.Length, StringComparer.Ordinal);
            foreach (string line in before)
                counts[line] = counts.TryGetValue(line, out int n) ? n + 1 : 1;

            // Lines that came back byte-identical: the wasted ones, and the only honest
            // measure of what the re-typing cost.
            int identical = 0;

            var added = new List<string>();
            foreach (string line in after)
            {
                if (counts.TryGetValue(line, out int n) && n > 0)
                {
                    counts[line] = n - 1;
                    identical++;
                }
                else if (added.Count <= MaxShownLines)
                    added.Add(line);
                else
                    added.Add(string.Empty);   // counted, not shown
            }

            var removed = new List<string>();
            foreach (KeyValuePair<string, int> pair in counts)
            {
                for (int i = 0; i < pair.Value && removed.Count <= MaxShownLines; i++)
                    removed.Add(pair.Key);
            }
            int removedCount = counts.Values.Sum();

            return (Math.Max(added.Count, removedCount), removed, added.Where(l => l.Length > 0).ToList(), identical);
        }

        /// <summary>Lines of the difference actually printed. Three is enough to make the point.</summary>
        private const int MaxShownLines = 3;

        /// <summary>Longest line echoed back. A minified bundle is one line of half a megabyte.</summary>
        private const int MaxLineChars = 160;

        private static string Clip(string line) =>
            line.Length <= MaxLineChars ? line : line.Substring(0, MaxLineChars) + " …";

        /// <summary>
        /// A NUL byte anywhere means this is not text, whatever its extension.
        ///
        /// <para>
        /// <see cref="File.ReadAllLines(string)"/> decodes UTF-16-without-a-BOM as Latin-1
        /// and yields lines laced with NULs, so both the line COUNT and the lines
        /// themselves would be nonsense — and the message states both to the model as
        /// facts about its own file.
        /// </para>
        /// </summary>
        private static bool LooksBinary(string[] lines)
        {
            int scanned = 0;
            foreach (string line in lines)
            {
                if (line.Contains('\0', StringComparison.Ordinal))
                    return true;
                if ((scanned += line.Length + 1) > 8192)
                    break;
            }
            return false;
        }
    }
}
