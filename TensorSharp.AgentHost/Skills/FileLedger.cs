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

namespace TensorSharp.AgentHost.Skills
{
    /// <summary>
    /// What the model has actually been shown of each file, so an edit can be checked
    /// against bytes it has seen rather than bytes it half-remembers.
    ///
    /// <para>
    /// <b>This is the mechanic the measurement said was missing.</b> Across 480 tool calls
    /// in this server's logs, 11 read file content at all, and 4 of the 5 files rewritten
    /// by a heredoc had never been read before being rewritten. Both reference
    /// implementations depend on the model holding a file's exact bytes before it edits
    /// one — Claude Code enforces it (an <c>Edit</c> is refused unless the file was
    /// <c>Read</c> first in the same conversation), and that invariant is the only reason
    /// its <c>old_string</c> is a COPY rather than a recollection. A model editing from
    /// memory writes anchors that do not match, and a model whose anchors do not match
    /// rewrites the file — which is exactly the behaviour being fixed.
    /// </para>
    /// <para>
    /// <b>Freshness is computed by comparing CONTENT, never by trusting a flag.</b> The
    /// ledger stores a hash of what it showed; the caller passes what is on disk now; the
    /// answer falls out of the comparison. That choice is what makes this correct where a
    /// command-line scanner cannot be: <c>cp</c>, <c>mv</c>, <c>sed -i</c>, a Python
    /// <c>open().write()</c>, a redirect through a shell variable, a background job that
    /// finished between calls, and every write path nobody has thought of yet are all seen,
    /// because the question is asked of the filesystem instead of of the command string.
    /// There is nothing to invalidate and therefore nothing that can be forgotten to
    /// invalidate — the <see cref="RewriteWatch"/> docstring's six escape routes exist
    /// precisely because that class has to enumerate write paths, and this one does not.
    /// </para>
    /// <para>
    /// <b>Writing counts as reading.</b> A file the model just created with a heredoc, a
    /// <c>write_file</c>, or a patch is a file whose bytes are in its context by
    /// construction, so it is recorded rather than left unread. Without that, the very
    /// first edit after creating a file would be gated on re-reading a file the model
    /// itself had just typed — a round spent proving something already true.
    /// </para>
    /// <para>
    /// <b>Why it lives beside the workspace and not beside the editor.</b> It is state
    /// ABOUT a session's files, with exactly the workspace's lifetime, and it has no
    /// dependency on anything that runs code. Putting it in the execution namespace would
    /// have made the skills namespace reference that one — reversing the direction
    /// <see cref="ICodeRunner"/> exists to keep one-way, and dragging the whole execution
    /// stack into every host that only ever wanted skills.
    /// </para>
    /// <para>
    /// <b>What is deliberately NOT here:</b> nothing parses <c>cat</c> or <c>sed -n</c> out
    /// of a shell command to claim the model has read something. The host would then be
    /// asserting the model attended to bytes that may have been truncated by the output
    /// cap or scrolled past unread, and a false "you have seen this" authorises exactly
    /// the blind edit the ledger exists to prevent. A model that keeps using <c>cat</c>
    /// simply keeps the <c>Unread</c> path, which applies anyway when the anchor is
    /// unambiguous.
    /// </para>
    /// </summary>
    public sealed class FileLedger
    {
        /// <summary>
        /// Paths remembered at once, LRU-evicted.
        ///
        /// <para>
        /// Deliberately not the clear-everything policy the repeated-command ledger uses.
        /// That one catches a loop, which is always recent; this one AUTHORISES an edit,
        /// and a false "unread" costs a round. Eviction degrades to
        /// <see cref="ReadFreshness.Unread"/>, which is the safe direction: the edit is
        /// still applied when its anchor is unambiguous, with a note.
        /// </para>
        /// </summary>
        private const int MaxTracked = 256;

        /// <summary>
        /// Largest file whose text is kept for comparison. Above this only the hash is
        /// held: freshness still works, and what is lost is the rewrite comparison, which
        /// was never going to be about a half-megabyte file.
        /// </summary>
        private const int MaxRememberedChars = 256 * 1024;

        /// <summary>
        /// Total text held across every tracked path.
        ///
        /// <para>
        /// A per-entry cap alone bounds nothing that matters: 256 paths at a quarter of a
        /// megabyte each is 64 MB of UTF-16 per SESSION, pinned for the session's whole
        /// lifetime, on a server that may hold many at once. The oldest entries give up
        /// their text — not their existence — until the total fits, so freshness still
        /// works for every tracked path and only the rewrite comparison degrades, which is
        /// the cheapest thing here to lose.
        /// </para>
        /// </summary>
        private const long MaxRememberedTotalChars = 8L * 1024 * 1024;

        private long _remembered;

        private readonly object _gate = new();
        private readonly Dictionary<string, Entry> _entries = new(StringComparer.Ordinal);
        private long _clock;

        private sealed class Entry
        {
            public ulong Hash;
            public int FirstLine;
            public int LastLine;
            public bool Complete;
            public string? Text;
            public long Touched;
        }

        /// <summary>What the model has seen of one file, relative to what is on disk now.</summary>
        public readonly record struct ReadState(
            ReadFreshness Freshness, int FirstLine, int LastLine, bool Complete, string? KnownText)
        {
            /// <summary>True when a span of lines lies inside what was actually shown.</summary>
            public bool Covers(int firstLine, int lastLine) =>
                Complete || (firstLine >= FirstLine && lastLine <= LastLine);
        }

        /// <summary>
        /// Remember that the model has been shown <paramref name="content"/> of
        /// <paramref name="fullPath"/> — from a read, or from having written it.
        /// </summary>
        /// <param name="fullPath">The RESOLVED absolute path, never a relative spelling.</param>
        /// <param name="content">The file's full text as it now stands on disk.</param>
        /// <param name="firstLine">1-based first line shown, for a partial read.</param>
        /// <param name="lastLine">1-based last line shown, for a partial read.</param>
        /// <param name="complete">Whether the whole file was shown.</param>
        public void Record(string fullPath, string content, int firstLine, int lastLine, bool complete)
        {
            if (string.IsNullOrEmpty(fullPath) || content == null)
                return;

            lock (_gate)
            {
                if (!_entries.TryGetValue(fullPath, out Entry? entry))
                {
                    entry = new Entry();
                    _entries[fullPath] = entry;
                }

                entry.Hash = Hash(content);
                _remembered -= entry.Text?.Length ?? 0;
                entry.Text = content.Length <= MaxRememberedChars ? content : null;
                _remembered += entry.Text?.Length ?? 0;
                entry.Touched = ++_clock;

                // A wider view never narrows. Reading lines 1-40 and then 30-80 has shown
                // the model 1-80, and telling it otherwise would refuse an edit it has
                // every byte for. A COMPLETE read subsumes everything.
                if (complete || entry.Complete)
                {
                    entry.Complete = true;
                    entry.FirstLine = 1;
                    entry.LastLine = int.MaxValue;
                }
                else if (entry.LastLine == 0)
                {
                    entry.FirstLine = firstLine;
                    entry.LastLine = lastLine;
                }
                else if (firstLine <= entry.LastLine + 1 && lastLine >= entry.FirstLine - 1)
                {
                    // Overlapping or adjoining: the union is contiguous and honest.
                    entry.FirstLine = Math.Min(entry.FirstLine, firstLine);
                    entry.LastLine = Math.Max(entry.LastLine, lastLine);
                }
                else
                {
                    // A disjoint second window. Claiming the union would claim the gap
                    // between them, which the model has not seen, so the newer window
                    // simply replaces the older one.
                    entry.FirstLine = firstLine;
                    entry.LastLine = lastLine;
                }

                Evict();
                Trim();
            }
        }

        /// <summary>
        /// How what the model has seen of <paramref name="fullPath"/> compares with
        /// <paramref name="currentContent"/>, which the caller has just read from disk.
        /// </summary>
        public ReadState Check(string fullPath, string currentContent)
        {
            if (string.IsNullOrEmpty(fullPath))
                return new ReadState(ReadFreshness.Unread, 0, 0, false, null);

            lock (_gate)
            {
                if (!_entries.TryGetValue(fullPath, out Entry? entry))
                    return new ReadState(ReadFreshness.Unread, 0, 0, false, null);

                entry.Touched = ++_clock;

                if (entry.Hash != Hash(currentContent ?? string.Empty))
                {
                    // The bytes moved under the model. The old text is still handed back:
                    // it is what tells a caller whether this was a small edit typed the
                    // long way, and it is the only copy of the pre-change file anyone has.
                    return new ReadState(
                        ReadFreshness.Stale, entry.FirstLine, entry.LastLine, entry.Complete, entry.Text);
                }

                return new ReadState(
                    entry.Complete ? ReadFreshness.Fresh : ReadFreshness.Partial,
                    entry.FirstLine, entry.LastLine, entry.Complete, entry.Text);
            }
        }

        /// <summary>The text last recorded for a path, when it is still held.</summary>
        public bool TryGetKnownText(string fullPath, out string text)
        {
            text = string.Empty;
            if (string.IsNullOrEmpty(fullPath))
                return false;

            lock (_gate)
            {
                if (_entries.TryGetValue(fullPath, out Entry? entry) && entry.Text != null)
                {
                    text = entry.Text;
                    return true;
                }
            }
            return false;
        }

        /// <summary>
        /// Drop a path entirely — for a file replaced behind the model's back by the HOST
        /// rather than by anything the model did, where even "stale" would overstate what
        /// it knows.
        /// </summary>
        public void Forget(string fullPath)
        {
            if (string.IsNullOrEmpty(fullPath))
                return;
            lock (_gate)
            {
                if (_entries.TryGetValue(fullPath, out Entry? entry))
                    _remembered -= entry.Text?.Length ?? 0;
                _entries.Remove(fullPath);
            }
        }

        /// <summary>Forget everything: for a ledger whose agent has finished for good.</summary>
        internal void Clear()
        {
            lock (_gate)
            {
                _entries.Clear();
                _remembered = 0;
            }
        }

        /// <summary>How many paths are tracked. For tests and for the structured log line.</summary>
        public int Count
        {
            get { lock (_gate) return _entries.Count; }
        }

        /// <summary>
        /// Give up remembered TEXT, oldest first, until the total fits — keeping every
        /// entry's hash and range, so freshness is unaffected.
        /// </summary>
        private void Trim()
        {
            while (_remembered > MaxRememberedTotalChars)
            {
                Entry? oldest = null;
                long oldestTouched = long.MaxValue;
                foreach (KeyValuePair<string, Entry> pair in _entries)
                {
                    if (pair.Value.Text != null && pair.Value.Touched < oldestTouched)
                    {
                        oldestTouched = pair.Value.Touched;
                        oldest = pair.Value;
                    }
                }
                if (oldest == null)
                    break;
                _remembered -= oldest.Text!.Length;
                oldest.Text = null;
            }
        }

        private void Evict()
        {
            while (_entries.Count > MaxTracked)
            {
                string? oldest = null;
                long oldestTouched = long.MaxValue;
                foreach (KeyValuePair<string, Entry> pair in _entries)
                {
                    if (pair.Value.Touched < oldestTouched)
                    {
                        oldestTouched = pair.Value.Touched;
                        oldest = pair.Key;
                    }
                }
                if (oldest == null)
                    break;
                if (_entries.TryGetValue(oldest, out Entry? dropped))
                    _remembered -= dropped.Text?.Length ?? 0;
                _entries.Remove(oldest);
            }
        }

        /// <summary>
        /// FNV-1a over the UTF-16 code units.
        ///
        /// <para>
        /// A hash and not a length-and-timestamp pair, because the case this has to catch
        /// sits inside both of those: the worst measured rewrite re-typed 188 lines to
        /// change one, leaving 6,780 of 6,808 characters identical — a length delta of 28
        /// bytes and a timestamp that moves for any write at all. Not cryptographic, and
        /// it does not need to be: a collision would mean calling a changed file fresh,
        /// and the edit is still matched against the file's real current bytes before
        /// anything is written.
        /// </para>
        /// </summary>
        private static ulong Hash(string content)
        {
            const ulong offset = 14695981039346656037UL;
            const ulong prime = 1099511628211UL;

            ulong hash = offset;
            for (int i = 0; i < content.Length; i++)
            {
                hash ^= content[i];
                hash *= prime;
            }
            return hash ^ (ulong)content.Length;
        }
    }

    /// <summary>How what the model has been shown of a file relates to what is on disk.</summary>
    public enum ReadFreshness
    {
        /// <summary>Never read in this conversation, and never written by it either.</summary>
        Unread,

        /// <summary>Read in full, and unchanged since.</summary>
        Fresh,

        /// <summary>Part of it was read, and that part is unchanged since.</summary>
        Partial,

        /// <summary>Read, but the file's content has changed since.</summary>
        Stale,
    }
}
