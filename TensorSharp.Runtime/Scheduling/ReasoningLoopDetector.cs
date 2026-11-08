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
using System.Text;

namespace TensorSharp.Runtime.Scheduling
{
    /// <summary>
    /// Recognises reasoning that has settled into a templated loop: line after line the
    /// same apart from a number or a word, which <see cref="RepetitionGuard"/> (exact
    /// periods only) never sees.
    ///
    /// <para>
    /// Observed with Qwen3.8-27B NVFP4 under Aider (issue #338): three turns counted
    /// through cases one line at a time - <c>Need maybe tests expect to_string for
    /// 1:1533 -> 02:33. Good.</c>, then <c>1:1534</c>, for thousands of tokens, or
    /// <c>"three hundred twenty-one" for 321</c>, <c>"... twenty-two" for 322</c> -
    /// until the model ended the turn mid-phrase inside its reasoning.
    /// </para>
    /// <para>
    /// The test works on the decoded text, a line at a time. A line's digit runs become
    /// one <c>#</c> and its whitespace runs one space; a line then repeats an earlier one
    /// when its edit distance to one of the previous <see cref="Window"/> lines is at most
    /// a fifth of the longer (so a changed number word, <c>twenty-nine</c> to
    /// <c>thirty</c>, still repeats). Blank lines and lines shorter than
    /// <see cref="MinLineChars"/> characters or without a letter are skipped: they
    /// neither extend nor break a run. <see cref="MinLines"/> repeating lines in a row
    /// is a loop.
    /// </para>
    /// <para>
    /// A finding only ever closes an open reasoning channel (the model then answers);
    /// it never stops a reply. Templated lines in an answer - a table, test cases, a
    /// lookup in code - are content the caller asked for, and a run inside reasoning
    /// that was legitimate costs the rest of that reasoning, not the answer.
    /// </para>
    /// </summary>
    public sealed class ReasoningLoopDetector
    {
        /// <summary>Repeating lines in a row that count as a loop. A checklist the model
        /// writes on purpose is shorter; the observed loops ran to hundreds of lines.</summary>
        public const int MinLines = 32;

        /// <summary>Shorter lines (after normalization) are skipped: "Good.", a rule.</summary>
        public const int MinLineChars = 12;

        /// <summary>A line may repeat any of this many previous lines (a two-line unit).</summary>
        public const int Window = 4;

        // A line with no break is cut here, so the buffer stays bounded; the piece
        // rarely repeats anything, which restarts the run.
        private const int MaxLineBytes = 1024;
        // Longer lines are compared on their first this-many characters.
        private const int MaxCompareChars = 384;

        private readonly List<byte> _line = new();
        private readonly string[] _recent = new string[Window];
        private int _recentCount;
        private int _recentNext;

        /// <summary>Repeating lines in the current run.</summary>
        public int RunLines { get; private set; }

        /// <summary>The newest line of the run, normalized, for a log line.</summary>
        public string? LastLine { get; private set; }

        /// <summary>True once the current run is <see cref="MinLines"/> lines long.</summary>
        public bool Looping => RunLines >= MinLines;

        /// <summary>Feed one emitted token's bytes.</summary>
        public void Observe(ReadOnlySpan<byte> tokenBytes)
        {
            foreach (byte b in tokenBytes)
            {
                if (b == (byte)'\n')
                {
                    EndLine();
                    continue;
                }
                _line.Add(b);
                if (_line.Count >= MaxLineBytes)
                    EndLine();
            }
        }

        /// <summary>Forget the run and the previous lines (after a finding was acted on).</summary>
        public void Reset()
        {
            RunLines = 0;
            LastLine = null;
            Array.Clear(_recent);
            _recentCount = 0;
            _recentNext = 0;
            _line.Clear();
        }

        private void EndLine()
        {
            if (_line.Count == 0)
                return;
            // '\n' is never inside a UTF-8 sequence; a cut line may end mid-character,
            // which decodes to a replacement character and is harmless here.
            string text = Encoding.UTF8.GetString(System.Runtime.InteropServices.CollectionsMarshal.AsSpan(_line));
            _line.Clear();
            string? line = Normalize(text);
            if (line == null)
                return;
            bool repeats = false;
            for (int i = 0; i < _recentCount && !repeats; i++)
                repeats = Similar(line, _recent[i]);
            RunLines = repeats ? RunLines + 1 : 1;
            LastLine = line;
            _recent[_recentNext] = line;
            _recentNext = (_recentNext + 1) % Window;
            if (_recentCount < Window)
                _recentCount++;
        }

        /// <summary>The line with digit runs as <c>#</c> and whitespace runs as one space,
        /// trimmed; null for a line the test skips.</summary>
        internal static string? Normalize(string text)
        {
            var sb = new StringBuilder(Math.Min(text.Length, MaxCompareChars));
            bool letter = false, pendingSpace = false;
            for (int i = 0; i < text.Length && sb.Length < MaxCompareChars; i++)
            {
                char c = text[i];
                if (char.IsWhiteSpace(c))
                {
                    pendingSpace = sb.Length > 0;
                    continue;
                }
                if (pendingSpace)
                {
                    sb.Append(' ');
                    pendingSpace = false;
                }
                if (c >= '0' && c <= '9')
                {
                    while (i + 1 < text.Length && text[i + 1] >= '0' && text[i + 1] <= '9')
                        i++;
                    sb.Append('#');
                    continue;
                }
                letter |= char.IsLetter(c);
                sb.Append(c);
            }
            return letter && sb.Length >= MinLineChars ? sb.ToString() : null;
        }

        /// <summary>Edit distance at most a fifth of the longer line (at least 2).</summary>
        internal static bool Similar(string a, string b)
        {
            int allowed = Math.Max(2, Math.Max(a.Length, b.Length) / 5);
            return Math.Abs(a.Length - b.Length) <= allowed && EditDistanceWithin(a, b, allowed);
        }

        // Levenshtein distance restricted to the diagonal band |i - j| <= k, with an
        // early exit once a whole row exceeds k: O(n * k).
        private static bool EditDistanceWithin(string a, string b, int k)
        {
            int n = a.Length, m = b.Length;
            const int Far = int.MaxValue / 2;
            Span<int> prev = m + 1 <= 512 ? stackalloc int[m + 1] : new int[m + 1];
            Span<int> cur = m + 1 <= 512 ? stackalloc int[m + 1] : new int[m + 1];
            for (int j = 0; j <= m; j++)
                prev[j] = j <= k ? j : Far;
            for (int i = 1; i <= n; i++)
            {
                int lo = Math.Max(1, i - k), hi = Math.Min(m, i + k);
                // Only the band is computed; the cells just outside it, which the next
                // row reads, are marked unreachable.
                cur[0] = i <= k ? i : Far;
                if (lo > 1)
                    cur[lo - 1] = Far;
                int best = cur[0];
                for (int j = lo; j <= hi; j++)
                {
                    int v = prev[j - 1] + (a[i - 1] == b[j - 1] ? 0 : 1);
                    v = Math.Min(v, prev[j] + 1);
                    v = Math.Min(v, cur[j - 1] + 1);
                    cur[j] = v;
                    best = Math.Min(best, v);
                }
                if (hi < m)
                    cur[hi + 1] = Far;
                if (best > k)
                    return false;
                Span<int> swap = prev;
                prev = cur;
                cur = swap;
            }
            return prev[m] <= k;
        }
    }
}
