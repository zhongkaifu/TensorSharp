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
using System.Buffers;
using System.Diagnostics.CodeAnalysis;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace TensorSharp.Runtime.Scheduling
{
    /// <summary>
    /// Recognises a generation that has locked into a loop, so the engine can end it
    /// instead of streaming the same few tokens until somebody presses Stop.
    ///
    /// <para>
    /// Observed on a phone: a 4-bit 9B model, asked for a slide deck, was writing a
    /// long block of XML inside a Python string when it fell into <c>","+","+","+"</c>
    /// and stayed there — 230 repetitions, five and a half minutes, ended only by the
    /// user tapping Stop. Nothing in the pipeline could end it: the penalties are
    /// deliberately off for code (they corrupt legitimately repetitive structure), the
    /// reply limit was hundreds of thousands of tokens, and a loop never reaches EOS.
    /// The output was not wrong because of the cache (99.8% reuse) or the prompt; it
    /// was a sampling degeneration in a long, low-entropy stretch, which is the one
    /// failure that sampling settings cannot rule out for every model and every text.
    /// </para>
    /// <para>
    /// The test is purely structural — no decoding, no language model — and errs on
    /// the side of letting output through: a loop is declared only when the LAST
    /// <c>max(MinSpan, MinRepeats × period)</c> generated tokens are exactly periodic
    /// with a period of at most <see cref="MaxPeriod"/> tokens. That is 128 identical
    /// tokens in a row, or a 3-token phrase 43 times over, or a 32-token paragraph 8
    /// times word for word. Real writing repeats shorter than that (a table separator
    /// row, a rule of dashes, a list of identical values) or with variation; and the
    /// cost of the rare false stop is one round in which the model is told why and
    /// writes it differently, while the cost of a miss is minutes of garbage on a
    /// battery.
    /// </para>
    /// <para>
    /// Cost: bounded by <c>MaxPeriod × MinSpan</c> comparisons per token in the
    /// pathological case and a handful in the normal one, because a period is only
    /// examined when the newest token equals the one that period back.
    /// </para>
    /// <para>
    /// A second rule catches a loop whose unit is a whole block rather than a phrase.
    /// Observed on TensorSharp.Server: LFM3-Omni 2B, after a refused <c>spawn_agent</c>,
    /// wrote an <c>apply_patch</c> body that repeated one 16-line markdown block (222
    /// tokens) word for word, 520 times. That was 115,498 tokens in 17 minutes,
    /// ended only when the client gave up. A unit that long is past
    /// <see cref="MaxPeriod"/>, so the first rule never looked at it. The second rule
    /// looks at units of up to <see cref="MaxBlockPeriod"/> tokens, and it is much
    /// stricter, because a long unit is where real writing does repeat itself: table
    /// rows, getters, CSV lines, sections under the same headings. It needs the LAST
    /// <c>max(MinBlockSpan, MinBlockRepeats × period)</c> tokens to be exactly
    /// periodic. That is at least 4,096 tokens of verbatim copy: 18 copies of the
    /// 222-token block, four of a 1,024-token one, and four of the longest (16,384
    /// tokens). Real writing that repeats a long unit varies it (the row's values, the
    /// getter's name, the section's text), and one changed token ends the run. What
    /// is left is thousands of tokens copied word for word, which is a loop.
    /// </para>
    /// <para>
    /// Cost of the second rule: it runs only once the output is
    /// <see cref="MinBlockSpan"/> tokens long, and then only when the length is a
    /// multiple of <see cref="BlockCheckInterval"/>. Each run reads a fixed window at
    /// the end of the output, never the whole of it. That is one vectorized scan of
    /// about 4,000 tokens for the newest token one unit back, an 8-token probe at
    /// each end of the span for each hit, and at most
    /// <see cref="MaxBlockVerifications"/> full comparisons of the span. So the work
    /// per token is a constant, the same on a 262K-token reply as on a 5K one.
    /// </para>
    /// </summary>
    public static class RepetitionGuard
    {
        /// <summary>The longest repeating unit, in tokens, that counts as a loop.</summary>
        public const int MaxPeriod = 64;

        /// <summary>
        /// A loop must repeat its unit at least this many times over.
        ///
        /// <para>
        /// Eight rather than four, and that is the parameter that decides the false
        /// positives. A degenerate loop's unit is SHORT — the observed one was three
        /// tokens — and a short unit is bounded by <see cref="MinSpan"/> anyway, so
        /// raising this costs the real case nothing. What it buys is the long-unit
        /// corner: four byte-identical copies of a 32-token block is ordinary
        /// structured output (four empty paragraphs in a row of OOXML, four identical
        /// rows of a table), and stopping that would be wrong. Eight in a row, with no
        /// variation at all, is not something a model writes on purpose.
        /// </para>
        /// </summary>
        public const int MinRepeats = 8;

        /// <summary>
        /// ...and the periodic tail must be at least this long in tokens, so that a
        /// short unit has to recur many more times than a long one before it counts:
        /// 128 copies of one token, 43 of a three-token phrase, 8 of a 32-token block.
        /// </summary>
        public const int MinSpan = 128;

        /// <summary>
        /// The longest repeating unit, in tokens, that the long-block rule looks for:
        /// a whole section or file part written out again and again.
        /// </summary>
        public const int MaxBlockPeriod = 4096;

        /// <summary>
        /// A unit longer than <see cref="MaxPeriod"/> must repeat at least this many whole
        /// times. This decides only for units of 1,024 tokens or more. For a shorter unit,
        /// <see cref="MinBlockSpan"/> asks for more copies.
        /// </summary>
        public const int MinBlockRepeats = 4;

        /// <summary>
        /// ...and the exactly periodic tail must hold at least this many tokens. This is
        /// the parameter that decides the long-block rule's false positives. About 16 KB
        /// of verbatim copy is not something real writing contains: a table, a list or a
        /// file with repeated structure changes something in every copy. A loop changes
        /// nothing, and it runs far past this length. The observed one went to 115,498
        /// tokens.
        /// </summary>
        public const int MinBlockSpan = 4096;

        /// <summary>
        /// The long-block rule is asked only when the output length is a multiple of this.
        /// A loop it catches has already run for thousands of tokens, so a few dozen more
        /// cost nothing.
        /// </summary>
        public const int BlockCheckInterval = 64;

        /// <summary>
        /// The most full comparisons of a span that one long-block check makes. A
        /// candidate period gets one only after the newest and the oldest few tokens of
        /// its span both recur one unit back. In a real loop the first such candidate is
        /// nearly always the loop's own period.
        /// </summary>
        internal const int MaxBlockVerifications = 8;

        /// <summary>
        /// The tokens at each end of a candidate span that must recur one unit back
        /// before the whole span is compared.
        /// </summary>
        private const int BlockProbe = 8;

        /// <summary>
        /// How far back, in tokens, a block loop that was found is followed to count its
        /// copies. It does not change what counts as a loop, only how large the reported
        /// count can be.
        /// </summary>
        private const int BlockCountWindow = 65536;

        /// <summary>The finish reason the engine reports when this guard ends a sequence.</summary>
        public const string FinishReason = "repetition";

        /// <summary>
        /// Whether the first <paramref name="count"/> entries of
        /// <paramref name="tokens"/> end in a loop, by either rule: a unit of at most
        /// <see cref="MaxPeriod"/> tokens, or a block of at most
        /// <see cref="MaxBlockPeriod"/>. Only the newest token is new since the previous
        /// call, so the check is meant to run once per emitted token. The answer depends
        /// only on those tokens, never on earlier calls.
        /// </summary>
        /// <param name="tokens">The generated tokens (never the prompt).</param>
        /// <param name="count">How many of them count; the rest may be an unpublished speculative tail.</param>
        /// <param name="period">The loop's period, when one is found.</param>
        /// <param name="repeats">How many whole copies of the unit the periodic tail holds.</param>
        public static bool IsLooping(IReadOnlyList<int> tokens, int count, out int period, out int repeats)
        {
            period = 0;
            repeats = 0;
            if (tokens == null)
                return false;
            if (count > tokens.Count)
                count = tokens.Count;
            if (count < MinSpan)
                return false;

            int last = tokens[count - 1];
            for (int p = 1; p <= MaxPeriod; p++)
            {
                int span = Math.Max(MinSpan, MinRepeats * p);
                if (span > count)
                    break;
                // Cheap pre-check: the newest token must equal the one a period back,
                // which for ordinary text prunes almost every period immediately.
                if (tokens[count - 1 - p] != last)
                    continue;

                // The tail is p-periodic when every token in it equals the one p
                // before it; the first p tokens of the tail are the unit itself.
                bool periodic = true;
                for (int i = count - span; i < count - p; i++)
                {
                    if (tokens[i] != tokens[i + p])
                    {
                        periodic = false;
                        break;
                    }
                }
                if (!periodic)
                    continue;

                period = p;
                // Extend backwards past the minimum span so the report says how far
                // the loop really reached, capped so this stays cheap.
                int start = count - span;
                int limit = Math.Max(0, count - 64 * MaxPeriod);
                while (start - 1 >= limit && tokens[start - 1] == tokens[start - 1 + p])
                    start--;
                repeats = (count - start) / p;
                return true;
            }

            // No short unit. A long block is looked for far less often, and only once
            // the output can hold the span the rule asks for.
            return count >= MinBlockSpan && count % BlockCheckInterval == 0
                && TryFindBlockLoop(tokens, count, out period, out repeats);
        }

        /// <summary>
        /// The long-block rule on the first <paramref name="count"/> tokens. It reads the
        /// list's own storage when the list is a <see cref="List{T}"/> or an array, as the
        /// engine's output tokens are. Any other list is copied, once per check, up to the
        /// window the rule reads.
        /// </summary>
        private static bool TryFindBlockLoop(IReadOnlyList<int> tokens, int count, out int period, out int repeats)
        {
            int window = Math.Min(count, BlockCountWindow);
            int offset = count - window;
            int[]? rented = null;
            try
            {
                ReadOnlySpan<int> tail;
                if (tokens is List<int> list)
                    tail = CollectionsMarshal.AsSpan(list).Slice(offset, window);
                else if (tokens is int[] array)
                    tail = array.AsSpan(offset, window);
                else
                {
                    rented = ArrayPool<int>.Shared.Rent(window);
                    for (int i = 0; i < window; i++)
                        rented[i] = tokens[offset + i];
                    tail = rented.AsSpan(0, window);
                }
                long work = 0;
                return FindBlockLoop(tail, out period, out repeats, ref work);
            }
            finally
            {
                if (rented != null)
                    ArrayPool<int>.Shared.Return(rented);
            }
        }

        /// <summary>
        /// Whether <paramref name="tail"/> ends in a block loop: a unit longer than
        /// <see cref="MaxPeriod"/> and at most <see cref="MaxBlockPeriod"/> tokens long,
        /// with the last <c>max(MinBlockSpan, MinBlockRepeats × period)</c> tokens exactly
        /// periodic. The smallest such period is reported.
        /// </summary>
        /// <param name="tail">The end of the output. The rule reads at most its last
        /// <c>MinBlockRepeats × MaxBlockPeriod</c> tokens. The rest is only used to count
        /// the copies of a loop that was found.</param>
        /// <param name="period">The loop's period, when one is found.</param>
        /// <param name="repeats">How many whole copies of the unit the periodic tail holds.</param>
        /// <param name="work">Increased by the number of tokens this call compared or
        /// scanned (an upper bound). Tests use it to show the cost does not grow with
        /// the output.</param>
        internal static bool FindBlockLoop(ReadOnlySpan<int> tail, out int period, out int repeats, ref long work)
        {
            period = 0;
            repeats = 0;
            int n = tail.Length;
            if (n < MinBlockSpan)
                return false;

            // A period d can only count when its span, max(MinBlockSpan, 4d), fits.
            int maxPeriod = Math.Min(MaxBlockPeriod, n / MinBlockRepeats);
            if (maxPeriod <= MaxPeriod)
                return false;

            // The candidates are the periods d for which the newest token recurs d back:
            // tail[n - 1 - d] == last. They are found with one vectorized scan of the
            // positions n - 1 - maxPeriod ... n - 2 - MaxPeriod, nearest first, so the
            // smallest period is tried first.
            int last = tail[n - 1];
            int lo = n - 1 - maxPeriod;
            ReadOnlySpan<int> region = tail.Slice(lo, maxPeriod - MaxPeriod);
            work += region.Length;
            ReadOnlySpan<int> newest = tail.Slice(n - BlockProbe, BlockProbe);
            int verifications = 0;
            int end = region.Length;
            while (end > 0)
            {
                int at = region.Slice(0, end).LastIndexOf(last);
                if (at < 0)
                    break;
                end = at;
                int d = n - 1 - (lo + at);
                int span = Math.Max(MinBlockSpan, MinBlockRepeats * d);

                // Cheap probes first. The newest few tokens must recur one unit back, and
                // so must the oldest few of the span. Ordinary text fails one of them at
                // once, and so does a loop that has already ended (or not yet reached the
                // span), whatever its period.
                int from = n - span;
                work += BlockProbe;
                if (!newest.SequenceEqual(tail.Slice(n - BlockProbe - d, BlockProbe)))
                    continue;
                work += BlockProbe;
                if (!tail.Slice(from, BlockProbe).SequenceEqual(tail.Slice(from + d, BlockProbe)))
                    continue;

                // The whole span, oldest first: every token equals the one d after it.
                if (++verifications > MaxBlockVerifications)
                    break;
                work += span - d;
                if (!tail.Slice(from, span - d).SequenceEqual(tail.Slice(from + d, span - d)))
                    continue;

                period = d;
                // Count the copies back to where the loop began, within the window.
                int start = from;
                while (start > 0 && tail[start - 1] == tail[start - 1 + d])
                    start--;
                work += from - start;
                repeats = (n - start) / d;
                return true;
            }
            return false;
        }

        /// <summary>
        /// The same test on CHARACTERS, for a layer that has text rather than tokens:
        /// true when <paramref name="text"/> ends in a unit of at most 16,384 characters
        /// repeated at least 4 times over at least 64 characters. Used to quote the loop
        /// back to the model; it is not what decides to stop.
        /// </summary>
        public static bool TryFindTextLoop(string text, [NotNullWhen(true)] out string? unit, out int repeats)
        {
            unit = null;
            repeats = 0;
            if (string.IsNullOrEmpty(text) || text.Length < 64)
                return false;
            // Looser than the token test on purpose, and in the safe direction: this
            // one only decides how the loop is DESCRIBED, and a 64-token unit (the
            // short rule's cap) can be several hundred characters, so a tighter cap here
            // would fall back to the vague sentence for exactly the cases the token
            // guard just stopped. The cap covers the long-block rule too: a unit of
            // MaxBlockPeriod tokens is up to about 16K characters of English or code.
            // Run once per looping round, never per token.
            const int maxPeriod = 16384, minRepeats = 4, minSpan = 64;
            int n = text.Length;
            char last = text[n - 1];
            for (int p = 1; p <= maxPeriod; p++)
            {
                int span = Math.Max(minSpan, minRepeats * p);
                if (span > n)
                    break;
                if (text[n - 1 - p] != last)
                    continue;
                bool periodic = true;
                for (int i = n - span; i < n - p; i++)
                {
                    if (text[i] != text[i + p])
                    {
                        periodic = false;
                        break;
                    }
                }
                if (!periodic)
                    continue;
                int start = n - span;
                while (start - 1 >= 0 && text[start - 1] == text[start - 1 + p])
                    start--;
                unit = text.Substring(n - p, p);
                repeats = (n - start) / p;
                return true;
            }
            return false;
        }

        /// <summary>
        /// One sentence for a log line or a note to the model: what repeated and how
        /// many times, with the unit rendered by <paramref name="decode"/> when a
        /// tokenizer is at hand.
        /// </summary>
        public static string Describe(IReadOnlyList<int> tokens, int count, int period, int repeats, Func<List<int>, string?> decode)
        {
            string unit = string.Empty;
            if (tokens != null && decode != null && period > 0 && count >= period)
            {
                var ids = new List<int>(period);
                for (int i = count - period; i < count; i++)
                    ids.Add(tokens[i]);
                try
                {
                    unit = decode(ids) ?? string.Empty;
                }
                catch (Exception)
                {
                    unit = string.Empty;
                }
                unit = unit.Replace("\r", "\\r").Replace("\n", "\\n");
                if (unit.Length > 48)
                    unit = unit.Substring(0, 48) + "…";
            }
            string what = unit.Length > 0 ? $"`{unit}`" : $"a {period}-token sequence";
            return $"the output repeated {what} {repeats} times in a row";
        }
    }
}
