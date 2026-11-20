// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Text;
using System.Text.RegularExpressions;

namespace TensorSharp.Models.Audio
{
    /// <summary>
    /// Cuts a reply's text, as it streams, into the pieces a TTS prompt speaks one at a time (a spoken reply made from a text
    /// reply: <see cref="SpeechReplyPath.Synthesized"/>). A piece ends at a sentence's end (。！？；… and ! ? ; or a period
    /// after a letter, each followed by whitespace), at a line's end, or, in a run longer than <see cref="MaxChars"/>
    /// without one, at its last comma or space. The first piece also ends at a comma once it has
    /// <see cref="FirstPieceMinChars"/> characters, so the speech starts before a long first sentence is complete. Each piece is the speakable text of the markdown the reply writes
    /// (<see cref="Speakable"/>): markers, emoji, fenced code and table rows are not read aloud, a long path or a URL is
    /// said by its last name or its host, a long inline code span not at all; a piece with nothing left to say is dropped.
    /// </summary>
    public sealed class SpeechTextSegmenter
    {
        /// <summary>A run of this many characters without a sentence end is cut at its last comma or space (LFM3-Omni's TTS
        /// speaks at most 512 frames, 41 s, which 100 Chinese characters stay well within).</summary>
        public const int DefaultMaxChars = 100;

        private readonly StringBuilder _line = new();
        private readonly StringBuilder _pending = new();
        private bool _atLineStart = true;
        private bool _inCode;

        /// <summary>The first piece ends at a comma after this many characters (0: at sentence ends only).</summary>
        public const int DefaultFirstPieceMinChars = 8;

        private int _emitted;

        public SpeechTextSegmenter(int maxChars = DefaultMaxChars, int firstPieceMinChars = DefaultFirstPieceMinChars)
        {
            if (maxChars < 8) throw new ArgumentOutOfRangeException(nameof(maxChars), "at least 8 characters");
            if (firstPieceMinChars < 0) throw new ArgumentOutOfRangeException(nameof(firstPieceMinChars), "0 or more");
            MaxChars = maxChars;
            FirstPieceMinChars = firstPieceMinChars;
        }

        public int MaxChars { get; }

        public int FirstPieceMinChars { get; }

        /// <summary>The pieces <paramref name="text"/> completes (possibly none).</summary>
        public IReadOnlyList<string> Append(string text)
        {
            var pieces = new List<string>();
            if (!string.IsNullOrEmpty(text))
            {
                _line.Append(text);
                Drain(pieces, final: false);
            }
            return pieces;
        }

        /// <summary>The rest, at the end of the reply or of a round of it (a tool call follows the text so far): what is
        /// pending is spoken now, and the text after it starts a new line.</summary>
        public IReadOnlyList<string> Flush()
        {
            var pieces = new List<string>();
            Drain(pieces, final: true);
            _atLineStart = true;
            _inCode = false;
            return pieces;
        }

        /// <summary>
        /// Take back <paramref name="text"/>, the end of what was appended (a reply's text that turned out to be reasoning):
        /// where it is still waiting to be spoken it is dropped and true is returned; where some of it was already handed out
        /// as a piece, everything still waiting is dropped (the rest of a piece that is gone means nothing) and false is
        /// returned.
        /// </summary>
        public bool Retract(string text)
        {
            if (string.IsNullOrEmpty(text)) return true;
            string waiting = _pending.ToString() + _line.ToString();
            if (waiting.EndsWith(text, StringComparison.Ordinal))
            {
                int fromLine = Math.Min(text.Length, _line.Length);
                _line.Length -= fromLine;
                _pending.Length -= text.Length - fromLine;
                return true;
            }
            _pending.Clear();
            _line.Clear();
            return false;
        }

        private void Drain(List<string> pieces, bool final)
        {
            while (true)
            {
                int nl = IndexOf(_line, '\n');
                if (nl >= 0)
                {
                    string line = _line.ToString(0, nl);
                    _line.Remove(0, nl + 1);
                    bool lineStart = _atLineStart;
                    _atLineStart = true;
                    if (lineStart && line.TrimStart().StartsWith("```", StringComparison.Ordinal))
                    {
                        Cut(pieces, lineEnded: true);
                        _inCode = !_inCode;
                        continue;
                    }
                    if (_inCode) continue;
                    // A table row is data, not a sentence: reading it cell by cell says nothing a listener can follow.
                    if (lineStart && IsTableRow(line))
                    {
                        Cut(pieces, lineEnded: true);
                        continue;
                    }
                    _pending.Append(line);
                    Cut(pieces, lineEnded: true);
                    continue;
                }
                if (_line.Length == 0) break;
                if (_inCode)
                {
                    if (final) _line.Clear();
                    break;
                }
                // A line that may still turn out to be a code fence or a table row waits for its end.
                if (_atLineStart && !final)
                {
                    string head = _line.ToString().TrimStart();
                    if (head.Length == 0 || "```".StartsWith(head, StringComparison.Ordinal) || head.StartsWith("```", StringComparison.Ordinal)
                        || IsTableRow(head))
                        break;
                }
                if (_atLineStart && final && (_line.ToString().TrimStart().StartsWith("```", StringComparison.Ordinal) || IsTableRow(_line.ToString())))
                {
                    _line.Clear();
                    break;
                }
                _pending.Append(_line);
                _line.Clear();
                _atLineStart = false;
                break;
            }
            Cut(pieces, lineEnded: final);
        }

        /// <summary>Emit the pending text's complete sentences; at a line's end (or the reply's) the rest too.</summary>
        private void Cut(List<string> pieces, bool lineEnded)
        {
            int start = 0;
            string s = _pending.ToString();
            for (int i = 0; i < s.Length; i++)
            {
                int end = SentenceEnd(s, i, lineEnded);
                if (end < 0 && _emitted == 0 && FirstPieceMinChars > 0 && i - start + 1 >= FirstPieceMinChars)
                    end = Comma(s, i);
                if (end < 0) continue;
                Emit(pieces, s.Substring(start, end - start));
                start = end;
                i = end - 1;
            }
            string rest = s.Substring(start);
            while (rest.Length > MaxChars)
            {
                int cut = SoftBreak(rest, MaxChars);
                Emit(pieces, rest.Substring(0, cut));
                rest = rest.Substring(cut);
            }
            if (lineEnded)
            {
                Emit(pieces, rest);
                rest = string.Empty;
            }
            _pending.Clear().Append(rest);
        }

        private void Emit(List<string> pieces, string text)
        {
            string speakable = Speakable(text);
            if (speakable.Length == 0) return;
            pieces.Add(speakable);
            _emitted++;
        }

        /// <summary>Where a clause ending at a comma at <paramref name="i"/> ends, or -1 (an ASCII comma before whitespace).</summary>
        private static int Comma(string s, int i)
        {
            char c = s[i];
            if (c is '，' or '、' or '：') return i + 1;
            if (c is ',' or ':' && i + 1 < s.Length && char.IsWhiteSpace(s[i + 1])) return i + 1;
            return -1;
        }

        private const string ClosingMarks = "\"'”’」』）)]】》*_";

        /// <summary>Where a sentence ending at <paramref name="i"/> ends (after its closing quotes), or -1.</summary>
        private static int SentenceEnd(string s, int i, bool lineEnded)
        {
            char c = s[i];
            bool wide = c is '。' or '！' or '？' or '；' or '…' or '｡';
            bool narrow = c is '!' or '?' or ';' or '.';
            if (!wide && !narrow) return -1;
            int end = i + 1;
            while (end < s.Length && (ClosingMarks.IndexOf(s[end]) >= 0 || s[end] is '。' or '！' or '？' or '…' or '!' or '?' or '.'))
                end++;
            // At the end of what has arrived, a closing quote may still follow: wait for the next piece.
            if (end == s.Length && !lineEnded) return -1;
            if (wide) return end;
            // ASCII: a sentence end only before whitespace (3.14, e.g., file.txt are not), and a period only after a
            // letter (a list's "1." is not).
            if (c == '.' && (i == 0 || !char.IsLetter(s[i - 1]))) return -1;
            if (end < s.Length) return char.IsWhiteSpace(s[end]) ? end : -1;
            return lineEnded ? end : -1;
        }

        /// <summary>Where to cut a run with no sentence end: after its last comma or space within <paramref name="max"/>.</summary>
        private static int SoftBreak(string s, int max)
        {
            for (int i = max - 1; i >= max / 2; i--)
                if (s[i] is '，' or ',' or '、' or '：' or ':' or ' ' or '　') return i + 1;
            return max;
        }

        /// <summary>A markdown table row: a line that starts with a pipe.</summary>
        private static bool IsTableRow(string line) => line.TrimStart().StartsWith("|", StringComparison.Ordinal);

        private static int IndexOf(StringBuilder sb, char c)
        {
            for (int i = 0; i < sb.Length; i++)
                if (sb[i] == c) return i;
            return -1;
        }

        /// <summary>A path's last name (<c>/a/b/c.txt</c>, <c>src/app/main.py</c>: <c>c.txt</c>, <c>main.py</c>). Only what reads
        /// as a path: rooted (<c>/</c>, <c>\</c>, <c>~</c>, <c>.</c>, a drive) or ending in a file name with an extension; a run
        /// of words with slashes (<c>TCP/IP/UDP</c>) or numbers (<c>1/2/3</c>) is said as it is.</summary>
        private static string LastName(string path)
        {
            string trimmed = path.TrimEnd('/', '\\');
            int cut = trimmed.LastIndexOfAny(new[] { '/', '\\' });
            if (cut < 0) return path;
            string last = trimmed.Substring(cut + 1);
            bool rooted = path[0] is '/' or '\\' or '~' or '.' || (path.Length > 1 && path[1] == ':');
            if (!rooted && !FileExtension.IsMatch(last.TrimEnd('.'))) return path;
            return last;
        }

        private static readonly Regex Image = new(@"!\[([^\]]*)\]\([^)]*\)", RegexOptions.Compiled);
        private static readonly Regex Link = new(@"\[([^\]]+)\]\([^)]*\)", RegexOptions.Compiled);
        private static readonly Regex LineMarkers = new(@"^\s*(?:#{1,6}\s+|>\s*|[-*+•]\s+)+", RegexOptions.Compiled);
        private static readonly Regex Rule = new(@"^\s*(?:[-*_=]\s*){3,}$|^\s*\|?(?:\s*:?-{2,}:?\s*\|)+\s*:?-*:?\s*\|?\s*$", RegexOptions.Compiled);
        private static readonly Regex Emphasis = new(@"\*{1,3}|_{2,3}|~~|`+", RegexOptions.Compiled);
        private static readonly Regex CodeSpan = new(@"`+([^`]+)`+", RegexOptions.Compiled);
        private static readonly Regex Url = new(@"(?<![A-Za-z0-9])(?:https?|ftp)://(?:www\.)?([A-Za-z0-9.-]*[A-Za-z0-9])[!-~-[)\]>]]*(?<![.,;:!?'""])", RegexOptions.Compiled | RegexOptions.IgnoreCase);
        private static readonly Regex PathLike = new(@"(?<![\w./\\])(?:[A-Za-z]:|~|\.{1,2})?[/\\]?[A-Za-z0-9_.@+-]+(?:[/\\][A-Za-z0-9_.@+-]+){2,}[/\\]?", RegexOptions.Compiled);
        private static readonly Regex FileExtension = new(@"\.[A-Za-z][A-Za-z0-9]{0,7}$", RegexOptions.Compiled);

        /// <summary>An inline code span this long or longer is not read (a command, an expression, JSON).</summary>
        public const int LongCodeSpanChars = 40;
        private static readonly Regex Spaces = new(@"\s+", RegexOptions.Compiled);

        /// <summary>
        /// What speech says of a piece of a markdown reply: headings', quotes' and bullets' markers, emphasis (* ** __ ~~),
        /// backticks, table pipes, rules, links' targets and emoji (symbols of Unicode categories So / Sk with their
        /// joiners and variation selectors) are dropped; a URL is said as its host (<c>docs.python.org</c>), a path of three
        /// or more parts as its last one (<c>/Users/me/out/result.csv</c> as <c>result.csv</c>), an inline code span of
        /// <see cref="LongCodeSpanChars"/> characters or more not at all; whitespace is collapsed. Empty when no letter or
        /// digit is left.
        /// </summary>
        public static string Speakable(string text)
        {
            if (string.IsNullOrEmpty(text)) return string.Empty;
            if (Rule.IsMatch(text)) return string.Empty;
            string t = Image.Replace(text, "$1");
            t = Link.Replace(t, "$1");
            t = Url.Replace(t, m => m.Groups[1].Value);
            t = CodeSpan.Replace(t, m =>
            {
                string code = m.Groups[1].Value.Trim();
                if (PathLike.Match(code) is { Success: true } path && path.Length == code.Length) return LastName(code);
                return code.Length >= LongCodeSpanChars ? string.Empty : code;
            });
            t = PathLike.Replace(t, m => LastName(m.Value));
            t = LineMarkers.Replace(t, "");
            t = Emphasis.Replace(t, "");
            t = t.Replace('|', ' ');
            var sb = new StringBuilder(t.Length);
            bool any = false;
            foreach (Rune r in t.EnumerateRunes())
            {
                int v = r.Value;
                if (v is 0x200D or 0x20E3 || (v >= 0xFE00 && v <= 0xFE0F) || (v >= 0x1F3FB && v <= 0x1F3FF)) continue;
                var category = Rune.GetUnicodeCategory(r);
                if (category is System.Globalization.UnicodeCategory.OtherSymbol or System.Globalization.UnicodeCategory.ModifierSymbol)
                    continue;
                if (Rune.IsLetterOrDigit(r)) any = true;
                sb.Append(r.ToString());
            }
            return any ? Spaces.Replace(sb.ToString(), " ").Trim() : string.Empty;
        }
    }
}
