// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace TensorSharp.Runtime
{
    /// <summary>
    /// Tool calls written as Python calls: <c>[read_file(path="notes.txt"), shell("wc -l notes.txt")]</c>.
    ///
    /// <para>
    /// Nemotron-H 8B Reasoning-128K writes this inside its tool-call tags whatever
    /// the prompt asks for. Measured on the Q4_K_M with the Hermes JSON instructions:
    /// every call it made was a Python list of calls, never the JSON object. The body
    /// failed JSON and the Qwen XML fallback, so the turn ended with the call consumed
    /// as tool text and nothing run.
    /// </para>
    /// <para>
    /// Python literals only: strings (single, double, triple-quoted, raw), numbers,
    /// <c>True</c>/<c>False</c>/<c>None</c> (and the JSON spellings), lists, tuples and
    /// dicts. A positional argument binds to the tool's declared parameter at that
    /// position, so it needs a declared tool. Anything else fails, including a body
    /// cut off by EOS or the token limit: a half-written command must not run.
    /// </para>
    /// </summary>
    internal static class PythonicToolCalls
    {
        // Below System.Text.Json's default 64, so every parsed call can be written back
        // out (the client's tool_calls, the next turn's history) like a JSON call can,
        // and model output cannot recurse the parser into a stack overflow.
        private const int MaxDepth = 60;

        public static bool TryParse(string body, IReadOnlyDictionary<string, ToolFunction> tools,
            out List<(string Name, Dictionary<string, object?> Arguments)> calls)
        {
            calls = new List<(string, Dictionary<string, object?>)>();
            if (string.IsNullOrWhiteSpace(body)) return false;
            var reader = new Reader(body);
            try
            {
                reader.SkipSpace();
                bool bracketed = reader.TryTake('[');
                while (true)
                {
                    reader.SkipSpace();
                    if (bracketed && reader.TryTake(']')) break;
                    if (!bracketed && reader.AtEnd) break;
                    calls.Add(ReadCall(ref reader, tools));
                    reader.SkipSpace();
                    if (reader.TryTake(',')) continue;
                    if (bracketed)
                    {
                        reader.Expect(']');
                        break;
                    }
                    if (!reader.AtEnd && !IsNameStart(reader.Peek)) throw new FormatException();
                }
                reader.SkipSpace();
                if (!reader.AtEnd) throw new FormatException();
                return calls.Count > 0;
            }
            catch (FormatException)
            {
                calls.Clear();
                return false;
            }
        }

        private static (string, Dictionary<string, object?>) ReadCall(ref Reader reader, IReadOnlyDictionary<string, ToolFunction> tools)
        {
            string name = reader.ReadName();
            reader.SkipSpace();
            reader.Expect('(');
            tools.TryGetValue(name, out ToolFunction? tool);
            List<string>? order = tool != null ? new List<string>(tool.Parameters.Keys) : null;
            var args = new Dictionary<string, object?>(StringComparer.Ordinal);
            int positional = 0;
            bool sawKeyword = false;
            while (true)
            {
                reader.SkipSpace();
                if (reader.TryTake(')')) break;
                int mark = reader.Position;
                string? key = null;
                if (IsNameStart(reader.Peek))
                {
                    string candidate = reader.ReadName();
                    reader.SkipSpace();
                    if (reader.Peek == '=' && reader.PeekAt(1) != '=')
                    {
                        reader.Take();
                        key = candidate;
                    }
                    else
                    {
                        reader.Position = mark;
                    }
                }
                object? value = ReadValue(ref reader, depth: 1);
                if (key == null)
                {
                    if (sawKeyword || order == null || positional >= order.Count) throw new FormatException();
                    key = order[positional++];
                }
                else
                {
                    sawKeyword = true;
                }
                if (args.ContainsKey(key)) throw new FormatException();
                args[key] = value;
                reader.SkipSpace();
                if (reader.TryTake(',')) continue;
                reader.Expect(')');
                break;
            }
            return (name, args);
        }

        private static object? ReadValue(ref Reader reader, int depth)
        {
            if (depth > MaxDepth) throw new FormatException();
            reader.SkipSpace();
            char c = reader.Peek;
            if (c == '"' || c == '\'') return reader.ReadString(raw: false);
            if ((c == 'r' || c == 'R') && (reader.PeekAt(1) == '"' || reader.PeekAt(1) == '\''))
            {
                reader.Take();
                return reader.ReadString(raw: true);
            }
            if (c == '[' || c == '(')
            {
                char close = c == '[' ? ']' : ')';
                reader.Take();
                var list = new List<object?>();
                while (true)
                {
                    reader.SkipSpace();
                    if (reader.TryTake(close)) return list;
                    list.Add(ReadValue(ref reader, depth + 1));
                    reader.SkipSpace();
                    if (reader.TryTake(',')) continue;
                    reader.Expect(close);
                    return list;
                }
            }
            if (c == '{')
            {
                reader.Take();
                var dict = new Dictionary<string, object?>(StringComparer.Ordinal);
                while (true)
                {
                    reader.SkipSpace();
                    if (reader.TryTake('}')) return dict;
                    object? key = ReadValue(ref reader, depth + 1);
                    if (key is not string text) throw new FormatException();
                    reader.SkipSpace();
                    reader.Expect(':');
                    dict[text] = ReadValue(ref reader, depth + 1);
                    reader.SkipSpace();
                    if (reader.TryTake(',')) continue;
                    reader.Expect('}');
                    return dict;
                }
            }
            if (c == '-' || c == '+' || c == '.' || char.IsDigit(c)) return reader.ReadNumber();
            if (IsNameStart(c))
            {
                return reader.ReadName() switch
                {
                    "True" or "true" => true,
                    "False" or "false" => false,
                    "None" or "null" => null,
                    _ => throw new FormatException(),
                };
            }
            throw new FormatException();
        }

        private static bool IsNameStart(char c) => char.IsLetter(c) || c == '_';

        private struct Reader
        {
            private readonly string _text;

            public Reader(string text)
            {
                _text = text;
                Position = 0;
            }

            public int Position;
            public bool AtEnd => Position >= _text.Length;
            public char Peek => Position < _text.Length ? _text[Position] : '\0';
            public char PeekAt(int offset) => Position + offset < _text.Length ? _text[Position + offset] : '\0';

            public char Take()
            {
                if (AtEnd) throw new FormatException();
                return _text[Position++];
            }

            public bool TryTake(char c)
            {
                if (Peek != c || AtEnd) return false;
                Position++;
                return true;
            }

            public void Expect(char c)
            {
                if (!TryTake(c)) throw new FormatException();
            }

            public void SkipSpace()
            {
                while (!AtEnd && char.IsWhiteSpace(_text[Position])) Position++;
            }

            /// <summary>An identifier; dots join a qualified tool name (<c>functions.shell</c>).</summary>
            public string ReadName()
            {
                int start = Position;
                if (!IsNameStart(Peek)) throw new FormatException();
                while (!AtEnd && (char.IsLetterOrDigit(Peek) || Peek == '_' || Peek == '-'
                                  || (Peek == '.' && IsNameStart(PeekAt(1)))))
                    Position++;
                return _text.Substring(start, Position - start);
            }

            public object ReadNumber()
            {
                int start = Position;
                if (Peek == '-' || Peek == '+') Position++;
                while (!AtEnd && (char.IsDigit(Peek) || Peek == '.' || Peek == '_' || Peek == 'e' || Peek == 'E'
                                  || ((Peek == '-' || Peek == '+') && (PeekAt(-1) == 'e' || PeekAt(-1) == 'E'))))
                    Position++;
                string text = _text.Substring(start, Position - start).Replace("_", string.Empty);
                if (long.TryParse(text, NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture, out long l)) return l;
                if (double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out double d)) return d;
                throw new FormatException();
            }

            public string ReadString(bool raw)
            {
                char quote = Take();
                bool triple = Peek == quote && PeekAt(1) == quote;
                if (triple) Position += 2;
                var sb = new StringBuilder();
                while (true)
                {
                    if (AtEnd) throw new FormatException();
                    char c = _text[Position];
                    if (c == quote && (!triple || (PeekAt(1) == quote && PeekAt(2) == quote)))
                    {
                        Position += triple ? 3 : 1;
                        return sb.ToString();
                    }
                    if (!triple && c == '\n') throw new FormatException();
                    Position++;
                    if (c != '\\')
                    {
                        sb.Append(c);
                        continue;
                    }
                    char e = Take();
                    if (raw)
                    {
                        sb.Append('\\').Append(e);
                        continue;
                    }
                    switch (e)
                    {
                        case 'n': sb.Append('\n'); break;
                        case 't': sb.Append('\t'); break;
                        case 'r': sb.Append('\r'); break;
                        case >= '0' and <= '7':
                        {
                            // Up to three octal digits: "\033" is ESC, "\0" alone is NUL.
                            int code = e - '0';
                            for (int k = 0; k < 2 && Peek >= '0' && Peek <= '7'; k++)
                                code = code * 8 + (Take() - '0');
                            sb.Append((char)code);
                            break;
                        }
                        case '\\': sb.Append('\\'); break;
                        case '\'': sb.Append('\''); break;
                        case '"': sb.Append('"'); break;
                        case '\n': break; // a line continuation
                        case 'u':
                        case 'x':
                        {
                            int digits = e == 'u' ? 4 : 2;
                            if (Position + digits > _text.Length
                                || !int.TryParse(_text.AsSpan(Position, digits), NumberStyles.HexNumber,
                                    CultureInfo.InvariantCulture, out int code))
                                throw new FormatException();
                            Position += digits;
                            sb.Append((char)code);
                            break;
                        }
                        default:
                            // Python keeps an unknown escape as written: "\d" is backslash + d.
                            sb.Append('\\').Append(e);
                            break;
                    }
                }
            }
        }
    }
}
