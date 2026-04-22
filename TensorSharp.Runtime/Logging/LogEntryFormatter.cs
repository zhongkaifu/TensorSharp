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
using System.IO;
using System.Text;
using System.Text.Json;
using Microsoft.Extensions.Logging;

namespace TensorSharp.Runtime.Logging
{
    /// <summary>
    /// Renders <see cref="LogEntry"/> into either JSON-line or plain-text form.
    /// Used internally by <see cref="FileLoggerProvider"/>; surfaced as
    /// <c>internal</c> so the test project (granted via <c>InternalsVisibleTo</c>)
    /// can pin the on-disk shape.
    /// </summary>
    internal static class LogEntryFormatter
    {
        private static readonly JsonWriterOptions WriterOptions = new()
        {
            Indented = false,
            SkipValidation = true,
        };

        internal static string ToJsonLine(LogEntry entry, bool includeScopes)
        {
            using var stream = new MemoryStream(256);
            using (var writer = new Utf8JsonWriter(stream, WriterOptions))
            {
                writer.WriteStartObject();

                writer.WriteString("ts", entry.Timestamp.ToString("o"));
                writer.WriteString("level", entry.Level.ToString());
                writer.WriteString("category", entry.Category);

                if (entry.EventId.Id != 0 || !string.IsNullOrEmpty(entry.EventId.Name))
                {
                    writer.WritePropertyName("event");
                    writer.WriteStartObject();
                    writer.WriteNumber("id", entry.EventId.Id);
                    if (!string.IsNullOrEmpty(entry.EventId.Name))
                        writer.WriteString("name", entry.EventId.Name);
                    writer.WriteEndObject();
                }

                writer.WriteString("message", entry.Message ?? string.Empty);

                if (entry.Exception != null)
                    writer.WriteString("exception", entry.Exception);

                WriteState(writer, entry.State);

                if (includeScopes && entry.Scopes != null && entry.Scopes.Count > 0)
                    WriteScopes(writer, entry.Scopes);

                writer.WriteEndObject();
            }

            return Encoding.UTF8.GetString(stream.GetBuffer(), 0, (int)stream.Length);
        }

        internal static string ToTextLine(LogEntry entry, bool includeScopes)
        {
            var sb = new StringBuilder(192);
            sb.Append(entry.Timestamp.ToString("yyyy-MM-ddTHH:mm:ss.fffZ"));
            sb.Append(' ');
            sb.Append(LevelLabel(entry.Level));
            sb.Append(' ');
            if (entry.EventId.Id != 0)
            {
                sb.Append('[');
                sb.Append(entry.EventId.Id);
                if (!string.IsNullOrEmpty(entry.EventId.Name))
                {
                    sb.Append(':');
                    sb.Append(entry.EventId.Name);
                }
                sb.Append("] ");
            }
            sb.Append(entry.Category);
            sb.Append(": ");
            sb.Append(entry.Message);

            if (includeScopes && entry.Scopes != null && entry.Scopes.Count > 0)
                AppendScopes(sb, entry.Scopes);

            if (entry.Exception != null)
            {
                sb.AppendLine();
                sb.Append("    ");
                sb.Append(entry.Exception);
            }

            return sb.ToString();
        }

        private static void WriteState(Utf8JsonWriter writer, IReadOnlyList<KeyValuePair<string, object>> state)
        {
            if (state == null || state.Count == 0)
                return;

            // The default ILogger formatter inserts a "{OriginalFormat}" entry when a
            // structured message template is used (e.g. "Sent {Count} bytes"). It's
            // useful for downstream processing but is noisy when humans grep, so we
            // separate it under "template" and put the rest under "props".
            string template = null;
            int realCount = 0;
            for (int i = 0; i < state.Count; i++)
            {
                if (state[i].Key == "{OriginalFormat}")
                    template = state[i].Value?.ToString();
                else
                    realCount++;
            }

            if (template != null)
                writer.WriteString("template", template);

            if (realCount == 0)
                return;

            writer.WritePropertyName("props");
            writer.WriteStartObject();
            for (int i = 0; i < state.Count; i++)
            {
                var kv = state[i];
                if (kv.Key == "{OriginalFormat}")
                    continue;
                WriteProperty(writer, kv.Key, kv.Value);
            }
            writer.WriteEndObject();
        }

        private static void WriteScopes(Utf8JsonWriter writer, IReadOnlyList<KeyValuePair<string, object>> scopes)
        {
            writer.WritePropertyName("scope");
            writer.WriteStartObject();
            for (int i = 0; i < scopes.Count; i++)
            {
                var kv = scopes[i];
                if (kv.Key == "{OriginalFormat}")
                    continue;
                WriteProperty(writer, kv.Key, kv.Value);
            }
            writer.WriteEndObject();
        }

        private static void WriteProperty(Utf8JsonWriter writer, string name, object value)
        {
            writer.WritePropertyName(name);
            switch (value)
            {
                case null:
                    writer.WriteNullValue();
                    return;
                case string s:
                    writer.WriteStringValue(s);
                    return;
                case bool b:
                    writer.WriteBooleanValue(b);
                    return;
                case int i:
                    writer.WriteNumberValue(i);
                    return;
                case long l:
                    writer.WriteNumberValue(l);
                    return;
                case double d:
                    if (double.IsFinite(d))
                        writer.WriteNumberValue(d);
                    else
                        writer.WriteStringValue(d.ToString(System.Globalization.CultureInfo.InvariantCulture));
                    return;
                case float f:
                    if (float.IsFinite(f))
                        writer.WriteNumberValue(f);
                    else
                        writer.WriteStringValue(f.ToString(System.Globalization.CultureInfo.InvariantCulture));
                    return;
                case decimal m:
                    writer.WriteNumberValue(m);
                    return;
                case DateTime dt:
                    writer.WriteStringValue(dt.ToUniversalTime().ToString("o"));
                    return;
                case DateTimeOffset dto:
                    writer.WriteStringValue(dto.ToString("o"));
                    return;
                case TimeSpan ts:
                    writer.WriteStringValue(ts.ToString());
                    return;
                case Guid g:
                    writer.WriteStringValue(g.ToString());
                    return;
                default:
                    writer.WriteStringValue(value.ToString());
                    return;
            }
        }

        private static void AppendScopes(StringBuilder sb, IReadOnlyList<KeyValuePair<string, object>> scopes)
        {
            sb.Append(" =>");
            for (int i = 0; i < scopes.Count; i++)
            {
                var kv = scopes[i];
                if (kv.Key == "{OriginalFormat}")
                    continue;
                sb.Append(' ');
                sb.Append(kv.Key);
                sb.Append('=');
                sb.Append(kv.Value);
            }
        }

        private static string LevelLabel(LogLevel level) => level switch
        {
            LogLevel.Trace => "TRC",
            LogLevel.Debug => "DBG",
            LogLevel.Information => "INF",
            LogLevel.Warning => "WRN",
            LogLevel.Error => "ERR",
            LogLevel.Critical => "CRT",
            _ => "NON",
        };
    }
}
