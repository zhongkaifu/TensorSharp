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
using System.Diagnostics;
using System.Globalization;
using System.Text;
using Microsoft.Extensions.Logging;

namespace TensorSharp.Runtime.Logging
{
    /// <summary>
    /// Convenience extensions for wiring TensorSharp's file logger into an
    /// <see cref="ILoggingBuilder"/> and for emitting common operational events
    /// in a consistent shape.
    /// </summary>
    public static class LoggingExtensions
    {
        /// <summary>
        /// Register a JSON-line file logger that buffers and flushes from a
        /// background thread. Multiple registrations against the same builder
        /// are permitted; callers usually combine this with the built-in console
        /// logger (<c>builder.AddConsole()</c>).
        /// </summary>
        public static ILoggingBuilder AddTensorSharpFileLogger(this ILoggingBuilder builder, Action<FileLoggerOptions> configure = null)
        {
            if (builder == null)
                throw new ArgumentNullException(nameof(builder));

            var options = new FileLoggerOptions();
            configure?.Invoke(options);
            options.Validate();

            var provider = new FileLoggerProvider(options);
            builder.AddProvider(provider);
            return builder;
        }

        /// <summary>
        /// Begin a structured scope that includes a single key/value. Returns a no-op
        /// disposable when <paramref name="value"/> is null/empty so call sites can
        /// be uniform without sprinkling null-checks everywhere.
        /// </summary>
        public static IDisposable BeginValueScope(this ILogger logger, string key, string value)
        {
            if (logger == null || string.IsNullOrEmpty(value))
                return NullScope.Instance;
            return logger.BeginScope(new Dictionary<string, object>(StringComparer.Ordinal)
            {
                [key] = value,
            }) ?? NullScope.Instance;
        }

        /// <summary>
        /// Begin a structured scope that includes multiple key/value pairs. The
        /// dictionary is constructed eagerly so any mutation by the caller after
        /// the call has no effect on the captured scope.
        /// </summary>
        public static IDisposable BeginScopedProperties(this ILogger logger, params (string key, object value)[] properties)
        {
            if (logger == null || properties == null || properties.Length == 0)
                return NullScope.Instance;

            var dict = new Dictionary<string, object>(properties.Length, StringComparer.Ordinal);
            foreach (var (key, value) in properties)
            {
                if (!string.IsNullOrEmpty(key))
                    dict[key] = value;
            }
            return logger.BeginScope(dict) ?? NullScope.Instance;
        }

        /// <summary>
        /// Returns a substring suitable for inclusion in a log entry: trimmed,
        /// with control characters replaced and a length cap to prevent
        /// runaway log volume from large prompts/responses.
        /// </summary>
        public static string SanitizeForLog(string value, int maxLength = 240)
        {
            if (string.IsNullOrEmpty(value))
                return string.Empty;
            if (maxLength <= 0)
                return string.Empty;

            int length = Math.Min(value.Length, maxLength);
            var sb = new StringBuilder(length + 16);
            AppendEscaped(sb, value, length);
            if (value.Length > length)
                sb.Append("...(+").Append((value.Length - length).ToString(CultureInfo.InvariantCulture)).Append(" chars)");
            return sb.ToString();
        }

        /// <summary>
        /// Same control-character escaping as <see cref="SanitizeForLog"/> but
        /// never truncates. Use when the caller deliberately wants the full
        /// user input or assistant output captured in the log (e.g. for
        /// audit trails or when reproducing model behaviour from logs).
        /// Each newline / carriage return / tab is still escaped so the entry
        /// remains on a single log line.
        /// </summary>
        public static string SanitizeForLogFull(string value)
        {
            if (string.IsNullOrEmpty(value))
                return string.Empty;

            int length = value.Length;
            int initialCapacity = length > int.MaxValue - 16 ? length : length + 16;
            var sb = new StringBuilder(initialCapacity);
            AppendEscaped(sb, value, length);
            return sb.ToString();
        }

        private static void AppendEscaped(StringBuilder sb, string value, int length)
        {
            for (int i = 0; i < length; i++)
            {
                char c = value[i];
                if (c == '\n')
                    sb.Append("\\n");
                else if (c == '\r')
                    sb.Append("\\r");
                else if (c == '\t')
                    sb.Append("\\t");
                else if (c < 0x20)
                    sb.Append('?');
                else
                    sb.Append(c);
            }
        }

        /// <summary>
        /// Format a non-negative byte count as a human-friendly string.
        /// </summary>
        public static string FormatBytes(long bytes)
        {
            if (bytes < 0) bytes = 0;
            string[] units = { "B", "KB", "MB", "GB", "TB" };
            double v = bytes;
            int unit = 0;
            while (v >= 1024 && unit < units.Length - 1)
            {
                v /= 1024;
                unit++;
            }
            return unit == 0
                ? $"{(long)v} {units[unit]}"
                : $"{v.ToString("F2", CultureInfo.InvariantCulture)} {units[unit]}";
        }

        /// <summary>
        /// Time the execution of <paramref name="action"/> and emit a single
        /// structured "completed" log entry that records the duration. If the
        /// action throws, an error entry is logged before re-throwing.
        /// </summary>
        public static T LogTimedOperation<T>(this ILogger logger, EventId eventId, string operationName, Func<T> action)
        {
            if (logger == null) throw new ArgumentNullException(nameof(logger));
            if (action == null) throw new ArgumentNullException(nameof(action));

            var sw = Stopwatch.StartNew();
            try
            {
                T result = action();
                sw.Stop();
                logger.Log(LogLevel.Information, eventId,
                    "{Operation} completed in {ElapsedMs:F1} ms",
                    operationName, sw.Elapsed.TotalMilliseconds);
                return result;
            }
            catch (Exception ex)
            {
                sw.Stop();
                logger.Log(LogLevel.Error, eventId, ex,
                    "{Operation} failed after {ElapsedMs:F1} ms",
                    operationName, sw.Elapsed.TotalMilliseconds);
                throw;
            }
        }

        /// <inheritdoc cref="LogTimedOperation{T}(ILogger, EventId, string, Func{T})"/>
        public static void LogTimedOperation(this ILogger logger, EventId eventId, string operationName, Action action)
        {
            if (logger == null) throw new ArgumentNullException(nameof(logger));
            if (action == null) throw new ArgumentNullException(nameof(action));
            logger.LogTimedOperation<object>(eventId, operationName, () => { action(); return null; });
        }

        private sealed class NullScope : IDisposable
        {
            public static readonly NullScope Instance = new();
            public void Dispose() { }
        }
    }
}
