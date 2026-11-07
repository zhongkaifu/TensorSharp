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
using OpenTelemetry;
using OpenTelemetry.Logs;
using TensorSharp.Runtime.Logging;
using TensorSharp.Runtime.Telemetry;

namespace TensorSharp.Telemetry
{
    /// <summary>
    /// Keeps prompts, model output, upload names and paths, and tool arguments out of
    /// exported logs.
    ///
    /// <para>
    /// Call sites mark content by wrapping the argument in <see cref="LogContent"/>; this
    /// processor replaces every such attribute, whatever its key, with
    /// <c>[redacted: N chars]</c>, and clears the pre-rendered message so the exported body is
    /// the message template. The text of each marked value is also removed from the record's
    /// other string attributes and from its exception's message: a call site that marks a path
    /// and also logs <c>ex.Message</c>, or attaches the exception, would otherwise export the
    /// same path unmarked. Exception messages are otherwise kept, except that paths under a
    /// directory registered with <see cref="TensorSharpTelemetry.RegisterSensitivePath"/> (the
    /// upload directory, the code-execution workspaces), <c>/uploads/</c> URLs and upload
    /// storage names are replaced, because media exceptions quote the file they failed on and an
    /// upload's GUID name is the only thing protecting it.
    /// </para>
    /// <para>
    /// It runs only in the OpenTelemetry pipeline, registered before every exporter (log
    /// processors run in the order they are added, and the Azure Monitor exporter's Live
    /// Metrics processor rides on its exporter), so console and file logs are untouched. It is
    /// left out when <c>OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true</c>.
    /// </para>
    /// </summary>
    public sealed class ContentRedactionProcessor : BaseProcessor<LogRecord>
    {
        /// <summary>
        /// Marked text shorter than this is not searched for elsewhere in the record: a one- or
        /// two-character value would match unrelated text, and says nothing on its own.
        /// </summary>
        internal const int MinimumSearchedLength = 4;

        internal const string RemovedText = "[redacted]";

        // Microsoft.Extensions.Logging's key for the message template among the attributes.
        private const string OriginalFormatKey = "{OriginalFormat}";

        public override void OnEnd(LogRecord data)
        {
            List<string>? marked = RedactAttributes(data);
            if (data.Exception is { } exception && ScrubbedException.Scrub(exception, marked) is { } scrubbed)
                data.Exception = scrubbed;
        }

        /// <summary>Redact the marked attributes; returns the marked texts worth searching for, or null.</summary>
        private static List<string>? RedactAttributes(LogRecord data)
        {
            IReadOnlyList<KeyValuePair<string, object?>>? attributes = data.Attributes;
            if (attributes is null || attributes.Count == 0)
                return null;

            List<string>? marked = null;
            for (int i = 0; i < attributes.Count; i++)
            {
                if (attributes[i].Value is LogContent content)
                {
                    marked ??= new List<string>(2);
                    if (content.Text.Length >= MinimumSearchedLength)
                        marked.Add(content.Text);
                }
            }
            if (marked is null)
                return null;

            var redacted = new List<KeyValuePair<string, object?>>(attributes.Count);
            for (int i = 0; i < attributes.Count; i++)
            {
                var pair = attributes[i];
                if (pair.Value is LogContent content)
                    redacted.Add(new KeyValuePair<string, object?>(pair.Key, Redact(content)));
                // The message template is code, never content: it is the exported body.
                else if (pair.Key != OriginalFormatKey && pair.Value is string text && RemoveMarked(text, marked) is { } cleaned)
                    redacted.Add(new KeyValuePair<string, object?>(pair.Key, cleaned));
                else
                    redacted.Add(pair);
            }
            data.Attributes = redacted;
            // The rendered message has the content in it; without it, exporters send the
            // message template as the body.
            data.FormattedMessage = null;
            return marked.Count > 0 ? marked : null;
        }

        internal static string Redact(LogContent content) =>
            "[redacted: " + content.Text.Length.ToString(CultureInfo.InvariantCulture) + " chars]";

        /// <summary>
        /// <paramref name="text"/> with every occurrence of a marked text replaced by
        /// <see cref="RemovedText"/> (ignoring case, as paths on Windows are), or null when none occurs.
        /// </summary>
        internal static string? RemoveMarked(string text, IReadOnlyList<string>? marked)
        {
            if (marked is null || string.IsNullOrEmpty(text))
                return null;
            string result = text;
            foreach (string value in marked)
            {
                if (result.Contains(value, StringComparison.OrdinalIgnoreCase))
                    result = result.Replace(value, RemovedText, StringComparison.OrdinalIgnoreCase);
            }
            return ReferenceEquals(result, text) ? null : result;
        }
    }

    /// <summary>
    /// Removes content attributes from spans when content capture is off: a backstop for a
    /// call site that set <c>gen_ai.input.messages</c>, <c>gen_ai.output.messages</c> or the video
    /// prompt without checking <see cref="TensorSharpTelemetry.CaptureMessageContent"/>.
    /// Registered before the trace exporters.
    /// </summary>
    public sealed class ContentSpanProcessor : BaseProcessor<Activity>
    {
        internal static readonly string[] ContentAttributes =
        {
            TelemetryAttributes.GenAiInputMessages,
            TelemetryAttributes.GenAiOutputMessages,
            TelemetryAttributes.VideoPrompt,
        };

        public override void OnEnd(Activity data)
        {
            foreach (string key in ContentAttributes)
            {
                if (data.GetTagItem(key) is not null)
                    data.SetTag(key, null);
            }
        }
    }

    /// <summary>
    /// An exception whose message had content or a sensitive path in it, exported in its place.
    /// Exporters report its own type (<c>TensorSharp.Telemetry.ScrubbedException</c>), so its
    /// message starts with the original exception's full type name, and it keeps the original
    /// stack trace: still diagnosable.
    /// </summary>
    public sealed class ScrubbedException : Exception
    {
        private readonly string? _stackTrace;

        private ScrubbedException(Exception original, string originalType, string message)
            : base(originalType + ": " + message)
        {
            OriginalType = originalType;
            _stackTrace = original.StackTrace;
        }

        /// <summary>The original exception's full type name, which also opens <see cref="Exception.Message"/>.</summary>
        public string OriginalType { get; }

        public override string? StackTrace => _stackTrace;

        // The message already names the original type; this is what OTLP exports as the stack.
        public override string ToString() =>
            _stackTrace is null ? Message : Message + Environment.NewLine + _stackTrace;

        /// <summary>
        /// A scrubbed copy, or null when the message (and inner messages) mention neither a
        /// sensitive path nor any of <paramref name="marked"/> (the texts the record marked as content).
        /// </summary>
        internal static Exception? Scrub(Exception exception, IReadOnlyList<string>? marked = null)
        {
            string full = Describe(exception);
            string cleaned = TensorSharpTelemetry.ScrubSensitivePaths(full);
            cleaned = ContentRedactionProcessor.RemoveMarked(cleaned, marked) ?? cleaned;
            if (string.Equals(cleaned, full, StringComparison.Ordinal))
                return null;
            return new ScrubbedException(exception, exception.GetType().FullName ?? exception.GetType().Name, cleaned);
        }

        private static string Describe(Exception exception)
        {
            // The outer message plus inner messages, which media code chains.
            string message = exception.Message;
            for (Exception? inner = exception.InnerException; inner != null; inner = inner.InnerException)
                message += " ---> " + (inner.GetType().FullName ?? inner.GetType().Name) + ": " + inner.Message;
            return message;
        }
    }
}
