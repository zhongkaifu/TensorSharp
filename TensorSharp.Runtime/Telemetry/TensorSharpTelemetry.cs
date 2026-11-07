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
using System.Diagnostics;
using System.Diagnostics.Metrics;
using System.Reflection;
using System.Threading;

namespace TensorSharp.Runtime.Telemetry
{
    /// <summary>
    /// The activity sources and meter TensorSharp's libraries record traces and metrics on.
    ///
    /// <para>
    /// Only the in-box <see cref="System.Diagnostics"/> types are used here, so a library
    /// that instruments itself takes no package dependency, and a process that never turns
    /// telemetry on pays nothing: with no listener, <see cref="ActivitySource.StartActivity(string, ActivityKind)"/>
    /// returns <c>null</c> and every instrument drops its measurements. The OpenTelemetry SDK
    /// that listens lives in the host-only <c>TensorSharp.Telemetry</c> project, which
    /// subscribes to every source and meter whose name starts with <see cref="NamePrefix"/>.
    /// </para>
    /// <para>
    /// <c>TensorSharp.Studio</c> has no project references, so it declares its own source and
    /// meter under <see cref="StudioSourceName"/> / <see cref="StudioMeterName"/>; they are
    /// matched by name, not by type.
    /// </para>
    /// </summary>
    public static class TensorSharpTelemetry
    {
        /// <summary>Every TensorSharp source and meter name starts with this.</summary>
        public const string NamePrefix = "TensorSharp";

        public const string InferenceSourceName = "TensorSharp.Inference";
        public const string ModelsSourceName = "TensorSharp.Models";
        public const string MediaSourceName = "TensorSharp.Media";
        public const string AgentSourceName = "TensorSharp.Agent";
        public const string StudioSourceName = "TensorSharp.Studio";

        public const string MeterName = "TensorSharp";
        public const string StudioMeterName = "TensorSharp.Studio";

        /// <summary>
        /// Opt-in for exporting prompt and output text. The same variable GenAI
        /// instrumentations in other ecosystems read; off unless set to a true value.
        /// </summary>
        public const string CaptureMessageContentEnvironmentVariable = "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT";

        /// <summary>Opt-in for one span per engine step, for profiling a single run.</summary>
        public const string TraceStepsEnvironmentVariable = "TENSORSHARP_OTEL_TRACE_STEPS";

        /// <summary>The value of <c>gen_ai.provider.name</c> on TensorSharp spans and metrics.</summary>
        public const string ProviderName = "tensorsharp";

        public static string Version { get; } = ResolveVersion();

        /// <summary>Chat, completion and embedding requests, and the engine's phases.</summary>
        public static ActivitySource Inference { get; } = new(InferenceSourceName, Version);

        /// <summary>Model load and unload.</summary>
        public static ActivitySource Models { get; } = new(ModelsSourceName, Version);

        /// <summary>Image and video generation, and media export.</summary>
        public static ActivitySource Media { get; } = new(MediaSourceName, Version);

        /// <summary>Skills, tool rounds and code execution.</summary>
        public static ActivitySource Agent { get; } = new(AgentSourceName, Version);

        public static Meter Meter { get; } = new(MeterName, Version);

        // -1: read the environment; 0 / 1: set by a host or a test.
        private static int s_captureMessageContent = -1;
        private static int s_traceSteps = -1;

        /// <summary>
        /// Whether prompt and output text may be put on spans. Off by default; see
        /// <see cref="CaptureMessageContentEnvironmentVariable"/>. The host's telemetry setup
        /// pins it with <see cref="ConfigureCaptureMessageContent"/> so the span side and the
        /// log redaction side always agree.
        /// </summary>
        public static bool CaptureMessageContent => Resolve(ref s_captureMessageContent, CaptureMessageContentEnvironmentVariable);

        /// <summary>Whether the engine emits one span per step. See <see cref="TraceStepsEnvironmentVariable"/>.</summary>
        public static bool TraceSteps => Resolve(ref s_traceSteps, TraceStepsEnvironmentVariable);

        /// <summary>Pin <see cref="CaptureMessageContent"/>; <c>null</c> goes back to reading the environment.</summary>
        public static void ConfigureCaptureMessageContent(bool? enabled) =>
            Volatile.Write(ref s_captureMessageContent, enabled is null ? -1 : enabled.Value ? 1 : 0);

        /// <summary>Pin <see cref="TraceSteps"/>; <c>null</c> goes back to reading the environment.</summary>
        public static void ConfigureTraceSteps(bool? enabled) =>
            Volatile.Write(ref s_traceSteps, enabled is null ? -1 : enabled.Value ? 1 : 0);

        private static volatile bool s_logsExported;

        /// <summary>
        /// Whether this process exports logs through OpenTelemetry. Set by the host's telemetry
        /// setup. Log lines that exist only for an exported log (the video helper's one line per
        /// phase) are written only when this is true, so the local console and file logs read
        /// as they always have.
        /// </summary>
        public static bool LogsExported => s_logsExported;

        /// <summary>Set by the host's telemetry setup; see <see cref="LogsExported"/>.</summary>
        public static void ConfigureLogsExported(bool exported) => s_logsExported = exported;

        private static readonly object s_sensitiveGate = new();
        private static string[] s_sensitivePaths = Array.Empty<string>();

        /// <summary>
        /// A directory whose paths count as user content (the upload directory): exported
        /// exception messages have it replaced with <c>&lt;uploads&gt;</c>. Idempotent.
        /// </summary>
        public static void RegisterSensitivePath(string? directory)
        {
            if (string.IsNullOrWhiteSpace(directory))
                return;
            string full;
            try { full = System.IO.Path.GetFullPath(directory).TrimEnd('\\', '/'); }
            catch (Exception) { return; }
            if (full.Length == 0)
                return;
            lock (s_sensitiveGate)
            {
                foreach (string existing in s_sensitivePaths)
                {
                    if (string.Equals(existing, full, StringComparison.OrdinalIgnoreCase))
                        return;
                }
                var next = new string[s_sensitivePaths.Length + 1];
                s_sensitivePaths.CopyTo(next, 0);
                next[^1] = full;
                s_sensitivePaths = next;
            }
        }

        /// <summary>
        /// <paramref name="text"/> with every path under a registered sensitive directory
        /// replaced whole by <c>&lt;uploads&gt;</c>, <c>/uploads/&lt;name&gt;</c> URLs by
        /// <c>/uploads/{file}</c>, and the 32 hex digits of an upload storage name
        /// (<c>&lt;32 hex&gt;.&lt;ext&gt;</c>, also inside a name such as <c>video-&lt;32 hex&gt;.mp4</c>)
        /// by <c>{id}</c>. A path is replaced through its end: to the closing quote when it is
        /// quoted (so a file name with spaces goes too), otherwise to the next whitespace. The
        /// directory alone is not enough: below it are the names that matter, an upload's
        /// unguessable name (the only thing guarding its URL), a code-execution artifact's run
        /// id, a workspace file's own name. Returns the same instance when nothing matched.
        /// </summary>
        public static string ScrubSensitivePaths(string text)
        {
            if (string.IsNullOrEmpty(text))
                return text;
            string result = text;
            foreach (string path in Volatile.Read(ref s_sensitivePaths))
            {
                result = ReplacePathsUnder(result, path, "<uploads>");
                string forward = path.Replace('\\', '/');
                if (!string.Equals(forward, path, StringComparison.Ordinal))
                    result = ReplacePathsUnder(result, forward, "<uploads>");
            }
            result = ReplacePathsUnder(result, "/uploads/", "/uploads/{file}");
            result = s_storageName.Replace(result, "{id}");
            return string.Equals(result, text, StringComparison.Ordinal) ? text : result;
        }

        // The 32 hex digits of a storage name: an upload (Guid "N" format plus its extension) or
        // a generated file named after one. A bare 32-digit id with no extension (a trace id) is
        // left alone.
        private static readonly System.Text.RegularExpressions.Regex s_storageName = new(
            "(?<![0-9A-Fa-f])[0-9A-Fa-f]{32}(?=\\.[A-Za-z0-9]{1,8}(?![A-Za-z0-9]))",
            System.Text.RegularExpressions.RegexOptions.CultureInvariant);

        /// <summary>
        /// Replace each occurrence of <paramref name="prefix"/> (ignoring case) and the rest of
        /// the path that follows it with <paramref name="replacement"/>.
        /// </summary>
        private static string ReplacePathsUnder(string text, string prefix, string replacement)
        {
            int index = text.IndexOf(prefix, StringComparison.OrdinalIgnoreCase);
            if (index < 0)
                return text;
            var sb = new System.Text.StringBuilder(text.Length);
            int start = 0;
            while (index >= 0)
            {
                // A quoted path ends at its closing quote, spaces and all; an unquoted one at
                // the first character that cannot continue a path in a message.
                char? quote = FindOpeningQuote(text, start, index);
                int end = index + prefix.Length;
                if (quote is char q)
                {
                    int close = text.IndexOf(q, end);
                    end = close < 0 ? text.Length : close;
                }
                else
                {
                    while (end < text.Length && !char.IsWhiteSpace(text[end]) && text[end] is not ('"' or '\'' or ')' or ']' or ',' or ';' or '<' or '>'))
                        end++;
                    // A sentence's full stop is not part of the path.
                    if (end > index + prefix.Length && text[end - 1] == '.')
                        end--;
                }
                sb.Append(text, start, index - start).Append(replacement);
                start = end;
                index = start < text.Length ? text.IndexOf(prefix, start, StringComparison.OrdinalIgnoreCase) : -1;
            }
            sb.Append(text, start, text.Length - start);
            return sb.ToString();
        }

        // The quote that opens the path at index (a path quoted as '...' or "...", possibly with a
        // drive or root before the prefix), or null when the path is not quoted.
        private static char? FindOpeningQuote(string text, int from, int index)
        {
            for (int i = index - 1; i >= from; i--)
            {
                char c = text[i];
                if (c is '\'' or '"')
                    return c;
                if (char.IsWhiteSpace(c) || c is '(' or '[' or ',' or ';' or '=')
                    return null;
            }
            return null;
        }

        /// <summary>
        /// True for <c>1</c>, <c>true</c>, <c>yes</c> and <c>on</c> (any case, surrounding
        /// whitespace ignored), the spellings TensorSharp's other switches accept.
        /// </summary>
        public static bool IsTruthy(string? value)
        {
            if (string.IsNullOrWhiteSpace(value))
                return false;
            string v = value.Trim();
            return v == "1"
                || v.Equals("true", StringComparison.OrdinalIgnoreCase)
                || v.Equals("yes", StringComparison.OrdinalIgnoreCase)
                || v.Equals("on", StringComparison.OrdinalIgnoreCase);
        }

        /// <summary>
        /// A <see cref="DateTime"/> recorded with <c>DateTime.UtcNow</c> (as the scheduler's
        /// timestamps are) as a <see cref="DateTimeOffset"/> for an explicit span start or end.
        /// A local or unspecified time is treated as what it says it is.
        /// </summary>
        public static DateTimeOffset ToUtcOffset(DateTime time) => time.Kind switch
        {
            DateTimeKind.Utc => new DateTimeOffset(time, TimeSpan.Zero),
            DateTimeKind.Local => new DateTimeOffset(time.ToUniversalTime(), TimeSpan.Zero),
            _ => new DateTimeOffset(DateTime.SpecifyKind(time, DateTimeKind.Utc), TimeSpan.Zero),
        };

        private static bool Resolve(ref int field, string environmentVariable)
        {
            int state = Volatile.Read(ref field);
            if (state >= 0)
                return state == 1;
            return IsTruthy(Environment.GetEnvironmentVariable(environmentVariable));
        }

        private static string ResolveVersion()
        {
            var assembly = typeof(TensorSharpTelemetry).Assembly;
            string? informational = assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion;
            if (!string.IsNullOrEmpty(informational))
            {
                // Drop the "+<commit>" build metadata Source Link appends; the version is
                // what dashboards group by, the commit belongs in a resource attribute.
                int plus = informational.IndexOf('+');
                return plus > 0 ? informational[..plus] : informational;
            }
            return assembly.GetName().Version?.ToString() ?? "0.0.0";
        }
    }
}
