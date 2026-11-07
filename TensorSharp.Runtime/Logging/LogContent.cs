// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

namespace TensorSharp.Runtime.Logging
{
    /// <summary>
    /// Marks a log argument as user content: a prompt, a chat message, model output, an
    /// uploaded file's name, path or result URL, a tool-call argument, a command. Wrap the
    /// argument at the call site:
    /// <code>logger.LogInformation("chat.start ... userInput=\"{LastUserContent}\"", LogContent.Of(preview));</code>
    ///
    /// <para>
    /// Nothing changes locally: <see cref="ToString"/> returns the text, so the console and
    /// the JSON-line file logger write exactly what they wrote before. The OpenTelemetry log
    /// pipeline (TensorSharp.Telemetry) replaces every attribute whose value is a
    /// <see cref="LogContent"/> with <c>[redacted: N chars]</c> before export, unless
    /// <c>OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true</c>.
    /// </para>
    /// <para>
    /// Content is marked where it is logged, not recognised by its placeholder name, because
    /// names do not say what a value is: <c>{Prompt}</c> is a prompt in one line and a token
    /// count in another, and <c>{Path}</c> is an HTTP route in one and an uploaded file in the
    /// next.
    /// </para>
    /// </summary>
    public sealed class LogContent
    {
        private LogContent(string text) => Text = text;

        /// <summary>The content, as it is written to local logs.</summary>
        public string Text { get; }

        /// <summary>
        /// Wrap <paramref name="text"/>; <c>null</c> stays <c>null</c>, so a missing value is
        /// logged exactly as before.
        /// </summary>
        public static LogContent? Of(string? text) => text is null ? null : new LogContent(text);

        /// <summary>Wrap any value's string form (an upload manifest, a path object).</summary>
        public static LogContent? Of(object? value) => value switch
        {
            null => null,
            LogContent already => already,
            string s => new LogContent(s),
            _ => value.ToString() is { } rendered ? new LogContent(rendered) : null,
        };

        public override string ToString() => Text;
    }
}
