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
using System.IO;
using System.Reflection;
using System.Text;

namespace TensorSharp.Runtime
{
    /// <summary>
    /// A model load that was declined on purpose, with a reason the operator can act on:
    /// not enough VRAM for the requested context or offload, a tensor-parallel layout
    /// the devices cannot hold, an unsupported KV cache dtype, a missing sidecar.
    /// </summary>
    /// <remarks>
    /// Derives from <see cref="InvalidOperationException"/> so code that already catches
    /// that type keeps working. Throw it where a loader DECIDES not to load, never for a
    /// bug: the hosts turn it into one error line and
    /// <see cref="HostExitCodes.ModelLoadRefused"/>, without a stack trace.
    /// </remarks>
    public sealed class ModelLoadRefusedException : InvalidOperationException
    {
        public ModelLoadRefusedException(string message)
            : base(message)
        {
        }

        public ModelLoadRefusedException(string message, Exception innerException)
            : base(message, innerException)
        {
        }
    }

    /// <summary>
    /// Process exit codes shared by <c>TensorSharp.Server</c> and <c>TensorSharp.Cli</c>.
    /// Documented in USAGE.md ("Exit codes"); scripts depend on these numbers, so they
    /// never change meaning.
    /// </summary>
    public static class HostExitCodes
    {
        /// <summary>Normal exit.</summary>
        public const int Success = 0;

        /// <summary>A command-line or configuration-file mistake (unknown flag, bad value).</summary>
        public const int ConfigurationError = 1;

        /// <summary>
        /// The model load was refused: see <see cref="ModelLoadRefusal"/> for what counts.
        /// The reason is the last line on stderr.
        /// </summary>
        public const int ModelLoadRefused = 2;
    }

    /// <summary>
    /// Tells a load REFUSAL (the operator's to fix; one line is the whole story) apart
    /// from a genuinely unexpected failure (a bug; the stack trace is the story).
    /// </summary>
    /// <remarks>
    /// <para>
    /// Refusals: <see cref="ModelLoadRefusedException"/>; <see cref="NotSupportedException"/>,
    /// which is how the architectures decline a backend, a <c>--tp</c> layout or a KV
    /// cache dtype; <see cref="IOException"/> (a missing, truncated or unreadable model,
    /// shard or sidecar); <see cref="InvalidDataException"/> (a file that is not the
    /// GGUF it claims to be); <see cref="UnauthorizedAccessException"/>.
    /// </para>
    /// <para>
    /// Everything else, <see cref="NullReferenceException"/>, a plain
    /// <see cref="InvalidOperationException"/>, an out-of-memory or a CUDA error, stays
    /// visible with its stack trace, because nothing an operator can change explains it.
    /// Only the load call itself is classified: a <see cref="NotSupportedException"/>
    /// thrown while serving is not a refusal to load.
    /// </para>
    /// </remarks>
    public static class ModelLoadRefusal
    {
        /// <summary>
        /// True when <paramref name="exception"/> (or the one exception it wraps) is a
        /// refusal; <paramref name="reason"/> is then its message flattened to one line.
        /// </summary>
        public static bool TryDescribe(Exception exception, out string reason)
        {
            reason = null;
            Exception ex = Unwrap(exception);
            if (ex == null || !IsRefusal(ex))
                return false;

            reason = ToSingleLine(ex.Message);
            if (string.IsNullOrEmpty(reason))
                reason = ex.GetType().Name;
            return true;
        }

        /// <summary>True for the exception types listed on <see cref="ModelLoadRefusal"/>.</summary>
        public static bool IsRefusal(Exception exception)
        {
            Exception ex = Unwrap(exception);
            return ex is ModelLoadRefusedException
                or NotSupportedException
                or IOException
                or InvalidDataException
                or UnauthorizedAccessException;
        }

        /// <summary>The one line a host prints to stderr before exiting with <see cref="HostExitCodes.ModelLoadRefused"/>.</summary>
        public static string FormatErrorLine(string reason)
            => "error: model load refused: " + ToSingleLine(reason);

        /// <summary>
        /// Collapse a multi-line message into one line: the error line is meant to be the
        /// LAST line of stderr, readable on its own, and greppable.
        /// </summary>
        public static string ToSingleLine(string text)
        {
            if (string.IsNullOrEmpty(text))
                return text ?? string.Empty;

            var sb = new StringBuilder(text.Length);
            bool pendingSpace = false;
            foreach (char c in text)
            {
                if (c == '\r' || c == '\n' || c == '\t')
                {
                    pendingSpace = sb.Length > 0;
                    continue;
                }
                if (pendingSpace)
                {
                    if (sb[sb.Length - 1] != ' ' && c != ' ')
                        sb.Append(' ');
                    pendingSpace = false;
                }
                sb.Append(c);
            }
            return sb.ToString().Trim();
        }

        /// <summary>
        /// Wrappers that only carry another exception: reflection-invoked factories, a
        /// single-exception aggregate from a task, a static constructor.
        /// </summary>
        private static Exception Unwrap(Exception exception)
        {
            Exception ex = exception;
            for (int depth = 0; ex != null && depth < 8; depth++)
            {
                Exception inner = ex switch
                {
                    TargetInvocationException tie => tie.InnerException,
                    TypeInitializationException tie => tie.InnerException,
                    AggregateException { InnerExceptions.Count: 1 } ae => ae.InnerExceptions[0],
                    _ => null,
                };
                if (inner == null)
                    break;
                ex = inner;
            }
            return ex;
        }
    }
}
