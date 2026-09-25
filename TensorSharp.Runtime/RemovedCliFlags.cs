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

namespace TensorSharp.Runtime
{
    /// <summary>
    /// Host options whose feature was removed outright, shared by <c>TensorSharp.Cli</c>
    /// and <c>TensorSharp.Server</c>. The speculative and code-execution families keep
    /// their own tables (their removed spellings have a survivor to point at); these do
    /// not, so each entry carries a whole sentence of advice instead of a flag name.
    /// </summary>
    /// <remarks>
    /// <para>
    /// A hard error, never a silent ignore. The CLI's argument switch drops flags it has
    /// no case for, so without this a retired flag in a script would simply stop doing
    /// anything; the server's unknown-option trap would refuse it, but with a bare
    /// "Unknown option" and no word about why or what to do instead.
    /// </para>
    /// <para>
    /// Both hosts call <see cref="RejectRemoved"/> before any other option is applied,
    /// and <see cref="ConfigFileArgs"/> refuses the same names as <c>--config</c> keys
    /// before it resolves (or downloads) anything the file names.
    /// </para>
    /// </remarks>
    public static class RemovedCliFlags
    {
        /// <summary>
        /// The removed flags, each with what the operator should know instead. Both
        /// usage pages list these under "Removed options" and never as live options.
        /// </summary>
        public static readonly IReadOnlyList<(string Flag, string Advice)> RemovedFlags = new[]
        {
            ("--qwen-image-lora",
                "it only applied to the retired Qwen-Image-Edit-2511 pipeline. "
                + "Qwen-Image-2.1 takes LoRA plug-ins with --lora <file> (plus --lora-scale and --lora-config)."),
            ("--offload-cpu",
                "it only applied to the retired Qwen-Image-Edit-2511 DiT. "
                + "Qwen-Image-2.1 keeps its weights resident; use smaller --width/--height if memory is short."),
        };

        /// <summary>
        /// The error for <paramref name="arg"/> when it names a removed flag, in either the
        /// spaced (<c>--flag value</c>) or the joined (<c>--flag=value</c>) spelling,
        /// case-insensitively; null otherwise. A bare config-file key (no leading dashes)
        /// is matched too, so the file and the command line share one message.
        /// </summary>
        public static string? Describe(string? arg)
        {
            if (string.IsNullOrWhiteSpace(arg))
                return null;

            string name = arg.Trim();
            int equals = name.IndexOf('=');
            if (equals >= 0)
                name = name.Substring(0, equals);
            if (!name.StartsWith("--", StringComparison.Ordinal))
                name = "--" + name;

            foreach ((string flag, string advice) in RemovedFlags)
            {
                if (name.Equals(flag, StringComparison.OrdinalIgnoreCase))
                    return $"{flag} was removed: {advice}";
            }
            return null;
        }

        /// <summary>
        /// Throw for the first removed flag in <paramref name="args"/>. Only tokens that
        /// start with <c>--</c> are checked, so a plain value such as a prompt reading
        /// "offload-cpu" is left alone.
        /// </summary>
        /// <exception cref="ArgumentException">A removed flag was present.</exception>
        public static void RejectRemoved(IReadOnlyList<string>? args)
        {
            if (args == null)
                return;
            foreach (string arg in args)
            {
                if (arg == null || !arg.StartsWith("--", StringComparison.Ordinal))
                    continue;
                if (Describe(arg) is { } message)
                    throw new ArgumentException(message);
            }
        }
    }
}
