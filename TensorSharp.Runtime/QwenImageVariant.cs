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
    /// Which Qwen-Image-2.1 checkpoint a diffusion-transformer GGUF holds. The two have the same
    /// tensors, names and shapes, and the community GGUFs carry no metadata at all, so nothing in
    /// the file tells them apart; the host declares it (<see cref="QwenImageVariantFlag"/>).
    /// </summary>
    public enum QwenImageVariant
    {
        /// <summary>Qwen/Qwen-Image-2.1: 40 Euler steps on the scheduler's resolution-dependent
        /// shifted schedule, CFG 1.</summary>
        Base,
        /// <summary>Qwen/Qwen-Image-2.1-Turbo: the same transformer distilled to 8 Euler steps at
        /// CFG 1 on one fixed schedule (its <c>model_index.json</c> <c>sample_sigmas</c>).</summary>
        Turbo,
    }

    /// <summary>
    /// The <c>--qwen-image-variant</c> option shared by <c>TensorSharp.Cli</c> and
    /// <c>TensorSharp.Server</c> (a config file's <c>"qwen-image-variant"</c> key), and the
    /// environment channel (<see cref="EnvironmentVariable"/>) through which both hosts and the
    /// TensorAgent catalog hand the declaration to the image model.
    /// </summary>
    /// <remarks>
    /// Without a declaration the model reads the file name: a name with the word <c>turbo</c> is
    /// taken for Qwen-Image-2.1-Turbo and the load prints that it assumed so. That is a guess, not
    /// an identification (merges of step-distillation LoRAs into the base checkpoint are published
    /// under the very same names), which is why every configuration TensorSharp ships declares it.
    /// </remarks>
    public static class QwenImageVariantFlag
    {
        public const string Flag = "--qwen-image-variant";

        /// <summary>Read by <c>QwenImageModel</c> at load: <c>base</c> or <c>turbo</c>.</summary>
        public const string EnvironmentVariable = "TS_QWEN_IMAGE_VARIANT";

        /// <summary>The accepted values, in the spelling the usage pages show.</summary>
        public static readonly IReadOnlyList<string> Values = new[] { "base", "turbo" };

        /// <summary>The value's spelling for <paramref name="variant"/>.</summary>
        public static string Name(QwenImageVariant variant) => variant switch
        {
            QwenImageVariant.Base => "base",
            QwenImageVariant.Turbo => "turbo",
            _ => throw new ArgumentOutOfRangeException(nameof(variant)),
        };

        /// <summary>
        /// Take the declaration out of <paramref name="args"/>: <c>--qwen-image-variant VALUE</c> or
        /// <c>--qwen-image-variant=VALUE</c>, matched case-insensitively like the server's option
        /// parser, the last one winning. The other arguments are copied to <paramref name="remaining"/>.
        /// For the CLI, whose own switch ignores what it does not know: a spelling the server accepts
        /// is never dropped there.
        /// </summary>
        /// <returns>The declared variant, or null when none is given.</returns>
        /// <exception cref="ArgumentException">The flag has no value, or the value is not a variant.</exception>
        public static QwenImageVariant? Take(IReadOnlyList<string> args, List<string> remaining)
        {
            ArgumentNullException.ThrowIfNull(args);
            ArgumentNullException.ThrowIfNull(remaining);
            QwenImageVariant? result = null;
            for (int i = 0; i < args.Count; i++)
            {
                string arg = args[i];
                string? value;
                if (string.Equals(arg, Flag, StringComparison.OrdinalIgnoreCase))
                {
                    if (i + 1 >= args.Count) throw new ArgumentException($"{Flag} requires a value ({string.Join(" or ", Values)}).");
                    value = args[++i];
                }
                else if (arg.StartsWith(Flag + "=", StringComparison.OrdinalIgnoreCase))
                {
                    value = arg.Substring(Flag.Length + 1);
                }
                else
                {
                    remaining.Add(arg);
                    continue;
                }
                result = Parse(value, Flag);
            }
            return result;
        }

        /// <summary>Parse a declared value (case-insensitive, surrounding blanks ignored).</summary>
        /// <param name="source">Where it came from, for the error (the flag or the variable).</param>
        /// <exception cref="ArgumentException">The value is not one of <see cref="Values"/>.</exception>
        public static QwenImageVariant Parse(string? value, string source)
        {
            switch (value?.Trim().ToLowerInvariant())
            {
                case "base": return QwenImageVariant.Base;
                case "turbo": return QwenImageVariant.Turbo;
                default:
                    throw new ArgumentException(
                        $"{source} expects one of {string.Join(", ", Values)} (which Qwen-Image-2.1 checkpoint the GGUF holds), not '{value}'.");
            }
        }
    }
}
