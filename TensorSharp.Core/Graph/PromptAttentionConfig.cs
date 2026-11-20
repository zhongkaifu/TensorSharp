// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;

namespace TensorSharp.Graph
{
    /// <summary>The arithmetic a prompt chunk's attention runs on <c>--backend cuda</c> (the CUDA executor). Decode rows, verify
    /// windows and drafts never take it: they keep the 16-bit kernels whatever is chosen.</summary>
    public enum PromptAttentionPath
    {
        /// <summary><c>f16</c>, the default: the 16-bit-run prompt kernels (ts_flash_prompt_*), each 16-dim step of Q K^T and
        /// each 32-key run of P V rounded to F16 before an F32 add.</summary>
        F16,

        /// <summary><c>f32acc</c>: the prompt kernels with every tensor-core product accumulated in F32 (ts_flash_prompt2_*,
        /// flash prefill v13). Faster where the tensor cores accumulate F32 at full rate (workstation and datacenter parts);
        /// failed issue #376's pre-registered quality gates on long C# code windows, so not the default.</summary>
        F32Acc,

        /// <summary><c>int8</c>: Q K^T on the INT8 tensor cores and P V on the block-scaled E4M3 ones (ts_flash_prompt_i8e4_d256),
        /// for 256-wide heads over long windows, where the loaded image carries the kernels (an sm_120a / sm_121a build).
        /// Failed issue #376's pre-registered quality gates on long C# code windows, so not the default.</summary>
        Int8,
    }

    /// <summary>
    /// The process-wide prompt attention choice (issue #383): <c>--prompt-attention &lt;f16|f32acc|int8&gt;</c> on the CLI and
    /// the server (and so in a <c>--config</c> file), or <c>TS_PROMPT_ATTENTION</c> in the environment; the flag wins. The
    /// CUDA executor reads it when it is made. The default stays <see cref="PromptAttentionPath.F16"/>: the faster paths are
    /// opt-in until the quality gates say otherwise.
    /// </summary>
    public static class PromptAttentionConfig
    {
        /// <summary>The flag both hosts accept.</summary>
        public const string Flag = "--prompt-attention";

        /// <summary>The environment variable the flag sets and the executor reads.</summary>
        public const string EnvironmentVariable = "TS_PROMPT_ATTENTION";

        /// <summary>The path when nothing is chosen.</summary>
        public const PromptAttentionPath Default = PromptAttentionPath.F16;

        /// <summary>The accepted values, in <see cref="PromptAttentionPath"/> order (one spelling each).</summary>
        public static readonly string[] Values = { "f16", "f32acc", "int8" };

        /// <summary>The value's name as the flag and the variable spell it.</summary>
        public static string Name(PromptAttentionPath path) => Values[(int)path];

        /// <summary>Parses one of <see cref="Values"/> (case-insensitive, surrounding blanks ignored).</summary>
        public static bool TryParse(string value, out PromptAttentionPath path)
        {
            path = Default;
            if (string.IsNullOrWhiteSpace(value)) return false;
            string trimmed = value.Trim();
            for (int i = 0; i < Values.Length; i++)
                if (string.Equals(trimmed, Values[i], StringComparison.OrdinalIgnoreCase))
                {
                    path = (PromptAttentionPath)i;
                    return true;
                }
            return false;
        }

        /// <summary>The error for a value that is not one of <see cref="Values"/>, naming where it came from.</summary>
        public static string InvalidValueMessage(string source, string value)
            => $"Unknown {source} value '{value}'. Valid: {string.Join(", ", Values)} (default f16).";

        /// <summary>
        /// The configured path: <see cref="EnvironmentVariable"/> (which the hosts' flag sets) when present, else the default.
        /// A value that is not one of <see cref="Values"/> throws <see cref="ArgumentException"/>: a deployment that asked for a
        /// path must not run another one without a word. Unset or empty means the default.
        /// </summary>
        public static PromptAttentionPath FromEnvironment() => FromValue(Environment.GetEnvironmentVariable(EnvironmentVariable));

        /// <summary><see cref="FromEnvironment"/> for a given variable value (tests read it without the process environment).</summary>
        public static PromptAttentionPath FromValue(string value)
        {
            if (string.IsNullOrWhiteSpace(value)) return Default;
            if (!TryParse(value, out PromptAttentionPath path))
                throw new ArgumentException(InvalidValueMessage(EnvironmentVariable, value));
            return path;
        }
    }
}
