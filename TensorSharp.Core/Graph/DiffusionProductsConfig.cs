// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;

namespace TensorSharp.Graph
{
    /// <summary>The arithmetic of a diffusion model's wide weight products on <c>ggml_cuda</c> (MiniMax-H3-Image, MiniMax-H3
    /// video and Qwen-Image 2.1, where their projections take cuBLAS).</summary>
    public enum DiffusionProductsPath
    {
        /// <summary><c>bf16</c>: the products as the backend runs them (BF16 weights and Q4_K projections on cuBLAS BF16, the
        /// VAE's F16 projections on cuBLAS F16, other quantizations on ggml's kernels). The default.</summary>
        Bf16,

        /// <summary><c>int8</c>: the DiT's cuBLAS projections on int8 tensor cores, each weight row and each token rounded to
        /// 8 bits against its own peak (cuBLASLt, exact int32 sums, F32 result). Faster, and it changes the output.</summary>
        Int8,

        /// <summary><c>int8-all</c>: <see cref="Int8"/> plus the VAE decoders' F16 projections (MiniMax-H3's video VAE and
        /// MiniMax-H3-Image's), whose int8 rows stay on the device between tiles.</summary>
        Int8All,
    }

    /// <summary>
    /// The process-wide diffusion products choice (design D4, docs/design/diffusion-runtime.md):
    /// <c>--diffusion-products &lt;bf16|int8|int8-all&gt;</c> on the CLI and the server, the <c>diffusion-products</c> key of a
    /// <c>--config</c> file. The hosts publish it as <see cref="EnvironmentVariable"/>, which the native CUDA products read
    /// (tsg_cuda_q4k_gemm.cu); the flag wins over an inherited value. It replaces <c>TS_H3_INT8_MATMUL</c> (<c>1</c> is
    /// <c>int8</c>, <c>all</c> is <c>int8-all</c>), which is refused by name (RemovedCliFlags). The int8 routes are CUDA's
    /// only: elsewhere the products run as stored, as the earlier variable's did, and the hosts say so once.
    /// </summary>
    public static class DiffusionProductsConfig
    {
        /// <summary>The flag both hosts accept (a config file's <c>diffusion-products</c> key).</summary>
        public const string Flag = "--diffusion-products";

        /// <summary>The environment variable the flag sets and the native products read.</summary>
        public const string EnvironmentVariable = "TS_DIFFUSION_PRODUCTS";

        /// <summary>The products when nothing is chosen.</summary>
        public const DiffusionProductsPath Default = DiffusionProductsPath.Bf16;

        /// <summary>The accepted values, in <see cref="DiffusionProductsPath"/> order (one spelling each).</summary>
        public static readonly string[] Values = { "bf16", "int8", "int8-all" };

        /// <summary>The value's name as the flag and the variable spell it.</summary>
        public static string Name(DiffusionProductsPath path) => Values[(int)path];

        /// <summary>Parses one of <see cref="Values"/> (case-insensitive, surrounding blanks ignored).</summary>
        public static bool TryParse(string value, out DiffusionProductsPath path)
        {
            path = Default;
            if (string.IsNullOrWhiteSpace(value)) return false;
            string trimmed = value.Trim();
            for (int i = 0; i < Values.Length; i++)
                if (string.Equals(trimmed, Values[i], StringComparison.OrdinalIgnoreCase))
                {
                    path = (DiffusionProductsPath)i;
                    return true;
                }
            return false;
        }

        /// <summary>The error for a value that is not one of <see cref="Values"/>, naming where it came from.</summary>
        public static string InvalidValueMessage(string source, string value)
            => $"Unknown {source} value '{value}'. Valid: {string.Join(", ", Values)} (default: bf16).";

        /// <summary>The chosen products: <see cref="EnvironmentVariable"/> (which the hosts' flag sets) when present, else null
        /// (nothing chosen: <see cref="Default"/>). A value that is not one of <see cref="Values"/> throws
        /// <see cref="ArgumentException"/>: a deployment that asked for a route must not run another one without a word.</summary>
        public static DiffusionProductsPath? FromEnvironment() => FromValue(Environment.GetEnvironmentVariable(EnvironmentVariable));

        /// <summary><see cref="FromEnvironment"/> for a given variable value (tests read it without the process environment).</summary>
        public static DiffusionProductsPath? FromValue(string value)
        {
            if (string.IsNullOrWhiteSpace(value)) return null;
            if (!TryParse(value, out DiffusionProductsPath path))
                throw new ArgumentException(InvalidValueMessage(EnvironmentVariable, value));
            return path;
        }

        /// <summary>The line a host prints when an int8 route was chosen for a backend that has none (null when it applies):
        /// the products then run as stored, as they did with <c>TS_H3_INT8_MATMUL</c>.</summary>
        public static string IgnoredOn(DiffusionProductsPath? chosen, string backend)
        {
            if (chosen is not ({ } path) || path == DiffusionProductsPath.Bf16) return null;
            if (string.Equals(backend, "ggml_cuda", StringComparison.OrdinalIgnoreCase)) return null;
            return $"WARNING: {Flag} {Name(path)} takes effect on --backend ggml_cuda only; on {backend} a diffusion model's " +
                   "products run as stored (bf16).";
        }
    }
}
