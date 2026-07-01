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
using System.Buffers;
using TensorSharp;

namespace TensorSharp.Models
{
    /// <summary>
    /// Storage element type for the per-layer K/V cache tensors. Selected once per
    /// model load (env var <c>KV_CACHE_DTYPE</c> or per-CLI flag) and threaded into
    /// every model's <c>InitKVCache</c>, <c>CopyToCache*</c>, and attention helpers.
    ///
    /// <c>F32</c> is the historical default: each cache element occupies four bytes
    /// and all attention kernels can read it directly.
    ///
    /// <c>F16</c> halves the resident KV-cache size (e.g. Gemma 4 E4B drops from
    /// ~4136 MB to ~2068 MB at the full 131K context). Quality is essentially
    /// identical: post-RoPE K/V values live in the [-10, 10] range that F16
    /// represents with ~10 bits of mantissa.
    ///
    /// <c>Q8_0</c> is the "model-aligned" choice for Q8_0 / Q4_K / IQ4_XS GGUF
    /// files. The cache stores values in 32-element blocks of one int8 quant per
    /// element plus a single F16 scale, for ~1.0625 bytes/elem total. That is
    /// another 2x compression over F16 (e.g. Gemma 4 E4B drops to ~1097 MB) and
    /// matches the precision of the model weights themselves, so the dominant
    /// matmul / flash-attention kernels can avoid an F16↔F32 reconvert path.
    /// Q8_0 is only valid when the entire cache I/O path stays in native GGML
    /// kernels - models that fall back to the C# managed attention helpers
    /// (which expect to walk the cache as a flat F32/F16 buffer) will reject
    /// Q8_0 at <c>InitKVCache</c> time.
    ///
    /// <c>Q4_0</c> is the most aggressive tier: 32-element blocks of one int4
    /// quant per element plus a single F16 scale, for ~0.5625 bytes/elem - half
    /// of Q8_0 and roughly 1/7 of F32 (e.g. Gemma 4 E4B drops from ~4136 MB at
    /// F32 to ~582 MB at the full 131K context). This is the right tier for very
    /// long contexts (128K-256K) where the KV cache dominates memory. It shares
    /// Q8_0's constraint - block-quantized, so only valid on the native flash
    /// path; the model-level <c>IsBlockQuantized</c> guards route it exactly like
    /// Q8_0. ggml's CUDA <c>cpy</c> (F32->Q4_0 cache write) and
    /// <c>flash_attn_ext</c> (K=Q4_0/V=Q4_0) both support it directly. Quality is
    /// lower than Q8_0 - 4-bit K/V noise is visible on tasks needing precise
    /// long-range recall - so it is opt-in via <c>--kv-cache-dtype q4_0</c>.
    /// </summary>
    public enum KvCacheDtype
    {
        F32 = 0,
        F16 = 1,
        Q8_0 = 2,
        Q4_0 = 3,
    }

    public static class KvCacheDtypeExtensions
    {
        // ggml type ids - must match the enum in
        // ExternalProjects/ggml/include/ggml.h.
        private const int GGML_TYPE_F32  = 0;
        private const int GGML_TYPE_F16  = 1;
        private const int GGML_TYPE_Q4_0 = 2;
        private const int GGML_TYPE_Q8_0 = 8;

        public static int ElementBytes(this KvCacheDtype dtype) => dtype switch
        {
            KvCacheDtype.F32 => 4,
            KvCacheDtype.F16 => 2,
            // Lower-bound for budget logging; actual storage rounds up to the
            // 32-element block boundary (34 bytes per block for Q8_0,
            // 18 bytes per block for Q4_0).
            KvCacheDtype.Q8_0 => 1,
            KvCacheDtype.Q4_0 => 1,
            _ => throw new ArgumentOutOfRangeException(nameof(dtype)),
        };

        /// <summary>
        /// Bytes consumed by a contiguous block-quantized cache of the given length
        /// (Q8_0: 32-element blocks of 34 bytes; Q4_0: 32-element blocks of 18
        /// bytes). For F32/F16 this is just elementCount*Size.
        /// </summary>
        public static long ByteLengthFor(this KvCacheDtype dtype, long elementCount) => dtype switch
        {
            KvCacheDtype.F32 => elementCount * 4,
            KvCacheDtype.F16 => elementCount * 2,
            KvCacheDtype.Q8_0 => DTypeExtensions.Q8_0Bytes(elementCount),
            KvCacheDtype.Q4_0 => DTypeExtensions.Q4_0Bytes(elementCount),
            _ => throw new ArgumentOutOfRangeException(nameof(dtype)),
        };

        public static DType ToDType(this KvCacheDtype dtype) => dtype switch
        {
            KvCacheDtype.F32 => DType.Float32,
            KvCacheDtype.F16 => DType.Float16,
            KvCacheDtype.Q8_0 => DType.Q8_0,
            KvCacheDtype.Q4_0 => DType.Q4_0,
            _ => throw new ArgumentOutOfRangeException(nameof(dtype)),
        };

        public static string ToShortString(this KvCacheDtype dtype) => dtype switch
        {
            KvCacheDtype.F32 => "f32",
            KvCacheDtype.F16 => "f16",
            KvCacheDtype.Q8_0 => "q8_0",
            KvCacheDtype.Q4_0 => "q4_0",
            _ => throw new ArgumentOutOfRangeException(nameof(dtype)),
        };

        public static int GgmlType(this KvCacheDtype dtype) => dtype switch
        {
            KvCacheDtype.F32 => GGML_TYPE_F32,
            KvCacheDtype.F16 => GGML_TYPE_F16,
            KvCacheDtype.Q8_0 => GGML_TYPE_Q8_0,
            KvCacheDtype.Q4_0 => GGML_TYPE_Q4_0,
            _ => throw new ArgumentOutOfRangeException(nameof(dtype)),
        };

        /// <summary>
        /// Block-quantized cache types cannot be read element-by-element from C#
        /// managed code without an explicit dequantize pass. Models that need to
        /// walk the cache directly (e.g. for SWA prev-window gather) must reject
        /// these dtypes during initialization.
        /// </summary>
        public static bool IsBlockQuantized(this KvCacheDtype dtype) => dtype switch
        {
            KvCacheDtype.Q8_0 => true,
            KvCacheDtype.Q4_0 => true,
            _ => false,
        };
    }

    /// <summary>
    /// Process-wide configuration for the KV-cache storage type. The CLI / server
    /// front-end calls <see cref="ConfigureFromEnvironment"/> once at start-up;
    /// each model picks the value up via <see cref="Current"/> when it allocates
    /// its per-layer K/V tensors.
    /// </summary>
    public static class KvCacheDtypeConfig
    {
        private static KvCacheDtype _current = KvCacheDtype.F32;
        private static bool _explicitlySet;

        public static KvCacheDtype Current => _current;
        public static bool IsExplicitlySet => _explicitlySet;

        public static void Set(KvCacheDtype dtype)
        {
            _current = dtype;
            _explicitlySet = true;
        }

        /// <summary>
        /// Apply a model-aligned default cache dtype if the user hasn't explicitly
        /// chosen one. Models whose dominant weight tier is below F32 (Q8_0,
        /// Q4_K, IQ4_XS, F16, etc.) get an F16 cache: K/V values fit losslessly
        /// inside F16's ~10-bit mantissa for the [-10, 10] post-RoPE range, so
        /// outputs are byte-identical to the F32 baseline while halving cache
        /// memory and bandwidth. Pure F32 native models keep the F32 cache.
        /// Callers can opt in to the more aggressive Q8_0 cache via
        /// <c>--kv-cache-dtype q8_0</c>.
        /// </summary>
        public static void ApplyModelDtypeDefault(int dominantGgmlType)
        {
            if (_explicitlySet) return;
            // 0 = GGML_TYPE_F32. Anything else (F16, BF16, Q8_0, Q4_K, ...) is
            // already operating below F32 precision so an F16 cache adds no
            // measurable error.
            if (dominantGgmlType != 0)
                _current = KvCacheDtype.F16;
        }

        public static bool TryParse(string value, out KvCacheDtype dtype)
        {
            dtype = KvCacheDtype.F32;
            if (string.IsNullOrWhiteSpace(value)) return false;
            switch (value.Trim().ToLowerInvariant())
            {
                case "f32":
                case "float32":
                case "fp32":
                    dtype = KvCacheDtype.F32;
                    return true;
                case "f16":
                case "float16":
                case "fp16":
                case "half":
                    dtype = KvCacheDtype.F16;
                    return true;
                case "q8_0":
                case "q8":
                case "int8":
                    dtype = KvCacheDtype.Q8_0;
                    return true;
                case "q4_0":
                case "q4":
                case "int4":
                    dtype = KvCacheDtype.Q4_0;
                    return true;
                default:
                    return false;
            }
        }

        /// <summary>
        /// Read the <c>KV_CACHE_DTYPE</c> environment variable (if any) and apply
        /// it as the process-wide default. Unrecognized values are ignored.
        /// </summary>
        public static void ConfigureFromEnvironment()
        {
            string value = Environment.GetEnvironmentVariable("KV_CACHE_DTYPE");
            if (TryParse(value, out KvCacheDtype dtype))
                Set(dtype);
        }
    }
}
