// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using TensorSharp.Models.Architecture;

namespace TensorSharp.Models
{
    /// <summary>DeepSeek V4 (Flash) architecture plug-in.</summary>
    internal static class DeepSeek4Architecture
    {
        public static ModelArchitectureDescriptor Descriptor { get; } = new()
        {
            // Deliberately TensorParallel as far as the shared gate is concerned:
            // DeepSeek V4 drives several GPUs through its OWN executor, sized by
            // TS_DSV4_NGPU, and the gate must not interfere with that.
            Id = "deepseek4",
            DisplayName = "DeepSeek V4 (Flash)",
            Aliases = new[] { "deepseek4" },
            Factory = c => new DeepSeek4Model(c.GgufPath, c.Backend, c.TpDegree, c.TpGroup, c.DraftModelPath),
        };

        /// <summary>
        /// The K/V storage every DeepSeek V4 / V4.1 executor actually uses. The
        /// native ggml graph allocates the sliding-window raw ring, the compressed
        /// (CSA/HCA) rows, the indexer rows, the rewind-checkpoint shadows and the
        /// DSpark draft rings as <c>GGML_TYPE_F16</c>; the pure C# executor commits
        /// F16-rounded rows; the direct-CUDA engine owns the same layout. None of
        /// them reads <see cref="KvCacheDtypeConfig"/>, so this is what
        /// <see cref="ModelBase.KvCacheDtype"/> must report.
        /// </summary>
        internal const KvCacheDtype ExecutorKvCacheDtype = KvCacheDtype.F16;

        /// <summary>
        /// Refuse a block-quantized (<c>q8_0</c> / <c>q4_0</c>) K/V cache before a
        /// checkpoint is opened, naming why.
        ///
        /// <para>The V4/V4.1 attention caches are not the per-layer K/V tensors the
        /// shared families hand to ggml's flash-attention kernels (which do read
        /// q8_0/q4_0 K/V, but only at the 64/128-wide head sizes of their vector
        /// kernels). They are the MLA latent rows (K doubles as V at the latent
        /// width, which those kernels accept only as F16), kept
        /// in a sliding-window ring plus compressed and indexer rows that the
        /// architecture's own fused kernels write (<c>TSG_DSV4_FUSED_ATTN_PREP</c>,
        /// <c>TSG_DSV4_FUSED_COMPRESS</c>) and read (<c>TSG_DSV4_FUSED_KGATHER</c>,
        /// the compact/TP gathers, the checkpoint shadow copies) as F16 rows with
        /// no dequantize step. Honouring the request would mean re-typing every
        /// one of those kernels on the CUDA, CPU and direct-CUDA executors; the
        /// old behaviour was worse than refusing: the request was accepted,
        /// <see cref="ModelBase.KvCacheDtype"/> reported <c>q8_0</c>, and the
        /// caches were F16 all along.</para>
        ///
        /// <para>For V4.1 the point is moot besides: the checkpoint's own trained
        /// cache quantization (FP8 E4M3 raw rows, MXFP4 indexer rows, NVFP4
        /// compressed rows) is reproduced before every F16 store, so the F16
        /// cache already holds fewer distinct values than a q8_0 block would.</para>
        /// </summary>
        internal static void RefuseBlockQuantizedKvCache(string family, bool v41)
        {
            KvCacheDtype requested = KvCacheDtypeConfig.Current;
            if (!requested.IsBlockQuantized()) return;
            throw new NotSupportedException(BlockQuantizedKvCacheError(family, requested, v41));
        }

        internal static string BlockQuantizedKvCacheError(string family, KvCacheDtype requested, bool v41)
            => $"KV_CACHE_DTYPE={requested.ToShortString()} is not supported by {family}. Its executors keep every " +
               "persistent attention cache (the MLA latent K=V rows in the sliding-window ring, the compressed and " +
               "indexer rows, the rewind-checkpoint shadows and the DSpark draft rings) as F16 tensors that the " +
               "architecture's own attention, gather and compressor kernels read directly, with no block-dequantize " +
               "step" +
               (v41 ? "; the checkpoint's trained cache quantization (FP8 raw rows, MXFP4 indexer rows, NVFP4 " +
                      "compressed rows) is already applied before each F16 store, so a q8_0 block would not shrink " +
                      "the cache's information content either"
                    : string.Empty) +
               ". Unset KV_CACHE_DTYPE or set it to f16; a smaller cache is not available for this family.";

        /// <summary>
        /// The notice printed when <c>f32</c> was asked for explicitly: the caches
        /// are F16 on every executor, and saying so beats reporting a dtype the
        /// model does not use.
        /// </summary>
        internal static string DescribeF32KvCacheRequest(string family)
            => $"[dsv4] KV_CACHE_DTYPE=f32 requested, but {family} keeps its attention caches in F16 on every " +
               "executor (native ggml, direct CUDA and pure C#); running with kvCacheDtype=f16.";
    }
}
