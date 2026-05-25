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
using System.Runtime.InteropServices;

namespace TensorSharp.MLX
{
    /// <summary>
    /// Per-process configuration for MoE expert weight offload to SSD-backed mmap.
    /// Enable with <c>TS_MLX_EXPERT_OFFLOAD_MB=&lt;n&gt;</c>. When set to a positive
    /// value:
    ///   - <see cref="ModelBase.PrepareMlxQuantizedWeightsForInference"/> SKIPS the
    ///     eager device upload for tensors whose name matches <c>*_exps.*</c> and
    ///     for the stacked-experts views populated alongside them; their host data
    ///     (the GGUF mmap pointer) is kept alive so MLX can lazily upload an
    ///     expert when it is first routed in a matmul call.
    ///   - <see cref="MlxQuantizedOps"/> maintains an LRU over those "offloadable"
    ///     cache entries; when their resident byte total exceeds the configured
    ///     ceiling, the oldest entries are evicted (their MLX arrays are freed via
    ///     the FIFO-ordered worker, so any kernel currently using them completes
    ///     before the free runs).
    /// Non-expert weights (attention/embedding/lm_head) are NEVER offloaded — they
    /// are small in aggregate, hot on every forward, and remain permanently
    /// device-resident under the existing preload path.
    /// </summary>
    public static class MoeExpertOffload
    {
        private const string EnvVarMb = "TS_MLX_EXPERT_OFFLOAD_MB";
        private static readonly long _maxCacheBytes = ParseLimit();
        private static readonly HashSet<IntPtr> _offloadableKeys = new();
        private static readonly object _sync = new();

        /// <summary>True when <c>TS_MLX_EXPERT_OFFLOAD_MB</c> is set to a positive value.</summary>
        public static bool IsEnabled => _maxCacheBytes > 0;

        /// <summary>
        /// Maximum total raw-bytes of MLX-resident offloadable (expert) weights
        /// before LRU eviction kicks in. 0 disables the mechanism entirely; any
        /// positive value enables offload AND caps cached expert bytes at that
        /// value. The metric uses each weight's <c>RawBytes</c> (GGUF block byte
        /// count) as a proxy for MLX-side residency.
        /// </summary>
        public static long MaxCacheBytes => _maxCacheBytes;

        /// <summary>
        /// Per the GGUF / llama.cpp naming convention, all MoE expert weight
        /// tensors carry an <c>_exps.</c> infix (e.g. <c>blk.0.ffn_gate_exps.5.weight</c>
        /// and the stacked <c>blk.0.ffn_gate_exps.weight</c>). Non-expert
        /// weights never carry that infix.
        /// </summary>
        public static bool IsExpertWeightName(string name)
            => !string.IsNullOrEmpty(name) && name.IndexOf("_exps.", StringComparison.Ordinal) >= 0;

        /// <summary>
        /// Register a cache key as an eligible offload target. Called once per
        /// expert weight (and once per stacked-experts view) during model
        /// preparation. Subsequent <see cref="MlxQuantizedOps"/> cache lookups
        /// keyed by this pointer participate in the LRU and may be evicted; all
        /// other entries are pinned in the cache forever.
        /// </summary>
        public static void RegisterOffloadable(IntPtr cacheKey)
        {
            if (cacheKey == IntPtr.Zero)
                return;
            lock (_sync)
            {
                _offloadableKeys.Add(cacheKey);
            }
        }

        public static bool IsOffloadable(IntPtr cacheKey)
        {
            if (cacheKey == IntPtr.Zero)
                return false;
            lock (_sync)
            {
                return _offloadableKeys.Contains(cacheKey);
            }
        }

        public static void Clear()
        {
            lock (_sync)
            {
                _offloadableKeys.Clear();
            }
        }

        /// <summary>
        /// Hint to the OS that the given file-backed mmap region is no longer
        /// needed. Used by the MLX offload path to actively release expert
        /// weight pages: the baseline (non-offload) preload path achieves this
        /// implicitly by calling <c>ReleaseHostData</c> after upload (which
        /// drops Metal's claim on the buffer and madvises DONTNEED), but in
        /// offload mode we keep the host pointer alive for re-upload — so we
        /// have to call this explicitly on registration and on cache eviction
        /// to match baseline's eviction behaviour. No-op on Windows.
        /// The range is rounded outward to whole page boundaries (16 KB on
        /// Apple Silicon); for GGUF tensors aligned on 32-byte block
        /// boundaries the rounding may overlap adjacent tensors, which is
        /// fine — they're also file-backed and will page back in on next
        /// touch.
        /// </summary>
        public static unsafe void AdvisePagesNotNeeded(IntPtr data, long byteCount)
        {
            if (data == IntPtr.Zero || byteCount <= 0)
                return;
            if (!OperatingSystem.IsMacOS() && !OperatingSystem.IsLinux())
                return;

            long pageSize = Environment.SystemPageSize;
            long address = data.ToInt64();
            long pageMask = ~(pageSize - 1);
            long alignedAddress = address & pageMask;
            long prefixBytes = address - alignedAddress;
            ulong length = checked((ulong)(byteCount + prefixBytes));
            ulong roundedLength = (length + (ulong)pageSize - 1) & ~((ulong)pageSize - 1);

            try
            {
                _ = madvise((void*)alignedAddress, (nuint)roundedLength, MadvDontNeed);
            }
            catch (DllNotFoundException)
            {
            }
            catch (EntryPointNotFoundException)
            {
            }
        }

        private const int MadvDontNeed = 4;

        [DllImport("libc", SetLastError = true, EntryPoint = "madvise")]
        private static extern unsafe int madvise(void* addr, nuint len, int advice);

        private static long ParseLimit()
        {
            string value = Environment.GetEnvironmentVariable(EnvVarMb);
            if (!long.TryParse(value, out long mb) || mb <= 0)
                return 0;
            return mb * 1024L * 1024L;
        }
    }
}
