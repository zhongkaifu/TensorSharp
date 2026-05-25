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

namespace TensorSharp.Runtime
{
    /// <summary>
    /// Knobs for the paged KV cache: block size, RAM cap, optional SSD spill tier.
    /// Designed so the feature is fully opt-in - callers either supply an explicit
    /// config or invoke <see cref="FromEnvironment"/> to pick up the env-var defaults.
    /// When <see cref="Enabled"/> is false the manager short-circuits all operations
    /// and behaves like a no-op, preserving the legacy single-session reuse path.
    /// </summary>
    public sealed class PagedKvCacheConfig
    {
        /// <summary>Master switch. When false the manager performs no work.</summary>
        public bool Enabled { get; init; }

        /// <summary>Tokens per block. Larger blocks lower hashing overhead but make
        /// partial-prefix sharing coarser. Default 256 matches omlx's PagedAttention.</summary>
        public int BlockSize { get; init; } = 256;

        /// <summary>Hard cap on bytes resident in the in-memory tier. Blocks beyond
        /// the cap are dropped (or spilled to SSD if the spill tier is enabled).</summary>
        public long MaxRamBytes { get; init; } = 1L * 1024 * 1024 * 1024;

        /// <summary>Optional SSD spill directory. <c>null</c> disables the spill tier.
        /// The directory is created lazily on first write.</summary>
        public string SsdDirectory { get; init; }

        /// <summary>Hard cap on bytes resident on the SSD spill tier. Old blocks
        /// are deleted in LRU order once this is exceeded.</summary>
        public long MaxSsdBytes { get; init; } = 16L * 1024 * 1024 * 1024;

        /// <summary>
        /// Read configuration from environment variables:
        ///   TS_KV_PAGED_CACHE=1            - enable the manager
        ///   TS_KV_BLOCK_SIZE=&lt;tokens&gt; - block size (default 256)
        ///   TS_KV_CACHE_MAX_RAM_MB=&lt;mb&gt;- in-memory tier cap (default 1024)
        ///   TS_KV_CACHE_SSD_DIR=&lt;path&gt; - enable SSD spill tier rooted at path
        ///   TS_KV_CACHE_MAX_SSD_MB=&lt;mb&gt;- SSD tier cap (default 16384)
        /// Any value that fails to parse falls back to the documented default.
        /// </summary>
        public static PagedKvCacheConfig FromEnvironment()
        {
            bool enabled = ReadBool("TS_KV_PAGED_CACHE", false);
            int blockSize = ReadInt("TS_KV_BLOCK_SIZE", 256);
            long maxRamBytes = ReadLong("TS_KV_CACHE_MAX_RAM_MB", 1024) * 1024L * 1024L;
            string ssdDir = Environment.GetEnvironmentVariable("TS_KV_CACHE_SSD_DIR");
            if (string.IsNullOrWhiteSpace(ssdDir))
                ssdDir = null;
            long maxSsdBytes = ReadLong("TS_KV_CACHE_MAX_SSD_MB", 16 * 1024) * 1024L * 1024L;

            if (blockSize <= 0)
                blockSize = 256;
            if (maxRamBytes <= 0)
                maxRamBytes = 1L * 1024 * 1024 * 1024;
            if (maxSsdBytes <= 0)
                maxSsdBytes = 16L * 1024 * 1024 * 1024;

            return new PagedKvCacheConfig
            {
                Enabled = enabled,
                BlockSize = blockSize,
                MaxRamBytes = maxRamBytes,
                SsdDirectory = ssdDir,
                MaxSsdBytes = maxSsdBytes,
            };
        }

        private static bool ReadBool(string name, bool defaultValue)
        {
            string v = Environment.GetEnvironmentVariable(name);
            if (string.IsNullOrWhiteSpace(v))
                return defaultValue;
            v = v.Trim().ToLowerInvariant();
            return v is "1" or "true" or "yes" or "on";
        }

        private static int ReadInt(string name, int defaultValue)
        {
            string v = Environment.GetEnvironmentVariable(name);
            return int.TryParse(v, out int parsed) ? parsed : defaultValue;
        }

        private static long ReadLong(string name, long defaultValue)
        {
            string v = Environment.GetEnvironmentVariable(name);
            return long.TryParse(v, out long parsed) ? parsed : defaultValue;
        }
    }
}
