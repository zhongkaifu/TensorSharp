// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Diagnostics;
using System.Collections.Generic;

namespace TensorSharp.Models
{
    public partial class Qwen35Model
    {
        /// <summary>Headroom for optional cache capacity and idle holders. Both the
        /// host allocation and a GPU mirror have to fit. The host estimate uses the
        /// process working set and its available-memory limit; it is not a claim
        /// that all other host processes' allocations are accounted for. Unknown
        /// budgets leave the existing capacity policy intact.</summary>
        protected virtual long? GetCacheMemorySpareBytes()
        {
            long? spare = GpuMemoryBudget.TryGetReservationSpareBytes(_backend, out long deviceSpare)
                ? deviceSpare : null;
            long? hostSpare = GetHostMemorySpareBytes();
            if (hostSpare.HasValue)
                spare = spare.HasValue ? Math.Min(spare.Value, hostSpare.Value) : hostSpare;
            return spare;
        }

        /// <summary>The host half of <see cref="GetCacheMemorySpareBytes"/>, or null when the
        /// process cannot measure it.</summary>
        protected virtual long? GetHostMemorySpareBytes()
        {
            // The GC's available-memory limit accounts for a container/VM or an
            // explicit heap limit. WorkingSet includes the native host allocations
            // and mapped weights that managed-heap accounting alone would miss.
            long? spare = null;
            try
            {
                long available = GC.GetGCMemoryInfo().TotalAvailableMemoryBytes;
                if (available > 0)
                {
                    using var process = Process.GetCurrentProcess();
                    long reserve = Math.Max(GpuMemoryBudget.MinHeadroomBytes, available / 16);
                    spare = Math.Max(0, available - Math.Min(available, process.WorkingSet64) - reserve);
                }
            }
            catch (InvalidOperationException) { }
            catch (System.ComponentModel.Win32Exception) { }
            catch (NotSupportedException) { }
            // On a Mac the working set leaves out the GPU's buffers (an MLX server's stayed at 14 GB while
            // its footprint passed 34 GB) and other programs' memory: also only what macOS can give.
            if (OperatingSystem.IsMacOS() && UnifiedMemorySpareBytes() is long mac)
                spare = spare.HasValue ? Math.Min(spare.Value, Math.Max(0, mac)) : Math.Max(0, mac);
            return spare;
        }

        private const int LongCacheTokens = 16384;
        private const int MinGeometricSlackTokens = 4096;

        internal static int ResolveAttentionCacheGrowthCapacity(
            int currentCapacity, int requiredTokens, int maxContext, long bytesPerToken,
            long? spareBytes, bool geometricGrowth = true)
        {
            if (requiredTokens <= currentCapacity) return currentCapacity;
            if (requiredTokens < 0 || maxContext < requiredTokens)
                throw new ArgumentOutOfRangeException(nameof(requiredTokens));
            long aligned = ((long)requiredTokens + CacheCapacityAlignment - 1) / CacheCapacityAlignment * CacheCapacityAlignment;
            int minimum = (int)Math.Min(maxContext, aligned);
            if (!geometricGrowth) return minimum;
            long geometric = Math.Max(currentCapacity, 1);
            while (geometric < requiredTokens)
                geometric = Math.Min(maxContext, geometric * 2);
            // Bounded slack: doubling a long cache reserved a second copy of it. A
            // 22,016-row Qwen3.8-27B holder grew to 32,768 rows for a 22,987-token
            // prompt (370 MB of q8 K/V each), which four concurrent requests could not
            // afford next to their retained prefixes. Short caches still double; long
            // ones grow by a quarter.
            long slack = currentCapacity < LongCacheTokens ? currentCapacity
                : Math.Max(MinGeometricSlackTokens, currentCapacity / 4);
            long bounded = (long)currentCapacity + slack;
            bounded = (bounded + CacheCapacityAlignment - 1) / CacheCapacityAlignment * CacheCapacityAlignment;
            geometric = Math.Max(minimum, Math.Min(geometric, Math.Min(maxContext, bounded)));
            // Free bytes exclude the existing cache. Compare the additional KV
            // cost, leaving half the spare for graph scratch and replacement
            // transients. Required rows still grow even when no optional slack fits.
            if (spareBytes.HasValue && bytesPerToken > 0
                && geometric - currentCapacity > Math.Max(0, spareBytes.Value) / 2 / bytesPerToken)
                return minimum;
            return (int)geometric;
        }

        internal static bool CanPoolIdleCache(long holderBytes, long pooledBytes, int poolCount, int poolMax, long? spareBytes)
        {
            if (poolCount >= poolMax || poolMax <= 0) return false;
            if (!spareBytes.HasValue) return true;
            long budget = Math.Max(0, spareBytes.Value) / 2;
            return pooledBytes <= budget && holderBytes <= budget - pooledBytes;
        }

        private static long IdleHolderBytes(Qwen35KvCacheHolder holder)
        {
            long bytes = 0;
            var storages = new HashSet<Storage>();
            foreach (Tensor[] tensors in new[] { holder.K, holder.V, holder.DeltaState })
                if (tensors != null)
                    foreach (Tensor tensor in tensors)
                        if (tensor != null && storages.Add(tensor.Storage))
                            bytes = checked(bytes + tensor.Storage.ByteLength);
            if (holder.ConvState != null)
                foreach (float[] state in holder.ConvState)
                    if (state != null) bytes = checked(bytes + (long)state.Length * sizeof(float));
            if (holder.ConvWriteIdx != null) bytes = checked(bytes + (long)holder.ConvWriteIdx.Length * sizeof(int));
            if (holder.Logits != null) bytes = checked(bytes + (long)holder.Logits.Length * sizeof(float));
            // MLX GDN caches (conv tail + delta state per recurrent layer).
            bytes = checked(bytes + holder.MlxGdnStateBytes);
            // ConvScratch contains the same number of floats as the host conv rings.
            if (holder.ConvScratch != IntPtr.Zero && holder.ConvState != null)
                foreach (float[] state in holder.ConvState)
                    if (state != null) bytes = checked(bytes + (long)state.Length * sizeof(float));
            return bytes;
        }

        private long PooledHolderBytes()
        {
            long bytes = 0;
            if (_holderPool != null)
                foreach (var holder in _holderPool) bytes = checked(bytes + IdleHolderBytes(holder));
            return bytes;
        }
    }
}
