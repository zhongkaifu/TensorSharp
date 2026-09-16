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
            // The GC's available-memory limit accounts for a container/VM or an
            // explicit heap limit. WorkingSet includes the native host allocations
            // and mapped weights that managed-heap accounting alone would miss.
            try
            {
                long available = GC.GetGCMemoryInfo().TotalAvailableMemoryBytes;
                if (available > 0)
                {
                    using var process = Process.GetCurrentProcess();
                    long reserve = Math.Max(GpuMemoryBudget.MinHeadroomBytes, available / 16);
                    long hostSpare = Math.Max(0, available - Math.Min(available, process.WorkingSet64) - reserve);
                    spare = spare.HasValue ? Math.Min(spare.Value, hostSpare) : hostSpare;
                }
            }
            catch (InvalidOperationException) { }
            catch (System.ComponentModel.Win32Exception) { }
            catch (NotSupportedException) { }
            return spare;
        }

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
