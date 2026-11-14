// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;

using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Scheduling.PrefixCache;

namespace TensorSharp.Models
{
    /// <summary>
    /// Unified batching (<see cref="IUnifiedBatchModel"/>): every request has its own slot from its first step, and a
    /// step runs every scheduled request's prompt chunk or decode token in one forward over their slots
    /// (<see cref="TryForwardSlots"/>), whatever the count.
    /// </summary>
    public abstract partial class StandardDecoderModel : IUnifiedBatchModel, ISoloPrefillBudgetModel
    {
        public bool SupportsUnifiedBatching => SupportsPerSequenceFusedForward;

        /// <summary>A lone prompt's step on MLX, in tokens (see <see cref="PreferredSoloPrefillTokens"/>).</summary>
        private const int MlxPromptStepTokens = 512;

        /// <summary>
        /// On MLX a lone prompt prefills in steps of <see cref="MlxPromptStepTokens"/>: MLX holds memory of the order of
        /// each layer's feed-forward activations until a step ends (Qwen3.8-27B on an M5 Pro: 0.9 GB a layer for a
        /// 3.6K-token step, past 40 GB wired), so the scheduler's whole-prompt step ran out of memory where 512-token
        /// steps peak at 22 GB wired (Qwen35Model 19 GB) in the same prompt time (9.09 vs 9.07-9.11 s; 1024-token steps
        /// 8.99 s at 31 GB). Elsewhere the scheduler's own steps (0).
        /// </summary>
        public int PreferredSoloPrefillTokens(int start, int remaining)
            => _backend switch
            {
                BackendType.Mlx => Math.Min(remaining, MlxPromptStepTokens),
                BackendType.Cuda => CudaPromptStepTokens(remaining),
                _ => 0,
            };

        /// <summary>Device memory a direct-CUDA prompt step keeps beside its rows' activations (cuBLAS workspace, the
        /// dequantized-weight and activation scratch).</summary>
        private const long CudaPromptReserveBytes = 768L << 20;

        /// <summary>
        /// On direct CUDA a lone prompt prefills in steps whose activations fit the device memory the weights and caches
        /// leave: the tensor executor allocates a layer's activations as it runs, and a step that runs out part way has
        /// already written some layers' cache rows and states, so it is not retried (Qwen3.8-27B UD-IQ3_XXS, 10.5 GB
        /// resident on a 16 GB RTX 2000 Ada, ran out on a 3.6K-token prompt in one step). The scheduler's own step (0)
        /// where the whole remainder fits.
        /// </summary>
        private int CudaPromptStepTokens(int remaining)
        {
            if (_allocator is not TensorSharp.Cuda.CudaAllocator cuda || remaining <= 256) return 0;
            // Sized once, from the memory the first lone prompt finds beside the weights: a budget read at every prompt
            // followed what earlier requests' caches happened to hold, so the same prompt split differently from one
            // request to the next (and a near-tied first token flipped with the split's rounding).
            if (_cudaPromptRows == 0)
            {
                long free;
                try { free = cuda.GetMemoryInfo().freeBytes + cuda.GetStats().CachedBytes; }
                catch (Exception) { return 0; }
                _cudaPromptRows = (int)Math.Clamp((free - CudaPromptReserveBytes) / CudaPromptRowBytes() / 256 * 256, 256, MaxCudaPromptStepRows);
            }
            int rows = Math.Min(_cudaPromptRows, remaining);
            // What the first prompt found may be held since: recorded steps keep the pool blocks their kernels used, a
            // speculating request its verify buffers, retained slots their caches (an RTX PRO 6000 slice ran out of memory
            // part way through the second 4K prompt of a speculating run). The recordings go first (they record again), then
            // finished requests' retained slots, the oldest first; a step that still does not fit is narrower, this prompt only.
            long need = rows * CudaPromptRowBytes() + CudaPromptReserveBytes;
            if (PromptFreeBytes(cuda) < need)
            {
                ClearGraphCache();
                if (PromptFreeBytes(cuda) < need) ReleaseRetainedSlotsFor(need, cuda);
                long now = PromptFreeBytes(cuda);
                if (now < need)
                    return (int)Math.Clamp((now - CudaPromptReserveBytes) / CudaPromptRowBytes() / 256 * 256, 256, _cudaPromptRows);
            }
            return remaining <= _cudaPromptRows ? 0 : _cudaPromptRows;
        }

        /// <summary>
        /// Finished requests' retained slots (prefix-cache payloads), the oldest first, until a lone prompt's step of
        /// <paramref name="need"/> bytes fits, each reported to the prefix cache as reclaimed (DEC-23). Kept, they narrowed
        /// every later prompt's steps on a small card, and a narrower step stages every host expert again: a 16 GB RTX 2000
        /// Ada's third distinct 4K prompt of Qwen3.8 Flash Next UD-Q2_K_XL ran in 1,024- and 541-row steps, 17.4 s against
        /// 7.05 s for the first. The request being served holds its own slot, so its reuse has already happened.
        /// </summary>
        private void ReleaseRetainedSlotsFor(long need, TensorSharp.Cuda.CudaAllocator cuda)
        {
            if (_retainedSlots == null || _retainedSlots.Count == 0) return;
            foreach (var (key, slot) in System.Linq.Enumerable.ToList(System.Linq.Enumerable.OrderBy(_retainedSlots, kv => kv.Value.RetainedAt)))
            {
                if (PromptFreeBytes(cuda) >= need) break;
                _retainedSlots.Remove(key);
                DisposeSlot(slot);
                _prefixCacheSink?.OnPayloadInvalidated(key, Runtime.Scheduling.PrefixCache.InvalidationReason.NativeSlotReclaimed);
            }
        }

        /// <summary>Tests stand in for the device's free memory a lone prompt's step is fitted to (after its first sizing).</summary>
        internal Func<long> PromptFreeBytesForTests { get; set; }

        private long PromptFreeBytes(TensorSharp.Cuda.CudaAllocator cuda) => PromptFreeBytesForTests?.Invoke() ?? CudaFreeBytes(cuda);

        /// <summary>The device's free bytes and its pool's cached ones (long.MaxValue when unknown).</summary>
        private static long CudaFreeBytes(TensorSharp.Cuda.CudaAllocator cuda)
        {
            try { return cuda.GetMemoryInfo().freeBytes + cuda.GetStats().CachedBytes; }
            catch (Exception) { return long.MaxValue; }
        }

        private int _cudaPromptRows;

        /// <summary>The widest lone-prompt step the model chooses: wider ones were slower where the memory allowed them
        /// (Qwen3.8 Flash Next on an A40, a 3.6K-token prompt in one step 6.1 s against 2.1 s in 2,816 + 794 rows; UD-Q2_K_XL
        /// on a 16 GB RTX 2000 Ada, 2,560-row steps 11.5 s against 7.2 s in 2,048-row ones).</summary>
        private const int MaxCudaPromptStepRows = 2048;

        /// <summary>A prompt row's activations at their peak, in bytes, about twice what the CUDA executor's steps peak at
        /// (its pool's live bytes over a step, 11-14 on an RTX 2000 Ada and an RTX PRO 6000 slice): Qwen3.8 Flash Next
        /// 329 KB a row (ff 25,600, a 10,240-float stream), Qwen3.8-27B 196 KB (ff 17,408, 5,120), i.e. about 1.8 ff and
        /// 3.8 stream-wide F32 rows (the executor frees each layer's activations after their last reader). The estimate
        /// had been 6 ff and 48 rows (2,520 and 1,368 KB): a 16 GB card then read a 4K prompt in 512-row steps, 19.7 s
        /// against 7.2 s in 2,048-row ones, each step streaming the offloaded experts once.</summary>
        private long CudaPromptRowBytes()
        {
            long ff = Config.IntermediateSize;
            foreach (var stacked in _stackedExpertWeights.Values)
                ff = Math.Max(ff, stacked.PerExpertNe1 * Math.Max(1, _expertsUsed));
            long width = Math.Max(Config.HiddenSize, StreamWidth);
            return 4L * (3 * ff + 8 * width);
        }

        public int EnsureSequenceSlot(string requestId, int reserveTokens)
        {
            if (string.IsNullOrEmpty(requestId))
                throw new ArgumentException("RequestId required", nameof(requestId));
            RestorePrimaryCache();
            _slots ??= new Dictionary<string, Slot>(StringComparer.Ordinal);
            int reserve = Math.Min(_maxContextLength, reserveTokens);
            if (!_slots.TryGetValue(requestId, out Slot slot))
            {
                EnsureDeviceRoomForSlot(requestId, Math.Max(reserve, FreshSlotTokens));
                _slots[requestId] = slot = NewSlot(reserve);
            }
            if (reserve > slot.Capacity)
                EnsureSlotCapacity(slot, reserve, reservation: true);
            return slot.Length;
        }

        public bool TryForwardUnified(IReadOnlyList<string> requestIds, IReadOnlyList<int[]> tokens, int[] starts, float[][] lastRowLogits,
            float[][] features = null)
            => TryForwardSlots(requestIds, tokens, starts, lastRowLogits, features);

        /// <summary>On executors that return integer outputs (the CUDA executor), a greedy step's head ends in a top-1 over each
        /// sequence's last row, read back as an id: a plain decode step had read back Qwen3.8's 248,320 logits (1 MB) and
        /// scanned them on the host.</summary>
        public bool SupportsUnifiedGreedyIds => GraphExecutor?.Capabilities.GreedyIdOutputs == true && !_rankGraphs && _mesh.Stages == 1;

        public bool TryForwardUnifiedGreedy(IReadOnlyList<string> requestIds, IReadOnlyList<int[]> tokens, int[] starts, int[] nextTokens)
        {
            var logits = new float[requestIds.Count][];
            _greedyLastRows = SupportsUnifiedGreedyIds;
            try
            {
                if (!TryForwardSlots(requestIds, tokens, starts, logits, null)) return false;
            }
            finally
            {
                _greedyLastRows = false;
            }
            for (int s = 0; s < logits.Length; s++)
            {
                float[] row = logits[s];
                if (row.Length == 1)
                {
                    nextTokens[s] = (int)row[0];   // the id as its F32 value (exact below 2^24)
                    continue;
                }
                // A step that ran off the graphs read the logits back: the first largest, as the device picks.
                int best = 0;
                for (int v = 1; v < row.Length; v++)
                    if (row[v] > row[best]) best = v;
                nextTokens[s] = best;
            }
            return true;
        }

        /// <summary>Set while <see cref="TryForwardUnifiedGreedy"/> runs: the graph steps' last-row heads end in a top-1.</summary>
        private bool _greedyLastRows;

        public float[] ForwardUnifiedMedia(string requestId, int[] tokens, int start)
        {
            if (!_slots.ContainsKey(requestId))
                throw new InvalidOperationException($"{requestId} has no slot.");
            BindSequenceCache(requestId);
            try
            {
                if (_cacheSeqLen != start)
                    throw new InvalidOperationException($"{requestId}: its slot holds {_cacheSeqLen} tokens, the chunk starts at {start}.");
                return (float[])ForwardPrompt(tokens).Clone();
            }
            finally
            {
                RestorePrimaryCache();
            }
        }

        /// <summary>The request's own slot, whichever cache is active (a unified step leaves the primary active, and a
        /// checkpoint of it would be another request's state).</summary>
        bool IPrefixCacheModel.TryCaptureCopy(string requestId, string payloadKey, out PayloadFootprint footprint)
        {
            footprint = default;
            if (string.IsNullOrEmpty(requestId) || string.IsNullOrEmpty(payloadKey) || !SupportsPrefixCheckpoints) return false;
            if (!TryGetSlot(requestId, out Slot slot))
                return HolderPrefixCacheAdapter.TryCaptureCopy(this, requestId, payloadKey, out footprint);
            _retainedSlots ??= new Dictionary<string, Slot>(StringComparer.Ordinal);
            if (_retainedSlots.ContainsKey(payloadKey) || slot.Length <= 0) return false;
            _retainedSlots.Add(payloadKey, Retained(CopySlot(slot, slot.Length)));
            footprint = MeasureEndState(payloadKey);
            return true;
        }
    }
}
