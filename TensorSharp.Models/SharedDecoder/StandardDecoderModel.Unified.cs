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
            // part way through the second 4K prompt of a speculating run). On one device the model makes the room in its
            // fixed order (EnsureDeviceRoom: pooled slots, parked recordings, finished conversations the oldest first, the
            // cached graphs last; dropping every graph first had every running request trace and record again); elsewhere the
            // recordings go first, then finished requests' retained slots, the oldest first. A step that still does not fit
            // is narrower, this prompt only. On one device the step is sized from the room EnsureDeviceRoom counted: a reading
            // after it leaves out the retained window buffers it freed (they wait in the allocator's small-block cache), and
            // narrowed the step anyway, those buffers given up for nothing.
            long need = rows * CudaPromptRowBytes() + CudaPromptReserveBytes;
            if (PromptFreeBytes(cuda) < need)
            {
                long now;
                if (DeviceRoomManaged)
                    EnsureDeviceRoom(need, RunningSlots(), RoomUse.PromptStep, mayEvict: true, "a prompt step", out now);
                else
                {
                    ClearGraphCache();
                    if (PromptFreeBytes(cuda) < need) ReleaseRetainedSlotsFor(need, cuda);
                    now = PromptFreeBytes(cuda);
                }
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

        /// <summary>The free bytes a lone prompt's step is fitted to: on a device this model manages, the driver's free memory
        /// and the large blocks the allocator frees on demand (<see cref="DeviceMemory"/>); elsewhere also the shards' small
        /// blocks, as before.</summary>
        private long PromptFreeBytes(TensorSharp.Cuda.CudaAllocator cuda)
            => PromptFreeBytesForTests?.Invoke() ?? (DeviceRoomManaged ? DeviceMemory(fresh: true).Free : CudaFreeBytes(cuda));

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

        public int EnsureSequenceSlot(string requestId, int reserveTokens) => EnsureSequenceSlot(requestId, reserveTokens, reserveTokens);

        /// <summary>
        /// The request's slot, with room for its reservation (<paramref name="reserveTokens"/>: its prompt and generation budget)
        /// where it is held as asked; where slots are sized to the device's room (<see cref="SizesSlotsToRoom"/>, the CUDA executor
        /// on one device) the reservation is fitted to the device once (<see cref="SlotTarget"/>: its class with headroom where
        /// memory allows, else the largest capacity that fits, and finished conversations released only for the
        /// <paramref name="requiredTokens"/> the step needs), and the slot grows only when a step needs more than it holds. Held as
        /// asked, a 60K-token agent prompt with a 32K max_tokens pinned 92K cells (6.4 GB of Qwen3.8-27B's K/V) from its first
        /// step.
        /// </summary>
        public int EnsureSequenceSlot(string requestId, int reserveTokens, int requiredTokens)
        {
            if (string.IsNullOrEmpty(requestId))
                throw new ArgumentException("RequestId required", nameof(requestId));
            NoteEngineThread();
            RestorePrimaryCache();
            ForgetDeferredGraphs();
            _slots ??= new Dictionary<string, Slot>(StringComparer.Ordinal);
            int reserve = Math.Min(_maxContextLength, reserveTokens);
            bool fitted = SizesSlotsToRoom;
            int required = fitted ? Math.Clamp(requiredTokens, 1, Math.Max(1, reserve)) : reserve;
            if (!_slots.TryGetValue(requestId, out Slot slot))
            {
                if (DeviceRoomManaged)
                    slot = NewSlotWithRoom(requestId, reserve, required);
                else
                {
                    EnsureDeviceRoomForSlot(requestId, Math.Max(reserve, FreshSlotTokens));
                    slot = ReclaimRetainedSlot(reserve) ?? NewSlot(reserve);
                }
                _slots[requestId] = slot;
                if (fitted) (_fittedReservations ??= new(StringComparer.Ordinal))[requestId] = reserve;
            }
            if (!fitted)
            {
                if (reserve > slot.Capacity && !TryMoveToPooledSlot(requestId, ref slot, reserve))
                    EnsureSlotCapacity(slot, reserve, reservation: true);
                return slot.Length;
            }
            // A slot the request adopted (a conversation's finished slot, a copy of a cached prefix) or a reservation not fitted
            // yet is fitted once; later steps ask again only when they need more than the slot holds.
            _fittedReservations ??= new Dictionary<string, int>(StringComparer.Ordinal);
            bool asked = _fittedReservations.TryGetValue(requestId, out int tried) && tried >= reserve;
            if (!asked && reserve > slot.Capacity && !TryMoveToPooledSlot(requestId, ref slot, reserve))
            {
                int target = RoomySlotTarget(required, reserve, RunningSlots(slot), "a request's reservation", slot.Capacity);
                if (target > slot.Capacity) GrowSlotRolling(slot, target);
            }
            _fittedReservations[requestId] = Math.Max(tried, reserve);
            if (required > slot.Capacity) EnsureSlotCapacity(slot, required, wanted: reserve);
            return slot.Length;
        }

        /// <summary>Reservations fitted to the device (tests: a request's later steps do not ask again).</summary>
        internal int FittedReservationsForTests => _fittedReservations?.Count ?? 0;

        /// <summary>
        /// A request whose slot holds rows (the prefix it adopted: a retained conversation's slot, exactly its length) but not
        /// its reservation takes a pooled slot that holds the reservation, the rows copied there on the device and the pooled
        /// slot's recordings adopted where they are; its own slot goes back to the pool. Grown in place instead, the slot got new
        /// caches, a copy and lost its recordings: four Qwen3.8-27B conversations at once on an RTX PRO 4000 each grew their
        /// second turn's slot from 512 to 1,024 cells, 15-30 ms of its first token (the direct engine's slots never grew).
        /// False where no pooled slot holds the reservation (the slot then grows).
        /// </summary>
        private bool TryMoveToPooledSlot(string requestId, ref Slot slot, int reserve)
        {
            if (slot.Length <= 0 || _slotPool == null || PooledSlotMoveDisabledForTests) return false;
            int capacity = (int)Math.Min(_maxContextLength, (reserve + (long)ReservationStepTokens - 1) / ReservationStepTokens * ReservationStepTokens);
            if (!_slotPool.Exists(p => p.Capacity >= capacity)) return false;
            Slot moved = CopySlot(slot, capacity, runSlotFirst: true);
            RepointDependents(slot.K, moved);   // the copy holds every row the slot's checkpoints share
            DropReplaysOf(slot.K, slot.Recurrent);
            _slots[requestId] = moved;
            RecycleSlot(slot);
            slot = moved;
            PooledSlotMoves++;
            return true;
        }

        /// <summary>Requests moved into a pooled slot instead of growing theirs (tests check the route engaged).</summary>
        internal int PooledSlotMoves { get; private set; }

        /// <summary>Tests: a request's slot grows in place as before.</summary>
        internal bool PooledSlotMoveDisabledForTests { get; set; }

        public bool TryForwardUnified(IReadOnlyList<string> requestIds, IReadOnlyList<int[]> tokens, int[] starts, float[][] lastRowLogits,
            float[][] features = null)
        {
            if (!TryForwardSlots(requestIds, tokens, starts, lastRowLogits, features)) return false;
            ReplenishSpareSlot(tokens);
            return true;
        }

        /// <summary>Decode rows whose input is an embedding row and steps that return the last rows' final hidden rows serve
        /// wherever slot steps do (graphs on every executor, and the per-op layers).</summary>
        public bool SupportsUnifiedRowIO => SupportsUnifiedBatching;

        public bool TryForwardUnifiedRows(IReadOnlyList<string> requestIds, IReadOnlyList<int[]> tokens, int[] starts,
            float[][] inputRows, bool[] wantLogits, float[][] lastRowLogits, float[][] lastHidden)
        {
            int n = requestIds.Count;
            if (inputRows == null || wantLogits == null || lastHidden == null || inputRows.Length != n || wantLogits.Length != n || lastHidden.Length != n)
                throw new ArgumentException("one input row, logits flag and hidden buffer a request");
            for (int s = 0; s < n; s++)
                if (lastHidden[s] != null && lastHidden[s].Length < Config.HiddenSize)
                    throw new ArgumentException($"a hidden buffer of {lastHidden[s].Length} values, the model's rows {Config.HiddenSize}", nameof(lastHidden));
            var rowIO = new SlotRowIO
            {
                InputRows = Array.Exists(inputRows, r => r != null) ? inputRows : null,
                Logits = Array.Exists(wantLogits, w => w),
                Hidden = Array.Exists(lastHidden, h => h != null) ? lastHidden : null,
            };
            if (!TryForwardSlots(requestIds, tokens, starts, lastRowLogits, null, rowIO)) return false;
            ReplenishSpareSlot(tokens);
            return true;
        }

        /// <summary>Set while <see cref="ForwardUnifiedMediaRows"/> runs: the prompt forward also writes its last final hidden
        /// row here.</summary>
        private float[] _promptHiddenOut;

        public float[] ForwardUnifiedMediaRows(string requestId, int[] tokens, int start, float[] lastHidden)
        {
            ArgumentNullException.ThrowIfNull(lastHidden);
            if (lastHidden.Length < Config.HiddenSize)
                throw new ArgumentException($"a hidden buffer of {lastHidden.Length} values, the model's rows {Config.HiddenSize}", nameof(lastHidden));
            _promptHiddenOut = lastHidden;
            try { return ForwardUnifiedMedia(requestId, tokens, start); }
            finally { _promptHiddenOut = null; }
        }

        /// <summary>On executors that return greedy ids (<see cref="TensorSharp.Graph.GraphCapabilities.GreedyIdOutputs"/>: the CUDA executor, GGML
        /// on discrete GPUs), a greedy step's head ends in a top-1 over each sequence's last row, read back as an id: a plain
        /// decode step had read back Qwen3.8's 248,320 logits (1 MB) and scanned them on the host.</summary>
        public bool SupportsUnifiedGreedyIds => GraphGreedyIds && !_rankGraphs && _mesh.Stages == 1;

        /// <summary>This model's graphs may end in greedy ids (<see cref="TensorSharp.Graph.GraphCapabilities.GreedyIdOutputs"/>) - except
        /// tensor-parallel graphs on ggml's meta device, whose head reads its logits.</summary>
        private bool GraphGreedyIds => GraphExecutor?.Capabilities.GreedyIdOutputs == true && _graphTpRanks == null;

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
            ReplenishSpareSlot(tokens);
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

        /// <summary>
        /// A finished request's retention: its slot is donated to the retained set (zero copy), and the conversation's next turn
        /// takes it. Where recorded steps bind a slot's buffers (<see cref="SlotBoundPolicies"/>, the CUDA executor) that turn's
        /// decode steps and verify windows then replay the recordings the request parked there (<see cref="Retained"/> keeps the
        /// buffers they bind). There a short slot is copied instead (<see cref="RetainsAsCopy"/>), into a pooled slot that holds
        /// it: the slot stays the request's, so its release returns it to the pool and the next request - another
        /// conversation's, as a benchmark's short prompts are - runs on it, its recordings adopted where they are. Donated, each
        /// such request ran in a new slot (Qwen3.8-27B, a short prompt: 52 against 39 ms to the first token; Qwen3.8 Flash Next
        /// ISTA + MTP after a 4K prompt: 102 against 130 tok/s over 64 tokens without the prefix cache). Copied whatever its size,
        /// a long conversation paid a whole slot's allocation and copy at every turn's end (3.3 GB for Qwen3.8-27B at 47K tokens,
        /// seconds on a device near full) and its next turn, on the copy, recorded its windows again (issue #376).
        /// </summary>
        bool IPrefixCacheModel.TryCaptureDonate(string requestId, string payloadKey, int length, out PayloadFootprint footprint)
        {
            footprint = default;
            LastRetentionCopied = false;
            if (RetainsAsCopy(requestId, payloadKey, length, out Slot slot))
            {
                _retainedSlots ??= new Dictionary<string, Slot>(StringComparer.Ordinal);
                if (_retainedSlots.ContainsKey(payloadKey)) return false;
                Slot copy = CopySlot(slot, slot.Capacity);
                // The request's checkpoints share the copy's rows (the same bytes), so its own slot still goes back to the pool.
                RepointDependents(slot.K, copy);
                _retainedSlots.Add(payloadKey, Retained(copy));
                footprint = MeasureEndState(payloadKey);
                RetainedByCopy++;
                LastRetentionCopied = true;
                return true;
            }
            return HolderPrefixCacheAdapter.TryCaptureDonate(this, requestId, payloadKey, length, out footprint);
        }

        /// <summary>Finished requests retained as a copy of their slot (tests check the route engaged).</summary>
        internal int RetainedByCopy { get; private set; }

        /// <summary>The rows a finished request's slot holds at most where it is retained as a copy (<see cref="RetainsAsCopy"/>):
        /// about 7.7K tokens of Qwen3.8-27B's F16 K/V, a few milliseconds of device copy.</summary>
        private const long RetainCopyMaxBytes = 512L << 20;

        /// <summary>Tests stand in for <see cref="RetainCopyMaxBytes"/>.</summary>
        internal long? RetainCopyMaxBytesForTests { get; set; }

        /// <summary>
        /// Whether a finished request's slot of exactly <paramref name="length"/> rows is retained as a copy: where recorded steps
        /// bind slots, its rows are at most <see cref="RetainCopyMaxBytes"/>, and a pooled slot at least as large as it takes the
        /// copy, so nothing is allocated (a copy keeps two slots where a donation keeps one, which beside a full expert tier cost
        /// more than the recordings it kept: Qwen3.8 Flash Next after a 16K prompt on an A40, 52.0 against 54.9 tok/s).
        /// </summary>
        private bool RetainsAsCopy(string requestId, string payloadKey, int length, out Slot slot)
        {
            slot = null;
            if (!SlotBoundPolicies || !SupportsPrefixCheckpoints || string.IsNullOrEmpty(payloadKey) || length <= 0
                || !TryGetSlot(requestId, out slot) || slot.Length != length || _slotBytesPerToken < 0
                || (long)slot.Length * _slotBytesPerToken > (RetainCopyMaxBytesForTests ?? RetainCopyMaxBytes))
                return false;
            // The copy takes this capacity (CopySlot), a pooled slot's: the conversation's next turn holds as much as on its own.
            int capacity = AlignedSlotCapacity(Math.Max(FreshSlotTokens, slot.Capacity));
            return _slotPool?.Exists(p => p.Capacity >= capacity) == true;
        }

        /// <summary>
        /// Whether the device has room for two more slots of <paramref name="capacity"/> cells beside its headroom and a lone
        /// prompt's step (its rows' activations and scratch, <see cref="CudaPromptStepTokens"/>); null where it is not known. A
        /// slot the prefix cache's copies take is one the next prompt cannot use: beside only the headroom, two spare 4K slots
        /// beside Qwen3.8 Flash Next's expert tier on an A40 left the pool returning its blocks to the driver during the next
        /// prompts (a 4K prompt 2.35 against 1.98 s without the prefix cache), as the direct engine, which kept no copies there,
        /// never did.
        /// </summary>
        private bool? RoomForTwoSlots(int capacity) => RoomForSlots(capacity, 2);

        /// <summary>Retained slots a new request took over (tests check the route engaged).</summary>
        internal int RetainedSlotsReclaimed { get; private set; }

        /// <summary>
        /// Where recorded steps bind slots (the CUDA executor), a new request whose device has no pooled slot for it and no room
        /// for a fresh one beside its headroom and a prompt's step takes the oldest retained slot (a finished request the
        /// prefix cache keeps), its payload reclaimed (DEC-23), as the direct engines' slot arenas did when full; null
        /// otherwise. Allocated instead, the new slot held none of the recordings the retained one parked, so every request
        /// recorded its steps again (Qwen3.8 Flash Next UD-Q2_K_XL on an A40 with the prefix cache: each request's slot was new,
        /// 2-4% of its decode spent recording), and the allocation sat in its time to the first token.
        /// </summary>
        /// <param name="roomShort">The device was found short of a fresh slot's room (<see cref="NewSlotWithRoom"/>): only a
        /// retained slot that holds the request is taken, which then allocates nothing.</param>
        private Slot ReclaimRetainedSlot(int reserve, bool roomShort = false)
        {
            if (GraphExecutor?.Capabilities.SlotBoundRecordings != true || _retainedSlots is not { Count: > 0 }) return null;
            int capacity = AlignedSlotCapacity(Math.Max(FreshSlotTokens, reserve));
            if (_slotPool != null && _slotPool.Exists(p => p.Capacity >= capacity)) return null;
            if (!roomShort && RoomForSlots(capacity, 1) != false) return null;
            string key = null;
            Slot oldest = null;
            foreach (var (k, retained) in _retainedSlots)
                if (retained.K != null && !InAnyShare(retained) && (!roomShort || retained.Capacity >= capacity)
                    && (oldest == null || retained.RetainedAt < oldest.RetainedAt))
                {
                    key = k;
                    oldest = retained;
                }
            if (oldest == null) return null;
            _retainedSlots.Remove(key);
            _prefixCacheSink?.OnPayloadInvalidated(key, Runtime.Scheduling.PrefixCache.InvalidationReason.NativeSlotReclaimed);
            if (!ResetStatesInPlace(oldest.Recurrent)) ForgetGraphsOf(oldest);
            oldest.Spare = false;
            oldest.Length = 0;
            oldest.DraftStart = 0;
            oldest.DraftFollowed = -1;
            oldest.RopeDelta = 0;
            oldest.VerifyRows = 0;
            RetainedSlotsReclaimed++;
            return oldest;
        }

        /// <summary>
        /// Whether every device that holds part of a slot has room for <paramref name="count"/> more slots' share of it (their
        /// parts of <paramref name="capacity"/> cells) beside its headroom and a lone prompt's step; null where one is not known.
        /// A layer split's or a rank's devices each hold their layers' or heads' part, and each runs the prompt's rows: judged
        /// by the first device's memory alone, a split whose other GPU was the fuller one kept spare slots that device could
        /// not hold.
        /// </summary>
        private bool? RoomForSlots(int capacity, int count)
        {
            if (_slotShares == null || _slotShares.Length == 0) return null;
            foreach (var (stage, rank, perToken, states) in _slotShares)
            {
                long free, total;
                if (SlotDeviceMemoryForTests?.Invoke(_mesh.GlobalDevice(stage, rank)) is { } stand)
                    (free, total) = stand;
                else if (_mesh.Allocator(stage, rank) is not IDeviceMemoryInfo memory || memory.SharesHostMemory
                    || !memory.TryGetDeviceMemory(out free, out total))
                    return null;
                if (free < count * (perToken * capacity + states + SlotExtraBytes) + DeviceHeadroom(total) + PromptStepBytes)
                    return false;
            }
            return true;
        }

        /// <summary>Tests stand in for each device's free and total bytes (by its index in the mesh) the slots made for the
        /// prefix cache and new requests are fitted to.</summary>
        internal Func<int, (long Free, long Total)?> SlotDeviceMemoryForTests { get; set; }

        /// <summary><see cref="RoomForSlots"/> (tests).</summary>
        internal bool? RoomForSlotsForTests(int capacity, int count) => RoomForSlots(capacity, count);

        /// <summary>Each device's share of a slot (tests): its place in the mesh, its caches' bytes per token, its states' bytes.</summary>
        internal (int Stage, int Rank, long PerToken, long States)[] SlotSharesForTests => _slotShares;

        /// <summary>A lone prompt's step on direct CUDA: its widest rows' activations and its scratch (0 elsewhere).</summary>
        private long PromptStepBytes => _backend == BackendType.Cuda
            ? (_cudaPromptRows > 0 ? _cudaPromptRows : MaxCudaPromptStepRows) * CudaPromptRowBytes() + CudaPromptReserveBytes
            : 0;

        // Decode steps to wait after the device had no room for a spare slot (a device memory query costs milliseconds).
        private int _spareSlotWait;

        // A spare's making waits for the next recorded step's launch.
        private bool _sparePending;

        // The largest slot a request has run in: a spare holds that request's copies (see ReplenishSpareSlot).
        private int _spareCapacity;

        /// <summary>Tests: <see cref="SpareCapacity"/>.</summary>
        internal int SpareCapacityForTests => SpareCapacity;

        /// <summary>A spare slot's cells: as large as the largest slot a request has run in. Where slots are sized to the
        /// device's room (<see cref="SizesSlotsToRoom"/>) only slots whose finished request's copy fits a spare are counted
        /// (<see cref="RetainsShortSlotCopy"/>: up to <see cref="RetainCopyMaxBytes"/> of K/V), and a spare is at least the
        /// smallest class: a long conversation's slot is donated, not copied, so a spare of its size (gigabytes at long context)
        /// only took the room the next prompt needed, while a short slot's copy still takes a spare (<see cref="RetainsAsCopy"/>).</summary>
        private int SpareCapacity => SizesSlotsToRoom
            ? Math.Max(SlotCapacityClass(FreshSlotTokens), AlignedSlotCapacity(Math.Max(FreshSlotTokens, _spareCapacity)))
            : AlignedSlotCapacity(Math.Max(FreshSlotTokens, _spareCapacity));

        /// <summary>Whether a finished request's slot of <paramref name="capacity"/> cells is short enough to be retained as a copy
        /// (<see cref="RetainsAsCopy"/>): its K/V at most <see cref="RetainCopyMaxBytes"/>.</summary>
        private bool RetainsShortSlotCopy(int capacity)
            => _slotBytesPerToken >= 0 && (long)capacity * _slotBytesPerToken <= (RetainCopyMaxBytesForTests ?? RetainCopyMaxBytes);

        /// <summary>Spares the pool keeps: one for the next request's prompt-end checkpoint, one for its retained copy.</summary>
        private const int SpareSlotsWanted = 2;

        /// <summary>
        /// After a step that read no prompt rows (a decode or a verify window), where recorded steps bind slots
        /// (<see cref="TensorSharp.Graph.GraphCapabilities.SlotBoundRecordings"/>) and a prefix cache copies slots, the pool gets
        /// fresh slots, up to <see cref="SpareSlotsWanted"/>, while the device has room for each and one more beside its headroom
        /// and a prompt's step (<see cref="RoomForTwoSlots"/>): a request's prompt-end checkpoint and its retained copy take them
        /// instead of allocating. A slot is a few hundred tensors from the device allocator, made a slice at a time while the
        /// device runs the next recorded steps (<see cref="TensorSharp.Graph.IGraphExecutor.RunWhileNextRecordedStepRuns"/>,
        /// <see cref="MakeSpareSlot"/>), where the host has nothing to do. Made at once, after one decode step in 256 and only for
        /// plain decode, a spare of 512 cells held neither copy of a 4K prompt: Qwen3.8-27B on an RTX PRO 4000 spent 19-104 ms
        /// allocating at the first decode step of each 4K request (DFlash2: 79.1 against 82.3 tok/s without the prefix cache) and
        /// 12.4 of the 15 ms between short MTP requests. A spare is as large as the largest slot a request has run in. After the
        /// device had no room, the next 256 steps do not ask again.
        /// </summary>
        private void ReplenishSpareSlot(IReadOnlyList<int[]> tokens)
        {
            for (int i = 0; i < tokens.Count; i++)
                if (tokens[i].Length != 1) return;
            ReplenishSpareSlot();
        }

        /// <inheritdoc cref="ReplenishSpareSlot(IReadOnlyList{int[]})"/>
        private void ReplenishSpareSlot()
        {
            // Without a prefix cache nothing copies a slot: a finished request's slot goes back to the pool whole.
            if (GraphExecutor is not { Capabilities.SlotBoundRecordings: true } executor || _prefixCacheSink == null || _slotStateBytes < 0
                || _sparePending)
                return;
            if (_slots != null)
                foreach (Slot running in _slots.Values)
                    if (running?.K != null && (!SizesSlotsToRoom || RetainsShortSlotCopy(running.Capacity)))
                        _spareCapacity = Math.Max(_spareCapacity, running.Capacity);
            int capacity = SpareCapacity;
            int spares = 0;
            if (_slotPool != null)
                foreach (Slot pooled in _slotPool)
                    if (pooled.Spare && pooled.Capacity >= capacity) spares++;
            if (spares >= SpareSlotsWanted) return;
            if (_spareSlotWait > 0)
            {
                _spareSlotWait--;
                return;
            }
            QueueSpareSlice(executor, capacity);
        }

        /// <summary>The next slice of a spare beside the next recorded step, and after it the next, until the spare is made.</summary>
        private void QueueSpareSlice(TensorSharp.Graph.IGraphExecutor executor, int capacity)
        {
            _sparePending = executor.RunWhileNextRecordedStepRuns(() =>
            {
                _sparePending = false;
                if (MakeSpareSlot(capacity)) QueueSpareSlice(executor, capacity);
            });
        }

        // The spare being made, a slice of its caches and states beside each recorded step, and how far it has come.
        private Slot _spareBuild;
        private int _spareBuildNext;

        /// <summary>Host time a spare slot's making takes beside one recorded step: a slot is a few hundred tensors, 15-20 ms of
        /// allocation for Qwen3.8-27B's 4K slot on an RTX PRO 4000 and more beside Qwen3.8 Flash Next's expert tier on an A40,
        /// where a decode step's device time is 17 ms; made at once beside one step, the rest held the next step's outputs
        /// (six 25-84 ms steps in each 256-token request).</summary>
        private static readonly long SpareSliceTicks = System.Diagnostics.Stopwatch.Frequency * 3 / 1000;

        /// <summary>
        /// A slice of a fresh spare slot of <paramref name="capacity"/> cells: the next cache entries and recurrent layers until
        /// <see cref="SpareSliceTicks"/> have gone by, the finished slot into the pool. True when more remains (the caller
        /// queues the next slice beside the next recorded step). A spare is begun only where the device has room for it and
        /// one more slot beside its headroom and a prompt's step.
        /// </summary>
        private bool MakeSpareSlot(int capacity)
        {
            if (_spareBuild == null)
            {
                if (_slotStateBytes < 0 || (_slotPool?.Count ?? 0) >= Math.Min(SlotPoolMax, ExecutionOptions.FromEnvironment().KvHolderPoolMax))
                    return false;
                if (RoomForTwoSlots(capacity) != true)
                {
                    _spareSlotWait = 256;
                    return false;
                }
                _spareBuild = new Slot
                {
                    Capacity = capacity, K = new Tensor[CacheEntries][], V = new Tensor[CacheEntries][],
                    Recurrent = _anyRecurrent ? new RecurrentState[Config.NumLayers][] : null,
                };
                _spareBuildNext = 0;
            }
            Slot slot = _spareBuild;
            long start = System.Diagnostics.Stopwatch.GetTimestamp();
            int layers = _anyRecurrent ? Config.NumLayers : 0;
            // A spare is a convenience: the allocator's reclaimer releases no conversation for it.
            _slotMutationDepth++;
            NoteDeviceMemoryChanged();
            try
            {
                while (_spareBuildNext < CacheEntries + layers)
                {
                    if (_spareBuildNext < CacheEntries) AllocateCacheEntry(slot.K, slot.V, _spareBuildNext, slot.Capacity);
                    else NewRecurrentLayer(slot.Recurrent, _spareBuildNext - CacheEntries);
                    _spareBuildNext++;
                    if (System.Diagnostics.Stopwatch.GetTimestamp() - start >= SpareSliceTicks) break;
                }
            }
            finally
            {
                _slotMutationDepth--;
            }
            if (_spareBuildNext < CacheEntries + layers) return true;
            _spareBuild = null;
            PrimeDeviceCopies(slot, overwritten: false);
            slot.Spare = true;
            RecycleSlot(slot);
            SpareSlotsMade++;
            return false;
        }

        /// <summary>Spare slots made during decode or while idle (tests check the route engaged).</summary>
        internal int SpareSlotsMade { get; private set; }

        /// <summary>Slots in the pool, and how many it keeps at most (tests).</summary>
        internal (int Pooled, int Cap) SlotPoolForTests => (_slotPool?.Count ?? 0, Math.Min(SlotPoolMax, ExecutionOptions.FromEnvironment().KvHolderPoolMax));

        // The device had no room for an idle spare: none is tried again until a slot is given back.
        private bool _idleSpareRefused;

        /// <summary>
        /// While the engine is idle, where recorded steps bind slots, the slot pool is filled to its cap with spare slots, a slice
        /// of one (<see cref="MakeSpareSlot"/>) a call, while the device has room for each and one more beside its headroom and
        /// a prompt's step. Spares are otherwise made beside recorded decode steps, two at most: the prefix cache's copies of
        /// the startup warm-up's requests left three slots pooled, and the fourth of four Qwen3.8-27B conversations sent
        /// together allocated its slot inside their prompt step, 13-39 ms before any of their first tokens (an RTX PRO 6000
        /// slice; the step 131-160 ms, 110-117 with the pool filled), where the direct engines' slots were made at load.
        /// </summary>
        public bool RunIdleSlice()
        {
            if (GraphExecutor is not { Capabilities.SlotBoundRecordings: true } || _slotStateBytes < 0 || !SupportsUnifiedBatching)
                return false;
            // A shared checkpoint's states first (StandardDecoderModel.RowShares.cs): the next prompt's end takes them. Not held
            // back by a spare slot's slice waiting for a recorded step: a request's last decode step queues one, and the
            // next request's prompt end came with no states made.
            if (BuildSpareCheckpointStates()) return true;
            if (_sparePending || _idleSpareRefused) return false;
            int cap = Math.Min(SlotPoolMax, ExecutionOptions.FromEnvironment().KvHolderPoolMax);
            if (_spareBuild == null && (_slotPool?.Count ?? 0) >= cap) return false;
            int capacity = _spareBuild?.Capacity ?? SpareCapacity;
            if (_spareBuild == null && RoomForTwoSlots(capacity) != true)
            {
                _idleSpareRefused = true;
                return false;
            }
            return MakeSpareSlot(capacity) || (_slotPool?.Count ?? 0) < cap;
        }

        /// <summary>Requests run in slots of their own: <see cref="IPrefixCacheModel.TryCaptureCopy"/> reads a request's slot
        /// whether or not it is bound.</summary>
        bool IPrefixCacheModel.CapturesUnboundRequests => _slots != null;

        /// <summary>The request's own slot, whichever cache is active (a unified step leaves the primary active, and a
        /// checkpoint of it would be another request's state).</summary>
        bool IPrefixCacheModel.TryCaptureCopy(string requestId, string payloadKey, out PayloadFootprint footprint)
        {
            footprint = default;
            if (string.IsNullOrEmpty(requestId) || string.IsNullOrEmpty(payloadKey) || !SupportsPrefixCheckpoints) return false;
            if (!TryGetSlot(requestId, out Slot slot))
                return HolderPrefixCacheAdapter.TryCaptureCopy(this, requestId, payloadKey, out footprint);
            _retainedSlots ??= new Dictionary<string, Slot>(StringComparer.Ordinal);
            if (_retainedSlots.ContainsKey(payloadKey) || slot.Length <= 0 || !RoomForCheckpointCopy(slot)) return false;
            _retainedSlots.Add(payloadKey, Retained(CopySlot(slot, slot.Length)));
            footprint = MeasureEndState(payloadKey);
            return true;
        }

        /// <summary>Checkpoints declined for want of device room (tests check the guard).</summary>
        internal int CheckpointsDeclinedForRoom { get; private set; }

        /// <summary>
        /// Where recorded steps bind slots (the CUDA executor), a checkpoint of <paramref name="length"/> tokens is copied into a
        /// pooled slot that holds it, or where the device has room for the copy and one more slot as large beside its headroom
        /// (as a spare slot is made); otherwise there is none. The copy's slot was allocated where the
        /// device had no room for it: after a 16K prompt beside Qwen3.8 Flash Next's expert tier on an A40 the allocator first
        /// gave the prompt's cached blocks back to the driver, 190-200 ms at the first decode step (52.9 against 56.5 tok/s
        /// without the prefix cache), where the direct engine took no checkpoint at all; after a 4K prompt the copy fit beside
        /// the headroom alone and still cost the decode 2 tok/s. On a device this model manages, the copy's room is made as a
        /// slot's, finished conversations released for it the oldest first (the newest checkpoint is worth more than the oldest
        /// conversation: declined, a conversation's next turn prefilled its whole prompt again, issue #376), and it is declined
        /// only where even that made no room.
        /// </summary>
        private bool RoomForCheckpointCopy(Slot slot)
        {
            if (DeviceRoomManaged)
            {
                if (RoomForSlotCopy(slot, slot.Length, mayEvict: true, needed: false, "a prompt-end checkpoint")) return true;
                CheckpointsDeclinedForRoom++;
                return false;
            }
            if (GraphExecutor?.Capabilities.SlotBoundRecordings != true) return true;
            int capacity = AlignedSlotCapacity(Math.Max(FreshSlotTokens, slot.Length));
            if (_slotPool != null && _slotPool.Exists(p => p.Capacity >= capacity)) return true;
            if (RoomForTwoSlots(capacity) != false) return true;
            CheckpointsDeclinedForRoom++;
            return false;
        }
    }
}
