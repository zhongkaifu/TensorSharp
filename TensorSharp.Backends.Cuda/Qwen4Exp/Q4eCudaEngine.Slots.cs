// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Sequence slots of the Qwen3.8-Flash-Next direct-CUDA engine: one set of caches per sequence
// (the GDN convolution and recurrent states, the attention K/V, the QSA raw and block keys, the
// PLE history), and a decode step that advances several slots at once.
//
// A delta recurrence cannot be rewound, so a slot moves back only to position 0 (a reset) or
// stays where it is.
using System;
using System.Collections.Generic;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    public sealed unsafe partial class Q4eCudaEngine
    {
        /// <summary>Sequences one batched decode step can take.</summary>
        public const int MaxBatchedDecodeRows = 16;

        /// <summary>What every device keeps free beside the slots: the matmul scratch the shared ops
        /// grow on demand.</summary>
        private const long SlotReserveBytes = 1L << 30;

        private readonly Dictionary<int, Slot> _slots = new Dictionary<int, Slot>();
        private int _nextSlotId;

        /// <summary>Bytes of one device's per-row state pointer table: [layer][conv, ssm][row], then
        /// the PLE histories [row].</summary>
        private long BatchStateTableBytes => ((long)_m.NLayer * 2 + 1) * MaxBatchedDecodeRows * 8;

        private static IntPtr BatchStatePtr(Dev dev, int il, int kind)
            => (IntPtr)((long)Ptr(dev.BatchStatePtrs) + ((long)il * 2 + kind) * MaxBatchedDecodeRows * 8);

        /// <summary>The active slot's id.</summary>
        public int ActiveSlot => _active.Id;

        /// <summary>A new, empty slot, not made active; -1 when the devices have no room for it beside
        /// what the running sequences still need.</summary>
        public int SlotAlloc()
        {
            DrainDecodeLanes();
            if (!SlotFits())
            {
                // Captured batched steps are a cache, and their executable graphs hold device memory: a new sequence needs it
                // more. In Strata's concurrency rounds the graphs of earlier rounds left the fourth request no slot, and it
                // waited 4.2-4.6 s for a running one to finish.
                // Only when the graphs could make up the difference: with one lane every new request's slot falls short
                // by a whole slot, and giving up the decode graphs there only made each request capture them again (139
                // of 1,665 verify windows ran uncaptured, short-prompt decode 122-171 -> 95-116 tok/s).
                if (SlotShortfall() > GraphReleaseBytes)
                    return RefuseSlot();
                bool fits = false;
                while (!fits && ReleaseBatchGraphsForMemory())
                    fits = SlotFits();
                if (!fits)
                    return RefuseSlot();
            }
            Slot slot;
            try
            {
                slot = CreateSlot();
            }
            catch (OutOfMemoryException)
            {
                return -1;
            }
            _slots[slot.Id] = slot;
            CountSlot(slot, +1);
            return slot.Id;
        }

        /// <summary>Whether one more slot fits, counting what the allocators hold pooled: its caches
        /// at the initial capacity, plus room for one slot to grow to the whole context, so the
        /// sequences already running are not left without memory to grow into.</summary>
        private bool SlotFits() => SlotShortfall() <= 0;

        private long _slotRefusalNoted;

        /// <summary>-1, saying (at most every 30 s) how far the devices are from another slot: the model's own message names
        /// only the running requests.</summary>
        private int RefuseSlot()
        {
            long now = Environment.TickCount64;
            if (now - _slotRefusalNoted >= 30_000)
            {
                _slotRefusalNoted = now;
                long need = 0;
                for (int d = 0; d < _devs.Length; d++)
                    need = Math.Max(need, _slotInitialBytes[d] + _slotGrowBytes[d]);
                Console.Error.WriteLine($"[q4e-cuda] no room for another sequence slot: {SlotShortfall() / 1048576.0:F0} MiB short of "
                    + $"{need / 1048576.0:F0} MiB plus the {SlotReserveBytes / 1048576.0:F0} MiB reserve ({_slots.Count} slots, "
                    + $"{_graphs.Count} solo and {_batchGraphs.Count} batched graphs held)");
            }
            return -1;
        }

        /// <summary>Bytes the device shortest of room lacks for one more slot beside the reserve (0 or less when it fits),
        /// counting what the allocators hold pooled.</summary>
        private long SlotShortfall()
        {
            long worst = long.MinValue;
            for (int d = 0; d < _devs.Length; d++)
            {
                long need = _slotInitialBytes[d] + _slotGrowBytes[d];
                if (need == 0)
                    continue;
                _devs[d].MakeCurrent();
                (long free, _) = _devs[d].Alloc.GetMemoryInfo();
                free += _devs[d].Alloc.GetStats().CachedBytes;
                worst = Math.Max(worst, need + SlotReserveBytes - free);
            }
            return worst == long.MinValue ? 0 : worst;
        }

        /// <summary>A shortfall the captured graphs could make up (they hold well under this): beyond it they are kept.</summary>
        private const long GraphReleaseBytes = 1L << 30;

        // ---------------------------------------------------------------- growing slots
        //
        // A slot used to hold its attention and indexer caches for the whole context from the start
        // (~29 KB a token over the 12 attention layers, ~7.2 GiB at 262,144), so two H100 NVL beside
        // the Q8_0 weights held 7 sequences, while ten coding agents' conversations of 20-150K tokens
        // each needed a fraction of that. With fewer slots than agents, every new turn took the slot
        // a waiting conversation was retained in, and nothing was ever reused. Now a slot starts at
        // TS_Q4E_SLOT_TOKENS (32,768 by default) and doubles as its sequence grows: the new caches
        // take the old ones' rows and the slot's captured graphs are rebuilt.

        /// <summary>Called when a slot cannot grow for lack of device memory: frees one retained
        /// sequence slot (the model's oldest) and returns true, or false when it has none.</summary>
        public Func<bool> ReclaimRetainedSlot { get; set; }

        /// <summary>One slot's caches at the initial capacity, per device.</summary>
        private long[] _slotInitialBytes;
        /// <summary>What one slot growing from the initial capacity to n_ctx takes, per device.</summary>
        private long[] _slotGrowBytes;

        /// <summary>The initial capacity: TS_Q4E_SLOT_TOKENS, rounded up to whole QSA blocks, at most n_ctx.</summary>
        private int InitialSlotTokens
            => Math.Min(_m.NCtx, AlignSlotTokens(Math.Max(1, EnvInt("TS_Q4E_SLOT_TOKENS", 32768))));

        private int AlignSlotTokens(long tokens)
        {
            int a = MaxRatio;
            return (int)Math.Min(_m.NCtx, (tokens + a - 1) / a * a);
        }

        /// <summary>Make <paramref name="s"/> hold at least <paramref name="need"/> positions (at most
        /// n_ctx, which the callers check first). Throws when the devices have no room even after
        /// the retained slots are given back.</summary>
        private void EnsureCapacity(Slot s, int need)
        {
            if (need <= s.Capacity)
                return;
            int cap = AlignSlotTokens(Math.Max(need, 2L * s.Capacity));
            while (true)
            {
                try
                {
                    GrowSlot(s, cap);
                    return;
                }
                catch (Exception ex) when (IsOutOfMemory(ex))
                {
                    // The captured batched steps give their memory back first (BatchGraphs.cs), then the doubling shrinks to
                    // what this step needs, then a retained slot is reclaimed.
                    if (ReleaseBatchGraphsForMemory())
                        continue;
                    int least = AlignSlotTokens(need);
                    if (cap > least)
                        cap = least;   // the doubling does not fit; take only what this step needs
                    else if (ReclaimRetainedSlot?.Invoke() != true)
                    {
                        System.Threading.Interlocked.Increment(ref _tGrowFailures);
                        throw new InvalidOperationException(
                            $"[q4e-cuda] no device memory to grow sequence slot {s.Id} from {s.Capacity} to {need} tokens", ex);
                    }
                }
            }
        }

        private static bool IsOutOfMemory(Exception ex)
            => ex is OutOfMemoryException
               || (ex is CudaException cuda && cuda.ErrorCode == 2 /* CUDA_ERROR_OUT_OF_MEMORY */);

        /// <summary>Move every attention layer's caches of <paramref name="s"/> into buffers of
        /// <paramref name="cap"/> positions, all allocated before any is swapped, so a failure leaves
        /// the slot as it was. Its captured graphs read the old buffers and go.</summary>
        private void GrowSlot(Slot s, int cap)
        {
            CountSlot(s, -1);
            try
            {
                GrowSlotCaches(s, cap);
            }
            finally
            {
                CountSlot(s, +1);
            }
            System.Threading.Interlocked.Increment(ref _tGrows);
        }

        private void GrowSlotCaches(Slot s, int cap)
        {
            var fresh = new List<(int Layer, SlotLayer Caches, long[] Bytes)>();
            try
            {
                for (int il = 0; il < _m.NLayer; il++)
                {
                    // A streamed layer's window and host copy already cover the whole context.
                    if (_layers[il].Recurrent || s.Layers[il]?.Kv != null)
                        continue;
                    var bytes = new long[_devs.Length];
                    fresh.Add((il, AllocAttentionCaches(_devs[_layers[il].Device], _layers[il], cap, bytes), bytes));
                }
            }
            catch
            {
                foreach (var f in fresh)
                    DisposeAttentionCaches(f.Caches);
                throw;
            }
            foreach (var dev in _devs)
            {
                dev.MakeCurrent();
                CudaDriverApi.cuStreamSynchronize(dev.Stream).ThrowOnError();
            }
            DestroySlotGraphs(s.Id);
            foreach (var (il, n, bytes) in fresh)
            {
                var dev = _devs[_layers[il].Device];
                dev.MakeCurrent();
                var old = s.Layers[il];
                CopyPrefix(dev, n.KCache, old.KCache);
                CopyPrefix(dev, n.VCache, old.VCache);
                if (old.KScale != null)
                {
                    // Q8_0 KV: the scales are position-major like the codes (one per 32 values of a row).
                    CopyPrefix(dev, n.KScale, old.KScale);
                    CopyPrefix(dev, n.VScale, old.VScale);
                }
                if (old.IdxRaw != null)
                {
                    CopyPrefix(dev, n.IdxRaw, old.IdxRaw);
                    CopyPrefix(dev, n.IdxPooled, old.IdxPooled);
                }
                CudaDriverApi.cuStreamSynchronize(dev.Stream).ThrowOnError();
                foreach (Tensor t in new[] { old.KCache, old.VCache, old.KScale, old.VScale, old.IdxRaw, old.IdxPooled })
                {
                    if (t == null)
                        continue;
                    s.DeviceBytes[dev.Ordinal] -= t.ElementCount() * t.ElementType.Size();
                    s.Owned.Remove(t);
                    t.Dispose();
                }
                old.KCache = n.KCache; old.VCache = n.VCache; old.KScale = n.KScale; old.VScale = n.VScale;
                old.IdxRaw = n.IdxRaw; old.IdxPooled = n.IdxPooled;
                foreach (Tensor t in new[] { n.KCache, n.VCache, n.KScale, n.VScale, n.IdxRaw, n.IdxPooled })
                    if (t != null)
                        s.Owned.Add(t);
                for (int d = 0; d < bytes.Length; d++)
                    s.DeviceBytes[d] += bytes[d];
            }
            s.Capacity = cap;
        }

        /// <summary>The whole of <paramref name="src"/> into the start of the larger <paramref name="dst"/>.</summary>
        private static void CopyPrefix(Dev dev, Tensor dst, Tensor src)
            => CudaDriverApi.cuMemcpyDtoDAsync(Ptr(dst), Ptr(src),
                new UIntPtr((ulong)src.ElementCount() * (ulong)src.ElementType.Size()), dev.Stream).ThrowOnError();

        private static void DisposeAttentionCaches(SlotLayer c)
        {
            c.KCache?.Dispose(); c.VCache?.Dispose(); c.KScale?.Dispose(); c.VScale?.Dispose();
            c.IdxRaw?.Dispose(); c.IdxPooled?.Dispose();
        }

        public bool SetActiveSlot(int id)
        {
            // Host state only: a prompt chunk in flight stays (its slot is selected again before its next chunk, and a
            // forward on any other slot waits for it).
            DrainLanes();
            if (!_slots.TryGetValue(id, out Slot slot))
                return false;
            _active = slot;
            return true;
        }

        /// <summary>
        /// Free every slot but the active one and those in <paramref name="keep"/> (the slots the owner still holds:
        /// bound to requests, its primary, retained conversations), returning how many went. A slot the owner lost track
        /// of (its free was refused while it was the selection) otherwise held a lane's memory for good: with 4 lanes on
        /// the RTX 5090 one such slot made a new request wait for a running one to finish.
        /// </summary>
        public int SlotFreeAllExcept(ICollection<int> keep)
        {
            DrainDecodeLanes();
            var lost = new List<int>();
            foreach (var (id, slot) in _slots)
                if (!ReferenceEquals(slot, _active) && !keep.Contains(id))
                    lost.Add(id);
            int freed = 0;
            foreach (int id in lost)
                if (SlotFree(id))
                    freed++;
            return freed;
        }

        /// <summary>Free a slot; the active slot cannot be freed.</summary>
        public bool SlotFree(int id)
        {
            DrainDecodeLanes();
            if (!_slots.TryGetValue(id, out Slot slot) || ReferenceEquals(slot, _active))
                return false;
            FreeSlot(slot);
            _slots.Remove(id);
            return true;
        }

        /// <summary>Head of a slot without selecting it; false when there is no such slot.</summary>
        public bool SlotHead(int id, out int head)
        {
            DrainLanes();
            head = 0;
            if (!_slots.TryGetValue(id, out Slot slot))
                return false;
            head = slot.NPast;
            return true;
        }

        /// <summary><see cref="Reset"/> that reports whether the active slot is usable at position 0.</summary>
        public bool ResetChecked()
        {
            DrainDecodeLanes();
            try
            {
                Reset();
                foreach (var dev in _devs)
                {
                    dev.MakeCurrent();
                    CudaDriverApi.cuStreamSynchronize(dev.Stream).ThrowOnError();
                }
                return true;
            }
            catch (Exception ex) when (ex is InvalidOperationException || ex is CudaException)
            {
                return false;
            }
        }

        /// <summary>
        /// Move the active slot back to <paramref name="nPast"/>: its head (nothing to do), 0 (a
        /// reset), or any position from its rewind checkpoint up to its head. The delta recurrence
        /// cannot run backwards, so the last kind restores the checkpoint's states and then forwards
        /// the slot's own tokens from the checkpoint to <paramref name="nPast"/> again (attention
        /// rows below the head are never overwritten, so they still hold those positions).
        ///
        /// An agent's next turn renders the previous reply back into its prompt, with its reasoning
        /// and tool calls re-serialized, so it matches the slot only up to somewhere in that reply:
        /// before this the whole prompt was prefilled again every turn (0% reuse over 83 opencode
        /// requests of a Terminal-Bench run, against 87% for vLLM on the same tasks).
        /// </summary>
        public bool Rewind(int nPast)
        {
            DrainDecodeLanes();
            Slot s = _active;
            if (nPast == s.NPast)
                return true;
            if (nPast == 0)
            {
                System.Threading.Interlocked.Increment(ref _tRewindZero);
                return ResetChecked();
            }
            if (!CanRewind(s, s.NPast, nPast))
            {
                System.Threading.Interlocked.Increment(ref _tRewindRefused);
                return false;
            }
            int replay = nPast - s.CpNPast;
            if (replay > 0)
                System.Threading.Interlocked.Increment(ref _tRewindReplay);
            else
                System.Threading.Interlocked.Increment(ref _tRewindCheckpoint);
            System.Threading.Interlocked.Add(ref _tReplayTokens, replay);
            int[] tokens = replay > 0 ? s.PleTokens.Tokens.AsSpan(s.CpNPast, replay).ToArray() : null;
            try
            {
                foreach (var dev in _devs)
                {
                    dev.MakeCurrent();
                    CudaDriverApi.cuStreamSynchronize(dev.Stream).ThrowOnError();
                }
                CopyCheckpoint(s, restore: true);
                s.NPast = s.CpNPast;
                SlotPositionsDropped(s, s.NPast);
                s.PleTokens.Length = s.CpNPast;
                // The MTP block's rows from nPast on belong to the continuation given up: the next draft rewrites them
                // from there, or restarts the block (MtpRestart) when it restarted after nPast. Below nPast the replay
                // feeds the same tokens and streams, so those rows stand.
                if (s.Mtp != null)
                {
                    if (nPast < s.MtpStart)
                        s.MtpStart = s.MtpPast = nPast;
                    else
                        s.MtpPast = Math.Min(s.MtpPast, nPast);
                }
                if (tokens != null)
                    Forward(tokens, null);
                foreach (var dev in _devs)
                {
                    dev.MakeCurrent();
                    CudaDriverApi.cuStreamSynchronize(dev.Stream).ThrowOnError();
                }
                return true;
            }
            catch (Exception ex) when (ex is InvalidOperationException || ex is CudaException)
            {
                s.Failed = true;
                return false;
            }
        }

        /// <summary>Whether <see cref="Rewind"/> would take slot <paramref name="slotId"/>, holding
        /// <paramref name="head"/> tokens, to <paramref name="target"/>. Side-effect free.</summary>
        public bool CanRewind(int slotId, int head, int target)
            => _slots.TryGetValue(slotId, out Slot s) && CanRewind(s, head, target);

        private bool CanRewind(Slot s, int head, int target)
        {
            DrainDecodeLanes();
            if (s.Failed || s.NPast != head || target < 0 || target > head)
                return false;
            if (target == head || target == 0)
                return true;
            // A replay reads the slot's tokens from its n-gram history.
            return s.CpNPast >= 0 && target >= s.CpNPast
                && (target == s.CpNPast || (_pleLayer >= 0 && s.PleTokens.Length >= target));
        }

        private readonly bool _rewindCheckpoint =
            !string.Equals(Environment.GetEnvironmentVariable("TS_Q4E_REWIND_CHECKPOINT"), "0", StringComparison.Ordinal);

        /// <summary>
        /// One token for each of several distinct slots at their heads, every row computed as that
        /// sequence's own step computes it. False (nothing written) when the step cannot be batched:
        /// fewer than two or more than <see cref="MaxBatchedDecodeRows"/> slots, an unknown or
        /// repeated slot, a position that is not the slot's head, or a full context.
        /// </summary>
        public bool ForwardBatchedDecode(int[] slotIds, int[] tokens, int[] positions, float[] logits)
        {
            DrainDecodeLanes();
            int n = slotIds?.Length ?? 0;
            if (n < 2 || n > MaxBatchedDecodeRows || tokens.Length != n || positions.Length != n
                || logits.Length < (long)n * _m.NVocab)
                return false;
            var rows = new Slot[n];
            var seen = new HashSet<int>();
            for (int i = 0; i < n; i++)
            {
                if (!seen.Add(slotIds[i]) || !_slots.TryGetValue(slotIds[i], out rows[i])
                    || rows[i].Failed || rows[i].NPast != positions[i] || positions[i] + 1 > _m.NCtx)
                    return false;
            }
            // A row that cannot grow goes back to its own step, which reports why.
            try
            {
                foreach (Slot r in rows)
                    EnsureCapacity(r, r.NPast + 1);
            }
            catch (InvalidOperationException)
            {
                return false;
            }

            StageBatchStates(rows);
            foreach (var r in rows)
                r.Failed = true;
            _rowsInFlight = n;
            ArmHostExperts();
            long start = System.Diagnostics.Stopwatch.GetTimestamp();
            bool sampled = BeginSampledForward();
            _issueEnd = 0;
            try
            {
                if (!ForwardBatchedDecodeGraphed(rows, slotIds, tokens, positions, logits))
                    ForwardUbatch(null, tokens, 0, n, 0, logits, rows, positions);
            }
            finally
            {
                _rowsInFlight = 0;
                if (sampled)
                {
                    try { EndSampledForward(n, decode: true); }
                    catch (Exception) { _sampling = false; }
                }
            }
            CheckHostExperts(n);
            AdaptResidency();
            CountForwardTime(start);
            foreach (var r in rows)
            {
                r.NPast++;
                r.Failed = false;
            }
            return true;
        }

        /// <summary>Each device's per-row state pointers for the layers it holds.</summary>
        private void StageBatchStates(Slot[] rows)
        {
            int n = rows.Length;
            foreach (var dev in _devs)
            {
                if (dev.BatchStatePtrs == null)
                    continue;
                long* table = (long*)dev.BatchStatePinned;
                for (int il = 0; il < _m.NLayer; il++)
                {
                    var L = _layers[il];
                    if (L.Device != dev.Ordinal)
                        continue;
                    for (int r = 0; r < n; r++)
                    {
                        var c = rows[r].Layers[il];
                        if (L.Recurrent)
                        {
                            table[((long)il * 2 + 0) * MaxBatchedDecodeRows + r] = (long)Ptr(c.ConvState);
                            table[((long)il * 2 + 1) * MaxBatchedDecodeRows + r] = (long)Ptr(c.Ssm);
                        }
                        if (L.Ple && rows[r].PleHist != null)
                            table[(long)_m.NLayer * 2 * MaxBatchedDecodeRows + r] = (long)Ptr(rows[r].PleHist);
                    }
                }
                dev.MakeCurrent();
                // The previous step's copy out of the pinned table finished with that step's logits.
                CudaDriverApi.cuMemcpyHtoDAsync(Ptr(dev.BatchStatePtrs), dev.BatchStatePinned,
                    new UIntPtr((ulong)BatchStateTableBytes), dev.Stream).ThrowOnError();
            }
        }
    }
}
