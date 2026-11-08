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
            if (!SlotFits())
                return -1;
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
            return slot.Id;
        }

        /// <summary>Whether one more slot fits, counting what the allocators hold pooled.</summary>
        private bool SlotFits()
        {
            long[] need = _active.DeviceBytes;
            for (int d = 0; d < _devs.Length; d++)
            {
                if (need[d] == 0)
                    continue;
                _devs[d].MakeCurrent();
                (long free, _) = _devs[d].Alloc.GetMemoryInfo();
                free += _devs[d].Alloc.GetStats().CachedBytes;
                if (free - need[d] < SlotReserveBytes)
                    return false;
            }
            return true;
        }

        public bool SetActiveSlot(int id)
        {
            if (!_slots.TryGetValue(id, out Slot slot))
                return false;
            _active = slot;
            return true;
        }

        /// <summary>Free a slot; the active slot cannot be freed.</summary>
        public bool SlotFree(int id)
        {
            if (!_slots.TryGetValue(id, out Slot slot) || ReferenceEquals(slot, _active))
                return false;
            FreeSlot(slot);
            _slots.Remove(id);
            return true;
        }

        /// <summary>Head of a slot without selecting it; false when there is no such slot.</summary>
        public bool SlotHead(int id, out int head)
        {
            head = 0;
            if (!_slots.TryGetValue(id, out Slot slot))
                return false;
            head = slot.NPast;
            return true;
        }

        /// <summary><see cref="Reset"/> that reports whether the active slot is usable at position 0.</summary>
        public bool ResetChecked()
        {
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

        /// <summary>Move the active slot's head to <paramref name="nPast"/>: its own head, or 0 (the
        /// recurrent states cannot go back to anything in between).</summary>
        public bool Rewind(int nPast)
        {
            if (nPast == _active.NPast)
                return true;
            if (nPast != 0)
                return false;
            return ResetChecked();
        }

        /// <summary>
        /// One token for each of several distinct slots at their heads, every row computed as that
        /// sequence's own step computes it. False (nothing written) when the step cannot be batched:
        /// fewer than two or more than <see cref="MaxBatchedDecodeRows"/> slots, an unknown or
        /// repeated slot, a position that is not the slot's head, or a full context.
        /// </summary>
        public bool ForwardBatchedDecode(int[] slotIds, int[] tokens, int[] positions, float[] logits)
        {
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

            StageBatchStates(rows);
            foreach (var r in rows)
                r.Failed = true;
            _rowsInFlight = n;
            ArmHostExperts();
            try
            {
                if (ForwardBatchGraphed(rows, tokens, positions, 1, greedy: false, hidden: false))
                    FinishHead(n, logits, true, null);
                else
                    ForwardUbatch(null, tokens, 0, n, 0, logits, rows, positions);
            }
            finally
            {
                _rowsInFlight = 0;
            }
            CheckHostExperts(n);
            AdaptResidency();
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
