// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// ---------------------------------------------------------------------------
// Sequence slots for the direct-CUDA DeepSeek engine: the same contract the
// native ggml executor exports (TSGgml_Dsv4Slot*, TSGgml_Dsv4Truncate,
// TSGgml_Dsv4ForwardBatchedDecode), so the server serves concurrent requests
// and keeps each conversation's caches between turns on either executor.
//
// A slot is one sequence: every layer's caches on that layer's device, its
// position, its Engram hash history and the drafter's key rings. Forward,
// Reset, Truncate and the DSpark calls act on the ACTIVE slot; batched decode
// advances several slots by one token each in one weight sweep.
// ---------------------------------------------------------------------------
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    public sealed unsafe partial class Dsv4CudaEngine
    {
        /// <summary>Rows one batched decode step takes: one per sequence. The native
        /// executor's limit (DSV4_MAX_BATCHED_SLOTS), so both serve the same steps.</summary>
        public const int MaxBatchedDecodeRows = 16;

        /// <summary>One sequence's caches for one layer, on that layer's device.</summary>
        private sealed class SlotLayer
        {
            public Tensor RingK, CompK, LidK, HistKv, HistScore, LidHistKv, LidHistScore;

            // V4.1 rewind checkpoint: shadows of the two caches addressed MODULARLY, where a later
            // position overwrites an earlier one's row (the raw sliding-window ring and the
            // compressor state ring). The compressed caches are addressed by absolute block and
            // only ever grow, so a truncation makes their tail invisible rather than wrong.
            public Tensor RingKCp, HistKvCp, HistScoreCp;

            // V4.1 group source layer whose group spans devices: replicas of CompK/LidK on each
            // later device that reads them, indexed by device (Dsv4CudaEngine.LayerSplit.cs).
            public Tensor[] CompKOn, LidKOn;
        }

        private sealed class Slot
        {
            public readonly int Id;
            public readonly SlotLayer[] Layers;
            public readonly long[] DeviceBytes;
            public readonly List<(Dev Dev, Tensor Tensor)> Owned = new List<(Dev, Tensor)>();
            public readonly Dsv41EngramHistory Engram = new Dsv41EngramHistory();
            // The drafter's key rings, one per stage, and their checkpoint shadows.
            public Tensor[] DsRing, DsRingCp;
            public int NPast;
            // Position of the rewind checkpoint: the rings as they stood at the end of the last
            // multi-token forward (a prompt boundary), or -1 for none.
            public int CpNPast = -1;
            // Set while a forward may have written part of the caches; a slot left failed must
            // be reset before it runs again.
            public bool Failed;

            public Slot(int id, int layers, int devices)
            {
                Id = id;
                Layers = new SlotLayer[layers];
                DeviceBytes = new long[devices];
            }
        }

        /// <summary>A batched decode step: row i is sequence <see cref="Rows"/>[i] at
        /// <see cref="Positions"/>[i].</summary>
        private sealed class DecodeBatch
        {
            public Slot[] Rows;
            public int[] Positions;
        }

        private readonly Dictionary<int, Slot> _slots = new Dictionary<int, Slot>();
        private Slot _active;
        private int _nextSlotId;
        // V4.1: a rewind keeps a prompt-boundary checkpoint; see dsv41_truncate.h for the rules.
        private readonly bool _rewindCheckpoint;
        private readonly int _rewindSpan;
        private IntPtr _pinnedPositions;
        private Tensor _batchLogits;
        private IntPtr _pinnedBatchLogits;

        /// <summary>The multiple a <see cref="Truncate"/> target must be (the lcm of the V4.1
        /// compression ratios), or 0 when this model cannot truncate: V4 compresses OVERLAPPING
        /// blocks, so a boundary still reads the rows a rewind dropped.</summary>
        public int TruncateAlign { get; }

        /// <summary>Slot the single-sequence calls act on.</summary>
        public int ActiveSlot => _active.Id;

        /// <summary>Rows of the V4.1 compressor state ring of a ratio: the ratio itself, plus the
        /// drafter's block so a rejected speculative tail cannot overwrite an accepted row.</summary>
        private int V41StateRows(int ratio) => ratio + _maxDraft;

        private static int ComputeTruncateAlign(ModelDesc m)
        {
            if (!m.V41)
                return 0;
            long align = 1;
            foreach (var layer in m.Layers)
                align = Lcm(align, Math.Max(1, layer.Ratio));
            return (int)align;
        }

        private static long Lcm(long a, long b)
        {
            long x = a, y = b;
            while (y != 0)
                (x, y) = (y, x % y);
            return a / x * b;
        }

        // -------------------------------------------------------------------
        // allocation
        // -------------------------------------------------------------------

        private Tensor SlotTensor(Slot s, Dev dev, DType type, params long[] sizes)
        {
            dev.MakeCurrent();
            var t = new Tensor(dev.Alloc, type, sizes);
            s.Owned.Add((dev, t));
            s.DeviceBytes[dev.Ordinal] += t.ElementCount() * type.Size();
            return t;
        }

        /// <summary>The caches of layer <paramref name="il"/>: F16 key rows (parity with the
        /// native executor and llama.cpp), F32 compressor state rings.</summary>
        private SlotLayer AllocSlotLayer(Slot s, int il)
        {
            DevLayer l = _layers[il];
            if (l.Device < 0)
            {
                // Another stage's layer: no caches here, except the replicas of a group source whose
                // readers run on this node (its committed rows arrive with the boundary payload).
                var remote = new SlotLayer();
                if (_m.V41 && l.Ratio != 0 && l.KvSource == il)
                    AllocReplicas(s, remote, il);
                return remote;
            }
            Dev dev = _devs[l.Device];
            int hd = _m.HeadDim;
            var c = new SlotLayer { RingK = SlotTensor(s, dev, DType.Float16, _ringRaw, hd) };
            if (_m.V41)
            {
                if (_rewindCheckpoint)
                    c.RingKCp = SlotTensor(s, dev, DType.Float16, _ringRaw, hd);
                // Only the per-ratio source layer owns the shared caches; every other
                // compressed layer in its group reads them.
                if (l.Ratio != 0 && l.KvSource == il)
                {
                    int rows = V41Rows(l.Ratio);
                    c.CompK = SlotTensor(s, dev, DType.Float16, rows, hd);
                    c.LidK = SlotTensor(s, dev, DType.Float16, rows, _m.IdxHeadSize);
                    AllocReplicas(s, c, il);
                    if (l.Ratio > 1)
                    {
                        int state = V41StateRows(l.Ratio);
                        c.HistKv = SlotTensor(s, dev, DType.Float32, state, hd);
                        c.HistScore = SlotTensor(s, dev, DType.Float32, state, hd);
                        if (_rewindCheckpoint)
                        {
                            c.HistKvCp = SlotTensor(s, dev, DType.Float32, state, hd);
                            c.HistScoreCp = SlotTensor(s, dev, DType.Float32, state, hd);
                        }
                    }
                }
                return c;
            }
            if (l.Ratio == CsaRatio)
            {
                c.CompK = SlotTensor(s, dev, DType.Float16, _compRowsCsa, hd);
                c.LidK = SlotTensor(s, dev, DType.Float16, _compRowsCsa, _m.IdxHeadSize);
                c.HistKv = SlotTensor(s, dev, DType.Float32, 2L * CsaRatio + _maxDraft, 2L * hd);
                c.HistScore = SlotTensor(s, dev, DType.Float32, 2L * CsaRatio + _maxDraft, 2L * hd);
                c.LidHistKv = SlotTensor(s, dev, DType.Float32, 2L * CsaRatio + _maxDraft, 2L * _m.IdxHeadSize);
                c.LidHistScore = SlotTensor(s, dev, DType.Float32, 2L * CsaRatio + _maxDraft, 2L * _m.IdxHeadSize);
            }
            else if (l.Ratio == HcaRatio)
            {
                c.CompK = SlotTensor(s, dev, DType.Float16, _compRowsHca, hd);
                c.HistKv = SlotTensor(s, dev, DType.Float32, HcaRatio + _maxDraft, hd);
                c.HistScore = SlotTensor(s, dev, DType.Float32, HcaRatio + _maxDraft, hd);
            }
            return c;
        }

        private Slot CreateSlot()
        {
            var s = new Slot(_nextSlotId, _m.NLayer, _devs.Length);
            try
            {
                for (int il = 0; il < _m.NLayer; il++)
                    s.Layers[il] = AllocSlotLayer(s, il);
                if (_ds != null)
                {
                    int stages = _ds.Stages.Length;
                    s.DsRing = new Tensor[stages];
                    s.DsRingCp = _rewindCheckpoint ? new Tensor[stages] : null;
                    for (int i = 0; i < stages; i++)
                    {
                        s.DsRing[i] = SlotTensor(s, _ds.Dev, DType.Float16, _ds.RingRows, _m.HeadDim);
                        if (s.DsRingCp != null)
                            s.DsRingCp[i] = SlotTensor(s, _ds.Dev, DType.Float16, _ds.RingRows, _m.HeadDim);
                    }
                }
                ClearSlot(s);
            }
            catch
            {
                FreeSlotTensors(s);
                throw;
            }
            _nextSlotId++;
            _slots.Add(s.Id, s);
            return s;
        }

        private void SynchronizeDevices()
        {
            foreach (var dev in _devs)
            {
                dev.MakeCurrent();
                CudaDriverApi.cuStreamSynchronize(dev.Stream).ThrowOnError();
            }
        }

        /// <summary>Back to position 0: every cache zeroed, the checkpoint and the Engram
        /// history dropped (its lookbacks reach earlier positions).</summary>
        private void ClearSlot(Slot s)
        {
            SynchronizeDevices();
            foreach (var (dev, t) in s.Owned)
            {
                dev.MakeCurrent();
                Memset0(t);
            }
            s.NPast = 0;
            s.CpNPast = -1;
            s.Engram.Length = 0;
            s.Failed = false;
        }

        private void FreeSlotTensors(Slot s)
        {
            try { SynchronizeDevices(); }
            catch (CudaException) { /* a failed device must not keep the memory */ }
            foreach (var (_, t) in s.Owned)
                t.Dispose();
            s.Owned.Clear();
        }

        // -------------------------------------------------------------------
        // the slot API (TSGgml_Dsv4Slot* on the native executor)
        // -------------------------------------------------------------------

        /// <summary>A new, empty sequence slot (own caches, shared weights); not made active.
        /// Returns its id, or -1 when the devices have no room for another set of caches.</summary>
        public int SlotAlloc()
        {
            // A distributed stage serves the one mirrored sequence; the other nodes could not follow.
            if (_stage != null)
                return -1;
            try
            {
                return CreateSlot().Id;
            }
            catch (Exception ex) when (ex is OutOfMemoryException || ex is CudaException { ErrorCode: 2 })
            {
                Console.Error.WriteLine($"[dsv4-cuda] slot alloc failed: {ex.Message}");
                return -1;
            }
        }

        /// <summary>Select the slot Forward, Reset, Truncate and the DSpark calls act on.</summary>
        public bool SetActiveSlot(int slotId)
        {
            if (!_slots.TryGetValue(slotId, out Slot s))
                return false;
            FlushBoundarySend();
            _active = s;
            return true;
        }

        /// <summary>Free a slot's caches. The active slot cannot be freed; select another first.</summary>
        public bool SlotFree(int slotId)
        {
            if (!_slots.TryGetValue(slotId, out Slot s))
                return false;
            if (s == _active)
            {
                Console.Error.WriteLine($"[dsv4-cuda] refusing to free the active slot {slotId}");
                return false;
            }
            _slots.Remove(slotId);
            FreeSlotTensors(s);
            return true;
        }

        /// <summary>Inspect a slot without selecting it: its head, its checkpoint (-1 for none)
        /// and whether it is usable. False, like the native executor, for a slot that cannot be
        /// kept between requests at all: V4 (no truncation) or a loaded drafter.</summary>
        public bool SlotStatus(int slotId, out int head, out int checkpoint, out bool healthy)
        {
            head = 0;
            checkpoint = -1;
            healthy = false;
            if (!_m.V41 || _ds != null || _stage != null || !_slots.TryGetValue(slotId, out Slot s))
                return false;
            head = s.NPast;
            checkpoint = _rewindCheckpoint ? s.CpNPast : -1;
            healthy = !s.Failed;
            return true;
        }

        /// <summary>Whether the slot, whose head is <paramref name="cachedHead"/>, can move its
        /// head back to <paramref name="target"/> exactly.</summary>
        public bool SlotCanReuse(int slotId, int cachedHead, int target)
            => SlotStatus(slotId, out int head, out int checkpoint, out bool healthy) && healthy && head == cachedHead
               && PlanTruncate(target, head, checkpoint, _rewindSpan, TruncateAlign) != TruncateRoute.Refuse;

        /// <summary>Retention admission: keeping this slot beside <paramref name="retainedCount"/>
        /// others stays inside <paramref name="budgetPerDevice"/> on every device, and leaves each
        /// device a reserve for the running requests.</summary>
        public bool SlotCanRetain(int slotId, int retainedCount, ulong budgetPerDevice)
        {
            if (retainedCount < 0 || !SlotStatus(slotId, out int head, out _, out bool healthy)
                || !healthy || head <= 0)
                return false;
            Slot s = _slots[slotId];
            for (int d = 0; d < _devs.Length; d++)
            {
                if ((ulong)s.DeviceBytes[d] > budgetPerDevice / (ulong)(retainedCount + 1))
                    return false;
            }
            return Fits(s, extra: false);
        }

        /// <summary>Whether one more slot fits on every device with the reserve left over. A
        /// request takes a new slot while this holds, and releases a retained conversation to
        /// make room only when it does not.</summary>
        public bool SlotCanAlloc() => Fits(_active, extra: true);

        /// <summary>Every device that holds a part of <paramref name="s"/> keeps
        /// <see cref="SlotReserveBytes"/> free, after one more slot like it when
        /// <paramref name="extra"/>. Blocks the allocator pool holds count as free: this engine
        /// gets them back without the driver.</summary>
        private bool Fits(Slot s, bool extra)
        {
            for (int d = 0; d < _devs.Length; d++)
            {
                long bytes = s.DeviceBytes[d];
                if (bytes == 0)
                    continue;
                _devs[d].MakeCurrent();
                (long free, _) = _devs[d].Alloc.GetMemoryInfo();
                free += _devs[d].Alloc.GetStats().CachedBytes;
                if (free - (extra ? bytes : 0) < SlotReserveBytes)
                    return false;
            }
            return true;
        }

        /// <summary>What every device keeps free beside the slots: the matmul scratch the shared
        /// ops grow on demand, and the engine's own working set.</summary>
        private const long SlotReserveBytes = 1L << 30;

        /// <summary><see cref="Reset"/> that reports whether the active slot is now usable at position 0.</summary>
        public bool ResetChecked()
        {
            try
            {
                Reset();
            }
            catch (CudaException ex)
            {
                Console.Error.WriteLine($"[dsv4-cuda] reset failed; slot remains unusable: {ex.Message}");
                _active.Failed = true;
            }
            return !_active.Failed && _active.NPast == 0;
        }

        // -------------------------------------------------------------------
        // truncation (partial KV reuse), V4.1 only
        // -------------------------------------------------------------------

        internal enum TruncateRoute
        {
            Refuse,      // nothing correct is available; reset and re-prefill instead
            None,        // the head is already there
            Reset,       // the target is 0, which is a full reset
            Live,        // the live rings still hold the window the new head reads
            Checkpoint,  // restore the shadow rings first, then move the head
        }

        /// <summary>
        /// How far back a V4.1 slot can move its head, and how. The native executor's
        /// dsv41_plan_truncate, which states the two conditions: the target is aligned to every
        /// compression ratio (no block straddles the new head), and the raw ring still holds the
        /// window the new head reads - live, or in the checkpoint taken at the last prompt
        /// boundary. <paramref name="checkpoint"/> is -1 when the slot has none.
        /// </summary>
        internal static TruncateRoute PlanTruncate(long target, long nPast, long checkpoint, long span, long align)
        {
            if (target < 0 || target > nPast || align <= 0) return TruncateRoute.Refuse;
            if (target == nPast) return TruncateRoute.None;
            if (target == 0) return TruncateRoute.Reset;
            if (target % align != 0) return TruncateRoute.Refuse;
            if (nPast - target <= span) return TruncateRoute.Live;
            if (checkpoint >= target && checkpoint - target <= span) return TruncateRoute.Checkpoint;
            return TruncateRoute.Refuse;
        }

        /// <summary>
        /// Move the active slot's head back to <paramref name="nPast"/>, so the next forward appends
        /// there and the first <paramref name="nPast"/> positions are reused instead of re-prefilled.
        /// True when the slot now holds exactly those positions and a forward from there matches a
        /// fresh prefill; false when it cannot (the caller resets and re-prefills).
        /// </summary>
        public bool Truncate(int nPast)
        {
            FlushBoundarySend();
            Slot s = _active;
            if (TruncateAlign <= 0 || s.Failed)
                return false;
            TruncateRoute route = PlanTruncate(nPast, s.NPast, _rewindCheckpoint ? s.CpNPast : -1,
                _rewindSpan, TruncateAlign);
            switch (route)
            {
                case TruncateRoute.Refuse:
                    return false;
                case TruncateRoute.None:
                    return true;
                case TruncateRoute.Reset:
                    return ResetChecked();
            }
            s.Failed = true;
            if (route == TruncateRoute.Checkpoint)
                CopyRewindRings(s, save: false);
            s.NPast = nPast;
            // The history is indexed by absolute position: dropping the tail restores the
            // hasher's expected next position.
            s.Engram.Length = Math.Min(s.Engram.Length, nPast);
            // A checkpoint past the new head belongs to the abandoned branch.
            if (s.CpNPast > nPast)
                s.CpNPast = -1;
            s.Failed = false;
            return true;
        }

        /// <summary>Copy the modular rings into (<paramref name="save"/>) or out of the slot's
        /// shadows. Both sides are on the layer's device, so these are device-local copies.</summary>
        private void CopyRewindRings(Slot s, bool save)
        {
            for (int il = 0; il < _m.NLayer; il++)
            {
                if (_layers[il].Device < 0)
                    continue;   // another stage's layer
                SlotLayer c = s.Layers[il];
                Dev dev = _devs[_layers[il].Device];
                dev.MakeCurrent();
                CopyTensor(dev, save ? c.RingK : c.RingKCp, save ? c.RingKCp : c.RingK);
                CopyTensor(dev, save ? c.HistKv : c.HistKvCp, save ? c.HistKvCp : c.HistKv);
                CopyTensor(dev, save ? c.HistScore : c.HistScoreCp, save ? c.HistScoreCp : c.HistScore);
            }
            if (s.DsRing != null && s.DsRingCp != null)
            {
                _ds.Dev.MakeCurrent();
                for (int i = 0; i < s.DsRing.Length; i++)
                    CopyTensor(_ds.Dev, save ? s.DsRing[i] : s.DsRingCp[i], save ? s.DsRingCp[i] : s.DsRing[i]);
            }
        }

        private static void CopyTensor(Dev dev, Tensor src, Tensor dst)
        {
            if (src == null || dst == null)
                return;
            long bytes = src.ElementCount() * src.ElementType.Size();
            CudaDriverApi.cuMemcpyDtoDAsync(Ptr(dst), Ptr(src), new UIntPtr((ulong)bytes), dev.Stream).ThrowOnError();
        }

        /// <summary>Record the active slot's rings as its rewind checkpoint. Called at the end of
        /// every multi-token forward; decode steps deliberately leave it at the prompt boundary,
        /// which is where the next turn's re-rendered prompt diverges.</summary>
        private void CheckpointActiveSlot()
        {
            Slot s = _active;
            if (!_rewindCheckpoint || s.Failed)
                return;
            s.Failed = true;
            CopyRewindRings(s, save: true);
            s.CpNPast = s.NPast;
            s.Failed = false;
        }

        // -------------------------------------------------------------------
        // batched decode
        // -------------------------------------------------------------------

        /// <summary>
        /// One token for each of several sequences in a single forward: the dense weights and each
        /// step's routed experts are read once for every row instead of once per sequence. Rows are
        /// distinct slots, each at its own head (<paramref name="positions"/>). Writes one row of
        /// logits per sequence and advances each by one position. False when the step cannot be
        /// batched (the caller runs the sequences one at a time); nothing is written then.
        /// </summary>
        public bool ForwardBatchedDecode(int[] slotIds, int[] tokens, int[] positions, float[] logitsOut)
        {
            int n = slotIds?.Length ?? 0;
            // The drafter's rings are fed by the per-sequence path only.
            if (n < 2 || n > MaxBatchedDecodeRows || n > _m.NUbatch || _ds != null || _stage != null
                || tokens == null || tokens.Length != n || positions == null || positions.Length != n
                || logitsOut == null || logitsOut.LongLength < (long)n * _m.NVocab)
                return false;
            var rows = new Slot[n];
            for (int i = 0; i < n; i++)
            {
                if (!_slots.TryGetValue(slotIds[i], out Slot s) || s.Failed || s.NPast != positions[i]
                    || positions[i] + 1 > _m.NCtx || (uint)tokens[i] >= (uint)_m.NVocab)
                    return false;
                for (int j = 0; j < i; j++)
                    if (rows[j] == s) return false;
                rows[i] = s;
            }

            foreach (Slot s in rows)
                s.Failed = true;
            ForwardUbatch(tokens, 0, n, 0, logitsOut, allLogitsRows: true, hAllOut: null, hRowOff: 0,
                batch: new DecodeBatch { Rows = rows, Positions = positions });
            foreach (Slot s in rows)
            {
                s.NPast++;
                s.Failed = false;
            }
            return true;
        }

        private void EnsureBatchBuffers()
        {
            if (_batchLogits != null)
                return;
            Dev last = _devs[_lastDev];
            _batchLogits = AllocF32(last, MaxBatchedDecodeRows, _m.NVocab);
            _devs[0].MakeCurrent();
            CudaDriverApi.cuMemHostAlloc(out _pinnedPositions, new UIntPtr((ulong)MaxBatchedDecodeRows * 4), 0x1 /*PORTABLE*/).ThrowOnError();
            CudaDriverApi.cuMemHostAlloc(out _pinnedBatchLogits,
                new UIntPtr((ulong)MaxBatchedDecodeRows * (ulong)_m.NVocab * 4UL), 0x1 /*PORTABLE*/).ThrowOnError();
            foreach (var dev in _devs)
                dev.Positions = AllocI32(dev, MaxBatchedDecodeRows);
        }

        /// <summary>Every row's position to every device. The step ends with the logits readback,
        /// which waits for all of it, so one pinned buffer serves every step.</summary>
        private void StagePositions(DecodeBatch b)
        {
            EnsureBatchBuffers();
            Marshal.Copy(b.Positions, 0, _pinnedPositions, b.Positions.Length);
            foreach (var dev in _devs)
            {
                dev.MakeCurrent();
                CudaDriverApi.cuMemcpyHtoDAsync(Ptr(dev.Positions), _pinnedPositions,
                    new UIntPtr((ulong)b.Positions.Length * 4), dev.Stream).ThrowOnError();
            }
        }

        /// <summary>Row <paramref name="row"/> of a scratch operand <paramref name="width"/> values wide.
        /// The operand's own width, not the buffer's: MatMul writes compact [rows, width] blocks at the
        /// front of buffers sized for their widest user (the V4.1 compressor's hd-wide rows sit in a
        /// buffer allocated 2*hd wide for V4's CSA).</summary>
        private static Tensor Row(Tensor t, int row, long width) => Block(t, row, 1, width);

        /// <summary>
        /// The attention block of a batched decode step. The projections are one GEMM over every
        /// row; what touches a sequence's caches (RoPE into its ring, its compressors, its indexer
        /// and the attention itself) runs row by row against that row's slot.
        /// </summary>
        private void AttentionRows(Dev dev, DevLayer l, int il, int nt, DecodeBatch b)
        {
            var m = _m;
            int ratio = l.Ratio;
            IntPtr ropeTab = Ptr(ratio != 0 ? dev.RopeComp : dev.RopeRaw);

            MatMul(dev, l.WqA, dev.Cur, dev.Qr, nt);
            RmsNorm(dev, dev.Qr, l.QANorm, nt);
            MatMul(dev, l.WqB, dev.Qr, dev.Q, nt);
            MatMul(dev, l.Wkv, dev.Cur, dev.KvRaw, nt);
            StageEnd(dev, 2);

            if (m.V41)
                CacheRowsV41(dev, l, il, nt, b, ropeTab);
            else
                CacheRowsV4(dev, l, il, nt, b, ropeTab);

            // Inverse RoPE at each row's own position, then the grouped LoRA out-projection.
            int hpg = m.NHead / m.OGroups;
            dev.DK.AttnFinish(dev.AttnO, ropeTab, dev.OGrouped, 0, m.NHead, m.HeadDim, m.NRot, hpg, nt,
                dev.Stream, Ptr(dev.Positions));
            OutProjectionGroups(dev, l, nt);
        }

        private void CacheRowsV41(Dev dev, DevLayer l, int il, int nt, DecodeBatch b, IntPtr ropeTab)
        {
            var m = _m;
            int nh = m.NHead, hd = m.HeadDim, ratio = l.Ratio;
            bool compress = ratio != 0 && l.KvSource == il;
            bool index = ratio != 0 && l.IndexSource == il;

            if (compress)
            {
                MatMul(dev, l.CompWkv, dev.Cur, dev.StKv, nt);
                if (ratio > 1)
                    MatMul(dev, l.CompWgate, dev.Cur, dev.StScore, nt);
            }

            // RoPE into each row's ring, and the latent of any block its position closes.
            Span<int> latentOf = stackalloc int[nt];
            int blocks = 0;
            for (int i = 0; i < nt; i++)
            {
                int p = b.Positions[i];
                SlotLayer c = b.Rows[i].Layers[il];
                using (Tensor q = Row(dev.Q, i, (long)nh * hd))
                using (Tensor kv = Row(dev.KvRaw, i, hd))
                {
                    dev.DK.V41AttnPrep(q, kv, Ptr(l.KvNorm), ropeTab, Ptr(c.RingK),
                        p, _ringRaw, nh, hd, m.NRot, m.RmsEps, 1, dev.Stream);
                }
                latentOf[i] = -1;
                if (compress && (p + 1) % ratio == 0)
                {
                    using Tensor stKv = Row(dev.StKv, i, hd);
                    using Tensor stScore = Row(dev.StScore, i, hd);
                    using Tensor latent = Row(dev.Latent, blocks, hd);
                    dev.DK.V41Compress(stKv, stScore, Ptr(c.HistKv), Ptr(c.HistScore), Ptr(l.CompNorm),
                        latent, p, 1, p, ratio, V41StateRows(ratio), hd, m.RmsEps, dev.Stream);
                    latentOf[i] = blocks++;
                }
            }
            StageEnd(dev, 2);

            if (compress)
            {
                if (blocks > 0)
                {
                    using Tensor latentRows = Rows(dev.Latent, blocks);
                    using Tensor keyRows = Rows(dev.LatentK, blocks);
                    MatMul(dev, l.IndexerK, latentRows, keyRows, blocks);
                    RmsNorm(dev, keyRows, l.IndexerKNorm, blocks);
                }
                for (int i = 0; i < nt; i++)
                {
                    int p = b.Positions[i];
                    SlotLayer c = b.Rows[i].Layers[il];
                    if (latentOf[i] >= 0)
                    {
                        using Tensor key = Row(dev.LatentK, latentOf[i], m.IdxHeadSize);
                        using Tensor latent = Row(dev.Latent, latentOf[i], hd);
                        dev.DK.V41Commit(key, Ptr(dev.RopeComp), Ptr(c.LidK), p, 1, ratio, m.IdxHeadSize, m.NRot, 1, dev.Stream);
                        dev.DK.V41Commit(latent, Ptr(dev.RopeComp), Ptr(c.CompK), p, 1, ratio, hd, m.NRot, 2, dev.Stream);
                        NoteCommitted(il, b.Rows[i], p, ratio, 1);
                    }
                    if (ratio > 1)
                    {
                        using Tensor stKv = Row(dev.StKv, i, hd);
                        using Tensor stScore = Row(dev.StScore, i, hd);
                        dev.DK.V41Persist(stKv, stScore, Ptr(c.HistKv), Ptr(c.HistScore), p, 1,
                            V41StateRows(ratio), hd, dev.Stream);
                    }
                }
                StageEnd(dev, 3);
            }

            if (index)
            {
                MatMul(dev, l.IdxQB, dev.Qr, dev.Iq, nt);
                MatMul(dev, l.IdxProj, dev.Cur, dev.Iw, nt);
                float iwScale = 1.0f / MathF.Sqrt((float)m.IdxHeadSize * m.IdxNHead);
                int rows = (int)dev.IdxScores.Sizes[1];
                bool prune = _v41CandActive && il > m.CandidateSource;
                for (int i = 0; i < nt; i++)
                {
                    int p = b.Positions[i];
                    Tensor lidK = LidKOn(b.Rows[i], l.KvSource, dev);
                    using Tensor iq = Row(dev.Iq, i, (long)m.IdxNHead * m.IdxHeadSize);
                    using Tensor iw = Row(dev.Iw, i, m.IdxNHead);
                    using Tensor scores = Row(dev.IdxScores, i, rows);
                    using Tensor mask = dev.CandMask != null ? Row(dev.CandMask, i, rows) : null;
                    using Tensor topkIdx = Row(dev.TopkIdx, i, Math.Max(m.IdxTopK, 1));
                    using Tensor topkCnt = Row(dev.TopkCnt, i, 1);
                    dev.DK.V41IdxPrep(iq, iw, Ptr(dev.RopeComp), p, m.IdxNHead, m.IdxHeadSize, m.NRot,
                        iwScale, 1, dev.Stream);
                    dev.DK.V41IdxScores(iq, iw, Ptr(lidK), prune ? Ptr(mask) : IntPtr.Zero,
                        scores, p, ratio, m.IdxNHead, m.IdxHeadSize, rows, (p + 1) / ratio, 1, dev.Stream);
                    if (il == m.CandidateSource)
                        dev.DK.V41Candidate(scores, Ptr(mask), p, ratio, rows, m.CandidateBlock, m.CandidateTopk, 1, dev.Stream);
                    dev.DK.TopK(scores, topkIdx, topkCnt, p, ratio, m.IdxTopK, rows, 1, dev.Stream);
                }
                if (il == m.CandidateSource)
                    _v41CandActive = true;
                StageEnd(dev, 4);
            }

            float kqScale = 1.0f / MathF.Sqrt(hd);
            for (int i = 0; i < nt; i++)
            {
                SlotLayer c = b.Rows[i].Layers[il];
                Tensor comp = ratio != 0 ? CompKOn(b.Rows[i], l.KvSource, dev) : null;
                using Tensor q = Row(dev.Q, i, (long)nh * hd);
                using Tensor topkIdx = Row(dev.TopkIdx, i, Math.Max(m.IdxTopK, 1));
                using Tensor topkCnt = Row(dev.TopkCnt, i, 1);
                using Tensor attnO = Row(dev.AttnO, i, (long)nh * hd);
                dev.DK.Attention(q, Ptr(c.RingK), Ptr(comp), topkIdx, topkCnt, Ptr(l.Sinks), attnO,
                    b.Positions[i], m.NSwa, _ringRaw, nh, hd, ratio != 0 ? 1 : 0, ratio == 0 ? 1 : ratio,
                    m.IdxTopK, kqScale, 1, dev.Stream);
            }
            StageEnd(dev, 5);
        }

        private void CacheRowsV4(Dev dev, DevLayer l, int il, int nt, DecodeBatch b, IntPtr ropeTab)
        {
            var m = _m;
            int nh = m.NHead, hd = m.HeadDim;
            for (int i = 0; i < nt; i++)
            {
                SlotLayer c = b.Rows[i].Layers[il];
                using Tensor q = Row(dev.Q, i, (long)nh * hd);
                using Tensor kv = Row(dev.KvRaw, i, hd);
                dev.DK.AttnPrep(q, kv, Ptr(l.KvNorm), ropeTab, Ptr(c.RingK), b.Positions[i], _ringRaw,
                    nh, hd, m.NRot, m.RmsEps, 1, dev.Stream);
            }
            StageEnd(dev, 2);

            // Each row's attention mode: 0 raw window only, 1 the indexer's selection, 2 every
            // visible compressed row (a CSA layer below the indexer's budget, or HCA).
            Span<int> mode = stackalloc int[nt];
            mode.Clear();
            if (l.Ratio == CsaRatio)
            {
                int cw = 2 * hd, lcw = 2 * m.IdxHeadSize;
                MatMul(dev, l.CompWkv, dev.Cur, dev.StKv, nt);
                MatMul(dev, l.CompWgate, dev.Cur, dev.StScore, nt);
                MatMul(dev, l.IdxCompWkv, dev.Cur, dev.LidStKv, nt);
                MatMul(dev, l.IdxCompWgate, dev.Cur, dev.LidStScore, nt);
                bool anyIndexed = false;
                for (int i = 0; i < nt; i++)
                {
                    int p = b.Positions[i];
                    SlotLayer c = b.Rows[i].Layers[il];
                    using (Tensor stKv = Row(dev.StKv, i, cw))
                    using (Tensor stScore = Row(dev.StScore, i, cw))
                    {
                        dev.DK.ApeAdd(stScore, Ptr(l.CompApe), p, CsaRatio, 1, cw, dev.Stream);
                        RunCompressor(dev, 1, p, CsaRatio, 2, hd, 2 * CsaRatio + _maxDraft, cw,
                            stKv, stScore, c.HistKv, c.HistScore, l.CompNorm, c.CompK, dev.RopeComp);
                    }
                    using (Tensor stKv = Row(dev.LidStKv, i, lcw))
                    using (Tensor stScore = Row(dev.LidStScore, i, lcw))
                    {
                        dev.DK.ApeAdd(stScore, Ptr(l.IdxCompApe), p, CsaRatio, 1, lcw, dev.Stream);
                        RunCompressor(dev, 1, p, CsaRatio, 2, m.IdxHeadSize, 2 * CsaRatio + _maxDraft, lcw,
                            stKv, stScore, c.LidHistKv, c.LidHistScore, l.IdxCompNorm, c.LidK, dev.RopeComp);
                    }
                    mode[i] = (p + 1) / CsaRatio > m.IdxTopK ? 1 : 2;
                    anyIndexed |= mode[i] == 1;
                }
                StageEnd(dev, 3);

                if (anyIndexed)
                {
                    MatMul(dev, l.IdxQB, dev.Qr, dev.Iq, nt);
                    MatMul(dev, l.IdxProj, dev.Cur, dev.Iw, nt);
                    float iwScale = 1.0f / MathF.Sqrt((float)m.IdxHeadSize * m.IdxNHead);
                    for (int i = 0; i < nt; i++)
                    {
                        if (mode[i] != 1)
                            continue;
                        int p = b.Positions[i];
                        SlotLayer c = b.Rows[i].Layers[il];
                        using Tensor iq = Row(dev.Iq, i, (long)m.IdxNHead * m.IdxHeadSize);
                        using Tensor iw = Row(dev.Iw, i, m.IdxNHead);
                        using Tensor scores = Row(dev.IdxScores, i, _compRowsCsa);
                        using Tensor topkIdx = Row(dev.TopkIdx, i, Math.Max(m.IdxTopK, 1));
                        using Tensor topkCnt = Row(dev.TopkCnt, i, 1);
                        int maxVis = (p + 1) / CsaRatio;
                        dev.DK.IdxPrep(iq, iw, Ptr(dev.RopeComp), p, m.IdxNHead, m.IdxHeadSize, m.NRot, iwScale, 1, dev.Stream);
                        dev.DK.IdxScores(iq, iw, Ptr(c.LidK), scores, p, CsaRatio, m.IdxNHead, m.IdxHeadSize,
                            1, _compRowsCsa, maxVis, dev.Stream);
                        dev.DK.TopK(scores, topkIdx, topkCnt, p, CsaRatio, m.IdxTopK, _compRowsCsa, 1, dev.Stream);
                    }
                    StageEnd(dev, 4);
                }
            }
            else if (l.Ratio == HcaRatio)
            {
                MatMul(dev, l.CompWkv, dev.Cur, dev.StKv, nt);
                MatMul(dev, l.CompWgate, dev.Cur, dev.StScore, nt);
                for (int i = 0; i < nt; i++)
                {
                    int p = b.Positions[i];
                    SlotLayer c = b.Rows[i].Layers[il];
                    using Tensor stKv = Row(dev.StKv, i, hd);
                    using Tensor stScore = Row(dev.StScore, i, hd);
                    dev.DK.ApeAdd(stScore, Ptr(l.CompApe), p, HcaRatio, 1, hd, dev.Stream);
                    RunCompressor(dev, 1, p, HcaRatio, 1, hd, HcaRatio + _maxDraft, hd,
                        stKv, stScore, c.HistKv, c.HistScore, l.CompNorm, c.CompK, dev.RopeComp);
                    mode[i] = 2;
                }
                StageEnd(dev, 3);
            }

            float kqScale = 1.0f / MathF.Sqrt(hd);
            for (int i = 0; i < nt; i++)
            {
                SlotLayer c = b.Rows[i].Layers[il];
                using Tensor q = Row(dev.Q, i, (long)nh * hd);
                using Tensor topkIdx = Row(dev.TopkIdx, i, Math.Max(m.IdxTopK, 1));
                using Tensor topkCnt = Row(dev.TopkCnt, i, 1);
                using Tensor attnO = Row(dev.AttnO, i, (long)nh * hd);
                dev.DK.Attention(q, Ptr(c.RingK), Ptr(c.CompK), topkIdx, topkCnt, Ptr(l.Sinks), attnO,
                    b.Positions[i], m.NSwa, _ringRaw, nh, hd, mode[i], l.Ratio == 0 ? 1 : l.Ratio, m.IdxTopK,
                    kqScale, 1, dev.Stream);
            }
            StageEnd(dev, 5);
        }

        private void FreeSlots()
        {
            foreach (Slot s in _slots.Values)
            {
                foreach (var (_, t) in s.Owned)
                    t.Dispose();
                s.Owned.Clear();
            }
            _slots.Clear();
            if (_pinnedPositions != IntPtr.Zero)
                CudaDriverApi.cuMemFreeHost(_pinnedPositions);
            if (_pinnedBatchLogits != IntPtr.Zero)
                CudaDriverApi.cuMemFreeHost(_pinnedBatchLogits);
            _pinnedPositions = _pinnedBatchLogits = IntPtr.Zero;
        }
    }
}
