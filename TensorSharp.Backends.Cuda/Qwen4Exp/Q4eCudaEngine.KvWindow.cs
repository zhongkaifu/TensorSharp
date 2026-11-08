// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// The resident KV window of the Qwen3.8-Flash-Next direct-CUDA engine (TS_Q4E_KV_RESIDENT, Strata's --kv-resident): a
// QSA attention layer keeps its authoritative Q8_0 K/V in pinned, device-mapped host memory and only a window of pages
// in VRAM, so the VRAM a long context's caches would hold goes to routed-expert slots instead.
//
// Past the QSA width a row attends its own incomplete block and the top-k blocks the indexer picks (2,051 cells), and
// below it every cell 0..p (at most as many): a decode step or verify window of up to PerSlotMaxRows rows reads at most
// PerSlotMaxRows x (2,051 / 4 + 2) pages per layer, which the window always holds. Before such a call's attention reads
// them, ts_q4e_kv_resolve makes them resident (hits stamped, misses given the slots nothing in the call uses, round
// robin) and ts_q4e_kv_page_copy copies the misses over the bus, all inside the captured step. The writer stores each
// new cell into the host copy and, when its page is resident, into the page's slot, so a slot never goes stale.
//
// A prompt (more than PerSlotMaxRows rows) reads whole contexts: its layer runs on a staging pool of the full context
// (one per device, shared by the layers in turn), filled from the host copy once per layer and prompt group; the
// ubatch's new cells go back to the host copy by DMA and leave the window (their old slots are stale).
//
// The QSA indexer of a streamed layer keeps its raw keys in a ring: a block's raw keys are read back only to pool the
// block, at most a prompt ubatch after its first member was written. Its pooled keys stay whole (every step scores
// every block).
//
// Reads go through the page table in the paged kernels; their arithmetic is the resident kernels', cell for cell, so a
// streamed layer answers bit for bit as a resident one.
using System;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    public sealed unsafe partial class Q4eCudaEngine
    {
        /// <summary>A streamed attention layer's page map and host copy (one per slot and layer).</summary>
        private sealed class KvStream
        {
            /// <summary>page -> VRAM slot (-1: not resident); slot -> page (-1: free); the epoch that last used a slot;
            /// the control words; the last resolve's misses and their slots.</summary>
            public Tensor Table, SlotPage, SlotStamp, Ctl, MissPage, MissSlot;
            /// <summary>The host copies, [nCtx, NKV, HD] codes and [nCtx, NKV, HD / 32] scales: pinned host pointers (DMA)
            /// and their device addresses (kernels).</summary>
            public IntPtr HostK, HostV, HostKs, HostVs, DevK, DevV, DevKs, DevVs;
            public int Slots;
        }

        // TS_Q4E_KV_RESIDENT: cells of each QSA attention layer's K/V kept in VRAM (Strata's --kv-resident); the rest of the
        // context lives in pinned host memory. 0 keeps every layer's K/V resident. Unset: Strata's 32,768 when the routed
        // experts do not fit in VRAM whole (expert residency: the VRAM goes to expert slots), every layer resident when
        // they do.
        private readonly int _kvResidentEnv = EnvInt("TS_Q4E_KV_RESIDENT", -1);

        /// <summary>The window an unset TS_Q4E_KV_RESIDENT takes under expert residency (Strata's --kv-resident).</summary>
        private const int KvResidentDefault = 32768;

        /// <summary>Cells a streamed layer keeps in VRAM, or 0 (every layer resident).</summary>
        private int _kvWindow;

        /// <summary>Raw indexer keys a streamed layer keeps (a ring).</summary>
        private int _idxRing;

        private bool KvStreamed(int il) => _kvWindow > 0 && !_m.Layers[il].Recurrent && _m.Layers[il].CompressRatio > 0;

        /// <summary>Decides the window before the residency plan sizes the caches: TS_Q4E_KV_RESIDENT, or when it is unset,
        /// <see cref="KvResidentDefault"/> unless the routed experts fit in VRAM whole.</summary>
        private void PlanKvWindow(bool expertsFitWhole)
        {
            var m = _m;
            _kvWindow = 0;
            bool auto = _kvResidentEnv < 0;
            int requested = auto ? (expertsFitWhole ? 0 : KvResidentDefault) : _kvResidentEnv;
            if (requested <= 0)
                return;
            string setting = auto ? $"the default {KvResidentDefault}-cell window (the experts do not fit whole)" : $"TS_Q4E_KV_RESIDENT={requested}";
            // A call of PerSlotMaxRows rows reads at most CellStride cells a row, in whole blocks: their pages, plus the
            // row's own incomplete block.
            int floor = PerSlotMaxRows * ((CellStride + Q4eKernels.KvPage - 1) / Q4eKernels.KvPage + 2) * Q4eKernels.KvPage;
            int cells = Math.Max(requested, floor);
            cells = (cells + Q4eKernels.KvPage - 1) / Q4eKernels.KvPage * Q4eKernels.KvPage;
            if (cells >= m.NCtx)
            {
                if (!auto)
                    Console.Error.WriteLine($"[q4e-cuda] {setting}: the {m.NCtx}-cell context fits; every layer's K/V stays in VRAM.");
                return;
            }
            bool anyQsa = false, aligned = true;
            foreach (var L in m.Layers)
                if (!L.Recurrent && L.CompressRatio > 0)
                {
                    anyQsa = true;
                    aligned &= L.CompressRatio % Q4eKernels.KvPage == 0;
                }
            string why = !anyQsa ? "the model has no QSA attention layer"
                : !m.KvQ8 ? "it streams the Q8_0 KV cache only (--kv-cache-dtype q8_0)"
                : !aligned ? $"a QSA block is not a whole number of {Q4eKernels.KvPage}-cell pages"
                : Array.Exists(_devs, d => !d.QK.HasKvStreaming)
                    ? "the loaded tensorsharp_q4e_kernels module predates it (rebuild the CUDA kernels)" : null;
            if (why != null)
            {
                Console.Error.WriteLine($"[q4e-cuda] {setting} not used: {why}; every layer's K/V stays in VRAM.");
                return;
            }
            _kvWindow = cells;
            _idxRing = m.NUbatch + 2 * MaxRatio;
            int streamed = 0;
            for (int il = 0; il < m.NLayer; il++)
                if (KvStreamed(il))
                    streamed++;
            long saved = (long)streamed * (2 * (KvCacheBytes(m.NCtx) - KvCacheBytes(cells))
                + (long)(m.NCtx - _idxRing) * m.IdxDim * 2) - 2 * KvCacheBytes(m.NCtx);
            Console.Error.WriteLine($"[q4e-cuda] resident KV window: {streamed} QSA layers keep {cells} of {m.NCtx} cells' K/V in VRAM "
                + $"(the rest in pinned host memory, {streamed * 2 * KvCacheBytes(m.NCtx) / 1073741824.0:F2} GiB a sequence) and "
                + $"{_idxRing} raw indexer keys; {saved / 1073741824.0:F2} GiB of VRAM a sequence for experts, less one "
                + $"{2 * KvCacheBytes(m.NCtx) / 1048576.0:F0} MiB prompt staging pool a GPU (TS_Q4E_KV_RESIDENT=0 keeps them resident"
                + (cells > requested ? $"; raised to one call's {floor} cells" : "") + ").");
        }

        /// <summary>A streamed layer's map and slot bytes beside its window (device).</summary>
        private long KvMapBytes()
        {
            long pages = _m.NCtx / Q4eKernels.KvPage + 1, slots = _kvWindow / Q4eKernels.KvPage;
            return CudaWeightArena.Align(pages * 4) + 4 * CudaWeightArena.Align(slots * 4) + CudaWeightArena.Align(Q4eKernels.KvCtlInts * 4);
        }

        /// <summary>The window's K/V tensors, its map and its pinned host copy for one streamed layer of a slot.</summary>
        private void AllocKvStream(Slot slot, Dev dev, SlotLayer c)
        {
            var m = _m;
            long window = (long)_kvWindow * m.NKvHead * m.HeadDim, full = (long)m.NCtx * m.NKvHead * m.HeadDim;
            c.KCache = SlotTensor(slot, dev, DType.UInt8, window);
            c.VCache = SlotTensor(slot, dev, DType.UInt8, window);
            c.KScale = SlotTensor(slot, dev, DType.Float16, window / Q4eKernels.KvQ8Block);
            c.VScale = SlotTensor(slot, dev, DType.Float16, window / Q4eKernels.KvQ8Block);
            int slots = _kvWindow / Q4eKernels.KvPage;
            var kv = new KvStream
            {
                Slots = slots,
                Table = SlotTensor(slot, dev, DType.Int32, m.NCtx / Q4eKernels.KvPage + 1),
                SlotPage = SlotTensor(slot, dev, DType.Int32, slots),
                SlotStamp = SlotTensor(slot, dev, DType.Int32, slots),
                Ctl = SlotTensor(slot, dev, DType.Int32, Q4eKernels.KvCtlInts),
                MissPage = SlotTensor(slot, dev, DType.Int32, slots),
                MissSlot = SlotTensor(slot, dev, DType.Int32, slots),
            };
            kv.HostK = HostMapped(slot, full, out kv.DevK);
            kv.HostV = HostMapped(slot, full, out kv.DevV);
            kv.HostKs = HostMapped(slot, full / Q4eKernels.KvQ8Block * 2, out kv.DevKs);
            kv.HostVs = HostMapped(slot, full / Q4eKernels.KvQ8Block * 2, out kv.DevVs);
            c.Kv = kv;
            ResetKvStream(dev, kv);
        }

        private static IntPtr HostMapped(Slot slot, long bytes, out IntPtr device)
        {
            CudaDriverApi.cuMemHostAlloc(out IntPtr host, new UIntPtr((ulong)bytes),
                CudaDriverApi.CU_MEMHOSTALLOC_PORTABLE | CudaDriverApi.CU_MEMHOSTALLOC_DEVICEMAP).ThrowOnError();
            slot.HostOwned.Add(host);
            CudaDriverApi.cuMemHostGetDevicePointer(out device, host, 0).ThrowOnError();
            return host;
        }

        /// <summary>Every page out of the window, every slot free, the counters zero.</summary>
        private static void ResetKvStream(Dev dev, KvStream kv)
        {
            dev.MakeCurrent();
            CudaDriverApi.cuMemsetD8Async(Ptr(kv.Table), 0xFF, new UIntPtr((ulong)kv.Table.ElementCount() * 4), dev.Stream).ThrowOnError();
            CudaDriverApi.cuMemsetD8Async(Ptr(kv.SlotPage), 0xFF, new UIntPtr((ulong)kv.Slots * 4), dev.Stream).ThrowOnError();
            CudaDriverApi.cuMemsetD8Async(Ptr(kv.SlotStamp), 0, new UIntPtr((ulong)kv.Slots * 4), dev.Stream).ThrowOnError();
            CudaDriverApi.cuMemsetD8Async(Ptr(kv.Ctl), 0, new UIntPtr((ulong)Q4eKernels.KvCtlInts * 4), dev.Stream).ThrowOnError();
        }

        /// <summary>The device's prompt staging pool: one layer's K/V at the full context.</summary>
        private void AllocKvStaging(Dev dev)
        {
            long full = (long)_m.NCtx * _m.NKvHead * _m.HeadDim;
            dev.KvStageK = AllocT(dev, DType.UInt8, full);
            dev.KvStageV = AllocT(dev, DType.UInt8, full);
            dev.KvStageKs = AllocT(dev, DType.Float16, full / Q4eKernels.KvQ8Block);
            dev.KvStageVs = AllocT(dev, DType.Float16, full / Q4eKernels.KvQ8Block);
        }

        /// <summary>A prompt call on a streamed layer: the staging pool holds cells [0, p0) of this layer and slot (copied
        /// from the host copy unless the previous call of the same layer and slot left them there).</summary>
        private void KvStageForPrompt(Dev dev, SlotLayer c, int p0)
        {
            if (ReferenceEquals(dev.KvStagedFor, c) && dev.KvStagedEnd == p0)
                return;
            var kv = c.Kv;
            long codes = (long)p0 * _m.NKvHead * _m.HeadDim, scales = codes / Q4eKernels.KvQ8Block * 2;
            if (codes > 0)
            {
                CudaDriverApi.cuMemcpyHtoDAsync(Ptr(dev.KvStageK), kv.HostK, new UIntPtr((ulong)codes), dev.Stream).ThrowOnError();
                CudaDriverApi.cuMemcpyHtoDAsync(Ptr(dev.KvStageV), kv.HostV, new UIntPtr((ulong)codes), dev.Stream).ThrowOnError();
                CudaDriverApi.cuMemcpyHtoDAsync(Ptr(dev.KvStageKs), kv.HostKs, new UIntPtr((ulong)scales), dev.Stream).ThrowOnError();
                CudaDriverApi.cuMemcpyHtoDAsync(Ptr(dev.KvStageVs), kv.HostVs, new UIntPtr((ulong)scales), dev.Stream).ThrowOnError();
                KvStagedBytes += 2 * (codes + scales);
            }
            dev.KvStagedFor = c;
            dev.KvStagedEnd = p0;
        }

        /// <summary>After a prompt call's K/V writes into the staging pool: cells [p0, p0 + nt) back to the host copy, and
        /// their pages out of the window (a resident slot of theirs holds the old cells).</summary>
        private void KvUnstagePrompt(Dev dev, SlotLayer c, int p0, int nt)
        {
            var kv = c.Kv;
            long row = (long)_m.NKvHead * _m.HeadDim, off = p0 * row, codes = nt * row;
            long sOff = off / Q4eKernels.KvQ8Block * 2, scales = codes / Q4eKernels.KvQ8Block * 2;
            CudaDriverApi.cuMemcpyDtoHAsync(kv.HostK + (nint)off, Ptr(dev.KvStageK) + (nint)off, new UIntPtr((ulong)codes), dev.Stream).ThrowOnError();
            CudaDriverApi.cuMemcpyDtoHAsync(kv.HostV + (nint)off, Ptr(dev.KvStageV) + (nint)off, new UIntPtr((ulong)codes), dev.Stream).ThrowOnError();
            CudaDriverApi.cuMemcpyDtoHAsync(kv.HostKs + (nint)sOff, Ptr(dev.KvStageKs) + (nint)sOff, new UIntPtr((ulong)scales), dev.Stream).ThrowOnError();
            CudaDriverApi.cuMemcpyDtoHAsync(kv.HostVs + (nint)sOff, Ptr(dev.KvStageVs) + (nint)sOff, new UIntPtr((ulong)scales), dev.Stream).ThrowOnError();
            int first = p0 / Q4eKernels.KvPage, last = (p0 + nt - 1) / Q4eKernels.KvPage;
            dev.QK.KvInvalidate(Ptr(kv.Table), Ptr(kv.SlotPage), first, last - first + 1, dev.Stream);
            dev.KvStagedEnd = p0 + nt;
        }

        /// <summary>Bytes the prompt path copied from the host copies into the staging pools so far.</summary>
        internal long KvStagedBytes { get; private set; }

        /// <summary>The active slot's streamed layers, summed: pages the resolves fetched over the bus, resolve calls, and
        /// calls that could not place every page (never, with the window's floor). Synchronizes; -1s when nothing streams.</summary>
        internal (long Misses, long Calls, long Overflows) KvStreamCounters() => KvStreamCounters(_active);

        private (long Misses, long Calls, long Overflows) KvStreamCounters(Slot slot)
        {
            if (_kvWindow == 0 || slot == null)
                return (-1, -1, -1);
            long misses = 0, calls = 0, over = 0;
            for (int il = 0; il < _m.NLayer; il++)
            {
                var kv = slot.Layers[il]?.Kv;
                if (kv == null)
                    continue;
                var dev = _devs[_layers[il].Device];
                dev.MakeCurrent();
                CudaDriverApi.cuStreamSynchronize(dev.Stream).ThrowOnError();
                var ctl = new int[Q4eKernels.KvCtlInts];
                fixed (int* p = ctl)
                    CudaDriverApi.cuMemcpyDtoH((IntPtr)p, Ptr(kv.Ctl), new UIntPtr((ulong)ctl.Length * 4)).ThrowOnError();
                misses += (uint)ctl[4];
                calls += (uint)ctl[5];
                over += (uint)ctl[3];
            }
            return (misses, calls, over);
        }

        private long _kvFetchedLogged, _kvStagedLogged;
        private int _kvNoteCalls;

        /// <summary>TS_Q4E_PERF lines: the pages fetched over the bus and the bytes staged for prompts since the last note
        /// (reads every streamed layer's counters, so only every <paramref name="every"/>-th call; empty when nothing
        /// streams).</summary>
        private string KvFetchedNote(Slot slot, int every = 1)
        {
            if (_kvWindow == 0 || ++_kvNoteCalls % every != 0)
                return "";
            var (misses, _, over) = KvStreamCounters(slot);
            string note = $", KV {misses - _kvFetchedLogged} pages fetched, {(KvStagedBytes - _kvStagedLogged) / 1048576.0:F0} MiB staged"
                + (over > 0 ? $", {over} overflowing resolves" : "");
            _kvFetchedLogged = misses;
            _kvStagedLogged = KvStagedBytes;
            return note;
        }

        /// <summary>The cells a streamed layer keeps in VRAM (0: every layer resident).</summary>
        internal int KvWindowCells => _kvWindow;
    }
}
