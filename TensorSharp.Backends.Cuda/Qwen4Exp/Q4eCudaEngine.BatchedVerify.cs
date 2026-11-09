// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Batched speculative verify on the Qwen3.8-Flash-Next direct-CUDA engine: the draft windows of several sequences, each
// in its own slot, verified in one forward. Without it concurrent requests decoded one token a step in a batched step
// while a lone request verified a whole window a step, so a second request in flight cut every request to plain-decode
// speed (four at once on the RTX 5090: 192-203 tok/s in all, against 293-329 for Strata's batch slots).
//
// The forward is the batched decode step's (a row per slot and position, each row's attention over its own slot's
// caches, the GDN recurrence and the PLE history per sequence), with `window` consecutive rows a sequence: the
// recurrences run window rows a sequence (rowsPerSeq = window), and each row's first-max token comes from the device.
// Each sequence's recurrent states, the inputs its recurrence consumed and its PLE history are saved first, so the
// commit puts them back and replays the kept rows only, sequence by sequence, as the single-sequence verify does.
using System;
using System.Collections.Generic;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    public sealed unsafe partial class Q4eCudaEngine
    {
        /// <summary>One sequence's recurrent states and PLE history before a batched verify.</summary>
        private sealed class BatchVerifySeqSave
        {
            public Tensor[] Conv, Ssm;      // per layer (recurrent layers only)
            public Tensor PleHist;
        }

        /// <summary>Per sequence of the batched verify in flight (its position in the step): its saves. A pool the steps reuse
        /// (the saves live from the verify to its commit), grown to the widest step, which the lane plan reserves
        /// (BatchVerifyBytes). Slot-owned saves allocated on each slot's first batched verify and kept with the slot (retained
        /// conversations too) ran the RTX 5090 out of memory in Strata's concurrency rounds.</summary>
        private readonly List<BatchVerifySeqSave> _bvPool = new List<BatchVerifySeqSave>();
        // Per layer: the window rows' recurrence inputs ([PerSlotMaxRows, ConvDim] and the scan scratch), and the PLE rows.
        private Tensor[] _bvQkv, _bvScr;
        private Tensor _bvPleNorm;
        // Rows a sequence holds in the batched step in flight (GDN and PLE run rowsPerSeq = this); 1 for a batched decode.
        private int _batchRowsPerSeq = 1;
        // The batched verify in flight saves each GDN layer's recurrence inputs.
        private bool _batchVerifyCapture;
        private Slot[] _bvSlots;
        private int[] _bvP0;
        private int _bvWindow;

        /// <summary>Rows one batched verify takes at most (the decode-size kernels' bound).</summary>
        public int BatchedVerifyMaxRows => PerSlotMaxRows;

        /// <summary>
        /// Rows a batched verify step keeps to (TS_Q4E_BATCHED_VERIFY_ROWS, 2 to <see cref="PerSlotMaxRows"/>, default 8): the
        /// sequences split them into their windows. Rows of different sequences share few experts on this MoE, so a verify
        /// costs nearly in proportion to its rows (4 / 8 / 12 rows in 9.5 / 14.5 / 24.4 ms on the RTX 5090, each draft kept at
        /// about 0.62): three sequences with 4-row windows decoded 211 tok/s in all against 296 for four sequences' plain
        /// batched steps. At 8, two sequences verify 4 rows each, three or four 2; more decode plainly together. 8 rows also
        /// keep a step's missed experts on the host (TS_Q4E_HOST_EXPERT_ROWS) and its graph captured.
        /// </summary>
        public int BatchedVerifyRowBudget => _bvRowBudget;

        private readonly int _bvRowBudget = Math.Clamp(EnvInt("TS_Q4E_BATCHED_VERIFY_ROWS", 8), 2, PerSlotMaxRows);

        /// <summary>Whether the engine verifies several sequences' windows in one forward (one GPU, the device argmax).</summary>
        public bool SupportsBatchedVerify => _devs.Length == 1 && _devs[_lastDev].QK.HasArgmaxRows;

        /// <summary>Bytes of one sequence's saves: the recurrent layers' conv and SSM states, and the PLE history.</summary>
        private long SeqSaveBytes()
        {
            var m = _m;
            long b = 0;
            for (int il = 0; il < m.NLayer; il++)
                if (m.Layers[il].Recurrent)
                    b += ((long)(m.DConv - 1) * ConvDim + (long)m.GdnVHeads * Q4eKernels.GdnHeadDim * Q4eKernels.GdnHeadDim) * 4;
            if (m.Ple != null && m.PleConvKernel > 1)
                b += (long)PleHistRows * m.Hc * m.NEmbd * 4;
            return b;
        }

        /// <summary>Bytes a batched verify of up to <paramref name="sequences"/> sequences holds: the window rows' recurrence
        /// inputs and scan scratch, the PLE rows, and each sequence's saves.</summary>
        private long BatchVerifyBytes(int sequences)
        {
            var m = _m;
            long rows = 0;
            for (int il = 0; il < m.NLayer; il++)
                if (m.Layers[il].Recurrent)
                    rows += ((long)PerSlotMaxRows * ConvDim + (long)PerSlotMaxRows * m.GdnVHeads * GlmKernels.KdaScratch) * 4;
            rows += (long)PerSlotMaxRows * m.Hc * m.NEmbd * 4;
            return rows + sequences * SeqSaveBytes();
        }

        private BatchVerifySeqSave NewSeqSave()
        {
            var m = _m;
            var s = new BatchVerifySeqSave { Conv = new Tensor[m.NLayer], Ssm = new Tensor[m.NLayer] };
            for (int il = 0; il < m.NLayer; il++)
            {
                if (!m.Layers[il].Recurrent)
                    continue;
                var dev = _devs[_layers[il].Device];
                dev.MakeCurrent();
                s.Conv[il] = AllocF32(dev, (long)(m.DConv - 1) * ConvDim);
                s.Ssm[il] = AllocF32(dev, (long)m.GdnVHeads * Q4eKernels.GdnHeadDim * Q4eKernels.GdnHeadDim);
            }
            if (_pleLayer >= 0 && PleHistRows > 0)
                s.PleHist = AllocF32(_devs[_layers[_pleLayer].Device], (long)PleHistRows * HcDim);
            return s;
        }

        /// <summary>The window buffers and <paramref name="sequences"/> sequences' saves; false (nothing in flight changed) when
        /// the device has no room for them, and the step is then served without the batched verify.</summary>
        private bool EnsureBatchVerifyBuffers(int sequences)
        {
            var m = _m;
            try
            {
                if (_bvQkv == null)
                {
                    var qkv = new Tensor[m.NLayer];
                    var scr = new Tensor[m.NLayer];
                    for (int il = 0; il < m.NLayer; il++)
                    {
                        if (!m.Layers[il].Recurrent)
                            continue;
                        var dev = _devs[_layers[il].Device];
                        dev.MakeCurrent();
                        qkv[il] = AllocF32(dev, PerSlotMaxRows, ConvDim);
                        scr[il] = AllocF32(dev, (long)PerSlotMaxRows * m.GdnVHeads * GlmKernels.KdaScratch);
                    }
                    if (_pleLayer >= 0 && PleHistRows > 0)
                        _bvPleNorm = AllocF32(_devs[_layers[_pleLayer].Device], PerSlotMaxRows, HcDim);
                    _bvScr = scr;
                    _bvQkv = qkv;
                }
                while (_bvPool.Count < sequences)
                    _bvPool.Add(NewSeqSave());
                return true;
            }
            catch (Exception ex) when (ex is CudaException || ex is OutOfMemoryException)
            {
                if (!_bvOomWarned)
                {
                    _bvOomWarned = true;
                    Console.Error.WriteLine($"[q4e-cuda] a batched verify of {sequences} sequences has no device memory for its saves "
                        + $"({ex.Message}); such steps decode without their drafts. Reported once.");
                }
                return false;
            }
        }

        private bool _bvOomWarned;

        /// <summary>
        /// Verify <paramref name="window"/> tokens of each of several slots in one forward: <paramref name="tokens"/> is
        /// [slots * window], each slot's tokens from its head on. Each row's first-max token goes to
        /// <paramref name="drawnOut"/> and, when given, its hidden streams to <paramref name="hAll"/>. Every slot's head
        /// moves past its window; <see cref="CommitBatchedVerify"/> keeps each slot's accepted rows. False (nothing
        /// written) when the step cannot be served this way.
        /// </summary>
        public bool ForwardBatchedVerify(int[] slotIds, int window, int[] tokens, int[] drawnOut, float[] hAll,
            float[] logitsOut = null)
        {
            int n = slotIds?.Length ?? 0;
            if (!SupportsBatchedVerify || n < 1 || window < 1 || window > VerifyMaxRows || (long)n * window > PerSlotMaxRows
                || tokens == null || tokens.Length < n * window)
                return false;
            if (logitsOut == null ? drawnOut == null || drawnOut.Length < n * window : logitsOut.LongLength < (long)n * window * _m.NVocab)
                return false;
            var slots = new Slot[n];
            var seen = new HashSet<int>();
            for (int i = 0; i < n; i++)
            {
                if (!seen.Add(slotIds[i]) || !_slots.TryGetValue(slotIds[i], out slots[i]) || slots[i].Failed
                    || slots[i].NPast + window > _m.NCtx)
                    return false;
            }
            // Each window's rows land in its slot's caches: grow them first. A slot that cannot grow sends the step back
            // to the sequences' own steps, which report why.
            try
            {
                foreach (var slot in slots)
                    EnsureCapacity(slot, slot.NPast + window);
            }
            catch (InvalidOperationException)
            {
                return false;
            }
            int nt = n * window;
            var rows = new Slot[nt];
            var positions = new int[nt];
            var p0 = new int[n];
            for (int i = 0; i < n; i++)
            {
                p0[i] = slots[i].NPast;
                for (int j = 0; j < window; j++)
                {
                    rows[i * window + j] = slots[i];
                    positions[i * window + j] = p0[i] + j;
                }
            }

            if (!EnsureBatchVerifyBuffers(n))
                return false;
            var m = _m;
            // Each sequence's states before the window.
            for (int i = 0; i < n; i++)
            {
                var slot = slots[i];
                var s = _bvPool[i];
                for (int il = 0; il < m.NLayer; il++)
                {
                    if (!m.Layers[il].Recurrent)
                        continue;
                    var dev = _devs[_layers[il].Device];
                    dev.MakeCurrent();
                    var c = slot.Layers[il];
                    CopyD2D(dev, s.Conv[il], c.ConvState, c.ConvState.ElementCount() * 4);
                    CopyD2D(dev, s.Ssm[il], c.Ssm, c.Ssm.ElementCount() * 4);
                }
                if (s.PleHist != null && slot.PleHist != null)
                    CopyD2D(_devs[_layers[_pleLayer].Device], s.PleHist, slot.PleHist, s.PleHist.ElementCount() * 4);
            }

            StageBatchStates(slots);
            foreach (var slot in slots)
                slot.Failed = true;
            _rowsInFlight = nt;
            _batchRowsPerSeq = window;
            _batchVerifyCapture = true;
            ArmHostExperts();
            try
            {
                // Each row's first-max token from the device, or (a sampled verify) every row's logits.
                if (!ForwardBatchGraphed(rows, tokens, positions, window, greedy: logitsOut == null, hidden: hAll != null))
                {
                    ForwardUbatch(null, tokens, 0, nt, 0, null, rows, positions);
                    IssueHead(nt, logits: true, allRows: true, hidden: hAll != null, greedy: logitsOut == null);
                }
                FinishHead(nt, logitsOut, true, hAll, 0, logitsOut == null ? drawnOut : null);
            }
            finally
            {
                _rowsInFlight = 0;
                _batchRowsPerSeq = 1;
                _batchVerifyCapture = false;
            }
            CheckHostExperts(nt);
            for (int i = 0; i < n; i++)
            {
                slots[i].NPast = p0[i] + window;
                slots[i].Failed = false;
            }
            _bvSlots = slots;
            _bvP0 = p0;
            _bvWindow = window;
            AdaptResidency();
            return true;
        }

        /// <summary>In a batched verify: a GDN layer's recurrence inputs for every row, once its prep has run.</summary>
        private void BatchVerifySaveGdnInputs(Dev dev, int il, int nt)
        {
            CopyD2D(dev, _bvQkv[il], dev.Qkv, (long)nt * ConvDim * 4);
            CopyD2D(dev, _bvScr[il], dev.Scr, (long)nt * _m.GdnVHeads * GlmKernels.KdaScratch * 4);
        }

        /// <summary>In a batched verify: the PLE rows the histories append.</summary>
        private void BatchVerifySavePle(Dev dev, int nt)
        {
            if (_bvPleNorm != null)
                CopyD2D(dev, _bvPleNorm, dev.PleNorm, (long)nt * HcDim * 4);
        }

        /// <summary>
        /// Keep <paramref name="kept"/>[i] rows (1 to the window) of slot i of the last batched verify: its recurrent states
        /// and PLE history are put back and stepped over those rows, and its head moves to its window's start + kept.
        /// False when there is no batched verify to commit or the slots differ from it.
        /// </summary>
        public bool CommitBatchedVerify(int[] slotIds, int[] kept)
        {
            var slots = _bvSlots;
            if (slots == null || slotIds == null || kept == null || slotIds.Length != slots.Length || kept.Length < slots.Length)
                return false;
            for (int i = 0; i < slots.Length; i++)
                if (slots[i].Id != slotIds[i] || kept[i] < 1 || kept[i] > _bvWindow)
                    return false;
            _bvSlots = null;
            var m = _m;
            int window = _bvWindow, scrRow = GlmKernels.KdaScratch * 4;
            for (int i = 0; i < slots.Length; i++)
            {
                var slot = slots[i];
                int keep = kept[i], p0 = _bvP0[i];
                if (keep < window)
                {
                    var s = _bvPool[i];
                    for (int il = 0; il < m.NLayer; il++)
                    {
                        if (!m.Layers[il].Recurrent)
                            continue;
                        var dev = _devs[_layers[il].Device];
                        dev.MakeCurrent();
                        var c = slot.Layers[il];
                        CopyD2D(dev, c.ConvState, s.Conv[il], c.ConvState.ElementCount() * 4);
                        CopyD2D(dev, c.Ssm, s.Ssm[il], c.Ssm.ElementCount() * 4);
                        // Sequence i's rows of the saved inputs: its window's first `keep` rows.
                        dev.QK.ConvUpdate((IntPtr)((long)Ptr(_bvQkv[il]) + (long)i * window * ConvDim * 4), StatePtr(slot, dev, il, 0),
                            keep, 1, ConvDim, m.DConv - 1, dev.Stream);
                        // The scan inputs are [sequence][heads][rows][scratch]: sequence i's kept rows of each head, packed.
                        var copy = new CudaDriverApi.CudaMemcpy2D
                        {
                            SrcMemoryType = CudaDriverApi.CU_MEMORYTYPE_DEVICE,
                            SrcDevice = (IntPtr)((long)Ptr(_bvScr[il]) + (long)i * m.GdnVHeads * window * scrRow),
                            SrcPitch = new UIntPtr((ulong)(window * scrRow)),
                            DstMemoryType = CudaDriverApi.CU_MEMORYTYPE_DEVICE, DstDevice = Ptr(dev.Scr),
                            DstPitch = new UIntPtr((ulong)(keep * scrRow)),
                            WidthInBytes = new UIntPtr((ulong)(keep * scrRow)), Height = new UIntPtr((ulong)m.GdnVHeads),
                        };
                        CudaDriverApi.cuMemcpy2DAsync(copy, dev.Stream).ThrowOnError();
                        dev.GK.KdaScan(Ptr(dev.Scr), StatePtr(slot, dev, il, 1), Ptr(dev.Core), keep, 1, m.GdnVHeads, dev.Stream);
                    }
                    if (s.PleHist != null && slot.PleHist != null && _bvPleNorm != null)
                    {
                        var dev = _devs[_layers[_pleLayer].Device];
                        dev.MakeCurrent();
                        CopyD2D(dev, slot.PleHist, s.PleHist, s.PleHist.ElementCount() * 4);
                        dev.QK.ConvUpdate((IntPtr)((long)Ptr(_bvPleNorm) + (long)i * window * HcDim * 4), StatePtr(slot, dev, m.NLayer, 0),
                            keep, 1, HcDim, PleHistRows, dev.Stream);
                    }
                }
                slot.PleTokens.Length = Math.Min(slot.PleTokens.Length, p0 + keep);
                slot.NPast = p0 + keep;
            }
            foreach (var dev in _devs)
            {
                dev.MakeCurrent();
                CudaDriverApi.cuStreamSynchronize(dev.Stream).ThrowOnError();
            }
            return true;
        }
    }
}
