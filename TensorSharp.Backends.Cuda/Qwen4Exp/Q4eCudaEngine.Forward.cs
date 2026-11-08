// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// The Qwen3.8-Flash-Next direct-CUDA forward: scratch, one sequence's caches, and the per-layer
// pipeline (see Q4eCudaEngine.cs for the model and its references).
using System;
using System.Runtime.InteropServices;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    public sealed unsafe partial class Q4eCudaEngine
    {
        private void AllocateScratch(Dev dev)
        {
            dev.MakeCurrent();
            var m = _m;
            int nt = m.NUbatch, e = m.NEmbd;
            long hcE = HcDim;
            bool anyGdn = false, anyAttn = false, anyQsa = false, ple = false;
            for (int il = 0; il < m.NLayer; il++)
            {
                var L = _layers[il];
                if (L.Device != dev.Ordinal)
                    continue;
                anyGdn |= L.Recurrent;
                anyAttn |= !L.Recurrent;
                anyQsa |= L.Ratio > 0;
                ple |= L.Ple;
            }

            dev.Xs = AllocF32(dev, nt, hcE);
            dev.Xn = AllocF32(dev, nt, hcE);
            dev.G = AllocF32(dev, nt, hcE);
            dev.Lo = AllocF32(dev, nt, m.HcLowRank);
            dev.Inject = AllocF32(dev, nt, m.Hc);
            dev.HcPartials = AllocF32(dev, PerSlotMaxRows, hcE / Q4eKernels.HcSlice * Q4eKernels.HcPartialFloats);
            dev.Cur = AllocF32(dev, nt, e);
            dev.BlockOut = AllocF32(dev, nt, e);
            // Every decode-size projection input: the widest is the hyper-connection streams.
            int sharedIn = Math.Max(Math.Max(Math.Max(e, (int)hcE), Math.Max(ValueDim, m.NHead * m.HeadDim)), Math.Max(m.NFfShexp, m.HcLowRank));
            dev.SharedQ81 = AllocT(dev, DType.UInt8, PerSlotMaxRows, (long)(sharedIn / 32) * 36);
            dev.GemvScratch = AllocF32(dev, Dsv4Kernels.GemvScratchFloats);
            if (anyGdn)
            {
                dev.Qkv = AllocF32(dev, nt, ConvDim);
                dev.Z = AllocF32(dev, nt, ValueDim);
                dev.Alpha = AllocF32(dev, nt, m.GdnVHeads);
                dev.Beta = AllocF32(dev, nt, m.GdnVHeads);
                dev.Scr = AllocF32(dev, (long)nt * m.GdnVHeads * GlmKernels.KdaScratch);
                dev.Core = AllocF32(dev, nt, ValueDim);
                dev.GOut = AllocF32(dev, nt, ValueDim);
            }
            if (anyAttn)
            {
                long qd = (long)m.NHead * m.HeadDim;
                dev.Qg = AllocF32(dev, nt, 2 * qd);
                dev.K = AllocF32(dev, nt, (long)m.NKvHead * m.HeadDim);
                dev.V = AllocF32(dev, nt, (long)m.NKvHead * m.HeadDim);
                dev.Q = AllocF32(dev, nt, qd);
                dev.Gate = AllocF32(dev, nt, qd);
                dev.AttnO = AllocF32(dev, nt, qd);
            }
            if (anyQsa)
            {
                dev.IdxRaw = AllocF32(dev, nt, m.IdxDim);
                dev.IdxQ = AllocF32(dev, nt, (long)m.IdxHeads * m.IdxDim);
                dev.Scores = AllocF32(dev, nt, ScoreStride);
                dev.Cells = AllocT(dev, DType.Int32, nt, CellStride);
                dev.CellCnt = AllocT(dev, DType.Int32, nt);
            }
            if (ple)
            {
                dev.PleEmb = AllocF32(dev, nt, e);
                dev.PleKeyOut = AllocF32(dev, nt, hcE);
                dev.PleValueOut = AllocF32(dev, nt, e);
                dev.PleGated = AllocF32(dev, nt, hcE);
                dev.PleNorm = AllocF32(dev, nt, hcE);
                CudaDriverApi.cuMemHostAlloc(out dev.PlePinned, new UIntPtr((ulong)nt * (ulong)e * 4), 0x1).ThrowOnError();
            }
            dev.RouterLogits = AllocF32(dev, nt, m.NExpert);
            dev.Moe = new CudaMoeScratch((type, sizes) => AllocT(dev, type, sizes), nt, m.NExpert, m.NExpertUsed, e, m.NFfExp);
            dev.ShGate = AllocF32(dev, nt, m.NFfShexp);
            dev.ShUp = AllocF32(dev, nt, m.NFfShexp);
            dev.ShDown = AllocF32(dev, nt, e);
            // One position per row of a captured step: a solo decode reads element 0, a batched one
            // row r's at element r.
            dev.PosDev = AllocT(dev, DType.Int32, MaxBatchedDecodeRows);
            if (dev.Ordinal == 0)
            {
                dev.Tokens = AllocT(dev, DType.Int32, nt);
                // PORTABLE: every device may read them.
                CudaDriverApi.cuMemHostAlloc(out _pinnedTokens, new UIntPtr((ulong)nt * 4), 0x1).ThrowOnError();
                CudaDriverApi.cuMemHostAlloc(out _posPinned, new UIntPtr((ulong)MaxBatchedDecodeRows * 4), 0x1).ThrowOnError();
            }
            if (dev.Ordinal == _lastDev)
            {
                dev.Logits = AllocF32(dev, 1, m.NVocab);
                dev.BatchLogits = AllocF32(dev, MaxBatchedDecodeRows, m.NVocab);
                CudaDriverApi.cuMemHostAlloc(out _pinnedLogits,
                    new UIntPtr((ulong)MaxBatchedDecodeRows * (ulong)m.NVocab * 4), 0x1).ThrowOnError();
                CudaDriverApi.cuMemHostAlloc(out _pinnedHidden,
                    new UIntPtr((ulong)MaxBatchedDecodeRows * (ulong)m.NEmbd * 4), 0x1).ThrowOnError();
            }
            if (anyGdn || ple)
            {
                dev.BatchStatePtrs = AllocT(dev, DType.UInt8, BatchStateTableBytes);
                CudaDriverApi.cuMemHostAlloc(out dev.BatchStatePinned, new UIntPtr((ulong)BatchStateTableBytes), 0x1).ThrowOnError();
            }
            CudaDriverApi.cuMemHostAlloc(out dev.BoundaryPinned, new UIntPtr((ulong)((long)nt * hcE * 4)), 0x1).ThrowOnError();
            CudaDriverApi.cuEventCreate(out dev.XsReadyEv, 0x02 /*DISABLE_TIMING*/).ThrowOnError();
            CudaDriverApi.cuEventCreate(out dev.CopyDoneEv, 0x02).ThrowOnError();
        }

        /// <summary>Row stride of the block scores: every block the context holds.</summary>
        private int ScoreStride => Math.Max(1, _m.NCtx / MinRatio);

        /// <summary>Cells a sparse row attends: top_k plus the rest of the query's own block.</summary>
        private int CellStride => _m.IdxTopK + MaxRatio - 1;

        private int MinRatio
        {
            get
            {
                int r = int.MaxValue;
                foreach (var L in _layers)
                    if (L.Ratio > 0) r = Math.Min(r, L.Ratio);
                return r == int.MaxValue ? 1 : r;
            }
        }

        private int MaxRatio
        {
            get
            {
                int r = 1;
                foreach (var L in _layers)
                    if (L.Ratio > 0) r = Math.Max(r, L.Ratio);
                return r;
            }
        }

        // -------------------------------------------------------------------
        // Sequence caches
        // -------------------------------------------------------------------

        private Slot CreateSlot()
        {
            var m = _m;
            int capacity = InitialSlotTokens;
            var slot = new Slot
            {
                Id = _nextSlotId++,
                Capacity = capacity,
                Layers = new SlotLayer[m.NLayer],
                StatePtrs = new Tensor[_devs.Length],
                DeviceBytes = new long[_devs.Length],
            };
            var ptrs = new long[_devs.Length][];
            for (int d = 0; d < _devs.Length; d++)
                ptrs[d] = new long[2 * m.NLayer + 1];
            for (int il = 0; il < m.NLayer; il++)
            {
                var L = _layers[il];
                var dev = _devs[L.Device];
                dev.MakeCurrent();
                var c = new SlotLayer();
                if (L.Recurrent)
                {
                    c.ConvState = SlotTensor(slot, dev, DType.Float32, (long)(m.DConv - 1) * ConvDim);
                    c.Ssm = SlotTensor(slot, dev, DType.Float32, (long)m.GdnVHeads * Q4eKernels.GdnHeadDim * Q4eKernels.GdnHeadDim);
                    ptrs[dev.Ordinal][2 * il] = (long)Ptr(c.ConvState);
                    ptrs[dev.Ordinal][2 * il + 1] = (long)Ptr(c.Ssm);
                    if (_rewindCheckpoint)
                    {
                        slot.ConvCp ??= new Tensor[m.NLayer];
                        slot.SsmCp ??= new Tensor[m.NLayer];
                        slot.ConvCp[il] = SlotTensor(slot, dev, DType.Float32, c.ConvState.ElementCount());
                        slot.SsmCp[il] = SlotTensor(slot, dev, DType.Float32, c.Ssm.ElementCount());
                    }
                }
                else
                {
                    c = AllocAttentionCaches(dev, L, capacity, slot.DeviceBytes);
                    foreach (Tensor t in new[] { c.KCache, c.VCache, c.IdxRaw, c.IdxPooled })
                        if (t != null)
                            slot.Owned.Add(t);
                }
                if (L.Ple && PleHistRows > 0)
                {
                    slot.PleHist = SlotTensor(slot, dev, DType.Float32, (long)PleHistRows * HcDim);
                    ptrs[dev.Ordinal][2 * m.NLayer] = (long)Ptr(slot.PleHist);
                    if (_rewindCheckpoint)
                        slot.PleHistCp = SlotTensor(slot, dev, DType.Float32, (long)PleHistRows * HcDim);
                }
                slot.Layers[il] = c;
            }
            for (int d = 0; d < _devs.Length; d++)
            {
                var dev = _devs[d];
                dev.MakeCurrent();
                long bytes = 8L * (2 * m.NLayer + 1);
                slot.StatePtrs[d] = SlotTensor(slot, dev, DType.UInt8, bytes);
                fixed (long* p = ptrs[d])
                    CudaDriverApi.cuMemcpyHtoD(Ptr(slot.StatePtrs[d]), (IntPtr)p, new UIntPtr((ulong)bytes)).ThrowOnError();
            }
            ClearSlot(slot);
            return slot;
        }

        /// <summary>An attention layer's caches for <paramref name="capacity"/> positions (K/V, and the
        /// QSA raw and block keys), their bytes added to <paramref name="deviceBytes"/>.</summary>
        private SlotLayer AllocAttentionCaches(Dev dev, DevLayer L, int capacity, long[] deviceBytes)
        {
            var m = _m;
            dev.MakeCurrent();
            var c = new SlotLayer();
            Tensor Make(DType type, long elements)
            {
                var t = new Tensor(dev.Alloc, type, elements);
                deviceBytes[dev.Ordinal] += elements * type.Size();
                return t;
            }
            try
            {
                long kv = (long)capacity * m.NKvHead * m.HeadDim;
                c.KCache = Make(DType.Float16, kv);
                c.VCache = Make(DType.Float16, kv);
                if (L.Ratio > 0)
                {
                    c.IdxRaw = Make(DType.Float16, (long)capacity * m.IdxDim);
                    c.IdxPooled = Make(DType.Float32, (long)(capacity / L.Ratio) * m.IdxDim);
                }
                return c;
            }
            catch
            {
                c.KCache?.Dispose(); c.VCache?.Dispose(); c.IdxRaw?.Dispose();
                throw;
            }
        }

        private static Tensor SlotTensor(Slot slot, Dev dev, DType type, long elements)
        {
            dev.MakeCurrent();
            var t = new Tensor(dev.Alloc, type, elements);
            slot.Owned.Add(t);
            slot.DeviceBytes[dev.Ordinal] += elements * type.Size();
            return t;
        }

        /// <summary>Zero the recurrent states and the PLE history (an attention cache needs no
        /// clearing: nothing past the head is ever read).</summary>
        private void ClearSlot(Slot slot)
        {
            for (int il = 0; il < _m.NLayer; il++)
            {
                var c = slot.Layers[il];
                var dev = _devs[_layers[il].Device];
                dev.MakeCurrent();
                if (c.ConvState != null)
                {
                    Zero(dev, c.ConvState, 4);
                    Zero(dev, c.Ssm, 4);
                }
                if (_layers[il].Ple && slot.PleHist != null)
                    Zero(dev, slot.PleHist, 4);
            }
            slot.NPast = 0;
            slot.Failed = false;
            slot.PleTokens.Length = 0;
            slot.CpNPast = -1;
        }

        /// <summary>Copy <paramref name="slot"/>'s recurrent states into its checkpoint
        /// (<paramref name="restore"/> false) or back (true), on each layer's device stream.</summary>
        private void CopyCheckpoint(Slot slot, bool restore)
        {
            for (int il = 0; il < _m.NLayer; il++)
            {
                if (slot.ConvCp?[il] == null)
                    continue;
                var dev = _devs[_layers[il].Device];
                dev.MakeCurrent();
                var c = slot.Layers[il];
                CopyD(dev, restore ? c.ConvState : slot.ConvCp[il], restore ? slot.ConvCp[il] : c.ConvState);
                CopyD(dev, restore ? c.Ssm : slot.SsmCp[il], restore ? slot.SsmCp[il] : c.Ssm);
            }
            if (slot.PleHistCp != null)
            {
                var dev = _devs[_layers[_pleLayer].Device];
                dev.MakeCurrent();
                CopyD(dev, restore ? slot.PleHist : slot.PleHistCp, restore ? slot.PleHistCp : slot.PleHist);
            }
        }

        private static void CopyD(Dev dev, Tensor dst, Tensor src)
            => CudaDriverApi.cuMemcpyDtoDAsync(Ptr(dst), Ptr(src), new UIntPtr((ulong)src.ElementCount() * 4), dev.Stream).ThrowOnError();

        /// <summary>The active slot's states, at its head, become its rewind checkpoint: called at the
        /// end of every multi-token forward, a prompt boundary.</summary>
        private void CheckpointActive()
        {
            Slot s = _active;
            if (!_rewindCheckpoint || s.Failed || s.ConvCp == null)
                return;
            CopyCheckpoint(s, restore: false);
            s.CpNPast = s.NPast;
        }

        private static void Zero(Dev dev, Tensor t, int elementBytes)
            => CudaDriverApi.cuMemsetD8Async(Ptr(t), 0, new UIntPtr((ulong)t.ElementCount() * (ulong)elementBytes), dev.Stream).ThrowOnError();

        private void FreeSlot(Slot slot)
        {
            if (slot == null)
                return;
            foreach (var dev in _devs)
            {
                dev.MakeCurrent();
                CudaDriverApi.cuStreamSynchronize(dev.Stream);
            }
            DestroySlotGraphs(slot.Id);
            CountSlot(slot, -1);
            foreach (var t in slot.Owned)
                t.Dispose();
            slot.Owned.Clear();
        }

        /// <summary>Back to position 0 on the active sequence.</summary>
        public void Reset() => ClearSlot(_active);

        /// <summary>Address of this slot's pointer to its [conv, ssm] state of a GDN layer (kind 0, 1)
        /// on <paramref name="dev"/>, or to its PLE history (il = NLayer, kind 0): a one-sequence
        /// pointer array for the kernels.</summary>
        private static IntPtr StatePtr(Slot slot, Dev dev, int il, int kind)
            => (IntPtr)((long)Ptr(slot.StatePtrs[dev.Ordinal]) + (2L * il + kind) * 8);

        // -------------------------------------------------------------------
        // Forward
        // -------------------------------------------------------------------

        public void Forward(int[] tokens, float[] logitsOut)
        {
            if (tokens == null || tokens.Length == 0)
                throw new ArgumentException("empty token batch", nameof(tokens));
            Slot slot = _active;
            if (slot.Failed)
                throw new InvalidOperationException("[q4e-cuda] the sequence's last forward failed; reset it before reuse");
            if (slot.NPast + tokens.Length > _m.NCtx)
                throw new InvalidOperationException($"[q4e-cuda] context overflow: n_past={slot.NPast} + {tokens.Length} > n_ctx={_m.NCtx}");

            var sw = System.Diagnostics.Stopwatch.StartNew();
            long start = System.Diagnostics.Stopwatch.GetTimestamp();
            EnsureCapacity(slot, slot.NPast + tokens.Length);
            bool sampled = logitsOut != null && BeginSampledForward();
            _issueEnd = 0;
            slot.Failed = true;
            try
            {
                if (tokens.Length == 1 && logitsOut != null && CanCaptureDecode && ForwardDecodeGraphed(slot, tokens[0], slot.NPast, logitsOut))
                {
                    slot.NPast++;
                }
                else
                {
                    int done = 0;
                    while (done < tokens.Length)
                    {
                        int nt = Math.Min(_m.NUbatch, tokens.Length - done);
                        bool last = done + nt == tokens.Length;
                        ForwardUbatch(slot, tokens, done, nt, slot.NPast, last ? logitsOut : null);
                        slot.NPast += nt;
                        done += nt;
                    }
                }
            }
            finally
            {
                if (sampled)
                {
                    try { EndSampledForward(tokens.Length, tokens.Length == 1); }
                    catch (Exception) { _sampling = false; }
                }
            }
            CountForwardTime(start);
            slot.Failed = false;
            if (tokens.Length >= 2)
                CheckpointActive();
            if (_perf > 0)
            {
                double secs = sw.Elapsed.TotalSeconds;
                Console.Error.WriteLine($"[q4e-cuda] forward {tokens.Length} tokens in {secs:F3}s ({tokens.Length / secs:F1} tok/s), "
                    + $"host issue {IssueMs(start):F1}ms");
            }
        }

        /// <param name="rows">A batched decode step: one row per sequence (<paramref name="slot"/> is
        /// then unused), each at <paramref name="positions"/>[row]; null for one sequence's ubatch at
        /// <paramref name="p0"/>.</param>
        private void ForwardUbatch(Slot slot, int[] tokens, int tokOff, int nt, int p0, float[] logitsOut,
            Slot[] rows = null, int[] positions = null, bool allRows = false, float[] hOut = null)
        {
            var m = _m;
            int e = m.NEmbd;
            var dev0 = _devs[0];
            dev0.MakeCurrent();
            // The pinned tokens are rewritten per ubatch: the previous upload must have run.
            CudaDriverApi.cuStreamSynchronize(dev0.Stream).ThrowOnError();
            Marshal.Copy(tokens, tokOff, _pinnedTokens, nt);
            CudaDriverApi.cuMemcpyHtoDAsync(Ptr(dev0.Tokens), _pinnedTokens, new UIntPtr((ulong)nt * 4), dev0.Stream).ThrowOnError();
            StageBegin(dev0);
            dev0.DK.Embed(_tokEmbd.Ptr, dev0.Tokens, dev0.Xs, _tokEmbd.Type, _tokEmbd.RowBytes, nt, e, dev0.Stream);
            StageEnd(dev0, StEmbed);

            // The PLE rows are hashed and gathered on the host before the step's GPU work, as the
            // native span does: the sequences' token histories take this step's tokens first.
            if (_pleLayer >= 0)
                GatherPleRows(slot, tokens, tokOff, nt, p0, rows, positions);

            int curDev = 0;
            for (int il = 0; il < m.NLayer; il++)
            {
                var L = _layers[il];
                if (L.Device != curDev)
                {
                    Handoff(_devs[curDev], _devs[L.Device], nt);
                    curDev = L.Device;
                    StageEnd(_devs[curDev], StBoundary);
                }
                var dev = _devs[curDev];
                dev.MakeCurrent();
                RunLayer(dev, il, slot, nt, p0, rows, positions, IntPtr.Zero);
            }

            if (logitsOut != null || hOut != null)
                Head(nt, logitsOut, rows != null || allRows, hOut);
        }

        /// <summary>One layer: the PLE block where it sits, then both mixed halves. <paramref name="pos"/>
        /// is the device copy of the row's position a captured decode step reads (zero: p0 is used); in
        /// a captured batched step (<paramref name="rows"/> set) row r reads element r.</summary>
        private void RunLayer(Dev dev, int il, Slot slot, int nt, int p0, Slot[] rows, int[] positions, IntPtr pos)
        {
            var m = _m;
            var L = _layers[il];
            int e = m.NEmbd;
            if (L.Ple)
                Ple(dev, L, slot, nt, rows);

            HcPre(dev, L.HcAttnNorm, L.HcAttnDown, L.HcAttnUp, L.HcAttnInject, nt);
            StageEnd(dev, StHc);
            if (L.Recurrent)
                Gdn(dev, L, il, slot, nt, rows);
            else if (rows != null)
                for (int r = 0; r < nt; r++)
                    Attention(dev, L, il, rows[r].Layers[il], r, 1, positions[r],
                        pos == IntPtr.Zero ? IntPtr.Zero : (IntPtr)((long)pos + 4L * r));
            else
                Attention(dev, L, il, slot.Layers[il], 0, nt, p0, pos);
            dev.QK.HcPost(Ptr(dev.Xs), Ptr(dev.BlockOut), Ptr(dev.Inject), nt, e, m.Hc, dev.Stream);

            HcPre(dev, L.HcFfnNorm, L.HcFfnDown, L.HcFfnUp, L.HcFfnInject, nt);
            StageEnd(dev, StHc);
            Moe(dev, L, il, nt, pos == IntPtr.Zero && rows == null ? p0 : -1);
            dev.QK.HcPost(Ptr(dev.Xs), Ptr(dev.BlockOut), Ptr(dev.Inject), nt, e, m.Hc, dev.Stream);
            StageEnd(dev, StHc);
        }

        /// <summary>Hand the streams to the next device through pinned host memory. Events are
        /// recorded on their own context's stream; only the waits cross contexts.</summary>
        private void Handoff(Dev src, Dev dst, int nt)
        {
            var bytes = new UIntPtr((ulong)((long)nt * HcDim * 4));
            src.MakeCurrent();
            CudaDriverApi.cuMemcpyDtoHAsync(src.BoundaryPinned, Ptr(src.Xs), bytes, src.Stream).ThrowOnError();
            StageEnd(src, StBoundary);
            CudaDriverApi.cuEventRecord(src.XsReadyEv, src.Stream).ThrowOnError();
            dst.MakeCurrent();
            CudaDriverApi.cuStreamWaitEvent(dst.Stream, src.XsReadyEv, 0).ThrowOnError();
            StageBegin(dst);
            CudaDriverApi.cuMemcpyHtoDAsync(Ptr(dst.Xs), src.BoundaryPinned, bytes, dst.Stream).ThrowOnError();
            CudaDriverApi.cuEventRecord(dst.CopyDoneEv, dst.Stream).ThrowOnError();
            // Nothing may overwrite the source streams or the pinned slice before the copy ran.
            src.MakeCurrent();
            CudaDriverApi.cuStreamWaitEvent(src.Stream, dst.CopyDoneEv, 0).ThrowOnError();
        }

        /// <summary>The low-rank mixer: the streams grouped-RMS-normed into Xn (and the scatter
        /// logits into Inject), the sigmoid gate from the rank-R bottleneck, and the gated stream
        /// mean into Cur.</summary>
        private void HcPre(Dev dev, Tensor normW, in DeviceWeight down, in DeviceWeight up, Tensor injW, int nt,
            Tensor xs = null)
        {
            var m = _m;
            IntPtr xsPtr = Ptr(xs ?? dev.Xs), inj = injW == null ? IntPtr.Zero : Ptr(injW);
            if (nt <= PerSlotMaxRows)
            {
                // Decode size: the norm in two passes over every slice of the streams (one block per
                // row would leave all but one SM idle), each pass quantizing what the next projection
                // reads.
                dev.QK.HcPartials(xsPtr, Ptr(normW), inj, Ptr(dev.HcPartials), nt, m.NEmbd, m.Hc, dev.Stream);
                dev.QK.HcApply(xsPtr, Ptr(normW), Ptr(dev.HcPartials), Ptr(dev.Xn), Ptr(dev.SharedQ81),
                    injW == null ? IntPtr.Zero : Ptr(dev.Inject), nt, m.NEmbd, m.Hc, m.RmsEps, dev.Stream);
                MatMul(dev, down, dev.Xn, Ptr(dev.SharedQ81), dev.Lo, nt);
                dev.QK.SiluQ81(Ptr(dev.Lo), Ptr(dev.SharedQ81), (long)nt * m.HcLowRank, 1.0f / m.Hc, dev.Stream);
                MatMul(dev, up, dev.Lo, Ptr(dev.SharedQ81), dev.G, nt);
            }
            else
            {
                dev.QK.HcNorm(xsPtr, Ptr(normW), inj, Ptr(dev.Xn), Ptr(dev.Inject), nt, m.NEmbd, m.Hc, m.RmsEps, dev.Stream);
                MatMulQ(dev, down, dev.Xn, dev.Lo, nt);
                dev.QK.SiluScale(Ptr(dev.Lo), (long)nt * m.HcLowRank, 1.0f / m.Hc, dev.Stream);
                MatMulQ(dev, up, dev.Lo, dev.G, nt);
            }
            dev.QK.HcCollapse(Ptr(dev.Xn), Ptr(dev.G), Ptr(dev.Cur), nt, m.NEmbd, m.Hc, dev.Stream);
        }

        private void Gdn(Dev dev, DevLayer L, int il, Slot slot, int nt, Slot[] rows)
        {
            var m = _m;
            IntPtr cur = QuantizeShared(dev, dev.Cur, m.NEmbd, nt);
            MatMul(dev, L.Qkv, dev.Cur, cur, dev.Qkv, nt);
            MatMul(dev, L.Gate, dev.Cur, cur, dev.Z, nt);
            MatMulF32(dev, L.SsmAlpha, dev.Cur, dev.Alpha, m.NEmbd, m.GdnVHeads, nt);
            MatMulF32(dev, L.SsmBeta, dev.Cur, dev.Beta, m.NEmbd, m.GdnVHeads, nt);
            StageEnd(dev, StGdnProj);

            // One sequence's nt rows in its own state, or one row of each batched sequence in theirs.
            IntPtr conv, ssm;
            int rowsPerSeq = nt, nSeq = 1;
            if (rows == null)
            {
                conv = StatePtr(slot, dev, il, 0);
                ssm = StatePtr(slot, dev, il, 1);
            }
            else
            {
                conv = BatchStatePtr(dev, il, 0);
                ssm = BatchStatePtr(dev, il, 1);
                rowsPerSeq = 1;
                nSeq = nt;
            }
            dev.QK.GdnPrep(Ptr(dev.Qkv), conv, Ptr(L.ConvW), Ptr(dev.Alpha), Ptr(dev.Beta), Ptr(L.DtBias), Ptr(L.SsmA),
                Ptr(dev.Scr), rowsPerSeq, nSeq, m.GdnKHeads, m.GdnVHeads, m.DConv, m.RmsEps, dev.Stream);
            // After the prep read the previous inputs.
            dev.QK.ConvUpdate(Ptr(dev.Qkv), conv, rowsPerSeq, nSeq, ConvDim, m.DConv - 1, dev.Stream);
            dev.GK.KdaScan(Ptr(dev.Scr), ssm, Ptr(dev.Core), rowsPerSeq, nSeq, m.GdnVHeads, dev.Stream);
            dev.GK.KdaOut(Ptr(dev.Core), Ptr(dev.Z), Ptr(L.SsmNorm), Ptr(dev.GOut), nt, m.GdnVHeads, m.RmsEps, dev.Stream);
            StageEnd(dev, StGdnCore);
            MatMulQ(dev, L.SsmOut, dev.GOut, dev.BlockOut, nt);
            StageEnd(dev, StAttnOut);
        }

        /// <summary>Gated attention for rows [row0, row0 + nt) of the ubatch, one sequence's rows at
        /// p0, p0 + 1, ... against its caches <paramref name="c"/>. A batched decode step calls it
        /// per row; the projections then run for every row on the first call.</summary>
        private void Attention(Dev dev, DevLayer L, int il, SlotLayer c, int row0, int nt, int p0, IntPtr pos = default)
        {
            var m = _m;
            int qd = m.NHead * m.HeadDim, kvd = m.NKvHead * m.HeadDim;
            int allRows = row0 == 0 ? RowsInFlight(nt) : 0;
            if (allRows > 0)
            {
                IntPtr cur = QuantizeShared(dev, dev.Cur, m.NEmbd, allRows);
                MatMul(dev, L.Wq, dev.Cur, cur, dev.Qg, allRows);
                MatMul(dev, L.Wk, dev.Cur, cur, dev.K, allRows);
                MatMul(dev, L.Wv, dev.Cur, cur, dev.V, allRows);
                if (L.Ratio > 0)
                    MatMulF32(dev, L.IdxK, dev.Cur, dev.IdxRaw, m.NEmbd, m.IdxDim, allRows);
                StageEnd(dev, StAttnProj);
            }

            dev.QK.AttnPrep(At(dev.Qg, (long)row0 * 2 * qd), At(dev.K, (long)row0 * kvd), At(dev.V, (long)row0 * kvd),
                Ptr(L.QNorm), Ptr(L.KNorm), At(dev.Q, (long)row0 * qd), At(dev.Gate, (long)row0 * qd),
                Ptr(c.KCache), Ptr(c.VCache), pos, p0, m.NHead, m.NKvHead, m.NRot, m.RopeBase, m.RopeFreqScale,
                m.RmsEps, nt, dev.Stream);

            IntPtr cells = IntPtr.Zero, cellCnt = IntPtr.Zero;
            if (L.Ratio > 0)
            {
                int R = L.Ratio, d = m.IdxDim;
                dev.QK.QsaStore(At(dev.IdxRaw, (long)row0 * d), Ptr(c.IdxRaw), pos, p0, d, nt, dev.Stream);
                // The blocks this ubatch completes (their last member now cached).
                int firstBlock = p0 / R, endBlock = (p0 + nt) / R;
                dev.QK.QsaPool(Ptr(c.IdxRaw), Ptr(L.IdxKNorm), Ptr(c.IdxPooled), pos, firstBlock, endBlock - firstBlock, R, d,
                    m.NRot, m.RopeBase, m.RopeFreqScale, m.RmsEps, dev.Stream);
                // Past the width the selection bites: score the complete blocks, keep the best.
                if (p0 + nt > m.IdxTopK + R - 1)
                {
                    IntPtr qIdx = At(dev.IdxQ, (long)row0 * m.IdxHeads * d);
                    using (Tensor rowsIn = Block(dev.Cur, row0, nt, m.NEmbd))
                    using (Tensor rowsOut = Block(dev.IdxQ, row0, nt, (long)m.IdxHeads * d))
                        MatMulF32(dev, L.IdxQ, rowsIn, rowsOut, m.NEmbd, m.IdxHeads * d, nt);
                    dev.QK.QsaQuery(qIdx, Ptr(L.IdxQNorm), pos, p0, m.IdxHeads, d, m.NRot, m.RopeBase, m.RopeFreqScale,
                        m.RmsEps, nt, dev.Stream);
                    IntPtr scores = At(dev.Scores, (long)row0 * ScoreStride);
                    cells = (IntPtr)((long)Ptr(dev.Cells) + (long)row0 * CellStride * 4);
                    cellCnt = (IntPtr)((long)Ptr(dev.CellCnt) + (long)row0 * 4);
                    // A captured step's grid covers every block the context can hold (its blocks past
                    // the row's position exit at once).
                    dev.QK.QsaScores(qIdx, Ptr(c.IdxPooled), scores, pos, p0, m.IdxHeads, d, R, ScoreStride,
                        pos != IntPtr.Zero ? ScoreStride : (p0 + nt) / R, nt, dev.Stream);
                    dev.QK.QsaSelect(scores, cells, cellCnt, pos, p0, R, m.IdxTopK, ScoreStride, CellStride, nt, dev.Stream);
                    if (pos == IntPtr.Zero)
                        SparseProbe?.Invoke(CaptureSparse(dev, L, il, c, row0, nt, p0));
                }
            }
            StageEnd(dev, StAttnPrep);

            dev.QK.Attention(At(dev.Q, (long)row0 * qd), Ptr(c.KCache), Ptr(c.VCache), At(dev.Gate, (long)row0 * qd),
                cells, cellCnt, At(dev.AttnO, (long)row0 * qd), pos, p0, m.NHead, m.NKvHead, m.AttnScale,
                CellStride, nt, dev.Stream);
            StageEnd(dev, StAttnCore);

            if (row0 + nt == RowsInFlight(nt))
            {
                MatMulQ(dev, L.Wo, dev.AttnO, dev.BlockOut, row0 + nt);
                StageEnd(dev, StAttnOut);
            }
        }

        // The rows of the step in flight: a batched decode step sets it before its layers run.
        private int _rowsInFlight;

        private int RowsInFlight(int nt) => _rowsInFlight > 0 ? _rowsInFlight : nt;

        /// <summary>This step's n-gram rows into the PLE device's pinned staging: the tokens join their
        /// sequences' histories, then the host hashes and dequantizes.</summary>
        private void GatherPleRows(Slot slot, int[] tokens, int tokOff, int nt, int p0, Slot[] rows, int[] positions)
        {
            var histories = new Qwen4ExpPleHistory[nt];
            var pos = new int[nt];
            if (rows == null)
                slot.PleTokens.Put(new ReadOnlySpan<int>(tokens, tokOff, nt), p0);
            for (int t = 0; t < nt; t++)
            {
                if (rows != null)
                    rows[t].PleTokens.Put(new ReadOnlySpan<int>(tokens, tokOff + t, 1), positions[t]);
                histories[t] = rows == null ? slot.PleTokens : rows[t].PleTokens;
                pos[t] = rows == null ? p0 + t : positions[t];
            }
            var dev = _devs[_layers[_pleLayer].Device];
            dev.MakeCurrent();
            // The staging is reused per ubatch: the previous ubatch's upload must have run.
            CudaDriverApi.cuStreamSynchronize(dev.Stream).ThrowOnError();
            _m.Ple.GatherPleRows(histories, new ReadOnlySpan<int>(tokens, tokOff, nt), pos, (float*)dev.PlePinned);
            _devs[0].MakeCurrent();
        }

        private void Ple(Dev dev, DevLayer L, Slot slot, int nt, Slot[] rows)
        {
            var m = _m;
            int e = m.NEmbd, hcE = HcDim;
            CudaDriverApi.cuMemcpyHtoDAsync(Ptr(dev.PleEmb), dev.PlePinned, new UIntPtr((ulong)nt * (ulong)e * 4), dev.Stream).ThrowOnError();

            IntPtr emb = QuantizeShared(dev, dev.PleEmb, e, nt);
            MatMul(dev, L.PleKey, dev.PleEmb, emb, dev.PleKeyOut, nt);
            MatMul(dev, L.PleValue, dev.PleEmb, emb, dev.PleValueOut, nt);
            dev.QK.PleGate(Ptr(dev.PleKeyOut), Ptr(dev.Xs), Ptr(L.PleNormKey), Ptr(L.PleNormQuery), Ptr(dev.PleValueOut),
                Ptr(dev.PleGated), nt, e, m.Hc, m.RmsEps, dev.Stream);
            dev.QK.GroupNorm(Ptr(dev.PleGated), Ptr(L.PleNormConv), Ptr(dev.PleNorm), nt, e, m.Hc, m.RmsEps, dev.Stream);
            IntPtr hists;
            int rowsPerSeq = nt, nSeq = 1;
            if (rows == null)
            {
                hists = StatePtr(slot, dev, m.NLayer, 0);
            }
            else
            {
                hists = BatchStatePtr(dev, m.NLayer, 0);
                rowsPerSeq = 1;
                nSeq = nt;
            }
            dev.QK.PleConv(Ptr(dev.PleGated), Ptr(dev.PleNorm), hists, Ptr(L.PleConvT), Ptr(dev.Xs), rowsPerSeq, nSeq, hcE,
                m.PleConvKernel, m.PleDilation, dev.Stream);
            if (PleHistRows > 0)
                dev.QK.ConvUpdate(Ptr(dev.PleNorm), hists, rowsPerSeq, nSeq, hcE, PleHistRows, dev.Stream);
            StageEnd(dev, StPle);
        }

        /// <param name="probeP0">The position of the ubatch's first row on the single-sequence uncaptured
        /// path, where <see cref="RouterProbe"/> may read the router back; -1 elsewhere.</param>
        private void Moe(Dev dev, DevLayer L, int il, int nt, int probeP0)
        {
            var m = _m;
            int e = m.NEmbd;
            MatMulF32(dev, L.Router, dev.Cur, dev.RouterLogits, e, m.NExpert, nt);
            dev.DK.MoeSelect(dev.RouterLogits, IntPtr.Zero, IntPtr.Zero, null, dev.Moe.Sel, dev.Moe.SelW,
                m.NExpert, m.NExpertUsed, 1, 1.0f, nt, dev.Stream, Dsv4Kernels.RouterSoftmax);
            if (probeP0 >= 0)
                RouterProbe?.Invoke(CaptureRouter(dev, il, probeP0, nt));
            StageEnd(dev, StRouter);

            IntPtr cur = QuantizeShared(dev, dev.Cur, e, nt);
            MatMul(dev, L.GateShexp, dev.Cur, cur, dev.ShGate, nt);
            MatMul(dev, L.UpShexp, dev.Cur, cur, dev.ShUp, nt);
            CudaMoe.SwigluClamp(dev.ShGate, dev.ShUp, (long)nt * m.NFfShexp, 0f);
            MatMulQ(dev, L.DownShexp, dev.ShGate, dev.ShDown, nt);
            dev.QK.SharedGate(Ptr(dev.Cur), Ptr(L.ShexpGate), Ptr(dev.ShDown), nt, e, dev.Stream);
            StageEnd(dev, StShexp);

            CudaMoe.Experts(dev.DK, dev.Alloc.Kernels, dev.Moe, L.GateExps, L.UpExps, L.DownExps,
                dev.Cur, dev.ShDown, dev.BlockOut, nt, m.NExpertUsed, m.NExpert, e, m.NFfExp, 0f,
                PerSlotMaxRows, dev.Stream);
            StageEnd(dev, StExperts);
        }

        /// <summary>The output mixer, then the LM head: the last row's logits, or every row's (a
        /// batched decode step). <paramref name="hOut"/> takes every row's mixed hidden state.</summary>
        private void Head(int nt, float[] logitsOut, bool allRows, float[] hOut = null)
        {
            IssueHead(nt, logitsOut != null, allRows, hOut != null);
            // When the host had queued the step: telemetry's issue/wait split, and TS_Q4E_PERF.
            _issueEnd = System.Diagnostics.Stopwatch.GetTimestamp();
            FinishHead(nt, logitsOut, allRows, hOut);
        }

        /// <summary>The head's kernels and the copies of its results into pinned memory.</summary>
        private void IssueHead(int nt, bool logits, bool allRows, bool hidden)
        {
            var m = _m;
            var dev = _devs[_lastDev];
            dev.MakeCurrent();
            int normRows = allRows || hidden ? nt : 1;
            using (Tensor xs = normRows == nt ? dev.Xs.CopyRef() : Block(dev.Xs, nt - 1, 1, HcDim))
                HcPre(dev, _outHcNorm, _outHcDown, _outHcUp, null, normRows, xs);
            int headRows = allRows ? nt : 1;
            long count = (long)headRows * m.NVocab;
            if (logits)
            {
                Tensor output = allRows ? dev.BatchLogits : dev.Logits;
                using (Tensor headIn = headRows == normRows ? dev.Cur.CopyRef() : Block(dev.Cur, nt - 1, 1, m.NEmbd))
                    MatMulQ(dev, _output, headIn, output, headRows);
                StageEnd(dev, StHead);
                CudaDriverApi.cuMemcpyDtoHAsync(_pinnedLogits, Ptr(output), new UIntPtr((ulong)count * 4), dev.Stream).ThrowOnError();
            }
            long hCount = (long)normRows * m.NEmbd;
            if (hidden)
                CudaDriverApi.cuMemcpyDtoHAsync(_pinnedHidden, Ptr(dev.Cur), new UIntPtr((ulong)hCount * 4), dev.Stream).ThrowOnError();
        }

        /// <summary>Wait for the head and copy its results out of pinned memory.</summary>
        private void FinishHead(int nt, float[] logitsOut, bool allRows, float[] hOut)
        {
            var m = _m;
            var dev = _devs[_lastDev];
            dev.MakeCurrent();
            int normRows = allRows || hOut != null ? nt : 1;
            long count = (long)(allRows ? nt : 1) * m.NVocab;
            long hCount = (long)normRows * m.NEmbd;
            CudaDriverApi.cuStreamSynchronize(dev.Stream).ThrowOnError();
            if (logitsOut != null)
                fixed (float* dst = logitsOut)
                    Buffer.MemoryCopy((void*)_pinnedLogits, dst, logitsOut.LongLength * 4L, count * 4L);
            if (hOut != null)
                fixed (float* dst = hOut)
                    Buffer.MemoryCopy((void*)_pinnedHidden, dst, hOut.LongLength * 4L, hCount * 4L);
            _stages.Report();
        }

        // ---------------------------------------------------------------- test probe

        /// <summary>Test hook: each QSA layer's sparse selection on the single-sequence path, read
        /// back once its selection ran.</summary>
        internal Action<SparseLayerProbe> SparseProbe { get; set; }

        /// <summary>One QSA layer's indexer inputs and selection for a ubatch.</summary>
        internal sealed class SparseLayerProbe
        {
            public int Layer, P0, Nt, Ratio, TopK, IdxHeads, IdxDim, ScoreStride, CellStride, NRot;
            public float RopeBase, RopeFreqScale, Eps;
            /// <summary>The cached raw keys [P0 + Nt, IdxDim], as stored (F16).</summary>
            public System.Half[] RawKeys;
            /// <summary>The block keys [(P0 + Nt) / Ratio, IdxDim] and the normed, rotated queries [Nt, IdxHeads, IdxDim].</summary>
            public float[] Pooled, Queries;
            /// <summary>The block scores [Nt, ScoreStride], the cell lists [Nt, CellStride] and counts [Nt].</summary>
            public float[] Scores;
            public int[] Cells, CellCnt;
        }

        private SparseLayerProbe CaptureSparse(Dev dev, DevLayer L, int il, SlotLayer c, int row0, int nt, int p0)
        {
            var m = _m;
            CudaDriverApi.cuStreamSynchronize(dev.Stream).ThrowOnError();
            int cached = p0 + nt, d = m.IdxDim;
            return new SparseLayerProbe
            {
                Layer = il, P0 = p0, Nt = nt, Ratio = L.Ratio, TopK = m.IdxTopK, IdxHeads = m.IdxHeads, IdxDim = d,
                ScoreStride = ScoreStride, CellStride = CellStride, NRot = m.NRot, RopeBase = m.RopeBase,
                RopeFreqScale = m.RopeFreqScale, Eps = m.RmsEps,
                RawKeys = Download<System.Half>(Ptr(c.IdxRaw), (long)cached * d),
                Pooled = Download<float>(Ptr(c.IdxPooled), (long)(cached / L.Ratio) * d),
                Queries = Download<float>(At(dev.IdxQ, (long)row0 * m.IdxHeads * d), (long)nt * m.IdxHeads * d),
                Scores = Download<float>(At(dev.Scores, (long)row0 * ScoreStride), (long)nt * ScoreStride),
                Cells = Download<int>((IntPtr)((long)Ptr(dev.Cells) + (long)row0 * CellStride * 4), (long)nt * CellStride),
                CellCnt = Download<int>((IntPtr)((long)Ptr(dev.CellCnt) + (long)row0 * 4), nt),
            };
        }

        /// <summary>Test hook: each layer's router logits and chosen experts on the single-sequence path, read
        /// back once the experts were chosen. Decode steps are not captured while it is set.</summary>
        internal Action<RouterLayerProbe> RouterProbe { get; set; }

        /// <summary>One layer's router for a ubatch: the logits [Nt, Experts] and the experts chosen [Nt, Used].</summary>
        internal sealed class RouterLayerProbe
        {
            public int Layer, P0, Nt, Experts, Used;
            public float[] Logits;
            public int[] Selected;
        }

        private RouterLayerProbe CaptureRouter(Dev dev, int il, int p0, int nt)
        {
            var m = _m;
            CudaDriverApi.cuStreamSynchronize(dev.Stream).ThrowOnError();
            return new RouterLayerProbe
            {
                Layer = il, P0 = p0, Nt = nt, Experts = m.NExpert, Used = m.NExpertUsed,
                Logits = Download<float>(Ptr(dev.RouterLogits), (long)nt * m.NExpert),
                Selected = Download<int>(Ptr(dev.Moe.Sel), (long)nt * m.NExpertUsed),
            };
        }

        private static T[] Download<T>(IntPtr src, long count) where T : unmanaged
        {
            var a = new T[count];
            fixed (T* p = a)
                CudaDriverApi.cuMemcpyDtoH((IntPtr)p, src, new UIntPtr((ulong)(count * sizeof(T)))).ThrowOnError();
            return a;
        }

        /// <summary>Tokens up to which the experts and projections run per row: a batched decode
        /// step's rows then compute exactly as each sequence's own step does.</summary>
        public const int PerSlotMaxRows = 16;

        // TS_Q4E_PERF=2 / 3: per-stage time, synchronized host time or GPU time (see CudaStageTimer).
        private const int StEmbed = 0, StHc = 1, StPle = 2, StGdnProj = 3, StGdnCore = 4, StAttnProj = 5, StAttnPrep = 6,
            StAttnCore = 7, StAttnOut = 8, StRouter = 9, StExperts = 10, StShexp = 11, StHead = 12, StBoundary = 13;
        private static readonly string[] StageNames =
        {
            "embed", "hc", "ple", "gdnproj", "gdncore", "attnproj", "attnprep", "attncore", "attnout", "router",
            "experts", "shexp", "head", "boundary",
        };
        // TS_Q4E_PERF>=1: when the host had queued the whole step (the last launch before the
        // logits sync). Close to the step's wall time = the step is bound by launch overhead.
        private long _issueEnd;

        private double IssueMs(long start)
            => _issueEnd > start ? (_issueEnd - start) * 1000.0 / System.Diagnostics.Stopwatch.Frequency : 0;

        private void StageBegin(Dev dev) => (_sampling ? _sampleStages : _stages).SpanStart(dev.Alloc.Context, dev.Stream);

        private void StageEnd(Dev dev, int stage) => (_sampling ? _sampleStages : _stages).End(dev.Alloc.Context, dev.Stream, stage);

        /// <summary>Whether the next forward is a sampled one (TS_Q4E_PERF_SAMPLE); sets
        /// <see cref="_sampling"/> for it.</summary>
        private bool BeginSampledForward()
        {
            if (_perfSample <= 0 || StageObserver == null || ++_forwardsSinceSample < _perfSample)
                return false;
            _forwardsSinceSample = 0;
            _sampling = true;
            return true;
        }

        /// <summary>A sampled forward's stage times to <see cref="StageObserver"/>, after every device
        /// finished it.</summary>
        private void EndSampledForward(int rows, bool decode)
        {
            if (!_sampling)
                return;
            _sampling = false;
            double[] ms = _sampleStages.CollectAndReset(synchronize: true);
            var observer = StageObserver;
            if (observer == null)
                return;
            for (int i = 0; i < ms.Length && i < StageNames.Length; i++)
            {
                if (ms[i] <= 0)
                    continue;
                try { observer(StageNames[i], ms[i] / 1000.0, rows, decode); }
                catch (Exception) { }
            }
        }
    }
}
