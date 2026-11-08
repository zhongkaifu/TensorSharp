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
            // The MTP block is an attention layer on the last device.
            anyAttn |= _mtpLayer != null && dev.Ordinal == _lastDev;

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
            // A captured step's row positions (a decode step's one, a verify window's up to VerifyMaxRows).
            dev.PosDev = AllocT(dev, DType.Int32, VerifyMaxRows);
            if (dev.Ordinal == 0)
            {
                dev.Tokens = AllocT(dev, DType.Int32, nt);
                // PORTABLE: every device may read them.
                CudaDriverApi.cuMemHostAlloc(out _pinnedTokens, new UIntPtr((ulong)nt * 4), 0x1).ThrowOnError();
                CudaDriverApi.cuMemHostAlloc(out _posPinned, new UIntPtr((ulong)VerifyMaxRows * 4), 0x1).ThrowOnError();
            }
            if (dev.Ordinal == _lastDev)
            {
                dev.Logits = AllocF32(dev, 1, m.NVocab);
                dev.BatchLogits = AllocF32(dev, MaxBatchedDecodeRows, m.NVocab);
                CudaDriverApi.cuMemHostAlloc(out _pinnedLogits,
                    new UIntPtr((ulong)MaxBatchedDecodeRows * (ulong)m.NVocab * 4), 0x1).ThrowOnError();
                // Every row's streams of a ubatch (a speculative prompt chunk), a verify window or a batched step.
                CudaDriverApi.cuMemHostAlloc(out _pinnedHidden,
                    new UIntPtr((ulong)Math.Max(nt, Math.Max(MaxBatchedDecodeRows, VerifyMaxRows)) * (ulong)hcE * 4), 0x1).ThrowOnError();
                if (_mtpLayer != null)
                    AllocateMtpScratch(dev);
            }
            if (anyGdn || ple)
            {
                dev.BatchStatePtrs = AllocT(dev, DType.UInt8, BatchStateTableBytes);
                CudaDriverApi.cuMemHostAlloc(out dev.BatchStatePinned, new UIntPtr((ulong)BatchStateTableBytes), 0x1).ThrowOnError();
            }
            CudaDriverApi.cuMemHostAlloc(out dev.BoundaryPinned, new UIntPtr((ulong)((long)nt * hcE * 4)), 0x1).ThrowOnError();
            CudaDriverApi.cuEventCreate(out dev.XsReadyEv, 0x02 /*DISABLE_TIMING*/).ThrowOnError();
            CudaDriverApi.cuEventCreate(out dev.CopyDoneEv, 0x02).ThrowOnError();
            if (_forkShexp)
            {
                CudaDriverApi.cuStreamCreate(out dev.ShStream, 0x1 /*NON_BLOCKING*/).ThrowOnError();
                CudaDriverApi.cuEventCreate(out dev.ShForkEv, 0x02).ThrowOnError();
                CudaDriverApi.cuEventCreate(out dev.ShJoinEv, 0x02).ThrowOnError();
            }
        }

        // TS_Q4E_FORK_SHEXP=0: the shared expert on the device's stream, between the router and the routed experts.
        private readonly bool _forkShexp = Environment.GetEnvironmentVariable("TS_Q4E_FORK_SHEXP") != "0";

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
            var slot = new Slot
            {
                Id = _nextSlotId++,
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
                }
                else
                {
                    AllocKv(slot, dev, c);
                    if (L.Ratio > 0)
                    {
                        c.IdxRaw = SlotTensor(slot, dev, DType.Float16, (long)m.NCtx * m.IdxDim);
                        c.IdxPooled = SlotTensor(slot, dev, DType.Float32, (long)(m.NCtx / L.Ratio) * m.IdxDim);
                    }
                }
                if (L.Ple && PleHistRows > 0)
                {
                    slot.PleHist = SlotTensor(slot, dev, DType.Float32, (long)PleHistRows * HcDim);
                    ptrs[dev.Ordinal][2 * m.NLayer] = (long)Ptr(slot.PleHist);
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

        /// <summary>One attention layer's K/V caches for a slot: F16, or Q8_0 codes and their scales.</summary>
        private void AllocKv(Slot slot, Dev dev, SlotLayer c)
        {
            long kv = (long)_m.NCtx * _m.NKvHead * _m.HeadDim;
            if (!_m.KvQ8)
            {
                c.KCache = SlotTensor(slot, dev, DType.Float16, kv);
                c.VCache = SlotTensor(slot, dev, DType.Float16, kv);
                return;
            }
            c.KCache = SlotTensor(slot, dev, DType.UInt8, kv);
            c.VCache = SlotTensor(slot, dev, DType.UInt8, kv);
            c.KScale = SlotTensor(slot, dev, DType.Float16, kv / Q4eKernels.KvQ8Block);
            c.VScale = SlotTensor(slot, dev, DType.Float16, kv / Q4eKernels.KvQ8Block);
        }

        private static IntPtr PtrOrZero(Tensor t) => t == null ? IntPtr.Zero : Ptr(t);

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
            slot.MtpPast = 0;
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

        public void Forward(int[] tokens, float[] logitsOut) => Forward(tokens, logitsOut, null);

        /// <summary>
        /// A forward that also returns the hidden streams (<see cref="HiddenWidth"/> floats a row, before the output
        /// mixer: what the MTP block reads): every row's when <paramref name="hOut"/> holds them all, else the last
        /// row's.
        /// </summary>
        public void Forward(int[] tokens, float[] logitsOut, float[] hOut)
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
            slot.Failed = true;
            ArmHostExperts();
            bool allHidden = hOut != null && hOut.LongLength >= (long)tokens.Length * HcDim;
            if (tokens.Length == 1 && logitsOut != null && hOut == null && CanCaptureDecode && ForwardDecodeGraphed(slot, tokens[0], slot.NPast, logitsOut))
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
                    ForwardUbatch(slot, tokens, done, nt, slot.NPast, last ? logitsOut : null,
                        hOut: allHidden || last ? hOut : null, hRowOffset: allHidden ? done : -1);
                    slot.NPast += nt;
                    done += nt;
                }
            }
            CheckHostExperts(tokens.Length <= _hostExpertRows ? tokens.Length : 0);
            if (tokens.Length <= _hostExpertRows)
                AdaptResidency();
            slot.Failed = false;
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
        /// <param name="hRowOffset">The row of <paramref name="hOut"/> this ubatch's first row goes to; -1 writes only
        /// the last row, to the start.</param>
        private void ForwardUbatch(Slot slot, int[] tokens, int tokOff, int nt, int p0, float[] logitsOut,
            Slot[] rows = null, int[] positions = null, bool allRows = false, float[] hOut = null, int hRowOffset = 0)
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
            bool ready = false;
            for (int il = 0; il < m.NLayer; il++)
            {
                var L = _layers[il];
                if (L.Device != curDev)
                {
                    ready = false;
                    Handoff(_devs[curDev], _devs[L.Device], nt);
                    curDev = L.Device;
                    StageEnd(_devs[curDev], StBoundary);
                }
                var dev = _devs[curDev];
                dev.MakeCurrent();
                ready = RunLayer(dev, il, slot, nt, p0, rows, positions, IntPtr.Zero, ready);
            }

            if (logitsOut != null || hOut != null)
                Head(nt, logitsOut, rows != null || allRows, hOut, hRowOffset);
        }

        /// <summary>One layer: the PLE block where it sits, then both mixed halves. <paramref name="pos"/>
        /// is the device copy of the row's position a captured decode step reads (zero: p0 is used).</summary>
        /// <param name="partialsReady">The previous layer's last scatter already ran this layer's first mixer pass.</param>
        /// <returns>Whether this layer's last scatter ran the next layer's first mixer pass (the next layer is on this
        /// device and has no PLE block ahead of it).</returns>
        private bool RunLayer(Dev dev, int il, Slot slot, int nt, int p0, Slot[] rows, int[] positions, IntPtr pos,
            bool partialsReady = false)
        {
            var m = _m;
            var L = _layers[il];
            int e = m.NEmbd;
            if (L.Ple)
                Ple(dev, L, slot, nt, rows);

            HcPre(dev, L.HcAttnNorm, L.HcAttnDown, L.HcAttnUp, L.HcAttnInject, nt, partialsReady: partialsReady && !L.Ple);
            StageEnd(dev, StHc);
            if (L.Recurrent)
                Gdn(dev, L, il, slot, nt, rows);
            else if (rows != null)
                for (int r = 0; r < nt; r++)
                    Attention(dev, L, il, rows[r].Layers[il], r, 1, positions[r]);
            else
                Attention(dev, L, il, slot.Layers[il], 0, nt, p0, pos);
            // Decode-size halves meet in one launch: the scatter and the next mixer's first pass.
            bool fuse = nt <= PerSlotMaxRows && dev.QK.HasHcPostPartials && _fuseHc;
            if (fuse)
                dev.QK.HcPostPartials(Ptr(dev.Xs), Ptr(dev.BlockOut), Ptr(dev.Inject), Ptr(L.HcFfnNorm), Ptr(L.HcFfnInject),
                    Ptr(dev.HcPartials), nt, e, m.Hc, dev.Stream);
            else
                dev.QK.HcPost(Ptr(dev.Xs), Ptr(dev.BlockOut), Ptr(dev.Inject), nt, e, m.Hc, dev.Stream);

            HcPre(dev, L.HcFfnNorm, L.HcFfnDown, L.HcFfnUp, L.HcFfnInject, nt, partialsReady: fuse);
            StageEnd(dev, StHc);
            Moe(dev, L, il, nt, pos == IntPtr.Zero && rows == null ? p0 : -1);
            var next = il + 1 < m.NLayer ? _layers[il + 1] : null;
            bool fuseNext = fuse && next != null && next.Device == dev.Ordinal && !next.Ple;
            if (fuseNext)
                dev.QK.HcPostPartials(Ptr(dev.Xs), Ptr(dev.BlockOut), Ptr(dev.Inject), Ptr(next.HcAttnNorm), Ptr(next.HcAttnInject),
                    Ptr(dev.HcPartials), nt, e, m.Hc, dev.Stream);
            else
                dev.QK.HcPost(Ptr(dev.Xs), Ptr(dev.BlockOut), Ptr(dev.Inject), nt, e, m.Hc, dev.Stream);
            StageEnd(dev, StHc);
            return fuseNext;
        }

        // TS_Q4E_FUSE_HC=0: the scatter and the next mixer's first pass as two launches (the fused kernel computes the
        // same values; this is for measuring it).
        private readonly bool _fuseHc = Environment.GetEnvironmentVariable("TS_Q4E_FUSE_HC") != "0";

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
        /// <param name="partialsReady">The first pass already ran, fused into the previous half's scatter
        /// (<see cref="HcPost"/>).</param>
        private void HcPre(Dev dev, Tensor normW, in DeviceWeight down, in DeviceWeight up, Tensor injW, int nt,
            Tensor xs = null, bool partialsReady = false)
        {
            var m = _m;
            IntPtr xsPtr = Ptr(xs ?? dev.Xs), inj = injW == null ? IntPtr.Zero : Ptr(injW);
            if (nt <= PerSlotMaxRows)
            {
                // Decode size: the norm in two passes over every slice of the streams (one block per
                // row would leave all but one SM idle), each pass quantizing what the next projection
                // reads.
                if (!partialsReady)
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
            bool capture = _verifyCapture && rows == null;
            if (capture)
                VerifySaveGdnState(dev, il, slot);
            dev.QK.GdnPrep(Ptr(dev.Qkv), conv, Ptr(L.ConvW), Ptr(dev.Alpha), Ptr(dev.Beta), Ptr(L.DtBias), Ptr(L.SsmA),
                Ptr(dev.Scr), rowsPerSeq, nSeq, m.GdnKHeads, m.GdnVHeads, m.DConv, m.RmsEps, dev.Stream);
            if (capture)
                VerifySaveGdnInputs(dev, il, nt);
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
                m.RmsEps, nt, dev.Stream, PtrOrZero(c.KScale), PtrOrZero(c.VScale));

            IntPtr cells = IntPtr.Zero, cellCnt = IntPtr.Zero;
            if (L.Ratio > 0)
            {
                int R = L.Ratio, d = m.IdxDim;
                dev.QK.QsaStore(At(dev.IdxRaw, (long)row0 * d), Ptr(c.IdxRaw), pos, p0, d, nt, dev.Stream);
                // The blocks this ubatch completes (their last member now cached).
                int firstBlock = p0 / R, endBlock = (p0 + nt) / R;
                if (pos == IntPtr.Zero || nt == 1)
                    dev.QK.QsaPool(Ptr(c.IdxRaw), Ptr(L.IdxKNorm), Ptr(c.IdxPooled), pos, firstBlock, endBlock - firstBlock, R, d,
                        m.NRot, m.RopeBase, m.RopeFreqScale, m.RmsEps, dev.Stream);
                else
                    // A captured window: the pooling kernel reads one row's position, so each row pools the block it
                    // completes, if it completes one.
                    for (int t = 0; t < nt; t++)
                        dev.QK.QsaPool(Ptr(c.IdxRaw), Ptr(L.IdxKNorm), Ptr(c.IdxPooled), (IntPtr)((long)pos + 4L * (row0 + t)), 0, 0,
                            R, d, m.NRot, m.RopeBase, m.RopeFreqScale, m.RmsEps, dev.Stream);
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
                CellStride, nt, dev.Stream, PtrOrZero(c.KScale), PtrOrZero(c.VScale));
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
            {
                if (_verifyCapture && rows == null)
                    VerifySavePle(dev, slot, nt);
                dev.QK.ConvUpdate(Ptr(dev.PleNorm), hists, rowsPerSeq, nSeq, hcE, PleHistRows, dev.Stream);
            }
            StageEnd(dev, StPle);
        }

        /// <param name="probeP0">The position of the ubatch's first row on the single-sequence uncaptured
        /// path, where <see cref="RouterProbe"/> may read the router back; -1 elsewhere.</param>
        private void Moe(Dev dev, DevLayer L, int il, int nt, int probeP0)
        {
            var m = _m;
            int e = m.NEmbd;
            // A decode-size step runs the shared expert beside the router and the routed experts (Strata's forked
            // shared-expert stream): it reads only the mixed input, and its output is first read by the routed experts'
            // combine, which waits for it. A captured step holds both branches.
            bool fork = dev.ShStream != IntPtr.Zero && nt <= PerSlotMaxRows && _perf < 2
                && DecodeKernelFor(dev, L.GateShexp, nt) && DecodeKernelFor(dev, L.UpShexp, nt) && DecodeKernelFor(dev, L.DownShexp, nt);
            if (fork)
            {
                CudaDriverApi.cuEventRecord(dev.ShForkEv, dev.Stream).ThrowOnError();
                CudaDriverApi.cuStreamWaitEvent(dev.ShStream, dev.ShForkEv, 0).ThrowOnError();
                SharedExpert(dev, L, nt, dev.ShStream);
                CudaDriverApi.cuEventRecord(dev.ShJoinEv, dev.ShStream).ThrowOnError();
            }
            if (L.RouterBf16 != null && nt <= Dsv4Kernels.GemvMaxRows)
                dev.DK.GemvBf16(Ptr(dev.Cur), Ptr(L.RouterBf16), Ptr(dev.RouterLogits), e, m.NExpert, nt, dev.Stream);
            else
                MatMulF32(dev, L.Router ?? FloatRouter(dev, L), dev.Cur, dev.RouterLogits, e, m.NExpert, nt);
            dev.DK.MoeSelect(dev.RouterLogits, IntPtr.Zero, IntPtr.Zero, null, dev.Moe.Sel, dev.Moe.SelW,
                m.NExpert, m.NExpertUsed, 1, 1.0f, nt, dev.Stream, Dsv4Kernels.RouterSoftmax);
            if (probeP0 >= 0)
                RouterProbe?.Invoke(CaptureRouter(dev, il, probeP0, nt));
            StageEnd(dev, StRouter);
            bool host = HostExpertsFor(L, nt);
            if (host)
                dev.DK.MoeDoorbellPublish(dev.Cur, dev.Moe.Sel, dev.Moe.SelW, L.Stage.IsHost, nt * e, nt * m.NExpertUsed, il,
                    _hostBox, dev.Stream, _doorbellMiss, _adaptCountsDev, m.NExpert);

            if (!fork)
            {
                SharedExpert(dev, L, nt, dev.Stream);
                StageEnd(dev, StShexp);
            }

            CudaMoe.Experts(dev.DK, dev.Alloc.Kernels, dev.Moe, L.GateExps, L.UpExps, L.DownExps,
                dev.Cur, dev.ShDown, dev.BlockOut, nt, m.NExpertUsed, m.NExpert, e, m.NFfExp, 0f,
                PerSlotMaxRows, dev.Stream, stage: host ? null : _expertStage, stageLayer: L.Stage,
                shDownReady: fork ? dev.ShJoinEv : IntPtr.Zero);
            if (host)
            {
                // The GPU's own experts end here; what follows is the wait for the CPU's share.
                StageEnd(dev, StExperts);
                dev.DK.MoeDoorbellWaitAdd(_hostBox, dev.BlockOut, nt * e, dev.Stream, _doorbellMiss);
                StageEnd(dev, StHostWait);
            }
            else
                StageEnd(dev, StExperts);
        }

        /// <summary>A prompt's router weights as F32 (cuBLAS reads them so) when the layer keeps them as BF16: widened
        /// into the device's scratch, which every layer of the ubatch reuses in turn.</summary>
        private Tensor FloatRouter(Dev dev, DevLayer L)
        {
            long n = (long)_m.NExpert * _m.NEmbd;
            dev.RouterF32 ??= AllocF32(dev, n);
            dev.QK.Bf16ToF32(Ptr(L.RouterBf16), Ptr(dev.RouterF32), n, dev.Stream);
            return dev.RouterF32;
        }

        /// <summary>The shared expert on <paramref name="stream"/>: gate and up from one quantized input, SwiGLU, down, and
        /// the sigmoid gate, into ShDown.</summary>
        private void SharedExpert(Dev dev, DevLayer L, int nt, IntPtr stream)
        {
            var m = _m;
            int e = m.NEmbd;
            IntPtr cur = QuantizeShared(dev, dev.Cur, e, nt, stream);
            MatMul(dev, L.GateShexp, dev.Cur, cur, dev.ShGate, nt, stream);
            MatMul(dev, L.UpShexp, dev.Cur, cur, dev.ShUp, nt, stream);
            dev.Alloc.Kernels.LaunchSiluMulClampF32(Ptr(dev.ShGate), Ptr(dev.ShGate), Ptr(dev.ShUp), (long)nt * m.NFfShexp, 0f, stream);
            MatMul(dev, L.DownShexp, dev.ShGate, QuantizeShared(dev, dev.ShGate, m.NFfShexp, nt, stream), dev.ShDown, nt, stream);
            dev.QK.SharedGate(Ptr(dev.Cur), Ptr(L.ShexpGate), Ptr(dev.ShDown), nt, e, stream);
        }

        /// <summary>The output mixer, then the LM head: the last row's logits, or every row's (a
        /// batched decode step). <paramref name="hOut"/> takes the rows' hidden streams (before the mixer)
        /// from row <paramref name="hRowOffset"/> on, or the last row's alone when that is -1 or they do not fit.</summary>
        private void Head(int nt, float[] logitsOut, bool allRows, float[] hOut = null, int hRowOffset = 0)
        {
            IssueHead(nt, logitsOut != null, allRows, hOut != null);
            if (_perf > 0)
                _issueEnd = System.Diagnostics.Stopwatch.GetTimestamp();
            FinishHead(nt, logitsOut, allRows, hOut, hRowOffset);
        }

        /// <summary>Floats of a row's hidden streams (<see cref="Forward(int[], float[], float[])"/>).</summary>
        public int HiddenWidth => HcDim;

        /// <summary>The head's kernels and the copies of its results into pinned memory.</summary>
        private void IssueHead(int nt, bool logits, bool allRows, bool hidden)
        {
            var m = _m;
            var dev = _devs[_lastDev];
            dev.MakeCurrent();
            if (hidden)
                CudaDriverApi.cuMemcpyDtoHAsync(_pinnedHidden, Ptr(dev.Xs), new UIntPtr((ulong)nt * (ulong)HcDim * 4), dev.Stream).ThrowOnError();
            if (!logits)
                return;
            int normRows = allRows ? nt : 1;
            using (Tensor xs = normRows == nt ? dev.Xs.CopyRef() : Block(dev.Xs, nt - 1, 1, HcDim))
                HcPre(dev, _outHcNorm, _outHcDown, _outHcUp, null, normRows, xs);
            int headRows = allRows ? nt : 1;
            long count = (long)headRows * m.NVocab;
            {
                Tensor output = allRows ? dev.BatchLogits : dev.Logits;
                using (Tensor headIn = headRows == normRows ? dev.Cur.CopyRef() : Block(dev.Cur, nt - 1, 1, m.NEmbd))
                    MatMulQ(dev, _output, headIn, output, headRows);
                StageEnd(dev, StHead);
                CudaDriverApi.cuMemcpyDtoHAsync(_pinnedLogits, Ptr(output), new UIntPtr((ulong)count * 4), dev.Stream).ThrowOnError();
            }
        }

        /// <summary>Wait for the head and copy its results out of pinned memory.</summary>
        private void FinishHead(int nt, float[] logitsOut, bool allRows, float[] hOut, int hRowOffset = 0)
        {
            var m = _m;
            var dev = _devs[_lastDev];
            dev.MakeCurrent();
            long count = (long)(allRows ? nt : 1) * m.NVocab;
            CudaDriverApi.cuStreamSynchronize(dev.Stream).ThrowOnError();
            if (logitsOut != null)
                fixed (float* dst = logitsOut)
                    Buffer.MemoryCopy((void*)_pinnedLogits, dst, logitsOut.LongLength * 4L, count * 4L);
            if (hOut != null)
            {
                long row = HcDim, all = nt * row;
                fixed (float* dst = hOut)
                {
                    if (hRowOffset >= 0 && hOut.LongLength >= (hRowOffset + nt) * row)
                        Buffer.MemoryCopy((void*)_pinnedHidden, dst + hRowOffset * row, (hOut.LongLength - hRowOffset * row) * 4L, all * 4L);
                    else
                        Buffer.MemoryCopy((float*)_pinnedHidden + all - row, dst, hOut.LongLength * 4L, row * 4L);
                }
            }
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
            StAttnCore = 7, StAttnOut = 8, StRouter = 9, StExperts = 10, StShexp = 11, StHead = 12, StBoundary = 13, StHostWait = 14;
        private static readonly string[] StageNames =
        {
            "embed", "hc", "ple", "gdnproj", "gdncore", "attnproj", "attnprep", "attncore", "attnout", "router",
            "experts", "shexp", "head", "boundary", "hostwait",
        };
        // TS_Q4E_PERF>=1: when the host had queued the whole step (the last launch before the
        // logits sync). Close to the step's wall time = the step is bound by launch overhead.
        private long _issueEnd;

        private double IssueMs(long start)
            => _issueEnd > start ? (_issueEnd - start) * 1000.0 / System.Diagnostics.Stopwatch.Frequency : 0;

        private void StageBegin(Dev dev) => _stages.SpanStart(dev.Alloc.Context, dev.Stream);

        private void StageEnd(Dev dev, int stage) => _stages.End(dev.Alloc.Context, dev.Stream, stage);
    }
}
