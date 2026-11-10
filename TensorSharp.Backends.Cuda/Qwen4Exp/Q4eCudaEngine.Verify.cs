// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Speculative verify on the direct CUDA engine: a window of up to VerifyMaxRows tokens runs as one ubatch (every
// row's logits and hidden state), and the accepted prefix is committed by replay, as the ggml_cuda Qwen 3.5 verify
// does. The attention KV and QSA caches are indexed by position, so rejected rows are simply overwritten later. The
// recurrent states are updated in place by the verify, so it first copies each Gated DeltaNet layer's conv and
// recurrent state, and keeps what each recurrence consumed (the conv inputs and the scan inputs); a commit puts the
// states back and steps the recurrence over the accepted rows only. That is the arithmetic those rows' own forward
// would run, row by row, so the committed state is the one a plain decode of the accepted tokens leaves. The PLE
// history is handled the same way. A fully accepted window keeps the verify's state.
using System;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    public sealed unsafe partial class Q4eCudaEngine
    {
        /// <summary>The widest verify window (a seven-token draft and the token before it).</summary>
        public const int VerifyMaxRows = 8;

        private sealed class VerifySave
        {
            public Tensor Conv, Ssm, Qkv, Scr;
        }

        private VerifySave[] _vsave;
        private Tensor _vPleHist, _vPleNorm;
        private bool _verifyCapture;
        private Slot _verifySlot;
        private int _verifyRows, _verifyP0 = -1;

        /// <summary>Device bytes the verify saves take (reserved by the residency plan, allocated on first use).</summary>
        private long VerifySaveBytes()
        {
            var m = _m;
            long perGdn = ((long)(m.DConv - 1) * ConvDim + (long)m.GdnVHeads * Q4eKernels.GdnHeadDim * Q4eKernels.GdnHeadDim
                + (long)VerifyMaxRows * ConvDim + (long)VerifyMaxRows * m.GdnVHeads * GlmKernels.KdaScratch) * 4;
            long b = 0;
            for (int il = 0; il < m.NLayer; il++)
                if (m.Layers[il].Recurrent)
                    b += CudaWeightArena.Align(perGdn) + 4 * 256;
            if (_pleLayer >= 0)
                b += ((long)PleHistRows * HcDim + (long)VerifyMaxRows * HcDim) * 4 + 512;
            return b;
        }

        private void EnsureVerifySaves()
        {
            if (_vsave != null)
                return;
            var m = _m;
            var saves = new VerifySave[m.NLayer];
            for (int il = 0; il < m.NLayer; il++)
            {
                if (!m.Layers[il].Recurrent)
                    continue;
                var dev = _devs[_layers[il].Device];
                dev.MakeCurrent();
                saves[il] = new VerifySave
                {
                    Conv = AllocF32(dev, (long)(m.DConv - 1) * ConvDim),
                    Ssm = AllocF32(dev, (long)m.GdnVHeads * Q4eKernels.GdnHeadDim * Q4eKernels.GdnHeadDim),
                    Qkv = AllocF32(dev, VerifyMaxRows, ConvDim),
                    Scr = AllocF32(dev, (long)VerifyMaxRows * m.GdnVHeads * GlmKernels.KdaScratch),
                };
            }
            if (_pleLayer >= 0 && PleHistRows > 0)
            {
                var dev = _devs[_layers[_pleLayer].Device];
                dev.MakeCurrent();
                _vPleHist = AllocF32(dev, (long)PleHistRows * HcDim);
                _vPleNorm = AllocF32(dev, VerifyMaxRows, HcDim);
            }
            _vsave = saves;
        }

        private static void CopyD2D(Dev dev, Tensor dst, Tensor src, long bytes)
            => CudaDriverApi.cuMemcpyDtoDAsync(Ptr(dst), Ptr(src), new UIntPtr((ulong)bytes), dev.Stream).ThrowOnError();

        /// <summary>In the verify: a GDN layer's states before the window, read before its prep and update.</summary>
        private void VerifySaveGdnState(Dev dev, int il, Slot slot)
        {
            var c = slot.Layers[il];
            var s = _vsave[il];
            CopyD2D(dev, s.Conv, c.ConvState, c.ConvState.ElementCount() * 4);
            CopyD2D(dev, s.Ssm, c.Ssm, c.Ssm.ElementCount() * 4);
        }

        /// <summary>In the verify: what the layer's recurrence consumed, once its prep has run.</summary>
        private void VerifySaveGdnInputs(Dev dev, int il, int nt)
        {
            var s = _vsave[il];
            CopyD2D(dev, s.Qkv, dev.Qkv, (long)nt * ConvDim * 4);
            CopyD2D(dev, s.Scr, dev.Scr, (long)nt * _m.GdnVHeads * GlmKernels.KdaScratch * 4);
        }

        /// <summary>In the verify: the PLE history before the window and the rows it appends.</summary>
        private void VerifySavePle(Dev dev, Slot slot, int nt)
        {
            if (_vPleHist == null)
                return;
            CopyD2D(dev, _vPleHist, slot.PleHist, _vPleHist.ElementCount() * 4);
            CopyD2D(dev, _vPleNorm, dev.PleNorm, (long)nt * HcDim * 4);
        }

        /// <summary>Whether the engine can verify a window (one device's states; not under a batched step).</summary>
        public bool SupportsVerify => _devs.Length >= 1;

        /// <summary>
        /// A verify window: <paramref name="tokens"/> (at most <see cref="VerifyMaxRows"/>) at the active slot's head, every
        /// row's logits into <paramref name="logitsAll"/> (rows x vocab) and, when given, every row's mixed hidden state into
        /// <paramref name="hAll"/>. The head advances over every row; <see cref="CommitVerify"/> then keeps a prefix.
        /// </summary>
        public void ForwardVerify(int[] tokens, float[] logitsAll, float[] hAll) => ForwardVerify(tokens, logitsAll, hAll, null);

        /// <summary>Whether <see cref="ForwardVerifyGreedy"/> computes the rows' tokens on the device.</summary>
        public bool HasDeviceGreedyVerify => _devs[_lastDev].QK.HasArgmaxRows && CanCaptureDecode;

        /// <summary>
        /// A verify window that returns each row's first-max token in <paramref name="drawnOut"/> instead of its logits
        /// (a plain greedy sampler's draw), the argmax computed on the device so the window's logits never cross the bus.
        /// Otherwise as <see cref="ForwardVerify(int[], float[], float[])"/>; without a captured window the logits come back
        /// and the host takes the same first maximum.
        /// </summary>
        public void ForwardVerifyGreedy(int[] tokens, int[] drawnOut, float[] hAll)
        {
            DrainDecodeLanes();
            ArgumentNullException.ThrowIfNull(drawnOut);
            if (drawnOut.Length < tokens.Length)
                throw new ArgumentException("a greedy verify needs one output token per row", nameof(drawnOut));
            if (HasDeviceGreedyVerify)
            {
                ForwardVerify(tokens, null, hAll, drawnOut);
                return;
            }
            int vocab = _m.NVocab;
            var all = new float[(long)tokens.Length * vocab];
            ForwardVerify(tokens, all, hAll, null);
            for (int r = 0; r < tokens.Length; r++)
            {
                int best = 0;
                float bestV = float.NegativeInfinity;
                long o = (long)r * vocab;
                for (int i = 0; i < vocab; i++)
                    if (all[o + i] > bestV) { bestV = all[o + i]; best = i; }
                drawnOut[r] = best;
            }
        }

        private void ForwardVerify(int[] tokens, float[] logitsAll, float[] hAll, int[] idsOut)
        {
            if (tokens == null || tokens.Length == 0 || tokens.Length > VerifyMaxRows)
                throw new ArgumentException($"a verify window takes 1 to {VerifyMaxRows} tokens", nameof(tokens));
            Slot slot = _active;
            if (slot.Failed)
                throw new InvalidOperationException("[q4e-cuda] the sequence's last forward failed; reset it before reuse");
            int nt = tokens.Length, p0 = slot.NPast;
            if (p0 + nt > _m.NCtx)
                throw new InvalidOperationException($"[q4e-cuda] context overflow: n_past={p0} + {nt} > n_ctx={_m.NCtx}");
            // The window's rows land in the slot's caches: grow them first (they start at TS_Q4E_SLOT_TOKENS).
            EnsureCapacity(slot, p0 + nt);
            EnsureVerifySaves();
            long start = System.Diagnostics.Stopwatch.GetTimestamp();
            long missedBefore = _perf > 0 ? HostExpertMisses : 0, tailBefore = _perf > 0 ? TailSkipped : 0;
            slot.Failed = true;
            ArmHostExperts();
            _verifyCapture = true;
            _verifySlot = slot;
            _verifyRows = nt;
            _verifyP0 = p0;
            try
            {
                // Captured per (sequence, regime, rows), as decode steps are; uncaptured when graphs are off.
                if (!(CanCaptureDecode && (logitsAll != null || idsOut != null) && ForwardVerifyGraphed(slot, tokens, p0, logitsAll, hAll, idsOut)))
                {
                    if (idsOut != null)
                        throw new InvalidOperationException("[q4e-cuda] a greedy verify needs a captured window");
                    ForwardUbatch(slot, tokens, 0, nt, p0, logitsAll, allRows: true, hOut: hAll);
                }
            }
            finally
            {
                _verifyCapture = false;
            }
            CheckHostExperts(nt);
            slot.NPast = p0 + nt;
            slot.Failed = false;
            AdaptResidency();
            if (_perf > 0)
            {
                double ms = System.Diagnostics.Stopwatch.GetElapsedTime(start).TotalMilliseconds, issue = IssueMs(start);
                Console.Error.WriteLine($"[q4e-cuda] verify {nt} rows in {ms:F2} ms, "
                    + $"host issue {issue:F2} ms, {HostExpertMisses - missedBefore} experts on the CPU"
                    + (_tailRank > 0 ? $", {TailSkipped - tailBefore} tail-skipped" : "") + KvFetchedNote(slot, 32));
            }
        }

        /// <summary>
        /// Keep the first <paramref name="keep"/> rows of the last verify window (1 to its width): the recurrent states and
        /// the PLE history are put back and stepped over those rows, and the head moves to the window's start + keep.
        /// False when there is no window to commit.
        /// </summary>
        public bool CommitVerify(int keep)
        {
            DrainDecodeLanes();
            Slot slot = _verifySlot;
            int nt = _verifyRows, p0 = _verifyP0;
            if (p0 < 0 || slot == null || keep < 1 || keep > nt)
                return false;
            _verifyP0 = -1;
            if (keep == nt)
                return true;
            var m = _m;
            int scrRow = GlmKernels.KdaScratch * 4;
            for (int il = 0; il < m.NLayer; il++)
            {
                if (!m.Layers[il].Recurrent)
                    continue;
                var dev = _devs[_layers[il].Device];
                dev.MakeCurrent();
                var c = slot.Layers[il];
                var s = _vsave[il];
                CopyD2D(dev, c.ConvState, s.Conv, c.ConvState.ElementCount() * 4);
                CopyD2D(dev, c.Ssm, s.Ssm, c.Ssm.ElementCount() * 4);
                dev.QK.ConvUpdate(Ptr(s.Qkv), StatePtr(slot, dev, il, 0), keep, 1, ConvDim, m.DConv - 1, dev.Stream);
                // The scan inputs are [heads][rows][scratch]: the kept rows of each head, packed for a keep-row scan.
                var copy = new CudaDriverApi.CudaMemcpy2D
                {
                    SrcMemoryType = CudaDriverApi.CU_MEMORYTYPE_DEVICE, SrcDevice = Ptr(s.Scr), SrcPitch = new UIntPtr((ulong)(nt * scrRow)),
                    DstMemoryType = CudaDriverApi.CU_MEMORYTYPE_DEVICE, DstDevice = Ptr(dev.Scr), DstPitch = new UIntPtr((ulong)(keep * scrRow)),
                    WidthInBytes = new UIntPtr((ulong)(keep * scrRow)), Height = new UIntPtr((ulong)m.GdnVHeads),
                };
                CudaDriverApi.cuMemcpy2DAsync(copy, dev.Stream).ThrowOnError();
                dev.GK.KdaScan(Ptr(dev.Scr), StatePtr(slot, dev, il, 1), Ptr(dev.Core), keep, 1, m.GdnVHeads, dev.Stream);
            }
            if (_vPleHist != null && _pleLayer >= 0)
            {
                var dev = _devs[_layers[_pleLayer].Device];
                dev.MakeCurrent();
                CopyD2D(dev, slot.PleHist, _vPleHist, _vPleHist.ElementCount() * 4);
                dev.QK.ConvUpdate(Ptr(_vPleNorm), StatePtr(slot, dev, m.NLayer, 0), keep, 1, HcDim, PleHistRows, dev.Stream);
            }
            slot.PleTokens.Length = Math.Min(slot.PleTokens.Length, p0 + keep);
            slot.NPast = p0 + keep;
            SlotPositionsDropped(slot, slot.NPast);
            foreach (var dev in _devs)
            {
                dev.MakeCurrent();
                CudaDriverApi.cuStreamSynchronize(dev.Stream).ThrowOnError();
            }
            return true;
        }

        /// <summary>Put the states back to before the last verify window (its rows all rejected); the head returns to its start.</summary>
        public bool RestoreVerify()
        {
            DrainDecodeLanes();
            Slot slot = _verifySlot;
            int p0 = _verifyP0;
            if (p0 < 0 || slot == null)
                return false;
            _verifyP0 = -1;
            var m = _m;
            for (int il = 0; il < m.NLayer; il++)
            {
                if (!m.Layers[il].Recurrent)
                    continue;
                var dev = _devs[_layers[il].Device];
                dev.MakeCurrent();
                CopyD2D(dev, slot.Layers[il].ConvState, _vsave[il].Conv, slot.Layers[il].ConvState.ElementCount() * 4);
                CopyD2D(dev, slot.Layers[il].Ssm, _vsave[il].Ssm, slot.Layers[il].Ssm.ElementCount() * 4);
            }
            if (_vPleHist != null && _pleLayer >= 0)
            {
                var dev = _devs[_layers[_pleLayer].Device];
                dev.MakeCurrent();
                CopyD2D(dev, slot.PleHist, _vPleHist, _vPleHist.ElementCount() * 4);
            }
            slot.PleTokens.Length = Math.Min(slot.PleTokens.Length, p0);
            slot.NPast = p0;
            SlotPositionsDropped(slot, p0);
            foreach (var dev in _devs)
            {
                dev.MakeCurrent();
                CudaDriverApi.cuStreamSynchronize(dev.Stream).ThrowOnError();
            }
            return true;
        }
    }
}
