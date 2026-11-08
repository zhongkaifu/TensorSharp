// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// The MTP (NextN) draft block on the direct CUDA engine, from a separate shared-target GGUF. Row t of a call is the
// token at position p0 + t with the trunk's hidden streams of the token before it:
//   join:  each stream s = eh_proj([rms(e) * enorm, rms(h_s) * hnorm_s])   (2E -> E per stream)
//   layer: the trunk's gated attention (dense; the block's own KV) and routed MoE, through its hyper-connections
//   head:  the block's output mixer, then the trunk's LM head (the last row)
// and the call returns the last row's streams, which the next draft step joins with its token. Every kernel is the
// trunk's: the embedding writes the token row into all four streams, so the per-stream norm with enorm repeated per
// stream is rms(e) * enorm, and one strided copy per half lays the join rows out for eh_proj (rows = tokens x streams).
// The reference is ggml_ops_qwen4exp_mtp.cpp.
using System;
using System.Runtime.InteropServices;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    public sealed unsafe partial class Q4eCudaEngine
    {
        private DevLayer _mtpLayer;
        private Tensor _mtpEnorm, _mtpHnorm, _mtpHeadNorm;
        private DeviceWeight _mtpEh, _mtpHeadDown, _mtpHeadUp;
        /// <summary>The join rows [ubatch * hc, 2E], on the last device.</summary>
        private Tensor _mtpJoin;
        /// <summary>A call's trunk streams staged for upload [ubatch, hc * E].</summary>
        private IntPtr _pinnedMtpH;

        /// <summary>Whether an MTP draft block is loaded.</summary>
        public bool HasMtp => _mtpLayer != null;

        private static System.Collections.Generic.IEnumerable<CudaWeightDesc> MtpWeights(MtpDesc mtp, bool experts)
        {
            foreach (var w in LayerWeights(mtp.Layer, experts))
                yield return w;
            foreach (var w in new[] { mtp.EhProj, mtp.HeadHcDown, mtp.HeadHcUp })
                if (w.IsValid)
                    yield return w;
        }

        /// <summary>The block's weights in the last device's arena.</summary>
        private long MtpArenaBytes()
        {
            if (_m.Mtp == null)
                return 0;
            long b = 0;
            foreach (var w in MtpWeights(_m.Mtp, experts: true))
                b += CudaWeightArena.Align(w.TotalBytes);
            return b;
        }

        /// <summary>Everything the block holds on its device: weights, F32 tensors, one sequence's KV and the join scratch.</summary>
        private long MtpDeviceBytes()
        {
            var m = _m;
            if (m.Mtp == null)
                return 0;
            long floats = LayerFloatBytes(m.Mtp.Layer)
                + CudaWeightArena.Align((long)m.Hc * m.NEmbd * 4) * 3;
            long kv = 2 * KvCacheBytes();
            long join = (long)m.NUbatch * m.Hc * 2 * m.NEmbd * 4;
            return MtpArenaBytes() + floats + kv + join;
        }

        private void UploadMtp(Dev dev)
        {
            var mtp = _m.Mtp;
            UploadLayer(dev, mtp.Layer, _mtpLayer, -1);
            // The per-stream norm takes [hc * E] weights: enorm repeated once per stream.
            var enorm = new float[_m.Hc * _m.NEmbd];
            for (int s = 0; s < _m.Hc; s++)
                Array.Copy(mtp.Enorm, 0, enorm, s * _m.NEmbd, _m.NEmbd);
            _mtpEnorm = UploadF32(dev, enorm);
            _mtpHnorm = UploadF32(dev, mtp.Hnorm);
            _mtpHeadNorm = UploadF32(dev, mtp.HeadHcNorm);
            _mtpEh = dev.Weights.Place(mtp.EhProj);
            _mtpHeadDown = dev.Weights.Place(mtp.HeadHcDown);
            _mtpHeadUp = dev.Weights.Place(mtp.HeadHcUp);
        }

        private void AllocateMtpScratch(Dev dev)
        {
            var m = _m;
            _mtpJoin = AllocF32(dev, (long)m.NUbatch * m.Hc, 2L * m.NEmbd);
            CudaDriverApi.cuMemHostAlloc(out _pinnedMtpH, new UIntPtr((ulong)m.NUbatch * (ulong)HcDim * 4), 0x1).ThrowOnError();
        }

        private void FreeMtpPinned()
        {
            if (_pinnedMtpH != IntPtr.Zero)
                CudaDriverApi.cuMemFreeHost(_pinnedMtpH);
            _pinnedMtpH = IntPtr.Zero;
        }

        /// <summary>The block's KV for a slot, allocated at its first draft.</summary>
        private void EnsureMtpCache(Slot slot)
        {
            if (slot.Mtp != null)
                return;
            var dev = _devs[_mtpLayer.Device];
            var c = new SlotLayer();
            AllocKv(slot, dev, c);
            slot.Mtp = c;
            slot.MtpPast = 0;
        }

        /// <summary>
        /// Run the MTP block over <paramref name="tokens"/> at positions <paramref name="p0"/>, ... of the active slot,
        /// row t with the trunk streams <paramref name="hPrev"/>[t] (<see cref="HiddenWidth"/> floats each) of the token
        /// before it. Fills the last row's draft logits and its streams when asked. The block keeps its own KV: a call
        /// may start at any position it already holds (rows past it are drafts the next calls rewrite), not past it.
        /// </summary>
        public void MtpForward(int[] tokens, float[] hPrev, int p0, float[] logitsOut, float[] hOut)
        {
            if (_mtpLayer == null)
                throw new InvalidOperationException("[q4e-cuda] no MTP block is loaded");
            ArgumentNullException.ThrowIfNull(tokens);
            ArgumentNullException.ThrowIfNull(hPrev);
            int n = tokens.Length;
            if (n == 0 || hPrev.LongLength < (long)n * HcDim || (logitsOut != null && logitsOut.Length < _m.NVocab)
                || (hOut != null && hOut.Length < HcDim))
                throw new ArgumentException("[q4e-cuda] the MTP inputs or outputs do not cover the requested rows");
            Slot slot = _active;
            if (slot.Failed)
                throw new InvalidOperationException("[q4e-cuda] the sequence's last forward failed; reset it before reuse");
            EnsureMtpCache(slot);
            if (p0 < 0 || p0 > slot.MtpPast)
                throw new InvalidOperationException($"[q4e-cuda] the MTP block holds {slot.MtpPast} positions and cannot draft from {p0}");
            if (p0 + n > _m.NCtx)
                throw new InvalidOperationException($"[q4e-cuda] MTP context overflow: {p0} + {n} > n_ctx={_m.NCtx}");
            slot.MtpPast = Math.Min(slot.MtpPast, p0);
            long start = System.Diagnostics.Stopwatch.GetTimestamp();
            int done = 0;
            while (done < n)
            {
                int nt = Math.Min(_m.NUbatch, n - done);
                bool last = done + nt == n;
                MtpUbatch(slot, tokens, hPrev, done, nt, p0 + done, last ? logitsOut : null, last ? hOut : null);
                done += nt;
            }
            slot.MtpPast = p0 + n;
            if (_perf > 0)
                Console.Error.WriteLine($"[q4e-cuda] mtp {n} rows in {System.Diagnostics.Stopwatch.GetElapsedTime(start).TotalMilliseconds:F2} ms");
        }

        private void MtpUbatch(Slot slot, int[] tokens, float[] hPrev, int off, int nt, int p0, float[] logitsOut, float[] hOut)
        {
            var m = _m;
            int e = m.NEmbd, hc = m.Hc;
            var dev0 = _devs[0];
            var dev = _devs[_mtpLayer.Device];

            // The token rows, embedded into all four streams on the embedding's device.
            dev0.MakeCurrent();
            CudaDriverApi.cuStreamSynchronize(dev0.Stream).ThrowOnError();
            Marshal.Copy(tokens, off, _pinnedTokens, nt);
            CudaDriverApi.cuMemcpyHtoDAsync(Ptr(dev0.Tokens), _pinnedTokens, new UIntPtr((ulong)nt * 4), dev0.Stream).ThrowOnError();
            dev0.DK.Embed(_tokEmbd.Ptr, dev0.Tokens, dev0.Xs, _tokEmbd.Type, _tokEmbd.RowBytes, nt, e, dev0.Stream);
            if (dev != dev0)
                Handoff(dev0, dev, nt);

            // The trunk streams of the previous tokens.
            dev.MakeCurrent();
            CudaDriverApi.cuStreamSynchronize(dev.Stream).ThrowOnError();
            Marshal.Copy(hPrev, off * HcDim, _pinnedMtpH, nt * HcDim);
            CudaDriverApi.cuMemcpyHtoDAsync(Ptr(dev.G), _pinnedMtpH, new UIntPtr((ulong)nt * (ulong)HcDim * 4), dev.Stream).ThrowOnError();

            // Join: per stream [rms(e) * enorm, rms(h_s) * hnorm_s], then eh_proj over tokens x streams rows.
            dev.QK.GroupNorm(Ptr(dev.Xs), Ptr(_mtpEnorm), Ptr(dev.Xn), nt, e, hc, m.RmsEps, dev.Stream);
            dev.QK.GroupNorm(Ptr(dev.G), Ptr(_mtpHnorm), Ptr(dev.Xs), nt, e, hc, m.RmsEps, dev.Stream);
            int rows = nt * hc;
            JoinHalf(dev, dev.Xn, 0, rows);
            JoinHalf(dev, dev.Xs, e, rows);
            MatMul(dev, _mtpEh, _mtpJoin, QuantizeShared(dev, _mtpJoin, 2 * e, rows), dev.Xs, rows);

            // The block's layer, on its own KV.
            var L = _mtpLayer;
            HcPre(dev, L.HcAttnNorm, L.HcAttnDown, L.HcAttnUp, L.HcAttnInject, nt);
            Attention(dev, L, -1, slot.Mtp, 0, nt, p0);
            dev.QK.HcPost(Ptr(dev.Xs), Ptr(dev.BlockOut), Ptr(dev.Inject), nt, e, hc, dev.Stream);
            HcPre(dev, L.HcFfnNorm, L.HcFfnDown, L.HcFfnUp, L.HcFfnInject, nt);
            Moe(dev, L, -1, nt, -1);
            dev.QK.HcPost(Ptr(dev.Xs), Ptr(dev.BlockOut), Ptr(dev.Inject), nt, e, hc, dev.Stream);

            if (hOut != null)
                CudaDriverApi.cuMemcpyDtoHAsync(_pinnedHidden, At(dev.Xs, (long)(nt - 1) * HcDim),
                    new UIntPtr((ulong)HcDim * 4), dev.Stream).ThrowOnError();
            if (logitsOut != null)
            {
                using (Tensor xs = Block(dev.Xs, nt - 1, 1, HcDim))
                    HcPre(dev, _mtpHeadNorm, _mtpHeadDown, _mtpHeadUp, null, 1, xs);
                MatMulQ(dev, _output, dev.Cur, dev.Logits, 1);
                CudaDriverApi.cuMemcpyDtoHAsync(_pinnedLogits, Ptr(dev.Logits), new UIntPtr((ulong)m.NVocab * 4), dev.Stream).ThrowOnError();
            }
            CudaDriverApi.cuStreamSynchronize(dev.Stream).ThrowOnError();
            if (logitsOut != null)
                fixed (float* dst = logitsOut)
                    Buffer.MemoryCopy((void*)_pinnedLogits, dst, logitsOut.LongLength * 4L, (long)m.NVocab * 4L);
            if (hOut != null)
                fixed (float* dst = hOut)
                    Buffer.MemoryCopy((void*)_pinnedHidden, dst, hOut.LongLength * 4L, (long)HcDim * 4L);
        }

        /// <summary>One half of the join rows: <paramref name="src"/>'s [rows, E] into columns [col, col + E) of the
        /// [rows, 2E] join.</summary>
        private void JoinHalf(Dev dev, Tensor src, int col, int rows)
        {
            long rowBytes = (long)_m.NEmbd * 4;
            var copy = new CudaDriverApi.CudaMemcpy2D
            {
                SrcMemoryType = CudaDriverApi.CU_MEMORYTYPE_DEVICE, SrcDevice = Ptr(src), SrcPitch = new UIntPtr((ulong)rowBytes),
                DstMemoryType = CudaDriverApi.CU_MEMORYTYPE_DEVICE, DstDevice = At(_mtpJoin, col), DstPitch = new UIntPtr((ulong)(2 * rowBytes)),
                WidthInBytes = new UIntPtr((ulong)rowBytes), Height = new UIntPtr((ulong)rows),
            };
            CudaDriverApi.cuMemcpy2DAsync(copy, dev.Stream).ThrowOnError();
        }
    }
}
