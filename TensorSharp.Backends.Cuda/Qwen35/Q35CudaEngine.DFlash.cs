// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// ---------------------------------------------------------------------------
// The DFlash2 block drafter on the direct-CUDA engine (kernels: tensorsharp_q35_dflash.cuh; the
// math is the graph ggml_ops_dflash.cpp builds, see DFlashConfig.cs for the three passes).
//
// The drafter's state lives in each slot: a KV ring per draft layer, F16 [nKv][ringRows][128],
// and the position each ring row holds (-1: none). It tracks the slot exactly, with no host
// round trip:
//   * taps:   at each tapped target layer, the residual rows go through that layer's column
//             slice of fc (Q4_K, 5120 of its 25600 inputs) and accumulate into one 5120-wide
//             pre-norm row per token - fc @ concat(taps) without ever forming the concatenation;
//   * inject: every row a slot commits (a prompt's last ringRows rows, a decode step's row, the
//             kept rows of a verify) is encoded and written to the ring at its position;
//   * draft:  [anchor, MASK x (b - 1)] at the slot's position over the ring.
// ---------------------------------------------------------------------------
using System;
using System.Collections.Generic;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    public sealed unsafe partial class Q35CudaEngine
    {
        private const int TQ4_K = 12;

        public sealed class DraftLayerDesc
        {
            public CudaWeightDesc Q, K, V, O, Gate, Up, Down, AttnConvProj, FfnConvProj;
            public float[] AttnNorm, FfnNorm, QNorm, KNorm, AttnConvBase, FfnConvBase;
        }

        /// <summary>A DFlash2 drafter (dflash.* keys of its GGUF). It borrows the target's
        /// embedding and LM head.</summary>
        public sealed class DraftDesc
        {
            public int NEmbd, NFf, NHead, NKvHead, HeadDim, BlockSize, MaskToken, SlidingWindow;
            public int ConvTaps, ConvGroup, SelRank, SelTopK;
            public float RopeBase, Eps, LogitScale = 1f;
            /// <summary>0-based target layers whose input residual feeds fc, in its column order.</summary>
            public int[] TargetLayers;
            public CudaWeightDesc Fc, SelHidden, SelPred, SelSucc;
            public float[] EncNorm, OutNorm;
            public DraftLayerDesc[] Layers;
        }

        private struct QMat
        {
            public IntPtr W;
            public long RowBytes;
            public int Rows, K, Type;
        }

        private sealed class DraftLayer
        {
            public QMat Q, K, V, O, GateUp, Down, AttnConv, FfnConv;
            public IntPtr AttnNorm, FfnNorm, QNorm, KNorm, AttnConvBase, FfnConvBase;
        }

        private DraftDesc _d;
        private DraftLayer[] _dl;
        private QMat _dFc, _dSelHidden, _dSelPred, _dSelSucc;
        private IntPtr _dEncNorm, _dOutNorm;
        private int _ringRows, _dTapCount;
        private int[] _dTapOf;                   // target layer -> fc column block, or -1

        // Drafter scratch: tap accumulators [NUbatch][e]; int8 activations; per-row metadata;
        // the draft pass's rows [MaxSeqs * b] ...
        private IntPtr _dAcc, _dQ8, _dQ8d, _dMeta, _dX, _dH, _dH2, _dCoef, _dQ, _dK, _dV, _dQn, _dAtt, _dO, _dGu, _dAct;
        private IntPtr _dBlockK, _dBlockV, _dBlockKTab, _dBlockVTab, _dPartial, _dLogits, _dCand, _dUnary, _dSel, _dDrafts, _dConf;
        private IntPtr _dTopPartialIdx, _dTopPartialVal;
        private IntPtr _tabRingK, _tabRingV, _tabRingPos;

        /// <summary>True when a drafter is attached.</summary>
        public bool HasDrafter => _d != null || HasNinferDrafter;
        /// <summary>The attached drafter's block (anchor included), 0 without one.</summary>
        public int DraftBlockSize => HasNinferDrafter ? _ninferDraftDesc.BlockSize : _d?.BlockSize ?? 0;
        /// <summary>Committed positions a slot's drafter ring keeps.</summary>
        public int DraftRingRows => RingRows;

        private int DraftRows => MaxSeqs * _d.BlockSize;
        private int Gamma => _d.BlockSize - 1;

        /// <summary>Loads a drafter. Slots created before this have no ring: create them after.</summary>
        public void AttachDrafter(DraftDesc d)
        {
            if (_m.Ninfer != null) throw new NotSupportedException("A GGUF drafter cannot be attached to the NInfer target numerical recipe.");
            if (d == null)
                throw new ArgumentNullException(nameof(d));
            if (_d != null)
                throw new InvalidOperationException("[q35-cuda] a drafter is already attached");
            if (d.NEmbd != _m.NEmbd || d.HeadDim != Q35Kernels.DraftHeadDim || d.NHead % d.NKvHead != 0)
                throw new NotSupportedException($"[q35-cuda] drafter shape (hidden {d.NEmbd}, head {d.HeadDim}, {d.NHead}/{d.NKvHead} heads)");
            if (d.BlockSize < 2 || (long)d.BlockSize * (d.NHead / d.NKvHead) > 64 || d.ConvTaps < 1 || d.ConvTaps > d.BlockSize
                || d.SelRank != 256 || d.SelTopK < 1 || d.SelTopK > 16)
                throw new NotSupportedException("[q35-cuda] the engine drafts with DFlash2 blocks (conv taps, a rank-256 selector, top-k <= 16)");
            if (d.TargetLayers == null || d.Fc.Ne0 != d.TargetLayers.Length * _m.NEmbd)
                throw new NotSupportedException("[q35-cuda] drafter fc width does not match its target taps");
            MakeCurrent();
            InvalidateVerifyGraphs();
            _dTapOf = new int[_m.NLayer];
            Array.Fill(_dTapOf, -1);
            for (int i = 0; i < d.TargetLayers.Length; i++)
            {
                int l = d.TargetLayers[i];
                if (l < 0 || l >= _m.NLayer || _dTapOf[l] >= 0)
                    throw new NotSupportedException($"[q35-cuda] drafter taps target layer {l}");
                _dTapOf[l] = i;
            }
            _dTapCount = d.TargetLayers.Length;
            using (var loader = new Loader(this))
            {
                _dFc = Quant(loader, d.Fc);
                _dSelHidden = Quant(loader, d.SelHidden);
                _dSelPred = Quant(loader, d.SelPred);
                _dSelSucc = Quant(loader, d.SelSucc);
                if (_dSelPred.Type != TQ4_K || _dSelSucc.Type != TQ4_K || _dSelPred.K != d.SelRank || _dSelSucc.K != d.SelRank)
                    throw new NotSupportedException("[q35-cuda] the selector codebooks must be Q4_K [vocab][256]");
                _dl = new DraftLayer[d.Layers.Length];
                for (int l = 0; l < d.Layers.Length; l++)
                {
                    var s = d.Layers[l];
                    _dl[l] = new DraftLayer
                    {
                        Q = Quant(loader, s.Q), K = Quant(loader, s.K), V = Quant(loader, s.V), O = Quant(loader, s.O),
                        GateUp = Quant(loader, s.Gate, s.Up), Down = Quant(loader, s.Down),
                        AttnConv = Quant(loader, s.AttnConvProj), FfnConv = Quant(loader, s.FfnConvProj),
                        AttnNorm = UploadF32(s.AttnNorm), FfnNorm = UploadF32(s.FfnNorm), QNorm = UploadF32(s.QNorm), KNorm = UploadF32(s.KNorm),
                        AttnConvBase = UploadF32(s.AttnConvBase), FfnConvBase = UploadF32(s.FfnConvBase),
                    };
                }
            }
            _dEncNorm = UploadF32(d.EncNorm);
            _dOutNorm = UploadF32(d.OutNorm);
            _d = d;
            _ringRows = (d.SlidingWindow + d.BlockSize + 1 + 31) / 32 * 32;

            int nt = RowCap, e = _m.NEmbd, rows = DraftRows, kv = d.NKvHead * d.HeadDim, qd = d.NHead * d.HeadDim;
            int coefWidth = _dl[0].AttnConv.Rows;
            int widestK = Math.Max(Math.Max(e, d.NFf), qd);
            _dAcc = Alloc((long)nt * e * 4);
            _dQ8 = Alloc(Math.Max((long)nt * e, (long)rows * widestK));
            _dQ8d = Alloc(Math.Max((long)nt * e, (long)rows * widestK) / 32 * 4);
            _dMeta = Alloc(4L * 3 * nt);
            _dX = Alloc((long)Math.Max(rows, nt) * e * 4);
            _dH = Alloc((long)Math.Max(rows, nt) * e * 4);
            _dH2 = Alloc((long)rows * e * 4);
            _dCoef = Alloc((long)rows * coefWidth * 4 * 2);
            _dQ = Alloc((long)rows * qd * 4);
            _dK = Alloc((long)Math.Max(rows, nt) * kv * 4);
            _dV = Alloc((long)Math.Max(rows, nt) * kv * 4);
            _dQn = Alloc((long)rows * qd * 4);
            _dAtt = Alloc((long)rows * qd * 4);
            _dO = Alloc((long)rows * e * 4);
            _dGu = Alloc((long)rows * 2 * d.NFf * 4);
            _dAct = Alloc((long)rows * d.NFf * 4);
            _dBlockK = Alloc((long)rows * kv * 2);
            _dBlockV = Alloc((long)rows * kv * 2);
            _dPartial = Alloc(Q35Kernels.DfAttnPartialFloats(rows, d.NHead, _ringRows, d.BlockSize) * 4);
            _dLogits = Alloc((long)MaxSeqs * Gamma * _m.NVocab * 4);
            _dCand = Alloc((long)MaxSeqs * Gamma * d.SelTopK * 4);
            _dUnary = Alloc((long)MaxSeqs * Gamma * d.SelTopK * 4);
            long topPartialBytes = (long)MaxSeqs * Gamma * Q35Kernels.TopKPartitions * d.SelTopK * 4;
            _dTopPartialIdx = Alloc(topPartialBytes);
            _dTopPartialVal = Alloc(topPartialBytes);
            _dSel = Alloc((long)MaxSeqs * Gamma * d.SelRank * 4);
            _dDrafts = Alloc((long)MaxSeqs * Gamma * 4);
            _dConf = Alloc((long)MaxSeqs * Gamma * 4);
            // Per lane, the block's own K / V heads [nKv][b][128]: fixed bases.
            var kt = new IntPtr[MaxSeqs];
            var vt = new IntPtr[MaxSeqs];
            for (int i = 0; i < MaxSeqs; i++)
            {
                kt[i] = _dBlockK + (nint)((long)i * d.BlockSize * kv * 2);
                vt[i] = _dBlockV + (nint)((long)i * d.BlockSize * kv * 2);
            }
            _dBlockKTab = Alloc(8L * MaxSeqs);
            _dBlockVTab = Alloc(8L * MaxSeqs);
            fixed (IntPtr* p = kt)
                CudaDriverApi.cuMemcpyHtoD(_dBlockKTab, (IntPtr)p, new UIntPtr(8UL * MaxSeqs)).ThrowOnError();
            fixed (IntPtr* p = vt)
                CudaDriverApi.cuMemcpyHtoD(_dBlockVTab, (IntPtr)p, new UIntPtr(8UL * MaxSeqs)).ThrowOnError();
            _tabRingK = Alloc(8L * d.Layers.Length * MaxSeqs);
            _tabRingV = Alloc(8L * d.Layers.Length * MaxSeqs);
            _tabRingPos = Alloc(8L * MaxSeqs);
            _bound = Array.Empty<Slot>();
            // The active slot predates the drafter: give it a ring.
            if (_active != null && _active.RingPos == IntPtr.Zero)
            {
                MakeCurrent();
                CreateRing(_active);
                ResetRing(_active);
                Sync();
            }
            Console.Error.WriteLine($"[q35-cuda] DFlash2 drafter: {d.Layers.Length} layers, block {d.BlockSize}, ring {_ringRows} rows, " +
                $"taps [{string.Join(",", d.TargetLayers)}]");
        }

        private QMat Quant(Loader loader, params CudaWeightDesc[] parts)
        {
            int type = parts[0].GgmlType, k = parts[0].Ne0, rows = 0;
            foreach (var w in parts)
            {
                if (w.GgmlType != type || w.Ne0 != k)
                    throw new NotSupportedException($"[q35-cuda] {w.Name}: fused parts differ in type or width");
                if (type != TQ4_K && type != TQ6_K)
                    throw new NotSupportedException($"[q35-cuda] {w.Name}: drafter weight type {type} (the engine takes Q4_K / Q6_K)");
                rows += w.Ne1;
            }
            var m = new QMat { Rows = rows, K = k, Type = type, RowBytes = parts[0].RowBytes };
            m.W = Alloc(m.RowBytes * rows);
            long off = 0;
            foreach (var w in parts)
            {
                loader.RawInto(w, m.W + (nint)off);
                off += w.TotalBytes;
            }
            return m;
        }

        /// <summary>y[n, w.Rows] (+)= x[n, w.K] W^T for rows already quantized into _dQ8.</summary>
        private void DMatMul(in QMat w, IntPtr y, int n, bool accumulate = false, long xOffsetRows = 0, int colBlock = -1)
        {
            IntPtr wp = w.W;
            int k = w.K;
            if (colBlock >= 0)
            {
                // fc's column block: k values from superblock colBlock * k / 256 of every row.
                k = _m.NEmbd;
                wp += (nint)((long)colBlock * k / 256 * 144);
            }
            _k.GemmQuant(w.Type, wp, w.RowBytes, w.Rows, k, _dQ8 + (nint)(xOffsetRows * k), _dQ8d + (nint)(xOffsetRows * (k / 32) * 4),
                n, y, accumulate, Stream);
        }

        private void DQuant(IntPtr x, int k, int rows) => _k.QuantizeQ81(x, _dQ8, _dQ8d, k, rows, Stream);

        // ---- slot rings ----

        /// <summary>A ring of its own for a slot made before the drafter was loaded.</summary>
        private void CreateRing(Slot s) => PlaceRing(s, Alloc(RingBytes, s.Owned));

        /// <summary>Lays a slot's ring out from <paramref name="ringBase"/> (<see cref="RingBytes"/>):
        /// K and V per draft layer, then the position each row holds.</summary>
        private void PlaceRing(Slot s, IntPtr ringBase)
        {
            int layers = RingLayers;
            long step = Align256(RingLayerBytes), off = 0;
            s.RingBase = ringBase;
            s.RingK = new IntPtr[layers];
            s.RingV = new IntPtr[layers];
            for (int l = 0; l < layers; l++)
            {
                s.RingK[l] = ringBase + (nint)off;
                off += step;
                s.RingV[l] = ringBase + (nint)off;
                off += step;
            }
            s.RingPos = ringBase + (nint)off;
        }

        private void ResetRing(Slot s)
        {
            if (s.RingPos != IntPtr.Zero)
                CudaDriverApi.cuMemsetD8Async(s.RingPos, 0xFF, new UIntPtr(4UL * (ulong)RingRows), Stream).ThrowOnError();
        }

        private void CopyRing(Slot src, Slot dst)
        {
            if (src.RingBase == IntPtr.Zero || dst.RingBase == IntPtr.Zero)
                return;
            CudaDriverApi.cuMemcpyDtoDAsync(dst.RingBase, src.RingBase, new UIntPtr((ulong)RingBytes), Stream).ThrowOnError();
        }

        private void BindRings(Slot[] slots, IntPtr[] host)
        {
            int layers = _dl.Length;
            for (int q = 0; q < slots.Length; q++)
            {
                if (slots[q].RingPos == IntPtr.Zero)
                    throw new InvalidOperationException("[q35-cuda] a slot created before the drafter has no ring");
                for (int l = 0; l < layers; l++)
                {
                    host[l * MaxSeqs + q] = slots[q].RingK[l];
                    host[(layers + l) * MaxSeqs + q] = slots[q].RingV[l];
                }
                host[2 * layers * MaxSeqs + q] = slots[q].RingPos;
            }
            fixed (IntPtr* h = host)
            {
                CudaDriverApi.cuMemcpyHtoD(_tabRingK, (IntPtr)h, new UIntPtr(8UL * (ulong)(layers * MaxSeqs))).ThrowOnError();
                CudaDriverApi.cuMemcpyHtoD(_tabRingV, (IntPtr)(h + layers * MaxSeqs), new UIntPtr(8UL * (ulong)(layers * MaxSeqs))).ThrowOnError();
                CudaDriverApi.cuMemcpyHtoD(_tabRingPos, (IntPtr)(h + 2 * layers * MaxSeqs), new UIntPtr(8UL * MaxSeqs)).ThrowOnError();
            }
        }

        // ---- taps and inject ----

        /// <summary>At target layer <paramref name="il"/>: rows [first, n) of the residual in _x
        /// through the layer's fc column block into the accumulators.</summary>
        private void DraftTap(int il, int first, int n)
        {
            int t = _dTapOf[il];
            if (t < 0 || first >= n)
                return;
            int e = _m.NEmbd, rows = n - first;
            IntPtr x = _x + (nint)((long)first * e * 4);
            DQuant(x, e, rows);
            DMatMul(_dFc, _dAcc + (nint)((long)first * e * 4), rows, accumulate: t > 0, colBlock: t);
        }

        /// <summary>Encodes accumulator rows idx[i] (i &lt; rows) and writes them into the ring of
        /// bound slot lane[i] at position pos[i].</summary>
        private void DraftInject(int rows, int[] idx, int[] lane, int[] pos)
        {
            if (rows == 0)
                return;
            int e = _m.NEmbd;
            var meta = new int[3 * rows];
            Array.Copy(idx, 0, meta, 0, rows);
            Array.Copy(lane, 0, meta, rows, rows);
            Array.Copy(pos, 0, meta, 2 * rows, rows);
            fixed (int* p = meta)
                Upload(_dMeta, p, 3 * rows);
            IntPtr idxDev = _dMeta, laneDev = _dMeta + (nint)(4L * rows), posDev = _dMeta + (nint)(8L * rows);
            _k.RowsByIndex(_dAcc, e, idxDev, _dX, rows, Stream);
            _k.AddRmsNorm(_dX, IntPtr.Zero, _dEncNorm, _dH, rows, e, _d.Eps, Stream);
            DQuant(_dH, e, rows);
            for (int l = 0; l < _dl.Length; l++)
            {
                var L = _dl[l];
                DMatMul(L.K, _dK, rows);
                DMatMul(L.V, _dV, rows);
                _k.DfPrep(IntPtr.Zero, _dK, _dV, rows, 1, _d.NHead, _d.NKvHead, IntPtr.Zero, L.KNorm, _d.Eps, posDev, _d.RopeBase,
                    IntPtr.Zero, _tabRingK + (nint)(8L * l * MaxSeqs), _tabRingV + (nint)(8L * l * MaxSeqs),
                    (long)_ringRows * _d.HeadDim, _ringRows, _tabRingPos, laneDev, Stream);
            }
        }

        /// <summary>Injects rows [first, n) of a group of the last RunGroups (its rows from rowOff, its
        /// slots bound from laneOff): rowsPerSeq rows per sequence, each at its slot's position before
        /// the step plus its row.</summary>
        private void DraftInjectRun(Slot[] slots, int rowsPerSeq, int first, int[] npastBefore, int[] counts = null, int rowOff = 0, int laneOff = 0)
        {
            int n = slots.Length * rowsPerSeq;
            if (first >= n)
                return;
            var idx = new List<int>(n - first);
            var lane = new List<int>(n - first);
            var pos = new List<int>(n - first);
            for (int r = first; r < n; r++)
            {
                int q = r / rowsPerSeq, t = r % rowsPerSeq;
                if (counts != null && t >= counts[q])
                    continue;   // padding
                idx.Add(rowOff + r);
                lane.Add(laneOff + q);
                pos.Add(npastBefore[q] + t);
            }
            int rows = idx.Count;
            for (int c = 0; c < rows; c += _m.NUbatch)
            {
                int m = Math.Min(_m.NUbatch, rows - c);
                DraftInject(m, idx.GetRange(c, m).ToArray(), lane.GetRange(c, m).ToArray(), pos.GetRange(c, m).ToArray());
            }
        }

        /// <summary>After a commit: each sequence's kept verify rows, at their positions.</summary>
        private void DraftInjectKept(Slot[] slots, int window, int[] kept, int[] npastBefore)
        {
            int rows = 0;
            foreach (int k in kept)
                rows += k;
            var idx = new int[rows];
            var lane = new int[rows];
            var pos = new int[rows];
            int i = 0;
            for (int q = 0; q < slots.Length; q++)
                for (int j = 0; j < kept[q]; j++, i++)
                {
                    idx[i] = _verifyRow0 + q * window + j;
                    lane[i] = q;
                    pos[i] = npastBefore[q] + j;
                }
            DraftInject(rows, idx, lane, pos);
        }

        // ---- draft ----

        /// <summary>
        /// Drafts one block for each slot: anchors[i] is slot i's next token, at the slot's
        /// position (not yet forwarded). Fills draftsOut[i] (b - 1 tokens) and confOut[i] (the
        /// selector's probability of each choice). Optional logitsOut receives every proposal
        /// row's full-vocabulary logits ([lanes * (b - 1)][vocab]) for numerical validation.
        /// </summary>
        public void Draft(Slot[] slots, int[] anchors, int[][] draftsOut, float[][] confOut, float[] logitsOut = null)
        {
            if (HasNinferDrafter)
            {
                ProposeNinfer(slots, anchors, draftsOut, confOut, logitsOut);
                return;
            }
            if (_d == null)
                throw new InvalidOperationException("[q35-cuda] no drafter attached");
            MakeCurrent();
            BindSlots(slots);
            int lanes = slots.Length, b = _d.BlockSize, rows = lanes * b, g = Gamma;
            if (logitsOut != null && logitsOut.Length < (long)lanes * g * _m.NVocab)
                throw new ArgumentException("[q35-cuda] draft logits output is too short", nameof(logitsOut));
            var meta = new int[2 * rows + lanes];
            for (int q = 0; q < lanes; q++)
            {
                for (int t = 0; t < b; t++)
                {
                    meta[q * b + t] = t == 0 ? anchors[q] : _d.MaskToken;
                    meta[rows + q * b + t] = slots[q].NPast + t;
                }
                meta[2 * rows + q] = slots[q].NPast;
            }
            fixed (int* p = meta)
                Upload(_dMeta, p, meta.Length);
            RunDraftPass(lanes);
            Sync();
            var drafts = new int[lanes * g];
            var conf = new float[lanes * g];
            fixed (int* p = drafts)
                CudaDriverApi.cuMemcpyDtoH((IntPtr)p, _dDrafts, new UIntPtr((ulong)drafts.Length * 4)).ThrowOnError();
            fixed (float* p = conf)
                CudaDriverApi.cuMemcpyDtoH((IntPtr)p, _dConf, new UIntPtr((ulong)conf.Length * 4)).ThrowOnError();
            if (logitsOut != null)
                fixed (float* p = logitsOut)
                    CudaDriverApi.cuMemcpyDtoH((IntPtr)p, _dLogits, new UIntPtr((ulong)lanes * (ulong)g * (ulong)_m.NVocab * 4)).ThrowOnError();
            for (int q = 0; q < lanes; q++)
            {
                Array.Copy(drafts, q * g, draftsOut[q], 0, g);
                Array.Copy(conf, q * g, confOut[q], 0, g);
            }
        }

        // Every address below belongs to the engine for its lifetime. Slot addresses are read
        // indirectly from _tabRing*; BindSlots refreshes those tables before each draft. Anchors
        // and positions are read from _dMeta, uploaded on the same stream before this pass.
        // Keep host transfers, allocations, synchronization and slot mutation outside this method.
        private void DraftPass(int lanes)
        {
            int b = _d.BlockSize, rows = lanes * b, e = _m.NEmbd, g = Gamma;
            int qd = _d.NHead * _d.HeadDim;
            IntPtr tokDev = _dMeta, posDev = _dMeta + (nint)(4L * rows), anchorDev = _dMeta + (nint)(8L * rows);
            _k.EmbedQ4_0(_tokEmbd, _tokEmbdRowBytes, tokDev, _dX, rows, e, Stream);
            IntPtr pending = IntPtr.Zero;
            int coefW = _dl[0].AttnConv.Rows;
            IntPtr coefA = _dCoef, coefF = _dCoef + (nint)((long)rows * coefW * 4);
            for (int l = 0; l < _dl.Length; l++)
            {
                var L = _dl[l];
                _k.AddRmsNorm(_dX, pending, L.AttnNorm, _dH, rows, e, _d.Eps, Stream);
                DQuant(_dH, e, rows);
                DMatMul(L.AttnConv, coefA, rows);
                _k.DfConv(_dH, coefA, coefW, L.AttnConvBase, 0, _d.ConvTaps, _d.ConvGroup, e, b, rows, _dH2, Stream);
                DQuant(_dH2, e, rows);
                DMatMul(L.Q, _dQ, rows);
                DMatMul(L.K, _dK, rows);
                DMatMul(L.V, _dV, rows);
                _k.DfPrep(_dQ, _dK, _dV, rows, b, _d.NHead, _d.NKvHead, L.QNorm, L.KNorm, _d.Eps, posDev, _d.RopeBase, _dQn,
                    _dBlockKTab, _dBlockVTab, (long)b * _d.HeadDim, 0, IntPtr.Zero, IntPtr.Zero, Stream);
                _k.DfAttention(_dQn, _tabRingK + (nint)(8L * l * MaxSeqs), _tabRingV + (nint)(8L * l * MaxSeqs), _tabRingPos, _ringRows,
                    _dBlockK, _dBlockV, b, lanes, _d.NHead, _d.NKvHead, anchorDev, _d.SlidingWindow, 1.0f / MathF.Sqrt(_d.HeadDim),
                    _dPartial, _dAtt, Stream);
                DQuant(_dAtt, qd, rows);
                DMatMul(L.O, _dO, rows);
                _k.DfConv(_dO, coefA, coefW, L.AttnConvBase, 1, _d.ConvTaps, _d.ConvGroup, e, b, rows, _dH2, Stream);
                _k.AddRmsNorm(_dX, _dH2, L.FfnNorm, _dH, rows, e, _d.Eps, Stream);
                DQuant(_dH, e, rows);
                DMatMul(L.FfnConv, coefF, rows);
                _k.DfConv(_dH, coefF, coefW, L.FfnConvBase, 0, _d.ConvTaps, _d.ConvGroup, e, b, rows, _dH2, Stream);
                DQuant(_dH2, e, rows);
                DMatMul(L.GateUp, _dGu, rows);
                _k.SwiGlu(_dGu, _dAct, _d.NFf, rows, Stream);
                DQuant(_dAct, _d.NFf, rows);
                DMatMul(L.Down, _dO, rows);
                _k.DfConv(_dO, coefF, coefW, L.FfnConvBase, 1, _d.ConvTaps, _d.ConvGroup, e, b, rows, _dH2, Stream);
                pending = _dH2;
            }
            _k.AddRmsNorm(_dX, pending, _dOutNorm, _dH, rows, e, _d.Eps, Stream);
            // The proposal rows (each block's rows 1 .. b-1) through the target's LM head and the selector.
            _k.DfGather(_dH, e, b, 1, g, lanes, _dH2, Stream);
            // Both head policies leave these Q81 inputs live for the selector's
            // immediately following DMatMul(_dSelHidden), which consumes _dQ8/_dQ8d.
            DQuant(_dH2, e, lanes * g);
            _k.GemvQ6kTiled(_output, _m.NVocab, e, _dQ8, _dQ8d, _dLogits, lanes * g, Stream);
            _k.TopK(_dLogits, lanes * g, _m.NVocab, _d.SelTopK, _dCand, _dUnary, Stream, _dTopPartialIdx, _dTopPartialVal);
            DMatMul(_dSelHidden, _dSel, lanes * g);
            _k.DfSelect(_dCand, _dUnary, _dSel, _dSelPred.W, _dSelSucc.W, _dSelPred.RowBytes, anchorDev, lanes, g, _d.SelTopK,
                _d.SelRank, _d.LogitScale, _dDrafts, _dConf, Stream);
        }

    }
}
