// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
using System;
using TensorSharp.GGML;

namespace TensorSharp.Models
{
    /// <summary>
    /// Routes the DFlash drafter's two passes through one fused GGML graph each.
    ///
    /// The per-op drafter issues ~150 dispatches per speculative step for ~2.5 GB of
    /// weight reads -- about a quarter of one target forward, but it was costing more
    /// wall time than the forward it was meant to save, which made speculation a net
    /// loss. llama.cpp does not solve this with graph reuse (its draft context rebuilds
    /// every step, because ENCODER -&gt; DECODER(embd) -&gt; DECODER(token) can never hit
    /// its single-slot graph cache); it just keeps the rebuild cheap. Here the graph is
    /// built once and replayed, so ggml-cuda can capture it.
    ///
    /// One thing this does that llama.cpp does not: the draft block's argmax and its
    /// winning probability are reduced ON DEVICE. llama.cpp pulls the whole
    /// [202048, 16] probability block back to the host every step -- 12.9 MB over PCIe
    /// plus a 3.2 M-element scan -- and that readback is a large share of its per-step
    /// cost. Two 16-element tensors carry everything the executor needs.
    /// </summary>
    public abstract partial class ModelBase
    {
        private sealed class DFlashArrays
        {
            public IntPtr[] AttnNorm, QNorm, KNorm, FfnNorm;
            public IntPtr[] Q, K, V, O, Gate, Up, Down;
            public int[] QType, KType, VType, OType, GateType, UpType, DownType;
            public long[] QNe0, QNe1, QBytes;
            public long[] KNe0, KNe1, KBytes;
            public long[] VNe0, VNe1, VBytes;
            public long[] ONe0, ONe1, OBytes;
            public long[] GateNe0, GateNe1, GateBytes;
            public long[] UpNe0, UpNe1, UpBytes;
            public long[] DownNe0, DownNe1, DownBytes;
            public IntPtr[] RingK, RingV;

            /// <summary>Encoder: the [featureSize -&gt; hidden] projection and its norm.</summary>
            public IntPtr Fc;
            public int FcType;
            public long FcNe0, FcNe1, FcBytes;
            public IntPtr EncNorm;

            /// <summary>Drafter's final norm, plus the TARGET's embedding table and LM head.</summary>
            public IntPtr OutNorm;
            public IntPtr TokEmbd;
            public int TokEmbdType;
            public long TokEmbdNe0, TokEmbdNe1, TokEmbdBytes;
            public IntPtr LmHead;
            public int LmHeadType;
            public long LmHeadNe0, LmHeadNe1, LmHeadBytes;

            public int RingRows;

            /// <summary>DFlash2 grouped convolution: per layer, the static base
            /// kernel [hidden, taps, 2] and the projection that produces both taps
            /// from the sublayer input. Null on a first-generation drafter.</summary>
            public IntPtr[] AttnConvBase, FfnConvBase;
            public IntPtr[] AttnConvProj, FfnConvProj;
            public int[] AttnConvProjType, FfnConvProjType;
            public long[] AttnConvProjNe0, AttnConvProjNe1, AttnConvProjBytes;
            public long[] FfnConvProjNe0, FfnConvProjNe1, FfnConvProjBytes;

            /// <summary>DFlash2 candidate selector: the hidden projection and the
            /// two [vocab, rank] transition codebooks. Zero when absent.</summary>
            public IntPtr SelHidden, SelPred, SelSucc;
            public int SelHiddenType, SelPredType, SelSuccType;
            public long SelHiddenNe0, SelHiddenNe1, SelHiddenBytes;
            public long SelPredNe0, SelPredNe1, SelPredBytes;
            public long SelSuccNe0, SelSuccNe1, SelSuccBytes;
        }

        private DFlashArrays _dflashArrays;
        private bool _dflashFusedProbed;

        /// <summary>Highest ring position written + 1. The ring slot for position p is
        /// p % ringRows, so this is all that is needed to reconstruct which slot holds
        /// which position (and which slots were never written).</summary>
        private int _dflashRingFilled;

        /// <summary>Set when the fused kernels have written the ring on the device, so
        /// the host mirror is behind.</summary>
        private bool _dflashRingHostStale;

        /// <summary>
        /// Pull device-side ring writes back to the host copy. Only the per-op drafter
        /// walks the ring with host pointers, so this is a no-op on the fused path.
        /// </summary>
        private void EnsureDFlashRingHostSynchronized()
        {
            if (!_dflashRingHostStale || _dflashRingK == null)
                return;
            for (int l = 0; l < _dflash.NumLayers; l++)
            {
                SyncTensorHostCache(_dflashRingK[l]);
                SyncTensorHostCache(_dflashRingV[l]);
            }
            _dflashRingHostStale = false;
        }

        private int[] _dflashRingSlotPos;
        private int[] _dflashDraftIds;
        private float[] _dflashDraftConf;
        private long[] _dflashInjectIdx;
        private int[] _dflashInjectPos;

        /// <summary>GGML_TYPE_F32. The ring is allocated as Float32 (see the ring
        /// construction in the loader), and the kernel needs the ggml enum value.</summary>
        private const int GgmlTypeF32 = 0;

        private static readonly bool DFlashFusedEnabled =
            !string.Equals(Environment.GetEnvironmentVariable("TS_DFLASH_FUSED"), "0", StringComparison.Ordinal);

        /// <summary>
        /// Same backend set as the trunk kernel (see
        /// <c>MuseGlimmerModel.Fused.CanUseFusedForward</c>): CUDA, Vulkan and
        /// Metal. CPU keeps the per-op drafter, which is correct.
        ///
        /// Metal was excluded with the trunk kernel and is included for the same
        /// reason - ggml-metal implements every op these graphs use, including
        /// GGML_OP_SET_ROWS. Leaving it out made speculation a NET LOSS there: the
        /// per-op drafter cost more per step than it saved, and a 300-token greedy
        /// generation took 46.7 s with `--draft-model` versus 17.7 s without.
        /// </summary>
        private bool CanUseFusedDFlash =>
            DFlashFusedEnabled && _hasDFlash && !IsTensorParallel &&
            (_backend == BackendType.GgmlCuda || _backend == BackendType.GgmlVulkan ||
             _backend == BackendType.GgmlMetal);

        /// <summary>A quantized weight the fused kernels can bind by cache key.
        /// (ModelBase has no such helper; the trunk kernels each roll their own.)</summary>
        private bool TryDFlashQuant(string name, out QuantizedWeight qw)
            => _quantWeights.TryGetValue(name, out qw) && qw != null && qw.CacheKey != IntPtr.Zero;

        private unsafe void BuildDFlashArrays()
        {
            _dflashFusedProbed = true;
            _dflashArrays = null;
            if (!CanUseFusedDFlash)
                return;

            var cfg = _dflash;
            int n = cfg.NumLayers;
            var a = new DFlashArrays
            {
                AttnNorm = new IntPtr[n], QNorm = new IntPtr[n], KNorm = new IntPtr[n], FfnNorm = new IntPtr[n],
                Q = new IntPtr[n], K = new IntPtr[n], V = new IntPtr[n], O = new IntPtr[n],
                Gate = new IntPtr[n], Up = new IntPtr[n], Down = new IntPtr[n],
                QType = new int[n], KType = new int[n], VType = new int[n], OType = new int[n],
                GateType = new int[n], UpType = new int[n], DownType = new int[n],
                QNe0 = new long[n], QNe1 = new long[n], QBytes = new long[n],
                KNe0 = new long[n], KNe1 = new long[n], KBytes = new long[n],
                VNe0 = new long[n], VNe1 = new long[n], VBytes = new long[n],
                ONe0 = new long[n], ONe1 = new long[n], OBytes = new long[n],
                GateNe0 = new long[n], GateNe1 = new long[n], GateBytes = new long[n],
                UpNe0 = new long[n], UpNe1 = new long[n], UpBytes = new long[n],
                DownNe0 = new long[n], DownNe1 = new long[n], DownBytes = new long[n],
                RingK = new IntPtr[n], RingV = new IntPtr[n],
                RingRows = _dflashRingRows,
            };
            if (cfg.HasConv)
            {
                a.AttnConvBase = new IntPtr[n]; a.FfnConvBase = new IntPtr[n];
                a.AttnConvProj = new IntPtr[n]; a.FfnConvProj = new IntPtr[n];
                a.AttnConvProjType = new int[n]; a.FfnConvProjType = new int[n];
                a.AttnConvProjNe0 = new long[n]; a.AttnConvProjNe1 = new long[n]; a.AttnConvProjBytes = new long[n];
                a.FfnConvProjNe0 = new long[n]; a.FfnConvProjNe1 = new long[n]; a.FfnConvProjBytes = new long[n];
            }

            for (int l = 0; l < n; l++)
            {
                string[] wn = _dflashLayerNames[l];
                if (!TryDFlashQuant(wn[DfAttnQ], out var q) || !TryDFlashQuant(wn[DfAttnK], out var k)
                    || !TryDFlashQuant(wn[DfAttnV], out var v) || !TryDFlashQuant(wn[DfAttnOutput], out var o)
                    || !TryDFlashQuant(wn[DfFfnGate], out var gate) || !TryDFlashQuant(wn[DfFfnUp], out var up)
                    || !TryDFlashQuant(wn[DfFfnDown], out var down))
                {
                    Console.WriteLine($"  DFlash fused drafter disabled: layer {l} has a non-quantized projection.");
                    return;
                }
                if (!_weights.TryGetValue(wn[DfAttnNorm], out var an) || !_weights.TryGetValue(wn[DfAttnQNorm], out var qn)
                    || !_weights.TryGetValue(wn[DfAttnKNorm], out var kn) || !_weights.TryGetValue(wn[DfFfnNorm], out var fn))
                {
                    Console.WriteLine($"  DFlash fused drafter disabled: layer {l} is missing a norm weight.");
                    return;
                }

                a.AttnNorm[l] = (IntPtr)GetFloatPtr(an);
                a.QNorm[l] = (IntPtr)GetFloatPtr(qn);
                a.KNorm[l] = (IntPtr)GetFloatPtr(kn);
                a.FfnNorm[l] = (IntPtr)GetFloatPtr(fn);

                a.Q[l] = q.CacheKey; a.QType[l] = q.GgmlType; a.QNe0[l] = q.Ne0; a.QNe1[l] = q.Ne1; a.QBytes[l] = q.RawBytes;
                a.K[l] = k.CacheKey; a.KType[l] = k.GgmlType; a.KNe0[l] = k.Ne0; a.KNe1[l] = k.Ne1; a.KBytes[l] = k.RawBytes;
                a.V[l] = v.CacheKey; a.VType[l] = v.GgmlType; a.VNe0[l] = v.Ne0; a.VNe1[l] = v.Ne1; a.VBytes[l] = v.RawBytes;
                a.O[l] = o.CacheKey; a.OType[l] = o.GgmlType; a.ONe0[l] = o.Ne0; a.ONe1[l] = o.Ne1; a.OBytes[l] = o.RawBytes;
                a.Gate[l] = gate.CacheKey; a.GateType[l] = gate.GgmlType; a.GateNe0[l] = gate.Ne0; a.GateNe1[l] = gate.Ne1; a.GateBytes[l] = gate.RawBytes;
                a.Up[l] = up.CacheKey; a.UpType[l] = up.GgmlType; a.UpNe0[l] = up.Ne0; a.UpNe1[l] = up.Ne1; a.UpBytes[l] = up.RawBytes;
                a.Down[l] = down.CacheKey; a.DownType[l] = down.GgmlType; a.DownNe0[l] = down.Ne0; a.DownNe1[l] = down.Ne1; a.DownBytes[l] = down.RawBytes;

                a.RingK[l] = TensorComputePrimitives.GetStoragePointer(_dflashRingK[l]);
                a.RingV[l] = TensorComputePrimitives.GetStoragePointer(_dflashRingV[l]);

                if (!cfg.HasConv)
                    continue;
                if (!TryDFlashQuant(wn[DfAttnConvProj], out var acp) || !TryDFlashQuant(wn[DfFfnConvProj], out var fcp))
                {
                    Console.WriteLine($"  DFlash fused drafter disabled: layer {l} has a non-quantized conv projection.");
                    return;
                }
                if (!_weights.TryGetValue(wn[DfAttnConvBase], out var acb) || !_weights.TryGetValue(wn[DfFfnConvBase], out var fcb))
                {
                    Console.WriteLine($"  DFlash fused drafter disabled: layer {l} is missing a conv base kernel.");
                    return;
                }
                a.AttnConvBase[l] = (IntPtr)GetFloatPtr(acb);
                a.FfnConvBase[l] = (IntPtr)GetFloatPtr(fcb);
                a.AttnConvProj[l] = acp.CacheKey; a.AttnConvProjType[l] = acp.GgmlType;
                a.AttnConvProjNe0[l] = acp.Ne0; a.AttnConvProjNe1[l] = acp.Ne1; a.AttnConvProjBytes[l] = acp.RawBytes;
                a.FfnConvProj[l] = fcp.CacheKey; a.FfnConvProjType[l] = fcp.GgmlType;
                a.FfnConvProjNe0[l] = fcp.Ne0; a.FfnConvProjNe1[l] = fcp.Ne1; a.FfnConvProjBytes[l] = fcp.RawBytes;
            }

            string fcName = DFlashConfig.WeightPrefix + "fc.weight";
            if (!TryDFlashQuant(fcName, out var fc))
            {
                Console.WriteLine("  DFlash fused drafter disabled: the encoder projection is not quantized.");
                return;
            }
            a.Fc = fc.CacheKey; a.FcType = fc.GgmlType; a.FcNe0 = fc.Ne0; a.FcNe1 = fc.Ne1; a.FcBytes = fc.RawBytes;

            if (!_weights.TryGetValue(DFlashConfig.WeightPrefix + "enc.output_norm.weight", out var encNorm)
                || !_weights.TryGetValue(DFlashConfig.WeightPrefix + "output_norm.weight", out var outNorm))
            {
                Console.WriteLine("  DFlash fused drafter disabled: a drafter norm weight is missing.");
                return;
            }
            a.EncNorm = (IntPtr)GetFloatPtr(encNorm);
            a.OutNorm = (IntPtr)GetFloatPtr(outNorm);

            // Both borrowed from the target, exactly as llama.cpp's dflash graph does
            // through cparams.ctx_other -- the drafter owns neither.
            if (!TryDFlashQuant("token_embd.weight", out var tok)
                || !TryDFlashQuant(DFlashTargetOutputWeightName, out var head))
            {
                Console.WriteLine("  DFlash fused drafter disabled: the target embedding/LM head is not quantized.");
                return;
            }
            a.TokEmbd = tok.CacheKey; a.TokEmbdType = tok.GgmlType; a.TokEmbdNe0 = tok.Ne0; a.TokEmbdNe1 = tok.Ne1; a.TokEmbdBytes = tok.RawBytes;
            a.LmHead = head.CacheKey; a.LmHeadType = head.GgmlType; a.LmHeadNe0 = head.Ne0; a.LmHeadNe1 = head.Ne1; a.LmHeadBytes = head.RawBytes;

            if (cfg.HasSelector)
            {
                if (!TryDFlashQuant(DFlashConfig.WeightPrefix + "selector_hidden.weight", out var selH)
                    || !TryDFlashQuant(DFlashConfig.WeightPrefix + "selector_predecessor.weight", out var selP)
                    || !TryDFlashQuant(DFlashConfig.WeightPrefix + "selector_successor.weight", out var selS))
                {
                    Console.WriteLine("  DFlash fused drafter disabled: a selector table is not quantized.");
                    return;
                }
                a.SelHidden = selH.CacheKey; a.SelHiddenType = selH.GgmlType;
                a.SelHiddenNe0 = selH.Ne0; a.SelHiddenNe1 = selH.Ne1; a.SelHiddenBytes = selH.RawBytes;
                a.SelPred = selP.CacheKey; a.SelPredType = selP.GgmlType;
                a.SelPredNe0 = selP.Ne0; a.SelPredNe1 = selP.Ne1; a.SelPredBytes = selP.RawBytes;
                a.SelSucc = selS.CacheKey; a.SelSuccType = selS.GgmlType;
                a.SelSuccNe0 = selS.Ne0; a.SelSuccNe1 = selS.Ne1; a.SelSuccBytes = selS.RawBytes;
            }

            _dflashArrays = a;
            Console.WriteLine($"  DFlash fused drafter armed ({n} draft layers, ring {a.RingRows} rows, "
                + (cfg.HasSelector ? "on-device lattice" : "on-device top-1")
                + (cfg.HasConv ? $", conv {cfg.ConvKernelSize}x{cfg.ConvGroupSize}" : string.Empty) + ").");
        }

        private DFlashArrays GetDFlashArrays()
        {
            if (!_dflashFusedProbed)
                BuildDFlashArrays();
            if (_dflashArrays != null && _dflashArrays.RingRows != _dflashRingRows)
            {
                GgmlBasicOps.DFlashResetCaches();
                BuildDFlashArrays();
            }
            return _dflashArrays;
        }

        /// <summary>Drops the persistent DFlash graphs (ring reallocation or KV reset).</summary>
        private void ResetFusedDFlashCaches()
        {
            if (_dflashArrays != null)
                GgmlBasicOps.DFlashResetCaches();
            _dflashRingFilled = 0;
        }

        /// <summary>
        /// Which absolute position each ring slot currently holds, or -1 for a slot
        /// that was never written. The kernel reads the WHOLE ring every step so its
        /// graph shape stays constant; this map is what lets its mask express liveness,
        /// causality and the sliding window without changing the shape.
        /// </summary>
        private int[] BuildRingSlotPositions()
        {
            int rows = _dflashRingRows;
            if (_dflashRingSlotPos == null || _dflashRingSlotPos.Length != rows)
                _dflashRingSlotPos = new int[rows];
            var map = _dflashRingSlotPos;
            for (int i = 0; i < rows; i++)
                map[i] = -1;
            int lo = Math.Max(0, _dflashRingFilled - rows);
            for (int p = lo; p < _dflashRingFilled; p++)
                map[p % rows] = p;
            return map;
        }

        /// <summary>
        /// PASS A+B in one graph: fc -&gt; RMSNorm -&gt; per layer {k/v projection, k head
        /// norm, NeoX RoPE, ring scatter}. Returns false when the fused path is not
        /// available, in which case the caller runs the per-op version.
        /// </summary>
        private bool TryFusedDFlashInject(float[] hRows, int rowOffset, int n, int startPos)
        {
            var a = GetDFlashArrays();
            if (a == null)
                return false;

            var cfg = _dflash;
            int feat = cfg.FeatureSize;

            // The ring drops rows older than its capacity, so only the last min(n, rows)
            // survive; writing the earlier ones would just be overwritten in place.
            int keep = Math.Min(n, _dflashRingRows);
            int skip = n - keep;

            float[] rows = hRows;
            if (rowOffset + skip != 0)
            {
                rows = new float[(long)keep * feat];
                Array.Copy(hRows, (long)(rowOffset + skip) * feat, rows, 0, (long)keep * feat);
            }

            if (_dflashInjectIdx == null || _dflashInjectIdx.Length < keep)
            {
                _dflashInjectIdx = new long[keep];
                _dflashInjectPos = new int[keep];
            }
            for (int i = 0; i < keep; i++)
            {
                int p = startPos + skip + i;
                _dflashInjectIdx[i] = p % _dflashRingRows;
                _dflashInjectPos[i] = p;
            }

            bool ok = GgmlBasicOps.DFlashInject(
                rows, feat, keep, _dflashInjectIdx, _dflashInjectPos,
                cfg.NumLayers, cfg.HiddenSize, cfg.HeadDim, cfg.NumKVHeads, _dflashRingRows,
                cfg.Eps, cfg.RopeBase, 1f,
                a.Fc, a.FcType, a.FcNe0, a.FcNe1, a.FcBytes, a.EncNorm,
                a.K, a.KType, a.KNe0, a.KNe1, a.KBytes,
                a.V, a.VType, a.VNe0, a.VNe1, a.VBytes,
                a.KNorm, a.RingK, a.RingV, GgmlTypeF32);
            if (!ok)
                return false;

            // See DFlashInjectKv: the frontier must be able to move backwards.
            _dflashRingFilled = startPos + n;
            _dflashRingHostStale = true;
            return true;
        }

        /// <summary>
        /// PASS C in one graph: [anchor, MASK x (b-1)] through the draft blocks, then
        /// the target's LM head, softmax, and an on-device argmax. Returns the number
        /// of drafted tokens, or -1 when the fused path is unavailable.
        /// </summary>
        private int TryFusedDFlashDraftBlock(int anchorToken, int position, int b, int[] draftOut, float[] confOut)
        {
            var a = GetDFlashArrays();
            if (a == null)
                return -1;

            var cfg = _dflash;

            // The fused graph has no Markov-head or attention-sink arms; the
            // per-op block draft covers both (their presence also changes the
            // draft count, see DFlashMarkovBlock).
            if (cfg.MarkovRank > 0 || cfg.HasAttentionSinks)
                return -1;

            if (_dflashDraftIds == null || _dflashDraftIds.Length < b)
            {
                _dflashDraftIds = new int[b];
                _dflashDraftConf = new float[b];
            }
            int[] ids = new int[b];
            int[] positions = new int[b];
            for (int i = 0; i < b; i++)
            {
                ids[i] = i == 0 ? anchorToken : cfg.MaskTokenId;
                positions[i] = position + i;
            }

            // The selector's lattice comes back instead of an argmax: k*k floats per
            // transition plus one k-wide row for the anchor's own position, which is
            // ~7 KB against the 12.9 MB a [vocab, b] readback would cost. The walk
            // itself is gamma steps over k candidates and belongs on the host.
            int gamma = b - 1;
            int k = cfg.SelectorTopK;
            if (cfg.HasSelector)
            {
                long need = (long)k + (long)k * k * Math.Max(0, gamma - 1);
                if (_dflashSelScores == null || _dflashSelScores.LongLength < need)
                    _dflashSelScores = new float[need];
                if (_dflashSelCand == null || _dflashSelCand.Length < gamma * k)
                    _dflashSelCand = new int[gamma * k];
            }

            bool ok = GgmlBasicOps.DFlashDraftBlock(
                ids, b, positions,
                cfg.NumLayers, cfg.HiddenSize, cfg.HeadDim, cfg.NumHeads, cfg.NumKVHeads, _dflashRingRows,
                cfg.Eps, cfg.RopeBase, 1f, 1f / MathF.Sqrt(cfg.HeadDim),
                BuildRingSlotPositions(), cfg.SlidingWindow,
                a.AttnNorm,
                a.Q, a.QType, a.QNe0, a.QNe1, a.QBytes,
                a.K, a.KType, a.KNe0, a.KNe1, a.KBytes,
                a.V, a.VType, a.VNe0, a.VNe1, a.VBytes,
                a.QNorm, a.KNorm,
                a.O, a.OType, a.ONe0, a.ONe1, a.OBytes,
                a.FfnNorm,
                a.Gate, a.GateType, a.GateNe0, a.GateNe1, a.GateBytes,
                a.Up, a.UpType, a.UpNe0, a.UpNe1, a.UpBytes,
                a.Down, a.DownType, a.DownNe0, a.DownNe1, a.DownBytes,
                a.RingK, a.RingV, GgmlTypeF32,
                a.OutNorm,
                a.TokEmbd, a.TokEmbdType, a.TokEmbdNe0, a.TokEmbdNe1, a.TokEmbdBytes,
                a.LmHead, a.LmHeadType, a.LmHeadNe0, a.LmHeadNe1, a.LmHeadBytes,
                Config.VocabSize, _dflashDraftIds, _dflashDraftConf,
                cfg.HasConv ? cfg.ConvKernelSize : 0, cfg.ConvGroupSize, cfg.ConvNumGroups,
                a.AttnConvBase, a.AttnConvProj, a.AttnConvProjType, a.AttnConvProjNe0, a.AttnConvProjNe1, a.AttnConvProjBytes,
                a.FfnConvBase, a.FfnConvProj, a.FfnConvProjType, a.FfnConvProjNe0, a.FfnConvProjNe1, a.FfnConvProjBytes,
                cfg.HasSelector ? cfg.SelectorRank : 0, cfg.HasSelector ? cfg.SelectorTopK : 0,
                cfg.LogitScale, cfg.FinalLogitSoftcap,
                a.SelHidden, a.SelHiddenType, a.SelHiddenNe0, a.SelHiddenNe1, a.SelHiddenBytes,
                a.SelPred, a.SelPredType, a.SelPredNe0, a.SelPredNe1, a.SelPredBytes,
                a.SelSucc, a.SelSuccType, a.SelSuccNe0, a.SelSuccNe1, a.SelSuccBytes,
                cfg.HasSelector ? _dflashSelScores : null, cfg.HasSelector ? _dflashSelCand : null);
            if (!ok)
                return -1;

            if (cfg.HasSelector)
                return DFlashWalkLattice(gamma, k, _dflashSelScores, _dflashSelCand, draftOut, confOut);

            // Row 0 is the anchor's own prediction; plain DFlash discards it.
            int drafted = b - 1;
            for (int i = 0; i < drafted; i++)
            {
                draftOut[i] = _dflashDraftIds[i + 1];
                if (confOut != null && i < confOut.Length)
                    confOut[i] = _dflashDraftConf[i + 1];
            }
            return drafted;
        }

        private float[] _dflashSelScores;
        private int[] _dflashSelCand;

        /// <summary>
        /// The greedy walk through the transition lattice the kernel produced.
        /// <paramref name="scores"/> holds the anchor row first (k floats: position
        /// 0's scores against the verified anchor) and then one [k(pred), k(cand)]
        /// matrix per following position, candidate-fastest. Each step takes the
        /// argmax over candidates of the row selected by the previous step's choice,
        /// which is exactly what the reference implementation's temperature-0 path
        /// does.
        /// </summary>
        internal static int DFlashWalkLattice(int gamma, int k, float[] scores, int[] cand, int[] draftOut, float[] confOut)
        {
            float[] row = new float[k];

            int chosen = 0;
            for (int e = 0; e < gamma; e++)
            {
                long baseIdx = e == 0
                    ? 0
                    : (long)k + ((long)(e - 1) * k + chosen) * k;
                Array.Copy(scores, baseIdx, row, 0, k);

                chosen = 0;
                for (int c = 1; c < k; c++)
                    if (row[c] > row[chosen]) chosen = c;

                draftOut[e] = cand[e * k + chosen];
                if (confOut != null && e < confOut.Length)
                    confOut[e] = DFlashSoftmaxAt(row, chosen);
            }
            return gamma;
        }
    }
}
