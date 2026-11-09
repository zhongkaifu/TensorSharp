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
using System.Collections.Generic;
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

            // Native packed scale table: fc/embedding/head/selector H/P/S,
            // then Q/K/V/O/gate/up/down/attention-conv/FFN-conv per layer.
            // Null preserves the existing unscaled ABI and graph.
            public float[] Scales;

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

        /// <summary>First position of the contiguous run of rows the ring holds for
        /// its current sequence; rows below it (never written, or written for another
        /// sequence) are not context. See <see cref="NoteDFlashRingWrite"/>.</summary>
        private int _dflashRingValidFrom;

        /// <summary>Set when the fused kernels have written the ring on the device, so
        /// the host mirror is behind.</summary>
        private bool _dflashRingHostStale;

        /// <summary>
        /// Pull device-side ring writes back to the host copy. Only the per-op drafter
        /// walks the ring with host pointers, so this is a no-op on the fused path.
        /// </summary>
        private void EnsureDFlashRingHostSynchronized()
        {
            FlushDeferredDFlashInjects();
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
            _hasDFlash && !IsTensorParallel &&
            (_backend == BackendType.GgmlCuda || _backend == BackendType.GgmlVulkan ||
             _backend == BackendType.GgmlMetal ||
             // DSpine has no per-op draft; its inject runs fused on ggml_cpu too.
             (_dflash.IsDSpine && _backend == BackendType.GgmlCpu));

        /// <summary>A quantized weight the fused kernels can bind by cache key.
        /// (ModelBase has no such helper; the trunk kernels each roll their own.)
        /// A DSpine drafter's BF16/F16 linears are bound in their file encoding.</summary>
        private bool TryDFlashQuant(string name, out QuantizedWeight qw)
            => (_quantWeights.TryGetValue(name, out qw) && qw != null && qw.CacheKey != IntPtr.Zero)
               || TryDSpineRaw(name, out qw);

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
                Scales = new float[6 + n * 9],
            };
            Array.Fill(a.Scales, 1f);
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

                int scaleOffset = 6 + l * 9;
                a.Scales[scaleOffset] = q.Scale;
                a.Scales[scaleOffset + 1] = k.Scale;
                a.Scales[scaleOffset + 2] = v.Scale;
                a.Scales[scaleOffset + 3] = o.Scale;
                a.Scales[scaleOffset + 4] = gate.Scale;
                a.Scales[scaleOffset + 5] = up.Scale;
                a.Scales[scaleOffset + 6] = down.Scale;

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
                a.Scales[scaleOffset + 7] = acp.Scale;
                a.Scales[scaleOffset + 8] = fcp.Scale;
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
            a.Scales[0] = fc.Scale;
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
            a.Scales[1] = tok.Scale;
            a.Scales[2] = head.Scale;
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
                a.Scales[3] = selH.Scale;
                a.Scales[4] = selP.Scale;
                a.Scales[5] = selS.Scale;
                a.SelHidden = selH.CacheKey; a.SelHiddenType = selH.GgmlType;
                a.SelHiddenNe0 = selH.Ne0; a.SelHiddenNe1 = selH.Ne1; a.SelHiddenBytes = selH.RawBytes;
                a.SelPred = selP.CacheKey; a.SelPredType = selP.GgmlType;
                a.SelPredNe0 = selP.Ne0; a.SelPredNe1 = selP.Ne1; a.SelPredBytes = selP.RawBytes;
                a.SelSucc = selS.CacheKey; a.SelSuccType = selS.GgmlType;
                a.SelSuccNe0 = selS.Ne0; a.SelSuccNe1 = selS.Ne1; a.SelSuccBytes = selS.RawBytes;
            }

            ApplyDraftVocabSubset(a, cfg);

            if (Array.TrueForAll(a.Scales, scale => scale == 1f))
                a.Scales = null;
            _dflashArrays = a;
            Console.WriteLine($"  {(cfg.IsDSpine ? "DSpine" : "DFlash")} fused drafter armed ({n} draft layers, ring {a.RingRows} rows, "
                + (cfg.IsDSpine ? "block-causal draft with the code bridge, on-device refinement lattice"
                    : cfg.HasSelector ? "on-device lattice" : "on-device top-1")
                + (cfg.HasConv ? $", conv {cfg.ConvKernelSize}x{cfg.ConvGroupSize}" : string.Empty) + ").");
        }

        // ---- Draft vocabulary subset (opt-in) ------------------------------------------------
        //
        // TS_DFLASH_DRAFT_VOCAB=<file>: the fused DFlash/DFlash2 draft scores only the listed tokens, as Strata's
        // drafter does (its data/draft_vocab*.bin: little-endian int32 token ids, ascending; Strata ships a 106K
        // default and 40K English, Cyrillic and French sets). The target's LM-head rows and the selector's
        // per-token tables are gathered into subset order once, the graph runs over the smaller vocabulary, and the
        // drafted ids are mapped back on the host. Output stays exact (the target verifies every token); a token
        // outside the subset can never be drafted, which costs acceptance where the subset under-covers the text.
        // DSpine keeps its own head. Off by default.
        private int[] _dflashVocabMap;
        private IntPtr _dflashSubHead, _dflashSubPred, _dflashSubSucc;
        private IntPtr _dflashSubSource;          // the full head the subset was gathered from
        private long _dflashSubHeadBytes, _dflashSubPredBytes, _dflashSubSuccBytes;

        /// <summary>The vocabulary the fused draft graph scores: the subset's size, or the target's.</summary>
        private int DFlashDraftVocabSize => _dflashVocabMap?.Length ?? Config.VocabSize;

        private void MapDraftVocab(int[] ids, int count) => DraftVocabSubset.MapToTokens(_dflashVocabMap, ids, count);

        private void ApplyDraftVocabSubset(DFlashArrays a, DFlashConfig cfg)
        {
            string path = Environment.GetEnvironmentVariable("TS_DFLASH_DRAFT_VOCAB");
            if (string.IsNullOrEmpty(path))
                return;
            if (cfg.IsDSpine)
            {
                Console.WriteLine("  TS_DFLASH_DRAFT_VOCAB ignored: a DSpine drafter scores its own head over the full vocabulary.");
                return;
            }
            long vocab = a.LmHeadNe1;
            if (_dflashVocabMap == null || _dflashSubSource != a.LmHead)
            {
                FreeDraftVocabSubset();
                if (!TryBuildDraftVocabSubset(a, cfg, path, vocab))
                    return;
            }
            a.LmHead = _dflashSubHead; a.LmHeadNe1 = _dflashVocabMap.Length; a.LmHeadBytes = _dflashSubHeadBytes;
            if (cfg.HasSelector)
            {
                a.SelPred = _dflashSubPred; a.SelPredNe1 = _dflashVocabMap.Length; a.SelPredBytes = _dflashSubPredBytes;
                a.SelSucc = _dflashSubSucc; a.SelSuccNe1 = _dflashVocabMap.Length; a.SelSuccBytes = _dflashSubSuccBytes;
            }
        }

        private bool TryBuildDraftVocabSubset(DFlashArrays a, DFlashConfig cfg, string path, long vocab)
        {
            int[] map = DraftVocabSubset.Load(path, vocab, cfg.HasSelector ? cfg.SelectorTopK : 1, out string problem);
            if (problem == null && a.LmHeadBytes % vocab != 0) problem = "the LM head's rows are not whole";
            if (problem == null && cfg.HasSelector
                && (a.SelPredNe1 != vocab || a.SelSuccNe1 != vocab || a.SelPredBytes % vocab != 0 || a.SelSuccBytes % vocab != 0))
                problem = "the selector tables are not one row per token";
            if (problem != null)
            {
                Console.WriteLine($"  TS_DFLASH_DRAFT_VOCAB={path} not used: {problem}; the draft scores the full vocabulary.");
                return false;
            }
            IntPtr source = a.LmHead;
            _dflashSubHead = DraftVocabSubset.Gather(a.LmHead, a.LmHeadBytes / vocab, map, out _dflashSubHeadBytes);
            if (cfg.HasSelector)
            {
                _dflashSubPred = DraftVocabSubset.Gather(a.SelPred, a.SelPredBytes / vocab, map, out _dflashSubPredBytes);
                _dflashSubSucc = DraftVocabSubset.Gather(a.SelSucc, a.SelSuccBytes / vocab, map, out _dflashSubSuccBytes);
            }
            // Resident device copies, as the model's own weights have (WeightLoading's preload): bound by their
            // host pointer alone, the rows were uploaded again on every draft (+3 ms per 8-row draft on the Mac).
            try
            {
                bool resident = GgmlBasicOps.PreloadQuantizedWeight(_dflashSubHead, _dflashSubHead, a.LmHeadType,
                    a.LmHeadNe0, map.Length, _dflashSubHeadBytes);
                if (resident && cfg.HasSelector)
                    resident = GgmlBasicOps.PreloadQuantizedWeight(_dflashSubPred, _dflashSubPred, a.SelPredType,
                            a.SelPredNe0, map.Length, _dflashSubPredBytes)
                        && GgmlBasicOps.PreloadQuantizedWeight(_dflashSubSucc, _dflashSubSucc, a.SelSuccType,
                            a.SelSuccNe0, map.Length, _dflashSubSuccBytes);
                if (!resident)
                    problem = "its rows do not fit one device buffer";
            }
            catch (Exception ex) { problem = "its rows could not be placed on the device: " + ex.Message; }
            if (problem != null)
            {
                FreeDraftVocabSubset();
                Console.WriteLine($"  TS_DFLASH_DRAFT_VOCAB={path} not used: {problem}; the draft scores the full vocabulary.");
                return false;
            }
            _dflashVocabMap = map;
            _dflashSubSource = source;
            double mib = (_dflashSubHeadBytes + _dflashSubPredBytes + _dflashSubSuccBytes) / 1048576.0;
            Console.WriteLine($"  DFlash draft vocabulary: {map.Length} of {vocab} tokens from {path} ({mib:F1} MiB of head and "
                + "selector rows); tokens outside it are never drafted (TS_DFLASH_DRAFT_VOCAB).");
            return true;
        }

        private void FreeDraftVocabSubset()
        {
            foreach (var (p, bytes) in new[] { (_dflashSubHead, _dflashSubHeadBytes), (_dflashSubPred, _dflashSubPredBytes),
                         (_dflashSubSucc, _dflashSubSuccBytes) })
            {
                if (p == IntPtr.Zero) continue;
                GgmlBasicOps.InvalidateHostBuffer(p);
                DraftVocabSubset.Free(p, bytes);
            }
            _dflashSubHead = _dflashSubPred = _dflashSubSucc = IntPtr.Zero;
            _dflashSubHeadBytes = _dflashSubPredBytes = _dflashSubSuccBytes = 0;
            _dflashVocabMap = null;
            _dflashSubSource = IntPtr.Zero;
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
            if (_dspineState != null)
                GgmlBasicOps.DSpineResetCaches();
            _dflashRingFilled = 0;
            _dflashRingValidFrom = 0;
        }

        /// <summary>
        /// Record a ring write of <paramref name="n"/> rows from <paramref name="startPos"/>.
        /// The frontier is an ASSIGNMENT (see DFlashInjectKv). The valid run restarts at
        /// a write that jumps past the frontier or lands below the run: a fused holder
        /// arms mid-sequence with an empty ring, and the slots under its first write's
        /// frontier hold zeros or another sequence's rows at colliding positions, which
        /// the slot map used to label as live context.
        /// </summary>
        private void NoteDFlashRingWrite(int startPos, int n)
        {
            if (startPos > _dflashRingFilled || startPos < _dflashRingValidFrom)
                _dflashRingValidFrom = startPos;
            _dflashRingFilled = startPos + n;
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
            int lo = Math.Max(Math.Max(0, _dflashRingFilled - rows), _dflashRingValidFrom);
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

            // The ring drops rows older than its capacity, so only the last min(n, rows)
            // survive; writing the earlier ones would just be overwritten in place.
            int keep = Math.Min(n, _dflashRingRows);
            int skip = n - keep;

            if (_dflashDeferInjects)
                DeferDFlashInject(hRows, rowOffset + skip, keep, startPos + skip, a);
            else if (!RunFusedDFlashInject(a, hRows, rowOffset + skip, keep, startPos + skip, a.RingK, a.RingV))
                return false;

            // See DFlashInjectKv: the frontier must be able to move backwards.
            NoteDFlashRingWrite(startPos + skip, keep);
            _dflashRingHostStale = true;
            return true;
        }

        private bool RunFusedDFlashInject(DFlashArrays a, float[] hRows, int rowOffset, int keep, int firstPos,
            IntPtr[] ringK, IntPtr[] ringV)
        {
            var cfg = _dflash;
            int feat = cfg.FeatureSize;
            float[] rows = hRows;
            if (rowOffset != 0)
            {
                rows = new float[(long)keep * feat];
                Array.Copy(hRows, (long)rowOffset * feat, rows, 0, (long)keep * feat);
            }

            if (_dflashInjectIdx == null || _dflashInjectIdx.Length < keep)
            {
                _dflashInjectIdx = new long[keep];
                _dflashInjectPos = new int[keep];
            }
            for (int i = 0; i < keep; i++)
            {
                int p = firstPos + i;
                _dflashInjectIdx[i] = p % _dflashRingRows;
                _dflashInjectPos[i] = p;
            }

            return GgmlBasicOps.DFlashInject(
                rows, feat, keep, _dflashInjectIdx, _dflashInjectPos,
                cfg.NumLayers, cfg.HiddenSize, cfg.HeadDim, cfg.NumKVHeads, _dflashRingRows,
                cfg.Eps, cfg.RopeBase, 1f,
                a.Fc, a.FcType, a.FcNe0, a.FcNe1, a.FcBytes, a.EncNorm,
                a.K, a.KType, a.KNe0, a.KNe1, a.KBytes,
                a.V, a.VType, a.VNe0, a.VNe1, a.VBytes,
                a.KNorm, ringK, ringV, GgmlTypeF32, a.Scales);
        }

        // ---- Deferred ring injects (batched speculation) ----

        /// <summary>A ring write waiting for <see cref="FlushDeferredDFlashInjects"/>: rows
        /// [FirstRow, FirstRow + Count) of the pending feature buffer, at positions from
        /// StartPos, into the ring whose layer pointers were bound when it was recorded.</summary>
        private readonly struct PendingDFlashInject
        {
            public PendingDFlashInject(int firstRow, int count, int startPos, IntPtr[] ringK, IntPtr[] ringV)
            {
                FirstRow = firstRow; Count = count; StartPos = startPos; RingK = ringK; RingV = ringV;
            }
            public int FirstRow { get; }
            public int Count { get; }
            public int StartPos { get; }
            public IntPtr[] RingK { get; }
            public IntPtr[] RingV { get; }
        }

        private bool _dflashDeferInjects;
        private readonly List<PendingDFlashInject> _dflashPendingInjects = new();
        private float[] _dflashPendingRows = Array.Empty<float>();
        private int _dflashPendingRowCount;
        private bool _dflashDeferredInjectFailureLogged;

        /// <summary>
        /// Queue fused ring injects until <see cref="EndDeferredDFlashInjects"/> (or the next
        /// draft): a batched speculative step injects every lane's accepted rows at commit
        /// and every lane's carried row before its draft, and one graph for all lanes
        /// replaces a launch and a stream sync per lane. The ring bookkeeping (frontier,
        /// valid run) still happens at each call; only the device write waits. CUDA only.
        /// </summary>
        private protected void BeginDeferredDFlashInjects()
        {
            if (s_dflashBatchedInject && _hasDFlash && _backend == BackendType.GgmlCuda)
                _dflashDeferInjects = true;
        }

        // TS_DFLASH_BATCHED_INJECT=0 injects lane by lane, as before.
        private static readonly bool s_dflashBatchedInject =
            Environment.GetEnvironmentVariable("TS_DFLASH_BATCHED_INJECT") != "0";

        private protected void EndDeferredDFlashInjects()
        {
            _dflashDeferInjects = false;
            FlushDeferredDFlashInjects();
        }

        private void DeferDFlashInject(float[] hRows, int rowOffset, int keep, int firstPos, DFlashArrays a)
        {
            int feat = _dflash.FeatureSize;
            long need = (long)(_dflashPendingRowCount + keep) * feat;
            if (_dflashPendingRows.LongLength < need)
                Array.Resize(ref _dflashPendingRows, (int)Math.Max(need, 2L * _dflashPendingRows.LongLength));
            Array.Copy(hRows, (long)rowOffset * feat, _dflashPendingRows, (long)_dflashPendingRowCount * feat, (long)keep * feat);
            _dflashPendingInjects.Add(new PendingDFlashInject(_dflashPendingRowCount, keep, firstPos,
                (IntPtr[])a.RingK.Clone(), (IntPtr[])a.RingV.Clone()));
            _dflashPendingRowCount += keep;
        }

        /// <summary>Write every queued inject: one graph over all of them on CUDA, else (one
        /// queued, or the batched graph declined) one inject each. Called before anything
        /// reads a ring.</summary>
        private void FlushDeferredDFlashInjects()
        {
            int lanes = _dflashPendingInjects.Count;
            if (lanes == 0)
                return;
            try
            {
                var a = GetDFlashArrays();
                var cfg = _dflash;
                int nl = cfg.NumLayers;
                if (lanes > 1)
                {
                    int maxRows = 0;
                    foreach (var p in _dflashPendingInjects)
                        maxRows = Math.Max(maxRows, p.Count);
                    int padded = lanes * maxRows;
                    var gather = new int[padded];
                    var idx = new long[padded];
                    var pos = new int[padded];
                    var ringK = new IntPtr[lanes * nl];
                    var ringV = new IntPtr[lanes * nl];
                    for (int lane = 0; lane < lanes; lane++)
                    {
                        var p = _dflashPendingInjects[lane];
                        for (int r = 0; r < maxRows; r++)
                        {
                            // Padding repeats the lane's last row: same bytes into the same slot.
                            int i = Math.Min(r, p.Count - 1);
                            gather[lane * maxRows + r] = p.FirstRow + i;
                            idx[lane * maxRows + r] = (p.StartPos + i) % _dflashRingRows;
                            pos[lane * maxRows + r] = p.StartPos + i;
                        }
                        Array.Copy(p.RingK, 0, ringK, lane * nl, nl);
                        Array.Copy(p.RingV, 0, ringV, lane * nl, nl);
                    }
                    if (GgmlBasicOps.DFlashInjectBatched(lanes, maxRows, _dflashPendingRowCount,
                            _dflashPendingRows, cfg.FeatureSize, gather, idx, pos,
                            nl, cfg.HiddenSize, cfg.HeadDim, cfg.NumKVHeads, _dflashRingRows,
                            cfg.Eps, cfg.RopeBase, 1f,
                            a.Fc, a.FcType, a.FcNe0, a.FcNe1, a.FcBytes, a.EncNorm,
                            a.K, a.KType, a.KNe0, a.KNe1, a.KBytes,
                            a.V, a.VType, a.VNe0, a.VNe1, a.VBytes,
                            a.KNorm, ringK, ringV, GgmlTypeF32, a.Scales))
                        return;
                }
                foreach (var p in _dflashPendingInjects)
                {
                    if (RunFusedDFlashInject(a, _dflashPendingRows, p.FirstRow, p.Count, p.StartPos, p.RingK, p.RingV)
                        || _dflashDeferredInjectFailureLogged)
                        continue;
                    // The ring keeps stale rows where these belong: drafts get worse until
                    // they are overwritten, verification keeps the output exact.
                    _dflashDeferredInjectFailureLogged = true;
                    Console.Error.WriteLine("[dflash] deferred ring inject failed; drafting continues on stale rows: "
                        + GgmlBasicOps.LastNativeError());
                }
            }
            finally
            {
                _dflashPendingInjects.Clear();
                _dflashPendingRowCount = 0;
            }
        }

        /// <summary>
        /// PASS C in one graph: [anchor, MASK x (b-1)] through the draft blocks, then
        /// the target's LM head, softmax, and an on-device argmax. Returns the number
        /// of drafted tokens, or -1 when the fused path is unavailable.
        /// </summary>
        private int TryFusedDFlashDraftBlock(int anchorToken, int position, int b, int[] draftOut, float[] confOut)
        {
            // DSpine has its own graph and no per-op fallback (never -1).
            if (_dflash.IsDSpine)
                return TryFusedDSpineDraftBlock(anchorToken, position, b, draftOut, confOut);

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
                DFlashDraftVocabSize, _dflashDraftIds, _dflashDraftConf,
                cfg.HasConv ? cfg.ConvKernelSize : 0, cfg.ConvGroupSize, cfg.ConvNumGroups,
                a.AttnConvBase, a.AttnConvProj, a.AttnConvProjType, a.AttnConvProjNe0, a.AttnConvProjNe1, a.AttnConvProjBytes,
                a.FfnConvBase, a.FfnConvProj, a.FfnConvProjType, a.FfnConvProjNe0, a.FfnConvProjNe1, a.FfnConvProjBytes,
                cfg.HasSelector ? cfg.SelectorRank : 0, cfg.HasSelector ? cfg.SelectorTopK : 0,
                cfg.LogitScale, cfg.FinalLogitSoftcap,
                a.SelHidden, a.SelHiddenType, a.SelHiddenNe0, a.SelHiddenNe1, a.SelHiddenBytes,
                a.SelPred, a.SelPredType, a.SelPredNe0, a.SelPredNe1, a.SelPredBytes,
                a.SelSucc, a.SelSuccType, a.SelSuccNe0, a.SelSuccNe1, a.SelSuccBytes,
                cfg.HasSelector ? _dflashSelScores : null, cfg.HasSelector ? _dflashSelCand : null, a.Scales);
            if (!ok)
                return -1;

            if (cfg.HasSelector)
            {
                MapDraftVocab(_dflashSelCand, gamma * k);
                return DFlashWalkLattice(gamma, k, _dflashSelScores, _dflashSelCand, draftOut, confOut);
            }
            MapDraftVocab(_dflashDraftIds, b);

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
        /// One block draft for each of several requests in a single graph, every
        /// request on its own ring (see <see cref="TryFusedDFlashDraftBlock"/> for the
        /// single-request pass this mirrors row for row). <paramref name="hPrev"/>[i]
        /// is request i's carried feature row and <paramref name="positions"/>[i] its
        /// anchor position. Fills each lane's full proposal block (b - 1 tokens) and
        /// returns false, with nothing drafted, when the batched graph is unavailable.
        /// </summary>
        private protected bool TryFusedDFlashDraftBatch(IReadOnlyList<string> requestIds, int[] anchors,
            float[][] hPrev, int[] positions, int[][] draftsOut, float[][] confOut)
        {
            var a = GetDFlashArrays();
            var cfg = _dflash;
            // DSpine drafts lane by lane (no batched DSpine graph yet).
            if (a == null || cfg.MarkovRank > 0 || cfg.HasAttentionSinks || cfg.IsDSpine || _backend != BackendType.GgmlCuda)
                return false;
            int lanes = requestIds.Count;
            int b = Math.Min(cfg.BlockSize, DFlashBatchDraftWidth + 1);
            int rows = _dflashRingRows;
            int nl = cfg.NumLayers;
            var ids = new int[lanes * b];
            var pos = new int[lanes * b];
            var slots = new int[lanes * rows];
            var ringK = new IntPtr[lanes * nl];
            var ringV = new IntPtr[lanes * nl];
            for (int lane = 0; lane < lanes; lane++)
            {
                if (positions[lane] <= 0)
                    return false;
                BindDFlashRing(requestIds[lane]);
                // The drafter's own key for the last committed position, as in DFlashPropose.
                DFlashEncodeAndInject(hPrev[lane], 0, 1, positions[lane] - 1);
                Array.Copy(BuildRingSlotPositions(), 0, slots, (long)lane * rows, rows);
                for (int i = 0; i < b; i++)
                {
                    ids[lane * b + i] = i == 0 ? anchors[lane] : cfg.MaskTokenId;
                    pos[lane * b + i] = positions[lane] + i;
                }
                for (int l = 0; l < nl; l++)
                {
                    ringK[lane * nl + l] = TensorComputePrimitives.GetStoragePointer(_dflashRingK[l]);
                    ringV[lane * nl + l] = TensorComputePrimitives.GetStoragePointer(_dflashRingV[l]);
                }
            }

            // The lanes' carried rows were queued above; write them in one graph.
            FlushDeferredDFlashInjects();

            int gamma = b - 1;
            int k = cfg.SelectorTopK;
            long laneScores = cfg.HasSelector ? (long)k + (long)k * k * Math.Max(0, gamma - 1) : 0;
            float[] scores = cfg.HasSelector ? new float[lanes * laneScores] : null;
            int[] cand = cfg.HasSelector ? new int[lanes * gamma * k] : null;
            var idsOut = new int[lanes * b];
            var conf = new float[lanes * b];
            bool ok = GgmlBasicOps.DFlashDraftBlockBatched(
                lanes, ids, b, pos,
                cfg.NumLayers, cfg.HiddenSize, cfg.HeadDim, cfg.NumHeads, cfg.NumKVHeads, rows,
                cfg.Eps, cfg.RopeBase, 1f, 1f / MathF.Sqrt(cfg.HeadDim),
                slots, cfg.SlidingWindow,
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
                ringK, ringV, GgmlTypeF32,
                a.OutNorm,
                a.TokEmbd, a.TokEmbdType, a.TokEmbdNe0, a.TokEmbdNe1, a.TokEmbdBytes,
                a.LmHead, a.LmHeadType, a.LmHeadNe0, a.LmHeadNe1, a.LmHeadBytes,
                DFlashDraftVocabSize, idsOut, conf,
                cfg.HasConv ? cfg.ConvKernelSize : 0, cfg.ConvGroupSize, cfg.ConvNumGroups,
                a.AttnConvBase, a.AttnConvProj, a.AttnConvProjType, a.AttnConvProjNe0, a.AttnConvProjNe1, a.AttnConvProjBytes,
                a.FfnConvBase, a.FfnConvProj, a.FfnConvProjType, a.FfnConvProjNe0, a.FfnConvProjNe1, a.FfnConvProjBytes,
                cfg.HasSelector ? cfg.SelectorRank : 0, cfg.HasSelector ? cfg.SelectorTopK : 0,
                cfg.LogitScale, cfg.FinalLogitSoftcap,
                a.SelHidden, a.SelHiddenType, a.SelHiddenNe0, a.SelHiddenNe1, a.SelHiddenBytes,
                a.SelPred, a.SelPredType, a.SelPredNe0, a.SelPredNe1, a.SelPredBytes,
                a.SelSucc, a.SelSuccType, a.SelSuccNe0, a.SelSuccNe1, a.SelSuccBytes,
                scores, cand, a.Scales);
            if (!ok)
                return false;
            if (cfg.HasSelector) MapDraftVocab(cand, lanes * gamma * k);
            else MapDraftVocab(idsOut, lanes * b);

            for (int lane = 0; lane < lanes; lane++)
            {
                if (cfg.HasSelector)
                {
                    var laneScoreSlice = new float[laneScores];
                    var laneCand = new int[gamma * k];
                    Array.Copy(scores, lane * laneScores, laneScoreSlice, 0, laneScores);
                    Array.Copy(cand, lane * gamma * k, laneCand, 0, gamma * k);
                    DFlashWalkLattice(gamma, k, laneScoreSlice, laneCand, draftsOut[lane], confOut[lane]);
                }
                else
                {
                    for (int i = 0; i < gamma; i++)
                    {
                        draftsOut[lane][i] = idsOut[lane * b + i + 1];
                        confOut[lane][i] = conf[lane * b + i + 1];
                    }
                }
            }
            return true;
        }

        /// <summary>Drafts a batched block proposes per lane: the trained block minus
        /// its anchor row, as the single-request draft always runs.</summary>
        private protected int DFlashBatchDraftWidth => _dflash.MaxDraftTokens;

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
