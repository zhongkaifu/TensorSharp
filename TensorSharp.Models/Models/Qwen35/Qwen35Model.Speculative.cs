// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
//
// Qwen3.5/3.6 NextN/MTP (multi-token prediction) draft head.
//
// Qwen3.6 GGUFs ship one extra decoder block past the main stack (blk.N where
// N == trunk layer count) flagged by `{arch}.nextn_predict_layers`. The block
// is a standard full-attention Qwen3.5 decoder block (dense FFN on 27B, MoE
// FFN on 35B-A3B) plus four NextN-specific tensors:
//   nextn.eh_proj          [2*hidden, hidden]  input projection
//   nextn.enorm            [hidden]            RMS norm over the token embedding
//   nextn.hnorm            [hidden]            RMS norm over the trunk hidden state
//   nextn.shared_head_norm [hidden]            final norm before the LM head
// (nextn.embed_tokens / nextn.shared_head_head are optional and absent in the
// stock GGUFs; we fall back to the trunk token embedding / LM head.)
//
// The MTP step consumes (token x_p, trunk hidden h_{p-1}) at position p and
// produces logits predicting x_{p+1} plus its own hidden state used to chain
// further draft steps. This mirrors llama.cpp's graph_mtp (src/models/qwen35.cpp)
// and vLLM's Qwen3_5MultiTokenPredictor (qwen3_5_mtp.py).
using System;
using System.Collections.Generic;
using System.Diagnostics;
using TensorSharp;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;

using TensorSharp.Runtime.Speculative;

namespace TensorSharp.Models
{
    // IBatchedSpeculativeModel (extends ISpeculativeModel) is the
    // Runtime-side contract BatchExecutor drives for engine-path speculation;
    // every member is implemented below or inherited from ModelBase
    // (CacheSeqLen, MaxContextLength).
    public partial class Qwen35Model : IBatchedSpeculativeModel
    {
        // NextN/MTP weights (cached once at load; null when the GGUF has no MTP block).
        private QuantizedWeight _mtpEhProjQW;
        private Tensor _mtpEhProjF32;
        private Tensor _mtpEnormW;
        private Tensor _mtpHnormW;
        private Tensor _mtpHeadNormW;       // nextn.shared_head_norm (falls back to output_norm)
        private QuantizedWeight _mtpEmbdQW; // optional nextn.embed_tokens
        private Tensor _mtpEmbdF32;
        private QuantizedWeight _mtpHeadQW; // optional nextn.shared_head_head
        private Tensor _mtpHeadF32;

        // Recurrent-state snapshot used to roll the trunk back when a verify
        // batch is partially rejected (GDN state cannot be truncated in place).
        private byte[][] _mtpGdnSnapshot;

        /// <summary>
        /// True when the loaded GGUF contains a usable NextN/MTP draft block.
        /// </summary>
        private bool HasMtpDraftHead { get; set; }

        /// <summary>
        /// True when SOME learned drafter is attached: the trunk's own NextN/MTP
        /// block, or an external DFlash/DFlash2 file. They are alternatives, not
        /// layers - see <see cref="DraftHeadKind"/>.
        /// </summary>
        public bool HasDraftHead => HasMtpDraftHead || HasDFlash;

        /// <summary>Qwen 3.6's NextN block drafts one token per pass, so it is
        /// served by <see cref="DraftHeadSpeculator"/>; a DFlash drafter proposes a
        /// whole block per pass and is served by
        /// <see cref="BlockDraftSpeculator"/>. An attached DFlash file wins, because
        /// the operator named it explicitly and the two consume different hidden
        /// rows.</summary>
        public DraftHeadKind DraftHeadKind => HasDFlash
            ? DraftHeadKind.Block
            : (HasMtpDraftHead ? DraftHeadKind.PerToken : DraftHeadKind.None);

        /// <summary>The drafter's hidden row: the trunk's own hidden size for MTP,
        /// and the concatenated dflash.target_layers residuals for DFlash.</summary>
        public int SpecFeatureSize => HasDFlash ? _dflash.FeatureSize : Config.HiddenSize;

        /// <summary>DFlash prefills the drafter's ring from the trunk's per-row
        /// features, so it wants whole micro-batches; MTP has no preference.</summary>
        public int SpecPrefillChunkSize => HasDFlash ? DFlashPrefillChunkSize : 0;

        /// <summary>
        /// Three, not the shared default of eight, whenever this checkpoint has
        /// GatedDeltaNet layers - which every Qwen 3.5/3.6/3.8 hybrid does.
        ///
        /// A wide window is priced by the trunk here, not by the drafter: a verify
        /// over N rows runs the GDN chunked scan rather than the single-token
        /// recurrent update, and a partial rejection has to restore the recurrent
        /// state (there is no per-row checkpoint to rewind to) and re-advance over
        /// the accepted prefix - a second whole-trunk forward, plus the state moving
        /// across PCIe twice. Both costs grow with the window while acceptance per
        /// position falls, so the marginal drafted token stops paying long before it
        /// would on a dense-attention trunk.
        ///
        /// Measured on Qwen3.8-27B-UD-IQ3_XXS (RTX 3080 Laptop, greedy, 256 tokens,
        /// DFlash2 drafter): window 8 -> 9.4 tok/s, 4 -> 13.7, 3 -> 15.5, against
        /// 18.3 plain. The trunk's own NextN/MTP head behaves the same way, so this
        /// is not a property of either drafter.
        ///
        /// An operator who passes --spec-draft still gets exactly that number.
        /// </summary>
        public int SpecPreferredDraftWindow => HasAnyRecurrentLayer ? 3 : 0;

        // Weight-free suffix matches can amortize a wider GEMM without paying
        // twelve learned draft-head forwards. IQ4_XS uses independent Metal
        // matvecs through eight verify rows; a twelve-token n-gram draft makes
        // the thirteen-row GEMM useful. MTP/DFlash keep the existing preference.
        public int SpecPreferredNGramDraftWindow
        {
            get
            {
                if (_backend != BackendType.GgmlMetal || _numExperts != 0 || IsTensorParallel || !HasAnyRecurrentLayer)
                    return 0;
                bool tiedOutput = _lmHeadQW != null
                    ? ReferenceEquals(_lmHeadQW, _quantWeights.GetValueOrDefault("token_embd.weight"))
                    : _lmHeadF32 != null && ReferenceEquals(_lmHeadF32, _weights.GetValueOrDefault("token_embd.weight"));
                return SelectMetalNGramDraftWindow(_quantWeights, _weights, Config.NumLayers, tiedOutput);
            }
        }

        internal static int SelectMetalNGramDraftWindow(
            IReadOnlyDictionary<string, QuantizedWeight> quantWeights,
            IReadOnlyDictionary<string, Tensor> weights, int trunkLayers, bool tiedOutput)
        {
            var (iq4Bytes, totalBytes) = MeasureMatmulWeightBytes(
                quantWeights, weights, (int)GgmlTensorType.IQ4_XS, IsActiveMatrix);
            return iq4Bytes > 0 && iq4Bytes >= totalBytes - iq4Bytes ? 12 : 0;

            bool HasWeight(string name) => quantWeights.ContainsKey(name) || weights.ContainsKey(name);
            bool IsActiveMatrix(string name)
            {
                if (name == "output.weight") return !tiedOutput;
                if (name == "token_embd.weight") return tiedOutput;
                if (!name.StartsWith("blk.", StringComparison.Ordinal)) return false;
                int separator = name.IndexOf('.', 4);
                if (separator < 0 || !int.TryParse(name.AsSpan(4, separator - 4), out int layer)
                    || layer < 0 || layer >= trunkLayers)
                    return false;
                string suffix = name[(separator + 1)..];
                string prefix = name[..(separator + 1)];
                // The non-TP verifier reads the four original recurrent input
                // projections, while the retained ssm_in_proj pack serves TP.
                if (suffix is "ssm_in_proj.weight" or "ssm_conv1d.weight"
                    || suffix.StartsWith("nextn.", StringComparison.Ordinal))
                    return false;
                if (suffix is "ffn_gate.weight" or "ffn_up.weight")
                    return !HasWeight(prefix + "ffn_gate_up.weight");
                if (suffix is "attn_q.weight" or "attn_k.weight" or "attn_v.weight")
                    return !HasWeight(prefix + "attn_qkv.weight");
                return true;
            }
        }

        private bool HasAnyRecurrentLayer
        {
            get
            {
                if (_isRecurrent == null)
                    return false;
                foreach (bool r in _isRecurrent)
                    if (r) return true;
                return false;
            }
        }

        /// <summary>Trunk layer count (excludes NextN/MTP blocks).</summary>
        public int NumTrunkLayers => Config.NumLayers;

        private void CacheMtpWeights()
        {
            if (_numNextnLayers <= 0 || _mtpLayerIdx < 0)
                return;

            string p = $"blk.{_mtpLayerIdx}.";
            _quantWeights.TryGetValue(p + "nextn.eh_proj.weight", out _mtpEhProjQW);
            _weights.TryGetValue(p + "nextn.eh_proj.weight", out _mtpEhProjF32);
            _weights.TryGetValue(p + "nextn.enorm.weight", out _mtpEnormW);
            _weights.TryGetValue(p + "nextn.hnorm.weight", out _mtpHnormW);
            _weights.TryGetValue(p + "nextn.shared_head_norm.weight", out _mtpHeadNormW);
            _quantWeights.TryGetValue(p + "nextn.embed_tokens.weight", out _mtpEmbdQW);
            _weights.TryGetValue(p + "nextn.embed_tokens.weight", out _mtpEmbdF32);
            _quantWeights.TryGetValue(p + "nextn.shared_head_head.weight", out _mtpHeadQW);
            _weights.TryGetValue(p + "nextn.shared_head_head.weight", out _mtpHeadF32);

            bool hasProj = _mtpEhProjQW != null || _mtpEhProjF32 != null;
            bool hasAttn = _attnQkvQW[_mtpLayerIdx] != null || _attnQkvF32[_mtpLayerIdx] != null
                || _attnQQW[_mtpLayerIdx] != null || _attnQF32[_mtpLayerIdx] != null;
            if (IsTensorParallel)
            {
                // Under tensor parallelism the fused Q/K/V (or separate Q) projections
                // are moved out of _quantWeights/_weights into per-rank shard tables,
                // so the per-layer cached arrays above stay null even though the MTP
                // attention block is fully sharded and usable. Consult the TP shard
                // tables, keyed by the same weight names the TP forward path uses.
                hasAttn = hasAttn
                    || _tpQuantWeights.ContainsKey(_attnQkvKey[_mtpLayerIdx])
                    || _tpWeights.ContainsKey(_attnQkvKey[_mtpLayerIdx])
                    || _tpQuantWeights.ContainsKey(_attnQKey[_mtpLayerIdx])
                    || _tpWeights.ContainsKey(_attnQKey[_mtpLayerIdx]);
            }
            // A draft block without its own head borrows the trunk LM head. Under
            // GGML tensor parallelism that head is split across ranks, so the
            // cached _lmHeadQW the draft path reads is no longer the head at all
            // (it falls through to token_embd). Refuse the combination rather than
            // draft from the wrong weight; the trunk itself is unaffected.
            bool borrowsSplitHead = _tpLmHeadKey != null && _mtpHeadQW == null && _mtpHeadF32 == null;

            HasMtpDraftHead = _numNextnLayers == 1 && hasProj && _mtpEnormW != null && _mtpHnormW != null
                && hasAttn && _attnNormW[_mtpLayerIdx] != null && _postAttnNormW[_mtpLayerIdx] != null
                && !borrowsSplitHead;

            if (borrowsSplitHead)
                Console.WriteLine("  NextN/MTP block has no own head and the LM head is column-parallel under TP; " +
                    "MTP drafting disabled.");
            else if (_numNextnLayers > 0 && !HasMtpDraftHead)
                Console.WriteLine("  NextN/MTP block present but incomplete; MTP drafting disabled.");
            else if (HasMtpDraftHead)
                Console.WriteLine($"  NextN/MTP draft head ready (layer {_mtpLayerIdx}, " +
                    $"moe={( _isMoeLayer != null && _isMoeLayer[_mtpLayerIdx] ? "yes" : "no")}, " +
                    $"ownHead={(_mtpHeadQW != null || _mtpHeadF32 != null ? "yes" : "no")})");
        }

        /// <summary>
        /// Token embedding lookup for the MTP block: prefers nextn.embed_tokens
        /// when shipped, otherwise reuses the trunk token embedding.
        /// </summary>
        private Tensor MtpEmbedding(int[] tokens)
        {
            if (_mtpEmbdQW != null && _mtpEmbdQW.HasHostData)
            {
                var result = new Tensor(_allocator, DType.Float32, tokens.Length, Config.HiddenSize);
                PopulateQuantizedRows(result, _mtpEmbdQW, tokens);
                return result;
            }
            return Embedding(tokens);
        }

        /// <summary>
        /// Shared MTP core: projects (token, previous trunk hidden) pairs into the
        /// MTP decoder block and runs it (updating the MTP block's KV cache rows at
        /// [startPos, startPos+n)). Returns the block output [n, hidden] BEFORE the
        /// shared head norm.
        /// <paramref name="hRows"/> holds n rows of post-final-norm trunk hidden
        /// states; row k must be the hidden state of the token PRECEDING tokens[k].
        /// </summary>
        private unsafe Tensor MtpForwardCore(int[] tokens, float[] hRows, int startPos)
        {
            int n = tokens.Length;
            EnsureCacheCapacity(startPos + n);
            EnsureKvCacheHostSynchronized();

            Tensor x = MtpProjectInput(tokens, hRows);

            // Full decoder block (attention + FFN/MoE with residuals); reuses the
            // trunk machinery — the MTP layer's weights/KV live at _mtpLayerIdx.
            x = AttentionBlock(x, _mtpLayerIdx, n, startPos);
            return x;
        }

        /// <summary>
        /// Projects (token, hPrev) pairs into the MTP block's input space:
        /// enorm(embed) and hnorm(hPrev) are concatenated and passed through
        /// eh_proj, yielding x [n, hidden] (llama.cpp graph_mtp's eh_proj output).
        /// Shared by the op-by-op and fused draft paths.
        /// </summary>
        private unsafe Tensor MtpProjectInput(int[] tokens, float[] hRows)
        {
            int n = tokens.Length;
            int hidden = Config.HiddenSize;

            Tensor emb = MtpEmbedding(tokens);
            Tensor eNorm = RMSNormOpCached(emb, _mtpEnormW);
            emb.Dispose();

            // hRows may be a reusable buffer larger than n*hidden; copy exactly
            // the rows we need (SetElementsAsFloat would write value.Length
            // elements and overrun the tensor allocation).
            var h = new Tensor(_allocator, DType.Float32, n, hidden);
            fixed (float* src = hRows)
            {
                float* dst = GetFloatPtr(h);
                Buffer.MemoryCopy(src, dst, (long)n * hidden * 4, (long)n * hidden * 4);
            }
            InvalidateTensorDeviceCache(h);
            Tensor hNorm = RMSNormOpCached(h, _mtpHnormW);
            h.Dispose();

            // concat([e_norm, h_norm], featureDim) -> eh_proj -> [n, hidden]
            var cat = new Tensor(_allocator, DType.Float32, n, 2L * hidden);
            using (var dstE = cat.Narrow(1, 0, hidden))
                Ops.Copy(dstE, eNorm);
            using (var dstH = cat.Narrow(1, hidden, hidden))
                Ops.Copy(dstH, hNorm);
            eNorm.Dispose();
            hNorm.Dispose();

            Tensor x = LinearForwardCached(cat, _mtpEhProjQW, _mtpEhProjF32);
            cat.Dispose();
            return x;
        }

        /// <summary>
        /// One MTP draft step: consume (token, hPrev) at <paramref name="pos"/>,
        /// fill <paramref name="logitsOut"/> (vocab floats) with next-token logits
        /// and <paramref name="hOut"/> (hidden floats) with the MTP hidden state
        /// used to chain the next draft step.
        /// </summary>
        public unsafe void DraftStep(int token, float[] hPrev, int pos, float[] logitsOut, float[] hOut)
        {
            if (HasDFlash)
                throw new NotSupportedException("A DFlash drafter proposes whole blocks; use DraftBlock.");
            if (!HasMtpDraftHead)
                throw new InvalidOperationException("Model has no NextN/MTP draft block.");
            EnterSpecSession();
            EnsureCacheCapacity(pos + 1);

            Tensor x = MtpProjectInput(new[] { token }, hPrev);

            // Fast path: run the MTP block + shared-head norm + LM head as ONE fused,
            // CUDA-graph-captured graph (llama.cpp's graph_mtp), mirroring the trunk's
            // fused verify. ~14 ms op-by-op -> ~1-2 ms captured. Falls through to the
            // op-by-op block on any unsupported shape/backend.
            if (TryFusedMtpBlock(x, pos, 1, hOut, logitsOut, nLogitRows: 1))
            {
                x.Dispose();
                return;
            }

            EnsureKvCacheHostSynchronized();
            x = AttentionBlock(x, _mtpLayerIdx, 1, pos);

            Tensor headNorm = _mtpHeadNormW ?? _finalNormW;
            Tensor hn = RMSNormOpCached(x, headNorm);
            x.Dispose();

            fixed (float* dst = hOut)
            {
                float* src = GetFloatPtr(hn);
                Buffer.MemoryCopy(src, dst, (long)hOut.Length * 4, (long)Config.HiddenSize * 4);
            }

            // nextn.shared_head_head when shipped, otherwise the trunk LM head.
            bool hasOwnHead = _mtpHeadQW != null || _mtpHeadF32 != null;
            QuantizedWeight headQW = hasOwnHead ? _mtpHeadQW : _lmHeadQW;
            Tensor headF32 = hasOwnHead ? _mtpHeadF32 : _lmHeadF32;
            Tensor logitsT = LinearForwardCached(hn, headQW, headF32);
            hn.Dispose();

            fixed (float* dst = logitsOut)
            {
                float* src = GetFloatPtr(logitsT);
                Buffer.MemoryCopy(src, dst, (long)logitsOut.Length * 4, (long)Config.VocabSize * 4);
            }
            logitsT.Dispose();
        }

        /// <summary>
        /// MTP catch-up pass (llama.cpp's draft-mtp process()): replays verified
        /// trunk tokens through the MTP block so its KV cache stays in sync with
        /// exact trunk hidden states. Logits are not needed — only the KV side
        /// effects matter.
        /// </summary>
        private float[] _mtpCatchupLogits;

        public void DraftCatchUp(int[] tokens, float[] hRows, int startPos)
        {
            if (HasDFlash)
            {
                // The DFlash ring holds committed positions only, and the executor
                // hands back exactly the rows it committed.
                EnterSpecSession();
                DFlashCommit(tokens, hRows, startPos);
                return;
            }
            if (!HasMtpDraftHead)
                throw new InvalidOperationException("Model has no NextN/MTP draft block.");
            EnterSpecSession();
            EnsureCacheCapacity(startPos + tokens.Length);

            Tensor x = MtpProjectInput(tokens, hRows);

            // Fused block (same captured/one-shot graph as the draft). Catch-up only
            // needs the MTP block's KV side effects — request a single logit row
            // (minimal lm_head) and no hidden capture. Falls back to op-by-op.
            _mtpCatchupLogits ??= new float[Config.VocabSize];
            if (TryFusedMtpBlock(x, startPos, tokens.Length, null, _mtpCatchupLogits, nLogitRows: 1))
            {
                x.Dispose();
                return;
            }

            EnsureKvCacheHostSynchronized();
            x = AttentionBlock(x, _mtpLayerIdx, tokens.Length, startPos);
            x.Dispose();
        }

        // Fold the MTP catch-up into the first draft step (llama.cpp's draft-mtp
        // runs its block over n_accepted + 1 rows). TS_MTP_FOLD_CATCHUP=0 goes back
        // to a catch-up pass plus a separate first DraftStep.
        private static readonly bool _mtpFoldCatchUpEnabled =
            !string.Equals(Environment.GetEnvironmentVariable("TS_MTP_FOLD_CATCHUP"), "0", StringComparison.Ordinal);

        /// <summary>Only the NextN/MTP head folds. A DFlash drafter proposes whole
        /// blocks and its commit is a ring write costing ~1 ms, so there is nothing
        /// worth folding there.</summary>
        public bool SupportsFusedCatchUpStep
            => _mtpFoldCatchUpEnabled && HasMtpDraftHead && !HasDFlash;

        private float[] _mtpFoldNormed;

        /// <inheritdoc />
        public unsafe void DraftCatchUpAndStep(int[] tokens, float[] hRows, int startPos,
            float[] logitsOut, float[] hOut)
        {
            if (!HasMtpDraftHead || HasDFlash)
                throw new InvalidOperationException("Model has no NextN/MTP draft block to fold.");
            int n = tokens.Length;
            int H = Config.HiddenSize;
            EnterSpecSession();
            EnsureCacheCapacity(startPos + n);

            // ONE block pass over every row. The kernel folds the LM head over the
            // last n_logits rows (here 1), so logitsOut is already the last row's;
            // the normed hidden comes back for all n rows and the last is the one
            // that chains the next draft step.
            Tensor x = MtpProjectInput(tokens, hRows);
            if (_mtpFoldNormed == null || _mtpFoldNormed.Length < (long)H * n)
                _mtpFoldNormed = new float[(long)H * n];
            if (TryFusedMtpBlock(x, startPos, n, _mtpFoldNormed, logitsOut, nLogitRows: 1))
            {
                x.Dispose();
                Array.Copy(_mtpFoldNormed, (long)(n - 1) * H, hOut, 0, H);
                return;
            }
            x.Dispose();

            // The fused block declined this shape. Fall back to the two-call form
            // rather than the op-by-op fold, so this path stays the one that is
            // already covered by DraftCatchUp/DraftStep.
            if (n > 1)
            {
                var replay = new int[n - 1];
                Array.Copy(tokens, replay, n - 1);
                DraftCatchUp(replay, hRows, startPos);
            }
            var hLast = new float[H];
            Array.Copy(hRows, (long)(n - 1) * H, hLast, 0, H);
            DraftStep(tokens[n - 1], hLast, startPos + n - 1, logitsOut, hOut);
        }

        /// <summary>
        /// Trunk forward for speculative decoding. Identical math to Forward()
        /// but additionally captures the post-final-norm hidden state of every
        /// row into <paramref name="hAllOut"/> (n*hidden floats; llama.cpp's
        /// h_nextn) and, when <paramref name="allLogitsRows"/> is set, computes
        /// LM-head logits for every row into <paramref name="logitsOut"/>
        /// (n*vocab floats) instead of only the last row.
        /// Advances the KV caches and _cacheSeqLen exactly like Forward().
        /// </summary>
        // SpecForward layer-type timing (speculative-path profiling; cheap
        // enough to keep always-on: one timestamp per layer per pass).
        public long SpecAttnLayerTicks { get; private set; }
        public long SpecRecurrentLayerTicks { get; private set; }
        public long SpecLmHeadTicks { get; private set; }
        public void ResetSpecLayerTimings()
        {
            SpecAttnLayerTicks = SpecRecurrentLayerTicks = SpecLmHeadTicks = 0;
        }

        public unsafe void SpecForward(int[] tokens, float[] hAllOut, float[] logitsOut, bool allLogitsRows)
        {
            EnterSpecSession();
            _forwardSw.Start();
            int seqLen = tokens.Length;
            int startPos = _cacheSeqLen;
            int hiddenSize = Config.HiddenSize;
            BeginRopePositions(startPos, seqLen);
            EnsureCacheCapacity(startPos + seqLen);

            long t0 = Stopwatch.GetTimestamp();
            Tensor hidden = Embedding(tokens);
            _embTicks += Stopwatch.GetTimestamp() - t0;

            if (HasDFlash)
            {
                // Buffer-size contract (ISpeculativeTarget.SpecForward): a hidden
                // buffer too small for one row per token means the caller only wants
                // the LAST row, written to row 0.
                int feat = _dflash.FeatureSize;
                bool captureAll = false, captureLast = false;
                if (hAllOut != null && hAllOut.LongLength > 0)
                {
                    captureAll = hAllOut.LongLength >= (long)seqLen * feat;
                    captureLast = !captureAll;
                    if (captureLast && hAllOut.LongLength < feat)
                    {
                        throw new ArgumentException(
                            $"DFlash hidden capture buffer holds {hAllOut.LongLength} floats; one feature row needs {feat}.",
                            nameof(hAllOut));
                    }
                }

                if (!TryDFlashSpecForwardFused(hidden, startPos, seqLen, hAllOut, logitsOut,
                        allLogitsRows, captureAll, captureLast))
                {
                    DFlashSpecForwardPerOp(hidden, startPos, seqLen, hAllOut, logitsOut,
                        allLogitsRows, captureAll, captureLast);
                }
                else
                {
                    hidden.Dispose();
                }
                _cacheSeqLen += seqLen;
                _forwardCount++;
                _forwardSw.Stop();
                return;
            }

            // Fast path: run the whole trunk over the N tokens as ONE fused GGML
            // graph (TSGgml_Qwen35ModelVerify) instead of the op-by-op layer loop.
            // Writes hAllOut (post-norm hidden) + logitsOut directly; advances KV +
            // GDN state by N. Env-gated TS_QWEN35_FUSED_VERIFY (default on).
            if (TryFusedVerifyTrunk(hidden, startPos, seqLen, hAllOut, logitsOut, allLogitsRows))
            {
                hidden.Dispose();
                _cacheSeqLen += seqLen;
                _forwardCount++;
                _forwardSw.Stop();
                return;
            }

            // A failed last-row-only verify may leave the only current recurrent
            // state in the native ping-pong buffer. Cross through the shared
            // device-to-host barrier before the per-op loop reads host mirrors.
            PrepareHostPrefillFallback();
            for (int layer = 0; layer < Config.NumLayers; layer++)
            {
                long tl = Stopwatch.GetTimestamp();
                if (_isRecurrent[layer])
                {
                    hidden = RecurrentBlock(hidden, layer, seqLen, startPos);
                    SpecRecurrentLayerTicks += Stopwatch.GetTimestamp() - tl;
                }
                else
                {
                    hidden = AttentionBlock(hidden, layer, seqLen, startPos);
                    SpecAttnLayerTicks += Stopwatch.GetTimestamp() - tl;
                }
                TryEvaluateMlxLayerBoundary(hidden, layer, seqLen);
            }

            // Final norm over ALL rows (the MTP draft head consumes per-row
            // post-norm hidden states, llama.cpp's t_h_nextn).
            Tensor normed = RMSNormOpCached(hidden, _finalNormW);
            hidden.Dispose();

            if (hAllOut != null)
            {
                fixed (float* dst = hAllOut)
                {
                    float* src = GetFloatPtr(normed);
                    Buffer.MemoryCopy(src, dst, (long)hAllOut.Length * 4, (long)seqLen * hiddenSize * 4);
                }
            }

            long t2 = Stopwatch.GetTimestamp();
            if (allLogitsRows)
            {
                Tensor logitsT = LinearForwardCached(normed, _lmHeadQW, _lmHeadF32);
                normed.Dispose();
                fixed (float* dst = logitsOut)
                {
                    float* src = GetFloatPtr(logitsT);
                    Buffer.MemoryCopy(src, dst, (long)logitsOut.Length * 4, (long)seqLen * Config.VocabSize * 4);
                }
                logitsT.Dispose();
            }
            else
            {
                Tensor lastRow;
                if (seqLen > 1)
                {
                    using var narrowed = normed.Narrow(0, seqLen - 1, 1);
                    lastRow = Ops.NewContiguous(narrowed);
                    normed.Dispose();
                }
                else
                {
                    lastRow = normed;
                }
                Tensor logitsT = LinearForwardCached(lastRow, _lmHeadQW, _lmHeadF32);
                lastRow.Dispose();
                fixed (float* dst = logitsOut)
                {
                    float* src = GetFloatPtr(logitsT);
                    Buffer.MemoryCopy(src, dst, (long)logitsOut.Length * 4, (long)Config.VocabSize * 4);
                }
                logitsT.Dispose();
            }
            _lmHeadTicks += Stopwatch.GetTimestamp() - t2;
            SpecLmHeadTicks += Stopwatch.GetTimestamp() - t2;

            _cacheSeqLen += seqLen;
            _forwardCount++;
            _forwardSw.Stop();
        }

        private float[] _fvLogitsAllBuf;
        private float[] _fvLogitsLastBuf;

        /// <summary>Fused trunk verify fast path for <see cref="SpecForward"/>: route
        /// all-N logit rows to logitsOut directly when the caller wants all rows
        /// (the true MTP verify batch, N &lt;= maxDraft+1). Last-row-only callers
        /// (prompt prefill chunks, rollback catch-up re-forwards) ask the kernel for
        /// ONE logit row (nLogitRows: 1): computing lm_head over every prompt row
        /// wastes vocab*N floats of device + host memory (248320-vocab * 2048-token
        /// chunk = 2 GB), and — because n_logits == N is also the kernel's persist-
        /// graph trigger — it used to leave a multi-GB persistent verify graph cached
        /// PER PROMPT LENGTH. Returns false (op-by-op fallback) when disabled or the
        /// kernel declines the shape.</summary>
        private bool TryFusedVerifyTrunk(Tensor hidden, int startPos, int seqLen,
            float[] hAllOut, float[] logitsOut, bool allLogitsRows)
        {
            if (!_fusedVerifyEnabled)
                return false;
            int vocab = Config.VocabSize;
            if (allLogitsRows)
            {
                float[] allLogits;
                if (logitsOut != null && logitsOut.Length >= (long)seqLen * vocab)
                {
                    allLogits = logitsOut;
                }
                else
                {
                    if (_fvLogitsAllBuf == null || _fvLogitsAllBuf.Length < (long)seqLen * vocab)
                        _fvLogitsAllBuf = new float[(long)seqLen * vocab];
                    allLogits = _fvLogitsAllBuf;
                }
                return TryFullModelVerify(hidden, startPos, seqLen, hAllOut, allLogits);
            }

            if (_fvLogitsLastBuf == null || _fvLogitsLastBuf.Length < vocab)
                _fvLogitsLastBuf = new float[vocab];
            if (!TryFullModelVerify(hidden, startPos, seqLen, hAllOut, _fvLogitsLastBuf, nLogitRows: 1))
                return false;
            if (logitsOut != null)
                Array.Copy(_fvLogitsLastBuf, 0, logitsOut, 0, vocab);
            return true;
        }

        /// <summary>
        /// Grow the KV caches up front to cover a full speculative window.
        /// EnsureCacheCapacity's growth path only preserves rows below
        /// _cacheSeqLen, so growing mid-draft would drop the MTP rows written
        /// past the trunk position; callers pre-grow before drafting instead.
        /// </summary>
        public void SpecEnsureCapacity(int requiredSeqLen) => EnsureCacheCapacity(requiredSeqLen);

        /// <summary>
        /// True when the accepted prefix of the last verify is already committed:
        /// its attention KV was written at the right positions by the verify itself,
        /// and its recurrent state came back out of a per-row snapshot in
        /// <see cref="SpecOnVerifyAccepted"/>. The executor then only has to rewind
        /// the position, instead of restoring a pre-verify state copy and
        /// re-forwarding the accepted prefix through all 64 layers.
        ///
        /// Necessarily per-step rather than a constant: whether the verify kept
        /// snapshots depends on the shape it ran at (only the persisted, all-rows,
        /// host-state verify does), and getting it wrong in either direction decodes
        /// from the wrong recurrent state.
        /// </summary>
        public bool SpecVerifyPersistsAcceptedKv => _fvAcceptedPrefixCommitted;

        /// <summary>
        /// Settle the recurrent state for the accepted prefix. When the verify left
        /// per-row snapshots on the device, the state after row <paramref name="acceptedRows"/>
        /// is slot (rows - 1 - accepted) - counting back from the end of the batch -
        /// and one fetch replaces the whole restore-and-re-forward.
        ///
        /// Called on EVERY speculative step, full acceptance included: with snapshots
        /// on, even a fully-accepted verify has not written its post-window state to
        /// the host mirror yet, and slot 0 is that state.
        /// </summary>
        public void SpecOnVerifyAccepted(int acceptedRows, int verifyRows)
        {
            _fvAcceptedPrefixCommitted = false;
            if (_fvSnapshotRows <= 0)
                return;                       // the old path already drained the state

            int rows = _fvSnapshotRows;
            _fvSnapshotRows = 0;
            int slot = rows - 1 - acceptedRows;
            if (slot < 0 || slot >= rows)
            {
                // Cannot happen for a well-formed accept count, and silently taking
                // the wrong slot would decode from the wrong state.
                throw new InvalidOperationException(
                    $"Recurrent-state snapshot slot {slot} outside [0, {rows}) (accepted {acceptedRows} of {verifyRows}).");
            }

            if (!CommitRecurrentStateSnapshot(slot))
            {
                // The state is still on the device and the host mirror is stale, so
                // the executor MUST take the restore-and-re-forward path - which is
                // exactly what leaving _fvAcceptedPrefixCommitted false selects, and
                // which is correct because the pre-verify snapshot is untouched.
                return;
            }
            _fvAcceptedPrefixCommitted = true;
        }

        /// <summary>
        /// Snapshot the GDN recurrent state of every trunk layer. Taken right
        /// before a speculative verify batch so a partial rejection can roll the
        /// recurrent state back (attention KV needs only a position rewind).
        /// </summary>
        /// <summary>
        /// Set when <see cref="SpecSnapshotRecurrentState"/> skipped its host copy
        /// because the pre-verify state was already sitting in the verify kernel's
        /// live device slices - which a verify only ever READS, so those slices ARE
        /// the snapshot until a snapshot commit overwrites them.
        /// </summary>
        private bool _fvSnapshotIsDeviceLive;

        public void SpecSnapshotRecurrentState()
        {
            // A parked run of plain steps goes through the fused decode, which keeps
            // the recurrent state device-resident in ITS slot; the executor takes
            // this snapshot before the verify forward that would sync it, so the
            // host mirrors below could still hold the pre-park state. Settle it
            // first, or a rollback that restores from this snapshot (the host-mode
            // verify, the per-op fallback) would continue from stale state.
            if (_fdStateResident)
                InvalidateFullDecodeState();
            // Nothing to copy: the state the verify is about to run from lives in the
            // shared device slices, the verify writes its results elsewhere (the
            // *_state_out slices and the snapshot slots), and the only thing that ever
            // overwrites the live slices is a snapshot commit - which happens after
            // the rollback decision. So the slices remain a perfectly good "snapshot"
            // for as long as one is needed, at no cost.
            _fvSnapshotIsDeviceLive = _fvDeviceStateCurrent;
            if (_fvSnapshotIsDeviceLive)
                return;

            // Direct-CUDA fast path: snapshot the GDN state device-to-device
            // (async cuMemcpyDtoD on the stream) instead of draining it to host
            // bytes. The host path does an EnsureHostReadable DtoH per recurrent
            // layer (48 syncs) every verify step, but the snapshot is only ever
            // consumed on a partial-rejection rollback (~1 in this model) -- so
            // those DtoH stalls were almost entirely wasted on a sync-bound
            // backend.
            if (_backend == BackendType.Cuda)
            {
                MtpSnapshotRecurrentStateCudaDevice();
                return;
            }

            _mtpGdnSnapshot ??= new byte[Config.NumLayers][];
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (!_isRecurrent[l])
                    continue;
                long bytes = GdnLayerStateBytes(l);
                if (_mtpGdnSnapshot[l] == null || _mtpGdnSnapshot[l].Length != bytes)
                    _mtpGdnSnapshot[l] = new byte[bytes];
                if (!CopyGdnStateOut(l, _mtpGdnSnapshot[l], out _))
                    throw new InvalidOperationException($"Failed to snapshot GDN state for layer {l}.");
            }
        }

        /// <summary>Restore the GDN recurrent state captured by <see cref="SpecSnapshotRecurrentState"/>.</summary>
        public void SpecRestoreRecurrentState()
        {
            if (_fvSnapshotIsDeviceLive)
            {
                // The pre-verify state is in the live device slices; the host mirrors
                // are whatever the verify left there. Bring the slices back so the
                // kept-prefix re-forward and any op-by-op path see the right state.
                _fvSnapshotIsDeviceLive = false;
                _fvDeviceStateCurrent = true;      // the slices are authoritative
                DrainDeviceRecurrentState();
                return;
            }

            if (_backend == BackendType.Cuda)
            {
                MtpRestoreRecurrentStateCudaDevice();
                return;
            }

            if (_mtpGdnSnapshot == null)
                throw new InvalidOperationException("No GDN snapshot to restore.");
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (!_isRecurrent[l])
                    continue;
                if (!CopyGdnStateIn(l, _mtpGdnSnapshot[l], out _))
                    throw new InvalidOperationException($"Failed to restore GDN state for layer {l}.");
            }
        }

        // Reusable device-resident GDN snapshot buffers for the CUDA linear-trunk
        // path (conv ring buffer + SSM/delta state + conv ring write index).
        private Tensor[] _mtpGdnConvDevSnap;
        private Tensor[] _mtpGdnDeltaDevSnap;
        private int[] _mtpGdnConvIdxDevSnap;

        private void MtpSnapshotRecurrentStateCudaDevice()
        {
            int layers = Config.NumLayers;
            _mtpGdnDeltaDevSnap ??= new Tensor[layers];
            _mtpGdnConvDevSnap ??= new Tensor[layers];
            _mtpGdnConvIdxDevSnap ??= new int[layers];
            for (int l = 0; l < layers; l++)
            {
                if (!_isRecurrent[l])
                    continue;
                Tensor delta = _deltaStateTensor[l];
                _mtpGdnDeltaDevSnap[l] ??= new Tensor(_allocator, delta.ElementType, _numVHeads, _headVDim, _headKDim);
                Ops.Copy(_mtpGdnDeltaDevSnap[l], delta);

                Tensor conv = _cudaGdnConvStateTensor?[l];
                if (conv != null)
                {
                    _mtpGdnConvDevSnap[l] ??= new Tensor(_allocator, conv.ElementType, conv.Sizes[0], conv.Sizes[1]);
                    Ops.Copy(_mtpGdnConvDevSnap[l], conv);
                }
                _mtpGdnConvIdxDevSnap[l] = _convStateWriteIdx[l];
            }
        }

        private void MtpRestoreRecurrentStateCudaDevice()
        {
            if (_mtpGdnDeltaDevSnap == null)
                throw new InvalidOperationException("No GDN snapshot to restore.");
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (!_isRecurrent[l])
                    continue;
                // Ops.Copy is device-to-device and MarkDeviceModified(), so the
                // recurrence kernel reads the restored device state directly (the
                // stale host mirror is never re-uploaded over it).
                Ops.Copy(_deltaStateTensor[l], _mtpGdnDeltaDevSnap[l]);
                if (_cudaGdnConvStateTensor?[l] != null && _mtpGdnConvDevSnap[l] != null)
                    Ops.Copy(_cudaGdnConvStateTensor[l], _mtpGdnConvDevSnap[l]);
                _convStateWriteIdx[l] = _mtpGdnConvIdxDevSnap[l];
            }
        }

        /// <summary>
        /// Rewind the attention KV position counter after rejected speculative
        /// tokens. Rows past <paramref name="length"/> are dead weight that the
        /// next forward simply overwrites (the causal mask never reads past the
        /// current position), so no data movement is needed.
        /// </summary>
        public void SpecRewindCache(int length)
        {
            if (length < 0 || length > _cacheSeqLen)
                throw new ArgumentOutOfRangeException(nameof(length),
                    $"Rewind length {length} outside [0, {_cacheSeqLen}].");
            _cacheSeqLen = length;
            // Speculation rewinds only generated rows, which all share the delta;
            // a rewind to nothing leaves no history for one to belong to.
            if (length == 0)
                _ropeDelta = 0;
        }

        // ====================================================================
        // Batched-trunk speculative decoding (IBatchedSpeculativeModel):
        // trunk passes run through ForwardBatch (paged KV via the sequence's
        // block table, per-slot GDN state) so speculation rides the same
        // kernels as the non-speculative batched baseline. The MTP draft head
        // above is unchanged — it runs on the linear cache at _mtpLayerIdx,
        // which is private to the speculative context.
        // ====================================================================

        // Per-slot GDN snapshot used to roll back a partially-rejected verify
        // batch: per recurrent layer, ONE slot's conv ring buffer + write
        // index + SSM state + init flag.
        private float[][] _mtpSlotConvSnapshot;
        private int[] _mtpSlotConvIdxSnapshot;
        private float[][] _mtpSlotSsmSnapshot;
        private bool[] _mtpSlotInitSnapshot;
        private int _mtpSlotSnapshotSlot = -1;

        /// <summary>
        /// MTP speculation is only a throughput WIN when the model's STANDARD
        /// decode is slow enough that drafting + verifying K tokens saves more
        /// than the speculation machinery costs. On <c>ggml_cuda</c> the standard
        /// Qwen3.6 decode IS the fused, CUDA-graph-captured whole-model decode
        /// (<see cref="TryFullModelDecode"/>): one graph replay per token, fully
        /// device-resident GDN/KV state, zero host orchestration — ~73 tok/s
        /// (~13.7 ms/token) on 35B-A3B IQ2_XXS, already at the memory-bandwidth
        /// floor (only ~3B of 35B params are active, so each decode token is cheap).
        ///
        /// <c>--spec</c> is an EXPLICIT operator opt-in, so we honor it whenever
        /// the model actually has an MTP/NextN head — even on ggml_cuda where the
        /// captured decode (~73 tok/s) may still beat speculation. (Earlier this gated
        /// OFF on ggml_cuda because the op-by-op verify made MTP ~34x slower; the
        /// fused multi-token verify <see cref="TryFullModelVerify"/>,
        /// <c>TS_QWEN35_FUSED_VERIFY=1</c>, cuts the verify ~20x and is the path that
        /// makes spec competitive — long context / large drafts.) The user asked that
        /// the flag be respected regardless, so don't second-guess it here.
        ///
        /// Not gated on <see cref="HasDraftHead"/>: this asks about the TRUNK's
        /// multi-token verify, which here is the ordinary layer stack with a
        /// hidden-state tap and does not touch the NextN block. A weight-free
        /// speculator (<c>--spec-type ngram</c>) therefore works on a Qwen 3.5/3.6
        /// checkpoint that ships no draft head at all; whether a LEARNED drafter
        /// exists is the registry's question, not this one.
        /// </summary>
        public bool SpeculationProfitable => true;

        /// <summary>
        /// Every field the speculative trunk touches - attention K/V, the GDN conv and
        /// delta state, the position, the residency latches - is what
        /// BindSequenceCache swapped in, and a holder switch drains the verify's
        /// device-live state into the outgoing holder's mirrors first
        /// (LoadCacheHolder -> InvalidateVerifyCache). So the trunk forwards on the
        /// BOUND holder, and TensorAgent's turns (all served from holders) can
        /// speculate. What used to break was not the state but the capability:
        /// see SupportsPerSequenceFusedForward.
        /// </summary>
        public bool SpecTrunkFollowsBoundCache => true;

        /// <summary>
        /// A parked plain step takes the fused whole-model decode (20.4 ms/token on
        /// 9B IQ4_XS, Metal) instead of a one-row pass through the verify family
        /// (25.6 ms): a governor that parks for 32-256 steps at a time made prose
        /// 25% slower than plain decoding while drafting nothing. Ordinary plain
        /// steps stay in the verify family - see SpecPlainStepCostsFamilySwitch.
        /// </summary>
        public bool SpecPlainStepUsesForward
            => IsGgmlBackend && !IsTensorParallel && !HasDFlash && _fullDecodeEnabled && !_fdUnsupported;

        public bool SpecPlainStepCostsFamilySwitch => true;

        /// <summary>Batched spec trunk needs the GGML batched paged path (the
        /// MLX backend keeps GDN state inside opaque per-slot MLX caches the
        /// snapshot/restore below cannot capture). When the fused multi-token
        /// verify (<see cref="TryFullModelVerify"/>) is enabled we route spec to the
        /// LINEAR trunk instead (SpecForward), whose KV/GDN state the fused verify
        /// reads/writes; the batched paged trunk uses a different (paged) store.</summary>
        // A DFlash drafter keeps ONE ring for ONE sequence and its catch-up is driven
        // from the linear trunk's per-row features, so it cannot ride the paged
        // multi-sequence trunk; those requests take the linear speculative path.
        public bool SupportsBatchedSpecTrunk => HasMtpDraftHead && !HasDFlash && IsGgmlBackend && IsBatchedPathEnabled() && !_fusedVerifyEnabled;

        public void SpecForwardBatched(SequenceState seq, int[] tokens, int startPos,
            float[] hAllOut, float[] logitsOut, bool allLogitsRows)
        {
            EnterSpecSession();
            ArgumentNullException.ThrowIfNull(seq);
            if (tokens == null || tokens.Length == 0)
                throw new ArgumentException("Tokens must not be empty.", nameof(tokens));
            // The batched path reads the sequence's committed length for its
            // attention extents, so every spec pass must start exactly there
            // (the executor advances the sequence only after the step).
            if (startPos != seq.NumComputedTokens)
                throw new InvalidOperationException(
                    $"SpecForwardBatched at position {startPos} but sequence has {seq.NumComputedTokens} computed tokens.");

            int n = tokens.Length;
            var bt = seq.BlockTable;
            if (bt.CapacityTokens < startPos + n)
                throw new InvalidOperationException(
                    $"Block table covers {bt.CapacityTokens} tokens but the spec pass needs {startPos + n}.");

            var positions = new System.Collections.Generic.List<int>(n);
            var slotMapping = new System.Collections.Generic.List<int>(n);
            for (int i = 0; i < n; i++)
            {
                int pos = startPos + i;
                positions.Add(pos);
                int blockIdx = pos / bt.BlockSize;
                slotMapping.Add(bt.Blocks[blockIdx].Id * bt.BlockSize + pos % bt.BlockSize);
            }
            var table = new int[bt.NumBlocks];
            for (int b = 0; b < bt.NumBlocks; b++)
                table[b] = bt.Blocks[b].Id;

            var ctx = new BatchedForwardContext
            {
                Sequences = new System.Collections.Generic.List<SequenceState> { seq },
                NumScheduledTokens = new System.Collections.Generic.List<int> { n },
                QueryStartLoc = new System.Collections.Generic.List<int> { 0, n },
                Positions = positions,
                SlotMapping = slotMapping,
                BlockTables = new[] { table },
                MaxQueryLen = n,
                MaxSeqLen = startPos + n,
                OverrideFlatTokens = tokens,
                CaptureHiddenAll = hAllOut,
                CaptureLogitsAll = allLogitsRows ? logitsOut : null,
            };

            var perSeq = ForwardBatch(ctx);
            if (!allLogitsRows && logitsOut != null)
                Array.Copy(perSeq[0], logitsOut, Config.VocabSize);
        }

        public unsafe void SpecSnapshotRecurrentStateSlots(SequenceState seq)
        {
            ArgumentNullException.ThrowIfNull(seq);
            if (_q35GdnSlotConvBuf == null)
                throw new InvalidOperationException(
                    "Batched GDN slot state not initialized (no batched forward has run for this sequence yet).");

            int slot = seq.BlockTable.Blocks[0].Id;
            int layers = Config.NumLayers;
            _mtpSlotConvSnapshot ??= new float[layers][];
            _mtpSlotSsmSnapshot ??= new float[layers][];
            _mtpSlotConvIdxSnapshot ??= new int[layers];
            _mtpSlotInitSnapshot ??= new bool[layers];

            int ssmLen = _numVHeads * _headVDim * _headKDim;
            for (int l = 0; l < layers; l++)
            {
                if (!_isRecurrent[l])
                    continue;
                EnsureGdnSlotAllocated(l, slot);
                float[] conv = _q35GdnSlotConvBuf[l][slot];
                if (_mtpSlotConvSnapshot[l] == null || _mtpSlotConvSnapshot[l].Length != conv.Length)
                    _mtpSlotConvSnapshot[l] = new float[conv.Length];
                Array.Copy(conv, _mtpSlotConvSnapshot[l], conv.Length);
                _mtpSlotConvIdxSnapshot[l] = _q35GdnSlotConvWriteIdx[l][slot];
                _mtpSlotInitSnapshot[l] = _q35GdnSlotInit[l][slot];
                // Pointer copy into a reused buffer: GetElementsAsFloat would
                // allocate a fresh ~3 MB array per layer per verify step
                // (gigabytes of GC churn per request — measured 92 ms/step
                // vs ~12 ms for the raw copy).
                if (_mtpSlotSsmSnapshot[l] == null || _mtpSlotSsmSnapshot[l].Length != ssmLen)
                    _mtpSlotSsmSnapshot[l] = new float[ssmLen];
                float* src = GetFloatPtr(_q35GdnSlotSsmTensor[l][slot]);
                fixed (float* dst = _mtpSlotSsmSnapshot[l])
                    Buffer.MemoryCopy(src, dst, (long)ssmLen * 4, (long)ssmLen * 4);
            }
            _mtpSlotSnapshotSlot = slot;
        }

        public unsafe void SpecRestoreRecurrentStateSlots(SequenceState seq)
        {
            ArgumentNullException.ThrowIfNull(seq);
            int slot = seq.BlockTable.Blocks[0].Id;
            if (_mtpSlotSnapshotSlot != slot)
                throw new InvalidOperationException(
                    $"No recurrent-state snapshot for slot {slot} (snapshot holds slot {_mtpSlotSnapshotSlot}).");

            int ssmLen = _numVHeads * _headVDim * _headKDim;
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (!_isRecurrent[l])
                    continue;
                Array.Copy(_mtpSlotConvSnapshot[l], _q35GdnSlotConvBuf[l][slot], _mtpSlotConvSnapshot[l].Length);
                _q35GdnSlotConvWriteIdx[l][slot] = _mtpSlotConvIdxSnapshot[l];
                _q35GdnSlotInit[l][slot] = _mtpSlotInitSnapshot[l];
                Tensor ssm = _q35GdnSlotSsmTensor[l][slot];
                float* dst = GetFloatPtr(ssm);
                fixed (float* src = _mtpSlotSsmSnapshot[l])
                    Buffer.MemoryCopy(src, dst, (long)ssmLen * 4, (long)ssmLen * 4);
                InvalidateTensorDeviceCache(ssm);
            }
        }
    }
}
