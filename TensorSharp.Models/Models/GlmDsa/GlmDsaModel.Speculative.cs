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
// NextN/MTP speculative decoding for glm-dsa (GLM-5.2).
//
// GLM-5.2 ships ONE trailing NextN block (`nextn_predict_layers = 1`, so
// `block_count = 79` and the trunk is 78 layers). Block 78 is a complete
// glm-dsa decoder block — MLA attention plus the sigmoid-gated 256-expert MoE
// with its shared expert — wrapped in the deepseek-family NextN wiring:
//
//     h_mtp = shared_head_norm( block( eh_proj( [enorm(embed(t)) ; hnorm(h)] ) ) )
//     logits = lm_head(h_mtp)
//
// where `h` is the trunk's POST-output_norm hidden state of the token before
// `t`. The block predicts token t+1 from token t, so chaining it drafts a
// window that the trunk then verifies in a single batched forward.
//
// Three details are worth stating because getting any of them wrong is silent:
//
//   * The draft block runs DENSE attention. It carries lightning-indexer
//     weights (blk.78.indexer.*) but llama.cpp's `graph_mtp` builds the plain
//     MLA attention input and never reads them, so reproducing llama.cpp means
//     leaving them alone — see the `dense` argument to Attention().
//   * GLM-5.2 ships neither `nextn.embed_tokens` nor `nextn.shared_head_head`,
//     so the block borrows the trunk's embedding table and LM head. Both are
//     optional in llama.cpp too, and both are honoured here when present.
//   * The MTP KV cache needs no rollback. Catch-up always rewrites from the
//     verified position forward, so rejected speculative rows are overwritten
//     before anything reads them.
//
// This file is the per-op (managed) implementation, which is the reference and
// what the synthetic-model tests exercise. GLM-5.2 itself runs through the
// native whole-model executor; GlmDsaModel.MtpNative.cs routes these same
// entry points there.
using System;
using TensorSharp.Core;
using TensorSharp.Runtime.Scheduling;

using TensorSharp.Runtime.Speculative;

namespace TensorSharp.Models
{
    public partial class GlmDsaModel : ISpeculativeModel
    {
        // NextN wiring of the draft block. Null until ResolveMtpLayer() finds a
        // complete block; _mtpLayer stays -1 in that case and HasDraftHead is false.
        private Tensor _mtpEnormW;
        private Tensor _mtpHnormW;
        private Tensor _mtpHeadNormW;      // nextn.shared_head_norm (falls back to output_norm)
        private Tensor _mtpEmbdF32;        // optional nextn.embed_tokens
        private string _mtpHeadName;       // LM head the draft block uses

        /// <summary>Highest MTP cache position written + 1. Diagnostics only —
        /// the draft/catch-up positions are supplied by the caller.</summary>
        private int _mtpCacheSeqLen;

        /// <summary>The micro-batch the native executor was loaded with, so a
        /// speculative prefill can chunk to match it. See <see cref="SpecPrefillChunkSize"/>.</summary>
        private int _nativeUbatch;

        // Scratch reused across draft steps so a 16-token window does not
        // allocate 16 hidden-sized arrays per decode step.
        private float[] _mtpHScratch;

        /// <summary>True when the loaded weights contain a usable NextN/MTP draft block.</summary>
        public bool HasDraftHead { get; private set; }

        /// <summary>GLM-5.2's NextN block drafts one token per pass, so it is
        /// served by <see cref="DraftHeadSpeculator"/>.</summary>
        public DraftHeadKind DraftHeadKind => HasDraftHead ? DraftHeadKind.PerToken : DraftHeadKind.None;

        /// <summary>
        /// Decide whether the checkpoint actually carries the MTP block, and cache
        /// the NextN wiring. A GGUF may declare <c>nextn_predict_layers</c> and
        /// still ship no MTP tensors (a trunk-only re-quantization); llama.cpp
        /// marks them NOT_REQUIRED for exactly that case, and so does this.
        /// </summary>
        private void ResolveMtpLayer()
        {
            HasDraftHead = false;
            _mtpLayer = -1;
            if (_numNextnLayers <= 0)
                return;
            if (_numNextnLayers != 1)
            {
                Console.WriteLine($"  NextN/MTP: {_numNextnLayers} blocks declared; only 1 is supported. Drafting disabled.");
                return;
            }

            int il = _numTrunkLayers;                  // the single trailing block
            string[] wn = _layerNames[il];

            _weights.TryGetValue(wn[WN_NEXTN_ENORM], out _mtpEnormW);
            _weights.TryGetValue(wn[WN_NEXTN_HNORM], out _mtpHnormW);
            _weights.TryGetValue(wn[WN_NEXTN_SHARED_HEAD_NORM], out _mtpHeadNormW);
            _weights.TryGetValue(wn[WN_NEXTN_EMBED_TOKENS], out _mtpEmbdF32);

            bool hasProj = HasAnyWeight(wn[WN_NEXTN_EH_PROJ]);
            bool hasAttn = _weights.ContainsKey(wn[WN_ATTN_NORM])
                        && HasAnyWeight(wn[WN_Q_A]) && HasAnyWeight(wn[WN_Q_B])
                        && HasAnyWeight(wn[WN_KV_A_MQA]) && HasAnyWeight(wn[WN_ATTN_OUT])
                        && _wkB[il] != null && _wvB[il] != null;
            // The three ways a checkpoint can store routed experts: stacked and
            // quantized, split per expert, or (F32) one [n_expert, out, in] block
            // that ExpertLinear slices. Check all of them — a check that only knew
            // the quantized layouts would disable drafting on every F32 model.
            bool hasFfn = _weights.ContainsKey(wn[WN_FFN_NORM])
                        && (il < _numDenseLead
                            ? HasAnyWeight(wn[WN_FFN_DOWN])
                            : _stackedGate[il] != null
                              || (_expertGateQW[il] != null && _expertGateQW[il][0] != null)
                              || _weights.ContainsKey($"blk.{il}.ffn_gate_exps.weight"));

            // A block with no head of its own borrows the trunk's. Under GGML
            // tensor parallelism the trunk head is column-parallel and the cached
            // handle is no longer the whole head, so refuse rather than draft
            // from the wrong weight — the trunk itself is unaffected.
            bool ownHead = HasAnyWeight(wn[WN_NEXTN_SHARED_HEAD_HEAD]);
            _mtpHeadName = ownHead ? wn[WN_NEXTN_SHARED_HEAD_HEAD]
                         : (HasAnyWeight("output.weight") ? "output.weight" : "token_embd.weight");
            bool borrowsSplitHead = !ownHead && IsTensorParallel;

            if (!hasProj || _mtpEnormW == null || _mtpHnormW == null || !hasAttn || !hasFfn)
            {
                Console.WriteLine("  NextN/MTP block declared but incomplete in this checkpoint; drafting disabled.");
                return;
            }
            if (borrowsSplitHead)
            {
                Console.WriteLine("  NextN/MTP block has no head of its own and the LM head is column-parallel " +
                                  "under tensor parallelism; drafting disabled.");
                return;
            }

            _mtpLayer = il;
            HasDraftHead = true;
            Console.WriteLine($"  NextN/MTP draft head ready (block {il}, dense attention, " +
                              $"ffn={(il < _numDenseLead ? "dense" : "moe")}, " +
                              $"ownEmbd={(_mtpEmbdF32 != null ? "yes" : "no")}, ownHead={(ownHead ? "yes" : "no")})");
        }

        // ---- ISpeculativeModel -------------------------------------------

        /// <summary>
        /// The draft block is one 256-expert MoE layer against a 78-layer trunk, so
        /// a draft step costs ~1.3% of a trunk decode step and the verify amortizes
        /// the whole trunk over the window. That holds on every backend GLM runs on;
        /// the runtime cost governor in SpeculativeExecution still measures it and
        /// parks drafting if a particular prompt/context makes it a loss.
        ///
        /// glm5next has no draft head of its own (its NextN block is not built),
        /// so what arms there is the weight-free n-gram speculator; its verify is
        /// the same one-graph trunk pass, and a rejected window is undone through
        /// the KDA recurrent-state snapshot (GlmDsaModel.Glm5NextSpeculative.cs).
        /// The one thing that can make that unavailable is a native library that
        /// predates the snapshot API, and that is declined here rather than
        /// discovered mid-verify.
        /// </summary>
        public bool SpeculationProfitable => !IsGlm5NextArch || Glm5NextRollbackAvailable;

        /// <summary>
        /// The verify batch writes MLA rows (and, on the trunk, indexer keys) for
        /// EVERY token it processes, and glm-dsa has no recurrent state, so a
        /// partially-accepted window keeps the accepted prefix's KV and only needs
        /// the position rewound. On a long context that removes the dominant
        /// rollback cost — a whole extra trunk forward.
        ///
        /// glm5next's KDA layers DO carry recurrent state that the verify advanced
        /// over the whole window, so there the executor must restore the snapshot
        /// and re-forward the accepted prefix (the Qwen 3.5 / Qwen 3.8 contract).
        /// </summary>
        public bool SpecVerifyPersistsAcceptedKv => !IsGlm5NextArch;

        /// <summary>
        /// On a recurrent trunk every extra verify row is dearer than on a dense
        /// one (the KDA scan runs over the window, and a partial rejection pays a
        /// state restore plus a re-forward of the kept prefix), so glm5next
        /// prefers the narrow default Qwen 3.8 measured best. It narrows the
        /// default only; <c>--spec-draft</c> still gets exactly what it asks for.
        /// glm-dsa proper keeps the shared default.
        /// </summary>
        public int SpecPreferredDraftWindow => IsGlm5NextArch ? 3 : 0;

        /// <summary>
        /// The native executor's SpecForward, snapshot and rewind all act on the
        /// ACTIVE slot - the one <see cref="BindSequenceCache"/> selected for the
        /// request being served - so a request served from its own slot (every
        /// request the engine serves on the native executor) can speculate on
        /// it. Without this the engine's fused-holder route declined speculation
        /// on every slot-served request with a warn-once and decoded plainly.
        /// The managed path has no per-request holders, so the flag is moot there.
        /// </summary>
        public bool SpecTrunkFollowsBoundCache => UsesNativeExecutor;

        /// <summary>
        /// The trunk re-reads the routed experts once per micro-batch, so a
        /// speculative prefill wants whole micro-batches rather than the caller's
        /// default. Matches the native executor's ubatch.
        ///
        /// This is not cosmetic on the native path. A speculative prefill is
        /// chunked by the CALLER — each chunk is one trunk forward whose per-token
        /// hidden states come back to the host, followed by one draft-block
        /// catch-up — while the native forward re-chunks whatever it is given by
        /// its own ubatch. Handing it the caller's generic 512 therefore buys two
        /// host round trips per micro-batch instead of one, and splits every
        /// expert tile in half (with 256 experts at top-8, a 512-token chunk
        /// leaves ~16 rows per expert, so half the tile is padding — the same
        /// effect that made the plain prefill 984 vs 696 tok/s at 1024).
        /// </summary>
        public int SpecPrefillChunkSize
            => UsesNativeExecutor ? Math.Max(1, _nativeUbatch) : ResolvePrefillChunkSize();


        public void SpecEnsureCapacity(int requiredSeqLen)
        {
            if (UsesNativeExecutor)
                return;                       // native caches are sized to n_ctx at load
            EnsureCacheCapacity(Math.Min(requiredSeqLen, _maxContextLength));
        }

        /// <summary>glm-dsa has no recurrent state: MLA rows and indexer keys are
        /// per-position and a rewind is exact. glm5next snapshots its KDA state
        /// (see GlmDsaModel.Glm5NextSpeculative.cs).</summary>
        public void SpecSnapshotRecurrentState()
        {
            if (IsGlm5NextArch)
                Glm5NextSnapshotRecurrentState();
        }

        /// <summary>See <see cref="SpecSnapshotRecurrentState"/>.</summary>
        public void SpecRestoreRecurrentState()
        {
            if (IsGlm5NextArch)
                Glm5NextRestoreRecurrentState();
        }

        public void SpecRewindCache(int length)
        {
            if (length < 0)
                throw new ArgumentOutOfRangeException(nameof(length));
            if (IsGlm5NextArch)
            {
                Glm5NextRewindCache(length);
                return;
            }
            if (UsesNativeExecutor)
            {
                RewindNative(length);
                return;
            }
            _cacheSeqLen = Math.Min(length, _cacheSeqLen);
        }

        /// <summary>
        /// Trunk forward that also captures the post-output_norm hidden state the
        /// draft block consumes. Identical to <see cref="ForwardCore"/> in every
        /// other respect — same layers, same caches, same position advance — so a
        /// speculative run and a plain run leave the trunk in the same state.
        /// </summary>
        public void SpecForward(int[] tokens, float[] hAllOut, float[] logitsOut, bool allLogitsRows)
        {
            ArgumentNullException.ThrowIfNull(tokens);
            if (tokens.Length == 0)
                throw new ArgumentException("At least one token is required.", nameof(tokens));

            if (UsesNativeExecutor)
            {
                SpecForwardNative(tokens, hAllOut, logitsOut, allLogitsRows);
                return;
            }
            if (IsGlm5Next)
            {
                SpecForwardGlm5Next(tokens, hAllOut, logitsOut, allLogitsRows);
                return;
            }

            int seqLen = tokens.Length;
            int hidden = Config.HiddenSize;
            int startPos = _cacheSeqLen;
            EnsureCacheCapacity(startPos + seqLen);

            Tensor hidden3 = Embedding(tokens);
            _sharedTopKCount = 0;
            for (int layer = 0; layer < _numTrunkLayers; layer++)
                hidden3 = DecoderBlock(hidden3, layer, seqLen, startPos);

            // The norm runs over every row (not just the LM-head rows) because the
            // draft head needs one hidden state per token. RMS norm is row-wise, so
            // the rows the LM head then reads are bit-identical to ForwardCore's.
            Tensor normed = RMSNormOp(hidden3, "output_norm.weight");
            hidden3.Dispose();
            Trace("result_norm", -1, normed);

            if (hAllOut != null)
                CopyRowsToBuffer(normed, hAllOut, seqLen, hidden);

            Tensor headIn;
            if (allLogitsRows || seqLen == 1)
            {
                headIn = normed.CopyRef();
            }
            else
            {
                using var narrowed = normed.Narrow(0, seqLen - 1, 1);
                headIn = Ops.NewContiguous(narrowed);
            }
            normed.Dispose();

            Tensor logitsTensor = LinearForward(headIn, "output.weight")
                                  ?? LinearForward(headIn, "token_embd.weight");
            headIn.Dispose();

            if (logitsOut != null)
            {
                int rows = allLogitsRows ? seqLen : 1;
                CopyRowsToBuffer(logitsTensor, logitsOut, rows, Config.VocabSize);
            }
            logitsTensor.Dispose();

            _cacheSeqLen += seqLen;
        }

        /// <summary>
        /// One draft step: consume <paramref name="token"/> (at trunk position
        /// <paramref name="pos"/>, not yet in the trunk cache) together with the
        /// hidden state of the token before it, and predict the token after it.
        /// <paramref name="hOut"/> receives the draft block's own hidden state so
        /// the next step in the window can chain from it.
        /// </summary>
        public void DraftStep(int token, float[] hPrev, int pos, float[] logitsOut, float[] hOut)
        {
            RequireMtp();
            ArgumentNullException.ThrowIfNull(hPrev);
            if (UsesNativeExecutor)
            {
                MtpDraftStepNative(token, hPrev, pos, logitsOut, hOut);
                return;
            }
            RunMtpBlock(new[] { token }, hPrev, pos, logitsOut, hOut, wantLogits: true);
        }

        /// <summary>
        /// Replay verified trunk tokens through the draft block so its KV cache
        /// tracks the real context. Row k of <paramref name="hRows"/> is the trunk
        /// hidden state of the token PRECEDING <c>tokens[k]</c>.
        /// </summary>
        public void DraftCatchUp(int[] tokens, float[] hRows, int startPos)
        {
            RequireMtp();
            ArgumentNullException.ThrowIfNull(tokens);
            ArgumentNullException.ThrowIfNull(hRows);
            if (tokens.Length == 0)
                return;
            if (UsesNativeExecutor)
            {
                MtpCatchUpNative(tokens, hRows, startPos);
                return;
            }
            RunMtpBlock(tokens, hRows, startPos, null, null, wantLogits: false);
        }

        // ---- the draft block ------------------------------------------------

        /// <summary>
        /// The NextN block over <paramref name="tokens"/> starting at
        /// <paramref name="startPos"/>. Writes this window's MLA rows into the
        /// draft block's own cache; fills <paramref name="hOut"/> (one row per
        /// token, or just the last when the buffer only holds one) and, when
        /// <paramref name="wantLogits"/>, the LAST row's logits.
        /// </summary>
        private void RunMtpBlock(int[] tokens, float[] hRows, int startPos,
                                 float[] logitsOut, float[] hOut, bool wantLogits)
        {
            int seqLen = tokens.Length;
            int hidden = Config.HiddenSize;
            if (startPos + seqLen > _maxContextLength)
                throw new InvalidOperationException(
                    $"MTP window [{startPos}, {startPos + seqLen}) exceeds max context {_maxContextLength}.");
            EnsureCacheCapacity(startPos + seqLen);

            string[] wn = _layerNames[_mtpLayer];

            // eh_proj( [ enorm(embed(t)) ; hnorm(h_prev) ] )
            Tensor eNorm;
            using (Tensor emb = MtpEmbedding(tokens))
                eNorm = Ops.RMSNorm(null, emb, _mtpEnormW, null, Config.Eps);

            Tensor hNorm;
            using (Tensor hPrev = MtpHiddenTensor(hRows, seqLen, hidden))
                hNorm = Ops.RMSNorm(null, hPrev, _mtpHnormW, null, Config.Eps);

            Tensor cur;
            using (Tensor cat = Ops.Concat(null, 1, eNorm, hNorm))
            {
                eNorm.Dispose();
                hNorm.Dispose();
                cur = LinearForward(cat, wn[WN_NEXTN_EH_PROJ]);
            }
            Trace("mtp_eh_proj", _mtpLayer, cur);

            // ... one ordinary glm-dsa decoder block, with the indexer left alone.
            using (Tensor normed = RMSNormOp(cur, wn[WN_ATTN_NORM]))
            using (Tensor attn = Attention(normed, _mtpLayer, wn, seqLen, startPos, dense: true))
                Ops.Add(cur, cur, attn);
            Trace("mtp_ffn_inp", _mtpLayer, cur);

            using (Tensor normed2 = RMSNormOp(cur, wn[WN_FFN_NORM]))
            using (Tensor ffn = _mtpLayer < _numDenseLead
                                    ? DenseFFN(normed2, wn, seqLen)
                                    : MoEBlock(normed2, _mtpLayer, wn, seqLen))
                Ops.Add(cur, cur, ffn);

            // shared_head_norm seeds the next draft step AND feeds the LM head.
            Tensor headIn = _mtpHeadNormW != null
                ? Ops.RMSNorm(null, cur, _mtpHeadNormW, null, Config.Eps)
                : RMSNormOp(cur, "output_norm.weight");
            cur.Dispose();
            Trace("mtp_h_nextn", _mtpLayer, headIn);

            if (hOut != null)
            {
                int rows = Math.Min(seqLen, hOut.Length / hidden);
                if (rows >= seqLen)
                {
                    CopyRowsToBuffer(headIn, hOut, seqLen, hidden);
                }
                else if (rows >= 1)
                {
                    // Contract: a buffer too small for one row per token gets the
                    // LAST row (the state the next window chains from).
                    using var last = headIn.Narrow(0, seqLen - 1, 1);
                    CopyRowsToBuffer(last, hOut, 1, hidden);
                }
            }

            if (wantLogits && logitsOut != null)
            {
                Tensor lastRow;
                if (seqLen == 1)
                {
                    lastRow = headIn.CopyRef();
                }
                else
                {
                    using var narrowed = headIn.Narrow(0, seqLen - 1, 1);
                    lastRow = Ops.NewContiguous(narrowed);
                }
                using (Tensor logits = LinearForward(lastRow, _mtpHeadName))
                    CopyRowsToBuffer(logits, logitsOut, 1, Config.VocabSize);
                lastRow.Dispose();
            }

            headIn.Dispose();
            _mtpCacheSeqLen = Math.Max(_mtpCacheSeqLen, startPos + seqLen);
        }

        /// <summary>
        /// Token embedding for the draft block. GLM-5.2 shares the trunk table;
        /// a checkpoint that ships <c>nextn.embed_tokens</c> uses that instead.
        /// </summary>
        private Tensor MtpEmbedding(int[] tokens)
        {
            if (_mtpEmbdF32 != null)
            {
                using var indices = CreateIntTensor(tokens, tokens.Length);
                return Ops.IndexSelect(null, _mtpEmbdF32, indices);
            }
            string[] wn = _layerNames[_mtpLayer];
            if (_quantWeights.ContainsKey(wn[WN_NEXTN_EMBED_TOKENS]))
            {
                // Never silently fall back to the trunk table: that is a different
                // embedding and the drafts would be wrong in a way only an
                // acceptance-rate collapse would reveal.
                throw new NotSupportedException(
                    $"{wn[WN_NEXTN_EMBED_TOKENS]} is quantized; the per-op MTP path needs it dequantized. " +
                    "Run this checkpoint on a GGML backend (native executor) or requantize the tensor to F32.");
            }
            return Embedding(tokens);
        }

        /// <summary>Wrap the first <c>rows*cols</c> floats of <paramref name="src"/>
        /// as a [rows, cols] tensor.</summary>
        private Tensor MtpHiddenTensor(float[] src, int rows, int cols)
        {
            long need = (long)rows * cols;
            if (src.Length < need)
                throw new ArgumentException(
                    $"MTP hidden buffer holds {src.Length} floats; {need} are needed for {rows} rows.", nameof(src));
            if (src.Length == need)
                return CreateFloatTensor(src, rows, cols);

            if (_mtpHScratch == null || _mtpHScratch.Length != need)
                _mtpHScratch = new float[need];
            Array.Copy(src, _mtpHScratch, need);
            return CreateFloatTensor(_mtpHScratch, rows, cols);
        }

        /// <summary>Copy the first <paramref name="rows"/> rows of
        /// <paramref name="t"/> into <paramref name="dst"/>.</summary>
        private void CopyRowsToBuffer(Tensor t, float[] dst, int rows, int cols)
        {
            long need = (long)rows * cols;
            if (dst.Length < need)
                throw new ArgumentException(
                    $"Destination buffer holds {dst.Length} floats; {need} are needed.", nameof(dst));
            float[] flat = TensorToFloatArray(t);
            Array.Copy(flat, 0, dst, 0, need);
        }

        private bool HasAnyWeight(string name)
            => _weights.ContainsKey(name) || _quantWeights.ContainsKey(name);

        private void RequireMtp()
        {
            if (!HasDraftHead)
                throw new InvalidOperationException("This glm-dsa checkpoint has no NextN/MTP draft block.");
        }
    }
}
