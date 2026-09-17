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
// ---------------------------------------------------------------------------
// Muse-Glimmer's half of DFlash speculative drafting.
//
// The drafter itself - loading, the KV ring, the encoder, the KV injection and
// the block draft, in both the per-op and the fused flavour - is shared and
// lives in TensorSharp.Models/Speculative/ModelBase.DFlash*.cs. What is left
// here is the part that genuinely belongs to this target model:
//
//   * the trunk forward that captures the per-layer residuals the drafter's
//     encoder consumes (SpecForward / SpecForwardCore), and
//   * the ISpeculativeModel wiring that points the shared draft/verify loop at
//     the shared drafter.
//
// Rollback is free here: the trunk verify writes correct KV for every row it
// processes into the linear (non-circular) cache and Muse-Glimmer has no
// recurrent state, so partial acceptance only rewinds the position counter
// (SpecVerifyPersistsAcceptedKv).
// ---------------------------------------------------------------------------
using System;
using System.Diagnostics;
using TensorSharp;
using TensorSharp.GGML;
using TensorSharp.MLX;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;

using TensorSharp.Runtime.Speculative;

namespace TensorSharp.Models
{
    public partial class MuseGlimmerModel : ModelBase, ISpeculativeModel
    {
        /// <summary>Path of the DFlash drafter GGUF, if one was configured
        /// (<c>--draft-model</c> / <c>TS_MUSE_GLIMMER_DFLASH</c>).</summary>
        internal static string ResolveDFlashPath(string explicitPath)
        {
            string path = !string.IsNullOrWhiteSpace(explicitPath)
                ? explicitPath
                : Environment.GetEnvironmentVariable("TS_MUSE_GLIMMER_DFLASH");
            return string.IsNullOrWhiteSpace(path) ? null : path;
        }

        /// <summary>The trunk's own ceiling on one speculative prefill chunk: its
        /// ForwardCore throws outright on a batch wider than the SWA ring can
        /// hold.</summary>
        private protected override int DFlashTrunkPrefillChunkCap
            => _kvSwaRows > 0 ? _kvSwaRows - _slidingWindow : int.MaxValue;

        /// <summary>Called from <see cref="Dispose"/> in MuseGlimmerModel.cs.</summary>
        private void DisposeDFlashDrafter() => DisposeDFlash();

        // ====================================================================
        // ISpeculativeModel
        // ====================================================================

        public bool HasDraftHead => HasDFlash;

        /// <summary>DFlash proposes a whole block per pass, so it is served by
        /// <see cref="BlockDraftSpeculator"/>.</summary>
        public DraftHeadKind DraftHeadKind => HasDFlash ? DraftHeadKind.Block : DraftHeadKind.None;

        /// <summary>Speculation is profitable exactly when a drafter is loaded: the
        /// verify batch is a single ordinary multi-token Muse-Glimmer forward (the
        /// same batched-attention path a prefill chunk takes), so it amortises the
        /// 30B weight read over the whole window on every backend the model runs
        /// on.</summary>
        // Gated on the drafter because SpecForward captures DFlash's own feature
        // rows (RequireDFlash) and has no meaning without it, so a weight-free
        // speculator has no trunk to verify with here.
        public bool SpeculationProfitable => HasDFlash;

        /// <summary>The concatenated target-layer input residuals the encoder
        /// consumes (5 * 6656 = 33280), NOT the model's hidden size.</summary>
        public int SpecFeatureSize => _dflash != null ? _dflash.FeatureSize : Config.HiddenSize;

        /// <summary>Number of DRAFTS a block produces, i.e. block_size - 1: row 0 of
        /// the block is the anchor's own prediction and DFlash discards it.</summary>
        public int DraftBlockSize => HasDFlash ? _dflash.MaxDraftTokens : 0;

        /// <summary>See <c>ModelBase.DFlashPrefillChunkSize</c>.</summary>
        public int SpecPrefillChunkSize => DFlashPrefillChunkSize;

        /// <summary>The drafter is fed from the host: every prefill chunk hands its
        /// per-row features back so <see cref="DraftCatchUp"/> can fill the ring.</summary>
        public bool DraftSelfCatchUp => false;

        /// <summary>The verify batch writes correct KV for every row at its true
        /// position into the linear cache, and Muse-Glimmer holds no recurrent
        /// state, so a partial acceptance only needs a position rewind.</summary>
        public bool SpecVerifyPersistsAcceptedKv => true;

        public int DraftBlock(int lastToken, float[] hPrev, int position, int[] draftOut, float[] confOut)
            => DFlashPropose(lastToken, hPrev, position, draftOut, confOut);

        public void DraftCatchUp(int[] tokens, float[] hRows, int startPos)
            => DFlashCommit(tokens, hRows, startPos);

        /// <summary>DFlash drafts whole blocks; the per-token entry point is never
        /// used.</summary>
        public void DraftStep(int token, float[] hPrev, int pos, float[] logitsOut, float[] hOut)
            => throw new NotSupportedException("Muse-Glimmer DFlash drafts whole blocks; use DraftBlock.");

        /// <summary>Pre-grows the trunk KV cache to cover the whole speculative
        /// window. The drafter's ring is fixed-size and needs nothing.</summary>
        public void SpecEnsureCapacity(int requiredSeqLen)
        {
            if (!HasDFlash)
                return;
            EnsureCacheCapacity(requiredSeqLen);
        }

        /// <summary>No recurrent (GDN/SSM) state in Muse-Glimmer -- drafting and
        /// verifying are stateless given the KV cache.</summary>
        public void SpecSnapshotRecurrentState() { }

        /// <summary>See <see cref="SpecSnapshotRecurrentState"/>.</summary>
        public void SpecRestoreRecurrentState() { }

        /// <summary>
        /// Rewinds the trunk KV position counter after rejected speculative tokens.
        /// Rows past <paramref name="length"/> are overwritten by later writes and
        /// the causal/SWA mask never reads past the live position, so no data moves.
        /// The drafter's ring is untouched: it only ever holds committed positions,
        /// and the next catch-up rewrites the ones a rejected tail would have needed.
        /// </summary>
        public void SpecRewindCache(int length)
        {
            if (length < 0 || length > _cacheSeqLen)
            {
                throw new ArgumentOutOfRangeException(nameof(length),
                    $"Rewind length {length} outside [0, {_cacheSeqLen}].");
            }
            NoteHeadBeforeRewind(length);
            _cacheSeqLen = length;
            _cachedSWAMaskStartPos = -1;
        }

        /// <summary>
        /// Trunk forward that additionally captures, per row, the concatenated
        /// INPUT residuals of the target layers in dflash.target_layers (row stride
        /// <see cref="SpecFeatureSize"/>) and, with <paramref name="allLogitsRows"/>,
        /// LM-head logits for every row instead of only the last. Advances the KV
        /// cache exactly like Forward().
        /// </summary>
        public void SpecForward(int[] tokens, float[] hAllOut, float[] logitsOut, bool allLogitsRows)
        {
            if (!HasDFlash)
                throw new InvalidOperationException("No DFlash drafter is loaded for this Muse-Glimmer model.");
            if (tokens == null || tokens.Length == 0)
                throw new ArgumentException("Token batch must not be empty.", nameof(tokens));
            // No lock here: like Gemma4Model.SpecForward, the caller owns
            // model.GpuComputeLock for the whole speculative step.
            SpecForwardCore(tokens, hAllOut, logitsOut, allLogitsRows);
        }

        /// <summary>
        /// ForwardCore() with a feature-capture hook in the layer loop and an
        /// optional all-rows LM head. Kept as a separate copy of the loop (rather
        /// than a hook inside MuseGlimmerModel.ForwardCore) so the non-speculative
        /// decode path stays byte-identical and branch-free.
        /// </summary>
        private unsafe void SpecForwardCore(int[] tokens, float[] hAllOut, float[] logitsOut, bool allLogitsRows)
        {
            _forwardSw.Start();
            int seqLen = tokens.Length;
            int startPos = _cacheSeqLen;
            int feat = _dflash.FeatureSize;
            EnsureCacheCapacity(startPos + seqLen);

            // Buffer-size contract (ISpeculativeModel.SpecForward): a hidden
            // buffer too small for one row per token means the caller only wants
            // the LAST row, written to row 0.
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

            long t0 = Stopwatch.GetTimestamp();
            Tensor hidden = Embedding(tokens);
            _embTicks += Stopwatch.GetTimestamp() - t0;

            DrainPendingVisionEmbeddings(hidden, startPos);
            ApplyInputEmbeddingNorm(hidden, seqLen);

            // Fused whole-model trunk. The kernel emits the same per-layer input
            // residuals the loop below captures by hand, so speculation runs on the
            // fast path instead of forcing the per-op forward just to observe them.
            if (TryFusedSpecForward(hidden, seqLen, startPos, hAllOut, logitsOut, allLogitsRows, captureAll, captureLast))
            {
                hidden.Dispose();
                _cacheSeqLen += seqLen;
                _forwardCount++;
                _forwardSw.Stop();
                return;
            }

            EnsureFusedKvHostSynchronized();

            for (int l = 0; l < Config.NumLayers; l++)
            {
                // res->t_layer_inp[l]: the residual ENTERING layer l (== the output
                // of layer l-1). Captured before the block runs.
                int slot = _dflashCaptureSlot[l];
                if (slot >= 0 && (captureAll || captureLast))
                    DFlashCaptureFeature(hidden, slot, seqLen, hAllOut, captureLast);

                hidden = TransformerBlock(hidden, l, seqLen, startPos);
                if (_backend == BackendType.Mlx && (l + 1) % MlxEvalEveryNLayers == 0
                    && l + 1 != Config.NumLayers && hidden != null)
                {
                    MlxFusedOps.TryAsyncEvaluate(hidden);
                }
            }

            Tensor normed = RMSNormOp(hidden, "output_norm.weight");
            hidden.Dispose();

            string outputWeight = DFlashTargetOutputWeightName;
            if (allLogitsRows)
            {
                t0 = Stopwatch.GetTimestamp();
                Tensor logitsTensor = LinearForward(normed, outputWeight);
                _lmHeadTicks += Stopwatch.GetTimestamp() - t0;
                normed.Dispose();

                ApplyOutputTransform(logitsTensor);

                t0 = Stopwatch.GetTimestamp();
                long need = (long)seqLen * Config.VocabSize;
                if (logitsOut == null || logitsOut.LongLength < need)
                    throw new ArgumentException($"Per-row logits need {need} floats.", nameof(logitsOut));
                float* src = GetFloatPtr(logitsTensor);
                fixed (float* dst = logitsOut)
                    Buffer.MemoryCopy(src, dst, logitsOut.LongLength * 4L, need * 4L);
                _logitsCopyTicks += Stopwatch.GetTimestamp() - t0;
                logitsTensor.Dispose();
            }
            else
            {
                Tensor lastHidden;
                if (seqLen > 1)
                {
                    using var lastRow = normed.Narrow(0, seqLen - 1, 1);
                    lastHidden = Ops.NewContiguous(lastRow);
                }
                else
                {
                    lastHidden = normed.CopyRef();
                }
                normed.Dispose();

                t0 = Stopwatch.GetTimestamp();
                Tensor logitsTensor = LinearForward(lastHidden, outputWeight);
                _lmHeadTicks += Stopwatch.GetTimestamp() - t0;
                lastHidden.Dispose();

                ApplyOutputTransform(logitsTensor);

                t0 = Stopwatch.GetTimestamp();
                _logitsBuffer = TensorToFloatArray(logitsTensor);
                _logitsCopyTicks += Stopwatch.GetTimestamp() - t0;
                logitsTensor.Dispose();
                if (logitsOut != null)
                    Array.Copy(_logitsBuffer, logitsOut, Config.VocabSize);
            }

            _cacheSeqLen += seqLen;
            _forwardCount++;
            _forwardSw.Stop();
        }

        /// <summary>
        /// SpecForward on the fused kernel. The kernel writes the captured residuals
        /// BLOCK-major (one [hidden, n_tokens] block per target layer); the
        /// ISpeculativeModel contract wants them ROW-major (per token, the blocks
        /// concatenated into one FeatureSize-wide row), so they are transposed on the
        /// way out. That is 5 x 6656 x n floats of host copying against a whole trunk
        /// forward - negligible, and it keeps the contract unchanged.
        /// </summary>
        private bool TryFusedSpecForward(Tensor hidden, int seqLen, int startPos,
            float[] hAllOut, float[] logitsOut, bool allLogitsRows, bool captureAll, bool captureLast)
        {
            int nCap = _dflash.TargetLayerIds.Length;
            int hs = Config.HiddenSize;
            int feat = _dflash.FeatureSize;
            bool wantCapture = captureAll || captureLast;

            if (wantCapture)
            {
                long need = (long)nCap * hs * seqLen;
                if (_specCaptureScratch == null || _specCaptureScratch.LongLength < need)
                    _specCaptureScratch = new float[need];
            }

            float[] produced = TryFusedForward(hidden, seqLen, startPos,
                wantCapture ? _specCaptureScratch : null,
                wantCapture ? _dflash.TargetLayerIds : null,
                null, allLogitsRows, allLogitsRows ? logitsOut : null);
            if (produced == null)
                return false;

            if (wantCapture)
            {
                for (int c = 0; c < nCap; c++)
                {
                    int slot = c;                       // capture_layers order == slot order
                    long blockBase = (long)c * hs * seqLen;
                    if (captureLast)
                    {
                        Array.Copy(_specCaptureScratch, blockBase + (long)(seqLen - 1) * hs,
                            hAllOut, (long)slot * hs, hs);
                        continue;
                    }
                    for (int r = 0; r < seqLen; r++)
                    {
                        Array.Copy(_specCaptureScratch, blockBase + (long)r * hs,
                            hAllOut, (long)r * feat + (long)slot * hs, hs);
                    }
                }
            }

            if (!allLogitsRows)
            {
                _logitsBuffer = produced;
                if (logitsOut != null)
                    Array.Copy(produced, logitsOut, Config.VocabSize);
            }
            return true;
        }

        private float[] _specCaptureScratch;
    }
}
