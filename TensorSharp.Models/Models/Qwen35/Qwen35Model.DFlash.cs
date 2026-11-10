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
// Qwen 3.5 / 3.8 as a DFlash (and DFlash2) speculation TARGET.
//
// The drafter itself is shared - see ModelBase.DFlash*.cs. What is here is the
// half that only this trunk can provide: the per-layer residuals the drafter's
// encoder was trained on, tapped out of a forward that otherwise runs exactly
// like the non-speculative one.
//
// The GGUF names those layers in dflash.target_layers (already +1-shifted by the
// converter, so id L means "the residual ENTERING 0-based layer L" == "the output
// of layer L-1"), and the drafter's fc projects the concatenation of them.
//
// A Qwen 3.8 checkpoint can carry BOTH drafters: a NextN/MTP block inside the
// trunk GGUF and an external DFlash file. When a DFlash drafter is attached it
// wins, because it is the one the operator explicitly asked for with
// --draft-model, and because the two cannot be mixed - they consume different
// hidden rows (SpecFeatureSize) and drive different speculators.
// ---------------------------------------------------------------------------
using System;
using System.Diagnostics;
using TensorSharp;
using TensorSharp.GGML;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Speculative;

namespace TensorSharp.Models
{
    public partial class Qwen35Model
    {
        /// <summary>
        /// Path of a DFlash drafter GGUF, if one was configured
        /// (<c>--draft-model</c>) AND it really is one.
        /// The same flag also names an MTP-only file for other architectures, so
        /// the architecture string decides rather than the extension.
        /// </summary>
        internal static string ResolveQwen35DFlashPath(string path)
        {
            if (string.IsNullOrWhiteSpace(path) || !System.IO.File.Exists(path))
                return null;
            try
            {
                using var probe = new GgufFile(path);
                string arch = probe.GetString("general.architecture") ?? string.Empty;
                return string.Equals(arch, DFlashConfig.ArchName, StringComparison.Ordinal) ? path : null;
            }
            catch
            {
                // A file we cannot open is not a drafter; the loader below will say
                // so if the operator really meant it.
                return null;
            }
        }

        /// <summary>
        /// Attach a DFlash drafter to this trunk. Called from the constructor after
        /// the trunk weights and caches exist, because the drafter's tensors are
        /// merged into the same dictionaries and its ring is allocated from the same
        /// allocator.
        /// </summary>
        private void TryLoadQwen35DFlash(string draftModelPath)
        {
            string path = ResolveQwen35DFlashPath(draftModelPath);
            if (path == null)
                return;

            if (IsTensorParallel)
            {
                // The drafter borrows the trunk's LM head and token embedding, both
                // of which are sharded under TP, and its capture path runs the
                // single-device fused verify. Refuse rather than draft from a shard.
                Console.WriteLine("  DFlash speculative decoding is not supported under tensor parallelism; ignoring the drafter.");
                return;
            }

            try
            {
                LoadDFlashDraftWeights(path);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"  DFlash drafter '{System.IO.Path.GetFileName(path)}' could not be attached: {ex.Message}");
                return;
            }

            if (HasDFlash && _numNextnLayers > 0)
                Console.WriteLine("  DFlash drafter attached; the checkpoint's NextN/MTP block will not be used.");
        }

        /// <summary>Number of DRAFTS a block produces (block_size - 1: row 0 is the
        /// anchor's own prediction, which DFlash discards).</summary>
        public int DraftBlockSize => HasDFlash ? _dflash.MaxDraftTokens : 0;

        /// <summary>A DSpine block is causal: a step drafts only the rows it verifies.</summary>
        public bool DraftBlockIsCausal => HasDFlash && _dflash.CausalBlock;

        /// <summary>The DFlash ring: older prompt rows are overwritten before any draft.</summary>
        public int DraftContextRows => HasDFlash && DFlashRingRows > 0 ? DFlashRingRows : int.MaxValue;

        /// <summary>One whole-block draft. See <c>ModelBase.DFlashPropose</c>.</summary>
        public int DraftBlock(int lastToken, float[] hPrev, int position, int[] draftOut, float[] confOut)
        {
            // A batched step drafts beside a verify that reads the holders directly;
            // entering the single-sequence session would drop every decode graph.
            if (!_batchedSpecScope)
                EnterSpecSession();
            return DFlashPropose(lastToken, hPrev, position, draftOut, confOut);
        }

        // ====================================================================
        // the trunk's half: capturing dflash.target_layers residuals
        // ====================================================================

        private float[] _dflashCaptureScratch;

        /// <summary>
        /// SpecForward for a DFlash-drafted trunk: the ordinary verify forward plus
        /// one [hidden, n] block per entry of dflash.target_layers, transposed into
        /// the ROW-major FeatureSize-wide layout the drafter's encoder consumes.
        ///
        /// Returns false when the fused kernel declines the shape, in which case the
        /// caller runs the op-by-op loop (which taps the same residuals by hand).
        /// </summary>
        private bool TryDFlashSpecForwardFused(Tensor hidden, int startPos, int seqLen,
            float[] hAllOut, float[] logitsOut, bool allLogitsRows, bool captureAll, bool captureLast)
        {
            int nCap = _dflash.TargetLayerIds.Length;
            int hs = Config.HiddenSize;
            int feat = _dflash.FeatureSize;
            bool wantCapture = captureAll || captureLast;

            if (wantCapture)
            {
                long need = (long)nCap * hs * seqLen;
                if (_dflashCaptureScratch == null || _dflashCaptureScratch.LongLength < need)
                    _dflashCaptureScratch = new float[need];
            }

            int vocab = Config.VocabSize;
            float[] logitTarget;
            int logitRows;
            if (allLogitsRows)
            {
                if (logitsOut != null && logitsOut.LongLength >= (long)seqLen * vocab)
                {
                    logitTarget = logitsOut;
                }
                else
                {
                    if (_fvLogitsAllBuf == null || _fvLogitsAllBuf.LongLength < (long)seqLen * vocab)
                        _fvLogitsAllBuf = new float[(long)seqLen * vocab];
                    logitTarget = _fvLogitsAllBuf;
                }
                logitRows = -1;
            }
            else
            {
                // Last-row-only callers (prompt prefill chunks, rollback re-forwards)
                // ask the kernel for ONE logit row: a whole-prompt lm_head is
                // vocab*N floats of device and host memory for rows nobody reads.
                if (_fvLogitsLastBuf == null || _fvLogitsLastBuf.Length < vocab)
                    _fvLogitsLastBuf = new float[vocab];
                logitTarget = _fvLogitsLastBuf;
                logitRows = 1;
            }

            // A last-row-only call is a prompt chunk or a rollback re-forward: neither is
            // rolled back again, so its recurrent state may stay on the device.
            if (!TryFullModelVerify(hidden, startPos, seqLen, normedOut: null, logitsOut: logitTarget,
                    nLogitRows: logitRows, rowOffset: 0,
                    captureData: wantCapture ? _dflashCaptureScratch : null,
                    captureLayers: wantCapture ? _dflash.TargetLayerIds : null,
                    keepDeviceState: !allLogitsRows))
            {
                return false;
            }

            if (wantCapture)
            {
                for (int c = 0; c < nCap; c++)
                {
                    long blockBase = (long)c * hs * seqLen;
                    if (captureLast)
                    {
                        Array.Copy(_dflashCaptureScratch, blockBase + (long)(seqLen - 1) * hs,
                            hAllOut, (long)c * hs, hs);
                        continue;
                    }
                    for (int r = 0; r < seqLen; r++)
                    {
                        Array.Copy(_dflashCaptureScratch, blockBase + (long)r * hs,
                            hAllOut, (long)r * feat + (long)c * hs, hs);
                    }
                }
            }

            if (!allLogitsRows && logitsOut != null)
                Array.Copy(logitTarget, logitsOut, vocab);
            return true;
        }

        /// <summary>
        /// Op-by-op SpecForward with the DFlash residual taps, for the shapes and
        /// backends the fused kernel declines. Same layer loop as
        /// <see cref="SpecForward"/>'s fallback, with the capture hook in front of
        /// each block.
        /// </summary>
        private unsafe void DFlashSpecForwardPerOp(Tensor hidden, int startPos, int seqLen,
            float[] hAllOut, float[] logitsOut, bool allLogitsRows, bool captureAll, bool captureLast)
        {
            // TryFullModelVerify may have declined while a prior chunk's recurrent
            // state was still device-authoritative. The capture/per-op path reads
            // host mirrors, so drain through the same fail-safe prefill barrier.
            PrepareHostPrefillFallback();
            for (int layer = 0; layer < Config.NumLayers; layer++)
            {
                int slot = _dflashCaptureSlot[layer];
                if (slot >= 0 && (captureAll || captureLast))
                    DFlashCaptureFeature(hidden, slot, seqLen, hAllOut, captureLast);

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

            Tensor normed = RMSNormOpCached(hidden, _finalNormW);
            hidden.Dispose();

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
        }
    }
}
