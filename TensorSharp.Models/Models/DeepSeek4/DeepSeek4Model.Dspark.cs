// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// DSpark speculative decoding for DeepSeek V4: the model-side contract over the
// direct-CUDA engine's drafter (see Dsv4CudaEngine.Dspark.cs for the algorithm).
//
// DSpark is a BLOCK drafter -- one forward pass proposes the whole speculative
// window -- so it plugs into the shared draft/verify/rollback core through
// ISpeculativeModel.DraftBlock instead of the per-token DraftStep, and
// reports the wider hidden row (the concatenated target-layer features its
// main_proj consumes) through SpecFeatureSize.
//
// Rollback retains accepted KV: the verify pass wrote correct KV for every token it
// processed, and the engine's rings are sized so a rejected tail can never
// alias a row a later pass still reads, so partial acceptance only rewinds the
// position counter (SpecVerifyPersistsAcceptedKv).
using System;
using TensorSharp.GGML;
using TensorSharp.Runtime.Scheduling;

using TensorSharp.Runtime.Speculative;

namespace TensorSharp.Models
{
    public partial class DeepSeek4Model : ISpeculativeModel
    {
        /// <summary>Path of the DSpark drafter GGUF, if one was configured
        /// (<c>--draft-model</c> / <c>TS_DSV4_DSPARK</c>).</summary>
        internal static string ResolveDsparkPath(string explicitPath)
        {
            string path = !string.IsNullOrWhiteSpace(explicitPath)
                ? explicitPath
                : Environment.GetEnvironmentVariable("TS_DSV4_DSPARK");
            return string.IsNullOrWhiteSpace(path) ? null : path;
        }

        /// <summary>True when a DSpark drafter is loaded and usable, on either
        /// executor: the direct-CUDA engine or the native ggml one.</summary>
        public bool HasDraftHead => (_cudaExec != null && _cudaExec.HasDspark) || _nativeDsparkBlock > 0;

        /// <summary>DSpark proposes a whole block per pass, so it is served by
        /// <see cref="BlockDraftSpeculator"/>.</summary>
        public DraftHeadKind DraftHeadKind => HasDraftHead ? DraftHeadKind.Block : DraftHeadKind.None;

        /// <summary>Both DSpark implementations run the batched verify on their
        /// own accelerated path.</summary>
        // Gated on the drafter because SpecForward runs the DSpark-aware executor
        // path (RequireDspark) and has no meaning without it, so a weight-free
        // speculator has no trunk to verify with here.
        public bool SpeculationProfitable => HasDraftHead;

        /// <summary>
        /// Width of the hidden row handed between the trunk and the drafter. The
        /// direct-CUDA engine feeds the drafter its target features through the
        /// host, so it reports their real width; the ggml engine commits the
        /// drafter key ring inside the trunk graph and needs nothing from the
        /// host, so its rows are a single (unused) float.
        /// </summary>
        public int SpecFeatureSize => _cudaExec != null && _cudaExec.HasDspark ? _cudaExec.DsparkFeatureSize : 1;

        public int DraftBlockSize => _cudaExec != null ? _cudaExec.DsparkBlockSize : _nativeDsparkBlock;

        /// <summary>Prefill in whole engine micro-batches: the MoE trunk reads
        /// every touched expert's weights once per micro-batch, so halving the
        /// chunk nearly doubles prefill weight traffic per token.</summary>
        public int SpecPrefillChunkSize => _cudaExec != null ? _cudaExec.UBatch : _nativeUBatch;

        /// <summary>Both engines replay the drafter key ring from their own
        /// on-device features, so prefill needs no per-row readback.</summary>
        public bool DraftSelfCatchUp => true;

        /// <summary>The verify batch writes reusable KV for every row, and the
        /// engine's rings tolerate a rejected tail, so partial acceptance needs
        /// no re-forward.</summary>
        public bool SpecVerifyPersistsAcceptedKv => true;

        /// <summary>Trunk position tracked by the executor (the base class's
        /// counter is not used by the DSV4 executors).</summary>
        public new int CacheSeqLen => _cudaExec != null
            ? _cudaExec.NPast
            : (_handle != IntPtr.Zero ? GgmlDeepSeek4Native.NPast(_handle) : base.CacheSeqLen);

        public void SpecForward(int[] tokens, float[] hAllOut, float[] logitsOut, bool allLogitsRows)
        {
            RequireDspark();
            lock (_sync)
            {
                if (_cudaExec != null)
                {
                    _cudaExec.ForwardSpec(tokens, hAllOut, logitsOut, allLogitsRows);
                    return;
                }

                // The native executor emits per-row logits only for the verify
                // batch; a prefill chunk takes the ordinary forward, which also
                // commits the drafter key ring, and reports the last row.
                bool ok = allLogitsRows
                    ? GgmlDeepSeek4Native.ForwardSpec(_handle, tokens, logitsOut)
                    : GgmlDeepSeek4Native.Forward(_handle, tokens, logitsOut);
                if (!ok)
                    throw new InvalidOperationException("DeepSeek V4 native speculative forward failed (see stderr).");
                if (hAllOut != null && hAllOut.Length > 0)
                    Array.Clear(hAllOut, 0, hAllOut.Length);   // the drafter reads its features on-device
            }
        }

        public int DraftBlock(int lastToken, float[] hPrev, int position, int[] draftOut, float[] confOut)
        {
            RequireDspark();
            lock (_sync)
            {
                if (_cudaExec != null)
                    return _cudaExec.DsparkDraft(lastToken, hPrev, position, draftOut, confOut);
                return GgmlDeepSeek4Native.DsparkDraft(_handle, lastToken, draftOut, confOut);
            }
        }

        /// <summary>Row k of <paramref name="hRows"/> is the hidden state of the
        /// token BEFORE tokens[k], i.e. the features of position startPos+k-1 --
        /// which is exactly the position whose drafter key it writes.</summary>
        public void DraftCatchUp(int[] tokens, float[] hRows, int startPos)
        {
            RequireDspark();
            lock (_sync)
            {
                // Only the direct-CUDA drafter is fed from the host; the native
                // one commits its ring inside the trunk graph.
                if (_cudaExec != null)
                    _cudaExec.DsparkCatchUp(hRows, tokens.Length, startPos - 1);
            }
        }

        public void SpecRewindCache(int length)
        {
            RequireDspark();
            lock (_sync)
            {
                if (_cudaExec != null)
                {
                    _cudaExec.Rewind(length);
                }
                // The native rewind refuses whatever it cannot serve, and a refusal that
                // is dropped here leaves the drafter's rejected tail in the cache with
                // every later token attending to it - fluent output from the wrong
                // positions, with nothing in the log to say so.
                else if (!GgmlDeepSeek4Native.Rewind(_handle, length))
                {
                    throw new InvalidOperationException(
                        $"DeepSeek {Config.Architecture} refused a speculative cache rewind to {length} tokens.");
                }
            }
        }

        /// <summary>DSpark drafts a whole block per step; the per-token draft
        /// entry point is never used.</summary>
        public void DraftStep(int token, float[] hPrev, int pos, float[] logitsOut, float[] hOut)
            => throw new NotSupportedException("DeepSeek V4 drafts whole blocks; use DraftBlock.");

        // Native DSpark pads the modular compressor rings by the complete
        // draft width. V4.1 also bounds Rewind to the last successful verify
        // and shrinks Engram history. This retains the accepted prefix at
        // partial compression boundaries without copying every cache per step.
        public void SpecEnsureCapacity(int requiredSeqLen) { }

        public void SpecSnapshotRecurrentState() { }

        public void SpecRestoreRecurrentState() { }

        private void RequireDspark()
        {
            if (!HasDraftHead)
                throw new InvalidOperationException("No DSpark drafter is loaded for this DeepSeek V4 model.");
        }
    }
}
