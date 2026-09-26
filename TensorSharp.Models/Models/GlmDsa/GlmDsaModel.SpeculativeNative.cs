// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// NextN/MTP on the native whole-model executor.
//
// GLM-5.2 is 226 GiB of weights, so on the GGML backends the trunk, the draft
// block and both KV caches live inside ggml_ops_glm_dsa.cpp and the managed
// side is a marshalling layer. The three entry points below mirror the per-op
// implementations in GlmDsaModel.Mtp.cs exactly; what differs is that the
// native side keeps everything device-resident, so a verify batch is ONE graph
// over the whole speculative window and drafting never round-trips the hidden
// state through host memory except at the interface boundary the shared
// draft/verify core requires.
using System;
using TensorSharp.GGML;
using TensorSharp.Runtime.Speculative;

namespace TensorSharp.Models
{
    public partial class GlmDsaModel
    {
        /// <summary>
        /// Whether the operator asked for speculation before this model loaded.
        /// The draft block is a whole extra 256-expert decoder layer (~3 GiB of
        /// GLM-5.2 at IQ2_XXS, and it competes with the KV cache for the same
        /// VRAM the loader sizes the context against), so it is only paged in
        /// when it is going to be used. <c>--spec</c> sets TS_SPEC and the legacy
        /// TS_MTP_SPEC before the startup model loads; both are honoured here so
        /// a deployment exporting the documented TS_SPEC directly does not get a
        /// scheduler that believes speculation is on while the loader never paged
        /// the block in. TS_GLM_MTP overrides either way for A/B runs.
        ///
        /// The two enable variables are parsed by <see cref="SpeculationOptions"/>
        /// itself, not re-derived here: the scheduler decides from the same
        /// resolution whether to speculate, and a private "anything but 0 is on"
        /// rule made <c>TS_SPEC=false</c> page the ~3 GiB block in for a
        /// scheduler that then never drafted with it.
        /// </summary>
        internal static bool NativeMtpRequested()
        {
            string glm = Environment.GetEnvironmentVariable("TS_GLM_MTP");
            if (!string.IsNullOrEmpty(glm))
                return glm != "0";
            return SpeculationOptions.FromEnvironment().Enabled;
        }

        private void SpecForwardNative(int[] tokens, float[] hAllOut, float[] logitsOut, bool allLogitsRows)
        {
            lock (_nativeSync)
            {
                int hidden = Config.HiddenSize;
                int vocab = Config.VocabSize;
                long needH = (long)tokens.Length * hidden;
                long needL = (allLogitsRows ? (long)tokens.Length : 1L) * vocab;

                // The native side writes exactly n*hidden / n*vocab floats, so a
                // caller buffer that is merely large enough is fine but a short one
                // would corrupt the heap. Stage through a scratch array in that case
                // (the plain-decode path deliberately hands a 1-row logits buffer).
                float[] hBuf = hAllOut != null && hAllOut.LongLength >= needH ? hAllOut : EnsureSpecH(needH);
                float[] lBuf = logitsOut != null && logitsOut.LongLength >= needL ? logitsOut : EnsureSpecLogits(needL);

                if (!GgmlGlmNative.SpecForward(_native, tokens, hBuf, lBuf, allLogitsRows))
                    throw new InvalidOperationException("glm-dsa native speculative forward failed (see stderr).");

                if (!ReferenceEquals(hBuf, hAllOut) && hAllOut != null)
                    Array.Copy(hBuf, 0, hAllOut, 0, Math.Min(hAllOut.LongLength, needH));
                if (!ReferenceEquals(lBuf, logitsOut) && logitsOut != null)
                    Array.Copy(lBuf, 0, logitsOut, 0, Math.Min(logitsOut.LongLength, needL));

                _cacheSeqLen = GgmlGlmNative.NPast(_native);
            }
        }

        private void MtpDraftStepNative(int token, float[] hPrev, int pos, float[] logitsOut, float[] hOut)
        {
            lock (_nativeSync)
            {
                float[] lBuf = logitsOut ?? EnsureSpecLogits(Config.VocabSize);
                float[] hBuf = hOut ?? EnsureSpecH(Config.HiddenSize);
                if (!GgmlGlmNative.DraftStep(_native, token, hPrev, pos, lBuf, hBuf))
                    throw new InvalidOperationException("glm-dsa native MTP draft step failed (see stderr).");
            }
        }

        private void MtpCatchUpNative(int[] tokens, float[] hRows, int startPos)
        {
            lock (_nativeSync)
            {
                if (!GgmlGlmNative.DraftCatchUp(_native, tokens, hRows, startPos))
                    throw new InvalidOperationException("glm-dsa native MTP catch-up failed (see stderr).");
            }
        }

        private void RewindNative(int length)
        {
            lock (_nativeSync)
            {
                if (GgmlGlmNative.Rewind(_native, length))
                    _cacheSeqLen = length;
            }
        }

        // Staging buffers for the short-buffer cases above. Grown, never shrunk;
        // a speculative window is at most MtpMaxDraftTokens+1 rows.
        private float[] _specHScratch;
        private float[] _specLogitsScratch;

        private float[] EnsureSpecH(long need)
        {
            if (_specHScratch == null || _specHScratch.LongLength < need)
                _specHScratch = new float[need];
            return _specHScratch;
        }

        private float[] EnsureSpecLogits(long need)
        {
            if (_specLogitsScratch == null || _specLogitsScratch.LongLength < need)
                _specLogitsScratch = new float[need];
            return _specLogitsScratch;
        }
    }
}
