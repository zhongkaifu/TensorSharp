// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// ============================================================================
// CFG-batched denoise forward. The true-CFG denoise runs the 60-block transformer
// TWICE per step (conditional + unconditional). The DiT is launch-bound — every
// per-block GPU sync + host round-trip leaves the GPU idle ~40% (measured ~60%
// util / ~55 W of 350 W). Running BOTH branches through one native dispatch per
// block (TSGgml_QwenImageBlockCfg, sharing the per-block weights) doubles the work
// between syncs and halves the per-block weight upload + sync/round-trip, filling
// the idle GPU. This mirrors the CFG-batching the reference engines do (ollama
// imagegen runs the two CFG branches as a single batch=2 transformer forward).
//
// First-Block-Cache (QwenImageDiT.Cache.cs) is preserved per branch: block 0 runs
// batched every step, then each branch independently decides whether to reuse its
// cached blocks-1..N-1 residual. When both branches compute, the combined kernel is
// used; when only one computes (the other cache-hits), it falls back to the single
// NativeBlock for that branch.
// ============================================================================
using System;
using TensorSharp.Core;

namespace TensorSharp.Models.QwenImage
{
    internal sealed partial class QwenImageDiT
    {
        /// <summary>
        /// Predict the FlowMatch velocity for BOTH true-CFG branches in one fused pass.
        /// Returns (conditional velocity, unconditional velocity), each [imgSeq, 64].
        /// </summary>
        public (float[] cond, float[] neg) PredictCfg(
            float[] imgTokens, int imgSeq,
            float[] condText, int txtSeqC, DitRope ropeC,
            float[] negText, int txtSeqN, DitRope ropeN,
            float timestep01, int[] modulateIndex,
            int stepIndex, int totalSteps, int genTokens)
        {
            // img_in is identical for both branches (same latent tokens); the streams diverge
            // only after block 0 (each attends to its own text conditioning).
            using Tensor imgT = HostToTensor(imgTokens, imgSeq, InCh);
            Tensor img0 = LinearBias(imgT, "img_in.weight", "img_in.bias");
            float[] imgHostC = TensorToHost(img0, (long)imgSeq * Dim); img0.Dispose();
            float[] imgHostN = (float[])imgHostC.Clone();

            // txt_norm + txt_in per branch (different prompts/lengths).
            float[] txtHostC = TxtIn(condText, txtSeqC);
            float[] txtHostN = TxtIn(negText, txtSeqN);

            Tensor temb = TimeEmbed(timestep01);   // [2, 3072] (shared)

            var blkSw = TimingOn ? System.Diagnostics.Stopwatch.StartNew() : null;
            bool cacheActive = CacheEnabled && stepIndex >= 0 && totalSteps > 1;
            if (!cacheActive)
            {
                for (int layer = 0; layer < NumLayers; layer++)
                    RunNativeLayerCfg(layer, imgHostC, imgSeq, txtHostC, txtSeqC, ropeC,
                                      imgHostN, txtHostN, txtSeqN, ropeN, temb, modulateIndex);
            }
            else
            {
                long imgLen = (long)imgSeq * Dim;
                int genLen = (int)Math.Min(imgLen, (long)(genTokens > 0 ? genTokens : imgSeq) * Dim);
                var stateC = _cache[0];
                var stateN = _cache[1];

                var beforeC = new float[genLen]; Array.Copy(imgHostC, 0, beforeC, 0, genLen);
                var beforeN = new float[genLen]; Array.Copy(imgHostN, 0, beforeN, 0, genLen);

                // block 0 always runs (batched) for both branches.
                RunNativeLayerCfg(0, imgHostC, imgSeq, txtHostC, txtSeqC, ropeC,
                                  imgHostN, txtHostN, txtSeqN, ropeN, temb, modulateIndex);

                var residC = new float[genLen]; for (int i = 0; i < genLen; i++) residC[i] = imgHostC[i] - beforeC[i];
                var residN = new float[genLen]; for (int i = 0; i < genLen; i++) residN[i] = imgHostN[i] - beforeN[i];
                bool useC = DecideUseCache(stateC, stepIndex, residC);
                bool useN = DecideUseCache(stateN, stepIndex, residN);

                float[] after0C = null, after0N = null;
                if (!useC) { after0C = new float[imgLen]; Array.Copy(imgHostC, after0C, imgLen); }
                if (!useN) { after0N = new float[imgLen]; Array.Copy(imgHostN, after0N, imgLen); }

                for (int layer = 1; layer < NumLayers; layer++)
                {
                    if (!useC && !useN)
                        RunNativeLayerCfg(layer, imgHostC, imgSeq, txtHostC, txtSeqC, ropeC,
                                          imgHostN, txtHostN, txtSeqN, ropeN, temb, modulateIndex);
                    else if (!useC)
                        RunNativeLayer(layer, imgHostC, imgSeq, txtHostC, txtSeqC, temb, modulateIndex, ropeC);
                    else if (!useN)
                        RunNativeLayer(layer, imgHostN, imgSeq, txtHostN, txtSeqN, temb, modulateIndex, ropeN);
                    // else: both cache-hit -> skip the whole block for both branches.
                }

                if (useC) { var rem = stateC.RemainingResidual; for (long i = 0; i < imgLen; i++) imgHostC[i] += rem[i]; _cacheSkipped++; }
                else { var rem = stateC.RemainingResidual ?? new float[imgLen]; for (long i = 0; i < imgLen; i++) rem[i] = imgHostC[i] - after0C[i]; stateC.RemainingResidual = rem; _cacheComputed++; }
                if (useN) { var rem = stateN.RemainingResidual; for (long i = 0; i < imgLen; i++) imgHostN[i] += rem[i]; _cacheSkipped++; }
                else { var rem = stateN.RemainingResidual ?? new float[imgLen]; for (long i = 0; i < imgLen; i++) rem[i] = imgHostN[i] - after0N[i]; stateN.RemainingResidual = rem; _cacheComputed++; }
            }
            if (blkSw != null)
            {
                blkSw.Stop();
                Console.WriteLine($"  [dit-timing] cfg-batched {NumLayers}-block loop imgSeq={imgSeq} txtSeq={txtSeqC}/{txtSeqN}: {blkSw.Elapsed.TotalMilliseconds:F0}ms");
            }
            temb.Dispose();

            float[] velC = FinishVelocity(imgHostC, imgSeq, timestep01);
            float[] velN = FinishVelocity(imgHostN, imgSeq, timestep01);
            return (velC, velN);
        }

        // txt_norm (RMSNorm) -> txt_in Linear, returning the host [seq, Dim] residual stream.
        private float[] TxtIn(float[] textCond, int txtSeq)
        {
            using Tensor txtT = HostToTensor(textCond, txtSeq, TxtDim);
            using Tensor txtNormed = RMSNormOp(txtT, "txt_norm.weight");
            Tensor txt = LinearBias(txtNormed, "txt_in.weight", "txt_in.bias");
            float[] h = TensorToHost(txt, (long)txtSeq * Dim);
            txt.Dispose();
            return h;
        }

        // norm_out (AdaLN continuous) + proj_out -> velocity [imgSeq, 64].
        private float[] FinishVelocity(float[] imgHost, int imgSeq, float timestep01)
        {
            using Tensor img = HostToTensor(imgHost, imgSeq, Dim);
            Tensor normed = AdaLayerNormOut(img, timestep01);
            Tensor outT = LinearBias(normed, "proj_out.weight", "proj_out.bias"); normed.Dispose();
            float[] velocity = TensorToHost(outT, (long)imgSeq * InCh);
            outT.Dispose();
            return velocity;
        }
    }
}
