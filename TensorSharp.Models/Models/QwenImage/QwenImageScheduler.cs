// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;

namespace TensorSharp.Models.QwenImage
{
    /// <summary>
    /// FlowMatch Euler discrete scheduler (the Qwen-Image scheduler), with the dynamic
    /// resolution-dependent time shift. Mirrors diffusers
    /// <c>FlowMatchEulerDiscreteScheduler</c> + the pipeline's <c>calculate_shift</c>.
    /// </summary>
    internal sealed class QwenImageScheduler
    {
        public float[] Sigmas { get; }      // length steps+1, ending at 0
        public float[] Timesteps { get; }   // length steps, = sigma*1000

        public int Steps => Timesteps.Length;

        /// <param name="muOverride">Fixed time-shift exponent overriding the resolution-dependent
        /// dynamic shift. Lightning step-distilled LoRAs are trained with a CONSTANT shift of 3
        /// (mu = ln 3; lightx2v generate_with_diffusers.py sets base_shift = max_shift = log(3)),
        /// so sampling them with the dynamic shift walks a different sigma schedule than the
        /// distillation targets and visibly degrades the few-step output.</param>
        public QwenImageScheduler(int numSteps, int imageSeqLen, float? muOverride = null)
        {
            float mu = muOverride ?? CalculateShift(imageSeqLen);
            var sigmas = new float[numSteps + 1];
            for (int i = 0; i < numSteps; i++)
            {
                // np.linspace(1.0, 1/N, N)
                double s = numSteps == 1 ? 1.0 : 1.0 + (1.0 / numSteps - 1.0) * i / (numSteps - 1);
                sigmas[i] = TimeShift(mu, 1.0, s);
            }
            sigmas[numSteps] = 0f;
            Sigmas = sigmas;
            Timesteps = new float[numSteps];
            for (int i = 0; i < numSteps; i++) Timesteps[i] = sigmas[i] * 1000f;
        }

        // time_shift(mu, sigma=1, t) = exp(mu) / (exp(mu) + (1/t - 1)^sigma)
        private static float TimeShift(float mu, double sigma, double t)
        {
            double em = Math.Exp(mu);
            return (float)(em / (em + Math.Pow(1.0 / t - 1.0, sigma)));
        }

        // calculate_shift(seq_len, base_seq=256, max_seq=8192, base_shift=0.5, max_shift=0.9).
        // NOTE: these are the QWEN-IMAGE FlowMatchEulerDiscreteScheduler constants (max_image_seq_len=8192,
        // max_shift=0.9), NOT the FLUX defaults (4096 / 1.15). diffusers' shared scheduler *signature*
        // defaults to the FLUX values, but the Qwen-Image scheduler_config overrides them — using the FLUX
        // constants makes mu (and every sigma) too large, mildly over-noising the whole trajectory.
        private static float CalculateShift(int imageSeqLen,
            int baseSeq = 256, int maxSeq = 8192, double baseShift = 0.5, double maxShift = 0.9)
        {
            double m = (maxShift - baseShift) / (maxSeq - baseSeq);
            double b = baseShift - m * baseSeq;
            return (float)(imageSeqLen * m + b);
        }

        /// <summary>Euler step: <c>x_next = x + (sigma_next - sigma) * velocity</c>.</summary>
        public void Step(float[] latents, float[] velocity, int stepIndex)
        {
            float dt = Sigmas[stepIndex + 1] - Sigmas[stepIndex];
            for (long i = 0; i < latents.Length; i++)
                latents[i] += dt * velocity[i];
        }
    }
}
