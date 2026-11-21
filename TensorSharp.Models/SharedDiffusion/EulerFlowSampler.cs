// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;

namespace TensorSharp.Models.SharedDiffusion
{
    /// <summary>
    /// The update a diffusion run applies after each model call. The prediction is the flow velocity
    /// dx/dsigma (what MiniMax-H3, Qwen-Image, Wan and LTX predict once their own sign conventions are applied), so the
    /// denoised estimate at sigma is <c>x - sigma * v</c>. A sampler with state (UniPC's history) keeps it between
    /// steps and drops it in <see cref="Reset"/>.
    /// </summary>
    public interface IDiffusionSampler
    {
        /// <summary>Advance <paramref name="x"/> in place from <paramref name="sigma"/> to <paramref name="next"/>.</summary>
        void Step(Span<float> x, ReadOnlySpan<float> velocity, float sigma, float next);

        /// <summary>Forget any state from earlier steps (a new run).</summary>
        void Reset();
    }

    /// <summary>First-order Euler on the flow ODE: <c>x += v * (next - sigma)</c>, in float32, element by element.</summary>
    public sealed class EulerFlowSampler : IDiffusionSampler
    {
        public void Step(Span<float> x, ReadOnlySpan<float> velocity, float sigma, float next)
        {
            if (x.Length != velocity.Length)
                throw new ArgumentException($"The velocity has {velocity.Length} values for {x.Length} latent values.", nameof(velocity));
            float dt = next - sigma;
            for (int k = 0; k < x.Length; k++) x[k] += velocity[k] * dt;
        }

        public void Reset() { }

        /// <summary>The denoised (x0) estimate of a flow state: <c>x - sigma * v</c>. After an update to sigma_next
        /// with the step's velocity it is the step's clean prediction, which previews decode.</summary>
        public static float[] Denoised(ReadOnlySpan<float> x, ReadOnlySpan<float> velocity, float sigma)
        {
            if (x.Length != velocity.Length)
                throw new ArgumentException("Latent and velocity lengths must match.");
            if (!float.IsFinite(sigma) || sigma < 0f || sigma > 1f)
                throw new ArgumentOutOfRangeException(nameof(sigma));
            var clean = new float[x.Length];
            for (int i = 0; i < clean.Length; i++)
                clean[i] = x[i] - sigma * velocity[i];
            return clean;
        }
    }
}
