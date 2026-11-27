// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Samplers that read more than the state and the velocity: the step's noise (ancestral Euler) and a per-token denoise mask
// with the clean latent it pins (a guide frame conditioned at strength s denoises with weight 1 - s: LTX's post_process_latent).
// Both go through the x0 estimate, x - v * (sigma * mask), and keep every value float32 in the reference order.
using System;
using System.Collections.Generic;

namespace TensorSharp.Models.SharedDiffusion
{
    /// <summary>What a conditioned sampler reads for one stream at one step beside the state and the velocity (each null when
    /// the stream has none).</summary>
    public sealed class SamplerStreamInputs
    {
        /// <summary>The step's standard-normal noise, one value per state value (an ancestral sampler's; none at the final step).</summary>
        public float[] Noise { get; init; }

        /// <summary>One denoise weight per token in [0, 1]: 1 denoises the token fully, 0 keeps <see cref="Clean"/>.</summary>
        public float[] DenoiseMask { get; init; }

        /// <summary>The clean values the mask pins toward, one per state value (with <see cref="DenoiseMask"/>).</summary>
        public float[] Clean { get; init; }
    }

    /// <summary>A sampler that reads the step's noise and a stream's denoise mask (<see cref="SamplerStreamInputs"/>).</summary>
    public interface IConditionedSampler : IDiffusionSampler
    {
        /// <summary>Whether a step that does not end at 0 needs noise.</summary>
        bool NeedsNoise { get; }

        /// <summary>Advance <paramref name="x"/> in place from <paramref name="sigma"/> to <paramref name="next"/> with the stream's
        /// noise and mask (null: none).</summary>
        void Step(Span<float> x, ReadOnlySpan<float> velocity, float sigma, float next, SamplerStreamInputs inputs);
    }

    /// <summary>
    /// The shared arithmetic of the conditioned samplers: the x0 estimate of a token's value and the mask's blend toward its
    /// clean value, as LTX's X0Model and post_process_latent compute them in float32.
    /// </summary>
    internal static class ConditionedSampling
    {
        public static void Check(int length, ReadOnlySpan<float> velocity, SamplerStreamInputs inputs, out float[] mask, out float[] clean,
            out int width)
        {
            if (velocity.Length != length)
                throw new ArgumentException($"The velocity has {velocity.Length} values for {length} latent values.", nameof(velocity));
            mask = inputs?.DenoiseMask;
            clean = inputs?.Clean;
            width = 0;
            if ((mask == null) != (clean == null))
                throw new ArgumentException("A denoise mask needs the clean values it pins, and clean values need their mask.");
            if (mask == null) return;
            if (clean.Length != length || mask.Length == 0 || length % mask.Length != 0)
                throw new ArgumentException("A denoise mask has one weight per token and the clean values one value per latent value.");
            width = length / mask.Length;
        }

        public static void RequireFinite(float value, string what)
        {
            if (!float.IsFinite(value)) throw new InvalidOperationException($"The diffusion sampler produced a nonfinite {what}.");
        }
    }

    /// <summary>
    /// Euler through the x0 estimate (LTX's X0Model then EulerDiffusionStep): denoised = x - v * (sigma * m), blended toward the
    /// clean value by the mask, then x + (x - denoised) / sigma * (next - sigma). Not <see cref="EulerFlowSampler"/>'s
    /// x + v * (next - sigma): the round trip rounds differently, and the reference keeps it.
    /// </summary>
    public sealed class EulerDenoisedSampler : IConditionedSampler
    {
        public bool NeedsNoise => false;

        public void Step(Span<float> x, ReadOnlySpan<float> velocity, float sigma, float next) => Step(x, velocity, sigma, next, null);

        public void Step(Span<float> x, ReadOnlySpan<float> velocity, float sigma, float next, SamplerStreamInputs inputs)
        {
            ConditionedSampling.Check(x.Length, velocity, inputs, out float[] mask, out float[] clean, out int width);
            for (int i = 0; i < x.Length; ++i)
            {
                ConditionedSampling.RequireFinite(velocity[i], "velocity");
                float denoise = mask == null ? 1 : mask[i / width];
                float denoised = x[i] - velocity[i] * (sigma * denoise);
                if (mask != null) denoised = denoised * denoise + clean[i] * (1 - denoise);
                float derivative = (x[i] - denoised) / sigma;
                float value = x[i] + derivative * (next - sigma);
                ConditionedSampling.RequireFinite(value, "latent");
                x[i] = value;
            }
        }

        public void Reset() { }
    }

    /// <summary>
    /// Ancestral Euler on the flow (the euler_ancestral denoising loop of LTX's DistilledPipeline, eta 1, noise scale 1): the x0
    /// estimate and the mask's first blend as <see cref="EulerDenoisedSampler"/>, then a step down to sigma_down and a renoise
    /// back up to the next sigma with the step's noise, and the mask's blend again; the final step (next 0) returns the x0
    /// estimate and reads no noise.
    /// </summary>
    public sealed class EulerAncestralSampler : IConditionedSampler
    {
        /// <summary>The reference's fully ancestral eta and noise scale.</summary>
        public const float Eta = 1f, NoiseScale = 1f;

        public bool NeedsNoise => true;

        public void Step(Span<float> x, ReadOnlySpan<float> velocity, float sigma, float next) => Step(x, velocity, sigma, next, null);

        public void Step(Span<float> x, ReadOnlySpan<float> velocity, float sigma, float next, SamplerStreamInputs inputs)
        {
            ConditionedSampling.Check(x.Length, velocity, inputs, out float[] mask, out float[] clean, out int width);
            float[] noise = inputs?.Noise;
            if (next != 0 && (noise == null || noise.Length != x.Length))
                throw new ArgumentException("An ancestral step that does not end at 0 needs one noise value per latent value.", nameof(inputs));
            float downstepRatio = 1f + (next / sigma - 1f) * Eta;
            float sigmaDown = next * downstepRatio;
            float sigmaDownRatio = sigmaDown / sigma;
            float alphaNext = 1f - next, alphaDown = 1f - sigmaDown;
            float alphaRatio = alphaNext / alphaDown;
            float renoise = MathF.Sqrt(MathF.Max(0f, next * next - sigmaDown * sigmaDown * (alphaNext * alphaNext) / (alphaDown * alphaDown)));
            for (int i = 0; i < x.Length; ++i)
            {
                ConditionedSampling.RequireFinite(velocity[i], "velocity");
                float denoise = mask == null ? 1 : mask[i / width];
                float denoised = x[i] - velocity[i] * (sigma * denoise);
                if (mask != null) denoised = denoised * denoise + clean[i] * (1 - denoise);
                float value;
                if (next == 0) value = denoised;
                else
                {
                    value = sigmaDownRatio * x[i] + (1f - sigmaDownRatio) * denoised;
                    value = alphaRatio * value + noise[i] * NoiseScale * renoise;
                    if (mask != null) value = value * denoise + clean[i] * (1 - denoise);
                }
                ConditionedSampling.RequireFinite(value, "latent");
                x[i] = value;
            }
        }

        public void Reset() { }
    }

    /// <summary>One stage of a clip's sampling (a family that samples in passes: LTX 2.5's half-size first pass, then the
    /// refinement at full size): its name for progress and its sigmas, ending in 0.</summary>
    public sealed record DiffusionVideoStage(string Name, float[] Sigmas)
    {
        public int Steps => Sigmas.Length - 1;
    }

    /// <summary>The sampler inputs of one step, by stream name (<see cref="IVideoDiffusionFamily.SamplerInputs"/>).</summary>
    public sealed class DiffusionSamplerInputs
    {
        public DiffusionSamplerInputs(IReadOnlyDictionary<string, SamplerStreamInputs> streams)
            => Streams = streams ?? throw new ArgumentNullException(nameof(streams));

        public IReadOnlyDictionary<string, SamplerStreamInputs> Streams { get; }

        public SamplerStreamInputs For(string stream) => Streams.TryGetValue(stream, out SamplerStreamInputs s) ? s : null;
    }
}
