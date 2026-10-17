// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// The Turbo plug-ins' endpoint-inclusive linspace grid (MiniMaxH3Scheduler.BuildUniformSigmas),
// pinned against closed-form values: ModelTC/Minimax-H3-Turbo samples q_i = (N - i) / N through
// the shift, and diffusers scheduling_minimax_h3.py set_timesteps gives the same grid. Also
// pins that the base sd.cpp grid is unchanged, and that H3's single-clock audio carrier on the
// linspace grid takes exactly the per-modality Euler trajectory the reference runs.
using System;
using TensorSharp.Models.MiniMaxH3;
using Xunit;

namespace InferenceWeb.Tests
{
    public class MiniMaxH3SchedulerTurboTests
    {
        private static void AssertGrid(double[] expected, float[] actual, double tolerance = 1e-6)
        {
            Assert.Equal(expected.Length, actual.Length);
            for (int i = 0; i < expected.Length; i++)
                Assert.True(Math.Abs(expected[i] - actual[i]) <= tolerance,
                    $"sigma[{i}] = {actual[i]:R}, expected {expected[i]:R}");
        }

        // ---- the Turbo grids -------------------------------------------------------------

        [Fact]
        public void UniformGrid_Video8StepsAtShift6()
        {
            // TimeSnrShift(6, 1 - i/8): 1, 42/43, 18/19, 10/11, 6/7, 18/23, 2/3, 6/13, then 0.
            AssertGrid(
                new[] { 1, 42 / 43.0, 18 / 19.0, 10 / 11.0, 6 / 7.0, 18 / 23.0, 2 / 3.0, 6 / 13.0, 0 },
                MiniMaxH3Scheduler.BuildUniformSigmas(8, 6f));
        }

        [Fact]
        public void UniformGrid_Video4StepsAtShift6()
        {
            AssertGrid(new[] { 1, 18 / 19.0, 6 / 7.0, 2 / 3.0, 0 }, MiniMaxH3Scheduler.BuildUniformSigmas(4, 6f));
        }

        [Theory]
        // The audio stream's grid is the video grid unshifted by 6 and reshifted by 3, which is
        // the same linspace grid at shift 3.
        [InlineData(4, new[] { 1, 0.9, 0.75, 0.5, 0 })]
        [InlineData(8, new[] { 1, 21 / 22.0, 0.9, 5 / 6.0, 0.75, 9 / 14.0, 0.5, 0.3, 0 })]
        public void UniformGrid_AudioThroughMapSigmaEqualsTheShift3Grid(int steps, double[] expected)
        {
            float[] video = MiniMaxH3Scheduler.BuildUniformSigmas(steps, 6f);
            var audio = new float[video.Length];
            for (int i = 0; i < video.Length; i++)
                audio[i] = MiniMaxH3Scheduler.MapSigma(video[i], 6f, 3f);

            AssertGrid(expected, audio);
            AssertGrid(expected, MiniMaxH3Scheduler.BuildUniformSigmas(steps, 3f));
        }

        [Fact]
        public void UniformGrid_StartsAtOneEndsAtZeroAndDecreases()
        {
            foreach (int steps in new[] { 1, 2, 4, 8, 20, 100 })
                foreach (float shift in new[] { 1f, 3f, 6f, 12f })
                {
                    float[] sigmas = MiniMaxH3Scheduler.BuildUniformSigmas(steps, shift);
                    Assert.Equal(steps + 1, sigmas.Length);
                    Assert.Equal(1f, sigmas[0]);
                    Assert.Equal(0f, sigmas[steps]);
                    for (int i = 1; i < sigmas.Length; i++)
                        Assert.True(sigmas[i] < sigmas[i - 1], $"step {i} did not decrease at {steps} steps, shift {shift}");
                }
        }

        [Fact]
        public void UniformGrid_AtShift1IsThePlainLinspace()
        {
            AssertGrid(new[] { 1, 0.75, 0.5, 0.25, 0 }, MiniMaxH3Scheduler.BuildUniformSigmas(4, 1f));
        }

        [Theory]
        [InlineData(0, 6f)]
        [InlineData(-1, 6f)]
        [InlineData(8, 0f)]
        [InlineData(8, -6f)]
        [InlineData(8, float.NaN)]
        [InlineData(8, float.PositiveInfinity)]
        public void UniformGrid_RefusesBadArguments(int steps, float shift)
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => MiniMaxH3Scheduler.BuildUniformSigmas(steps, shift));
        }

        // ---- the base grid is untouched ----------------------------------------------------

        [Fact]
        public void BaseGrid_20StepsAtShift12IsUnchanged()
        {
            AssertGrid(new[]
            {
                1, 0.9953966, 0.9903020, 0.9846334, 0.9782878, 0.9711363, 0.9630151, 0.9537123, 0.9429501,
                0.9303558, 0.9154179, 0.8974151, 0.8752965, 0.8474684, 0.8113925, 0.7627642, 0.6936523,
                0.5876611, 0.4045301, 0.0118694, 0,
            }, MiniMaxH3Scheduler.BuildSigmas(20, 12f));
        }

        [Fact]
        public void BaseGrid_8StepsAtShift12IsUnchanged()
        {
            // sd.cpp's discrete axis: the last nonzero sigma sits just above 0, unlike the
            // linspace grid's 1/N.
            AssertGrid(new[] { 1, 0.9863172, 0.9677856, 0.9412733, 0.9002098, 0.8280846, 0.6682165, 0.0118694, 0 },
                MiniMaxH3Scheduler.BuildSigmas(8, 12f));
        }

        // ---- single-clock carrier == per-modality Euler on the linspace grid ---------------

        /// <summary>A state-dependent denoiser stand-in: the flow-matching velocity for the
        /// clean estimate 0.5 tanh(x), v = (x - x0) / sigma. Any smooth function would do;
        /// what matters is that each step's velocity depends on the state it is given.</summary>
        private static double Oracle(double x, double sigma) => (x - 0.5 * Math.Tanh(x)) / sigma;

        [Theory]
        [InlineData(4)]
        [InlineData(8)]
        public void AudioCarrierOnTheLinspaceGrid_MatchesPerModalityEuler(int steps)
        {
            const float videoShift = 6f, audioShift = 3f;
            double[] start = { 0.37, -0.81, 1.9, -1.4, 0.05, 2.6 };

            // Reference: the audio stream stepped on its own shift-3 linspace grid, in double.
            var reference = (double[])start.Clone();
            for (int i = 0; i < steps; i++)
            {
                double sigma = MiniMaxH3Scheduler.TimeSnrShift(audioShift, 1.0 - (double)i / steps);
                double next = i + 1 < steps ? MiniMaxH3Scheduler.TimeSnrShift(audioShift, 1.0 - (double)(i + 1) / steps) : 0.0;
                for (int k = 0; k < reference.Length; k++)
                    reference[k] += Oracle(reference[k], sigma) * (next - sigma);
            }

            // TensorSharp: one sampler on the video grid carrying c = (sigma_v / sigma_a) x_a,
            // the model fed the restored audio and its chain-rule-scaled velocity converted back.
            float[] sigmas = MiniMaxH3Scheduler.BuildUniformSigmas(steps, videoShift);
            var carrier = Array.ConvertAll(start, v => (float)v);
            for (int i = 0; i < steps; i++)
            {
                float sigma = sigmas[i];
                float[] input = MiniMaxH3Scheduler.AudioInputFromCarrier(carrier, sigma, videoShift, audioShift);
                float audioSigma = MiniMaxH3Scheduler.MapSigma(sigma, videoShift, audioShift);
                float slope = MiniMaxH3Scheduler.ShiftSlope(sigma, videoShift, audioShift);
                var velocity = new float[input.Length];
                for (int k = 0; k < input.Length; k++)
                    velocity[k] = (float)Oracle(input[k], audioSigma) * slope;
                MiniMaxH3Scheduler.ConvertAudioVelocityToCarrier(input, velocity, sigma, videoShift, audioShift);
                MiniMaxH3Scheduler.EulerStep(carrier, velocity, sigma, sigmas[i + 1]);
            }
            MiniMaxH3Scheduler.AudioOutputFromCarrierInPlace(carrier, videoShift, audioShift);

            for (int k = 0; k < reference.Length; k++)
                Assert.True(Math.Abs(carrier[k] - reference[k]) <= 1e-5 * Math.Max(Math.Abs(reference[k]), 1e-2),
                    $"element {k}: carrier {carrier[k]:R}, per-modality {reference[k]:R} at {steps} steps");
        }
    }
}
