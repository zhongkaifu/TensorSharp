// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// MiniMax-H3 flow-matching schedule.
//
// H3 denoises video and audio jointly, but the two streams want DIFFERENT
// timestep shifts: 12 for video, 3 for audio. Running two samplers is not an
// option -- both streams live in one packed tensor and are stepped together.
//
// The trick H3 uses is to keep the sampler entirely on the video schedule and
// make the model do the conversion:
//
//   * the audio stream is conditioned on the timestep a shift-3 schedule would
//     have had at the same underlying uniform time -- MapSigma() below unshifts
//     by 12 and reshifts by 3; and
//   * the sampler carries c_a = (sigma_video / sigma_audio) * x_a. Before calling
//     the model it restores x_a, then converts the model's audio velocity to the
//     derivative of this carrier. The conversion includes a drift term; a
//     chain-rule scale alone leaves noise in the decoded audio.
//
// At sigma = 1 the carrier equals the physical audio. At sigma = 0 its limit is
// (videoShift / audioShift) * cleanAudio, so the final carrier must be scaled by
// audioShift / videoShift before the audio VAE, matching process_latent_out.
using System;

namespace TensorSharp.Models.MiniMaxH3
{
    /// <summary>Discrete flow-matching schedule for MiniMax-H3, with the dual video/audio
    /// timestep shift folded into a single sampler pass.</summary>
    public static class MiniMaxH3Scheduler
    {
        /// <summary>Default timestep shift for the video stream.</summary>
        public const float VideoShift = 12.0f;

        /// <summary>Timestep shift for the audio stream. Fixed by the checkpoint; it is
        /// NOT affected by a user-supplied flow shift, which only moves the video stream.</summary>
        public const float AudioShift = 3.0f;

        /// <summary>Timestep at which conditioning (keyframe / reference) latents are presented:
        /// essentially clean, but not exactly, matching the tiny noise blend applied to them.</summary>
        public const float VisualConditionTimestep = 0.999f;

        /// <summary>Default denoising steps, matching stable-diffusion.cpp's CLI.
        /// Guidance distillation does not guarantee quality at fewer steps. Fewer steps
        /// take a step-distillation adapter, the LightX2V Turbo plug-in
        /// (config/lora/minimax-h3-fl2v-turbo-*.json), which brings its own step count and
        /// grid (<see cref="BuildUniformSigmas"/>) and refuses any other.</summary>
        public const int DefaultSteps = 20;

        /// <summary>The flow-matching SNR shift: sigma' = a*t / (1 + (a-1)*t).</summary>
        public static double TimeSnrShift(double shift, double t) => shift * t / (1.0 + (shift - 1.0) * t);

        /// <summary>Inverse of <see cref="TimeSnrShift"/>: recovers the uniform time that
        /// produced <paramref name="sigma"/> under <paramref name="shift"/>.</summary>
        public static double InverseTimeSnrShift(double shift, double sigma) =>
            sigma / (shift + sigma * (1.0 - shift));

        /// <summary>Build the discrete sigma schedule for the video stream. Returns
        /// <paramref name="steps"/>+1 values running from 1.0 down to exactly 0.0.</summary>
        public static float[] BuildSigmas(int steps, float shift = VideoShift)
        {
            if (steps <= 0) throw new ArgumentOutOfRangeException(nameof(steps));
            if (!float.IsFinite(shift) || shift <= 0) throw new ArgumentOutOfRangeException(nameof(shift));
            var sigmas = new float[steps + 1];
            for (int i = 0; i < steps; i++)
            {
                // The reference walks the discrete timestep axis 999 -> 0 uniformly,
                // then maps each through the shift. t+1 (not t) is what gets divided
                // by 1000, so the last step lands just above zero rather than on it.
                double t = steps == 1 ? 999.0 : 999.0 - i * (999.0 / (steps - 1));
                sigmas[i] = (float)TimeSnrShift(shift, (t + 1.0) / 1000.0);
            }
            sigmas[steps] = 0f;
            return sigmas;
        }

        /// <summary>The endpoint-inclusive linspace grid the Turbo adapters were distilled on
        /// (ModelTC/Minimax-H3-Turbo, q_i = (N - i) / N; diffusers scheduling_minimax_h3.py
        /// set_timesteps): u_i = 1 - i/N for i = 0..N-1, sigma_i = TimeSnrShift(shift, u_i),
        /// then a terminal 0. Unlike <see cref="BuildSigmas"/>, which walks sd.cpp's discrete
        /// 999..0 axis, the steps are even in u and the last nonzero sigma sits at u = 1/N
        /// rather than just above 0, so the two grids differ after the first step. The audio stream's grid is this one mapped
        /// through <see cref="MapSigma"/>, which equals the same grid at the audio shift.
        /// Returns <paramref name="steps"/>+1 values from 1.0 down to exactly 0.0.</summary>
        public static float[] BuildUniformSigmas(int steps, float shift)
        {
            if (steps <= 0) throw new ArgumentOutOfRangeException(nameof(steps));
            if (!float.IsFinite(shift) || shift <= 0) throw new ArgumentOutOfRangeException(nameof(shift));
            var sigmas = new float[steps + 1];
            for (int i = 0; i < steps; i++)
                sigmas[i] = (float)TimeSnrShift(shift, 1.0 - (double)i / steps);
            sigmas[steps] = 0f;
            return sigmas;
        }

        /// <summary>Re-express a sigma from one shift's schedule on another's:
        /// unshift by <paramref name="fromShift"/>, reshift by <paramref name="toShift"/>.
        /// This is how the audio stream's timestep is derived from the video sigma.</summary>
        public static float MapSigma(float sigma, float fromShift = VideoShift, float toShift = AudioShift)
        {
            double b = InverseTimeSnrShift(fromShift, sigma);
            return (float)TimeSnrShift(toShift, b);
        }

        /// <summary>d(sigmaTo)/d(sigmaFrom) -- the chain-rule factor the audio velocity is
        /// scaled by so a single Euler step on the video schedule advances the audio stream
        /// along its own schedule.</summary>
        public static float ShiftSlope(float sigma, float fromShift = VideoShift, float toShift = AudioShift)
        {
            double b = InverseTimeSnrShift(fromShift, sigma);
            double a = 1.0 + (fromShift - 1.0) * b;
            double c = 1.0 + (toShift - 1.0) * b;
            return (float)(toShift * a * a / (fromShift * c * c));
        }

        /// <summary>Conditioning timestep fed to the video stream's AdaLN for a given sigma.</summary>
        public static float VideoTimestep(float sigma) => 1f - sigma;

        /// <summary>Conditioning timestep fed to the audio stream's AdaLN for a given
        /// VIDEO sigma (the sampler only ever knows the video schedule).</summary>
        public static float AudioTimestep(float sigma, float videoShift = VideoShift, float audioShift = AudioShift)
            => 1f - MapSigma(sigma, videoShift, audioShift);

        /// <summary>Clamp a video sigma into the domain the timestep mapping is defined on.</summary>
        public static float ClampSigma(float sigma) => Math.Clamp(sigma, 1e-6f, 1f);

        /// <summary>Restore the network's audio input from the sampler carrier,
        /// matching the H3 reference runner's sigma_audio / sigma_video scale.</summary>
        public static float[] AudioInputFromCarrier(float[] carrier, float sigma,
            float videoShift = VideoShift, float audioShift = AudioShift)
        {
            if (carrier is null) throw new ArgumentNullException(nameof(carrier));
            float videoSigma = ClampSigma(sigma);
            float scale = MapSigma(videoSigma, videoShift, audioShift) / videoSigma;
            var audio = new float[carrier.Length];
            for (int i = 0; i < audio.Length; ++i) audio[i] = carrier[i] * scale;
            return audio;
        }

        /// <summary>Convert DiT's chain-rule-scaled audio velocity in place to
        /// d(carrier)/d(sigma_video). DiT's standalone component contract uses
        /// ShiftSlope; the generation sampler additionally needs the carrier drift.</summary>
        public static void ConvertAudioVelocityToCarrier(float[] audioInput, float[] velocity,
            float sigma, float videoShift = VideoShift, float audioShift = AudioShift)
        {
            if (audioInput is null) throw new ArgumentNullException(nameof(audioInput));
            if (velocity is null) throw new ArgumentNullException(nameof(velocity));
            if (audioInput.Length != velocity.Length)
                throw new ArgumentException("velocity length must match the audio input length", nameof(velocity));
            float videoSigma = ClampSigma(sigma);
            float audioSigma = MapSigma(videoSigma, videoShift, audioShift);
            float ratio = videoShift / audioShift;
            float predictionScale = (1f + (ratio - 1f) * audioSigma)
                / ShiftSlope(videoSigma, videoShift, audioShift);
            for (int i = 0; i < velocity.Length; ++i)
                velocity[i] = (1f - ratio) * audioInput[i] + predictionScale * velocity[i];
        }

        /// <summary>Restore the clean physical audio at the sampler endpoint,
        /// matching H3AVFlowDenoiser.process_latent_out. Unlike the network-input
        /// conversion, this uses the sigma-zero limit and performs no clamping.</summary>
        public static void AudioOutputFromCarrierInPlace(float[] carrier,
            float videoShift = VideoShift, float audioShift = AudioShift)
        {
            if (carrier is null) throw new ArgumentNullException(nameof(carrier));
            if (!float.IsFinite(videoShift) || videoShift <= 0) throw new ArgumentOutOfRangeException(nameof(videoShift));
            if (!float.IsFinite(audioShift) || audioShift <= 0) throw new ArgumentOutOfRangeException(nameof(audioShift));
            float scale = audioShift / videoShift;
            for (int i = 0; i < carrier.Length; ++i) carrier[i] *= scale;
        }

        /// <summary>One Euler update: <c>x += velocity * (sigmaNext - sigma)</c>.
        /// Audio velocities must first be converted to the sampler carrier derivative.</summary>
        public static void EulerStep(float[] x, float[] velocity, float sigma, float sigmaNext)
        {
            if (x is null) throw new ArgumentNullException(nameof(x));
            if (velocity is null) throw new ArgumentNullException(nameof(velocity));
            if (x.Length != velocity.Length)
                throw new ArgumentException("velocity length must match the latent length", nameof(velocity));
            float dt = sigmaNext - sigma;
            for (int i = 0; i < x.Length; i++) x[i] += velocity[i] * dt;
        }
    }
}
