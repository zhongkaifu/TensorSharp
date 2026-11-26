// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// MiniMax-H3 long-clip numerics.
//
// H3's DiT attends bidirectionally over ONE packed sequence with no mask, so its
// key count is the whole clip rather than a window -- 2364 tokens for a 22-frame
// 640x384 request but 8646 for a 107-frame one. ggml's flash-attention kernels
// keep the softmax numerator in FP16 with three bits of headroom, so that key
// count is a numeric budget, and H3 spent it: a 107-frame clip came back with
// every pixel black and every audio sample clamped to -1, while 73 frames of the
// same request rendered correctly. Nothing reported an error, because a NaN latent
// decodes to a perfectly valid file.
//
// Every other numeric test in the suite runs the DiT at 533 tokens or fewer, a
// dozen times below the shortest length that works, which is why this shipped.
// These run it at the lengths that actually broke.
//
//   set TS_MINIMAX_H3_DIR=C:\Works\TensorSharp\models
//   set TS_TEST_GGML_BACKEND=cuda
//   dotnet test InferenceWeb.Tests --filter FullyQualifiedName~MiniMaxH3LongClipTests
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using TensorSharp.Models.MiniMaxH3;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests
{
    public class MiniMaxH3LongClipTests
    {
        private const string DirEnv = "TS_MINIMAX_H3_DIR";
        private readonly ITestOutputHelper _output;

        public MiniMaxH3LongClipTests(ITestOutputHelper output) { _output = output; }

        private static string Root => Environment.GetEnvironmentVariable(DirEnv);
        private static string DitPath =>
            Path.Combine(Root ?? ".", "minimax_h3_fl2va_pruned-Q4_K.gguf");

        // The shipped config's canvas, which is also the model card's operating point
        // and the geometry every measured failure was taken at.
        private const int Width = 640, Height = 384;

        // Roughly what the server sends: a paragraph of prompt plus the 240-token
        // vision span a conditioning keyframe contributes.
        private const int TextLength = 370;

        /// <summary>Every rung of H3's 17k+5 frame grid from the shortest clip that was
        /// ever tested up to the longest the failure was seen at. 73 rendered correctly
        /// and 107 did not, so the pair either side of that boundary is the point.</summary>
        private static readonly int[] ClipLengths = { 22, 39, 56, 73, 90, 107, 124 };

        /// <summary>One DiT step at every clip length, and none of them may come back with
        /// a NaN. A [Theory] would reload 11 GB of weights per case, so the sweep runs
        /// inside one test and reports every length before failing — which length breaks
        /// is the whole diagnostic.</summary>
        [ModelFact(DirEnv)]
        public void DitVelocitiesStayFiniteAcrossTheFrameGrid()
        {
            if (!File.Exists(DitPath))
            {
                _output.WriteLine("[h3-long] denoiser absent; skipping");
                return;
            }

            using var dit = new MiniMaxH3Denoiser(DitPath, TestGates.PinnedGgmlBackend);
            _output.WriteLine("[h3-long] " + dit.Config);

            // Fixed seeds, and a fresh latent generator per clip length so what a given
            // length sees does not depend on which lengths ran before it. The overflow is
            // marginal and therefore data-dependent: with this draw, 107 frames produced
            // 288 NaN video velocities on the very first step before the fix and 124
            // produced more, while 73 and everything shorter stayed clean.
            var textHidden = Fill(new float[(long)TextLength * dit.Config.TextDim], new Random(1234));
            var failures = new List<string>();

            foreach (int frames in ClipLengths)
            {
                var shape = MiniMaxH3Geometry.Resolve(Width, Height, frames);
                int videoCount = shape.VideoTokenCount, audioCount = shape.AudioTokenCount;
                int keyframeTokens = shape.TokenGridWidth * shape.TokenGridHeight;

                var latentRng = new Random(4321);
                var video = Fill(new float[(long)videoCount * dit.Config.VideoPatchDim], latentRng);
                var audio = Fill(new float[(long)audioCount * dit.Config.AudioLatentChannels], latentRng);
                var keyframe = Fill(new float[(long)keyframeTokens * dit.Config.VideoPatchDim], latentRng);

                const float sigma = 1.0f;   // the first step, where the latent is pure noise
                var layout = MiniMaxH3Layout.Build(
                    TextLength, shape, sigma,
                    conditionFrames: new List<int> { 1 },
                    keyframeAtEnd: new List<bool> { false });

                var sw = Stopwatch.StartNew();
                var (videoVelocity, audioVelocity) = dit.Forward(
                    video, videoCount, audio, audioCount,
                    textHidden, TextLength, layout, sigma,
                    conditionTokens: keyframe, conditionCount: keyframeTokens);
                sw.Stop();

                int videoBad = CountNonFinite(videoVelocity), audioBad = CountNonFinite(audioVelocity);
                double videoRms = Rms(videoVelocity), audioRms = Rms(audioVelocity);
                _output.WriteLine(
                    $"[h3-long] frames={shape.Frames,4} latentT={shape.LatentFrames,3} " +
                    $"tokens={layout.TokenCount,6} | {sw.Elapsed.TotalSeconds,6:F2}s " +
                    $"| video rms={videoRms,7:F4} absmax={AbsMax(videoVelocity),8:F2} bad={videoBad} " +
                    $"| audio rms={audioRms,7:F4} absmax={AbsMax(audioVelocity),8:F2} bad={audioBad}");

                if (videoBad > 0)
                    failures.Add($"{shape.Frames} frames ({layout.TokenCount} tokens): {videoBad} of " +
                                 $"{videoVelocity.LongLength} video velocities are NaN or infinite - " +
                                 "every frame of this clip would decode to black");
                if (audioBad > 0)
                    failures.Add($"{shape.Frames} frames ({layout.TokenCount} tokens): {audioBad} of " +
                                 $"{audioVelocity.LongLength} audio velocities are NaN or infinite");

                // A flow-matching velocity is O(1) by construction; across this whole grid
                // the magnitudes sat at rms ~1.5 (video) and ~4 (audio, which the schedule's
                // dSigmaAudio/dSigmaVideo scales up by 4 at sigma = 1). An order of magnitude
                // past that is a blow-up that simply has not reached infinity yet.
                if (!(videoRms < 20))
                    failures.Add($"{shape.Frames} frames: video velocity rms {videoRms:F2} is far " +
                                 "outside the flow-matching range");
                if (!(audioRms < 50))
                    failures.Add($"{shape.Frames} frames: audio velocity rms {audioRms:F2} is far " +
                                 "outside the flow-matching range");
            }

            Assert.True(failures.Count == 0, string.Join("\n", failures));
        }

        /// <summary>The whole sampler at the two lengths either side of the boundary,
        /// printing every step. Opt-in with <c>TS_H3_LONG_LOOP=1</c>: it is ~15 minutes
        /// of real denoising, which is too slow to sit in the model-gated run, but it is
        /// what shows a divergence spreading step by step rather than only its wreckage.
        /// <c>TS_H3_LONG_LOOP_FRAMES</c> overrides the lengths.</summary>
        [ModelFact(DirEnv)]
        public void DenoiseLoopStaysFiniteOnALongClip()
        {
            if (Environment.GetEnvironmentVariable("TS_H3_LONG_LOOP") != "1")
            {
                _output.WriteLine("[h3-long] set TS_H3_LONG_LOOP=1 to run the full sampler; skipping");
                return;
            }
            if (!File.Exists(DitPath))
            {
                _output.WriteLine("[h3-long] denoiser absent; skipping");
                return;
            }

            string spec = Environment.GetEnvironmentVariable("TS_H3_LONG_LOOP_FRAMES") ?? "107,73";
            const int steps = 24;

            using var dit = new MiniMaxH3Denoiser(DitPath, TestGates.PinnedGgmlBackend);
            _output.WriteLine($"[h3-long] {dit.Config}  steps={steps}");
            var textHidden = Fill(new float[(long)TextLength * dit.Config.TextDim], new Random(1234));

            foreach (string part in spec.Split(',', StringSplitOptions.RemoveEmptyEntries))
            {
                var shape = MiniMaxH3Geometry.Resolve(Width, Height, int.Parse(part.Trim()));
                int videoCount = shape.VideoTokenCount, audioCount = shape.AudioTokenCount;
                int keyframeTokens = shape.TokenGridWidth * shape.TokenGridHeight;

                var latentRng = new Random(4321);
                var video = Fill(new float[(long)videoCount * dit.Config.VideoPatchDim], latentRng);
                var audio = Fill(new float[(long)audioCount * dit.Config.AudioLatentChannels], latentRng);
                var keyframe = Fill(new float[(long)keyframeTokens * dit.Config.VideoPatchDim], latentRng);

                float[] sigmas = MiniMaxH3Scheduler.BuildSigmas(steps);
                _output.WriteLine($"[h3-long] ==== frames={shape.Frames} latentT={shape.LatentFrames} " +
                                  $"videoTokens={videoCount} audioTokens={audioCount} ====");

                for (int step = 0; step < steps; step++)
                {
                    float sigma = sigmas[step];
                    var layout = MiniMaxH3Layout.Build(
                        TextLength, shape, sigma,
                        conditionFrames: new List<int> { 1 },
                        keyframeAtEnd: new List<bool> { false });

                    var (v, a) = dit.Forward(video, videoCount, audio, audioCount,
                                             textHidden, TextLength, layout, sigma,
                                             conditionTokens: keyframe, conditionCount: keyframeTokens);
                    _output.WriteLine(
                        $"[h3-long] step {step,2} sigma={sigma,7:F4} " +
                        $"| x rms={Rms(video),8:F3} absmax={AbsMax(video),9:F2} " +
                        $"| v rms={Rms(v),8:F3} bad={CountNonFinite(v)} " +
                        $"| aX rms={Rms(audio),8:F3} | aV rms={Rms(a),8:F3} bad={CountNonFinite(a)}");

                    Assert.True(CountNonFinite(v) == 0 && CountNonFinite(a) == 0,
                        $"the sample diverged at step {step + 1}/{steps} on a {shape.Frames}-frame clip " +
                        $"({layout.TokenCount} packed tokens)");

                    MiniMaxH3Scheduler.EulerStep(video, v, sigma, sigmas[step + 1]);
                    MiniMaxH3Scheduler.EulerStep(audio, a, sigma, sigmas[step + 1]);
                }

                _output.WriteLine($"[h3-long] FINAL frames={shape.Frames} video rms={Rms(video):F4} " +
                                  $"absmax={AbsMax(video):F2} | audio rms={Rms(audio):F4}");
                Assert.Equal(0, CountNonFinite(video));
                Assert.Equal(0, CountNonFinite(audio));
            }
        }

        // ---- helpers --------------------------------------------------------

        private static float[] Fill(float[] buffer, Random rng)
        {
            for (long i = 0; i < buffer.LongLength; i++) buffer[i] = Gauss(rng);
            return buffer;
        }

        private static float Gauss(Random rng)
        {
            double u1 = 1.0 - rng.NextDouble(), u2 = rng.NextDouble();
            return (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
        }

        private static int CountNonFinite(float[] v)
        {
            int bad = 0;
            foreach (float x in v) if (!float.IsFinite(x)) bad++;
            return bad;
        }

        // Non-finite values are excluded so a partially poisoned buffer still reports a
        // usable magnitude for the rest; CountNonFinite is what decides pass or fail.
        private static double Rms(float[] v)
        {
            double sum = 0; long n = 0;
            foreach (float x in v) if (float.IsFinite(x)) { sum += (double)x * x; n++; }
            return n == 0 ? double.NaN : Math.Sqrt(sum / n);
        }

        private static float AbsMax(float[] v)
        {
            float m = 0;
            foreach (float x in v) if (float.IsFinite(x) && Math.Abs(x) > m) m = Math.Abs(x);
            return m;
        }
    }
}
