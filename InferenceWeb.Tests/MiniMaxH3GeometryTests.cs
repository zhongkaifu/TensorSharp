// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// MiniMax-H3 shape, schedule and packed-layout unit tests. Everything here is pure
// arithmetic with no weights, so it runs in ordinary CI. The expected values are the
// reference implementation's, not this implementation's: the frame grid and latent
// mapping come from stable-diffusion.cpp's align_video_frames /
// video_frames_to_latent_frames, the sigma schedule from its DiscreteFlowDenoiser at
// shift 12, and the dual-shift mapping/slope from time_shift_sigma / time_shift_slope.
using System;
using System.Collections.Generic;
using TensorSharp.Models.MiniMaxH3;
using TensorSharp.Models.Video;
using Xunit;

namespace InferenceWeb.Tests
{
    public class MiniMaxH3GeometryTests
    {
        // ---- frame + latent grid ---------------------------------------------

        [Theory]
        // The grid is 17k+5 with a floor of 5, so anything at or below 5 snaps up to 5
        // and every value in between climbs to the next rung.
        [InlineData(1, 5)]
        [InlineData(5, 5)]
        [InlineData(6, 22)]
        [InlineData(22, 22)]
        [InlineData(23, 39)]
        [InlineData(39, 39)]
        [InlineData(56, 56)]
        [InlineData(57, 73)]
        [InlineData(121, 124)]
        public void AlignFramesSnapsToThe17kPlus5Grid(int requested, int expected)
            => Assert.Equal(expected, MiniMaxH3Geometry.AlignFrames(requested));

        [Theory]
        [InlineData(5, 2)]
        [InlineData(22, 7)]
        [InlineData(39, 12)]
        [InlineData(56, 17)]
        [InlineData(90, 27)]
        public void LatentFrameCountMatchesReference(int frames, int latent)
            => Assert.Equal(latent, MiniMaxH3Geometry.VideoFramesToLatentFrames(frames));

        [Theory]
        [InlineData(2, 5)]
        [InlineData(7, 22)]
        [InlineData(12, 39)]
        [InlineData(17, 56)]
        [InlineData(27, 90)]
        public void LatentFrameCountRoundTrips(int latent, int frames)
        {
            Assert.Equal(frames, MiniMaxH3Geometry.LatentFramesToVideoFrames(latent));
            Assert.Equal(latent, MiniMaxH3Geometry.VideoFramesToLatentFrames(frames));
        }

        [Theory]
        [InlineData(0, 32)]
        [InlineData(1, 32)]
        [InlineData(32, 32)]
        [InlineData(33, 64)]
        [InlineData(480, 480)]
        [InlineData(864, 864)]
        [InlineData(500, 512)]
        [InlineData(544, 544)]
        public void SpatialDimensionsAlignTo32(int requested, int expected)
            => Assert.Equal(expected, MiniMaxH3Geometry.AlignSpatial(requested));

        // ---- the reference worked example ------------------------------------
        // 864x480, 56 frames at 24 fps is the command in the sd.cpp MiniMax-H3 docs.

        [Fact]
        public void WorkedExampleMatchesReferenceShapes()
        {
            var s = MiniMaxH3Geometry.Resolve(864, 480, 56);

            Assert.Equal(864, s.Width);
            Assert.Equal(480, s.Height);
            Assert.Equal(56, s.Frames);
            Assert.Equal(24, s.Fps);

            Assert.Equal(54, s.LatentWidth);     // 864 / 16
            Assert.Equal(30, s.LatentHeight);    // 480 / 16
            Assert.Equal(17, s.LatentFrames);    // ((56-5)/17)*5 + 2

            Assert.Equal(27, s.TokenGridWidth);  // 54 / 2 (patch)
            Assert.Equal(15, s.TokenGridHeight); // 30 / 2
            Assert.Equal(6885, s.VideoTokenCount);   // 17 * 27 * 15

            Assert.Equal(93, s.AudioLatentFrames);   // round(56 * 40 / 24)
            Assert.Equal(186, s.AudioTokenCount);    // stereo: 2 tokens per latent frame

            // The audio latent is flat-copied into channels appended after the 24 video
            // channels; one extra channel is enough here and its tail is zero padding.
            Assert.Equal(1, s.PackedExtraChannels);
            Assert.Equal(25, s.PackedChannels);
            Assert.Equal(660960L, s.VideoLatentLength);   // 54*30*17*24
            Assert.Equal(5952L, s.AudioLatentLength);     // 93*2*32
            Assert.Equal(688500L, s.PackedLatentLength);  // 54*30*17*25
            // The pad is the slack in the appended channel.
            Assert.Equal(21588L, s.PackedLatentLength - s.VideoLatentLength - s.AudioLatentLength);
        }

        [Theory]
        // The decoder's 5k+2 -> 17k+5 inverse: this is what makes a long clip come back
        // with exactly the frames that were asked for, and it is the arithmetic the
        // temporal chunking has to land on.
        [InlineData(2, 5)]
        [InlineData(7, 22)]
        [InlineData(12, 39)]
        [InlineData(17, 56)]
        [InlineData(27, 90)]
        [InlineData(37, 124)]
        public void DecodedFrameCountInvertsTheLatentGrid(int latentFrames, int frames)
        {
            Assert.Equal(frames, MiniMaxH3VideoVae.DecodedFrameCount(latentFrames));
            Assert.Equal(latentFrames, MiniMaxH3Geometry.VideoFramesToLatentFrames(frames));
        }

        [Theory]
        // The chunk loop must cover every latent frame for any length on the grid, and
        // it has to produce at least one chunk even for the shortest clip.
        [InlineData(2)]
        [InlineData(7)]
        [InlineData(12)]
        [InlineData(17)]
        [InlineData(27)]
        [InlineData(37)]
        [InlineData(52)]
        public void TemporalDecodeChunkingCoversTheWholeClip(int latentFrames)
        {
            const int perChunk = 5, drop = 3, overlap = 2, framesPerChunk = 20, prePad = 3;
            int pseudo = latentFrames + drop;
            int pad = (perChunk - pseudo % perChunk) % perChunk;
            pseudo += pad;
            int chunks = pseudo / perChunk - 1;
            if (chunks < 1) { pad += perChunk; chunks += 1; }
            Assert.True(chunks >= 1, "every clip needs at least one decode chunk");

            int paddedT = latentFrames + pad;
            // The last chunk must reach the end of the padded latent, or the tail of
            // the clip would silently never be decoded.
            int lastStart = (chunks - 1) * perChunk;
            Assert.True(lastStart < paddedT, $"chunk {chunks - 1} starts past the latent");

            int produced = 0;
            for (int i = 0; i < chunks; i++)
            {
                int end = Math.Min(lastStart == i * perChunk ? paddedT : i * perChunk + perChunk + overlap, paddedT);
                int take = end - i * perChunk;
                int decoded = take * MiniMaxH3Geometry.VaeTemporalRatio;
                int firstEnd = Math.Min(framesPerChunk, decoded);
                produced += firstEnd - Math.Min(prePad, firstEnd);
                if (i == chunks - 1 && decoded > framesPerChunk + prePad)
                    produced += decoded - (framesPerChunk + prePad);
            }
            Assert.True(produced >= MiniMaxH3VideoVae.DecodedFrameCount(latentFrames),
                $"chunking yields {produced} frames but the clip needs " +
                $"{MiniMaxH3VideoVae.DecodedFrameCount(latentFrames)}");
        }

        [Fact]
        public void KeyframesAreFittedToTheCanvasAndOrderedStartThenEnd()
        {
            // A keyframe is conditioned TWICE: the VAE latent pins its frame, and the
            // same picture goes through the vision tower into the prompt so it steers
            // the whole clip. Both halves must see the SAME canvas-sized pixels — the
            // tower's token count follows the image's own size, so presenting the
            // original photo is a different signal, not a sharper one.
            var shape = MiniMaxH3Geometry.Resolve(384, 288, 22, MiniMaxH3Geometry.Fps);
            var wide = new TensorSharp.Models.QwenImage.RgbImage(
                1024, 768, new float[1024L * 768 * 3]);
            var tall = new TensorSharp.Models.QwenImage.RgbImage(
                600, 800, new float[600L * 800 * 3]);

            var frames = new List<TensorSharp.Models.QwenImage.RgbImage>();
            var atEnd = new List<bool>();
            MiniMaxH3Pipeline.ResolveKeyframes(
                new VideoGenerationParams { Image = wide, EndImage = tall }, shape, frames, atEnd);

            Assert.Equal(2, frames.Count);
            Assert.Equal(new[] { false, true }, atEnd);
            foreach (var f in frames)
            {
                Assert.Equal(shape.Width, f.Width);
                Assert.Equal(shape.Height, f.Height);
            }

            // An end frame on its own is still marked as the END of the timeline.
            frames.Clear(); atEnd.Clear();
            MiniMaxH3Pipeline.ResolveKeyframes(
                new VideoGenerationParams { EndImage = tall }, shape, frames, atEnd);
            Assert.Single(frames);
            Assert.Equal(new[] { true }, atEnd);
        }

        [Fact]
        public void AudioLatentRateIs40Hz()
        {
            // 32 kHz / 800 (the audio VAE's total stride) = 40 latents per second.
            Assert.Equal(40, MiniMaxH3Geometry.AudioLatentCount(24));
            Assert.Equal(80, MiniMaxH3Geometry.AudioLatentCount(48));
            // Never degenerates to zero for a tiny clip.
            Assert.Equal(1, MiniMaxH3Geometry.AudioLatentCount(0));
        }

        [Fact]
        public void ResolveAlignsAndPinsFpsTo24()
        {
            var s = MiniMaxH3Geometry.Resolve(500, 300, 30, fps: 16);
            Assert.Equal(512, s.Width);
            Assert.Equal(320, s.Height);
            Assert.Equal(39, s.Frames);   // 30 -> next 17k+5 rung
            Assert.Equal(24, s.Fps);      // H3 is a 24 fps model
            // ...and the soundtrack is laid out on that 24 fps timeline too, not on the
            // rate that was asked for: 39 frames are 1.625 s, 65 audio latents at 40/s.
            Assert.Equal(MiniMaxH3Geometry.AudioLatentCount(39), s.AudioLatentFrames);
            Assert.Equal(65, s.AudioLatentFrames);
        }

        // ---- flow-matching schedule ------------------------------------------

        [Fact]
        public void SigmaScheduleMatchesReferenceAtShift12()
        {
            // DiscreteFlowDenoiser, 20 steps, shift 12 (the reference default).
            float[] expected =
            {
                1.0000f, 0.9954f, 0.9903f, 0.9846f, 0.9783f, 0.9711f, 0.9630f,
                0.9537f, 0.9430f, 0.9304f, 0.9154f, 0.8974f, 0.8753f, 0.8475f,
                0.8114f, 0.7628f, 0.6937f, 0.5877f, 0.4045f, 0.0119f, 0.0000f,
            };
            var sigmas = MiniMaxH3Scheduler.BuildSigmas(20);
            Assert.Equal(expected.Length, sigmas.Length);
            for (int i = 0; i < expected.Length; i++)
                Assert.Equal(expected[i], sigmas[i], 4);
        }

        [Fact]
        public void SigmaScheduleStartsAtOneAndEndsAtZeroAndDecreases()
        {
            foreach (int steps in new[] { 1, 4, 8, 20, 50 })
            {
                var sigmas = MiniMaxH3Scheduler.BuildSigmas(steps);
                Assert.Equal(steps + 1, sigmas.Length);
                Assert.Equal(1.0f, sigmas[0], 5);
                Assert.Equal(0.0f, sigmas[steps]);
                for (int i = 1; i < sigmas.Length; i++)
                    Assert.True(sigmas[i] < sigmas[i - 1], $"step {i} did not decrease at {steps} steps");
            }
        }

        // ---- the dual video/audio shift --------------------------------------

        [Theory]
        // Unshift by 12, reshift by 3. Both schedules pin the endpoints.
        [InlineData(1.00f, 1.0000f)]
        [InlineData(0.90f, 0.6923f)]
        [InlineData(0.75f, 0.4286f)]
        [InlineData(0.50f, 0.2000f)]
        [InlineData(0.25f, 0.0769f)]
        [InlineData(0.10f, 0.0270f)]
        public void AudioSigmaMappingMatchesReference(float videoSigma, float audioSigma)
            => Assert.Equal(audioSigma, MiniMaxH3Scheduler.MapSigma(videoSigma), 4);

        [Theory]
        // d(sigma_audio)/d(sigma_video): the factor the audio velocity is scaled by so a
        // single Euler step on the video schedule advances audio along its own.
        [InlineData(1.00f, 4.000f)]
        [InlineData(0.75f, 1.306f)]
        [InlineData(0.50f, 0.640f)]
        [InlineData(0.25f, 0.379f)]
        [InlineData(0.01f, 0.254f)]
        public void AudioShiftSlopeMatchesReference(float videoSigma, float slope)
            => Assert.Equal(slope, MiniMaxH3Scheduler.ShiftSlope(videoSigma), 3);

        [Fact]
        public void AudioShiftSlopeApproachesTheShiftRatioAtZero()
        {
            // As sigma -> 0 the slope tends to audioShift/videoShift = 3/12.
            Assert.Equal(0.25f, MiniMaxH3Scheduler.ShiftSlope(1e-6f), 4);
        }

        [Fact]
        public void ShiftSlopeIsTheDerivativeOfTheSigmaMapping()
        {
            // Independent check: finite-difference MapSigma and compare to ShiftSlope.
            foreach (float sigma in new[] { 0.05f, 0.2f, 0.4f, 0.6f, 0.8f, 0.95f })
            {
                const float h = 1e-4f;
                float numeric = (MiniMaxH3Scheduler.MapSigma(sigma + h)
                                 - MiniMaxH3Scheduler.MapSigma(sigma - h)) / (2 * h);
                Assert.Equal(numeric, MiniMaxH3Scheduler.ShiftSlope(sigma), 2);
            }
        }

        [Theory]
        [InlineData(12f, 3f)]
        [InlineData(5f, 3f)]
        [InlineData(3f, 3f)]
        public void AudioCarrierVelocityMatchesIndependentTrajectoryDerivative(float videoShift, float audioShift)
        {
            // Independent continuous flow: x(a)=clean+a*(noise-clean),
            // a=b/(r+(1-r)*b), c(b)=b*x(a)/a. Its derivative is noise-r*clean.
            const double clean = 0.37, noise = -1.23;
            double ratio = videoShift / (double)audioShift;
            double Carrier(double b)
            {
                double a = b / (ratio + (1 - ratio) * b);
                return b / a * (clean + a * (noise - clean));
            }
            foreach (float sigma in new[] { 1f, 0.9f, 0.5f, 0.01f, 1e-6f })
            {
                float[] carrier = [(float)Carrier(sigma)];
                float[] input = MiniMaxH3Scheduler.AudioInputFromCarrier(carrier, sigma, videoShift, audioShift);
                double a = sigma / (ratio + (1 - ratio) * sigma);
                Assert.Equal((float)(clean + a * (noise - clean)), input[0], 5);
                float[] velocity = [(float)(noise - clean) * MiniMaxH3Scheduler.ShiftSlope(sigma, videoShift, audioShift)];
                MiniMaxH3Scheduler.ConvertAudioVelocityToCarrier(input, velocity, sigma, videoShift, audioShift);
                Assert.Equal((float)(noise - ratio * clean), velocity[0], 5);
                if (sigma > 0.01f && sigma < 1f)
                {
                    const double h = 1e-6;
                    Assert.Equal((float)((Carrier(sigma + h) - Carrier(sigma - h)) / (2 * h)), velocity[0], 5);
                }
            }
        }

        [Fact]
        public void SingleAudioCarrierStepIncludesNoiseCancellingDrift()
        {
            float[] carrier = [1.25f, -0.5f];
            float[] input = MiniMaxH3Scheduler.AudioInputFromCarrier(carrier, 1f);
            // Zero clean endpoint: the physical flow velocity is the input noise.
            float[] velocity = [4f * input[0], 4f * input[1]];
            MiniMaxH3Scheduler.ConvertAudioVelocityToCarrier(input, velocity, 1f);
            MiniMaxH3Scheduler.EulerStep(carrier, velocity, 1f, 0f);
            Assert.Equal(new[] { 0f, 0f }, carrier);
        }

        [Theory]
        [InlineData(12f, 3f)]
        [InlineData(6f, 4f)]
        [InlineData(1.5f, 3f)]
        public void AudioCarrierEndpointRestoresCleanPhysicalAudio(float videoShift, float audioShift)
        {
            // Independent linear physical flow x(a)=clean+a*(noise-clean).
            // At b=1 the carrier equals noise; one Euler step produces r*clean,
            // and process_latent_out must undo r=videoShift/audioShift.
            float[] clean = [.37f, -.81f], noise = [-1.23f, .92f];
            float[] carrier = (float[])noise.Clone();
            float[] input = MiniMaxH3Scheduler.AudioInputFromCarrier(carrier, 1, videoShift, audioShift);
            float slope = MiniMaxH3Scheduler.ShiftSlope(1, videoShift, audioShift);
            float[] velocity = [(noise[0] - clean[0]) * slope, (noise[1] - clean[1]) * slope];
            MiniMaxH3Scheduler.ConvertAudioVelocityToCarrier(input, velocity, 1, videoShift, audioShift);
            MiniMaxH3Scheduler.EulerStep(carrier, velocity, 1, 0);
            for (int i = 0; i < clean.Length; ++i)
                Assert.InRange(Math.Abs(carrier[i] - clean[i] * videoShift / audioShift), 0, 1e-6);
            MiniMaxH3Scheduler.AudioOutputFromCarrierInPlace(carrier, videoShift, audioShift);
            for (int i = 0; i < clean.Length; ++i) Assert.InRange(Math.Abs(carrier[i] - clean[i]), 0, 1e-6);
        }

        [Fact]
        public void SigmaMappingIsAnInvolutionBetweenShifts()
        {
            // Mapping 12 -> 3 and back must be the identity.
            foreach (float sigma in new[] { 0.1f, 0.25f, 0.5f, 0.75f, 0.99f })
            {
                float audio = MiniMaxH3Scheduler.MapSigma(sigma, 12f, 3f);
                float back = MiniMaxH3Scheduler.MapSigma(audio, 3f, 12f);
                Assert.Equal(sigma, back, 5);
            }
        }

        [Fact]
        public void TimestepsAreOneMinusSigma()
        {
            Assert.Equal(0f, MiniMaxH3Scheduler.VideoTimestep(1f), 6);
            Assert.Equal(1f, MiniMaxH3Scheduler.VideoTimestep(0f), 6);
            // At sigma = 0.5 the audio stream is much further denoised than the video
            // stream, because shift 3 front-loads less noise than shift 12.
            Assert.Equal(0.5f, MiniMaxH3Scheduler.VideoTimestep(0.5f), 6);
            Assert.Equal(0.8f, MiniMaxH3Scheduler.AudioTimestep(0.5f), 4);
        }

        [Fact]
        public void EulerStepAppliesVelocityTimesSigmaDelta()
        {
            var x = new[] { 1f, 2f, 3f };
            var v = new[] { 0.5f, -1f, 2f };
            MiniMaxH3Scheduler.EulerStep(x, v, 0.8f, 0.6f);   // dt = -0.2
            Assert.Equal(1f + 0.5f * -0.2f, x[0], 5);
            Assert.Equal(2f + -1f * -0.2f, x[1], 5);
            Assert.Equal(3f + 2f * -0.2f, x[2], 5);
        }

        // ---- packed sequence layout ------------------------------------------

        [Fact]
        public void PackedLayoutOrdersTextConditionsAudioThenVideo()
        {
            var shape = MiniMaxH3Geometry.Resolve(864, 480, 56);
            var layout = MiniMaxH3Layout.Build(textLength: 12, shape, sigma: 0.5f);

            // Sequence order is fixed: text, then conditions, then TARGET AUDIO, then
            // TARGET VIDEO. Audio precedes video in the token sequence even though it is
            // stored after video in the packed latent buffer.
            Assert.Collection(layout.Segments,
                s => Assert.Equal(MiniMaxH3SegmentKind.Text, s.Kind),
                s => Assert.Equal(MiniMaxH3SegmentKind.TargetAudio, s.Kind),
                s => Assert.Equal(MiniMaxH3SegmentKind.TargetVideo, s.Kind));

            Assert.Equal(12, layout.AudioSpan.Start);
            Assert.Equal(12 + 186, layout.AudioSpan.End);
            Assert.Equal(12 + 186, layout.VideoSpan.Start);
            Assert.Equal(12 + 186 + 6885, layout.VideoSpan.End);
            Assert.Equal(12 + 186 + 6885, layout.TokenCount);

            // Three floats (t, h, w) per token.
            Assert.Equal(layout.TokenCount * 3, layout.Positions.Length);
        }

        [Fact]
        public void ModulationSpansTileTheSequenceWithoutGaps()
        {
            var shape = MiniMaxH3Geometry.Resolve(512, 320, 22);
            var layout = MiniMaxH3Layout.Build(textLength: 7, shape, sigma: 0.7f);

            int cursor = 0;
            foreach (var span in layout.ModulationSpans)
            {
                Assert.Equal(cursor, span.Start);
                Assert.True(span.End > span.Start);
                cursor = span.End;
            }
            Assert.Equal(layout.TokenCount, cursor);
        }

        [Fact]
        public void PlainTextProducesOneSpanWithAValidModality()
        {
            // Regression: the run-splitting loop used -1 as a terminator, and with no
            // explicit modalities that sentinel became the last text span's modality,
            // yielding a negative AdaLN row.
            var shape = MiniMaxH3Geometry.Resolve(512, 320, 22);
            var layout = MiniMaxH3Layout.Build(textLength: 5, shape, sigma: 0.5f);

            int textSpans = 0;
            foreach (var span in layout.ModulationSpans)
            {
                Assert.True(span.Row >= 0, $"span [{span.Start},{span.End}) has negative row {span.Row}");
                if (span.End <= 5) textSpans++;
            }
            Assert.Equal(1, textSpans);
            Assert.Equal(0 * 3 + MiniMaxH3Modality.Text, FindSpanCovering(layout, 4).Row);
        }

        [Fact]
        public void EveryModulationRowIsWithinTheProjectionsRange()
        {
            var shape = MiniMaxH3Geometry.Resolve(864, 480, 56);
            var modalities = new List<int>
            {
                MiniMaxH3Modality.Text, MiniMaxH3Modality.Visual, MiniMaxH3Modality.Text,
            };
            var layout = MiniMaxH3Layout.Build(3, shape, 0.7f, textTokenModalities: modalities);
            int rows = layout.Timesteps.Length * MiniMaxH3Modality.Count;
            foreach (var span in layout.ModulationSpans)
            {
                Assert.InRange(span.Row, 0, rows - 1);
                Assert.InRange(span.Row % MiniMaxH3Modality.Count, 0, MiniMaxH3Modality.Count - 1);
            }
        }

        [Fact]
        public void ModulationRowsEncodeTimestepTimesThreePlusModality()
        {
            var shape = MiniMaxH3Geometry.Resolve(512, 320, 22);
            var layout = MiniMaxH3Layout.Build(textLength: 4, shape, sigma: 0.5f);

            // Timesteps are registered video, audio, condition, audio-condition in that
            // order, and every row is timestepIndex*3 + modality.
            Assert.Equal(MiniMaxH3Scheduler.VideoTimestep(0.5f), layout.Timesteps[0], 5);
            Assert.Equal(MiniMaxH3Scheduler.AudioTimestep(0.5f), layout.Timesteps[1], 5);

            var text = layout.ModulationSpans[0];
            Assert.Equal(0 * 3 + MiniMaxH3Modality.Text, text.Row);

            var audio = FindSpanCovering(layout, layout.AudioSpan.Start);
            Assert.Equal(1 * 3 + MiniMaxH3Modality.Audio, audio.Row);

            var video = FindSpanCovering(layout, layout.VideoSpan.Start);
            Assert.Equal(0 * 3 + MiniMaxH3Modality.Visual, video.Row);
        }

        [Fact]
        public void VisionTaggedTextTokensModulateOnTheVisualStream()
        {
            var shape = MiniMaxH3Geometry.Resolve(512, 320, 22);
            // A run of image tokens embedded in the text stream: text, text, image, image, text.
            var modalities = new List<int>
            {
                MiniMaxH3Modality.Text, MiniMaxH3Modality.Text,
                MiniMaxH3Modality.Visual, MiniMaxH3Modality.Visual,
                MiniMaxH3Modality.Text,
            };
            var layout = MiniMaxH3Layout.Build(5, shape, 0.5f, textTokenModalities: modalities);

            Assert.Equal(0 * 3 + MiniMaxH3Modality.Text, FindSpanCovering(layout, 0).Row);
            Assert.Equal(0 * 3 + MiniMaxH3Modality.Visual, FindSpanCovering(layout, 2).Row);
            Assert.Equal(0 * 3 + MiniMaxH3Modality.Text, FindSpanCovering(layout, 4).Row);
        }

        [Fact]
        public void ConditioningFramesSitAtANearCleanTimestepAndPinToTheTimeline()
        {
            var shape = MiniMaxH3Geometry.Resolve(512, 320, 22);
            int textLen = 3;
            // Two keyframes: a start frame and an end frame (FL2VA).
            var layout = MiniMaxH3Layout.Build(
                textLen, shape, sigma: 0.5f,
                conditionFrames: new[] { 1, 1 },
                keyframeAtEnd: new[] { false, true });

            var cond = new List<MiniMaxH3SequenceSegment>();
            foreach (var s in layout.Segments)
                if (s.Kind == MiniMaxH3SegmentKind.ConditionVideo) cond.Add(s);
            Assert.Equal(2, cond.Count);

            // Conditioning uses the max(videoTimestep, 0.999) row, distinct from the target's.
            float condTimestep = Math.Max(MiniMaxH3Scheduler.VideoTimestep(0.5f),
                                          MiniMaxH3Scheduler.VisualConditionTimestep);
            int condRow = Array.IndexOf(layout.Timesteps, condTimestep);
            Assert.True(condRow >= 0, "conditioning timestep was not registered");
            Assert.Equal(condRow * 3 + MiniMaxH3Modality.Visual, FindSpanCovering(layout, cond[0].Start).Row);

            // The start keyframe sits at the head of the timeline; the end keyframe sits one
            // video span before the clip's end.
            float startT = layout.Positions[cond[0].Start * 3];
            float endT = layout.Positions[cond[1].Start * 3];
            Assert.Equal(textLen, startT, 4);
            Assert.Equal(textLen + MiniMaxH3Layout.VideoDuration(shape.LatentFrames)
                         - MiniMaxH3Layout.FrameRescale, endT, 4);
            Assert.True(endT > startT);
        }

        [Fact]
        public void ALastFrameAloneIsOneConditioningBlockPinnedToTheEndOfTheClip()
        {
            // L2VA: FL2VA with only its last frame. No placeholder takes the first frame's
            // place: one block, at the end time sglang/vllm-omni give frame_index -1.
            var shape = MiniMaxH3Geometry.Resolve(512, 320, 22);
            int textLen = 3;
            var layout = MiniMaxH3Layout.Build(
                textLen, shape, sigma: 0.5f, conditionFrames: new[] { 1 }, keyframeAtEnd: new[] { true });
            var cond = new List<MiniMaxH3SequenceSegment>();
            foreach (var s in layout.Segments)
                if (s.Kind == MiniMaxH3SegmentKind.ConditionVideo) cond.Add(s);
            var only = Assert.Single(cond);
            Assert.Equal(shape.TokenGridHeight * shape.TokenGridWidth, only.End - only.Start);
            Assert.Equal(textLen + MiniMaxH3Layout.VideoDuration(shape.LatentFrames)
                         - MiniMaxH3Layout.FrameRescale, layout.Positions[only.Start * 3], 4);

            // The pipeline's fit gives the lone last keyframe one end-pinned frame and nothing at the start.
            var frames = new List<TensorSharp.Models.QwenImage.RgbImage>(); var atEnd = new List<bool>();
            MiniMaxH3Pipeline.ResolveKeyframes(null, new(64, 40, new float[64 * 40 * 3]), shape, frames, atEnd, unsetCanvas: null);
            Assert.Equal((shape.Width, shape.Height), (Assert.Single(frames).Width, frames[0].Height));
            Assert.Equal(new[] { true }, atEnd);
        }

        [Fact]
        public void VideoTimeAxisAdvancesOnTheOneFourFourFourFourSpan()
        {
            // Latent frame 0 covers a single pixel frame; the next four cover four each.
            // 1 + 4*4 = 17, which is exactly the frame grid, and the 5/3 rescale puts the
            // axis in audio-latent units so both streams share one timeline.
            Assert.Equal(5f / 3f, MiniMaxH3Layout.VideoSpan(0), 5);
            Assert.Equal(20f / 3f, MiniMaxH3Layout.VideoSpan(1), 5);
            Assert.Equal(20f / 3f, MiniMaxH3Layout.VideoSpan(4), 5);
            Assert.Equal(5f / 3f, MiniMaxH3Layout.VideoSpan(5), 5);   // wraps every 5

            // A 17-latent-frame clip spans 56 pixel frames = 56 * 5/3 audio-latent units.
            Assert.Equal(56 * 5f / 3f, MiniMaxH3Layout.VideoDuration(17), 3);
            Assert.Equal(22 * 5f / 3f, MiniMaxH3Layout.VideoDuration(7), 3);
        }

        [Fact]
        public void AudioTimeAxisAdvancesOnePerLatentFrameAndStraddlesTheWidthAxis()
        {
            var shape = MiniMaxH3Geometry.Resolve(512, 320, 22);
            int textLen = 2;
            var layout = MiniMaxH3Layout.Build(textLen, shape, 0.5f);
            var (_, wAxis) = MiniMaxH3Layout.SpatialAxes(shape.LatentHeight, shape.LatentWidth);

            int a0 = layout.AudioSpan.Start;
            int n = shape.AudioLatentFrames;

            // Channel 0 runs first, anchored to the left edge of the width axis.
            Assert.Equal(textLen + 0, layout.Positions[a0 * 3], 4);
            Assert.Equal(0f, layout.Positions[a0 * 3 + 1], 4);
            Assert.Equal(wAxis[0], layout.Positions[a0 * 3 + 2], 4);
            Assert.Equal(textLen + 1, layout.Positions[(a0 + 1) * 3], 4);

            // Channel 1 restarts the clock, anchored to the right edge.
            Assert.Equal(textLen + 0, layout.Positions[(a0 + n) * 3], 4);
            Assert.Equal(wAxis[wAxis.Length - 1], layout.Positions[(a0 + n) * 3 + 2], 4);
        }

        [Fact]
        public void SpatialAxisIsAspectNormalisedAboutASharedCentre()
        {
            var axis = MiniMaxH3Layout.SpatialAxis(30, Math.Sqrt(30.0 * 30.0));
            Assert.Equal(15, axis.Length);                       // 30 / patch 2
            // Evenly spaced.
            float step = axis[1] - axis[0];
            for (int i = 1; i < axis.Length; i++)
                Assert.Equal(step, axis[i] - axis[i - 1], 4);

            // The invariant that makes this a sane positional encoding: both spatial axes
            // are centred on the SAME point whatever the aspect ratio, so only the relative
            // extent carries aspect information — a 30x54 latent and a 54x30 latent describe
            // the same geometry rather than being shifted apart.
            var (h, w) = MiniMaxH3Layout.SpatialAxes(30, 54);
            Assert.Equal(15, h.Length);
            Assert.Equal(27, w.Length);
            float hMid = (h[0] + h[h.Length - 1]) / 2f;
            float wMid = (w[0] + w[w.Length - 1]) / 2f;
            Assert.Equal(hMid, wMid, 4);

            // ...and the wider side spans the wider coordinate range.
            float hRange = h[h.Length - 1] - h[0];
            float wRange = w[w.Length - 1] - w[0];
            Assert.True(wRange > hRange,
                $"the wider axis should span more RoPE coordinate range ({wRange} vs {hRange})");

            // A square latent puts both axes on exactly the same coordinates.
            var (sh, sw) = MiniMaxH3Layout.SpatialAxes(32, 32);
            for (int i = 0; i < sh.Length; i++) Assert.Equal(sh[i], sw[i], 5);
        }

        // ---- AdaLN curve table -----------------------------------------------

        [Fact]
        public void CurveIndexInterpolatesAcrossTheGridAndClampsAtTheEnds()
        {
            const int grid = 1025;   // the checkpoint's adaln_t_table has 1025 entries

            var mid = MiniMaxH3Layout.CurveIndex(0.5f, grid);
            Assert.Equal(512, mid.Lower);
            Assert.Equal(513, mid.Upper);
            Assert.Equal(0f, mid.Fraction, 4);

            var low = MiniMaxH3Layout.CurveIndex(0f, grid);
            Assert.Equal(0, low.Lower);
            Assert.Equal(0f, low.Fraction, 5);

            // The top of the range must land on the last interval, never past it.
            var high = MiniMaxH3Layout.CurveIndex(1f, grid);
            Assert.Equal(grid - 2, high.Lower);
            Assert.Equal(grid - 1, high.Upper);
            Assert.Equal(1f, high.Fraction, 5);

            // Out-of-range timesteps clamp rather than index out of bounds.
            Assert.Equal(0, MiniMaxH3Layout.CurveIndex(-3f, grid).Lower);
            Assert.Equal(grid - 2, MiniMaxH3Layout.CurveIndex(9f, grid).Lower);
        }

        [Fact]
        public void VaeTilePlanCoversTheAxisExactlyWithSaneOverlaps()
        {
            // The reference's 640x384 decode is 3 tiles by 2, overlapping by 0.25 and
            // 0.5 of a tile. Tiles are always a full 16 latent voxels and the leftover
            // is absorbed into the seams, so coverage must land exactly on the axis.
            foreach (int length in new[] { 8, 16, 24, 40, 48, 60, 80 })
            {
                var tiles = MiniMaxH3VideoVae.PlanTiles(length);
                Assert.NotEmpty(tiles);
                Assert.Equal(0, tiles[0].Start);
                Assert.Equal(length, tiles[^1].Start + tiles[^1].Length);
                for (int i = 0; i < tiles.Count; i++)
                {
                    Assert.Equal(Math.Min(length, MiniMaxH3VideoVae.TileLatent), tiles[i].Length);
                    if (i > 0)
                    {
                        // Consecutive tiles must actually overlap by the declared amount.
                        int prevEnd = tiles[i - 1].Start + tiles[i - 1].Length;
                        Assert.Equal(prevEnd - tiles[i].Start, tiles[i].LeadOverlap);
                        Assert.Equal(tiles[i].LeadOverlap, tiles[i - 1].TrailOverlap);
                        Assert.True(tiles[i].LeadOverlap > 0,
                            $"tiles must overlap at length {length}");
                    }
                }
            }

            // The two extents the reference logged for 640x384.
            var w = MiniMaxH3VideoVae.PlanTiles(40);
            Assert.Equal(3, w.Count);
            Assert.Equal(4, w[1].LeadOverlap);      // 64 px = 0.25 of a 256 px tile
            var h = MiniMaxH3VideoVae.PlanTiles(24);
            Assert.Equal(2, h.Count);
            Assert.Equal(8, h[1].LeadOverlap);      // 128 px = 0.5 of a tile
        }

        // ---- conditioning modes ---------------------------------------------

        private static VideoGenerationParams P(string mode = null, bool first = false,
                                               bool last = false, bool refs = false) =>
            new()
            {
                Mode = mode,
                ImagePath = first ? "first.png" : null,
                EndImagePath = last ? "last.png" : null,
                ReferenceImagePaths = refs ? new List<string> { "ref.png" } : null,
            };

        [Fact]
        public void ModeIsInferredFromWhatTheRequestSupplies()
        {
            var fl = MiniMaxH3Partition.FirstLastFrame;
            Assert.Equal(MiniMaxH3Mode.TextToVideo, MiniMaxH3ModeResolver.Resolve(P(), fl));
            Assert.Equal(MiniMaxH3Mode.ImageToVideo,
                MiniMaxH3ModeResolver.Resolve(P(first: true), fl));
            Assert.Equal(MiniMaxH3Mode.FirstLastFrame,
                MiniMaxH3ModeResolver.Resolve(P(first: true, last: true), fl));
            // An end frame alone is still the first/last mode.
            Assert.Equal(MiniMaxH3Mode.FirstLastFrame,
                MiniMaxH3ModeResolver.Resolve(P(last: true), fl));
            Assert.Equal(MiniMaxH3Mode.Reference,
                MiniMaxH3ModeResolver.Resolve(P(refs: true), MiniMaxH3Partition.Reference));
        }

        [Theory]
        [InlineData("t2v", MiniMaxH3Mode.TextToVideo)]
        [InlineData("i2v", MiniMaxH3Mode.ImageToVideo)]
        [InlineData("image-to-video", MiniMaxH3Mode.ImageToVideo)]
        [InlineData("fl2v", MiniMaxH3Mode.FirstLastFrame)]
        [InlineData("first-last", MiniMaxH3Mode.FirstLastFrame)]
        [InlineData("ref", MiniMaxH3Mode.Reference)]
        [InlineData("ref2va", MiniMaxH3Mode.Reference)]
        public void ExplicitModeStringsParse(string text, MiniMaxH3Mode expected)
            => Assert.Equal(expected, MiniMaxH3ModeResolver.Parse(text));

        [Fact]
        public void UnknownModeStringIsRejected()
            => Assert.Throws<ArgumentException>(() => MiniMaxH3ModeResolver.Parse("animate"));

        [Fact]
        public void ModeMustMatchTheLoadedCheckpoint()
        {
            // The two denoisers are separate files, so asking for the wrong one has to
            // say so rather than silently doing something else.
            var ex = Assert.Throws<InvalidOperationException>(() =>
                MiniMaxH3ModeResolver.Resolve(P(refs: true), MiniMaxH3Partition.FirstLastFrame));
            Assert.Contains("Ref2VA", ex.Message);

            // An EXPLICIT keyframe mode on the reference checkpoint is still a mistake...
            var ex2 = Assert.Throws<InvalidOperationException>(() =>
                MiniMaxH3ModeResolver.Resolve(P("i2v", first: true), MiniMaxH3Partition.Reference));
            Assert.Contains("FL2VA", ex2.Message);
            var ex3 = Assert.Throws<InvalidOperationException>(() =>
                MiniMaxH3ModeResolver.Resolve(P(last: true), MiniMaxH3Partition.Reference));
            Assert.Contains("FL2VA", ex3.Message);
        }

        [Fact]
        public void PlainImageOnTheReferenceCheckpointIsTreatedAsAReference()
        {
            // A client that only knows how to attach "an image" — the Web UI among
            // them — should not be told to load a different model when the operator
            // already picked the reference one.
            Assert.Equal(MiniMaxH3Mode.Reference,
                MiniMaxH3ModeResolver.Resolve(P(first: true), MiniMaxH3Partition.Reference));
            // The same image on the keyframe checkpoint still means the first frame.
            Assert.Equal(MiniMaxH3Mode.ImageToVideo,
                MiniMaxH3ModeResolver.Resolve(P(first: true), MiniMaxH3Partition.FirstLastFrame));
        }

        [Fact]
        public void ExplicitModeWithoutItsInputsIsRejected()
        {
            var fl = MiniMaxH3Partition.FirstLastFrame;
            Assert.Throws<InvalidOperationException>(() =>
                MiniMaxH3ModeResolver.Resolve(P("i2v"), fl));
            Assert.Throws<InvalidOperationException>(() =>
                MiniMaxH3ModeResolver.Resolve(P("fl2v"), fl));
            Assert.Throws<InvalidOperationException>(() =>
                MiniMaxH3ModeResolver.Resolve(P("ref"), MiniMaxH3Partition.Reference));
            // ...and passing an image while asking for text-only is a mistake worth reporting.
            Assert.Throws<InvalidOperationException>(() =>
                MiniMaxH3ModeResolver.Resolve(P("t2v", first: true), fl));
        }

        [Fact]
        public void ExplicitModeOverridesInference()
        {
            // A reference request on the Ref2VA checkpoint stays a reference request
            // even though an image is present.
            Assert.Equal(MiniMaxH3Mode.Reference,
                MiniMaxH3ModeResolver.Resolve(P("ref", refs: true), MiniMaxH3Partition.Reference));
        }

        [Fact]
        public void ConditioningFramesLandBeforeTheTargetStreams()
        {
            // Keyframes are extra sequence tokens between the text and the target
            // audio, so both target spans must shift by their token count.
            var shape = MiniMaxH3Geometry.Resolve(512, 320, 22);
            int perFrame = shape.TokenGridWidth * shape.TokenGridHeight;
            var plain = MiniMaxH3Layout.Build(4, shape, 0.5f);
            var conditioned = MiniMaxH3Layout.Build(4, shape, 0.5f,
                conditionFrames: new[] { 1, 1 }, keyframeAtEnd: new[] { false, true });

            Assert.Equal(plain.TokenCount + 2 * perFrame, conditioned.TokenCount);
            Assert.Equal(plain.AudioSpan.Start + 2 * perFrame, conditioned.AudioSpan.Start);
            Assert.Equal(plain.VideoSpan.Start + 2 * perFrame, conditioned.VideoSpan.Start);
        }

        // ---- output size resolution -----------------------------------------

        private static TensorSharp.Models.QwenImage.RgbImage Img(int w, int h) =>
            new(w, h, new float[(long)w * h * 3]);

        [Fact]
        public void CanvasDefaultsToTheModelsWorkingAreaNotAThumbnail()
        {
            // Regression: the default used to be 256x256, which is far too small for
            // faces to hold together — the Web UI sends no size, so every clip came
            // back blurry with no way for the user to tell why.
            var (w, h) = MiniMaxH3Pipeline.ResolveCanvas(new VideoGenerationParams(), MiniMaxH3Mode.TextToVideo);
            Assert.Equal(640, w);
            Assert.Equal(384, h);
        }

        [Fact]
        public void CompactPreviewTakesItsAspectFromTheConditioningImage()
        {
            // "Animate this photo" should not letterbox or stretch the photo it is
            // meant to become. The compact host policy keeps 640x384's area.
            var compact = VideoCanvasPolicy.CompactPreview;
            var (w, h) = MiniMaxH3Pipeline.ResolveCanvas(
                new VideoGenerationParams { Image = Img(1024, 768), CanvasPolicy = compact }, MiniMaxH3Mode.ImageToVideo);
            Assert.InRange(w / (double)h, 1.30, 1.37);          // 4:3 preserved
            Assert.InRange((long)w * h, 640L * 384 / 2, 640L * 384 * 2);

            var (pw, ph) = MiniMaxH3Pipeline.ResolveCanvas(
                new VideoGenerationParams { Image = Img(768, 1024), CanvasPolicy = compact }, MiniMaxH3Mode.ImageToVideo);
            Assert.InRange(pw / (double)ph, 0.72, 0.78);        // 3:4 preserved
        }

        [Theory]
        [InlineData(1253, 836, 1152, 768)]     // the 3:2 banner of the reported fl2v run
        [InlineData(3808, 5712, 768, 1152)]    // a full-body phone portrait
        [InlineData(1920, 1080, 1344, 768)]    // 16:9 hits the 768x1344 pixel budget
        [InlineData(640, 384, 1280, 768)]
        [InlineData(4032, 3024, 1024, 768)]
        [InlineData(2560, 1080, 1568, 672)]    // 21:9 shrinks to the budget
        [InlineData(1, 4, 512, 2016)]
        [InlineData(4, 1, 2016, 512)]
        [InlineData(49, 48, 768, 768)]         // 784 is a tie (24.5 steps) and rounds to even
        public void KeyframeCanvasFollowsMiniMaxsReleasedRecipe(int w, int h, int expectedW, int expectedH)
        {
            // vllm-omni resolve_minimax_h3_output_canvas and sglang adapt_shape_v1 give
            // these sizes: a 768-pixel short edge at the keyframe's aspect, under
            // 768x1344's area, each side on the NEAREST 32-px step.
            Assert.Equal((expectedW, expectedH), MiniMaxH3Geometry.KeyframeCanvas(w, h));
        }

        [Theory]
        [InlineData(5, 1, 2016, 512)]
        [InlineData(1, 5, 512, 2016)]
        [InlineData(4000, 900, 2016, 512)]
        public void KeyframeCanvasClampsShapesBeyondTheRecipesRange(int w, int h, int expectedW, int expectedH)
        {
            // Upstream refuses these; a request with no size must still generate, so the ratio is clamped
            // to 4:1 (or 1:4) and the cover fit crops the rest, which the keyframe warning reports.
            Assert.Equal((expectedW, expectedH), MiniMaxH3Geometry.KeyframeCanvas(w, h));
        }

        [Fact]
        public void KeyframeCanvasStaysOnTheGridAndCloseToTheImagesAspect()
        {
            for (int w = 4; w <= 400; w += 3)
                for (int h = 4; h <= 400; h += 7)
                {
                    double ratio = w / (double)h;
                    if (ratio < MiniMaxH3Geometry.RecipeMinAspect || ratio > MiniMaxH3Geometry.RecipeMaxAspect) continue;
                    var (cw, ch) = MiniMaxH3Geometry.KeyframeCanvas(w, h);
                    Assert.Equal(0, cw % 32); Assert.Equal(0, ch % 32);
                    Assert.True(Math.Min(cw, ch) <= MiniMaxH3Geometry.RecipeShortEdge, $"{w}x{h} -> {cw}x{ch}");
                    Assert.InRange(cw / (double)ch / ratio, 0.96, 1.04);
                }
        }

        [Fact]
        public void KeyframeModesTakeTheRecipeCanvasFromTheFirstKeyframe()
        {
            // These modes keep the keyframe's framing, so the canvas decides how many
            // pixels its faces get: the 3:2 banner sets 1152x768 for the whole clip,
            // and the portrait LAST frame is cropped into it rather than steering it.
            var first = Img(1253 / 4, 836 / 4);
            var last = Img(3808 / 32, 5712 / 32);
            Assert.Equal((1152, 768), MiniMaxH3Pipeline.ResolveCanvas(
                new VideoGenerationParams { Image = first, EndImage = last }, MiniMaxH3Mode.FirstLastFrame));
            Assert.Equal((1152, 768), MiniMaxH3Pipeline.ResolveCanvas(
                new VideoGenerationParams(), MiniMaxH3Mode.FirstLastFrame, first, last));
            Assert.Equal((768, 1152), MiniMaxH3Pipeline.ResolveCanvas(
                new VideoGenerationParams { EndImage = last }, MiniMaxH3Mode.FirstLastFrame));
            Assert.Equal((1152, 768), MiniMaxH3Pipeline.ResolveCanvas(
                new VideoGenerationParams { Image = first }, MiniMaxH3Mode.ImageToVideo));
        }

        [Fact]
        public void TextToVideoAndReferencesKeepTheCompactCanvasUnderTheRecipePolicy()
        {
            Assert.Equal((640, 384), MiniMaxH3Pipeline.ResolveCanvas(new VideoGenerationParams(), MiniMaxH3Mode.TextToVideo));
            Assert.Equal((640, 384), MiniMaxH3Pipeline.ResolveCanvas(
                new VideoGenerationParams { ReferenceImagePaths = new List<string> { "unused.png" } }, MiniMaxH3Mode.Reference));
        }

        [Fact]
        public void ExplicitSizeAlwaysWins()
        {
            var (w, h) = MiniMaxH3Pipeline.ResolveCanvas(
                new VideoGenerationParams { Width = 832, Height = 480, Image = Img(1024, 768) }, MiniMaxH3Mode.ImageToVideo);
            Assert.Equal(832, w);
            Assert.Equal(480, h);
        }

        [Fact]
        public void OneGivenDimensionTakesTheOtherFromTheImage()
        {
            var (w, h) = MiniMaxH3Pipeline.ResolveCanvas(
                new VideoGenerationParams { Width = 800, Image = Img(1024, 768) }, MiniMaxH3Mode.ImageToVideo);
            Assert.Equal(800, w);
            Assert.InRange(h, 590, 610);                        // 800 / (4/3)

            var (w2, h2) = MiniMaxH3Pipeline.ResolveCanvas(
                new VideoGenerationParams { Height = 600, Image = Img(1024, 768) }, MiniMaxH3Mode.ImageToVideo);
            Assert.Equal(600, h2);
            Assert.InRange(w2, 790, 810);
        }

        [Fact]
        public void EndImageAloneStillSetsTheAspect()
        {
            var (w, h) = MiniMaxH3Pipeline.ResolveCanvas(
                new VideoGenerationParams { EndImage = Img(1920, 1080) }, MiniMaxH3Mode.FirstLastFrame);
            Assert.InRange(w / (double)h, 1.72, 1.82);          // 16:9
        }

        [Fact]
        public void KeyframeFitReportsTheCropAndTheDetailItLoses()
        {
            // The reported run: a 3808x5712 portrait as the LAST keyframe on a forced 640x384 canvas, where an
            // unset size would have followed the 3:2 first keyframe to 1152x768.
            var lines = MiniMaxH3Pipeline.DescribeKeyframeFit("last", 3808, 5712, 640, 384, anchor: false, unsetCanvas: (1152, 768));
            Assert.Equal(2, lines.Count);
            Assert.Contains("middle 40% of its height", lines[0]);
            Assert.DoesNotContain("Leave the size unset", lines[0]); // the last keyframe would be cropped either way
            Assert.Contains("scaled down ", lines[1]);
            Assert.Contains("768x1152", lines[1]);
            Assert.Contains("Leave the size unset for 1152x768", lines[1]);

            // On the recipe canvas the same photo is still cropped, but no longer starved of pixels.
            var recipe = MiniMaxH3Pipeline.DescribeKeyframeFit("last", 3808, 5712, 1152, 768, anchor: false, unsetCanvas: null);
            Assert.Contains("middle 44% of its height", Assert.Single(recipe));
            Assert.Empty(MiniMaxH3Pipeline.DescribeKeyframeFit("first", 1253, 836, 1152, 768, anchor: true, unsetCanvas: null));
            // A 16:9 last keyframe on a square first keyframe's 768x768 recipe canvas is scaled exactly as far as
            // its own 1344x768 recipe would scale it: cropped, but not reported as losing detail.
            Assert.Single(MiniMaxH3Pipeline.DescribeKeyframeFit("last", 1920, 1080, 768, 768, anchor: false, unsetCanvas: null));
        }

        private static MiniMaxH3ModulationSpan FindSpanCovering(MiniMaxH3PackedLayout layout, int token)
        {
            foreach (var s in layout.ModulationSpans)
                if (token >= s.Start && token < s.End) return s;
            throw new InvalidOperationException($"no modulation span covers token {token}");
        }
    }
}
