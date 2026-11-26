// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// MiniMax-H3 Ref2VA with SEVERAL references.
//
// Everything here is about behaviour that cannot exist with one reference: blocks
// laid head to tail on a shared timeline, per-block cursor advance, per-reference
// aspect ratios, audio-before-video ordering inside a block, and the counters that
// number "<Picture 2>" and "<Audio 2>". The existing parity test
// (MiniMaxH3RefLayoutTests) checks a single reference image against a dump from
// stable-diffusion.cpp and is fixture-gated on top; none of the rules below were
// covered anywhere, which is why a video-with-audio reference could silently stop
// the audio counter and nothing failed.
//
// None of these need weights. The expectations are the reference implementation's
// rules written out by hand (stable-diffusion.cpp src/model/diffusion/minimax_h3.hpp
// build_layout, src/stable-diffusion.cpp around 5977-6120), not a re-derivation
// through the code under test.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using TensorSharp.Models.MiniMaxH3;
using TensorSharp.Models.QwenImage;
using TensorSharp.Models.Video;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests
{
    public class MiniMaxH3MultiReferenceTests
    {
        private readonly ITestOutputHelper _output;

        public MiniMaxH3MultiReferenceTests(ITestOutputHelper output) { _output = output; }

        private static MiniMaxH3ReferenceBlock ImageBlock(int index, int pixelWidth, int pixelHeight) =>
            new()
            {
                Kind = MiniMaxH3ReferenceKind.Image,
                VideoIndex = index,
                LatentFrames = 1,
                LatentWidth = pixelWidth / MiniMaxH3Geometry.VaeSpatialRatio,
                LatentHeight = pixelHeight / MiniMaxH3Geometry.VaeSpatialRatio,
            };

        // ---- the timeline ---------------------------------------------------

        [Fact]
        public void ThreeReferenceImagesTakeThreeConsecutiveSlotsOnTheTimeline()
        {
            // 288x224, 224x288 and 256x256: landscape, portrait and square, so a
            // reference that borrowed another's axes would be caught rather than
            // coincidentally agreeing.
            var shape = MiniMaxH3Geometry.Resolve(256, 256, 22);
            const int textLength = 10;
            var blocks = new[]
            {
                ImageBlock(0, 288, 224),
                ImageBlock(1, 224, 288),
                ImageBlock(2, 256, 256),
            };

            var layout = MiniMaxH3Layout.Build(textLength, shape, sigma: 1f, references: blocks);

            // 63 + 63 + 64 reference tokens: each is (latentH/2) * (latentW/2) of its OWN
            // latent, never the generated clip's.
            int[] expectedCounts = { 7 * 9, 9 * 7, 8 * 8 };
            Assert.Equal(63, expectedCounts[0]);
            int refTokens = expectedCounts.Sum();
            int expectedTotal = textLength + refTokens + shape.AudioTokenCount + shape.VideoTokenCount;
            _output.WriteLine($"[multi-ref] tokens={layout.TokenCount} (text {textLength} + refs " +
                              $"{refTokens} + audio {shape.AudioTokenCount} + video {shape.VideoTokenCount})");
            Assert.Equal(expectedTotal, layout.TokenCount);

            var refSegments = layout.Segments
                .Where(s => s.Kind == MiniMaxH3SegmentKind.ConditionVideo)
                .ToArray();
            Assert.Equal(3, refSegments.Length);

            // Head to tail, in the caller's order, immediately after the text.
            int cursor = textLength;
            for (int i = 0; i < 3; i++)
            {
                Assert.Equal(i, refSegments[i].SourceIndex);
                Assert.Equal(cursor, refSegments[i].Start);
                Assert.Equal(cursor + expectedCounts[i], refSegments[i].End);
                cursor += expectedCounts[i];
            }

            // A still costs exactly ONE unit of the shared timeline no matter how many
            // tokens it takes, so reference k sits at textLength + k and the generated
            // clip starts after the last one.
            for (int i = 0; i < 3; i++)
            {
                var times = TimesOf(layout, refSegments[i]);
                Assert.All(times, t => Assert.Equal(textLength + i, t, 4));
            }

            var video = layout.Segments.Single(s => s.Kind == MiniMaxH3SegmentKind.TargetVideo);
            Assert.Equal(textLength + 3f, TimesOf(layout, video).First(), 4);
        }

        [Fact]
        public void EachReferenceKeepsItsOwnAspectRatio()
        {
            var shape = MiniMaxH3Geometry.Resolve(256, 256, 22);
            var blocks = new[] { ImageBlock(0, 288, 224), ImageBlock(1, 224, 288) };
            var layout = MiniMaxH3Layout.Build(4, shape, sigma: 1f, references: blocks);

            var refSegments = layout.Segments
                .Where(s => s.Kind == MiniMaxH3SegmentKind.ConditionVideo).ToArray();

            foreach (var (segment, block) in refSegments.Zip(blocks))
            {
                var (expectedH, expectedW) =
                    MiniMaxH3Layout.SpatialAxes(block.LatentHeight, block.LatentWidth);
                var actualH = new SortedSet<float>();
                var actualW = new SortedSet<float>();
                for (int token = segment.Start; token < segment.End; token++)
                {
                    actualH.Add(layout.Positions[token * 3 + 1]);
                    actualW.Add(layout.Positions[token * 3 + 2]);
                }
                Assert.Equal(expectedH.Length, actualH.Count);
                Assert.Equal(expectedW.Length, actualW.Count);
                Assert.All(expectedH, h => Assert.Contains(actualH, a => Math.Abs(a - h) < 1e-4f));
                Assert.All(expectedW, w => Assert.Contains(actualW, a => Math.Abs(a - w) < 1e-4f));
            }

            // The landscape and portrait references must NOT come out with the same axes;
            // that is the failure a single-reference test cannot see.
            var first = refSegments[0];
            var second = refSegments[1];
            Assert.NotEqual(layout.Positions[first.Start * 3 + 1],
                            layout.Positions[second.Start * 3 + 1]);
        }

        [Fact]
        public void AudioComesBeforeVideoInsideOneReferenceBlock()
        {
            // The reference pushes an AUDIO presentation item and then a VIDEO one for a
            // clip that carries sound (stable-diffusion.cpp:6079-6097), and the layout
            // emits the same two segments in that order.
            var shape = MiniMaxH3Geometry.Resolve(256, 256, 22);
            var blocks = new[]
            {
                ImageBlock(0, 256, 256),
                new MiniMaxH3ReferenceBlock
                {
                    Kind = MiniMaxH3ReferenceKind.VideoAudio,
                    VideoIndex = 1,
                    AudioIndex = 0,
                    LatentFrames = 2,
                    LatentWidth = 16,
                    LatentHeight = 16,
                    AudioLatentFrames = 5,
                },
                new MiniMaxH3ReferenceBlock
                {
                    Kind = MiniMaxH3ReferenceKind.Audio,
                    VideoIndex = -1,
                    AudioIndex = 1,
                    AudioLatentFrames = 3,
                },
            };

            var layout = MiniMaxH3Layout.Build(6, shape, sigma: 1f, references: blocks);
            var kinds = layout.Segments.Select(s => s.Kind).ToArray();
            _output.WriteLine("[multi-ref] segments = " + string.Join(", ", kinds));

            Assert.Equal(new[]
            {
                MiniMaxH3SegmentKind.Text,
                MiniMaxH3SegmentKind.ConditionVideo,   // the still
                MiniMaxH3SegmentKind.ConditionAudio,   // the clip's soundtrack, FIRST
                MiniMaxH3SegmentKind.ConditionVideo,   // then the clip
                MiniMaxH3SegmentKind.ConditionAudio,   // the standalone soundtrack
                MiniMaxH3SegmentKind.TargetAudio,
                MiniMaxH3SegmentKind.TargetVideo,
            }, kinds);

            // Stereo: an audio reference of n latent frames is 2n tokens.
            var clipAudio = layout.Segments[2];
            Assert.Equal(5 * MiniMaxH3Geometry.AudioChannels, clipAudio.End - clipAudio.Start);
            var loneAudio = layout.Segments[4];
            Assert.Equal(3 * MiniMaxH3Geometry.AudioChannels, loneAudio.End - loneAudio.Start);
        }

        [Fact]
        public void ReferenceBlocksShareOneTimestepTableWhateverTheirCount()
        {
            // Every reference is noised to the same near-clean timestep, so adding
            // references must not grow the AdaLN table — the native side indexes it by
            // column and a fourth row would silently shift every column base.
            var shape = MiniMaxH3Geometry.Resolve(256, 256, 22);
            var one = MiniMaxH3Layout.Build(4, shape, 1f, references: new[] { ImageBlock(0, 256, 256) });
            var many = MiniMaxH3Layout.Build(4, shape, 1f, references: Enumerable.Range(0, 9)
                .Select(i => ImageBlock(i, 256, 256)).ToArray());

            Assert.Equal(one.Timesteps, many.Timesteps);
            var oneRows = one.ModulationSpans
                .Where(s => s.Start >= 4 && s.End <= 4 + 64).Select(s => s.Row).Distinct().ToArray();
            var manyRows = many.ModulationSpans
                .Where(s => s.Start >= 4 && s.End <= 4 + 9 * 64).Select(s => s.Row).Distinct().ToArray();
            Assert.Equal(oneRows, manyRows);
            _output.WriteLine($"[multi-ref] timesteps={string.Join(",", many.Timesteps)} " +
                              $"reference row={manyRows.Single()}");
        }

        [Fact]
        public void EveryTokenIsCoveredByExactlyOneModulationSpan()
        {
            // The native side rejects a layout whose spans do not tile the sequence, and
            // it rebuilds the hidden state by concatenating them in order — so a gap or
            // an overlap is a silent permutation of the conditioning, not a crash.
            var shape = MiniMaxH3Geometry.Resolve(320, 192, 39);
            var blocks = new List<MiniMaxH3ReferenceBlock>
            {
                ImageBlock(0, 288, 224),
                new MiniMaxH3ReferenceBlock
                {
                    Kind = MiniMaxH3ReferenceKind.VideoAudio, VideoIndex = 1, AudioIndex = 0,
                    LatentFrames = 2, LatentWidth = 14, LatentHeight = 18, AudioLatentFrames = 4,
                },
                ImageBlock(2, 224, 288),
                new MiniMaxH3ReferenceBlock
                {
                    Kind = MiniMaxH3ReferenceKind.Audio, VideoIndex = -1, AudioIndex = 1,
                    AudioLatentFrames = 6,
                },
            };

            var layout = MiniMaxH3Layout.Build(17, shape, 0.5f, references: blocks);

            var covered = new int[layout.TokenCount];
            foreach (var span in layout.ModulationSpans)
                for (int i = span.Start; i < span.End; i++) covered[i]++;
            Assert.All(covered, c => Assert.Equal(1, c));

            // Same for the segment list, which the native validator sums independently.
            var segCovered = new int[layout.TokenCount];
            foreach (var seg in layout.Segments)
                for (int i = seg.Start; i < seg.End; i++) segCovered[i]++;
            Assert.All(segCovered, c => Assert.Equal(1, c));

            Assert.Equal(layout.TokenCount * 3, layout.Positions.Length);
        }

        // ---- the prompt -----------------------------------------------------

        [Fact]
        public void EachImageGetsItsOwnVisualSpanInTheModalityTags()
        {
            // BuildTextModalities is what tells the DiT which prompt tokens are pictures.
            // With one image any off-by-one is invisible; with three, a span that ran long
            // would swallow the text between two pictures.
            var images = new[]
            {
                new MiniMaxH3PromptImage { TokenIndex = 5, Vision = Vision(4) },
                new MiniMaxH3PromptImage { TokenIndex = 20, Vision = Vision(6) },
                new MiniMaxH3PromptImage { TokenIndex = 40, Vision = Vision(2) },
            };

            var tags = MiniMaxH3Requests.BuildTextModalities(60, images);
            Assert.Equal(60, tags.Count);

            // The span covers the placeholders AND the <|vision_start|>/<|vision_end|>
            // markers either side, so it runs [index-1, index+count+1).
            foreach (var image in images)
            {
                int begin = image.TokenIndex - 1;
                int end = image.TokenIndex + image.Vision.TokenCount + 1;
                for (int i = begin; i < end; i++)
                    Assert.Equal(MiniMaxH3Modality.Visual, tags[i]);
                Assert.Equal(MiniMaxH3Modality.Text, tags[begin - 1]);
                Assert.Equal(MiniMaxH3Modality.Text, tags[end]);
            }

            int visual = tags.Count(t => t == MiniMaxH3Modality.Visual);
            Assert.Equal((4 + 2) + (6 + 2) + (2 + 2), visual);
            _output.WriteLine($"[multi-ref] {visual} visual tokens of {tags.Count}");
        }

        [Fact]
        public void AVisualSpanIsClampedToThePromptRatherThanRunningOffTheEnd()
        {
            var images = new[] { new MiniMaxH3PromptImage { TokenIndex = 0, Vision = Vision(9) } };
            var tags = MiniMaxH3Requests.BuildTextModalities(10, images);
            Assert.Equal(10, tags.Count);
            Assert.All(tags, t => Assert.Equal(MiniMaxH3Modality.Visual, t));
        }

        private static MiniMaxH3VisionOutput Vision(int tokens) =>
            new() { TokenCount = tokens, GridHeight = 2, GridWidth = tokens, Merged = Array.Empty<float>() };

        // ---- loading and fitting --------------------------------------------

        [Fact]
        public void ReferenceImagesKeepTheirOrderAndAreFittedIndividually()
        {
            using var dir = new TempDir();
            // Deliberately different shapes; the fit is per image, to the clip's AREA.
            string a = dir.WritePng("a.png", 640, 480);
            string b = dir.WritePng("b.png", 200, 400);
            string c = dir.WritePng("c.png", 96, 96);

            var loaded = MiniMaxH3Requests.LoadReferenceImages(
                new VideoGenerationParams { ReferenceImagePaths = new List<string> { a, b, c } },
                width: 256, height: 256);

            Assert.Equal(3, loaded.Count);
            // Order is the request's order: reference 1 is "<Picture 1>".
            Assert.True(loaded[0].Width > loaded[0].Height, "the landscape source stayed landscape");
            Assert.True(loaded[1].Height > loaded[1].Width, "the portrait source stayed portrait");
            // Only ever scaled DOWN: a source under the target area keeps its own size.
            Assert.Equal(96, loaded[2].Width);
            Assert.Equal(96, loaded[2].Height);

            foreach (var image in loaded)
            {
                Assert.Equal(0, image.Width % MiniMaxH3Geometry.SpatialMultiple);
                Assert.Equal(0, image.Height % MiniMaxH3Geometry.SpatialMultiple);
            }
            _output.WriteLine("[multi-ref] fitted " +
                string.Join(", ", loaded.Select(i => $"{i.Width}x{i.Height}")));
        }

        [Fact]
        public void ReferenceSnapRoundsHalfAwayFromZeroLikeTheReferenceImplementation()
        {
            // 80/32 = 2.5 and 144/32 = 4.5 both land exactly on a tie. std::round in
            // stable-diffusion.cpp goes away from zero (96 and 160); .NET's default goes
            // to even, which would give 64 and 128 — a whole latent row out on both axes.
            using var dir = new TempDir();
            string p = dir.WritePng("tie.png", 80, 144);

            var loaded = MiniMaxH3Requests.LoadReferenceImages(
                new VideoGenerationParams { ReferenceImagePaths = new List<string> { p } },
                width: 640, height: 384);   // far larger area, so scale is exactly 1

            Assert.Equal(96, loaded[0].Width);
            Assert.Equal(160, loaded[0].Height);
        }

        [Fact]
        public void ATenthReferenceImageIsRefusedByName()
        {
            using var dir = new TempDir();
            var paths = Enumerable.Range(0, MiniMaxH3Geometry.MaxReferences + 1)
                .Select(i => dir.WritePng($"r{i}.png", 64, 64)).ToList();

            var ok = MiniMaxH3Requests.LoadReferenceImages(
                new VideoGenerationParams
                {
                    ReferenceImagePaths = paths.Take(MiniMaxH3Geometry.MaxReferences).ToList()
                },
                width: 256, height: 256);
            Assert.Equal(MiniMaxH3Geometry.MaxReferences, ok.Count);

            var ex = Assert.Throws<ArgumentException>(() => MiniMaxH3Requests.LoadReferenceImages(
                new VideoGenerationParams { ReferenceImagePaths = paths },
                width: 256, height: 256));
            Assert.Contains(MiniMaxH3Geometry.MaxReferences.ToString(), ex.Message);
            _output.WriteLine("[multi-ref] " + ex.Message);
        }

        [Fact]
        public void APlainImageIsNotQuietlyAddedAsAnExtraReference()
        {
            // A stray --image next to named references would otherwise become an unnumbered
            // "<Picture 4>" that the user never asked for.
            using var dir = new TempDir();
            string named = dir.WritePng("named.png", 128, 128);
            string plain = dir.WritePng("plain.png", 128, 128);

            var loaded = MiniMaxH3Requests.LoadReferenceImages(
                new VideoGenerationParams
                {
                    ReferenceImagePaths = new List<string> { named },
                    ImagePath = plain,
                },
                width: 256, height: 256);

            Assert.Single(loaded);
        }

        // ---- mode resolution -------------------------------------------------

        [Fact]
        public void SeveralReferenceImagesResolveToReferenceModeOnTheRef2VaCheckpoint()
        {
            var p = new VideoGenerationParams
            {
                ReferenceImagePaths = new List<string> { "a.png", "b.png", "c.png" },
            };
            Assert.Equal(MiniMaxH3Mode.Reference,
                MiniMaxH3ModeResolver.Resolve(p, MiniMaxH3Partition.Reference));

            var ex = Assert.Throws<InvalidOperationException>(() =>
                MiniMaxH3ModeResolver.Resolve(p, MiniMaxH3Partition.FirstLastFrame));
            Assert.Contains("Ref2VA", ex.Message);
        }

        [Fact]
        public void KeyframesAndReferencesInOneRequestAreRefused()
        {
            // stable-diffusion.cpp: "Ref2VA cannot be combined with --init-img or --end-img
            // in one request." Silently dropping one of them returns a clip that looks like
            // the request was honoured.
            var p = new VideoGenerationParams
            {
                ReferenceImagePaths = new List<string> { "a.png", "b.png" },
                ImagePath = "start.png",
            };
            var ex = Assert.Throws<InvalidOperationException>(() =>
                MiniMaxH3ModeResolver.Resolve(p, MiniMaxH3Partition.Reference));
            Assert.Contains("keyframes and references", ex.Message);
            _output.WriteLine("[multi-ref] " + ex.Message);
        }

        [Fact]
        public void ALoneImageStillMeansOneReferenceOnTheRef2VaCheckpoint()
        {
            // The accommodation for clients that only know how to attach "an image" must
            // survive the rule above.
            var p = new VideoGenerationParams { ImagePath = "only.png" };
            Assert.Equal(MiniMaxH3Mode.Reference,
                MiniMaxH3ModeResolver.Resolve(p, MiniMaxH3Partition.Reference));
        }

        // ---- the checkpoint switch -------------------------------------------

        [Theory]
        [InlineData("minimax_h3_ref2va_pruned-Q4_K.gguf", MiniMaxH3Partition.Reference)]
        [InlineData("/models/MiniMax_H3_REF2VA_pruned-Q8_0.gguf", MiniMaxH3Partition.Reference)]
        [InlineData("minimax_h3_fl2va_pruned-Q4_K.gguf", MiniMaxH3Partition.FirstLastFrame)]
        [InlineData("some-other-video-model.gguf", MiniMaxH3Partition.FirstLastFrame)]
        public void ThePartitionIsReadOffTheFileName(string path, MiniMaxH3Partition expected)
        {
            // Everything about references hangs off this substring match, and a requantized
            // file that lost "ref2va" from its name would silently become a keyframe model.
            Assert.Equal(expected, MiniMaxH3Config.PartitionFromFileName(path));
        }

        // ---- what each extra reference costs ---------------------------------

        [Fact]
        public void ReferenceCostInTokensIsLinearAndSmallAgainstTheClip()
        {
            // No weights: the packed sequence length is pure arithmetic, and it is the
            // number that decides both the attention cost and how close a request gets to
            // the FP16 flash-attention ceiling. A 640x384 reference is 240 tokens; nine of
            // them are 2160, against 1680 for a 22-frame clip and 7680 for a 107-frame one.
            var shape = MiniMaxH3Geometry.Resolve(640, 384, 22);
            int baseline = MiniMaxH3Layout.Build(60, shape, 1f).TokenCount;
            int previous = baseline;
            foreach (int n in new[] { 1, 2, 4, 8, MiniMaxH3Geometry.MaxReferences })
            {
                var blocks = Enumerable.Range(0, n).Select(i => ImageBlock(i, 640, 384)).ToArray();
                int tokens = MiniMaxH3Layout.Build(60, shape, 1f, references: blocks).TokenCount;
                _output.WriteLine($"[ref-cost] refs={n,2} packedTokens={tokens,6} " +
                                  $"(+{tokens - baseline} over text-to-video, " +
                                  $"+{tokens - previous} for the last batch)");
                Assert.Equal(baseline + n * 240, tokens);
                previous = tokens;
            }
        }

        /// <summary>Wall-clock cost of a denoise step against the number of references,
        /// on the real Ref2VA weights. Opt in with <c>TS_H3_REF_BENCH=1</c>; needs
        /// <c>TS_MINIMAX_H3_DIR</c> to hold minimax_h3_ref2va_pruned-Q4_K.gguf.</summary>
        [ModelFact("TS_MINIMAX_H3_DIR")]
        public void BenchmarkStepTimeAgainstReferenceCount()
        {
            if (Environment.GetEnvironmentVariable("TS_H3_REF_BENCH") != "1")
            {
                _output.WriteLine("[ref-bench] set TS_H3_REF_BENCH=1 to run; skipping");
                return;
            }
            string dit = Path.Combine(Environment.GetEnvironmentVariable("TS_MINIMAX_H3_DIR") ?? ".",
                                      "minimax_h3_ref2va_pruned-Q4_K.gguf");
            if (!File.Exists(dit))
            {
                _output.WriteLine("[ref-bench] Ref2VA denoiser absent; skipping");
                return;
            }

            using var model = new MiniMaxH3Denoiser(dit, TestGates.PinnedGgmlBackend);
            _output.WriteLine("[ref-bench] " + model.Config);
            var shape = MiniMaxH3Geometry.Resolve(640, 384, 22);
            int videoCount = shape.VideoTokenCount, audioCount = shape.AudioTokenCount;
            const int textLength = 548;   // what two references plus a sentence actually cost

            var rng = new Random(7);
            var textHidden = new float[(long)textLength * model.Config.TextDim];
            for (long i = 0; i < textHidden.LongLength; i++) textHidden[i] = Gauss(rng);
            var video = new float[(long)videoCount * model.Config.VideoPatchDim];
            var audio = new float[(long)audioCount * model.Config.AudioLatentChannels];
            for (long i = 0; i < video.LongLength; i++) video[i] = Gauss(rng);
            for (long i = 0; i < audio.LongLength; i++) audio[i] = Gauss(rng);

            double first = 0;
            foreach (int n in new[] { 0, 1, 2, 4, 8 })
            {
                var blocks = Enumerable.Range(0, n).Select(i => ImageBlock(i, 640, 384)).ToArray();
                int perRef = (640 / 32) * (384 / 32);
                var cond = new float[(long)n * perRef * model.Config.VideoPatchDim];
                for (long i = 0; i < cond.LongLength; i++) cond[i] = Gauss(rng);

                var layout = MiniMaxH3Layout.Build(textLength, shape, 1f,
                    references: n > 0 ? blocks : null);

                // One untimed pass so weight upload and graph planning are not in the number.
                model.Forward(video, videoCount, audio, audioCount, textHidden, textLength,
                              layout, 1f, conditionTokens: n > 0 ? cond : null,
                              conditionCount: n * perRef);

                var sw = Stopwatch.StartNew();
                const int reps = 3;
                for (int r = 0; r < reps; r++)
                    model.Forward(video, videoCount, audio, audioCount, textHidden, textLength,
                                  layout, 1f, conditionTokens: n > 0 ? cond : null,
                                  conditionCount: n * perRef);
                sw.Stop();

                double ms = sw.Elapsed.TotalMilliseconds / reps;
                if (n == 0) first = ms;
                _output.WriteLine(
                    $"[ref-bench] refs={n,2} packedTokens={layout.TokenCount,6} " +
                    $"step={ms,8:F1} ms  ({ms / first:F2}x of text-to-video, " +
                    $"{(n == 0 ? 0 : (ms - first) / n):F1} ms per reference)");
            }
        }

        private static float Gauss(Random rng)
        {
            double u1 = 1.0 - rng.NextDouble(), u2 = rng.NextDouble();
            return (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
        }

        // ---- helpers ---------------------------------------------------------

        private static IEnumerable<float> TimesOf(MiniMaxH3PackedLayout layout,
                                                  MiniMaxH3SequenceSegment segment)
        {
            for (int token = segment.Start; token < segment.End; token++)
                yield return layout.Positions[token * 3];
        }

        private sealed class TempDir : IDisposable
        {
            public string Path { get; } = System.IO.Path.Combine(
                System.IO.Path.GetTempPath(), "h3refs_" + Guid.NewGuid().ToString("N"));

            public TempDir() => Directory.CreateDirectory(Path);

            public string WritePng(string name, int width, int height)
            {
                var pixels = new float[(long)width * height * 3];
                for (int y = 0; y < height; y++)
                    for (int x = 0; x < width; x++)
                    {
                        long i = ((long)y * width + x) * 3;
                        pixels[i] = x / (float)width;
                        pixels[i + 1] = y / (float)height;
                        pixels[i + 2] = 0.5f;
                    }
                string full = System.IO.Path.Combine(Path, name);
                ImageIO.SavePng(full, new RgbImage(width, height, pixels));
                return full;
            }

            public void Dispose()
            {
                try { Directory.Delete(Path, recursive: true); } catch { /* best effort */ }
            }
        }
    }
}
