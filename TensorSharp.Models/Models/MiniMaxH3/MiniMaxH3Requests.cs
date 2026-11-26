// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// MiniMax-H3's request rules: the conditioning mode's canvas and keyframes, Ref2VA references (pictures, clips and
// soundtracks) and how the language model is shown them, the sampling a request takes (the base schedule or a Turbo
// plug-in's recipe) and its refusals, and the finite checks that stop a diverged clip. The family on the diffusion runtime
// (MiniMaxH3Family) runs a clip with them.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using TensorSharp.GGML;
using TensorSharp.Models.QwenImage;
using TensorSharp.Models.Video;
using TensorSharp.Models.WanVideo;
using TensorSharp.Runtime;

namespace TensorSharp.Models.MiniMaxH3
{
    /// <summary>MiniMax-H3's request rules (see the file header).</summary>
    internal static class MiniMaxH3Requests
    {
        /// <summary>One resolved Ref2VA reference: the media it carries, how it is
        /// presented to the language model, and (once encoded) the shape of its
        /// latents.</summary>
        internal sealed class MiniMaxH3Reference
        {
            public MiniMaxH3ReferenceKind Kind { get; set; }
            /// <summary>Frames at the VAE's canvas — one for an image, a clip for a video.</summary>
            public List<RgbImage> Frames { get; init; }
            /// <summary>The subset shown to the vision tower: every frame for an image,
            /// every 12th (i.e. 2 fps) for a video.</summary>
            public List<RgbImage> Presented { get; init; }
            /// <summary>Seconds into the clip for each presented frame.</summary>
            public List<float> Timestamps { get; init; }
            /// <summary>Soundtrack as channel planes at the audio VAE's rate.</summary>
            public float[][] Audio { get; init; }
        }

        /// <summary>Pixel frames between presented frames: 24 fps / 12 = the 2 fps the
        /// language model sees a reference clip at.</summary>
        private const int PresentationStride = 12;

        /// <summary>Resolve every reference input into media the encoders can consume.</summary>
        internal static List<MiniMaxH3Reference> LoadReferences(
            VideoGenerationParams p, int width, int height, int maxFrames)
        {
            var refs = new List<MiniMaxH3Reference>();

            foreach (RgbImage image in LoadReferenceImages(p, width, height)
                                       ?? new List<RgbImage>())
            {
                refs.Add(new MiniMaxH3Reference
                {
                    Kind = MiniMaxH3ReferenceKind.Image,
                    Frames = new List<RgbImage> { image },
                    Presented = new List<RgbImage> { image },
                    Timestamps = new List<float> { 0f },
                });
            }

            var videos = p.ReferenceVideoPaths;
            var videoAudios = p.ReferenceVideoAudioPaths;
            for (int i = 0; videos != null && i < videos.Count; i++)
            {
                var clip = LoadReferenceVideo(videos[i], maxFrames);
                // A soundtrack pairs with its clip BY INDEX, which is why they are two
                // lists rather than one: a video file's own audio track is not readable
                // through the frame decoder.
                float[][] audio = null;
                if (videoAudios != null && i < videoAudios.Count &&
                    !string.IsNullOrWhiteSpace(videoAudios[i]))
                {
                    audio = LoadReferenceAudio(videoAudios[i], clip.Count, MiniMaxH3Geometry.Fps);
                }
                refs.Add(new MiniMaxH3Reference
                {
                    Kind = audio != null ? MiniMaxH3ReferenceKind.VideoAudio : MiniMaxH3ReferenceKind.Video,
                    Frames = clip,
                    Presented = Subsample(clip, PresentationStride),
                    Timestamps = Stamps(clip.Count, PresentationStride),
                    Audio = audio,
                });
            }

            var audios = p.ReferenceAudioPaths;
            for (int i = 0; audios != null && i < audios.Count; i++)
            {
                refs.Add(new MiniMaxH3Reference
                {
                    Kind = MiniMaxH3ReferenceKind.Audio,
                    Audio = LoadReferenceAudio(audios[i], maxFrames, MiniMaxH3Geometry.Fps),
                });
            }
            return refs;
        }

        private static List<RgbImage> Subsample(List<RgbImage> frames, int stride)
        {
            var outp = new List<RgbImage>();
            for (int i = 0; i < frames.Count; i += stride) outp.Add(frames[i]);
            return outp;
        }

        private static List<float> Stamps(int frameCount, int stride)
        {
            var outp = new List<float>();
            for (int i = 0; i < frameCount; i += stride)
                outp.Add(i / (float)MiniMaxH3Geometry.Fps);
            return outp;
        }

        /// <summary>Load a reference clip onto H3's own 24 fps timeline and canvas.
        ///
        /// <para>The canvas is derived from the aspect ratio around a nominal 768, then
        /// capped by area and snapped to 32 — the reference's own rule, which does NOT
        /// take the generated clip's size. The frame count is pulled down onto the
        /// 17k+5 grid because that is the only length the causal encoder maps cleanly
        /// onto latent frames.</para></summary>
        private static List<RgbImage> LoadReferenceVideo(string path, int maxFrames)
        {
            // Only the first `cap` frames can ever be used, and hold-and-drop picks frame i
            // from i alone, so asking for no more than that reads exactly the same frames.
            // Asking for all of them wrote every frame of the clip to disk as a PNG - 720 of
            // them for a 30-second clip - to use a few dozen.
            int cap = Math.Max(maxFrames, MiniMaxH3Geometry.MinFrames);
            var (files, sourceFps) = MediaHelper.ExtractFramesAtRate(
                path, MiniMaxH3Geometry.Fps, maxFrames: cap);
            // A clip FILE is extracted into a folder of its own, which is ours to remove
            // once the frames are in memory; a folder of frames was the caller's.
            string extracted = File.Exists(path) && files.Count > 0 ? Path.GetDirectoryName(files[0]) : null;
            if (extracted != null && !Path.GetFileName(extracted).StartsWith("refvid_", StringComparison.Ordinal))
                extracted = null;
            try
            {
                int frames = Math.Min(files.Count, cap);
                if (frames < MiniMaxH3Geometry.MinFrames)
                    throw new ArgumentException(
                        $"'{path}' gives {frames} frames at 24 fps; a reference clip needs at least " +
                        $"{MiniMaxH3Geometry.MinFrames}.", nameof(path));
                // Down onto the grid: 5, 22, 39, ... Anything between rungs cannot become a
                // whole number of latent frames.
                while (frames % 17 != 5) frames--;

                RgbImage first = QwenImage.ImageIO.Load(files[0]);
                var (w, h) = ReferenceVideoCanvas(first.Width, first.Height);
                var clip = new List<RgbImage>(frames);
                for (int i = 0; i < frames; i++)
                {
                    RgbImage src = i == 0 ? first : QwenImage.ImageIO.Load(files[i]);
                    clip.Add(src.Width == w && src.Height == h ? src : QwenImage.ImageIO.Resize(src, w, h));
                }
                Console.WriteLine($"  [h3] reference video: {files.Count} frames at 24 fps " +
                                  $"(source {sourceFps:F2} fps) -> {frames} frames at {w}x{h}");
                return clip;
            }
            finally
            {
                if (extracted != null)
                {
                    try { Directory.Delete(extracted, recursive: true); }
                    catch (IOException) { }
                    catch (UnauthorizedAccessException) { }
                }
            }
        }

        /// <summary>The reference-clip canvas: aspect ratio around a nominal 768, capped
        /// at 768x1344 of area, snapped to the VAE's 32-pixel grid.
        ///
        /// <para>A source SMALLER than that keeps its own size instead of being blown
        /// up to it. Upscaling invents no detail for the model to reference and the
        /// cost is steep — a 448x320 clip would otherwise become 1088x768, which is
        /// 5712 conditioning tokens against 1680 for the clip being generated.</para></summary>
        internal static (int Width, int Height) ReferenceVideoCanvas(int sourceWidth, int sourceHeight)
        {
            const double nominal = 768.0, maxArea = 768.0 * 1344.0;
            double ratio = sourceWidth / (double)sourceHeight;
            double w = ratio >= 1.0 ? nominal * ratio : nominal;
            double h = ratio >= 1.0 ? nominal : nominal / ratio;
            if (w * h > maxArea)
            {
                double scale = Math.Sqrt(maxArea / (w * h));
                w *= scale; h *= scale;
            }
            if ((double)sourceWidth * sourceHeight < w * h)
                return (SnapToVaeGrid(sourceWidth), SnapToVaeGrid(sourceHeight));
            return (SnapToVaeGrid(w), SnapToVaeGrid(h));
        }

        /// <summary>Decode a reference soundtrack to the audio VAE's stereo 32 kHz, and
        /// trim it to the duration the clip covers — a longer reference is truncated,
        /// not stretched.</summary>
        private static float[][] LoadReferenceAudio(string path, int frames, int fps)
        {
            var decoded = AudioIO.Decode(path, MiniMaxH3AudioVae.SampleRate,
                                         MiniMaxH3Geometry.AudioChannels);
            int wanted = (int)Math.Round(frames / (double)fps * MiniMaxH3AudioVae.SampleRate);
            // Whole latent frames only: the encoder's strided convolutions land on a
            // frame boundary or not at all.
            wanted = Math.Max(MiniMaxH3AudioVae.TotalUpsample,
                              wanted / MiniMaxH3AudioVae.TotalUpsample * MiniMaxH3AudioVae.TotalUpsample);
            var planes = new float[decoded.ChannelCount][];
            for (int c = 0; c < planes.Length; c++)
            {
                planes[c] = new float[wanted];
                Array.Copy(decoded.Channels[c], planes[c],
                           Math.Min(wanted, decoded.Channels[c].Length));
            }
            return planes;
        }

        /// <summary>Load the Ref2VA reference images and fit each to the grid the video
        /// VAE needs.
        ///
        /// <para>References are only ever scaled DOWN, to the generated clip's area,
        /// and keep their own aspect ratio: a reference exists to carry identity and
        /// appearance, so distorting it to the output canvas would work against the one
        /// thing it is there for.</para></summary>
        internal static List<RgbImage> LoadReferenceImages(VideoGenerationParams p, int width, int height)
        {
            double targetArea = (double)width * height;
            RgbImage Fit(RgbImage image)
            {
                double scale = Math.Min(1.0, Math.Sqrt(targetArea / ((double)image.Width * image.Height)));
                int w = SnapToVaeGrid(image.Width * scale);
                int h = SnapToVaeGrid(image.Height * scale);
                return w == image.Width && h == image.Height ? image : QwenImage.ImageIO.Resize(image, w, h);
            }

            var paths = p.ReferenceImagePaths;
            if (paths is not { Count: > 0 })
            {
                // A client that only knows how to attach "an image" still gets a
                // reference when the Ref2VA checkpoint is loaded; the mode resolver
                // has already decided that is what the image means. A request that
                // named clips or soundtracks explicitly is not that client, so a
                // stray --image does not silently become an extra reference.
                bool namedOthers = (p.ReferenceVideoPaths?.Count ?? 0) > 0
                                   || (p.ReferenceAudioPaths?.Count ?? 0) > 0;
                RgbImage plain = namedOthers ? null : p.ResolveImage();
                return plain == null ? null : new List<RgbImage> { Fit(plain) };
            }

            // The published Ref2VA takes at most nine references; more is a mistake
            // worth naming rather than a sequence quietly ballooning past what the
            // model was trained on.
            if (paths.Count > MiniMaxH3Geometry.MaxReferences)
                throw new ArgumentException(
                    $"MiniMax-H3 Ref2VA accepts at most {MiniMaxH3Geometry.MaxReferences} " +
                    $"reference images; {paths.Count} were supplied.", nameof(p));

            var images = new List<RgbImage>(paths.Count);
            foreach (string path in paths) images.Add(Fit(QwenImage.ImageIO.Load(path)));
            return images;
        }

        /// <summary>Snap a reference dimension onto the VAE's 32-pixel grid, breaking ties
        /// the way the reference implementation does.
        ///
        /// <para>The tie-break is asked for explicitly because the two languages disagree:
        /// the reference computes <c>(int)std::round(size * scale / 32.f) * 32</c>, and
        /// <c>std::round</c> goes half AWAY FROM ZERO, while .NET's <see cref="Math.Round(double)"/>
        /// goes half to EVEN. A reference whose scaled height lands on 2.5 grid units
        /// becomes 96 px there and 64 px here — a whole latent row's difference in the
        /// conditioning, from a default nobody chose. Same reasoning as
        /// <see cref="FormatSeconds"/>, opposite direction.</para></summary>
        private static int SnapToVaeGrid(double value) =>
            Math.Max(MiniMaxH3Geometry.SpatialMultiple,
                     (int)Math.Round(value / MiniMaxH3Geometry.SpatialMultiple,
                                     MidpointRounding.AwayFromZero)
                        * MiniMaxH3Geometry.SpatialMultiple);

        /// <summary>Present the references to the language model and run the vision
        /// tower over each.
        ///
        /// <para>An image becomes a numbered picture; a clip becomes a numbered video
        /// whose frames are shown two at a time, each pair stamped with the time it
        /// sits at; a soundtrack contributes only its label, because the language model
        /// never hears it — the audio reaches the denoiser as conditioning latents
        /// instead. Every visual block is followed by one placeholder token per merged
        /// patch, and those placeholders are what the tower's output later replaces.</para></summary>
        internal static (string Prompt, List<MiniMaxH3PromptImage> Images) BuildReferencePrompt(
            MiniMaxH3TextEncoder te, IReadOnlyList<MiniMaxH3Reference> references, string text)
        {
            const string visionStart = "<|vision_start|>";
            const string visionEnd = "<|vision_end|>";
            const string placeholder = "<|image_pad|>";
            RequireVisionTokens(te.Tokenizer, references);

            var builder = new System.Text.StringBuilder();
            var images = new List<MiniMaxH3PromptImage>();

            void AddBlock(RgbImage first, RgbImage second)
            {
                MiniMaxH3VisionOutput vision = te.Vision.Encode(first, second);
                builder.Append(visionStart);
                images.Add(new MiniMaxH3PromptImage
                {
                    TokenIndex = te.Tokenize(builder.ToString()).Count,
                    Vision = vision,
                });
                for (int t = 0; t < vision.TokenCount; t++) builder.Append(placeholder);
                builder.Append(visionEnd);
            }

            int pictures = 0, videos = 0, sounds = 0;
            foreach (var reference in references)
            {
                switch (reference.Kind)
                {
                    case MiniMaxH3ReferenceKind.Audio:
                        builder.Append("<Audio ").Append(++sounds).Append(">: ");
                        break;

                    case MiniMaxH3ReferenceKind.Image:
                        builder.Append("<Picture ").Append(++pictures).Append(">: ");
                        AddBlock(reference.Presented[0], null);
                        break;

                    default:
                        // A clip that came with its own soundtrack is presented as TWO
                        // things, not one: the reference pushes an AUDIO item and then a
                        // VIDEO item, so the prompt reads "<Audio 1>: <Video 1>: ...".
                        // Emitting only the video label also stopped the audio counter,
                        // which then renumbered every later standalone soundtrack — a
                        // divergence that cannot appear with a single reference and grows
                        // with each extra one.
                        if (reference.Kind == MiniMaxH3ReferenceKind.VideoAudio)
                            builder.Append("<Audio ").Append(++sounds).Append(">: ");
                        builder.Append("<Video ").Append(++videos).Append(">: ");
                        var shown = reference.Presented;
                        for (int i = 0; i < shown.Count; i += 2)
                        {
                            int next = Math.Min(i + 1, shown.Count - 1);
                            float t0 = reference.Timestamps[i];
                            float t1 = reference.Timestamps[next];
                            builder.Append('<').Append(FormatSeconds((t0 + t1) * 0.5f))
                                   .Append(" seconds>");
                            AddBlock(shown[i], i == next ? null : shown[next]);
                        }
                        break;
                }
            }
            builder.Append(text ?? string.Empty);

            string prompt = builder.ToString();
            int length = te.Tokenize(prompt).Count;
            foreach (var image in images)
            {
                if (image.TokenIndex + image.Vision.TokenCount > length)
                    throw new InvalidOperationException(
                        "the reference placeholders did not tokenize one-to-one; " +
                        "the tokenizer is missing the <|image_pad|> special token.");
            }
            return (prompt, images);
        }

        /// <summary>Refuse to present a picture to a tokenizer that has no vision tokens.
        ///
        /// <para>They are not in vocab.json: only tokenizer_config.json's added tokens define
        /// them. Without that file each marker is split into ordinary text pieces, the prompt
        /// comes out LONGER than the placeholders it was built from, and the one-to-one check
        /// after tokenizing (which catches a prompt that is shorter) passes - so the vision
        /// features were written over the wrong positions and the clip quietly ignored the
        /// picture. Checked before the vision tower runs, so the failure costs nothing.</para></summary>
        internal static void RequireVisionTokens(BpeTokenizer tokenizer)
        {
            foreach (string marker in new[] { "<|vision_start|>", "<|image_pad|>", "<|vision_end|>" })
            {
                if (tokenizer.Encode(marker, addSpecial: false).Count != 1)
                    throw new InvalidOperationException(
                        $"the MiniMax-H3 tokenizer has no {marker} token, so a picture cannot be placed in " +
                        "the prompt. It is defined by tokenizer_config.json (MiniMaxAI/MiniMax-H3, " +
                        "processor/), which has to sit beside vocab.json and merges.txt.");
            }
        }

        /// <summary>The same refusal, for references that show the language model a
        /// picture: a soundtrack on its own reaches the prompt only as its label, so
        /// references that are all sounds need no vision markers.</summary>
        internal static void RequireVisionTokens(BpeTokenizer tokenizer, IReadOnlyList<MiniMaxH3Reference> references)
        {
            if (references.Any(reference => reference.Kind != MiniMaxH3ReferenceKind.Audio))
                RequireVisionTokens(tokenizer);
        }

        /// <summary>Audio references are presented as text labels and encoded by the
        /// audio VAE, so they do not require a vision tower in the text checkpoint.</summary>
        internal static void RequireReferenceVision(bool hasVision, IReadOnlyList<MiniMaxH3Reference> references)
        {
            if (!hasVision && references.Any(reference => reference.Kind != MiniMaxH3ReferenceKind.Audio))
                throw new InvalidOperationException(
                    "visual reference conditioning needs the Qwen3-VL vision tower, and this " +
                    "text-encoder checkpoint does not carry one. Point --video-text-encoder " +
                    "at the full qwen3vl_32b_minimax_h3 GGUF.");
        }

        /// <summary>Format a timestamp exactly as the reference's C++ stream does.
        ///
        /// <para>Presented frames sit 12 apart at 24 fps, so a pair's midpoint is always
        /// an exact quarter-second — every value lands precisely on a rounding tie.
        /// C++ streams break ties to EVEN, and .NET's default breaks them away from
        /// zero, so "0.25" would become "0.3" here and "0.2" there. One character of
        /// prompt is one different token, so the tie-break is asked for explicitly.</para></summary>
        internal static string FormatSeconds(float seconds) =>
            Math.Round(seconds, 1, MidpointRounding.ToEven)
                .ToString("F1", System.Globalization.CultureInfo.InvariantCulture);

        /// <summary>Fit the request's keyframes to the generation canvas, in timeline
        /// order (start then end).
        ///
        /// <para>Fitted ONCE, here: the same pixels go to the VAE and to the vision
        /// tower, which is what the reference presents. Handing the tower the original
        /// photo instead gives it a far finer patch grid — a 1024x768 source is 768
        /// merged tokens against 108 for a 384x288 canvas — and that is a different
        /// conditioning signal, not a better one.</para></summary>
        internal static void ResolveKeyframes(VideoGenerationParams p, MiniMaxH3Shape shape,
                                              List<RgbImage> frames, List<bool> atEnd) =>
            ResolveKeyframes(p.ResolveImage(), p.ResolveEndImage(), shape, frames, atEnd, unsetCanvas: null);

        /// <summary>Resolve the canvas and, in a keyframe mode, fit the keyframes to it.
        ///
        /// <para>Each keyframe is decoded once, here: the canvas comes from the first of them,
        /// and the canvas-sized copies then go to the VAE and the vision tower. The full-size
        /// decodes live only in this frame (a 24 MP phone photo is ~290 MB of float RGB), so
        /// they are unreachable before the 17 GB text encoder loads; held in Generate's frame
        /// they would stay reachable through the whole run, under Tier-0 and Mono alike.</para></summary>
        /// <returns>The canvas before grid alignment (references are sized from it, as before)
        /// and the resolved shape.</returns>
        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
        internal static (int Width, int Height, MiniMaxH3Shape Shape) ResolveCanvasAndKeyframes(
            VideoGenerationParams p, MiniMaxH3Mode mode, List<RgbImage> frames, List<bool> atEnd)
        {
            bool keyframeMode = mode is MiniMaxH3Mode.ImageToVideo or MiniMaxH3Mode.FirstLastFrame;
            RgbImage first = keyframeMode ? p.ResolveImage() : null;
            RgbImage last = keyframeMode ? p.ResolveEndImage() : null;
            var (width, height) = ResolveCanvas(p, mode, first, last);
            var shape = MiniMaxH3Geometry.Resolve(
                width, height,
                p.Frames > 0 ? p.Frames : 22,
                p.Fps > 0 ? p.Fps : MiniMaxH3Geometry.Fps);
            if (!keyframeMode) return (width, height, shape);

            RgbImage anchor = first ?? last;
            bool requested = p.Width > 0 || p.Height > 0;
            Console.WriteLine($"  [h3] canvas {shape.Width}x{shape.Height} " +
                              (requested ? "(requested size)"
                               : $"from the {(first != null ? "first" : "last")} keyframe " +
                                 (p.CanvasPolicy == VideoCanvasPolicy.ModelRecipe ? "(MiniMax 768p recipe)" : "(compact preview)")));
            // What an unset size would have given, so a warning can say whether it would help.
            (int Width, int Height)? unset = null;
            if (requested && anchor != null)
            {
                var canvas = ResolveCanvas(new VideoGenerationParams { CanvasPolicy = p.CanvasPolicy }, mode, first, last);
                unset = (MiniMaxH3Geometry.AlignSpatial(canvas.Width), MiniMaxH3Geometry.AlignSpatial(canvas.Height));
            }
            // Fitted to the generation canvas ONCE, here: the same pixels go to the VAE and
            // to the vision tower, which is what the reference presents. The original photo
            // would give the tower a far finer patch grid — a 1024x768 source is 768 merged
            // tokens against 108 for a 384x288 canvas — and that is a different conditioning
            // signal, not a better one.
            ResolveKeyframes(first, last, shape, frames, atEnd, unset);
            return (width, height, shape);
        }

        /// <summary>The same fit, over keyframes the caller has already decoded.
        ///
        /// <para>Both keyframes are centre-cropped to the canvas. On the recipe canvas the
        /// first keyframe sets the aspect, so its crop is under 2%; sglang stretches that one
        /// instead, which is the same picture to within the rounding. The last keyframe is
        /// cropped exactly as sglang crops its follower keyframe.</para></summary>
        /// <param name="unsetCanvas">The canvas an unset size would have given, when the size was
        /// requested; null when the canvas already is that one.</param>
        internal static void ResolveKeyframes(RgbImage first, RgbImage last, MiniMaxH3Shape shape,
                                              List<RgbImage> frames, List<bool> atEnd,
                                              (int Width, int Height)? unsetCanvas)
        {
            if (first != null)
            {
                ReportKeyframeFit("first", first, shape, anchor: true, unsetCanvas);
                frames.Add(QwenImage.ImageIO.ResizeCover(first, shape.Width, shape.Height));
                atEnd.Add(false);
            }
            if (last != null)
            {
                ReportKeyframeFit("last", last, shape, anchor: first == null, unsetCanvas);
                frames.Add(QwenImage.ImageIO.ResizeCover(last, shape.Width, shape.Height));
                atEnd.Add(true);
            }
        }

        private static void ReportKeyframeFit(string which, RgbImage img, MiniMaxH3Shape shape,
                                              bool anchor, (int Width, int Height)? unsetCanvas)
        {
            foreach (string line in DescribeKeyframeFit(which, img.Width, img.Height,
                                                        shape.Width, shape.Height, anchor, unsetCanvas))
                Console.WriteLine("  [h3] " + line);
        }

        /// <summary>What fitting a keyframe to the canvas costs it, in words, or nothing
        /// when it costs nothing worth saying.
        ///
        /// <para>Two losses, and neither should be a surprise. A crop drops the edges of the
        /// shot: centre-cropping is the right trade against stretching, since a squeezed face
        /// is baked into every frame, but a portrait photo on a landscape canvas keeps only
        /// a band of it. A downscale drops detail: these modes keep the keyframe's framing,
        /// and the video VAE downsamples 16x in each direction, so a canvas smaller than the
        /// keyframe's own recipe leaves the faces of a wide shot a few latent cells across.
        /// An encode then decode alone smears them, which no denoiser can undo.</para></summary>
        /// <param name="anchor">This keyframe is the one an unset size takes the canvas from.</param>
        /// <param name="unsetCanvas">The canvas an unset size would give, when a size was
        /// requested; null otherwise.</param>
        internal static List<string> DescribeKeyframeFit(string which, int sourceWidth, int sourceHeight,
                                                         int canvasWidth, int canvasHeight, bool anchor,
                                                         (int Width, int Height)? unsetCanvas)
        {
            var lines = new List<string>();
            if (sourceWidth <= 0 || sourceHeight <= 0) return lines;
            bool unsetDiffers = unsetCanvas is { } u && (u.Width != canvasWidth || u.Height != canvasHeight);
            double want = canvasWidth / (double)canvasHeight;
            double have = sourceWidth / (double)sourceHeight;
            if (Math.Abs(want - have) / have > 0.02)
            {
                // The same arithmetic ImageIO.ResizeCover crops by.
                bool cropsWidth = have > want;
                double kept = cropsWidth ? want / have : have / want;
                lines.Add(
                    $"the {which} keyframe is {sourceWidth}x{sourceHeight} ({have:F2}:1) but the canvas is " +
                    $"{canvasWidth}x{canvasHeight} ({want:F2}:1); centre-cropping it keeps the middle " +
                    $"{kept * 100:F0}% of its {(cropsWidth ? "width" : "height")}." +
                    (anchor && unsetDiffers
                        ? $" Leave the size unset to take the canvas from this keyframe ({unsetCanvas.Value.Width}x{unsetCanvas.Value.Height})."
                        : ""));
            }

            // A cover fit scales by the LARGER ratio and crops the other axis. The keyframe's own
            // recipe canvas is the yardstick: scaling it further down than that loses detail the
            // recipe keeps. Comparing linear scale, not area, keeps a last keyframe that is merely
            // cropped onto the first keyframe's recipe canvas from being reported as starved.
            var recipe = MiniMaxH3Geometry.KeyframeCanvas(sourceWidth, sourceHeight);
            double fit = Math.Max(canvasWidth / (double)sourceWidth, canvasHeight / (double)sourceHeight);
            double recipeFit = Math.Max(recipe.Width / (double)sourceWidth, recipe.Height / (double)sourceHeight);
            if (fit < 1.0 && fit < 0.9 * recipeFit)
                lines.Add(
                    $"the {which} keyframe is scaled down {1.0 / fit:F1}x into {canvasWidth}x{canvasHeight}, " +
                    $"{recipeFit / fit:F1}x further than MiniMax-H3's 768p recipe would ({recipe.Width}x{recipe.Height}); " +
                    "faces and hands that are small in it come out blurred and distorted at this size" +
                    (unsetDiffers ? $". Leave the size unset for {unsetCanvas.Value.Width}x{unsetCanvas.Value.Height}." : "."));
            return lines;
        }

        /// <summary>Mark which prompt tokens are modulated on the visual stream.
        ///
        /// <para>The span covers the placeholders <i>and</i> the surrounding
        /// <c>&lt;|vision_start|&gt;</c> / <c>&lt;|vision_end|&gt;</c> markers, so the
        /// AdaLN switch happens one token early and ends one token late.</para></summary>
        internal static List<int> BuildTextModalities(
            int textLength, IReadOnlyList<MiniMaxH3PromptImage> images)
        {
            var modalities = new List<int>(textLength);
            for (int i = 0; i < textLength; i++) modalities.Add(MiniMaxH3Modality.Text);
            foreach (var image in images)
            {
                int begin = Math.Max(0, image.TokenIndex - 1);
                int end = Math.Min(textLength, image.TokenIndex + image.Vision.TokenCount + 1);
                for (int i = begin; i < end; i++) modalities[i] = MiniMaxH3Modality.Visual;
            }
            return modalities;
        }

        internal static float[] Concat(List<float[]> parts)
        {
            long total = 0;
            foreach (var part in parts) total += part.LongLength;
            var all = new float[total];
            long off = 0;
            foreach (var part in parts) { Array.Copy(part, 0, all, off, part.Length); off += part.Length; }
            return all;
        }

        /// <summary>Check that the conditioning chunk list describes the same runs, in the
        /// same order, that the layout put in the sequence.
        ///
        /// <para>These are built by two independent loops — this file fills
        /// <c>condChunks</c> while encoding, and <see cref="MiniMaxH3Layout.Build"/> emits
        /// segments — and the native side reassembles the conditioning rows POSITIONALLY
        /// from the chunk list, discarding the segment's source index. So they agree by
        /// construction rather than by contract, and if they ever stopped agreeing the
        /// symptom would be reference 2's tokens sitting where reference 1's belong: a
        /// plausible clip conditioned on the wrong pictures, with nothing reported. That
        /// cannot happen with a single reference, which is exactly why it is worth
        /// asserting now that several are the point.</para>
        ///
        /// <para>Runs once per request, over a handful of chunks.</para></summary>
        internal static void RequireChunksMatchLayout(
            IReadOnlyList<(int Kind, int Count)> chunks, MiniMaxH3PackedLayout layout)
        {
            if (chunks is not { Count: > 0 }) return;

            var expected = new List<(int Kind, int Count)>();
            foreach (var segment in layout.Segments)
            {
                if (segment.Kind == MiniMaxH3SegmentKind.ConditionVideo)
                    expected.Add((0, segment.Length));
                else if (segment.Kind == MiniMaxH3SegmentKind.ConditionAudio)
                    expected.Add((1, segment.Length));
            }

            string Describe(IEnumerable<(int Kind, int Count)> runs) =>
                string.Join(", ", runs.Select(r => $"{(r.Kind == 0 ? "video" : "audio")}x{r.Count}"));

            var actual = chunks.ToList();
            if (actual.Count != expected.Count || !actual.SequenceEqual(expected))
                throw new InvalidOperationException(
                    "MiniMax-H3 conditioning is inconsistent: the encoder produced [" +
                    Describe(actual) + "] but the packed layout expects [" + Describe(expected) +
                    "]. The denoiser rebuilds these rows by position, so continuing would " +
                    "condition the clip on the wrong references rather than fail.");
        }

        /// <summary>Stop a diverged sample rather than writing it out as a black clip.
        ///
        /// <para>This failure mode is silent by construction: a NaN or -Inf video latent
        /// decodes to a pixel the RGB clamp pins at 0 and an audio latent the WAV writer
        /// clamps to -1, so the user gets a plausible-looking file — right length, right
        /// frame rate, right soundtrack duration — that is uniformly black with a
        /// saturated buzz, and nothing anywhere reported an error. A flow-matching
        /// velocity is O(1) by construction, so a non-finite one is a numeric failure
        /// upstream and there is nothing left to salvage from the run.</para></summary>
        internal static void RequireFinite(float[] values, string what, int step, int steps, int tokens)
        {
            for (long i = 0; i < values.LongLength; i++)
            {
                if (float.IsFinite(values[i])) continue;
                throw new InvalidOperationException(
                    $"MiniMax-H3 denoising diverged: the {what} is " +
                    $"{(float.IsNaN(values[i]) ? "NaN" : "infinite")} at step {step + 1}/{steps} " +
                    $"(element {i} of {values.LongLength}, {tokens} packed tokens). Every frame " +
                    "would decode to black and the soundtrack to a saturated buzz, so the run is " +
                    "stopped instead. Re-run with a shorter clip to confirm the length is what " +
                    "triggers it.");
            }
        }

        /// <summary>Check decoder values before RGB or PCM conversion can hide a
        /// numeric failure by saturating infinities or replacing NaNs.</summary>
        internal static void RequireFiniteOutput(float[] values, string stage)
        {
            for (long i = 0; i < values.LongLength; i++)
                if (!float.IsFinite(values[i]))
                    throw new InvalidOperationException(
                        $"MiniMax-H3 {stage} contains a non-finite value at element {i} " +
                        $"of {values.LongLength}. Generation stopped before media conversion.");
        }

        /// <summary>Reject unsupported settings before loading the text encoder.
        /// Zero retains the shared request type's model-default convention.</summary>
        internal static void ValidateSamplingParameters(VideoGenerationParams p)
        {
            if (!float.IsFinite(p.CfgScale) || (p.CfgScale != 0f && p.CfgScale != 1f))
                throw new ArgumentException(
                    "MiniMax-H3 is CFG-distilled and requires guidance 1.0. " +
                    "Use --cfg 1.0 (the default).", nameof(p));
            if (!float.IsFinite(p.FlowShift) || p.FlowShift < 0f)
                throw new ArgumentException("MiniMax-H3 flow shift must be finite and positive, or 0 for the default.", nameof(p));
            if (p.Steps < 0)
                throw new ArgumentException("MiniMax-H3 steps must be positive, or 0 for the default.", nameof(p));
        }

        /// <summary>The same check with a LoRA plug-in loaded: the request is held to the
        /// plug-in's recipe (<see cref="MiniMaxH3SamplingRecipe.Validate"/>), so steps, flow shift
        /// and sampler may be unset or the recipe's own, and anything else is refused, never
        /// overridden. A null recipe is the base check.</summary>
        internal static void ValidateSamplingParameters(VideoGenerationParams p, MiniMaxH3SamplingRecipe recipe)
        {
            if (recipe != null) recipe.Validate(p);
            else ValidateSamplingParameters(p);
        }

        /// <summary>The steps, video shift and sigma grid a clip samples with. A plug-in's recipe
        /// fixes all three: its N Euler steps on the endpoint-inclusive linspace grid at its own
        /// shift (<see cref="MiniMaxH3Scheduler.BuildUniformSigmas"/>). Without one, the request's
        /// steps and shift, else 20 at 12, on stable-diffusion.cpp's grid
        /// (<see cref="MiniMaxH3Scheduler.BuildSigmas"/>).</summary>
        internal static (int Steps, float VideoShift, float[] Sigmas) ResolveSampling(
            VideoGenerationParams p, MiniMaxH3SamplingRecipe recipe)
        {
            if (recipe != null) return (recipe.Steps, recipe.VideoShift, recipe.VideoSigmas());
            int steps = p.Steps > 0 ? p.Steps : MiniMaxH3Scheduler.DefaultSteps;
            float videoShift = p.FlowShift > 0 ? p.FlowShift : MiniMaxH3Scheduler.VideoShift;
            return (steps, videoShift, MiniMaxH3Scheduler.BuildSigmas(steps, videoShift));
        }

        /// <summary>Refuse a reference clip with a plug-in loaded. The Turbo adapters were
        /// distilled on the FL2VA checkpoint's text, first-frame and first/last-frame clips;
        /// reference conditioning belongs to Ref2VA, which takes no plug-in.</summary>
        internal static void RequirePluginMode(MiniMaxH3SamplingRecipe recipe, MiniMaxH3Mode mode)
        {
            if (recipe != null && mode == MiniMaxH3Mode.Reference)
                throw new InvalidOperationException(
                    $"{recipe.Name} ({recipe.WeightsFileName}) is a MiniMax-H3 FL2VA plug-in for text, first-frame and " +
                    "first/last-frame clips; reference conditioning does not take it. Drop the plug-in for a reference clip.");
        }

        /// <summary>The Turbo adapters were trained at 768p; a clip whose short edge is another
        /// size says so once, and still runs. Null when there is nothing to say.</summary>
        internal static string DescribePluginCanvas(MiniMaxH3SamplingRecipe recipe, MiniMaxH3Shape shape)
        {
            const int TrainedShortEdge = 768;
            int shortEdge = Math.Min(shape.Width, shape.Height);
            if (recipe == null || shortEdge == TrainedShortEdge) return null;
            return $"  [lora] {recipe.Name} was trained at 768p; this clip is {shape.Width}x{shape.Height} " +
                   $"(short edge {shortEdge}).";
        }

        /// <summary>Area of the compact canvas: 640x384, TensorSharp's own validation
        /// workload and the size text-to-video and references get when no size is given.
        /// It is not MiniMax's working point (the released recipe uses a 768-pixel short
        /// edge), and for a keyframe it is too small: a full-body photo fitted to it leaves
        /// a face about 60 px tall, which the video VAE cannot reproduce.</summary>
        private const long PreviewArea = 640L * 384L;

        /// <summary>Resolve the output canvas.
        ///
        /// <para>An explicit width and height always win. One given side takes the other
        /// from the image's aspect. With neither, i2v and fl2v size the canvas from the
        /// first keyframe (the last one when it is the only one), because those modes keep
        /// its framing: under <see cref="VideoCanvasPolicy.ModelRecipe"/> by MiniMax's 768p
        /// recipe (<see cref="MiniMaxH3Geometry.KeyframeCanvas"/>), under
        /// <see cref="VideoCanvasPolicy.CompactPreview"/> at 640x384's area. Text-to-video
        /// gets 640x384.</para></summary>
        /// <param name="firstKeyframe">The decoded first keyframe, when the caller has one;
        /// otherwise the request's images are decoded here.</param>
        internal static (int Width, int Height) ResolveCanvas(
            VideoGenerationParams p, MiniMaxH3Mode mode,
            RgbImage firstKeyframe = null, RgbImage lastKeyframe = null)
        {
            if (p.Width > 0 && p.Height > 0) return (p.Width, p.Height);

            bool keyframeMode = mode is MiniMaxH3Mode.ImageToVideo or MiniMaxH3Mode.FirstLastFrame;
            RgbImage reference = keyframeMode && (firstKeyframe ?? lastKeyframe) != null
                ? firstKeyframe ?? lastKeyframe
                : p.ResolveImage() ?? p.ResolveEndImage();
            double aspect = 0;
            if (reference is { Width: > 0, Height: > 0 })
                aspect = reference.Width / (double)reference.Height;

            if (keyframeMode && aspect > 0 && p.Width <= 0 && p.Height <= 0
                && p.CanvasPolicy == VideoCanvasPolicy.ModelRecipe)
                return MiniMaxH3Geometry.KeyframeCanvas(reference.Width, reference.Height);

            if (p.Width > 0 && aspect > 0) return (p.Width, (int)Math.Round(p.Width / aspect));
            if (p.Height > 0 && aspect > 0) return ((int)Math.Round(p.Height * aspect), p.Height);
            if (p.Width > 0) return (p.Width, p.Width);
            if (p.Height > 0) return (p.Height, p.Height);

            if (aspect <= 0) return (640, 384);
            int w = (int)Math.Round(Math.Sqrt(PreviewArea * aspect));
            int h = (int)Math.Round(Math.Sqrt(PreviewArea / aspect));
            return (Math.Max(w, 32), Math.Max(h, 32));
        }
    }
}
