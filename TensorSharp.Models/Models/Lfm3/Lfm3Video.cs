// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Numerics;

using TensorSharp.Models.Media;

namespace TensorSharp.Models
{
    /// <summary>One temporal patch of a clip: the sampled frames it merges (indices into
    /// <see cref="Lfm3VideoPlan.FrameIndices"/>; <see cref="Second"/> equals <see cref="First"/> for an odd clip's last
    /// frame, which fills both slices) and the time it is labelled with.</summary>
    public readonly record struct Lfm3VideoGroup(int First, int Second, double Seconds)
    {
        /// <summary>The group's label, <c>"&lt;t.t seconds&gt;"</c>.</summary>
        public string Label => Lfm3Video.FormatTimestamp(Seconds);
    }

    /// <summary>How a clip is read: the source frames sampled, the size every frame is resized to, and the frame pairs
    /// with their time labels (<see cref="Lfm3Video.Plan(int, double, int, int, int)"/>).</summary>
    public sealed class Lfm3VideoPlan
    {
        internal Lfm3VideoPlan(int sourceFrames, double sourceFps, int sourceWidth, int sourceHeight, int[] frameIndices,
            int width, int height, IReadOnlyList<Lfm3VideoGroup> groups)
        {
            SourceFrames = sourceFrames;
            SourceFps = sourceFps;
            SourceWidth = sourceWidth;
            SourceHeight = sourceHeight;
            FrameIndices = frameIndices;
            Width = width;
            Height = height;
            Groups = groups;
        }

        public int SourceFrames { get; }
        public double SourceFps { get; }
        public int SourceWidth { get; }
        public int SourceHeight { get; }

        /// <summary>The source frames read, in order (a repeated index reads a frame twice).</summary>
        public int[] FrameIndices { get; }

        /// <summary>The size every sampled frame is resized to (multiples of 32).</summary>
        public int Width { get; }

        public int Height { get; }

        public IReadOnlyList<Lfm3VideoGroup> Groups { get; }

        /// <summary>Embedding rows of one group.</summary>
        public int RowsPerGroup => Lfm3ImageProcessor.TokenCount(Width, Height);

        /// <summary>Embedding rows of the whole clip.</summary>
        public int RowCount => RowsPerGroup * Groups.Count;

        /// <summary>The clip's prompt span (outer markers, per group a label, markers and rows).</summary>
        public Lfm3VisionLayout Layout() => Lfm3VisionLayout.Video(Groups, RowsPerGroup);
    }

    /// <summary>
    /// LFM3-VL / LFM3-Omni video sampling and labels, as llama.cpp's mtmd (<c>mtmd_video_config_init</c>,
    /// <c>mtmd_helper_video_seq</c>) and HF (<c>Lfm3VlVideoProcessor.sample_frames</c>,
    /// <c>Lfm3OmniProcessor._calculate_timestamps</c>) define them: 2 frames a second of video, clamped to 4..384 (Omni) or
    /// 4..768 (VL) and to the clip's frame count, at <c>round(linspace(0, frames - 1, n))</c> (half to even); one size for all
    /// frames (<see cref="Lfm3ImageProcessor.SmartResizeVideo"/>); consecutive frames paired into one temporal patch, an odd
    /// last frame on its own (both kernels); each pair labelled with the mean time of its two source frames.
    /// <para>Frames are the decoder's frames at their source indices, resized with torchvision's bicubic
    /// (<see cref="Lfm3ImageProcessor.Resize"/>), as HF reads and resizes them. llama.cpp differs in three ways: ffmpeg's
    /// swscale resizes (bicubic B = 0, C = 0.5); its frames come from ffmpeg's constant-rate output, which repeats or drops
    /// frames where the timestamps leave gaps (frame n of that stream is then not source frame n); and its clock is the
    /// stream's nominal rate (r_frame_rate) where HF and this read the average one. The frame size follows the decoder's
    /// probed size: OpenCV and AVFoundation report the displayed size (sample aspect ratio applied), as llama.cpp's ffprobe
    /// reading does.</para>
    /// </summary>
    public static class Lfm3Video
    {
        /// <summary>Frames sampled per second of video.</summary>
        public const double SampleFps = 2.0;

        public const int MinFrames = 4;

        /// <summary>Most frames LFM3-Omni samples (it hears the soundtrack: mtmd's <c>MTMD_VIDEO_TEMPLATE_LFM3_OMNI</c>).</summary>
        public const int MaxFramesOmni = 384;

        /// <summary>Most frames LFM3-VL samples.</summary>
        public const int MaxFramesVl = 768;

        /// <summary>Frames sampled from a clip of <paramref name="sourceFrames"/> frames at <paramref name="sourceFps"/>:
        /// <c>int(frames / fps * 2)</c> clamped to [4, <paramref name="maxFrames"/>] and to the clip.</summary>
        public static int FrameCount(int sourceFrames, double sourceFps, int maxFrames = MaxFramesOmni)
        {
            if (sourceFrames < 1) throw new ArgumentOutOfRangeException(nameof(sourceFrames), $"a clip of {sourceFrames} frames");
            if (!(sourceFps > 0) || double.IsInfinity(sourceFps)) throw new ArgumentOutOfRangeException(nameof(sourceFps), $"a clip at {sourceFps} fps");
            if (maxFrames < MinFrames) throw new ArgumentOutOfRangeException(nameof(maxFrames));
            int n = (int)(sourceFrames / sourceFps * SampleFps);
            return Math.Min(Math.Clamp(n, MinFrames, maxFrames), sourceFrames);
        }

        /// <summary><c>round(linspace(0, sourceFrames - 1, count))</c>, half to even.</summary>
        public static int[] FrameIndices(int sourceFrames, int count)
        {
            if (count < 1 || count > sourceFrames) throw new ArgumentOutOfRangeException(nameof(count), $"{count} of {sourceFrames} frames");
            double step = count > 1 ? (double)(sourceFrames - 1) / (count - 1) : 0.0;
            var indices = new int[count];
            for (int i = 0; i < count; i++)
                indices[i] = (int)Math.Round(i * step, MidpointRounding.ToEven);
            return indices;
        }

        /// <summary>The sampling, frame size and pairs of a clip.</summary>
        public static Lfm3VideoPlan Plan(int sourceFrames, double sourceFps, int sourceWidth, int sourceHeight, int maxFrames = MaxFramesOmni)
        {
            int n = FrameCount(sourceFrames, sourceFps, maxFrames);
            int[] indices = FrameIndices(sourceFrames, n);
            (int w, int h) = Lfm3ImageProcessor.SmartResizeVideo(n, sourceWidth, sourceHeight);
            // HF rounds the clip's fps to 6 decimals first (torchcodec's average carries float noise).
            double fps = Math.Round(sourceFps, 6, MidpointRounding.ToEven);
            return new Lfm3VideoPlan(sourceFrames, sourceFps, sourceWidth, sourceHeight, indices, w, h, PairGroups(indices, fps));
        }

        /// <summary><see cref="Plan(int, double, int, int, int)"/> of a probed clip.</summary>
        public static Lfm3VideoPlan Plan(VideoInfo info, int maxFrames = MaxFramesOmni)
        {
            ArgumentNullException.ThrowIfNull(info);
            if (info.FrameCount < 1 || !(info.Fps > 0) || info.Width < 1 || info.Height < 1)
                throw new InvalidOperationException($"lfm3 video: the container reports {info.FrameCount} frames at {info.Fps} fps, {info.Width} x {info.Height}; " +
                    "the model's sampling needs the frame count and the frame rate");
            return Plan(info.FrameCount, info.Fps, info.Width, info.Height, maxFrames);
        }

        /// <summary>
        /// A plan for frames sampled elsewhere (a client's <c>video_url</c> sampling, extracted frame files): consecutive
        /// frames paired, each pair labelled with the mean of its two <paramref name="seconds"/>, an odd last frame on its
        /// own with its own time, all resized to <see cref="Lfm3ImageProcessor.SmartResizeVideo"/> of the frame count and the
        /// first frame's size. Frames further apart than <paramref name="maxPairGapSeconds"/> are not merged (each fills its
        /// own patch with its own time): the tower was trained on pairs half a second apart, and two different scenes merged
        /// into one patch blend. Encode it with the frames (<see cref="Lfm3VisionEncoder.EncodeVideo(Lfm3VideoPlan,
        /// IReadOnlyList{Lfm3RgbImage}, Action)"/>); its <see cref="Lfm3VideoPlan.FrameIndices"/> index those frames.
        /// </summary>
        public static Lfm3VideoPlan PlanFrames(int width, int height, IReadOnlyList<double> seconds, double maxPairGapSeconds = double.PositiveInfinity)
        {
            ArgumentNullException.ThrowIfNull(seconds);
            if (seconds.Count < 1) throw new ArgumentException("a clip without frames", nameof(seconds));
            var groups = new List<Lfm3VideoGroup>();
            for (int f = 0; f < seconds.Count;)
            {
                double t = seconds[f];
                if (!double.IsFinite(t) || t < 0) throw new ArgumentOutOfRangeException(nameof(seconds), $"frame {f} at {t} s");
                if (f + 1 < seconds.Count && seconds[f + 1] - t <= maxPairGapSeconds)
                {
                    groups.Add(new Lfm3VideoGroup(f, f + 1, (t + seconds[f + 1]) / 2));
                    f += 2;
                }
                else
                {
                    groups.Add(new Lfm3VideoGroup(f, f, t));
                    f++;
                }
            }
            (int w, int h) = Lfm3ImageProcessor.SmartResizeVideo(seconds.Count, width, height);
            var indices = new int[seconds.Count];
            for (int i = 0; i < indices.Length; i++) indices[i] = i;
            return new Lfm3VideoPlan(seconds.Count, 0, width, height, indices, w, h, groups);
        }

        /// <summary>Pairs of consecutive sampled frames labelled <c>(index[i] + index[i + 1]) / (2 fps)</c>; an odd last frame
        /// pairs with itself (HF pads the index list by repeating the last).</summary>
        private static List<Lfm3VideoGroup> PairGroups(int[] indices, double fps)
        {
            var groups = new List<Lfm3VideoGroup>((indices.Length + 1) / 2);
            for (int i = 0; i < indices.Length; i += 2)
            {
                int j = Math.Min(i + 1, indices.Length - 1);
                groups.Add(new Lfm3VideoGroup(i, j, (indices[i] + indices[j]) / (2 * fps)));
            }
            return groups;
        }

        /// <summary>
        /// <c>"&lt;t.t seconds&gt;"</c> with <paramref name="seconds"/> rounded to one decimal half to even on its exact binary
        /// value, as Python's <c>f"{t:.1f}"</c> (and C's <c>%.1f</c>) round it: 0.25 is "0.2", and 0.15 (0.1499...) is "0.1".
        /// </summary>
        public static string FormatTimestamp(double seconds) => "<" + FormatOneDecimal(seconds) + " seconds>";

        internal static string FormatOneDecimal(double value)
        {
            if (!double.IsFinite(value)) throw new ArgumentOutOfRangeException(nameof(value), $"a time of {value}");
            bool negative = value < 0;
            value = Math.Abs(value);
            long bits = BitConverter.DoubleToInt64Bits(value);
            int exponent = (int)((bits >> 52) & 0x7FF);
            long mantissa = bits & 0xFFFFFFFFFFFFFL;
            if (exponent == 0) exponent++;
            else mantissa |= 1L << 52;
            exponent -= 1075; // value = mantissa * 2^exponent
            BigInteger tenths;
            if (exponent >= 0)
            {
                tenths = new BigInteger(mantissa) * 10 << exponent;
            }
            else
            {
                BigInteger num = new BigInteger(mantissa) * 10, den = BigInteger.One << -exponent;
                tenths = BigInteger.DivRem(num, den, out BigInteger rem);
                int cmp = (rem * 2).CompareTo(den);
                if (cmp > 0 || (cmp == 0 && !tenths.IsEven)) tenths += 1;
            }
            BigInteger whole = BigInteger.DivRem(tenths, 10, out BigInteger frac);
            string text = whole.ToString(CultureInfo.InvariantCulture) + "." + frac.ToString(CultureInfo.InvariantCulture);
            return negative && !tenths.IsZero ? "-" + text : text;
        }

        // ================================================================ frames

        /// <summary>Probe and plan the clip at <paramref name="path"/> with <paramref name="decoder"/> (null:
        /// <see cref="MediaCodecs.Video"/>).</summary>
        public static Lfm3VideoPlan Plan(string path, int maxFrames = MaxFramesOmni, IVideoDecoder decoder = null)
            => Plan((decoder ?? MediaCodecs.Video).Probe(path), maxFrames);

        /// <summary>
        /// Probe, plan and read the clip at <paramref name="path"/> (<see cref="ReadFrames"/>). A clip whose decoder runs out
        /// before the frame count its container reports (OpenCV counts from the duration) is counted frame by frame past the
        /// last frame it delivered and sampled again over the frames it has, with a warning naming both counts: the frame
        /// indices, and with them the time labels, then follow the frames that exist.
        /// </summary>
        public static (Lfm3VideoPlan Plan, Lfm3RgbImage[] Frames) Read(string path, int maxFrames = MaxFramesOmni, IVideoDecoder decoder = null)
        {
            decoder ??= MediaCodecs.Video;
            VideoInfo info = decoder.Probe(path);
            Lfm3VideoPlan plan = Plan(info, maxFrames);
            for (int attempt = 0; ; attempt++)
            {
                Lfm3RgbImage[] frames = TryReadFrames(path, plan, decoder, out int delivered);
                if (frames != null) return (plan, frames);
                int count = attempt == 0 && delivered > 0 ? CountFrames(path, decoder, delivered - 1, plan.FrameIndices[^1]) : 0;
                if (count < 1)
                    throw new InvalidOperationException($"lfm3 video: '{path}' delivered {delivered} frames, short of the sampled frame {plan.FrameIndices[^1]} " +
                        $"(it reports {plan.SourceFrames} frames)");
                Console.Error.WriteLine($"lfm3 video: '{path}' reports {plan.SourceFrames} frames but its decoder delivers {count}; " +
                    $"sampling again over those {count} frames (frame indices and time labels follow them).");
                plan = Plan(count, info.Fps, info.Width, info.Height, maxFrames);
            }
        }

        /// <summary>Frames the decoder delivers: it reached <paramref name="reached"/>, and ended before
        /// <paramref name="missing"/>; every frame between is requested.</summary>
        private static int CountFrames(string path, IVideoDecoder decoder, int reached, int missing)
        {
            var request = new List<int>();
            for (int i = MediaHelper.SequentialStepMaxGap; i < reached; i += MediaHelper.SequentialStepMaxGap) request.Add(i);
            for (int i = reached; i <= missing; i++) request.Add(i);
            int last = -1;
            decoder.ReadFrames(path, request, (index, _, _, _, _, _) => last = index);
            return last + 1;
        }

        /// <summary>
        /// Decode the plan's frames of the clip at <paramref name="path"/> and resize each to the plan's size as it arrives
        /// (only resized frames are kept): one RGB picture per <see cref="Lfm3VideoPlan.FrameIndices"/> entry. A clip that
        /// ends before its last planned frame throws, naming how many arrived (<see cref="Read"/> samples again instead).
        /// </summary>
        public static Lfm3RgbImage[] ReadFrames(string path, Lfm3VideoPlan plan, IVideoDecoder decoder = null)
        {
            ArgumentNullException.ThrowIfNull(plan);
            return TryReadFrames(path, plan, decoder ?? MediaCodecs.Video, out int delivered)
                ?? throw new InvalidOperationException($"lfm3 video: '{path}' delivered {delivered} frames, short of the sampled frame {plan.FrameIndices[^1]} " +
                    $"(it reports {plan.SourceFrames} frames)");
        }

        /// <summary>
        /// The plan's frames, or null when the clip ends first (<paramref name="delivered"/>: the frames it delivered).
        /// The request carries extra frames so no two requested frames are more than
        /// <see cref="MediaHelper.SequentialStepMaxGap"/> apart: decoders that follow
        /// <see cref="MediaHelper.PrefersSequentialStepping"/> then step through the clip, which reads exactly frame i,
        /// where a seek can land frames away (OpenCV seeking in an MP4 with an edit list read the wrong frames and ended
        /// early). The extra frames are dropped as they arrive; every source frame is decoded either way, as llama.cpp's
        /// ffmpeg reader decodes them.
        /// </summary>
        private static Lfm3RgbImage[] TryReadFrames(string path, Lfm3VideoPlan plan, IVideoDecoder decoder, out int delivered)
        {
            if (!(plan.SourceFps > 0))
                throw new ArgumentException("lfm3 video: a plan of given frames (PlanFrames) is encoded with those frames, not read from a file");
            int[] wanted = plan.FrameIndices;
            var request = new List<int>(wanted.Length * 2);
            int previous = 0;
            foreach (int index in wanted)
            {
                if (request.Count > 0 && index <= previous)
                    throw new ArgumentException($"lfm3 video: frame {index} after frame {previous} (a plan's frames increase)");
                while (index - previous > MediaHelper.SequentialStepMaxGap)
                {
                    previous += MediaHelper.SequentialStepMaxGap;
                    request.Add(previous);
                }
                request.Add(index);
                previous = index;
            }
            var frames = new Lfm3RgbImage[wanted.Length];
            int next = 0, last = -1;
            decoder.ReadFrames(path, request, (index, pixels, width, height, stride, layout) =>
            {
                last = index;
                if (next < frames.Length && index == wanted[next])
                    frames[next++] = Lfm3ImageProcessor.Resize(ToRgb(pixels, width, height, stride, layout), plan.Width, plan.Height);
            });
            delivered = last + 1;
            return next == frames.Length ? frames : null;
        }

        /// <summary>RGB24 from a decoder's frame buffer (any of the four layouts, rows <paramref name="stride"/> bytes
        /// apart; alpha dropped).</summary>
        public static Lfm3RgbImage ToRgb(byte[] pixels, int width, int height, int stride, PixelLayout layout)
        {
            ArgumentNullException.ThrowIfNull(pixels);
            int bpp = layout is PixelLayout.Bgra or PixelLayout.Rgba ? 4 : 3;
            bool bgr = layout is PixelLayout.Bgr or PixelLayout.Bgra;
            if (stride < width * bpp || (long)stride * (height - 1) + (long)width * bpp > pixels.LongLength)
                throw new ArgumentException($"a {width} x {height} {layout} frame with stride {stride} in {pixels.LongLength} bytes");
            var rgb = new byte[(long)width * height * 3];
            for (int y = 0; y < height; y++)
            {
                long src = (long)y * stride, dst = (long)y * width * 3;
                for (int x = 0; x < width; x++, src += bpp, dst += 3)
                {
                    rgb[dst] = pixels[src + (bgr ? 2 : 0)];
                    rgb[dst + 1] = pixels[src + 1];
                    rgb[dst + 2] = pixels[src + (bgr ? 0 : 2)];
                }
            }
            return new Lfm3RgbImage(width, height, rgb);
        }
    }
}
