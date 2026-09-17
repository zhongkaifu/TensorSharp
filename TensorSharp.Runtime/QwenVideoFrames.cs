// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace TensorSharp.Runtime
{
    /// <summary>
    /// How a Qwen-VL family model (Qwen 3.8 Flash Next, and the Qwen 3.5/3.6 tower it
    /// reuses) sees the frames of a sampled video, shared by the prompt renderer and
    /// the multimodal injector so the two can never disagree about how many
    /// placeholders a clip renders to.
    ///
    /// <para>The layout is the Qwen3-VL processor's. The chat template renders a video
    /// part as <c>&lt;|vision_start|&gt;&lt;|video_pad|&gt;&lt;|vision_end|&gt;</c>; the
    /// processor then merges the sampled frames in pairs (the tower's temporal patch
    /// size) and replaces the single <c>&lt;|video_pad|&gt;</c> with one
    /// <c>&lt;{t:F1} seconds&gt;&lt;|vision_start|&gt;&lt;|video_pad|&gt;&lt;|vision_end|&gt;</c>
    /// block per pair, where <c>t</c> is the mean source time of the pair. Frames are
    /// paired only when they are at most <see cref="MaxPairedFrameGapSeconds"/> apart
    /// (the processor's own 2 fps sampling); a sparser frame fills its own temporal
    /// patch and is labelled with its own time. The
    /// processor replaces the clip's outer start/pad/end sequence; it does not
    /// nest a second pair of vision delimiters around the temporal blocks. A clip with
    /// an odd number of frames repeats its last frame to complete the final pair, and
    /// that pair's time is the repeated frame's time. Every pair is then expanded to its
    /// merged-patch token count and positioned like a still image whose temporal
    /// coordinate is the running position at that pair, so consecutive pairs carry
    /// increasing temporal M-RoPE ids.</para>
    ///
    /// <para>A clip is a run of consecutive frames in <see cref="ChatMessage.ImagePaths"/>
    /// whose <see cref="ChatMessage.ImageTimestamps"/> are set and strictly increasing;
    /// a frame whose time does not increase starts a new clip, which is how two
    /// <c>video_url</c> parts in one message stay two videos (each clip's sampling starts
    /// at its own first frame). Frames without a source time, including the frames of a
    /// legacy Web UI video upload, are still images.</para>
    /// </summary>
    public static class QwenVideoFrames
    {
        public const string VisionStart = "<|vision_start|>";
        public const string VisionEnd = "<|vision_end|>";
        public const string ImagePad = "<|image_pad|>";
        public const string VideoPad = "<|video_pad|>";

        /// <summary>Frames merged into one temporal patch by the Qwen-VL tower.</summary>
        public const int TemporalPatchSize = 2;

        /// <summary>
        /// The widest gap between two sampled frames that still share a temporal patch.
        /// The Qwen3-VL video processor samples at 2 fps, so the tower's temporal patch
        /// merges frames 0.5 s apart - one patch per second of video. A client that
        /// samples sparser (<c>fps: 1</c>, or a long clip spread over <c>max_frames</c>)
        /// hands over frames that are different scenes; merging two of them blends both
        /// into one patch, and the model reads neither (a 17 / 42 / 86 slide clip at
        /// 1 fps read back as "12", "47", "86"). Such frames each fill their own patch
        /// instead. The 15% margin absorbs the rounding of 2 fps samples onto a 25 or
        /// 29.97 fps source.
        /// </summary>
        public const double MaxPairedFrameGapSeconds = 0.575;

        /// <summary>
        /// One temporal patch of a clip: the indices (into the message's image list) of
        /// the two frames it merges, and the source time it is labelled with.
        /// <see cref="Second"/> equals <see cref="First"/> when the clip's last frame is
        /// repeated to complete the pair.
        /// </summary>
        public readonly record struct Group(int First, int Second, double Seconds);

        /// <summary>A still image (one index) or a clip (one or more groups), in prompt order.</summary>
        public sealed class Item
        {
            public int ImageIndex { get; init; } = -1;
            public IReadOnlyList<Group>? Groups { get; init; }
            public bool IsVideo => Groups != null;
        }

        /// <summary>
        /// The stills and clips of one message, in the order their placeholders render.
        /// </summary>
        public static List<Item> Layout(ChatMessage message)
        {
            var items = new List<Item>();
            var paths = message?.ImagePaths;
            if (paths == null || paths.Count == 0)
                return items;

            var times = message.ImageTimestamps;
            bool timed = times != null && times.Count == paths.Count;
            int i = 0;
            while (i < paths.Count)
            {
                if (!timed || times![i] is not double first)
                {
                    items.Add(new Item { ImageIndex = i });
                    i++;
                    continue;
                }

                // Extend the clip while the source time keeps increasing.
                var frames = new List<(int Index, double Seconds)> { (i, Validate(first)) };
                int j = i + 1;
                while (j < paths.Count && times![j] is double next && next > frames[^1].Seconds)
                {
                    frames.Add((j, Validate(next)));
                    j++;
                }

                var groups = new List<Group>(frames.Count);
                int f = 0;
                while (f < frames.Count)
                {
                    // Pair a frame with the next only when the two are as close as the
                    // frames the tower was trained to merge. Sparser samples are
                    // different scenes: merging them blends both into one patch.
                    if (f + 1 < frames.Count
                        && frames[f + 1].Seconds - frames[f].Seconds <= MaxPairedFrameGapSeconds)
                    {
                        groups.Add(new Group(frames[f].Index, frames[f + 1].Index,
                            (frames[f].Seconds + frames[f + 1].Seconds) / TemporalPatchSize));
                        f += TemporalPatchSize;
                    }
                    else
                    {
                        // The frame fills its own temporal patch (repeated, exactly as the
                        // processor completes an odd clip and as a still image is encoded)
                        // and keeps its own time label.
                        groups.Add(new Group(frames[f].Index, frames[f].Index, frames[f].Seconds));
                        f++;
                    }
                }
                items.Add(new Item { Groups = groups });
                i = j;
            }
            return items;
        }

        private static double Validate(double seconds)
        {
            if (!double.IsFinite(seconds) || seconds < 0)
                throw new ArgumentOutOfRangeException(nameof(ChatMessage.ImageTimestamps),
                    "A video frame's source time must be a finite, non-negative number of seconds.");
            return seconds;
        }

        /// <summary>
        /// The time label of one temporal patch, one decimal as the Qwen3-VL processor's
        /// <c>f"{t:.1f}"</c> writes it. Ties round to even, as Python's formatting does
        /// (0.25 s is "0.2", 0.75 s is "0.8"); the label is prompt text, so one different
        /// character is one different token.
        /// </summary>
        public static string FormatSeconds(double seconds) =>
            Math.Round(seconds, 1, MidpointRounding.ToEven).ToString("F1", CultureInfo.InvariantCulture);

        /// <summary>
        /// Append the message's vision placeholders in prompt order: one
        /// <c>&lt;|vision_start|&gt;&lt;|image_pad|&gt;&lt;|vision_end|&gt;</c> per still
        /// image, and per clip the timestamped per-pair blocks described on this class.
        /// </summary>
        public static void AppendPlaceholders(ChatMessage message, StringBuilder text)
        {
            foreach (var item in Layout(message))
            {
                if (!item.IsVideo)
                {
                    text.Append(VisionStart).Append(ImagePad).Append(VisionEnd);
                    continue;
                }
                // The tower labels a temporal pair with its mean timestamp. That
                // loses the individual sampling times, especially when a frame
                // cap selects nonadjacent frames. Preserve those source facts in
                // text so later turns can resolve time within a pair. Keep the
                // trained per-pair vision-token layout itself unchanged.
                text.Append("Sampled video frame times in chronological order: ");
                bool firstFrame = true;
                foreach (var group in item.Groups!)
                {
                    AppendFrameTime(group.First);
                    if (group.Second != group.First) AppendFrameTime(group.Second);
                }
                text.Append(" seconds.\n");
                foreach (var group in item.Groups!)
                {
                    text.Append('<').Append(FormatSeconds(group.Seconds)).Append(" seconds>")
                        .Append(VisionStart).Append(VideoPad).Append(VisionEnd);
                }

                void AppendFrameTime(int index)
                {
                    if (!firstFrame) text.Append(", ");
                    text.Append(message.ImageTimestamps![index]!.Value.ToString("R", CultureInfo.InvariantCulture));
                    firstFrame = false;
                }
            }
        }
    }
}
