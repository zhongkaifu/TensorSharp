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
    /// block per pair, where <c>t</c> is the mean source time of the pair, so the
    /// template's own start/end tokens stay wrapped around the whole clip. A clip with
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

                var groups = new List<Group>(frames.Count / TemporalPatchSize + 1);
                for (int f = 0; f < frames.Count; f += TemporalPatchSize)
                {
                    int last = Math.Min(f + TemporalPatchSize - 1, frames.Count - 1);
                    double seconds = 0;
                    for (int k = f; k < f + TemporalPatchSize; k++)
                        seconds += frames[Math.Min(k, frames.Count - 1)].Seconds;
                    groups.Add(new Group(frames[f].Index, frames[last].Index, seconds / TemporalPatchSize));
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
        /// image, and per clip the wrapped per-pair blocks described on this class.
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
                text.Append(VisionStart);
                foreach (var group in item.Groups!)
                {
                    text.Append('<').Append(FormatSeconds(group.Seconds)).Append(" seconds>")
                        .Append(VisionStart).Append(VideoPad).Append(VisionEnd);
                }
                text.Append(VisionEnd);
            }
        }
    }
}
