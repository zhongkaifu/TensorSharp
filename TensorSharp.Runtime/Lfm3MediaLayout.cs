// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;

namespace TensorSharp.Runtime
{
    /// <summary>What one LFM3 media placeholder stands for.</summary>
    public enum Lfm3MediaKind
    {
        Image,
        Video,
        Audio,
    }

    /// <summary>
    /// One media item of a message, in prompt order (<see cref="Lfm3MediaLayout.Items"/>).
    /// </summary>
    public sealed class Lfm3MediaItem
    {
        public Lfm3MediaKind Kind { get; init; }

        /// <summary>An image's index in <see cref="ChatMessage.ImagePaths"/>, an audio clip's in
        /// <see cref="ChatMessage.AudioPaths"/>; -1 for a video.</summary>
        public int Index { get; init; } = -1;

        /// <summary>A video read from its file (an entry of <see cref="ChatMessage.VideoFilePaths"/>): the model samples
        /// its frames and hears its soundtrack. Null for a video of frames sampled by the host.</summary>
        public string? VideoFile { get; init; }

        /// <summary>A video of host-sampled frames: their indices in <see cref="ChatMessage.ImagePaths"/>, in order.
        /// Null for a file video.</summary>
        public IReadOnlyList<int>? Frames { get; init; }

        /// <summary>The source times of <see cref="Frames"/> when the message carries them
        /// (<see cref="ChatMessage.ImageTimestamps"/>), else null.</summary>
        public IReadOnlyList<double>? FrameSeconds { get; init; }
    }

    /// <summary>
    /// How an LFM3 / LFM3-Omni message's attachments become media spans, shared by the prompt renderer (which writes one
    /// template span per item, <see cref="ChatTemplate.AppendLfm3MediaPlaceholders"/>) and the model's expander (which
    /// fills them), so the two always agree on the count and order.
    /// <list type="number">
    /// <item><see cref="ChatMessage.ImagePaths"/> splits into still images and runs of video frames: with
    /// <see cref="ChatMessage.ImageTimestamps"/> aligned, an entry without a time is a still and a run of strictly
    /// increasing times is one video (two <c>video_url</c> parts stay two videos, <see cref="QwenVideoFrames"/>' rule);
    /// without times, a <see cref="ChatMessage.IsVideo"/> message's entries are frames, except an entry the user attached
    /// (<see cref="ChatMessage.AttachmentPaths"/> never lists an extracted frame), which is a still; otherwise every entry
    /// is a still.</item>
    /// <item>A message that names its video files (<see cref="ChatMessage.VideoFilePaths"/>) is read from them: when
    /// there is one file per frame run, file k takes run k's place; otherwise (the runs cannot be matched to the files)
    /// the stills keep their order and the files follow them. The host-sampled frames are not used then.</item>
    /// <item>Without files, each frame run is a video of those frames.</item>
    /// <item>Audio clips follow the visual items, in order.</item>
    /// </list>
    /// </summary>
    public static class Lfm3MediaLayout
    {
        public static List<Lfm3MediaItem> Items(ChatMessage? message)
        {
            var items = new List<Lfm3MediaItem>();
            if (message == null)
                return items;

            List<string>? paths = message.ImagePaths;
            List<string>? files = message.VideoFilePaths is { Count: > 0 } named ? named : null;
            // Visual entries in order: a still (Image) or a frame run (Video with Frames).
            var visual = new List<Lfm3MediaItem>();
            int runs = 0;
            if (paths is { Count: > 0 })
            {
                List<double?>? times = message.ImageTimestamps;
                if (times != null && times.Count == paths.Count)
                {
                    int i = 0;
                    while (i < paths.Count)
                    {
                        if (times[i] is not double first)
                        {
                            visual.Add(new Lfm3MediaItem { Kind = Lfm3MediaKind.Image, Index = i });
                            i++;
                            continue;
                        }
                        var frames = new List<int> { i };
                        var seconds = new List<double> { first };
                        int j = i + 1;
                        while (j < paths.Count && times[j] is double next && next > seconds[^1])
                        {
                            frames.Add(j);
                            seconds.Add(next);
                            j++;
                        }
                        visual.Add(new Lfm3MediaItem { Kind = Lfm3MediaKind.Video, Frames = frames, FrameSeconds = seconds });
                        runs++;
                        i = j;
                    }
                }
                else if (message.IsVideo)
                {
                    var attached = message.AttachmentPaths is { Count: > 0 } a
                        ? new HashSet<string>(a, StringComparer.Ordinal) : null;
                    List<int>? run = null;
                    for (int i = 0; i < paths.Count; i++)
                    {
                        if (attached != null && attached.Contains(paths[i]))
                        {
                            run = null;
                            visual.Add(new Lfm3MediaItem { Kind = Lfm3MediaKind.Image, Index = i });
                            continue;
                        }
                        if (run == null)
                        {
                            run = new List<int>();
                            visual.Add(new Lfm3MediaItem { Kind = Lfm3MediaKind.Video, Frames = run });
                            runs++;
                        }
                        run.Add(i);
                    }
                }
                else
                {
                    for (int i = 0; i < paths.Count; i++)
                        visual.Add(new Lfm3MediaItem { Kind = Lfm3MediaKind.Image, Index = i });
                }
            }

            if (files == null)
            {
                items.AddRange(visual);
            }
            else if (runs == files.Count)
            {
                int k = 0;
                foreach (Lfm3MediaItem item in visual)
                    items.Add(item.Kind == Lfm3MediaKind.Video
                        ? new Lfm3MediaItem { Kind = Lfm3MediaKind.Video, VideoFile = files[k++] }
                        : item);
            }
            else
            {
                foreach (Lfm3MediaItem item in visual)
                    if (item.Kind == Lfm3MediaKind.Image)
                        items.Add(item);
                foreach (string file in files)
                    items.Add(new Lfm3MediaItem { Kind = Lfm3MediaKind.Video, VideoFile = file });
            }

            if (message.AudioPaths is { Count: > 0 } audio)
                for (int i = 0; i < audio.Count; i++)
                    items.Add(new Lfm3MediaItem { Kind = Lfm3MediaKind.Audio, Index = i });
            return items;
        }
    }
}
