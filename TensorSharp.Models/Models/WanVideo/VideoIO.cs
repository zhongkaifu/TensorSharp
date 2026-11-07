// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// MP4 writing for generated video. The encoder itself is the platform's
// (MediaCodecs.VideoEncoder): on desktop the ffmpeg -> OpenCV 'avc1' -> OpenCV 'mp4v'
// ladder in Media/Desktop, on iOS an AVAssetWriter the app registers. Callers that
// need in-browser playback should check the returned codec ("h264" plays everywhere,
// "mp4v" does not).
using System;
using System.IO;
using TensorSharp.Models.Media;
using TensorSharp.Models.QwenImage;

namespace TensorSharp.Models.WanVideo
{
    public static class VideoIO
    {
        /// <summary>Why the registered encoder cannot write any MP4 on this machine, or null
        /// when it has one to try (<see cref="IVideoEncoder.Mp4ExportUnavailableReason"/>).
        /// Hosts check it before generating a clip that is to be saved as MP4.</summary>
        public static string Mp4ExportUnavailableReason() => MediaCodecs.VideoEncoder.Mp4ExportUnavailableReason;

        /// <summary>Write frames as an MP4. Returns the codec actually used
        /// ("h264" or "mp4v").</summary>
        public static string SaveMp4(string path, RgbImage[] frames, int fps)
        {
            if (frames == null || frames.Length == 0)
                throw new ArgumentException("no frames to save", nameof(frames));
            if (fps <= 0) fps = 16;

            string full = Path.GetFullPath(path);
            Directory.CreateDirectory(Path.GetDirectoryName(full) ?? ".");

            return MediaCodecs.VideoEncoder.SaveMp4(full, frames, fps);
        }

        /// <summary>Write frames as an MP4 with the soundtrack inside it when the platform's
        /// encoder can mux one (<paramref name="audioMuxed"/> says whether it did). Returns
        /// the codec actually used, as <see cref="SaveMp4(string, RgbImage[], int)"/>.</summary>
        public static string SaveMp4(string path, RgbImage[] frames, int fps,
                                     TensorSharp.Models.Video.GeneratedVideoAudio audio, out bool audioMuxed)
        {
            if (frames == null || frames.Length == 0)
                throw new ArgumentException("no frames to save", nameof(frames));
            if (fps <= 0) fps = 16;

            string full = Path.GetFullPath(path);
            Directory.CreateDirectory(Path.GetDirectoryName(full) ?? ".");

            if (audio is not { ChannelCount: > 0, SampleCount: > 0, SampleRate: > 0 })
            {
                audioMuxed = false;
                return MediaCodecs.VideoEncoder.SaveMp4(full, frames, fps);
            }
            return MediaCodecs.VideoEncoder.SaveMp4(full, frames, fps, audio, out audioMuxed);
        }

        /// <summary>Encode to an in-memory MP4 (for server responses). Returns null codec info
        /// via <paramref name="codec"/> as in <see cref="SaveMp4"/>.</summary>
        public static byte[] EncodeMp4(RgbImage[] frames, int fps, out string codec)
        {
            string tmp = Path.Combine(Path.GetTempPath(), $"tswan-{Guid.NewGuid():N}.mp4");
            try
            {
                codec = SaveMp4(tmp, frames, fps);
                return File.ReadAllBytes(tmp);
            }
            finally
            {
                try { if (File.Exists(tmp)) File.Delete(tmp); } catch { /* best effort */ }
            }
        }
    }
}
