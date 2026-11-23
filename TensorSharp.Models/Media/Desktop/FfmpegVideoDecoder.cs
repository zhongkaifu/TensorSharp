// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;

namespace TensorSharp.Models.Media.Desktop
{
    /// <summary>
    /// Frame access through the ffprobe and ffmpeg executables, for hosts where OpenCvSharp's native library does not
    /// load (OpenCvSharp does not ship it for every platform). <see cref="DesktopMediaProvider"/> hands its video calls
    /// here only then, so a host with OpenCV keeps OpenCV's frames.
    ///
    /// <para>Probe reads the first video stream's frame rate, frame count, size, sample aspect ratio and rotation, and
    /// reports what OpenCV's FFmpeg backend reports: the displayed width (the sample aspect ratio applied) and, for a clip
    /// rotated a quarter turn, the rotated size, since both decoders deliver upright frames. ReadFrames runs one ffmpeg
    /// over the clip with a <c>select</c> filter that keeps the wanted frame numbers, decoded upright at their encoded
    /// size as RGB, so a clip is decoded once however many frames are wanted.</para>
    /// </summary>
    public sealed class FfmpegVideoDecoder : IVideoDecoder
    {
        private readonly string _ffmpeg;
        private readonly string _ffprobe;

        private FfmpegVideoDecoder(string ffmpeg, string ffprobe)
        {
            _ffmpeg = ffmpeg;
            _ffprobe = ffprobe;
        }

        /// <summary>The decoder over the ffmpeg and ffprobe this host has (TS_FFMPEG / TS_FFPROBE, PATH, the
        /// application folder), or null when either is missing.</summary>
        public static FfmpegVideoDecoder TryCreate()
        {
            string ffmpeg = MediaExecutableLocator.Find("ffmpeg", "TS_FFMPEG");
            if (ffmpeg == null) return null;
            string ffprobe = MediaExecutableLocator.Find("ffprobe", "TS_FFPROBE", siblingOf: ffmpeg);
            return ffprobe == null ? null : new FfmpegVideoDecoder(ffmpeg, ffprobe);
        }

        public VideoInfo Probe(string path)
        {
            if (string.IsNullOrWhiteSpace(path)) throw new ArgumentNullException(nameof(path));
            if (!File.Exists(path)) throw new FileNotFoundException($"video not found: {path}", path);
            var (width, height, sar, rotation, fps, frames, duration) = ProbeStream(path);
            if (frames <= 0 && fps > 0 && duration > 0)
                frames = (int)Math.Round(duration * fps, MidpointRounding.AwayFromZero);
            int shown = sar is >= 0.1 and <= 10.0 ? (int)Math.Round(width * sar, MidpointRounding.ToEven) : width;
            return QuarterTurn(rotation) ? new VideoInfo(fps, frames, height, shown) : new VideoInfo(fps, frames, shown, height);
        }

        public void ReadFrames(string path, IReadOnlyList<int> frameIndices, FrameCallback onFrame)
        {
            if (string.IsNullOrWhiteSpace(path)) throw new ArgumentNullException(nameof(path));
            if (frameIndices == null) throw new ArgumentNullException(nameof(frameIndices));
            if (onFrame == null) throw new ArgumentNullException(nameof(onFrame));
            if (frameIndices.Count == 0) return;
            if (!File.Exists(path)) throw new FileNotFoundException($"video not found: {path}", path);
            for (int i = 0; i < frameIndices.Count; i++)
            {
                if (frameIndices[i] < 0) throw new ArgumentOutOfRangeException(nameof(frameIndices), "frame indices must be >= 0");
                if (i > 0 && frameIndices[i] < frameIndices[i - 1])
                    throw new ArgumentException("frame indices must be non-decreasing", nameof(frameIndices));
            }

            var (encodedWidth, encodedHeight, _, rotation, _, _, _) = ProbeStream(path);
            int width = QuarterTurn(rotation) ? encodedHeight : encodedWidth;
            int height = QuarterTurn(rotation) ? encodedWidth : encodedHeight;
            if (width <= 0 || height <= 0)
                throw new InvalidOperationException($"Failed to open video file: {path} (no video stream size)");

            int[] distinct = frameIndices.Distinct().ToArray();
            string select = "select=" + string.Join("+", distinct.Select(i => string.Create(CultureInfo.InvariantCulture, $"eq(n\\,{i})")));
            var start = new ProcessStartInfo(_ffmpeg)
            {
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true,
            };
            foreach (string a in new[] { "-nostdin", "-hide_banner", "-loglevel", "error", "-i", path, "-map", "0:v:0",
                         "-vf", select, "-fps_mode", "passthrough", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1" })
                start.ArgumentList.Add(a);

            using var process = new Process { StartInfo = start };
            try { process.Start(); }
            catch (Exception ex) when (ex is System.ComponentModel.Win32Exception or InvalidOperationException)
            {
                throw new NotSupportedException($"could not start ffmpeg ('{_ffmpeg}') to read '{Path.GetFileName(path)}': {ex.Message}", ex);
            }
            Task<string> errors = process.StandardError.ReadToEndAsync();
            int stride = width * 3;
            var frame = new byte[(long)stride * height];
            Stream output = process.StandardOutput.BaseStream;
            int next = 0;   // into frameIndices
            try
            {
                foreach (int index in distinct)
                {
                    if (!ReadFull(output, frame)) break;   // a short read is how a clip ends
                    while (next < frameIndices.Count && frameIndices[next] == index)
                    {
                        onFrame(index, frame, width, height, stride, PixelLayout.Rgb);
                        next++;
                    }
                }
            }
            finally
            {
                if (!process.HasExited)
                {
                    try { process.Kill(entireProcessTree: true); } catch (InvalidOperationException) { }
                }
                process.WaitForExit();
            }
            // ffmpeg killed after the last wanted frame exits nonzero; only a run that delivered nothing is a failure.
            if (next == 0 && process.ExitCode != 0)
                throw new InvalidOperationException(
                    $"ffmpeg could not decode '{Path.GetFileName(path)}' (exit {process.ExitCode}): " +
                    DesktopMediaProvider.SummarizeFfmpegError(MediaInputException.Scrub(errors.GetAwaiter().GetResult(), new[] { path })));
        }

        private static bool QuarterTurn(int rotation) => Math.Abs(rotation) % 180 == 90;

        private static bool ReadFull(Stream stream, byte[] buffer)
        {
            int read = 0;
            while (read < buffer.Length)
            {
                int n = stream.Read(buffer, read, buffer.Length - read);
                if (n <= 0) return false;
                read += n;
            }
            return true;
        }

        /// <summary>The first video stream's encoded size, sample aspect ratio, rotation (degrees), frame rate, frame count
        /// (0 when the container does not say) and duration in seconds.</summary>
        private (int Width, int Height, double Sar, int Rotation, double Fps, int Frames, double Duration) ProbeStream(string path)
        {
            var start = new ProcessStartInfo(_ffprobe)
            {
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true,
            };
            foreach (string a in new[] { "-v", "error", "-select_streams", "v:0", "-show_entries",
                         "stream=width,height,sample_aspect_ratio,avg_frame_rate,r_frame_rate,nb_frames,duration:" +
                         "stream_side_data=rotation:format=duration", "-of", "json", path })
                start.ArgumentList.Add(a);
            using var process = new Process { StartInfo = start };
            try { process.Start(); }
            catch (Exception ex) when (ex is System.ComponentModel.Win32Exception or InvalidOperationException)
            {
                throw new NotSupportedException($"could not start ffprobe ('{_ffprobe}') to read '{Path.GetFileName(path)}': {ex.Message}", ex);
            }
            Task<string> errors = process.StandardError.ReadToEndAsync();
            string json = process.StandardOutput.ReadToEnd();
            process.WaitForExit();
            if (process.ExitCode != 0)
                throw new InvalidOperationException($"Failed to open video file: {Path.GetFileName(path)}: " +
                    DesktopMediaProvider.SummarizeFfmpegError(MediaInputException.Scrub(errors.GetAwaiter().GetResult(), new[] { path })));

            using JsonDocument doc = JsonDocument.Parse(json);
            if (!doc.RootElement.TryGetProperty("streams", out JsonElement streams) || streams.GetArrayLength() == 0)
                throw new InvalidOperationException($"Failed to open video file: {Path.GetFileName(path)} has no video stream");
            JsonElement s = streams[0];
            int width = Int(s, "width"), height = Int(s, "height");
            double sar = Ratio(Str(s, "sample_aspect_ratio"), ':');
            double fps = Ratio(Str(s, "avg_frame_rate"), '/');
            if (fps <= 0) fps = Ratio(Str(s, "r_frame_rate"), '/');
            int frames = int.TryParse(Str(s, "nb_frames"), NumberStyles.Integer, CultureInfo.InvariantCulture, out int n) ? n : 0;
            double duration = Dbl(Str(s, "duration"));
            if (duration <= 0 && doc.RootElement.TryGetProperty("format", out JsonElement format))
                duration = Dbl(Str(format, "duration"));
            int rotation = 0;
            if (s.TryGetProperty("side_data_list", out JsonElement sides))
                foreach (JsonElement side in sides.EnumerateArray())
                    if (side.TryGetProperty("rotation", out JsonElement r) && r.TryGetInt32(out int deg)) rotation = deg;
            return (width, height, sar <= 0 ? 1.0 : sar, rotation, fps, frames, duration);
        }

        private static string Str(JsonElement e, string name)
            => e.TryGetProperty(name, out JsonElement v) ? (v.ValueKind == JsonValueKind.String ? v.GetString() : v.ToString()) : null;

        private static int Int(JsonElement e, string name) => e.TryGetProperty(name, out JsonElement v) && v.TryGetInt32(out int i) ? i : 0;

        private static double Dbl(string s)
            => double.TryParse(s, NumberStyles.Float, CultureInfo.InvariantCulture, out double d) ? d : 0;

        /// <summary>"a/b" (or "a:b") as a / b; 0 when absent or b is 0.</summary>
        private static double Ratio(string s, char separator)
        {
            if (string.IsNullOrEmpty(s)) return 0;
            int at = s.IndexOf(separator);
            if (at < 0) return Dbl(s);
            double a = Dbl(s[..at]), b = Dbl(s[(at + 1)..]);
            return b == 0 ? 0 : a / b;
        }
    }
}
