// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using OpenCvSharp;

namespace TensorSharp.Models.Media.Desktop
{
    /// <summary>Frame access through OpenCvSharp's <c>VideoCapture</c>.</summary>
    public sealed partial class DesktopMediaProvider
    {
        /// <summary>
        /// Opens <paramref name="path"/> with OpenCV. OpenCvSharp's native library
        /// (OpenCvSharpExtern) has no Windows on Arm build, so there the first
        /// <see cref="VideoCapture"/> fails inside a type initializer with a
        /// DllNotFoundException; report what is missing instead of that.
        /// </summary>
        private static VideoCapture OpenVideoCapture(string path)
        {
            try
            {
                return new VideoCapture(path);
            }
            catch (Exception ex) when (ex is DllNotFoundException || ex.InnerException is DllNotFoundException)
            {
                throw new PlatformNotSupportedException(
                    "Video input needs OpenCV's native library (OpenCvSharpExtern), which could not be loaded on " +
                    $"{RuntimeInformation.RuntimeIdentifier}. OpenCvSharp ships no Windows on Arm build of it; " +
                    "elsewhere, check that its runtime package for this platform is installed.", ex);
            }
        }

        public VideoInfo Probe(string path)
        {
            if (string.IsNullOrWhiteSpace(path))
                throw new ArgumentNullException(nameof(path));
            if (!File.Exists(path))
                throw new FileNotFoundException($"video not found: {path}", path);

            using var capture = OpenVideoCapture(path);
            if (!capture.IsOpened())
                throw new InvalidOperationException($"Failed to open video file: {path}");

            return new VideoInfo(
                capture.Get(VideoCaptureProperties.Fps),
                (int)capture.Get(VideoCaptureProperties.FrameCount),
                (int)capture.Get(VideoCaptureProperties.FrameWidth),
                (int)capture.Get(VideoCaptureProperties.FrameHeight));
        }

        /// <summary>
        /// Decoding is strictly sequential: one <see cref="VideoCapture"/>, one reused
        /// <see cref="Mat"/>, and one reused managed buffer the frame is copied into for the
        /// callback (the Mat is overwritten by the next decode, so the callback must not keep
        /// the buffer either — <see cref="MediaHelper"/> copies it into PNG scanlines at once).
        /// </summary>
        public void ReadFrames(string path, IReadOnlyList<int> frameIndices, FrameCallback onFrame)
        {
            if (string.IsNullOrWhiteSpace(path))
                throw new ArgumentNullException(nameof(path));
            if (frameIndices == null)
                throw new ArgumentNullException(nameof(frameIndices));
            if (onFrame == null)
                throw new ArgumentNullException(nameof(onFrame));
            if (frameIndices.Count == 0)
                return;
            if (!File.Exists(path))
                throw new FileNotFoundException($"video not found: {path}", path);

            using var capture = OpenVideoCapture(path);
            if (!capture.IsOpened())
                throw new InvalidOperationException($"Failed to open video file: {path}");

            using var mat = new Mat();
            byte[] buffer = null;

            // Walking forward with Grab() skips a frame for a fraction of the cost of a
            // decode, while Set(PosFrames) pays a keyframe seek plus decode-forward that
            // is roughly constant in distance. Below the crossover, stepping is both
            // cheaper and exact — it never depends on the container's index being
            // trustworthy. Above it, seeking wins and is the only sane option on a long
            // clip sampled sparsely. The two are never mixed within one extraction: a
            // seek that lands off-by-a-frame would then silently skew every subsequent
            // step. The crossover itself was measured with this decoder and lives on
            // MediaHelper so every provider follows the same policy.
            bool stepForward = MediaHelper.PrefersSequentialStepping(frameIndices);
            int cursor = 0;

            foreach (int frameIdx in frameIndices)
            {
                if (frameIdx < 0)
                    throw new ArgumentOutOfRangeException(nameof(frameIndices), "frame indices must be >= 0");

                if (stepForward)
                {
                    bool exhausted = false;
                    while (cursor < frameIdx)
                    {
                        if (!capture.Grab()) { exhausted = true; break; }
                        cursor++;
                    }
                    if (exhausted)
                        return;
                }
                else
                {
                    capture.Set(VideoCaptureProperties.PosFrames, frameIdx);
                }

                if (!capture.Read(mat) || mat.Empty())
                    return;
                cursor = frameIdx + 1;

                int width = mat.Cols, height = mat.Rows, channels = mat.Channels();
                PixelLayout layout = channels switch
                {
                    3 => PixelLayout.Bgr,
                    4 => PixelLayout.Bgra,
                    _ => throw new NotSupportedException(
                        $"OpenCV decoded a {channels}-channel frame; VideoCapture is expected to deliver BGR or BGRA."),
                };
                int stride = (int)mat.Step();
                long bytes = (long)stride * height;
                if (bytes > int.MaxValue)
                    throw new NotSupportedException($"frame of {width}x{height} exceeds the 2 GB buffer limit");
                if (buffer == null || buffer.Length < bytes)
                    buffer = new byte[bytes];
                Marshal.Copy(mat.Data, buffer, 0, (int)bytes);

                onFrame(frameIdx, buffer, width, height, stride, layout);
            }
        }
    }
}
