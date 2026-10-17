// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
using TensorSharp.Models.QwenImage;
using TensorSharp.Models.Video;
using System.Threading;

namespace TensorSharp.Models.Media
{
    /// <summary>Writes generated frames as an MP4 file.</summary>
    public interface IVideoEncoder
    {
        /// <summary>Whether this provider supplies the libx264 conditioning transform
        /// used by LTX image guides. An OS H.264 export encoder alone is insufficient.</summary>
        bool SupportsH264ImageConditioning => false;

        /// <summary>Encode and decode one packed RGB8 frame using libx264 at 1 fps,
        /// the veryfast preset, the requested CRF and yuv420p. RGB/YUV conversion uses
        /// bilinear scaling and the encoder uses slice threading, matching PyAV's
        /// defaults in the publisher's image-conditioning implementation. Dimensions
        /// must be positive and even; the returned RGB8 buffer has the same geometry.
        /// No alternate codec or bitrate fallback is permitted.</summary>
        byte[] RecompressH264Image(byte[] rgb8, int width, int height, int crf,
            CancellationToken cancellationToken = default) => throw new System.NotSupportedException(
                $"{GetType().Name} does not provide the libx264 image-conditioning transform required by LTX. " +
                "Use the desktop media provider with FFmpeg (including libx264) on PATH or set TS_FFMPEG.");

        /// <summary>
        /// Write <paramref name="frames"/> (all the same size, HWC RGB float [0,1]) at
        /// <paramref name="fps"/> to <paramref name="path"/>, whose directory already exists.
        /// Returns the codec actually used: <c>"h264"</c> plays in browsers, <c>"mp4v"</c>
        /// (MPEG-4 Part 2) does not — callers that serve the file check this. A provider that
        /// cannot encode at all throws <see cref="System.InvalidOperationException"/> (or
        /// <see cref="System.NotSupportedException"/> when no encoder is registered) rather
        /// than writing a file nothing can play.
        /// </summary>
        string SaveMp4(string path, RgbImage[] frames, int fps);

        /// <summary>
        /// As <see cref="SaveMp4(string, RgbImage[], int)"/>, with the soundtrack a model
        /// generated alongside the frames (null when it made none) written into the same
        /// file when the provider can. <paramref name="audioMuxed"/> says whether it was, so a
        /// caller knows whether the MP4 alone carries the sound. A provider without an audio
        /// path writes the frames alone, which is what every provider did before this
        /// overload existed.
        /// </summary>
        string SaveMp4(string path, RgbImage[] frames, int fps, GeneratedVideoAudio audio, out bool audioMuxed)
        {
            audioMuxed = false;
            return SaveMp4(path, frames, fps);
        }
    }
}
