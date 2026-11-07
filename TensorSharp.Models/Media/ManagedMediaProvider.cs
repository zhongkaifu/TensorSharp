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
using StbImageSharp;
using TensorSharp.Models.QwenImage;
using TensorSharp.Models.Video;

namespace TensorSharp.Models.Media
{
    /// <summary>
    /// The provider every <see cref="MediaCodecs"/> slot starts on: pure managed code, so it
    /// runs unchanged under Mono AOT on a phone.
    ///
    /// <para>Images: PNG (own decoder, stb_image for palette / 16-bit / interlaced), JPEG, BMP,
    /// GIF (first frame) and PSD through StbImageSharp, EXIF orientation applied; PNG encode
    /// with the encoder video frames use; Lanczos = the Pillow port, bilinear = managed.
    /// Audio: WAV, MP3 (NLayer) and Ogg Vorbis (NVorbis) via <see cref="AudioIO"/>.
    /// Video decode and MP4 encode: none — those need a platform library, and rather than
    /// quietly returning nothing they throw a <see cref="NotSupportedException"/> that names
    /// the slot to fill and the library that fills it.</para>
    /// </summary>
    public sealed class ManagedMediaProvider : IImageCodec, IVideoDecoder, IVideoEncoder, IAudioDecoder
    {
        /// <summary>The shared instance <see cref="MediaCodecs"/> defaults to. The class is
        /// stateless, so creating more is harmless but pointless.</summary>
        public static ManagedMediaProvider Instance { get; } = new ManagedMediaProvider();

        // ------------------------------------------------------------------
        // IImageCodec
        // ------------------------------------------------------------------

        public bool CanDecode(ReadOnlySpan<byte> header) =>
            ImageFormatSniffer.Detect(header) switch
            {
                ImageFormat.Png or ImageFormat.Jpeg or ImageFormat.Bmp or ImageFormat.Gif or ImageFormat.Psd => true,
                _ => false,
            };

        public byte[] DecodeRgba(byte[] file, out int width, out int height)
        {
            if (file == null) throw new ArgumentNullException(nameof(file));

            switch (ImageFormatSniffer.Detect(file))
            {
                case ImageFormat.Png:
                {
                    byte[] rgba = PngCodec.Decode(file, out width, out height);
                    return ExifOrientation.Apply(rgba, ref width, ref height, PngCodec.ReadExifOrientation(file));
                }
                case ImageFormat.Jpeg:
                {
                    byte[] rgba = DecodeWithStb(file, "JPEG", out width, out height);
                    return ExifOrientation.Apply(rgba, ref width, ref height, ExifOrientation.ReadFromJpeg(file));
                }
                case ImageFormat.Bmp:
                    return DecodeWithStb(file, "BMP", out width, out height);
                case ImageFormat.Gif:
                    return DecodeWithStb(file, "GIF", out width, out height);   // first frame, as ImageMagick did
                case ImageFormat.Psd:
                    return DecodeWithStb(file, "PSD", out width, out height);
                case ImageFormat.Heic:
                    throw HeicUnsupported();
                case ImageFormat.WebP:
                    throw Unsupported("WebP");
                case ImageFormat.Tiff:
                    throw Unsupported("TIFF");
                default:
                    throw new NotSupportedException(
                        "Unrecognised image format (the managed image codec reads PNG, JPEG, BMP, GIF and PSD " +
                        "by their magic bytes; a platform IImageCodec registered on MediaCodecs.Image can add more).");
            }
        }

        public (int width, int height) ReadDimensions(byte[] file)
        {
            if (file == null) throw new ArgumentNullException(nameof(file));

            switch (ImageFormatSniffer.Detect(file))
            {
                case ImageFormat.Png:
                    return PngCodec.ReadDimensions(file);
                case ImageFormat.Jpeg:
                case ImageFormat.Bmp:
                case ImageFormat.Gif:
                case ImageFormat.Psd:
                {
                    // Header-only: stb_image's info pass never decodes pixel data.
                    using var stream = new MemoryStream(file, writable: false);
                    ImageInfo? info = ImageInfo.FromStream(stream);
                    if (info == null)
                        throw new InvalidDataException("Failed to read the image header.");
                    return (info.Value.Width, info.Value.Height);
                }
                case ImageFormat.Heic:
                    throw HeicUnsupported();
                case ImageFormat.WebP:
                    throw Unsupported("WebP");
                case ImageFormat.Tiff:
                    throw Unsupported("TIFF");
                default:
                    throw new NotSupportedException(
                        "Unrecognised image format (the managed image codec reads PNG, JPEG, BMP, GIF and PSD).");
            }
        }

        public byte[] EncodePng(byte[] rgb8, int width, int height, int channels) =>
            PngCodec.Encode(rgb8, width, height, channels);

        public byte[] ResizeRgb8(byte[] rgb8, int srcW, int srcH, int dstW, int dstH, ResizeFilter filter)
        {
            if (rgb8 == null) throw new ArgumentNullException(nameof(rgb8));
            if (rgb8.Length != (long)srcW * srcH * 3)
                throw new ArgumentException($"pixel buffer {rgb8.Length} != {srcW}x{srcH}x3", nameof(rgb8));

            return filter switch
            {
                ResizeFilter.Lanczos => ManagedResampler.ResizeLanczosPillow(rgb8, srcW, srcH, dstW, dstH),
                ResizeFilter.Bilinear => ManagedResampler.ResizeBilinearRgb8(rgb8, srcW, srcH, dstW, dstH),
                _ => throw new ArgumentOutOfRangeException(nameof(filter), filter, "unknown resize filter"),
            };
        }

        private static byte[] DecodeWithStb(byte[] data, string formatName, out int width, out int height)
        {
            try
            {
                ImageResult decoded = ImageResult.FromMemory(data, ColorComponents.RedGreenBlueAlpha);
                width = decoded.Width;
                height = decoded.Height;
                return decoded.Data;
            }
            catch (Exception ex)
            {
                throw new InvalidDataException($"Failed to decode {formatName} image.", ex);
            }
        }

        private static NotSupportedException HeicUnsupported() => new(
            "HEIC/HEIF decoding needs a platform image provider: the managed media provider has no HEIF decoder. " +
            "On iOS register an ImageIO/UIImage-backed IImageCodec on MediaCodecs.Image; desktop builds register " +
            "Magick.NET (libheif) automatically.");

        private static NotSupportedException Unsupported(string format) => new(
            $"{format} decoding is not available in the managed image codec (it reads PNG, JPEG, BMP, GIF and PSD). " +
            "Register a platform IImageCodec on MediaCodecs.Image — ImageIO on iOS, Magick.NET on desktop — to decode it.");

        // ------------------------------------------------------------------
        // IVideoDecoder — nothing managed can demux H.264; say so, precisely.
        // ------------------------------------------------------------------

        public VideoInfo Probe(string path) => throw NoVideoDecoder(path);

        public void ReadFrames(string path, IReadOnlyList<int> frameIndices, FrameCallback onFrame) =>
            throw NoVideoDecoder(path);

        private static NotSupportedException NoVideoDecoder(string path) => new(
            $"No video decoder is registered, so '{Path.GetFileName(path ?? string.Empty)}' cannot be read: the managed " +
            "media provider does not demux video. Register an IVideoDecoder on MediaCodecs.Video — AVFoundation " +
            "(AVAssetReader) on iOS; desktop builds register OpenCV automatically.");

        // ------------------------------------------------------------------
        // IVideoEncoder
        // ------------------------------------------------------------------

        public string Mp4ExportUnavailableReason =>
            "No MP4 encoder is registered: the managed media provider cannot encode video.";

        public string SaveMp4(string path, RgbImage[] frames, int fps) => throw new NotSupportedException(
            "No MP4 encoder is registered: the managed media provider cannot encode video. Register an " +
            "IVideoEncoder on MediaCodecs.VideoEncoder — AVAssetWriter on iOS; desktop builds register " +
            "ffmpeg / OpenCV automatically. The generated frames are still available as PNGs.");

        // ------------------------------------------------------------------
        // IAudioDecoder
        // ------------------------------------------------------------------

        public DecodedAudio Decode(string path) => AudioIO.DecodeManaged(path);
    }
}
