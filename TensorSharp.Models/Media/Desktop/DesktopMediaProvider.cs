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
using System.IO;
using System.Runtime.CompilerServices;
using ImageMagick;

namespace TensorSharp.Models.Media.Desktop
{
    /// <summary>
    /// Puts the desktop stack on <see cref="MediaCodecs"/> before any user code runs.
    ///
    /// <para>A module initializer rather than a call every host has to remember: the server,
    /// the CLI, the benches and the tests all decode media, and one of them forgetting the
    /// call would throw the managed provider's "no video decoder" at the first upload. The
    /// whole <c>Media/Desktop/</c> folder is compiled out of the iOS target (see the csproj),
    /// so there the initializer does not exist and the app registers its own providers.</para>
    /// </summary>
    internal static class DesktopMediaRegistration
    {
        // CA2255 warns that module initializers belong in application code. This is the
        // documented exception — a library wiring its own default provider before any of
        // its callers can observe the unwired state.
#pragma warning disable CA2255
        [ModuleInitializer]
        internal static void Register() => DesktopMediaProvider.Register();
#pragma warning restore CA2255
    }

    /// <summary>
    /// The desktop media stack, exactly as before the seam existed: Magick.NET for every
    /// still-image format (HEIC via its bundled libheif, EXIF auto-orientation, Lanczos
    /// resize, PNG encode), OpenCvSharp <c>VideoCapture</c> for frame access, and ffmpeg →
    /// OpenCV <c>avc1</c> → OpenCV <c>mp4v</c> for MP4 export. Audio is not overridden: the
    /// desktop build never had a native audio decoder, so WAV/MP3/OGG stay on the managed one.
    /// </summary>
    public sealed partial class DesktopMediaProvider : IImageCodec, IVideoDecoder, IVideoEncoder
    {
        /// <summary>The instance the module initializer registers. Stateless.</summary>
        public static DesktopMediaProvider Instance { get; } = new DesktopMediaProvider();

        /// <summary>Register this provider for images, video decode and MP4 encode. Idempotent;
        /// a host that swapped in something else can call it to get the desktop stack back.</summary>
        public static void Register()
        {
            MediaCodecs.Image = Instance;
            MediaCodecs.Video = Instance;
            MediaCodecs.VideoEncoder = Instance;
        }

        // ------------------------------------------------------------------
        // IImageCodec — Magick.NET
        // ------------------------------------------------------------------

        /// <summary>ImageMagick decodes far more than the sniffer names, but the contract is
        /// "false means DecodeRgba throws", so the answer is tied to what is sniffed and
        /// <see cref="DecodeRgba"/> refuses the unrecognised rather than guessing.</summary>
        public bool CanDecode(ReadOnlySpan<byte> header) =>
            ImageFormatSniffer.Detect(header) != ImageFormat.Unknown;

        public byte[] DecodeRgba(byte[] file, out int width, out int height)
        {
            if (file == null) throw new ArgumentNullException(nameof(file));
            if (ImageFormatSniffer.Detect(file) == ImageFormat.Unknown)
            {
                throw new NotSupportedException(
                    "Unrecognised image format: the desktop image codec decodes PNG, JPEG, HEIC/HEIF, GIF, BMP, " +
                    "WebP, PSD and TIFF, identified by their magic bytes.");
            }

            try
            {
                using var image = new MagickImage(file);
                // Apply the EXIF orientation (phone photos are stored rotated with an
                // orientation tag; the reference pipelines' loaders all honor it).
                image.AutoOrient();
                if (image.ColorSpace != ColorSpace.sRGB)
                    image.ColorSpace = ColorSpace.sRGB;
                // An opaque source gets an all-255 alpha plane so the export below is
                // uniformly RGBA; a source with alpha keeps it (the vision processors
                // composite over white themselves, the generative pipelines drop it).
                if (!image.HasAlpha)
                    image.Alpha(AlphaOption.Set);

                width = (int)image.Width;
                height = (int)image.Height;

                using var pixels = image.GetPixels();
                byte[] rgba = pixels.ToByteArray(0, 0, image.Width, image.Height, "RGBA");
                if (rgba == null || rgba.Length != (long)width * height * 4)
                    throw new InvalidDataException("ImageMagick returned an unexpected pixel buffer.");
                return rgba;
            }
            catch (Exception ex) when (ex is not InvalidDataException)
            {
                throw new InvalidDataException(
                    "Failed to decode image with ImageMagick. Ensure the Magick.NET native binaries " +
                    "(with libheif support for HEIC/HEIF) are available.",
                    ex);
            }
        }

        public (int width, int height) ReadDimensions(byte[] file)
        {
            if (file == null) throw new ArgumentNullException(nameof(file));
            try
            {
                var info = new MagickImageInfo(file);
                return ((int)info.Width, (int)info.Height);
            }
            catch (Exception ex)
            {
                throw new InvalidDataException("Failed to read image dimensions with ImageMagick.", ex);
            }
        }

        public byte[] EncodePng(byte[] rgb8, int width, int height, int channels)
        {
            if (rgb8 == null) throw new ArgumentNullException(nameof(rgb8));
            if (channels != 3 && channels != 4)
                throw new ArgumentOutOfRangeException(nameof(channels), "PNG encoding takes 3 (RGB) or 4 (RGBA) channels");
            if (rgb8.Length != (long)width * height * channels)
                throw new ArgumentException($"pixel buffer {rgb8.Length} != {width}x{height}x{channels}", nameof(rgb8));

            var settings = new PixelReadSettings((uint)width, (uint)height,
                StorageType.Char, channels == 4 ? PixelMapping.RGBA : PixelMapping.RGB);
            using var image = new MagickImage(rgb8, settings);
            image.Format = MagickFormat.Png;
            // PNG quality is zlib level x 10 + row filter: 15 is level 1 with adaptive
            // filters. ImageMagick's default, 75 (level 7), spent 8.3 s on a 16 MP
            // MiniMax-H3-Image picture for 19.3 MB; level 1 takes 0.9 s for 22.0 MB (level 4,
            // 45, 2.0 s for 19.6 MB), so a large picture no longer waits on its own encode.
            image.Quality = 15;
            return image.ToByteArray();
        }

        public byte[] ResizeRgb8(byte[] rgb8, int srcW, int srcH, int dstW, int dstH, ResizeFilter filter)
        {
            if (rgb8 == null) throw new ArgumentNullException(nameof(rgb8));
            if (rgb8.Length != (long)srcW * srcH * 3)
                throw new ArgumentException($"pixel buffer {rgb8.Length} != {srcW}x{srcH}x3", nameof(rgb8));
            if (dstW <= 0 || dstH <= 0)
                throw new ArgumentException($"Invalid target size {dstW}x{dstH}");
            if (dstW == srcW && dstH == srcH)
                return rgb8;

            // Bilinear never goes native: it is the cheap filter the chat vision processors
            // define for themselves, and keeping it managed makes it identical on every platform.
            if (filter == ResizeFilter.Bilinear)
                return ManagedResampler.ResizeBilinearRgb8(rgb8, srcW, srcH, dstW, dstH);
            if (filter != ResizeFilter.Lanczos)
                throw new ArgumentOutOfRangeException(nameof(filter), filter, "unknown resize filter");

            var readSettings = new PixelReadSettings((uint)srcW, (uint)srcH, StorageType.Char, PixelMapping.RGB);
            using var image = new MagickImage(rgb8, readSettings);
            image.FilterType = FilterType.Lanczos;
            image.Resize(new MagickGeometry((uint)dstW, (uint)dstH) { IgnoreAspectRatio = true });
            image.Alpha(AlphaOption.Off);
            using var pixels = image.GetPixels();
            byte[] rgb = pixels.ToByteArray(0, 0, (uint)dstW, (uint)dstH, "RGB");
            if (rgb == null || rgb.Length != (long)dstW * dstH * 3)
                throw new InvalidDataException("ImageMagick returned an unexpected pixel buffer after resize.");
            return rgb;
        }
    }
}
