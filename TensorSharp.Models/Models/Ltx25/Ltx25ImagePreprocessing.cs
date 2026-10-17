// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System;
using System.Threading;
using TensorSharp.Models.Media;
using TensorSharp.Models.QwenImage;

namespace TensorSharp.Models.Ltx25;

/// <summary>The publisher's LTX 2.5 temporal-image preprocessing. Compression precedes
/// each stage's float bilinear resize; cropping happens in the resized coordinate system.</summary>
internal static class Ltx25ImagePreprocessing
{
    internal const int ImageCrf = 18;

    internal static RgbImage PrepareSource(RgbImage image, CancellationToken cancellationToken = default,
        IVideoEncoder encoder = null)
    {
        ArgumentNullException.ThrowIfNull(image);
        cancellationToken.ThrowIfCancellationRequested();
        if (image.Width <= 0 || image.Height <= 0) throw new ArgumentException("An image must have positive dimensions.", nameof(image));
        // The official codec clips the bottom/right odd edge before yuv420p encoding.
        // A one-pixel dimension bypasses compression in the publisher's preprocess().
        if (image.Width < 2 || image.Height < 2) return image;
        encoder ??= MediaCodecs.VideoEncoder;
        if (!encoder.SupportsH264ImageConditioning)
            throw new NotSupportedException($"{encoder.GetType().Name} does not support LTX temporal image conditioning. " +
                "It requires the desktop FFmpeg/libx264 image transform; install FFmpeg on PATH or set TS_FFMPEG.");
        int width = image.Width / 2 * 2, height = image.Height / 2 * 2;
        var rgb = new byte[checked(width * height * 3)];
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width * 3; ++x)
                rgb[y * width * 3 + x] = (byte)Math.Clamp(image.Pixels[y * image.Width * 3 + x] * 255f + .5f, 0, 255);
        byte[] compressed = encoder.RecompressH264Image(rgb, width, height, ImageCrf, cancellationToken);
        if (compressed.Length != rgb.Length) throw new InvalidOperationException("The LTX image-conditioning codec returned an unexpected RGB buffer size.");
        var pixels = new float[compressed.Length];
        for (int i = 0; i < pixels.Length; ++i) pixels[i] = compressed[i] / 255f;
        return new RgbImage(width, height, pixels);
    }

    internal static RgbImage ResizeAndCenterCrop(RgbImage image, int width, int height)
    {
        ArgumentNullException.ThrowIfNull(image);
        if (image.Width <= 0 || image.Height <= 0 || width <= 0 || height <= 0)
            throw new ArgumentException("Source and target image dimensions must be positive.");
        double scale = Math.Max((double)height / image.Height, (double)width / image.Width);
        int resizedHeight = checked((int)Math.Ceiling(image.Height * scale));
        int resizedWidth = checked((int)Math.Ceiling(image.Width * scale));
        int top = (resizedHeight - height) / 2, left = (resizedWidth - width) / 2;
        float yScale = (float)((double)image.Height / resizedHeight), xScale = (float)((double)image.Width / resizedWidth);
        var result = new float[checked(width * height * 3)];
        // Evaluate only the crop's pixels, using the coordinates and weights of the
        // entire resized image. No RGB8 round trip: interpolate in [0,255] Float32
        // as torch.interpolate does, then retain [0,1] for EncodeImage's normalization.
        for (int y = 0; y < height; ++y)
        {
            float sy = Math.Max(0, (y + top + .5f) * yScale - .5f);
            int y0 = Math.Min((int)sy, image.Height - 1), y1 = Math.Min(y0 + 1, image.Height - 1);
            float dy = sy - y0;
            for (int x = 0; x < width; ++x)
            {
                float sx = Math.Max(0, (x + left + .5f) * xScale - .5f);
                int x0 = Math.Min((int)sx, image.Width - 1), x1 = Math.Min(x0 + 1, image.Width - 1);
                float dx = sx - x0;
                for (int channel = 0; channel < 3; ++channel)
                {
                    float a = image.Pixels[(y0 * image.Width + x0) * 3 + channel] * 255f;
                    float b = image.Pixels[(y0 * image.Width + x1) * 3 + channel] * 255f;
                    float c = image.Pixels[(y1 * image.Width + x0) * 3 + channel] * 255f;
                    float d = image.Pixels[(y1 * image.Width + x1) * 3 + channel] * 255f;
                    result[(y * width + x) * 3 + channel] = ((1 - dy) * ((1 - dx) * a + dx * b) + dy * ((1 - dx) * c + dx * d)) / 255f;
                }
            }
        }
        return new RgbImage(width, height, result);
    }
}
