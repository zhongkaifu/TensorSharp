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
using StbImageSharp;

namespace InferenceEngine
{
    public class Gemma3ImageProcessor
    {
        public int TokensPerImage { get; }
        public int ImageSize { get; }
        public int PatchSize { get; }

        public const int StartOfImageToken = 255999;
        public const int EndOfImageToken = 256000;
        public const int NewlineNewlineToken = 108;
        public const int PadToken = 0;

        private const float ImageMean = 0.5f;
        private const float ImageStd = 0.5f;

        public Gemma3ImageProcessor(int tokensPerImage = 256, int imageSize = 896, int patchSize = 14)
        {
            TokensPerImage = tokensPerImage;
            ImageSize = imageSize;
            PatchSize = patchSize;
        }

        /// <summary>
        /// Process an image file into normalized pixel values in channel-first format [C, H, W].
        /// Matches Ollama's processing: composite over white, bilinear resize, normalize with mean=0.5, std=0.5.
        /// </summary>
        public float[] ProcessImage(string imagePath)
        {
            byte[] fileBytes = File.ReadAllBytes(imagePath);
            int origWidth, origHeight;
            byte[] rgba = DecodeImageToRGBA(fileBytes, out origWidth, out origHeight);

            byte[] composited = CompositeOverWhite(rgba, origWidth, origHeight);

            byte[] resized = BilinearResize(composited, origWidth, origHeight, ImageSize, ImageSize);

            return PackChannelFirst(resized, ImageSize, ImageSize);
        }

        internal static byte[] DecodeImageToRGBA(byte[] fileBytes, out int width, out int height)
        {
            if (IsPng(fileBytes))
                return DecodePNG(fileBytes, out width, out height);

            if (IsJpeg(fileBytes))
                return DecodeJPEG(fileBytes, out width, out height);

            throw new NotSupportedException("Only PNG and JPEG image formats are supported");
        }

        internal static (int width, int height) ReadImageDimensions(string imagePath)
        {
            byte[] fileBytes = File.ReadAllBytes(imagePath);

            if (IsPng(fileBytes))
                return ReadPngDimensions(fileBytes);

            if (IsJpeg(fileBytes))
            {
                DecodeJPEG(fileBytes, out int width, out int height);
                return (width, height);
            }

            throw new NotSupportedException("Only PNG and JPEG image formats are supported");
        }

        private static bool IsPng(byte[] fileBytes) =>
            fileBytes.Length >= 8 &&
            fileBytes[0] == 0x89 &&
            fileBytes[1] == 0x50 &&
            fileBytes[2] == 0x4E &&
            fileBytes[3] == 0x47;

        private static bool IsJpeg(byte[] fileBytes) =>
            fileBytes.Length >= 2 &&
            fileBytes[0] == 0xFF &&
            fileBytes[1] == 0xD8;

        private static byte[] DecodePNG(byte[] data, out int width, out int height)
        {
            using var ms = new MemoryStream(data);
            using var reader = new BinaryReader(ms);

            byte[] sig = reader.ReadBytes(8);
            if (sig[0] != 0x89 || sig[1] != 0x50 || sig[2] != 0x4E || sig[3] != 0x47)
                throw new InvalidDataException("Not a PNG file");

            width = 0; height = 0;
            int bitDepth = 0, colorType = 0;
            using var idatStream = new MemoryStream();

            while (ms.Position < ms.Length)
            {
                int length = ReadBigEndianInt32(reader);
                byte[] chunkType = reader.ReadBytes(4);
                string type = System.Text.Encoding.ASCII.GetString(chunkType);

                if (type == "IHDR")
                {
                    width = ReadBigEndianInt32(reader);
                    height = ReadBigEndianInt32(reader);
                    bitDepth = reader.ReadByte();
                    colorType = reader.ReadByte();
                    reader.ReadBytes(3 + 4); // compression, filter, interlace + CRC
                }
                else if (type == "IDAT")
                {
                    byte[] idatData = reader.ReadBytes(length);
                    idatStream.Write(idatData, 0, idatData.Length);
                    reader.ReadBytes(4); // CRC
                }
                else if (type == "IEND")
                {
                    break;
                }
                else
                {
                    reader.ReadBytes(length + 4);
                }
            }

            idatStream.Position = 0;
            using var deflateStream = new System.IO.Compression.DeflateStream(
                new MemoryStream(idatStream.ToArray(), 2, (int)idatStream.Length - 2),
                System.IO.Compression.CompressionMode.Decompress);

            int channels = colorType switch { 0 => 1, 2 => 3, 4 => 2, 6 => 4, _ => 3 };
            int bytesPerPixel = channels * (bitDepth / 8);
            int stride = width * bytesPerPixel;
            byte[] rawPixels = new byte[height * stride];
            byte[] prevRow = new byte[stride];

            for (int y = 0; y < height; y++)
            {
                int filterByte = deflateStream.ReadByte();
                byte[] row = new byte[stride];
                int read = 0;
                while (read < stride)
                {
                    int n = deflateStream.Read(row, read, stride - read);
                    if (n <= 0) break;
                    read += n;
                }

                for (int x = 0; x < stride; x++)
                {
                    byte a = x >= bytesPerPixel ? row[x - bytesPerPixel] : (byte)0;
                    byte b = prevRow[x];
                    byte c = x >= bytesPerPixel ? prevRow[x - bytesPerPixel] : (byte)0;

                    row[x] = filterByte switch
                    {
                        0 => row[x],
                        1 => (byte)(row[x] + a),
                        2 => (byte)(row[x] + b),
                        3 => (byte)(row[x] + (a + b) / 2),
                        4 => (byte)(row[x] + PaethPredictor(a, b, c)),
                        _ => row[x],
                    };
                }

                Buffer.BlockCopy(row, 0, rawPixels, y * stride, stride);
                Buffer.BlockCopy(row, 0, prevRow, 0, stride);
            }

            byte[] rgba = new byte[width * height * 4];
            for (int i = 0; i < width * height; i++)
            {
                switch (colorType)
                {
                    case 2: // RGB
                        rgba[i * 4] = rawPixels[i * 3];
                        rgba[i * 4 + 1] = rawPixels[i * 3 + 1];
                        rgba[i * 4 + 2] = rawPixels[i * 3 + 2];
                        rgba[i * 4 + 3] = 255;
                        break;
                    case 6: // RGBA
                        rgba[i * 4] = rawPixels[i * 4];
                        rgba[i * 4 + 1] = rawPixels[i * 4 + 1];
                        rgba[i * 4 + 2] = rawPixels[i * 4 + 2];
                        rgba[i * 4 + 3] = rawPixels[i * 4 + 3];
                        break;
                    case 0: // Grayscale
                        rgba[i * 4] = rgba[i * 4 + 1] = rgba[i * 4 + 2] = rawPixels[i];
                        rgba[i * 4 + 3] = 255;
                        break;
                    case 4: // Grayscale + Alpha
                        rgba[i * 4] = rgba[i * 4 + 1] = rgba[i * 4 + 2] = rawPixels[i * 2];
                        rgba[i * 4 + 3] = rawPixels[i * 2 + 1];
                        break;
                }
            }

            return rgba;
        }

        private static (int width, int height) ReadPngDimensions(byte[] data)
        {
            if (data.Length < 24 || !IsPng(data))
                throw new InvalidDataException("Not a PNG file");

            int width = (data[16] << 24) | (data[17] << 16) | (data[18] << 8) | data[19];
            int height = (data[20] << 24) | (data[21] << 16) | (data[22] << 8) | data[23];
            return (width, height);
        }

        private static byte PaethPredictor(byte a, byte b, byte c)
        {
            int p = a + b - c;
            int pa = Math.Abs(p - a);
            int pb = Math.Abs(p - b);
            int pc = Math.Abs(p - c);
            return pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
        }

        private static int ReadBigEndianInt32(BinaryReader reader)
        {
            byte[] bytes = reader.ReadBytes(4);
            return (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
        }

        private static byte[] DecodeJPEG(byte[] data, out int width, out int height)
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
                throw new InvalidDataException("Failed to decode JPEG image.", ex);
            }
        }

        internal static byte[] CompositeOverWhite(byte[] rgba, int width, int height)
        {
            byte[] result = new byte[width * height * 4];
            for (int i = 0; i < width * height; i++)
            {
                int a = rgba[i * 4 + 3];
                if (a == 255)
                {
                    result[i * 4] = rgba[i * 4];
                    result[i * 4 + 1] = rgba[i * 4 + 1];
                    result[i * 4 + 2] = rgba[i * 4 + 2];
                }
                else
                {
                    float alpha = a / 255f;
                    result[i * 4] = (byte)(rgba[i * 4] * alpha + 255 * (1 - alpha));
                    result[i * 4 + 1] = (byte)(rgba[i * 4 + 1] * alpha + 255 * (1 - alpha));
                    result[i * 4 + 2] = (byte)(rgba[i * 4 + 2] * alpha + 255 * (1 - alpha));
                }
                result[i * 4 + 3] = 255;
            }
            return result;
        }

        internal static byte[] BilinearResize(byte[] rgba, int srcW, int srcH, int dstW, int dstH)
        {
            byte[] result = new byte[dstW * dstH * 4];
            double xRatio = (double)srcW / dstW;
            double yRatio = (double)srcH / dstH;

            for (int dy = 0; dy < dstH; dy++)
            {
                double srcY = (dy + 0.5) * yRatio - 0.5;
                int y0 = Math.Max(0, (int)srcY);
                int y1 = Math.Min(srcH - 1, y0 + 1);
                double fy = srcY - y0;

                for (int dx = 0; dx < dstW; dx++)
                {
                    double srcX = (dx + 0.5) * xRatio - 0.5;
                    int x0 = Math.Max(0, (int)srcX);
                    int x1 = Math.Min(srcW - 1, x0 + 1);
                    double fx = srcX - x0;

                    for (int c = 0; c < 3; c++)
                    {
                        double v00 = rgba[(y0 * srcW + x0) * 4 + c];
                        double v01 = rgba[(y0 * srcW + x1) * 4 + c];
                        double v10 = rgba[(y1 * srcW + x0) * 4 + c];
                        double v11 = rgba[(y1 * srcW + x1) * 4 + c];

                        double v = v00 * (1 - fx) * (1 - fy) + v01 * fx * (1 - fy) +
                                   v10 * (1 - fx) * fy + v11 * fx * fy;
                        result[(dy * dstW + dx) * 4 + c] = (byte)Math.Clamp(v + 0.5, 0, 255);
                    }
                    result[(dy * dstW + dx) * 4 + 3] = 255;
                }
            }

            return result;
        }

        /// <summary>
        /// Pack RGBA pixels into channel-first float format [R..., G..., B...] normalized with mean/std.
        /// Matches Ollama's pack(): channel-first with (pixel/255 - mean) / std.
        /// </summary>
        private float[] PackChannelFirst(byte[] rgba, int width, int height)
        {
            int pixels = width * height;
            float[] result = new float[3 * pixels];

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    int pixIdx = y * width + x;
                    float r = (rgba[pixIdx * 4] / 255f - ImageMean) / ImageStd;
                    float g = (rgba[pixIdx * 4 + 1] / 255f - ImageMean) / ImageStd;
                    float b = (rgba[pixIdx * 4 + 2] / 255f - ImageMean) / ImageStd;

                    result[pixIdx] = r;
                    result[pixels + pixIdx] = g;
                    result[2 * pixels + pixIdx] = b;
                }
            }

            return result;
        }
    }
}
