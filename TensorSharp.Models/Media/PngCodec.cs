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
using System.IO.Compression;
using StbImageSharp;

namespace TensorSharp.Models.Media
{
    /// <summary>
    /// The managed PNG codec: the hand-written decoder the vision processors have always
    /// used (8-bit, non-interlaced grey / RGB / grey+alpha / RGBA), with stb_image taking the
    /// shapes it never handled (palette, 16-bit, Adam7), and the encoder
    /// <see cref="MediaHelper"/> writes video frames with. No native code, so it is the same
    /// on the desktop and on the phone.
    /// </summary>
    internal static class PngCodec
    {
        internal readonly record struct Header(int Width, int Height, int BitDepth, int ColorType, int Interlace);

        /// <summary>Width and height from IHDR, which the spec puts first (bytes 16..23).</summary>
        internal static (int width, int height) ReadDimensions(byte[] data)
        {
            if (data.Length < 24 || !ImageFormatSniffer.IsPng(data))
                throw new InvalidDataException("Not a PNG file");

            int width = (data[16] << 24) | (data[17] << 16) | (data[18] << 8) | data[19];
            int height = (data[20] << 24) | (data[21] << 16) | (data[22] << 8) | data[23];
            return (width, height);
        }

        internal static Header ReadHeader(byte[] data)
        {
            if (data.Length < 33 || !ImageFormatSniffer.IsPng(data))
                throw new InvalidDataException("Not a PNG file");
            // IHDR is mandated to be the first chunk: length at 8, type at 12, payload at 16.
            if (data[12] != (byte)'I' || data[13] != (byte)'H' || data[14] != (byte)'D' || data[15] != (byte)'R')
                throw new InvalidDataException("PNG does not start with IHDR");
            var (w, h) = ReadDimensions(data);
            return new Header(w, h, data[24], data[25], data[28]);
        }

        /// <summary>Decode to RGBA8 as stored (no orientation applied).</summary>
        internal static byte[] Decode(byte[] data, out int width, out int height)
        {
            Header header = ReadHeader(data);
            bool simple = header.BitDepth == 8 && header.Interlace == 0 &&
                          header.ColorType is 0 or 2 or 4 or 6;
            if (simple)
                return DecodeSimple(data, out width, out height);

            // Palette (type 3), 16-bit and interlaced PNGs: stb_image expands all of them to
            // 8-bit RGBA. The hand-written path used to fall into these silently and emit
            // garbage; routing them here is the behavioural fix.
            try
            {
                ImageResult decoded = ImageResult.FromMemory(data, ColorComponents.RedGreenBlueAlpha);
                width = decoded.Width;
                height = decoded.Height;
                return decoded.Data;
            }
            catch (Exception ex)
            {
                throw new InvalidDataException(
                    $"Failed to decode PNG (bit depth {header.BitDepth}, colour type {header.ColorType}, interlace {header.Interlace}).", ex);
            }
        }

        /// <summary>The EXIF orientation carried in an <c>eXIf</c> chunk, or 1 when there is none.
        /// Rare in the wild (cameras write JPEG/HEIC), but ImageMagick honours it and so does this.</summary>
        internal static int ReadExifOrientation(byte[] data)
        {
            int pos = 8;
            while (pos + 8 <= data.Length)
            {
                int length = (data[pos] << 24) | (data[pos + 1] << 16) | (data[pos + 2] << 8) | data[pos + 3];
                if (length < 0 || pos + 12 + (long)length > data.Length)
                    return 1;
                if (data[pos + 4] == (byte)'e' && data[pos + 5] == (byte)'X' && data[pos + 6] == (byte)'I' && data[pos + 7] == (byte)'f')
                    return ExifOrientation.ReadFromTiff(data, pos + 8, length);
                if (data[pos + 4] == (byte)'I' && data[pos + 5] == (byte)'D' && data[pos + 6] == (byte)'A' && data[pos + 7] == (byte)'T')
                    return 1;   // eXIf must precede IDAT; nothing after it can carry one
                pos += 12 + length;
            }
            return 1;
        }

        /// <summary>Whether the PNG carries transparency: an alpha channel (colour types 4 and
        /// 6) or a tRNS chunk, which gives a grey, truecolour or palette image transparent
        /// entries.</summary>
        internal static bool CarriesAlpha(byte[] data) =>
            ReadHeader(data).ColorType is 4 or 6 || HasChunkBeforeImageData(data, "tRNS");

        /// <summary>Whether a chunk of <paramref name="type"/> comes before the image data, which
        /// is where every ancillary chunk that changes how the pixels read (iCCP, sRGB, gAMA,
        /// tRNS, eXIf) has to be.</summary>
        internal static bool HasChunkBeforeImageData(byte[] data, string type)
        {
            long pos = 8;
            while (pos + 8 <= data.Length)
            {
                long length = ((long)data[pos] << 24) | ((long)data[pos + 1] << 16) | ((long)data[pos + 2] << 8) | data[pos + 3];
                if (pos + 12 + length > data.Length)
                    return false;
                string name = System.Text.Encoding.ASCII.GetString(data, (int)pos + 4, 4);
                if (name == type)
                    return true;
                if (name is "IDAT" or "IEND")
                    return false;
                pos += 12 + length;
            }
            return false;
        }

        // Decoded on every image turn; a server's first screenshots otherwise run the
        // unfiltering loops as unoptimized tier-0 code (tens of milliseconds per image).
        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveOptimization)]
        private static byte[] DecodeSimple(byte[] data, out int width, out int height)
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

            // One inflate of the whole image ([filter byte | row] per row), then unfiltering in
            // place with a loop per filter type and a loop per color type: a screenshot-sized
            // image decoded row by row (a read per row, a switch per byte and per pixel) took
            // about 23 ms on the request's critical path.
            int channels = colorType switch { 0 => 1, 2 => 3, 4 => 2, 6 => 4, _ => 3 };
            int bytesPerPixel = channels * (bitDepth / 8);
            int stride = width * bytesPerPixel;
            byte[] filtered = new byte[(long)height * (stride + 1)];
            using (var deflateStream = new DeflateStream(
                new MemoryStream(idatStream.GetBuffer(), 2, (int)idatStream.Length - 2), CompressionMode.Decompress))
            {
                int read = 0;
                while (read < filtered.Length)
                {
                    int n = deflateStream.Read(filtered, read, filtered.Length - read);
                    if (n <= 0) break;
                    read += n;
                }
            }

            byte[] rawPixels = new byte[(long)height * stride];
            Unfilter(filtered, rawPixels, height, stride, bytesPerPixel);
            return ToRgba(rawPixels, width, height, colorType);
        }

        private static unsafe void Unfilter(byte[] filtered, byte[] raw, int height, int stride, int bpp)
        {
            byte[] zero = new byte[stride];
            fixed (byte* fbase = filtered, rbase = raw, zbase = zero)
            {
                for (int y = 0; y < height; y++)
                {
                    byte* src = fbase + (long)y * (stride + 1) + 1;
                    int filter = src[-1];
                    byte* cur = rbase + (long)y * stride;
                    byte* prev = y > 0 ? cur - stride : zbase;
                    int x = 0;
                    switch (filter)
                    {
                        case 1:
                            for (; x < bpp; x++) cur[x] = src[x];
                            for (; x < stride; x++) cur[x] = (byte)(src[x] + cur[x - bpp]);
                            break;
                        case 2:
                            for (; x + 16 <= stride; x += 16)
                                *(System.Runtime.Intrinsics.Vector128<byte>*)(cur + x) =
                                    *(System.Runtime.Intrinsics.Vector128<byte>*)(src + x) + *(System.Runtime.Intrinsics.Vector128<byte>*)(prev + x);
                            for (; x < stride; x++) cur[x] = (byte)(src[x] + prev[x]);
                            break;
                        case 3:
                            for (; x < bpp; x++) cur[x] = (byte)(src[x] + (prev[x] >> 1));
                            for (; x < stride; x++) cur[x] = (byte)(src[x] + ((cur[x - bpp] + prev[x]) >> 1));
                            break;
                        case 4:
                            for (; x < bpp; x++) cur[x] = (byte)(src[x] + prev[x]);   // Paeth(0, b, 0) = b
                            for (; x < stride; x++) cur[x] = (byte)(src[x] + PaethPredictor(cur[x - bpp], prev[x], prev[x - bpp]));
                            break;
                        default:
                            Buffer.MemoryCopy(src, cur, stride, stride);
                            break;
                    }
                }
            }
        }

        private static unsafe byte[] ToRgba(byte[] raw, int width, int height, int colorType)
        {
            long pixels = (long)width * height;
            byte[] rgba = new byte[pixels * 4];
            fixed (byte* s = raw, d = rgba)
            {
                switch (colorType)
                {
                    case 2: // RGB
                        for (long i = 0; i < pixels; i++)
                        {
                            d[4 * i] = s[3 * i];
                            d[4 * i + 1] = s[3 * i + 1];
                            d[4 * i + 2] = s[3 * i + 2];
                            d[4 * i + 3] = 255;
                        }
                        break;
                    case 6: // RGBA
                        Buffer.MemoryCopy(s, d, pixels * 4, pixels * 4);
                        break;
                    case 0: // Grayscale
                        for (long i = 0; i < pixels; i++)
                        {
                            d[4 * i] = d[4 * i + 1] = d[4 * i + 2] = s[i];
                            d[4 * i + 3] = 255;
                        }
                        break;
                    case 4: // Grayscale + Alpha
                        for (long i = 0; i < pixels; i++)
                        {
                            d[4 * i] = d[4 * i + 1] = d[4 * i + 2] = s[2 * i];
                            d[4 * i + 3] = s[2 * i + 1];
                        }
                        break;
                }
            }
            return rgba;
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

        // ------------------------------------------------------------------
        // Encoder
        // ------------------------------------------------------------------

        /// <summary>Scanline bytes per deflate band of <see cref="Encode"/> (at least 8 rows).
        /// Fixed by the picture's geometry rather than the core count, so the same pixels
        /// encode to the same bytes on every machine.</summary>
        private const int BandBytes = 1 << 20;

        /// <summary>zlib level of <see cref="Encode"/>. On a 4096x4096 photograph (zlib-ng,
        /// 40 cores, Up filter) level 1 gave 29.3 MiB, 2 gave 20.2 MiB, 4 gave 19.0 MiB in
        /// 42 ms and 6 gave 18.6 MiB in 74 ms; ImageMagick's level 1 with adaptive filters
        /// took 0.9 s for 21 MiB.</summary>
        private const int DeflateLevel = 4;

        /// <summary>Encode packed 8-bit pixels (3 = RGB, 4 = RGBA; row-major, unpadded) as a
        /// complete PNG file. Thread-safe: it touches nothing but its arguments.
        ///
        /// <para>Row 0 is stored unfiltered and every other row with the Up filter (one
        /// vector subtraction a row). The rows are deflated in bands of about
        /// <see cref="BandBytes"/> in parallel: every band but the last ends with a sync
        /// flush, so their concatenation is one valid deflate stream, and the bands'
        /// Adler-32s combine into the stream's (the pigz construction). Each band is its own
        /// IDAT chunk, which a decoder concatenates.</para></summary>
        internal static byte[] Encode(byte[] pixels, int width, int height, int channels)
        {
            if (pixels == null) throw new ArgumentNullException(nameof(pixels));
            if (width <= 0 || height <= 0) throw new ArgumentOutOfRangeException(nameof(width), $"invalid size {width}x{height}");
            if (channels != 3 && channels != 4)
                throw new ArgumentOutOfRangeException(nameof(channels), "PNG encoding takes 3 (RGB) or 4 (RGBA) channels");
            long expected = (long)width * height * channels;
            if (pixels.Length != expected)
                throw new ArgumentException($"pixel buffer {pixels.Length} != {width}x{height}x{channels}", nameof(pixels));

            int rowBytes = width * channels;
            int rowStride = 1 + rowBytes;
            int bandRows = Math.Max(8, BandBytes / rowStride);
            int bands = (height + bandRows - 1) / bandRows;
            var parts = new byte[bands][];
            var adlers = new uint[bands];
            var lengths = new long[bands];
            System.Threading.Tasks.Parallel.For(0, bands, i =>
            {
                int y0 = i * bandRows, y1 = Math.Min(height, y0 + bandRows);
                var rows = new byte[(long)(y1 - y0) * rowStride];
                for (int y = y0; y < y1; y++)
                    FilterUp(pixels, y, rowBytes, rows, (y - y0) * rowStride);
                adlers[i] = Adler32(rows);
                lengths[i] = rows.Length;
                using var ms = new MemoryStream(rows.Length / 2 + 64);
                if (i == 0)
                {
                    ms.WriteByte(0x78);   // zlib header: deflate, 32 KiB window, "fast" level
                    ms.WriteByte(0x5E);
                }
                using (var deflate = new DeflateStream(ms, new ZLibCompressionOptions { CompressionLevel = DeflateLevel }, leaveOpen: true))
                {
                    deflate.Write(rows, 0, rows.Length);
                    if (i < bands - 1)
                    {
                        deflate.Flush();          // sync flush: byte-aligned, not the final block
                        parts[i] = ms.ToArray();  // before Dispose writes a final block
                    }
                }
                if (i == bands - 1) parts[i] = ms.ToArray();
            });

            uint adler = adlers[0];
            for (int i = 1; i < bands; i++)
                adler = Adler32Combine(adler, adlers[i], lengths[i]);
            byte[] last = parts[bands - 1];
            Array.Resize(ref last, last.Length + 4);
            last[^4] = (byte)(adler >> 24); last[^3] = (byte)(adler >> 16);
            last[^2] = (byte)(adler >> 8);  last[^1] = (byte)adler;
            parts[bands - 1] = last;

            byte[] idat = System.Text.Encoding.ASCII.GetBytes("IDAT");
            var crcs = new uint[bands];
            System.Threading.Tasks.Parallel.For(0, bands, i => crcs[i] = Crc32Png(idat, parts[i]));

            long total = 8 + (12 + 13) + 12;
            foreach (byte[] p in parts) total += 12 + p.Length;
            using var png = new MemoryStream(checked((int)total));
            png.Write(new byte[] { 0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A }, 0, 8);
            WritePngChunk(png, "IHDR", BuildIHDR(width, height, channels));
            for (int i = 0; i < bands; i++)
                WritePngChunk(png, idat, parts[i], crcs[i]);
            WritePngChunk(png, "IEND", Array.Empty<byte>());
            return png.ToArray();
        }

        // Scanline y as [filter byte | samples] at rows[at]: None for the first row, Up
        // (sample minus the one above, mod 256) for the others.
        private static void FilterUp(byte[] pixels, int y, int rowBytes, byte[] rows, int at)
        {
            var cur = new ReadOnlySpan<byte>(pixels, y * rowBytes, rowBytes);
            var dst = new Span<byte>(rows, at + 1, rowBytes);
            if (y == 0)
            {
                rows[at] = 0;
                cur.CopyTo(dst);
                return;
            }
            rows[at] = 2;
            var prev = new ReadOnlySpan<byte>(pixels, (y - 1) * rowBytes, rowBytes);
            int x = 0, lanes = System.Numerics.Vector<byte>.Count;
            for (; x + lanes <= rowBytes; x += lanes)
                (new System.Numerics.Vector<byte>(cur.Slice(x)) - new System.Numerics.Vector<byte>(prev.Slice(x))).CopyTo(dst.Slice(x));
            for (; x < rowBytes; x++)
                dst[x] = (byte)(cur[x] - prev[x]);
        }

        /// <summary>The Adler-32 of A followed by B from theirs and B's length (zlib's
        /// adler32_combine).</summary>
        internal static uint Adler32Combine(uint adlerA, uint adlerB, long lengthB)
        {
            const uint Base = 65521;
            uint rem = (uint)(lengthB % Base);
            uint sum1 = adlerA & 0xFFFF;
            uint sum2 = (uint)((ulong)rem * sum1 % Base);
            sum1 += (adlerB & 0xFFFF) + Base - 1;
            sum2 += (adlerA >> 16) + (adlerB >> 16) + Base - rem;
            if (sum1 >= Base) sum1 -= Base;
            if (sum1 >= Base) sum1 -= Base;
            if (sum2 >= Base << 1) sum2 -= Base << 1;
            if (sum2 >= Base) sum2 -= Base;
            return sum1 | (sum2 << 16);
        }

        /// <summary>Assembles a complete PNG file from filter-prefixed scanlines (each row is one
        /// filter-type byte followed by <c>width * channels</c> samples). Thread-safe.</summary>
        internal static byte[] EncodeScanlines(byte[] rawRows, int width, int height, int channels)
        {
            byte[] compressed;
            using (var ms = new MemoryStream())
            {
                ms.WriteByte(0x78); // zlib header
                ms.WriteByte(0x01);
                using (var deflate = new DeflateStream(ms, CompressionLevel.Fastest, true))
                    deflate.Write(rawRows, 0, rawRows.Length);

                uint adler = Adler32(rawRows);
                ms.WriteByte((byte)(adler >> 24));
                ms.WriteByte((byte)(adler >> 16));
                ms.WriteByte((byte)(adler >> 8));
                ms.WriteByte((byte)adler);
                compressed = ms.ToArray();
            }

            using var png = new MemoryStream(compressed.Length + 128);
            png.Write(new byte[] { 0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A }, 0, 8);
            WritePngChunk(png, "IHDR", BuildIHDR(width, height, channels));
            WritePngChunk(png, "IDAT", compressed);
            WritePngChunk(png, "IEND", Array.Empty<byte>());
            return png.ToArray();
        }

        /// <summary>
        /// Adler-32 over the raw scanlines, for the zlib trailer.
        ///
        /// <para>Reducing modulo 65521 on every byte costs two divisions per byte, which on a
        /// 1080p frame is more time than the deflate that follows it. The sums instead run
        /// unreduced for <see cref="AdlerNmax"/> bytes — the largest block for which
        /// <c>b</c> provably cannot overflow 32 bits — and are reduced once per block, the
        /// standard zlib formulation. The result is bit-identical.</para>
        /// </summary>
        private static uint Adler32(byte[] data)
        {
            const uint Base = 65521;
            uint a = 1, b = 0;
            int offset = 0, remaining = data.Length;

            while (remaining > 0)
            {
                int block = remaining < AdlerNmax ? remaining : AdlerNmax;
                remaining -= block;
                int end = offset + block;
                for (; offset < end; offset++)
                {
                    a += data[offset];
                    b += a;
                }
                a %= Base;
                b %= Base;
            }

            return (b << 16) | a;
        }

        /// <summary>Largest byte count for which the unreduced Adler-32 <c>b</c> accumulator
        /// stays inside 32 bits; the constant zlib uses.</summary>
        private const int AdlerNmax = 5552;

        private static byte[] BuildIHDR(int width, int height, int channels)
        {
            byte[] ihdr = new byte[13];
            ihdr[0] = (byte)(width >> 24); ihdr[1] = (byte)(width >> 16);
            ihdr[2] = (byte)(width >> 8);  ihdr[3] = (byte)width;
            ihdr[4] = (byte)(height >> 24); ihdr[5] = (byte)(height >> 16);
            ihdr[6] = (byte)(height >> 8);  ihdr[7] = (byte)height;
            ihdr[8] = 8;  // bit depth
            ihdr[9] = (byte)(channels == 4 ? 6 : 2);  // color type RGBA / RGB
            return ihdr;
        }

        private static void WritePngChunk(Stream s, string type, byte[] data)
        {
            byte[] typeBuf = System.Text.Encoding.ASCII.GetBytes(type);
            WritePngChunk(s, typeBuf, data, Crc32Png(typeBuf, data));
        }

        private static void WritePngChunk(Stream s, byte[] typeBuf, byte[] data, uint crc)
        {
            byte[] lenBuf = { (byte)(data.Length >> 24), (byte)(data.Length >> 16),
                              (byte)(data.Length >> 8),  (byte)data.Length };
            s.Write(lenBuf, 0, 4);
            s.Write(typeBuf, 0, 4);
            if (data.Length > 0) s.Write(data, 0, data.Length);

            byte[] crcBuf = { (byte)(crc >> 24), (byte)(crc >> 16),
                              (byte)(crc >> 8),  (byte)crc };
            s.Write(crcBuf, 0, 4);
        }

        private static readonly uint[] Crc32Table = BuildCrc32Table();
        private static uint[] BuildCrc32Table()
        {
            var t = new uint[256];
            for (uint n = 0; n < 256; n++)
            {
                uint c = n;
                for (int k = 0; k < 8; k++)
                    c = (c & 1) != 0 ? 0xEDB88320 ^ (c >> 1) : c >> 1;
                t[n] = c;
            }
            return t;
        }

        private static uint Crc32Png(byte[] type, byte[] data)
        {
            uint crc = 0xFFFFFFFF;
            foreach (byte b in type)
                crc = Crc32Table[(crc ^ b) & 0xFF] ^ (crc >> 8);
            foreach (byte b in data)
                crc = Crc32Table[(crc ^ b) & 0xFF] ^ (crc >> 8);
            return crc ^ 0xFFFFFFFF;
        }

        /// <summary>CRC over a chunk type + payload, exposed so a test can synthesise ancillary
        /// chunks (an <c>eXIf</c> with a known orientation) that decoders will accept.</summary>
        internal static uint ChunkCrc(string type, byte[] data) =>
            Crc32Png(System.Text.Encoding.ASCII.GetBytes(type), data);
    }
}
