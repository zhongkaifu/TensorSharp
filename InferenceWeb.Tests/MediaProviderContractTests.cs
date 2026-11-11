// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using ImageMagick;
using ImageMagick.Drawing;
using StbImageSharp;
using TensorSharp.Models.Media;
using TensorSharp.Models.Media.Desktop;
using TensorSharp.Models.QwenImage;
using TensorSharp.Models.Video;
using TensorSharp.Models.WanVideo;

namespace InferenceWeb.Tests;

/// <summary>
/// The contract behind <see cref="MediaCodecs"/>, pinned against the managed default so the
/// iOS provider can be validated in the simulator against the same expectations, and against
/// the desktop provider so the seam demonstrably changed nothing there.
///
/// <para>Where two decoders are compared, PNG parity is exact (lossless either way) and JPEG
/// parity allows a few levels per channel: stb_image and libjpeg-turbo use different IDCT
/// and upsampling arithmetic, which is codec noise, not a contract violation.</para>
/// </summary>
[Collection(MediaCodecSwaps.Name)]
public class MediaProviderContractTests : IDisposable
{
    private readonly string _dir;

    public MediaProviderContractTests()
    {
        _dir = Path.Combine(Path.GetTempPath(), "ts-media-seam-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_dir);
    }

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { /* best effort */ }
    }

    private static ManagedMediaProvider Managed => ManagedMediaProvider.Instance;
    private static DesktopMediaProvider Desktop => DesktopMediaProvider.Instance;

    // ---- images: PNG -----------------------------------------------------------------

    [Fact]
    public void ManagedPng_RoundTripsExactly_AndAgreesWithTheVisionPathAndMagick()
    {
        const int w = 7, h = 5;
        byte[] rgba = Gradient(w, h, channels: 4);

        byte[] png = Managed.EncodePng(rgba, w, h, 4);
        Assert.True(Managed.CanDecode(png));

        byte[] back = Managed.DecodeRgba(png, out int bw, out int bh);
        Assert.Equal((w, h), (bw, bh));
        Assert.Equal(rgba, back);
        Assert.Equal((w, h), Managed.ReadDimensions(png));

        // The chat vision processors decode PNG on the managed path: same bytes.
        Assert.Equal(rgba, ImageProcessorUtils.DecodeImageToRGBA(png, out _, out _));

        // The old path (Magick.NET) sees the identical image.
        Assert.Equal(rgba, Desktop.DecodeRgba(png, out int dw, out int dh));
        Assert.Equal((w, h), (dw, dh));
        Assert.Equal((w, h), Desktop.ReadDimensions(png));
    }

    [Theory]
    [InlineData(3)]
    [InlineData(4)]
    public void ManagedPng_SeveralDeflateBands_RoundTripExactly(int channels)
    {
        // About 1 MiB of scanlines a band: 300 pixels a row and 4,000 rows make three or four
        // bands, each its own IDAT chunk, every band but the last ending in a sync flush.
        const int w = 300, h = 4000;
        var rng = new Random(7);
        byte[] pixels = new byte[w * h * channels];
        for (int i = 0; i < pixels.Length; i++)
            pixels[i] = (byte)((i / channels % w) + (i / (w * channels)) * 3 + rng.Next(8));
        byte[] png = Managed.EncodePng(pixels, w, h, channels);

        Assert.True(CountChunks(png, "IDAT") >= 3);
        byte[] rgba = Managed.DecodeRgba(png, out int dw, out int dh);
        Assert.Equal((w, h), (dw, dh));
        ImageResult stb = ImageResult.FromMemory(png, ColorComponents.RedGreenBlueAlpha);
        Assert.Equal(rgba, stb.Data);
        for (int p = 0; p < w * h; p++)
            for (int c = 0; c < 4; c++)
                Assert.Equal(c < channels ? pixels[p * channels + c] : (byte)255, rgba[p * 4 + c]);

        // The same pixels give the same file: the bands follow the geometry, not the cores.
        Assert.Equal(png, Managed.EncodePng(pixels, w, h, channels));
    }

    [Fact]
    public void Adler32Combine_MatchesTheChecksumOfTheConcatenation()
    {
        var rng = new Random(3);
        byte[] a = new byte[70_001], b = new byte[123_457];
        rng.NextBytes(a);
        rng.NextBytes(b);
        Assert.Equal(Adler32(a.Concat(b).ToArray()), PngCodec.Adler32Combine(Adler32(a), Adler32(b), b.Length));
        Assert.Equal(Adler32(a), PngCodec.Adler32Combine(Adler32(a), 1, 0));
    }

    private static uint Adler32(byte[] data)
    {
        uint s1 = 1, s2 = 0;
        foreach (byte x in data) { s1 = (s1 + x) % 65521; s2 = (s2 + s1) % 65521; }
        return (s2 << 16) | s1;
    }

    private static int CountChunks(byte[] png, string type)
    {
        int count = 0;
        for (int at = 8; at + 8 <= png.Length;)
        {
            int length = (png[at] << 24) | (png[at + 1] << 16) | (png[at + 2] << 8) | png[at + 3];
            if (System.Text.Encoding.ASCII.GetString(png, at + 4, 4) == type) count++;
            at += 12 + length;
        }
        return count;
    }

    [Fact]
    public void ManagedPng_EncodesRgbAsOpaque_AndIsReadableByThirdPartyDecoders()
    {
        const int w = 9, h = 4;
        byte[] rgb = Gradient(w, h, channels: 3);
        byte[] png = Managed.EncodePng(rgb, w, h, 3);

        byte[] rgba = Managed.DecodeRgba(png, out _, out _);
        for (int i = 0; i < w * h; i++)
        {
            Assert.Equal(rgb[i * 3], rgba[i * 4]);
            Assert.Equal(rgb[i * 3 + 1], rgba[i * 4 + 1]);
            Assert.Equal(rgb[i * 3 + 2], rgba[i * 4 + 2]);
            Assert.Equal(255, rgba[i * 4 + 3]);
        }

        // stb_image (what llama.cpp uses) and ImageMagick both accept the file as written.
        ImageResult stb = ImageResult.FromMemory(png, ColorComponents.RedGreenBlueAlpha);
        Assert.Equal(rgba, stb.Data);
        using var magick = new MagickImage(png);
        Assert.Equal((uint)w, magick.Width);
        Assert.Equal((uint)h, magick.Height);

        // The generic encoder is the same one video frames go through.
        Assert.Throws<ArgumentOutOfRangeException>(() => Managed.EncodePng(rgb, w, h, 2));
        Assert.Throws<ArgumentException>(() => Managed.EncodePng(rgb, w + 1, h, 3));
    }

    [Fact]
    public void ManagedPng_ReadsPaletteImagesThroughStb_MatchingMagick()
    {
        // The hand-written decoder never handled colour type 3; a palette PNG used to come out
        // as garbage. PNG8 forces ImageMagick to write one.
        using var image = new MagickImage(MagickColors.Orange, 6, 3);
        image.Format = MagickFormat.Png8;
        byte[] png = image.ToByteArray();
        Assert.Equal(3, PngCodec.ReadHeader(png).ColorType);

        byte[] managed = Managed.DecodeRgba(png, out int mw, out int mh);
        byte[] magick = Desktop.DecodeRgba(png, out int dw, out int dh);
        Assert.Equal((6, 3), (mw, mh));
        Assert.Equal((dw, dh), (mw, mh));
        Assert.Equal(magick, managed);
        Assert.Equal(magick, ImageProcessorUtils.DecodeImageToRGBA(png, out _, out _));
    }

    [Fact]
    public void Png_TransparentPixelsKeepTheirStoredColour_AndTransparencyIsDetected()
    {
        // The generative pipelines drop alpha and read the colour as stored (ImageIO.Decode),
        // so the colour under alpha 0 or a low alpha is what a model sees. Magick keeps it and
        // so does the managed codec - which is why the Apple provider reads a PNG with
        // transparency through the latter instead of drawing it premultiplied, where a
        // mostly transparent picture reached MiniMax-H3 as black blotches.
        byte[] rgba = { 40, 120, 230, 0, 40, 120, 230, 1, 40, 120, 230, 8, 200, 10, 10, 255 };
        byte[] png = Managed.EncodePng(rgba, 4, 1, 4);
        Assert.Equal(rgba, Managed.DecodeRgba(png, out _, out _));
        Assert.Equal(rgba, Desktop.DecodeRgba(png, out _, out _));

        Assert.True(PngCodec.CarriesAlpha(png));
        Assert.False(PngCodec.CarriesAlpha(Managed.EncodePng(new byte[] { 1, 2, 3, 4, 5, 6 }, 2, 1, 3)));

        // A colour profile is found where the spec puts it - before the image data - and only there.
        Assert.False(PngCodec.HasChunkBeforeImageData(png, "iCCP"));
        Assert.True(PngCodec.HasChunkBeforeImageData(InsertAfterHeader(png, "iCCP", new byte[] { 0x73, 0x52, 0x47, 0x42, 0, 0 }), "iCCP"));
        Assert.True(PngCodec.HasChunkBeforeImageData(InsertAfterHeader(png, "tRNS", new byte[] { 0, 0 }), "tRNS"));
    }

    /// <summary>A copy of <paramref name="png"/> with one more chunk right after IHDR.</summary>
    private static byte[] InsertAfterHeader(byte[] png, string type, byte[] payload)
    {
        const int afterHeader = 8 + 12 + 13;   // signature + IHDR (length, type, 13 bytes, CRC)
        var chunk = new byte[12 + payload.Length];
        chunk[0] = (byte)(payload.Length >> 24); chunk[1] = (byte)(payload.Length >> 16);
        chunk[2] = (byte)(payload.Length >> 8); chunk[3] = (byte)payload.Length;
        System.Text.Encoding.ASCII.GetBytes(type).CopyTo(chunk, 4);
        payload.CopyTo(chunk, 8);
        uint crc = PngCodec.ChunkCrc(type, payload);
        chunk[^4] = (byte)(crc >> 24); chunk[^3] = (byte)(crc >> 16); chunk[^2] = (byte)(crc >> 8); chunk[^1] = (byte)crc;
        return png[..afterHeader].Concat(chunk).Concat(png[afterHeader..]).ToArray();
    }

    // ---- images: JPEG and EXIF -------------------------------------------------------

    [Fact]
    public void ManagedJpeg_AgreesWithMagickWithinCodecTolerance()
    {
        byte[] jpeg = Convert.FromBase64String(EmbeddedJpegBase64);
        byte[] managed = Managed.DecodeRgba(jpeg, out int mw, out int mh);
        byte[] magick = Desktop.DecodeRgba(jpeg, out int dw, out int dh);

        Assert.Equal((2, 2), (mw, mh));
        Assert.Equal((mw, mh), (dw, dh));
        Assert.Equal((2, 2), Managed.ReadDimensions(jpeg));
        AssertClose(magick, managed, tolerance: 4);

        // The vision processors' JPEG path is this exact decoder.
        Assert.Equal(managed, ImageProcessorUtils.DecodeImageToRGBA(jpeg, out _, out _));
    }

    [Fact]
    public void ManagedDecode_AppliesExifOrientation_LikeMagickAutoOrient()
    {
        // 8x4: left half white, right half black; stored with orientation 6 (rotate 90 CW to
        // view), so upright it is 4x8 with the white half on top.
        byte[] jpeg = InjectJpegOrientation(WhiteLeftBlackRightJpeg(8, 4), orientation: 6);
        Assert.Equal(6, ExifOrientation.ReadFromJpeg(jpeg));

        byte[] managed = Managed.DecodeRgba(jpeg, out int mw, out int mh);
        Assert.Equal((4, 8), (mw, mh));
        for (int y = 0; y < 8; y++)
            for (int x = 0; x < 4; x++)
            {
                int p = (y * 4 + x) * 4;
                if (y < 4) Assert.True(managed[p] > 200 && managed[p + 1] > 200 && managed[p + 2] > 200, $"({x},{y}) should be white");
                else Assert.True(managed[p] < 55 && managed[p + 1] < 55 && managed[p + 2] < 55, $"({x},{y}) should be black");
            }

        byte[] magick = Desktop.DecodeRgba(jpeg, out int dw, out int dh);
        Assert.Equal((4, 8), (dw, dh));
        AssertClose(magick, managed, tolerance: 4);

        // Stored dimensions are what ReadDimensions reports, on both providers.
        Assert.Equal((8, 4), Managed.ReadDimensions(jpeg));
        Assert.Equal((8, 4), Desktop.ReadDimensions(jpeg));

        // The chat vision path deliberately does NOT orient JPEG (its references don't).
        ImageProcessorUtils.DecodeImageToRGBA(jpeg, out int vw, out int vh);
        Assert.Equal((8, 4), (vw, vh));
    }

    [Fact]
    public void ManagedDecode_AppliesEveryExifOrientation_ToPng()
    {
        // A 4x2 image whose pixels encode their own coordinates, so each transform can be
        // checked against the EXIF definition directly.
        const int w = 4, h = 2;
        byte[] rgba = new byte[w * h * 4];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                int p = (y * w + x) * 4;
                rgba[p] = (byte)x; rgba[p + 1] = (byte)y; rgba[p + 2] = 0; rgba[p + 3] = 255;
            }
        byte[] plain = Managed.EncodePng(rgba, w, h, 4);

        for (int orientation = 1; orientation <= 8; orientation++)
        {
            byte[] png = orientation == 1 ? plain : InjectPngOrientation(plain, orientation);
            Assert.Equal(orientation, PngCodec.ReadExifOrientation(png));

            byte[] outp = Managed.DecodeRgba(png, out int ow, out int oh);
            bool transposed = orientation >= 5;
            Assert.Equal(transposed ? (h, w) : (w, h), (ow, oh));

            for (int y = 0; y < oh; y++)
                for (int x = 0; x < ow; x++)
                {
                    (int sx, int sy) = orientation switch
                    {
                        1 => (x, y),
                        2 => (w - 1 - x, y),
                        3 => (w - 1 - x, h - 1 - y),
                        4 => (x, h - 1 - y),
                        5 => (y, x),
                        6 => (y, h - 1 - x),
                        7 => (w - 1 - y, h - 1 - x),
                        _ => (w - 1 - y, x),
                    };
                    int p = (y * ow + x) * 4;
                    Assert.True(outp[p] == sx && outp[p + 1] == sy,
                        $"orientation {orientation}: output ({x},{y}) came from ({outp[p]},{outp[p + 1]}), expected ({sx},{sy})");
                }
        }
    }

    // ---- images: resize --------------------------------------------------------------

    [Fact]
    public void ManagedLanczos_IsTheMuseGlimmerPort_AndCloseToMagick()
    {
        const int sw = 64, sh = 48, dw = 32, dh = 24;
        byte[] rgb = Gradient(sw, sh, channels: 3);

        byte[] managed = Managed.ResizeRgb8(rgb, sw, sh, dw, dh, ResizeFilter.Lanczos);
        Assert.Equal(dw * dh * 3, managed.Length);
        Assert.Equal(MuseGlimmerImageProcessor.ResizeLanczosPillow(rgb, sw, sh, dw, dh), managed);

        // Two Lanczos-3 implementations (Pillow fixed-point vs ImageMagick float) on a smooth
        // gradient: the same filter up to rounding.
        byte[] magick = Desktop.ResizeRgb8(rgb, sw, sh, dw, dh, ResizeFilter.Lanczos);
        AssertClose(magick, managed, tolerance: 8);

        // Identity and flat colour are exact on both.
        Assert.Same(rgb, Managed.ResizeRgb8(rgb, sw, sh, sw, sh, ResizeFilter.Lanczos));
        Assert.Same(rgb, Desktop.ResizeRgb8(rgb, sw, sh, sw, sh, ResizeFilter.Lanczos));
        byte[] flat = Enumerable.Repeat((byte)137, sw * sh * 3).ToArray();
        Assert.All(Managed.ResizeRgb8(flat, sw, sh, dw, dh, ResizeFilter.Lanczos), b => Assert.Equal(137, b));
        Assert.All(Desktop.ResizeRgb8(flat, sw, sh, dw, dh, ResizeFilter.Lanczos), b => Assert.Equal(137, b));
    }

    [Fact]
    public void Bilinear_IsManagedOnEveryProvider_AndMatchesTheRgbaHelper()
    {
        const int sw = 20, sh = 12, dw = 13, dh = 7;
        byte[] rgb = Gradient(sw, sh, channels: 3);
        byte[] rgba = new byte[sw * sh * 4];
        for (int i = 0; i < sw * sh; i++)
        {
            rgba[i * 4] = rgb[i * 3]; rgba[i * 4 + 1] = rgb[i * 3 + 1]; rgba[i * 4 + 2] = rgb[i * 3 + 2]; rgba[i * 4 + 3] = 255;
        }

        byte[] expectedRgba = ImageProcessorUtils.BilinearResize(rgba, sw, sh, dw, dh);
        byte[] expected = new byte[dw * dh * 3];
        for (int i = 0; i < dw * dh; i++)
        {
            expected[i * 3] = expectedRgba[i * 4]; expected[i * 3 + 1] = expectedRgba[i * 4 + 1]; expected[i * 3 + 2] = expectedRgba[i * 4 + 2];
        }

        Assert.Equal(expected, Managed.ResizeRgb8(rgb, sw, sh, dw, dh, ResizeFilter.Bilinear));
        Assert.Equal(expected, Desktop.ResizeRgb8(rgb, sw, sh, dw, dh, ResizeFilter.Bilinear));
    }

    // ---- images: refusals ------------------------------------------------------------

    [Fact]
    public void ManagedImages_RefuseHeicAndUnknownFormats_NamingTheProviderThatWould()
    {
        byte[] heic = HeicHeader();
        Assert.False(Managed.CanDecode(heic));
        var ex = Assert.Throws<NotSupportedException>(() => Managed.DecodeRgba(heic, out _, out _));
        Assert.Contains("ImageIO", ex.Message);
        Assert.Contains("MediaCodecs.Image", ex.Message);
        Assert.Throws<NotSupportedException>(() => Managed.ReadDimensions(heic));

        byte[] webp = "RIFF\0\0\0\0WEBPVP8 "u8.ToArray();
        Assert.False(Managed.CanDecode(webp));
        Assert.Contains("MediaCodecs.Image", Assert.Throws<NotSupportedException>(() => Managed.DecodeRgba(webp, out _, out _)).Message);

        byte[] junk = Enumerable.Range(0, 64).Select(i => (byte)(i * 37)).ToArray();
        Assert.False(Managed.CanDecode(junk));
        Assert.Throws<NotSupportedException>(() => Managed.DecodeRgba(junk, out _, out _));

        // HEIC goes through the seam from the vision path, so with the managed default in
        // place the same actionable error surfaces there.
        using (new CodecScope(image: Managed))
        {
            var visionEx = Assert.Throws<NotSupportedException>(() => ImageProcessorUtils.DecodeImageToRGBA(heic, out _, out _));
            Assert.Contains("ImageIO", visionEx.Message);
        }

        // The desktop provider recognises HEIC (and would decode it via libheif).
        Assert.True(Desktop.CanDecode(heic));
        Assert.False(Desktop.CanDecode(junk));
        Assert.Throws<NotSupportedException>(() => Desktop.DecodeRgba(junk, out _, out _));
    }

    // ---- ImageIO (generative pipelines) over the seam --------------------------------

    [Fact]
    public void ImageIO_UsesWhateverCodecIsRegistered()
    {
        const int w = 6, h = 4;
        byte[] rgb = Gradient(w, h, channels: 3);
        byte[] png = Managed.EncodePng(rgb, w, h, 3);

        using (new CodecScope(image: Managed))
        {
            RgbImage img = ImageIO.Decode(png);
            Assert.Equal((w, h), (img.Width, img.Height));
            for (int i = 0; i < rgb.Length; i++)
                Assert.Equal(rgb[i] / 255f, img.Pixels[i]);

            // Encode -> decode is exact: [0,1] floats quantise back to the same bytes.
            byte[] again = ImageIO.EncodePng(img);
            Assert.Equal(rgb, Rgb(Managed.DecodeRgba(again, out _, out _)));

            RgbImage small = ImageIO.Resize(img, 3, 2);
            Assert.Equal((3, 2), (small.Width, small.Height));
            Assert.Equal(Managed.ResizeRgb8(rgb, w, h, 3, 2, ResizeFilter.Lanczos), Rgb8(small));
            Assert.Same(img, ImageIO.Resize(img, w, h));
        }

        // And the desktop registration gives the Magick-backed results the pipelines expect.
        RgbImage viaMagick = ImageIO.Decode(png);
        Assert.Equal((w, h), (viaMagick.Width, viaMagick.Height));
        for (int i = 0; i < rgb.Length; i++)
            Assert.Equal(rgb[i] / 255f, viaMagick.Pixels[i]);
    }

    // ---- audio -----------------------------------------------------------------------

    [Fact]
    public void ManagedAudio_WavDecode_MatchesAudioIOExactly()
    {
        const int rate = 22050, n = 2205;
        var left = new float[n];
        var right = new float[n];
        for (int i = 0; i < n; i++)
        {
            left[i] = (float)Math.Sin(2 * Math.PI * 440 * i / rate) * 0.5f;
            right[i] = (float)Math.Sin(2 * Math.PI * 660 * i / rate) * 0.25f;
        }
        string wav = Path.Combine(_dir, "tone.wav");
        WavWriter.Write(wav, new[] { left, right }, rate);

        DecodedAudio viaProvider = Managed.Decode(wav);
        DecodedAudio viaAudioIO = AudioIO.DecodeWav(File.ReadAllBytes(wav));
        Assert.Equal(rate, viaProvider.SampleRate);
        Assert.Equal(2, viaProvider.ChannelCount);
        Assert.Equal(n, viaProvider.SampleCount);
        Assert.Equal(viaAudioIO.Channels[0], viaProvider.Channels[0]);
        Assert.Equal(viaAudioIO.Channels[1], viaProvider.Channels[1]);
        // The tone survives the 16-bit round trip at the amplitude it was written with. Checked
        // as the peak over the buffer, not the sample at n/4: 440 Hz at 22050 Hz is ~50 samples
        // per period, so index 551 is an arbitrary point on the sine, not its crest.
        Assert.Equal(0.5f, viaProvider.Channels[0].Max(Math.Abs), precision: 3);
        Assert.Equal(0.25f, viaProvider.Channels[1].Max(Math.Abs), precision: 3);

        // The rate-conforming entry point still serves the model pipelines.
        DecodedAudio mono16k = AudioIO.Decode(wav, 16000, 1);
        Assert.Equal(16000, mono16k.SampleRate);
        Assert.Equal(1, mono16k.ChannelCount);
        Assert.Equal(1600, mono16k.SampleCount);

        // The chat audio preprocessors' WAV paths are untouched by the seam.
        Assert.Equal(1600, Gemma4AudioPreprocessor.DecodeAudioFile(wav).Length);
        Assert.Equal(1600, NemotronAudioPreprocessor.DecodeAudioFile(wav).Length);
    }

    [Fact]
    public void UnsupportedAudio_GoesToTheRegisteredDecoder_AndIsRefusedByTheManagedOne()
    {
        string m4a = Path.Combine(_dir, "memo.m4a");
        File.WriteAllBytes(m4a, new byte[] { 0, 0, 0, 0x1C, (byte)'f', (byte)'t', (byte)'y', (byte)'p', (byte)'M', (byte)'4', (byte)'A', (byte)' ' });

        var ex = Assert.Throws<NotSupportedException>(() => Managed.Decode(m4a));
        Assert.Contains("MediaCodecs.Audio", ex.Message);
        Assert.Contains("AVFoundation", ex.Message);

        using (new CodecScope(audio: Managed))
        {
            Assert.Throws<NotSupportedException>(() => AudioIO.Decode(m4a, 16000, 1));
            Assert.Throws<NotSupportedException>(() => Gemma4AudioPreprocessor.DecodeAudioFile(m4a));
            Assert.Throws<NotSupportedException>(() => NemotronAudioPreprocessor.DecodeAudioFile(m4a));
        }

        // A platform decoder (an AVAudioFile on iOS) returning 8 kHz stereo is folded and
        // resampled by each caller's own code, exactly as MP3/OGG are today.
        var stub = new StubAudioDecoder(rate: 8000, samples: 8000, channels: 2);
        using (new CodecScope(audio: stub))
        {
            float[] gemma = Gemma4AudioPreprocessor.DecodeAudioFile(m4a);
            Assert.Equal(16000, gemma.Length);
            float[] nemotron = NemotronAudioPreprocessor.DecodeAudioFile(m4a);
            Assert.Equal(16000, nemotron.Length);
            DecodedAudio stereo = AudioIO.Decode(m4a, 16000, 2);
            Assert.Equal(2, stereo.ChannelCount);
            Assert.Equal(16000, stereo.SampleCount);
            Assert.Equal(3, stub.Calls.Count);
            Assert.All(stub.Calls, p => Assert.Equal(m4a, p));

            // Formats the managed decoders own never reach the platform decoder.
            string wav = Path.Combine(_dir, "owned.wav");
            WavWriter.Write(wav, new[] { new float[800] }, 8000);
            Gemma4AudioPreprocessor.DecodeAudioFile(wav);
            NemotronAudioPreprocessor.DecodeAudioFile(wav);
            AudioIO.Decode(wav, 16000, 1);
            Assert.Equal(3, stub.Calls.Count);
        }
    }

    // ---- video: managed refusals -----------------------------------------------------

    [Fact]
    public void ManagedVideo_RefusesWithTheSlotToFill()
    {
        string clip = Path.Combine(_dir, "clip.mp4");
        File.WriteAllBytes(clip, new byte[16]);

        Assert.Contains("MediaCodecs.Video", Assert.Throws<NotSupportedException>(() => Managed.Probe(clip)).Message);
        Assert.Contains("AVFoundation", Assert.Throws<NotSupportedException>(() => Managed.ReadFrames(clip, new[] { 0 }, (_, _, _, _, _, _) => { })).Message);
        Assert.Contains("MediaCodecs.VideoEncoder", Assert.Throws<NotSupportedException>(
            () => Managed.SaveMp4(Path.Combine(_dir, "out.mp4"), new[] { new RgbImage(2, 2, new float[12]) }, 8)).Message);

        using (new CodecScope(video: Managed, encoder: Managed))
        {
            Assert.Throws<NotSupportedException>(() => MediaHelper.ExtractVideoFrames(clip, _dir, "x", maxFrames: 0, fps: 1.0));
            Assert.Throws<NotSupportedException>(() => MediaHelper.ExtractFramesAtRate(clip, 24));
            Assert.Throws<NotSupportedException>(() => VideoIO.SaveMp4(Path.Combine(_dir, "out.mp4"), new[] { new RgbImage(2, 2, new float[12]) }, 8));
        }
    }

    // ---- video: MediaHelper drives any decoder through the seam ----------------------

    [Theory]
    [InlineData(PixelLayout.Bgr, 0)]
    [InlineData(PixelLayout.Bgra, 0)]
    [InlineData(PixelLayout.Rgb, 5)]
    [InlineData(PixelLayout.Rgba, 12)]
    public void ExtractVideoFrames_SamplesAndEncodesWhateverTheDecoderDelivers(PixelLayout layout, int rowPadding)
    {
        var fake = new FakeVideoDecoder { Fps = 24, FrameCount = 48, Width = 16, Height = 8, Layout = layout, RowPadding = rowPadding };
        string clip = Path.Combine(_dir, "fake.mp4");
        File.WriteAllBytes(clip, new byte[1]);

        using (new CodecScope(video: fake))
        {
            List<string> frames = MediaHelper.ExtractVideoFrames(clip, _dir, "p", maxFrames: 0, fps: 1.0);

            Assert.Equal(new[] { clip }, fake.Probed);
            Assert.Equal(new[] { 0, 24 }, fake.Requested);
            Assert.Equal(new[] { "p_0001.png", "p_0002.png" }, frames.Select(Path.GetFileName));

            for (int f = 0; f < frames.Count; f++)
            {
                byte[] rgba = Managed.DecodeRgba(File.ReadAllBytes(frames[f]), out int w, out int h);
                Assert.Equal((16, 8), (w, h));
                (byte r, byte g, byte b) = FakeVideoDecoder.ColorOf(fake.Requested[f]);
                for (int i = 0; i < w * h; i++)
                {
                    Assert.Equal(r, rgba[i * 4]);
                    Assert.Equal(g, rgba[i * 4 + 1]);
                    Assert.Equal(b, rgba[i * 4 + 2]);
                    Assert.Equal(255, rgba[i * 4 + 3]);
                }
            }
        }
    }

    [Fact]
    public void ExtractVideoFrames_DenseSamplingCapsAndShortReads()
    {
        var fake = new FakeVideoDecoder { Fps = 24, FrameCount = 48 };
        string clip = Path.Combine(_dir, "fake.mp4");
        File.WriteAllBytes(clip, new byte[1]);

        using (new CodecScope(video: fake))
        {
            // Every frame at the source rate, evenly capped to 4.
            var frames = MediaHelper.ExtractVideoFrames(clip, _dir, "cap", maxFrames: 4, fps: 24.0);
            Assert.Equal(4, frames.Count);
            Assert.Equal(MediaHelper.SelectEvenlySpacedIndices(48, 4), fake.Requested);

            // A decoder that runs dry hands back what it produced; a short clip is not an error.
            fake.Requested.Clear();
            fake.StopAfter = 5;
            var partial = MediaHelper.ExtractVideoFrames(clip, _dir, "short", maxFrames: 0, fps: 24.0);
            Assert.Equal(5, partial.Count);
            Assert.All(partial, f => Assert.True(File.Exists(f)));

            // An unusable probe is rejected before any frame is asked for.
            fake.Requested.Clear();
            fake.Fps = 0;
            Assert.ThrowsAny<Exception>(() => MediaHelper.ExtractVideoFrames(clip, _dir, "bad", maxFrames: 0, fps: 1.0));
            Assert.Empty(fake.Requested);
        }
    }

    [Fact]
    public void ExtractFramesAtRate_HoldsAndDropsThroughTheSeam()
    {
        string clip = Path.Combine(_dir, "ref.mov");
        File.WriteAllBytes(clip, new byte[1]);

        // 30 fps onto 24 fps: drop every fifth frame.
        var fast = new FakeVideoDecoder { Fps = 30, FrameCount = 30 };
        using (new CodecScope(video: fast))
        {
            var (frames, sourceFps) = MediaHelper.ExtractFramesAtRate(clip, 24);
            Assert.Equal(30, sourceFps);
            Assert.Equal(24, frames.Count);
            Assert.Equal(Enumerable.Range(0, 24).Select(i => (int)Math.Floor(i * 30 / 24.0)).ToList(), fast.Requested);
            Assert.False(MediaHelper.PrefersSequentialStepping(fast.Requested) == false && fast.Requested.Distinct().Count() == fast.Requested.Count && fast.Requested.Max() <= MediaHelper.SequentialStepMaxGap,
                "a dense strictly-increasing request should step");
        }

        // 12 fps onto 24 fps: every frame held twice, which the decoder must deliver twice.
        var slow = new FakeVideoDecoder { Fps = 12, FrameCount = 12 };
        using (new CodecScope(video: slow))
        {
            var (frames, _) = MediaHelper.ExtractFramesAtRate(clip, 24, maxFrames: 10);
            Assert.Equal(10, frames.Count);
            Assert.Equal(new[] { 0, 0, 1, 1, 2, 2, 3, 3, 4, 4 }, slow.Requested);
            Assert.False(MediaHelper.PrefersSequentialStepping(slow.Requested));

            byte[] a = File.ReadAllBytes(frames[0]);
            byte[] b = File.ReadAllBytes(frames[1]);
            Assert.Equal(a, b);
            (byte r, byte g, byte bl) = FakeVideoDecoder.ColorOf(4);
            byte[] last = Managed.DecodeRgba(File.ReadAllBytes(frames[9]), out _, out _);
            Assert.Equal((r, g, bl), (last[0], last[1], last[2]));
        }
    }

    [Fact]
    public void ExtractFramesAtRate_DirectoryMode_IsPureManaged()
    {
        string dir = Path.Combine(_dir, "frames");
        Directory.CreateDirectory(dir);
        byte[] png = Managed.EncodePng(Gradient(4, 4, channels: 3), 4, 4, 3);
        for (int i = 1; i <= 10; i++)
            File.WriteAllBytes(Path.Combine(dir, $"f_{i:D4}.png"), png);
        File.WriteAllText(Path.Combine(dir, "notes.txt"), "ignored");

        using (new CodecScope(video: Managed))   // no video decoder needed for a directory
        {
            // Delivered at the target rate already: one output per file.
            var (same, fps) = MediaHelper.ExtractFramesAtRate(dir, 24);
            Assert.Equal(24, fps);
            Assert.Equal(Enumerable.Range(1, 10).Select(i => $"f_{i:D4}.png"), same.Select(Path.GetFileName));

            // 12 fps hint onto 24 fps: each file held twice.
            var (held, heldFps) = MediaHelper.ExtractFramesAtRate(dir, 24, maxFrames: 0, sourceFpsHint: 12);
            Assert.Equal(12, heldFps);
            Assert.Equal(20, held.Count);
            Assert.Equal(Enumerable.Range(0, 20).Select(i => $"f_{i / 2 + 1:D4}.png"), held.Select(Path.GetFileName));

            var (capped, _) = MediaHelper.ExtractFramesAtRate(dir, 24, maxFrames: 5, sourceFpsHint: 12);
            Assert.Equal(5, capped.Count);

            string empty = Path.Combine(_dir, "empty");
            Directory.CreateDirectory(empty);
            Assert.Throws<InvalidOperationException>(() => MediaHelper.ExtractFramesAtRate(empty, 24));
        }
    }

    // ---- MediaHelper policy knobs ----------------------------------------------------

    [Fact]
    public void InFlightBudgetAndParallelism_AreTunableForAPhone()
    {
        long budget = MediaHelper.InFlightScanlineBudgetBytes;
        int parallel = MediaHelper.MaxParallelFrameEncodes;
        try
        {
            Assert.Equal(256L * 1024 * 1024, MediaHelper.DefaultInFlightScanlineBudgetBytes);
            Assert.Equal(MediaHelper.DefaultInFlightScanlineBudgetBytes, budget);
            Assert.Equal(Environment.ProcessorCount, parallel);

            // Two 1080p frames' worth of budget allows at most two in flight.
            MediaHelper.InFlightScanlineBudgetBytes = 1920L * 1080 * 4 * 2;
            MediaHelper.MaxParallelFrameEncodes = 8;
            Assert.Equal(2, MediaHelper.InFlightLimit(1920, 1080));
            Assert.Equal(8, MediaHelper.InFlightLimit(16, 16));

            MediaHelper.MaxParallelFrameEncodes = 1;
            Assert.Equal(1, MediaHelper.InFlightLimit(16, 16));

            // Even a frame larger than the whole budget gets one slot.
            MediaHelper.InFlightScanlineBudgetBytes = 1;
            Assert.Equal(1, MediaHelper.InFlightLimit(4096, 2160));

            Assert.Throws<ArgumentOutOfRangeException>(() => MediaHelper.InFlightScanlineBudgetBytes = 0);
            Assert.Throws<ArgumentOutOfRangeException>(() => MediaHelper.MaxParallelFrameEncodes = 0);
        }
        finally
        {
            MediaHelper.InFlightScanlineBudgetBytes = budget;
            MediaHelper.MaxParallelFrameEncodes = parallel;
        }
    }

    [Fact]
    public void PrefersSequentialStepping_IsTheMeasuredPolicy()
    {
        Assert.Equal(12, MediaHelper.SequentialStepMaxGap);
        Assert.True(MediaHelper.PrefersSequentialStepping(new[] { 0, 1, 2, 3 }));
        Assert.True(MediaHelper.PrefersSequentialStepping(new[] { 12, 24, 36 }));
        Assert.False(MediaHelper.PrefersSequentialStepping(new[] { 0, 13 }));
        Assert.False(MediaHelper.PrefersSequentialStepping(new[] { 13 }));
        Assert.False(MediaHelper.PrefersSequentialStepping(new[] { 0, 0, 1 }), "a repeated index cannot be stepped to twice");
        Assert.False(MediaHelper.PrefersSequentialStepping(new[] { 5, 3 }));
        Assert.True(MediaHelper.PrefersSequentialStepping(Array.Empty<int>()));
    }

    // ---- helpers ---------------------------------------------------------------------

    private const string EmbeddedJpegBase64 =
        "/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAMCAgMCAgMDAwMEAwMEBQgFBQQEBQoHBwYIDAoMDAsKCwsNDhIQDQ4RDgsLEBYQERMUFRUVDA8XGBYUGBIUFRT/2wBDAQMEBAUEBQkFBQkUDQsNFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBT/wAARCAACAAIDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD7V/Z2+C3w91v9n74ZajqPgTwzf6heeF9MuLm7utHt5JZ5XtImd3dkJZmJJJJySSTRRRXyOL/3ip/if5nwmO/3qr/il+bP/9k=";

    /// <summary>Deterministic, non-degenerate pixels (no two rows alike, every channel used).</summary>
    private static byte[] Gradient(int w, int h, int channels)
    {
        byte[] px = new byte[w * h * channels];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                int p = (y * w + x) * channels;
                px[p] = (byte)(x * 255 / Math.Max(1, w - 1));
                px[p + 1] = (byte)(y * 255 / Math.Max(1, h - 1));
                px[p + 2] = (byte)((x * 31 + y * 17) % 256);
                if (channels == 4) px[p + 3] = (byte)(255 - (x + y) * 7 % 256);
            }
        return px;
    }

    private static byte[] Rgb(byte[] rgba)
    {
        byte[] rgb = new byte[rgba.Length / 4 * 3];
        for (int i = 0; i < rgb.Length / 3; i++)
        {
            rgb[i * 3] = rgba[i * 4]; rgb[i * 3 + 1] = rgba[i * 4 + 1]; rgb[i * 3 + 2] = rgba[i * 4 + 2];
        }
        return rgb;
    }

    private static byte[] Rgb8(RgbImage img) =>
        img.Pixels.Select(v => (byte)Math.Clamp((int)(v * 255f + 0.5f), 0, 255)).ToArray();

    private static void AssertClose(byte[] expected, byte[] actual, int tolerance)
    {
        Assert.Equal(expected.Length, actual.Length);
        int worst = 0, at = -1;
        for (int i = 0; i < expected.Length; i++)
        {
            int d = Math.Abs(expected[i] - actual[i]);
            if (d > worst) { worst = d; at = i; }
        }
        Assert.True(worst <= tolerance, $"max channel difference {worst} at index {at} exceeds {tolerance}");
    }

    private static byte[] HeicHeader()
    {
        var bytes = new List<byte> { 0, 0, 0, 0x18 };
        bytes.AddRange("ftypheic"u8.ToArray());
        bytes.AddRange(new byte[] { 0, 0, 0, 0 });
        bytes.AddRange("mif1heic"u8.ToArray());
        bytes.AddRange(new byte[40]);
        return bytes.ToArray();
    }

    private static byte[] WhiteLeftBlackRightJpeg(int w, int h)
    {
        using var image = new MagickImage(MagickColors.White, (uint)w, (uint)h);
        image.Draw(new Drawables().FillColor(MagickColors.Black).Rectangle(w / 2, 0, w - 1, h - 1));
        image.Format = MagickFormat.Jpeg;
        image.Quality = 100;
        // No chroma subsampling, so a hard vertical edge stays where it is put.
        image.Settings.SetDefine(MagickFormat.Jpeg, "sampling-factor", "1x1");
        return image.ToByteArray();
    }

    /// <summary>A minimal little-endian TIFF block with just IFD0 { Orientation = value }.</summary>
    private static byte[] TiffOrientation(int orientation) => new byte[]
    {
        (byte)'I', (byte)'I', 0x2A, 0x00,         // byte order, 42
        0x08, 0x00, 0x00, 0x00,                   // IFD0 at offset 8
        0x01, 0x00,                               // one entry
        0x12, 0x01, 0x03, 0x00,                   // tag 0x0112, type SHORT
        0x01, 0x00, 0x00, 0x00,                   // count 1
        (byte)orientation, 0x00, 0x00, 0x00,      // value (left-justified in 4 bytes)
        0x00, 0x00, 0x00, 0x00,                   // no next IFD
    };

    /// <summary>Insert an APP1 Exif segment straight after SOI.</summary>
    private static byte[] InjectJpegOrientation(byte[] jpeg, int orientation)
    {
        byte[] payload = "Exif\0\0"u8.ToArray().Concat(TiffOrientation(orientation)).ToArray();
        int length = payload.Length + 2;
        var outp = new List<byte>(jpeg.Length + length + 2);
        outp.AddRange(jpeg.Take(2));
        outp.Add(0xFF); outp.Add(0xE1);
        outp.Add((byte)(length >> 8)); outp.Add((byte)length);
        outp.AddRange(payload);
        outp.AddRange(jpeg.Skip(2));
        return outp.ToArray();
    }

    /// <summary>Insert an eXIf chunk straight after IHDR (byte 33 of every PNG).</summary>
    private static byte[] InjectPngOrientation(byte[] png, int orientation)
    {
        byte[] tiff = TiffOrientation(orientation);
        uint crc = PngCodec.ChunkCrc("eXIf", tiff);
        var chunk = new List<byte>();
        chunk.AddRange(new[] { (byte)(tiff.Length >> 24), (byte)(tiff.Length >> 16), (byte)(tiff.Length >> 8), (byte)tiff.Length });
        chunk.AddRange("eXIf"u8.ToArray());
        chunk.AddRange(tiff);
        chunk.AddRange(new[] { (byte)(crc >> 24), (byte)(crc >> 16), (byte)(crc >> 8), (byte)crc });
        return png.Take(33).Concat(chunk).Concat(png.Skip(33)).ToArray();
    }

    /// <summary>Swap providers for the duration of a test and put the desktop stack back after.</summary>
    private sealed class CodecScope : IDisposable
    {
        private readonly IImageCodec _image = MediaCodecs.Image;
        private readonly IVideoDecoder _video = MediaCodecs.Video;
        private readonly IVideoEncoder _encoder = MediaCodecs.VideoEncoder;
        private readonly IAudioDecoder _audio = MediaCodecs.Audio;

        public CodecScope(IImageCodec? image = null, IVideoDecoder? video = null, IVideoEncoder? encoder = null, IAudioDecoder? audio = null)
        {
            if (image != null) MediaCodecs.Image = image;
            if (video != null) MediaCodecs.Video = video;
            if (encoder != null) MediaCodecs.VideoEncoder = encoder;
            if (audio != null) MediaCodecs.Audio = audio;
        }

        public void Dispose()
        {
            MediaCodecs.Image = _image;
            MediaCodecs.Video = _video;
            MediaCodecs.VideoEncoder = _encoder;
            MediaCodecs.Audio = _audio;
        }
    }

    /// <summary>What an AVFoundation decoder would look like from MediaHelper's side: frames of
    /// one flat colour per index, in a caller-chosen layout, possibly with padded rows.</summary>
    private sealed class FakeVideoDecoder : IVideoDecoder
    {
        public double Fps = 24;
        public int FrameCount = 48;
        public int Width = 16, Height = 8;
        public PixelLayout Layout = PixelLayout.Bgr;
        public int RowPadding;
        public int StopAfter = int.MaxValue;
        public readonly List<string> Probed = new();
        public readonly List<int> Requested = new();

        public static (byte r, byte g, byte b) ColorOf(int index) => ((byte)index, (byte)(index * 2), (byte)(index * 3 + 1));

        public VideoInfo Probe(string path)
        {
            Probed.Add(path);
            return new VideoInfo(Fps, FrameCount, Width, Height);
        }

        public void ReadFrames(string path, IReadOnlyList<int> frameIndices, FrameCallback onFrame)
        {
            int channels = Layout is PixelLayout.Bgra or PixelLayout.Rgba ? 4 : 3;
            int stride = Width * channels + RowPadding;
            byte[] buffer = new byte[stride * Height];
            int delivered = 0;
            foreach (int index in frameIndices)
            {
                Requested.Add(index);
                if (index >= FrameCount || delivered >= StopAfter)
                    return;

                (byte r, byte g, byte b) = ColorOf(index);
                bool bgr = Layout is PixelLayout.Bgr or PixelLayout.Bgra;
                for (int y = 0; y < Height; y++)
                    for (int x = 0; x < Width; x++)
                    {
                        int p = y * stride + x * channels;
                        buffer[p] = bgr ? b : r;
                        buffer[p + 1] = g;
                        buffer[p + 2] = bgr ? r : b;
                        if (channels == 4) buffer[p + 3] = 255;
                    }
                onFrame(index, buffer, Width, Height, stride, Layout);
                delivered++;
            }
        }
    }

    private sealed class StubAudioDecoder : IAudioDecoder
    {
        private readonly int _rate, _samples, _channels;
        public readonly List<string> Calls = new();

        public StubAudioDecoder(int rate, int samples, int channels)
        {
            _rate = rate; _samples = samples; _channels = channels;
        }

        public DecodedAudio Decode(string path)
        {
            Calls.Add(path);
            var planes = new float[_channels][];
            for (int c = 0; c < _channels; c++)
            {
                planes[c] = new float[_samples];
                for (int i = 0; i < _samples; i++)
                    planes[c][i] = (float)Math.Sin(2 * Math.PI * (220 * (c + 1)) * i / _rate) * 0.5f;
            }
            return new DecodedAudio { Channels = planes, SampleRate = _rate };
        }
    }
}
