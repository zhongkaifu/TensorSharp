using ImageMagick;

namespace InferenceWeb.Tests;

public class ImageProcessorTests
{
    private const string EmbeddedJpegBase64 =
        "/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAMCAgMCAgMDAwMEAwMEBQgFBQQEBQoHBwYIDAoMDAsKCwsNDhIQDQ4RDgsLEBYQERMUFRUVDA8XGBYUGBIUFRT/2wBDAQMEBAUEBQkFBQkUDQsNFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBT/wAARCAACAAIDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD7V/Z2+C3w91v9n74ZajqPgTwzf6heeF9MuLm7utHt5JZ5XtImd3dkJZmJJJJySSTRRRXyOL/3ip/if5nwmO/3qr/il+bP/9k=";

    [Fact]
    public void Gemma3ImageProcessorProcessImageSupportsJpeg()
    {
        string path = WriteEmbeddedJpeg();
        try
        {
            var processor = new Gemma3ImageProcessor(imageSize: 32);
            float[] pixels = processor.ProcessImage(path);

            Assert.Equal(3 * 32 * 32, pixels.Length);
            Assert.All(pixels, value => Assert.InRange(value, -1f, 1f));
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void Gemma4ImageProcessorProcessImageSupportsJpeg()
    {
        string path = WriteEmbeddedJpeg();
        try
        {
            var processor = new Gemma4ImageProcessor(patchSize: 1, nMerge: 1, minTokens: 1, maxTokens: 4);
            var (pixels, width, height) = processor.ProcessImage(path);

            Assert.Equal(2, width);
            Assert.Equal(2, height);
            Assert.Equal(12, pixels.Length);
            Assert.All(pixels, value => Assert.InRange(value, -1f, 1f));
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void Gemma4ImageProcessorPreservesAspectRatioAndDoesNotUpscale()
    {
        // Regression test for the gemma4uv image-preprocessing bug.
        //
        // A 750x500 photo has 375k pixels, already inside the [40,280]-token budget
        // (92160..645120 px at patch16*merge3). The reference (llama.cpp
        // calc_size_preserved_ratio + PAD_CEIL) therefore keeps the aspect-aligned
        // native size 768x480 (16x10 = 160 patches) and letter-boxes with black.
        // The old SmartResize always rescaled to the *max* pixel budget, producing a
        // stretched, upscaled 960x624 (260 patches) image -> the model then reported a
        // "very wide", "blurry" picture and hallucinated cut/sliced apples.
        string path = WriteSyntheticPng(750, 500);
        try
        {
            var processor = new Gemma4ImageProcessor(
                patchSize: 16, nMerge: 3, minTokens: 40, maxTokens: 280,
                imageMean: new[] { 0f, 0f, 0f }, imageStd: new[] { 1f, 1f, 1f });
            var (pixels, width, height) = processor.ProcessImage(path);

            Assert.Equal(768, width);   // not the old upscaled 960
            Assert.Equal(480, height);  // not the old upscaled 624
            Assert.Equal(3 * width * height, pixels.Length);

            // Content (red) is centred (720x480, offsetX=24); the 24px left/right
            // borders are black letter-box padding -> normalized 0 (mean=0, std=1).
            int rowMid = (height / 2) * width;
            Assert.InRange(pixels[rowMid + 0], -0.01f, 0.01f);             // left pad -> black
            Assert.True(pixels[rowMid + width / 2] > 0.9f);                // centre -> red (~1)
            Assert.InRange(pixels[rowMid + (width - 1)], -0.01f, 0.01f);   // right pad -> black
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void Qwen35ImageProcessorComputeImageTokenCountSupportsJpeg()
    {
        string path = WriteEmbeddedJpeg();
        try
        {
            var processor = new Qwen35ImageProcessor(patchSize: 1, mergeSize: 1, shortestEdge: 1, longestEdge: 16);
            int tokenCount = processor.ComputeImageTokenCount(path);

            Assert.Equal(4, tokenCount);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void NemotronImageProcessorCapsTilesForLatencyByDefault()
    {
        string path = WriteSyntheticPng(96, 64);
        string? oldMaxTiles = Environment.GetEnvironmentVariable("TS_NEMOTRON_IMAGE_MAX_TILES");
        try
        {
            Environment.SetEnvironmentVariable("TS_NEMOTRON_IMAGE_MAX_TILES", null);
            var processor = CreateNemotronTestProcessor(maxTiles: 12);
            var tiles = processor.ProcessImage(path);

            Assert.Single(tiles);
            Assert.All(tiles, tile =>
            {
                Assert.Equal(64, tile.Width);
                Assert.Equal(64, tile.Height);
                Assert.Equal(3 * 64 * 64, tile.Pixels.Length);
            });
        }
        finally
        {
            Environment.SetEnvironmentVariable("TS_NEMOTRON_IMAGE_MAX_TILES", oldMaxTiles);
            File.Delete(path);
        }
    }

    [Fact]
    public void NemotronImageProcessorHonorsFullTileOverride()
    {
        string path = WriteSyntheticPng(96, 64);
        string? oldMaxTiles = Environment.GetEnvironmentVariable("TS_NEMOTRON_IMAGE_MAX_TILES");
        try
        {
            Environment.SetEnvironmentVariable("TS_NEMOTRON_IMAGE_MAX_TILES", "12");
            var processor = CreateNemotronTestProcessor(maxTiles: 12);
            var tiles = processor.ProcessImage(path);

            Assert.Equal(7, tiles.Count); // 3x2 tiles plus thumbnail.
        }
        finally
        {
            Environment.SetEnvironmentVariable("TS_NEMOTRON_IMAGE_MAX_TILES", oldMaxTiles);
            File.Delete(path);
        }
    }

    [Fact]
    public void UserSuppliedJpegSmokeTestWhenConfigured()
    {
        string? path = Environment.GetEnvironmentVariable("TENSORSHARP_JPEG_SMOKE_PATH");
        if (string.IsNullOrWhiteSpace(path) || !File.Exists(path))
            return;

        var gemma3 = new Gemma3ImageProcessor(imageSize: 32);
        float[] gemma3Pixels = gemma3.ProcessImage(path);

        var (width, height) = Qwen35ImageProcessor.ReadImageDimensions(path);
        var qwen = new Qwen35ImageProcessor(patchSize: 1, mergeSize: 1, shortestEdge: 1, longestEdge: width * height);
        int qwenTokens = qwen.ComputeImageTokenCount(path);

        var gemma4 = new Gemma4ImageProcessor(patchSize: 1, nMerge: 1, minTokens: 1, maxTokens: width * height);
        var (gemma4Pixels, gemma4Width, gemma4Height) = gemma4.ProcessImage(path);

        Assert.Equal(3 * 32 * 32, gemma3Pixels.Length);
        Assert.True(width > 0);
        Assert.True(height > 0);
        Assert.True(qwenTokens > 0);
        Assert.Equal(width, gemma4Width);
        Assert.Equal(height, gemma4Height);
        Assert.Equal(3 * width * height, gemma4Pixels.Length);
    }

    [Fact]
    public void Gemma3ImageProcessorProcessImageSupportsHeic()
    {
        string? heicPath = TryWriteSyntheticHeic(64, 64);
        if (heicPath == null)
            return; // libheif/x265 encoder not present in this Magick.NET build

        try
        {
            var processor = new Gemma3ImageProcessor(imageSize: 32);
            float[] pixels = processor.ProcessImage(heicPath);

            Assert.Equal(3 * 32 * 32, pixels.Length);
            Assert.All(pixels, value => Assert.InRange(value, -1f, 1f));
        }
        finally
        {
            File.Delete(heicPath);
        }
    }

    [Fact]
    public void Qwen35ImageProcessorReadDimensionsSupportsHeic()
    {
        string? heicPath = TryWriteSyntheticHeic(80, 48);
        if (heicPath == null)
            return;

        try
        {
            var (width, height) = Qwen35ImageProcessor.ReadImageDimensions(heicPath);
            Assert.Equal(80, width);
            Assert.Equal(48, height);
        }
        finally
        {
            File.Delete(heicPath);
        }
    }

    [Fact]
    public void UserSuppliedHeicSmokeTestWhenConfigured()
    {
        string? path = Environment.GetEnvironmentVariable("TENSORSHARP_HEIC_SMOKE_PATH");
        if (string.IsNullOrWhiteSpace(path) || !File.Exists(path))
            return;

        var gemma3 = new Gemma3ImageProcessor(imageSize: 32);
        float[] gemma3Pixels = gemma3.ProcessImage(path);

        var (width, height) = Qwen35ImageProcessor.ReadImageDimensions(path);

        Assert.Equal(3 * 32 * 32, gemma3Pixels.Length);
        Assert.All(gemma3Pixels, value => Assert.InRange(value, -1f, 1f));
        Assert.True(width > 0);
        Assert.True(height > 0);
    }

    private static string WriteEmbeddedJpeg()
    {
        string path = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid():N}.jpg");
        File.WriteAllBytes(path, Convert.FromBase64String(EmbeddedJpegBase64));
        return path;
    }

    private static NemotronImageProcessor CreateNemotronTestProcessor(int maxTiles)
    {
        return new NemotronImageProcessor(
            imageSize: 64,
            patchSize: 16,
            numChannels: 3,
            maxTiles: maxTiles,
            minNumPatches: 0,
            maxNumPatches: 0,
            useThumbnail: true,
            projectorScaleFactor: 2,
            imageMean: null,
            imageStd: null);
    }

    private static string WriteSyntheticPng(uint width, uint height)
    {
        string path = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid():N}.png");
        using var image = new MagickImage(MagickColors.Red, width, height);
        image.Format = MagickFormat.Png;
        image.Write(path);
        return path;
    }

    // Synthesize a HEIC file on disk so the round-trip exercise does not require
    // a vendored binary blob. Returns null when the bundled Magick.NET native
    // binaries cannot encode HEIC on this platform (e.g. when x265 is not
    // available), which keeps the test environment-resilient.
    private static string? TryWriteSyntheticHeic(uint width, uint height)
    {
        string path = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid():N}.heic");
        try
        {
            using var image = new MagickImage(MagickColors.Crimson, width, height);
            image.Format = MagickFormat.Heic;
            image.Write(path);

            if (!File.Exists(path) || new FileInfo(path).Length == 0)
            {
                if (File.Exists(path)) File.Delete(path);
                return null;
            }

            return path;
        }
        catch
        {
            if (File.Exists(path)) File.Delete(path);
            return null;
        }
    }
}
