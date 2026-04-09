using InferenceEngine;

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

    private static string WriteEmbeddedJpeg()
    {
        string path = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid():N}.jpg");
        File.WriteAllBytes(path, Convert.FromBase64String(EmbeddedJpegBase64));
        return path;
    }
}
