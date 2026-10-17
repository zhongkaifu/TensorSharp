using System.Text.Json;
using TensorSharp.Models.Ltx25;
using TensorSharp.Models.Media;
using TensorSharp.Models.Media.Desktop;
using TensorSharp.Models.QwenImage;

namespace InferenceWeb.Tests;

public sealed class Ltx25ImageCodecFactAttribute : FactAttribute
{
    public Ltx25ImageCodecFactAttribute()
    {
        if (!DesktopMediaProvider.Instance.SupportsH264ImageConditioning)
            Skip = "LTX image preprocessing oracle requires operator-installed FFmpeg with libx264 (PATH or TS_FFMPEG).";
    }
}

public sealed class Ltx25LocalImageOracleFactAttribute : FactAttribute
{
    public Ltx25LocalImageOracleFactAttribute()
    {
        if (!File.Exists(Environment.GetEnvironmentVariable("TS_LTX25_IMAGE_ORACLE")))
            Skip = "Generate a local unchanged-source PyAV fixture with eng/validation/generate_ltx25_image_preprocessing_reference.py and set TS_LTX25_IMAGE_ORACLE to its path. Codec pixels vary with x264/library version and CPU thread count.";
        else if (!DesktopMediaProvider.Instance.SupportsH264ImageConditioning)
            Skip = "The local PyAV comparison requires installed FFmpeg with libx264.";
    }
}

public sealed class Ltx25PosixCodecProbeFactAttribute : FactAttribute
{
    public Ltx25PosixCodecProbeFactAttribute()
    {
        if (OperatingSystem.IsWindows()) Skip = "The fake FFmpeg capability process uses a POSIX script; Windows process integration is not covered by this fixture.";
    }
}

public sealed class Ltx25ImagePreprocessingTests
{
    private static JsonDocument Fixture() => JsonDocument.Parse(typeof(Ltx25ImagePreprocessingTests).Assembly.GetManifestResourceStream(
        "InferenceWeb.Tests.Fixtures.Ltx25.image-preprocessing.json")!);

    private static void CheckNormalized(JsonElement expected, RgbImage actual)
    {
        float[] values = expected.EnumerateArray().Select(value => value.GetSingle()).ToArray();
        Assert.Equal(values.Length, actual.Pixels.Length);
        for (int i = 0; i < values.Length; ++i)
            Assert.True(Math.Abs(values[i] - (actual.Pixels[i] * 2f - 1f)) <= 2e-6f,
                $"Normalized pixel {i}: expected {values[i]:R}, actual {actual.Pixels[i] * 2f - 1f:R}");
    }

    [Fact]
    public void FloatBilinearResizeThenCropMatchesUnchangedOfficialTorchOnEveryPixel()
    {
        using var fixture = Fixture();
        Assert.Equal("be6d6d0a69d11a0a399c462ed92606692e5f8697", fixture.RootElement.GetProperty("source_revision").GetString());
        foreach (var test in fixture.RootElement.GetProperty("resize_cases").EnumerateArray())
        {
            var source = new RgbImage(test.GetProperty("width").GetInt32(), test.GetProperty("height").GetInt32(),
                test.GetProperty("source").EnumerateArray().Select(value => value.GetSingle() / 255f).ToArray());
            float[] unchanged = (float[])source.Pixels.Clone();
            var actual = Ltx25ImagePreprocessing.ResizeAndCenterCrop(source,
                test.GetProperty("target_width").GetInt32(), test.GetProperty("target_height").GetInt32());
            CheckNormalized(test.GetProperty("normalized"), actual);
            Assert.Equal(unchanged, source.Pixels);
        }
    }

    [Fact]
    public void HalfAndFullGenerationResizeCoordinatesMatchOfficialAtUserImageGeometry()
    {
        using var fixture = Fixture();
        foreach (var test in fixture.RootElement.GetProperty("large_resize_cases").EnumerateArray())
        {
            int width = test.GetProperty("width").GetInt32(), height = test.GetProperty("height").GetInt32();
            var source = new float[width * height * 3];
            for (int y = 0; y < height; ++y)
                for (int x = 0; x < width; ++x)
                    for (int c = 0; c < 3; ++c)
                        source[(y * width + x) * 3 + c] = ((x * 17 + y * 29 + c * 71 + x * y * 3) % 256) / 255f;
            var actual = Ltx25ImagePreprocessing.ResizeAndCenterCrop(new RgbImage(width, height, source),
                test.GetProperty("target_width").GetInt32(), test.GetProperty("target_height").GetInt32());
            int[] indices = test.GetProperty("indices").EnumerateArray().Select(value => value.GetInt32()).ToArray();
            var expected = test.GetProperty("normalized").EnumerateArray().ToArray();
            for (int i = 0; i < indices.Length; ++i)
                Assert.InRange(Math.Abs(expected[i].GetSingle() - (actual.Pixels[indices[i]] * 2f - 1f)), 0, 2e-6f);
        }
    }

    [Ltx25LocalImageOracleFact]
    public void LocalH264CompressionAndResizedGuidesMatchUnchangedOfficialPyAvPixels()
    {
        // This fixture is regenerated with the local PyAV codec stack. It must not
        // assert universal bit equality across x264 versions or CPU slice counts.
        using var fixture = JsonDocument.Parse(File.ReadAllText(Environment.GetEnvironmentVariable("TS_LTX25_IMAGE_ORACLE")!));
        foreach (var test in fixture.RootElement.GetProperty("codec_cases").EnumerateArray())
        {
            byte[] source = Convert.FromBase64String(test.GetProperty("source_rgb_base64").GetString()!);
            byte[] expected = Convert.FromBase64String(test.GetProperty("compressed_rgb_base64").GetString()!);
            var image = new RgbImage(test.GetProperty("width").GetInt32(), test.GetProperty("height").GetInt32(),
                source.Select(value => value / 255f).ToArray());
            var compressed = Ltx25ImagePreprocessing.PrepareSource(image, encoder: DesktopMediaProvider.Instance);
            Assert.Equal(test.GetProperty("compressed_width").GetInt32(), compressed.Width);
            Assert.Equal(test.GetProperty("compressed_height").GetInt32(), compressed.Height);
            Assert.Equal(expected, compressed.Pixels.Select(value => (byte)(value * 255f + .5f)));
            var actual = Ltx25ImagePreprocessing.ResizeAndCenterCrop(compressed,
                test.GetProperty("target_width").GetInt32(), test.GetProperty("target_height").GetInt32());
            CheckNormalized(test.GetProperty("normalized"), actual);
        }
    }

    [Ltx25ImageCodecFact]
    public void CompressionDropsOnlyOddEdgesAndProducesResizedFloatGuides()
    {
        using var fixture = Fixture();
        var test = fixture.RootElement.GetProperty("codec_cases")[0];
        byte[] source = Convert.FromBase64String(test.GetProperty("source_rgb_base64").GetString()!);
        int width = test.GetProperty("width").GetInt32(), height = test.GetProperty("height").GetInt32();
        var image = new RgbImage(width, height, source.Select(value => value / 255f).ToArray());
        var compressed = Ltx25ImagePreprocessing.PrepareSource(image, encoder: DesktopMediaProvider.Instance);
        Assert.Equal((width - 1, height - 1), (compressed.Width, compressed.Height));
        var altered = (float[])image.Pixels.Clone();
        for (int y = 0; y < height; ++y)
            for (int channel = 0; channel < 3; ++channel) altered[(y * width + width - 1) * 3 + channel] = 1;
        Array.Fill(altered, 1f, (height - 1) * width * 3, width * 3);
        var same = Ltx25ImagePreprocessing.PrepareSource(new RgbImage(width, height, altered), encoder: DesktopMediaProvider.Instance);
        Assert.Equal(compressed.Pixels, same.Pixels);
        Assert.True(compressed.Pixels.Where((value, index) => value != image.Pixels[(index / ((width - 1) * 3)) * width * 3 + index % ((width - 1) * 3)]).Any());
        Assert.All(compressed.Pixels, value => Assert.InRange(value, 0f, 1f));
        var resized = Ltx25ImagePreprocessing.ResizeAndCenterCrop(compressed, 9, 7);
        Assert.Equal((9, 7), (resized.Width, resized.Height));
        Assert.All(resized.Pixels, value => Assert.InRange(value, 0f, 1f));
    }

    [Fact]
    public void UnsupportedCodecFailsAtSourcePreparationWithoutEncodingOrFallback()
    {
        var codec = new UnsupportedCodec();
        var error = Assert.Throws<NotSupportedException>(() => Ltx25ImagePreprocessing.PrepareSource(
            new RgbImage(2, 2, new float[12]), encoder: codec));
        Assert.Contains("FFmpeg", error.Message);
        Assert.False(codec.WasCalled);
        Assert.False(((IVideoEncoder)ManagedMediaProvider.Instance).SupportsH264ImageConditioning);
        Assert.Throws<NotSupportedException>(() => ((IVideoEncoder)ManagedMediaProvider.Instance).RecompressH264Image(new byte[12], 2, 2, 18));
    }

    [Fact]
    public void SingletonImageUsesOfficialCompressionBypassAndCancellationStopsBeforeCodec()
    {
        var codec = new UnsupportedCodec();
        var singleton = new RgbImage(1, 2, new float[6]);
        Assert.Same(singleton, Ltx25ImagePreprocessing.PrepareSource(singleton, encoder: codec));
        Assert.Throws<OperationCanceledException>(() => Ltx25ImagePreprocessing.PrepareSource(
            new RgbImage(2, 2, new float[12]), new CancellationToken(canceled: true), codec));
        Assert.False(codec.WasCalled);
    }

    [Fact]
    public void MissingEncoderExecutableCannotAdvertiseImageConditioning()
    {
        Assert.False(DesktopMediaProvider.HasH264ConditioningEncoder(null!));
        Assert.False(DesktopMediaProvider.HasH264ConditioningEncoder(Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"))));
    }

    [Ltx25PosixCodecProbeFact]
    public void EncoderCapabilityRequiresLibx264AndCachesUntilExecutableChanges()
    {
        string directory = Path.Combine(Path.GetTempPath(), "ts-codec-capability-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        string executable = Path.Combine(directory, "ffmpeg");
        try
        {
            void WriteEncoderList(string encoders)
            {
                File.WriteAllText(executable, "#!/bin/sh\nprintf x >> \"$0.count\"\nprintf '%s\\n' '" + encoders + "'\n");
                File.SetUnixFileMode(executable, UnixFileMode.UserRead | UnixFileMode.UserWrite | UnixFileMode.UserExecute);
            }
            WriteEncoderList(" V....D h264_videotoolbox VideoToolbox H264; V....D libx264rgb H264 RGB");
            Assert.False(DesktopMediaProvider.HasH264ConditioningEncoder(executable));
            Assert.False(DesktopMediaProvider.HasH264ConditioningEncoder(executable));
            Assert.Equal("x", File.ReadAllText(executable + ".count"));
            DateTime before = File.GetLastWriteTimeUtc(executable);
            WriteEncoderList(" V....D libx264 libx264 H.264 / AVC / MPEG-4 AVC");
            File.SetLastWriteTimeUtc(executable, before.AddSeconds(2));
            Assert.True(DesktopMediaProvider.HasH264ConditioningEncoder(executable));
            Assert.True(DesktopMediaProvider.HasH264ConditioningEncoder(executable));
            Assert.Equal("xx", File.ReadAllText(executable + ".count"));
        }
        finally { Directory.Delete(directory, recursive: true); }
    }

    [Ltx25PosixCodecProbeFact]
    public void CapabilityDeadlineAlsoCoversPipesHeldByAChildAfterTheWrapperExits()
    {
        string directory = Path.Combine(Path.GetTempPath(), "ts-codec-deadline-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        string executable = Path.Combine(directory, "ffmpeg");
        try
        {
            File.WriteAllText(executable, "#!/bin/sh\nsleep 20 &\nprintf '%s' \"$!\" > \"$0.pid\"\nprintf '%s\\n' ' V....D libx264 H264'\nexit 0\n");
            File.SetUnixFileMode(executable, UnixFileMode.UserRead | UnixFileMode.UserWrite | UnixFileMode.UserExecute);
            var timer = System.Diagnostics.Stopwatch.StartNew();
            Assert.False(DesktopMediaProvider.HasH264ConditioningEncoder(executable));
            Assert.InRange(timer.Elapsed.TotalSeconds, 4, 10);
        }
        finally
        {
            if (File.Exists(executable + ".pid"))
            {
                try
                {
                    using var child = System.Diagnostics.Process.GetProcessById(int.Parse(File.ReadAllText(executable + ".pid")));
                    child.Kill();
                    child.WaitForExit();
                }
                catch (ArgumentException) { }
                catch (InvalidOperationException) { }
            }
            Directory.Delete(directory, recursive: true);
        }
    }

    private sealed class UnsupportedCodec : IVideoEncoder
    {
        public bool WasCalled { get; private set; }
        public byte[] RecompressH264Image(byte[] rgb8, int width, int height, int crf, CancellationToken cancellationToken = default)
        { WasCalled = true; throw new InvalidOperationException("Unsupported codec must never be invoked."); }
        public string SaveMp4(string path, RgbImage[] frames, int fps)
        { WasCalled = true; throw new InvalidOperationException("Output encoder must never be a conditioning fallback."); }
    }
}
