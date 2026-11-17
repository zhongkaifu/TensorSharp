// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Cuda;
using TensorSharp.GGML;
using TensorSharp.Graph;
using TensorSharp.MLX;
using TensorSharp.Models;
using TensorSharp.Models.Graph;
using TensorSharp.Models.Media;
using TensorSharp.Runtime;
using Xunit.Abstractions;
using Xunit.Sdk;

namespace InferenceWeb.Tests;

/// <summary>[Fact] for one executor lane of the LFM3 vision tower over a real mmproj (TS_LFM3_MMPROJ): skips visibly
/// without the weights or when the lane's backend is not this process's.</summary>
[TraitDiscoverer("InferenceWeb.Tests.RequiresTraitDiscoverer", "InferenceWeb.Tests")]
[AttributeUsage(AttributeTargets.Method)]
public sealed class Lfm3VisionFactAttribute : FactAttribute, ITraitAttribute
{
    public string RequiresValue { get; }

    public Lfm3VisionFactAttribute(Lfm3VisionTests.Lane lane, string extraEnvVar = null)
    {
        RequiresValue = lane switch
        {
            Lfm3VisionTests.Lane.Mlx => "Models,Mlx",
            Lfm3VisionTests.Lane.Cuda => "Models,Cuda",
            Lfm3VisionTests.Lane.GgmlMetal => "Models,GgmlMetal",
            Lfm3VisionTests.Lane.GgmlCpu => "Models,GgmlCpu",
            Lfm3VisionTests.Lane.GgmlCuda => "Models,GgmlCuda",
            _ => "Models",
        };
        Skip = TestGates.ModelSkip("TS_LFM3_MMPROJ")
            ?? (extraEnvVar != null ? TestGates.ModelSkip(extraEnvVar) : null)
            ?? lane switch
            {
                Lfm3VisionTests.Lane.Mlx => TestGates.MlxSkip,
                Lfm3VisionTests.Lane.Cuda => TestGates.CudaSkip,
                Lfm3VisionTests.Lane.GgmlMetal => TestGates.GgmlPinSkip(BackendType.GgmlMetal),
                Lfm3VisionTests.Lane.GgmlCpu => TestGates.GgmlPinSkip(BackendType.GgmlCpu),
                Lfm3VisionTests.Lane.GgmlCuda => TestGates.GgmlPinSkip(BackendType.GgmlCuda),
                _ => null,
            };
    }
}

/// <summary>
/// LFM3-Omni vision input (<see cref="Lfm3VisionEncoder"/>, <see cref="Lfm3ImageProcessor"/>, <see cref="Lfm3Video"/>,
/// <see cref="Lfm3VisionLayout"/>) against llama.cpp's mtmd: Fixtures/Lfm3Vision/reference.json holds, for synthetic
/// pictures (Lfm3VisionSynthetic), a synthetic clip and a fixture PNG with alpha, llama.cpp's chunk layout, the SHA-256
/// of every graph input it built (preprocessing must be bit-identical) and sample projector rows of both Omni mmprojs
/// on ggml-metal (eng/lfm3/vision-reference.sh + vision-fixtures.py). The tower runs on every executor lane against them.
/// </summary>
public sealed class Lfm3VisionTests
{
    public enum Lane { Managed, Reference, GgmlCpu, GgmlMetal, Mlx, Cuda, GgmlCuda }

    private readonly ITestOutputHelper _output;

    public Lfm3VisionTests(ITestOutputHelper output) => _output = output;

    // ================================================================ fixture

    internal sealed record Case(JsonElement Json)
    {
        public string Name => Json.GetProperty("name").GetString();
        public string Kind => Json.GetProperty("kind").GetString();
        public JsonElement Chunks => Json.GetProperty("chunks");
        public JsonElement Mmproj(string type) => Json.GetProperty("mmproj").GetProperty(type);
        public int Int(string name) => Json.GetProperty(name).GetInt32();
    }

    private static readonly Lazy<JsonDocument> s_reference = new(() =>
    {
        using Stream s = typeof(Lfm3VisionTests).Assembly.GetManifestResourceStream("InferenceWeb.Tests.Fixtures.Lfm3Vision.reference.json")
            ?? throw new InvalidOperationException("embedded Fixtures/Lfm3Vision/reference.json is missing");
        return JsonDocument.Parse(s);
    });

    internal static IEnumerable<Case> Cases => s_reference.Value.RootElement.GetProperty("cases").EnumerateArray().Select(c => new Case(c));

    internal static Case Get(string name) => Cases.Single(c => c.Name == name);

    /// <summary>A committed fixture file (the case's path is relative to InferenceWeb.Tests/Fixtures).</summary>
    private static string FixturePath(string relative)
    {
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir != null; dir = dir.Parent)
        {
            string candidate = Path.Combine(dir.FullName, "InferenceWeb.Tests", "Fixtures", relative);
            if (File.Exists(candidate)) return candidate;
        }
        throw new FileNotFoundException($"Cannot locate the committed fixture {relative}.");
    }

    /// <summary>The case's source picture (synth or file).</summary>
    private static Lfm3RgbImage Source(Case c) => c.Kind switch
    {
        "synth" => Lfm3VisionSynthetic.Image(c.Int("width"), c.Int("height"), (uint)c.Int("seed")),
        "file" => Lfm3ImageProcessor.LoadImage(FixturePath(c.Json.GetProperty("file").GetString())),
        _ => throw new ArgumentException(c.Kind),
    };

    /// <summary>The synthetic clip's plan and frames, read through <see cref="Lfm3Video.ReadFrames"/>.</summary>
    private static (Lfm3VideoPlan Plan, Lfm3RgbImage[] Frames, Lfm3SyntheticVideoDecoder Decoder) Clip(Case c)
    {
        var decoder = new Lfm3SyntheticVideoDecoder(c.Int("width"), c.Int("height"), c.Int("frames"), c.Json.GetProperty("fps").GetDouble());
        Lfm3VideoPlan plan = Lfm3Video.Plan("synthetic.mp4", Lfm3Video.MaxFramesOmni, decoder);
        return (plan, Lfm3Video.ReadFrames("synthetic.mp4", plan, decoder), decoder);
    }

    // ================================================================ preprocessing (no weights)

    [Fact]
    public void SyntheticPictureMatchesTheHarnessFormula()
    {
        // synth_rgb as the C++ harness computes it (any change to either side breaks the fixtures)
        Assert.Equal(new byte[] { 22, 66, 27, 27, 113, 98 }, Lfm3VisionSynthetic.Image(7, 5, 3).Rgb.Take(6).ToArray());
        ulong hash = 0;
        unchecked
        {
            foreach (byte b in Lfm3VisionSynthetic.Image(333, 77, 12345).Rgb) hash = hash * 31 + b;
        }
        Assert.Equal(15333685675319718117UL, hash);
    }

    [Theory]
    [InlineData(1920, 1080, 1920, 1088)]
    [InlineData(640, 480, 640, 480)]
    [InlineData(224, 224, 256, 256)]
    [InlineData(100, 100, 256, 256)]
    [InlineData(3000, 4000, 1248, 1664)]
    [InlineData(1024, 768, 1024, 768)]
    [InlineData(100, 80, 288, 256)]
    [InlineData(1000, 250, 992, 256)]
    [InlineData(360, 1280, 352, 1280)]
    [InlineData(19, 25, 224, 320)]
    [InlineData(10, 1000, 32, 2560)]
    [InlineData(80, 48, 352, 224)]
    public void ImageSizesFollowSmartResize(int w, int h, int ew, int eh)
    {
        Assert.Equal((ew, eh), Lfm3ImageProcessor.SmartResize(w, h));
        int tokens = Lfm3ImageProcessor.TokenCount(ew, eh);
        Assert.InRange(tokens, 64, 2048);
    }

    [Theory]
    // clip frames, w, h -> frame size (mtmd get_video_frame_size; the 10 s 1080p and 60 s 720p rows are the mapping report's)
    [InlineData(20, 1920, 1080, 1472, 832)]
    [InlineData(120, 1280, 720, 608, 320)]
    [InlineData(4, 640, 480, 640, 480)]
    [InlineData(384, 1920, 1080, 320, 192)]
    [InlineData(20, 720, 358, 704, 352)]
    [InlineData(5, 320, 240, 320, 256)]   // t_bar = round(2.5) * 2 = 4 (half to even)
    [InlineData(4, 20, 10, 64, 32)]       // sides below 32 are first scaled up (to 64 x 32)
    [InlineData(1, 1280, 720, 96, 64)]    // one frame: t_bar 0, the min branch (llama.cpp proceeds where HF raises)
    [InlineData(20, 854, 358, 864, 352)]  // test-3.mp4 as llama.cpp sees it (720 x 358 at SAR 427:360 displays 854 wide)
    public void VideoFrameSizesFollowTheVideoSmartResize(int frames, int w, int h, int ew, int eh)
        => Assert.Equal((ew, eh), Lfm3ImageProcessor.SmartResizeVideo(frames, w, h));

    [Theory]
    [InlineData(240, 24.0, 384, 20)]
    [InlineData(240, 337500.0 / 14131, 384, 20)]
    [InlineData(75, 30.0, 384, 5)]
    [InlineData(30, 30.0, 384, 4)]       // 2 frames at 2 fps -> at least 4
    [InlineData(3, 30.0, 384, 3)]        // never more than the clip
    [InlineData(9000, 30.0, 384, 384)]   // 300 s
    [InlineData(9000, 30.0, 768, 600)]
    public void FrameCountsSampleTwoPerSecond(int frames, double fps, int max, int expected)
        => Assert.Equal(expected, Lfm3Video.FrameCount(frames, fps, max));

    [Fact]
    public void FrameIndicesAreRoundedLinspace()
    {
        Assert.Equal(new[] { 0, 18, 37, 56, 74 }, Lfm3Video.FrameIndices(75, 5));       // 18.5 -> 18, 55.5 -> 56
        Assert.Equal(new[] { 0 }, Lfm3Video.FrameIndices(1, 1));
        int[] twenty = Lfm3Video.FrameIndices(240, 20);
        Assert.Equal(20, twenty.Length);
        Assert.Equal(0, twenty[0]);
        Assert.Equal(239, twenty[^1]);
        Assert.Equal(13, twenty[1]); // 12.578...
    }

    [Theory]
    [InlineData(0.25, "<0.2 seconds>")]
    [InlineData(0.75, "<0.8 seconds>")]
    [InlineData(0.15, "<0.1 seconds>")]  // 0.1499999999999999944...
    [InlineData(0.45, "<0.5 seconds>")]  // 0.4500000000000000111...
    [InlineData(1.55, "<1.6 seconds>")]
    [InlineData(9.95, "<9.9 seconds>")]  // 9.9499999999999992894...
    [InlineData(0.0, "<0.0 seconds>")]
    [InlineData(123.04999999, "<123.0 seconds>")]
    [InlineData(1e6 + 0.05, "<1000000.1 seconds>")]  // 1000000.0500000000465...
    public void TimestampsRoundHalfToEvenOnTheExactValue(double seconds, string expected)
        => Assert.Equal(expected, Lfm3Video.FormatTimestamp(seconds));

    [Fact]
    public void PreSampledFramesPairUnlessTooFarApart()
    {
        Lfm3VideoPlan dense = Lfm3Video.PlanFrames(640, 480, new[] { 0.0, 0.5, 1.0, 1.5, 2.0 });
        Assert.Equal(new[] { new Lfm3VideoGroup(0, 1, 0.25), new Lfm3VideoGroup(2, 3, 1.25), new Lfm3VideoGroup(4, 4, 2.0) }, dense.Groups);
        Assert.Equal((640, 480), (dense.Width, dense.Height));
        Lfm3VideoPlan sparse = Lfm3Video.PlanFrames(640, 480, new[] { 0.0, 0.5, 1.5, 3.0 }, maxPairGapSeconds: 0.575);
        Assert.Equal(new[] { new Lfm3VideoGroup(0, 1, 0.25), new Lfm3VideoGroup(2, 2, 1.5), new Lfm3VideoGroup(3, 3, 3.0) }, sparse.Groups);
    }

    [Fact]
    public void ShortClipIsSampledAgainOverTheFramesItHas()
    {
        // the container says 75 frames at 30 fps, the decoder delivers 61: 4 frames over 0..60 instead of 5 over 0..74
        var decoder = new Lfm3SyntheticVideoDecoder(320, 256, 75, 30, deliverable: 61);
        var error = new StringWriter();
        TextWriter previous = Console.Error;
        Console.SetError(error);
        (Lfm3VideoPlan plan, Lfm3RgbImage[] frames) read;
        try
        {
            read = Lfm3Video.Read("short.mp4", Lfm3Video.MaxFramesOmni, decoder);
        }
        finally
        {
            Console.SetError(previous);
        }
        Assert.Equal(new[] { 0, 20, 40, 60 }, read.plan.FrameIndices);
        Assert.Equal(4, read.frames.Length);
        Assert.Equal(3, decoder.Requests.Count); // read, count past the last delivered frame, read again
        Assert.Contains("reports 75 frames but its decoder delivers 61", error.ToString());
        Assert.Throws<InvalidOperationException>(() => Lfm3Video.ReadFrames("short.mp4", Lfm3Video.Plan(75, 30, 320, 256), decoder));
    }

    [Fact]
    public void BytePatchesEqualNormalizedPlanarPatches()
    {
        foreach ((int w, int h) in new[] { (64, 32), (2048, 1024) })
        {
            Lfm3RgbImage img = Lfm3VisionSynthetic.Image(w, h, 77);
            float[] planar = Lfm3ImageProcessor.Patchify(Lfm3ImageProcessor.Normalize(img), w, h);
            Assert.Equal(planar, Lfm3ImageProcessor.Patchify(img));
            // a pair's second frame in the upper half of 1536-wide rows
            float[] pair = Lfm3ImageProcessor.Patchify(img, new float[planar.Length * 2], 2 * Lfm3ImageProcessor.PatchValues, Lfm3ImageProcessor.PatchValues);
            for (int r = 0; r < planar.Length / 768; r += 97)
                Assert.Equal(planar.AsSpan(r * 768, 768).ToArray(), pair.AsSpan(r * 1536 + 768, 768).ToArray());
        }
    }

    /// <summary>The sampled frames of a real H.264 / MJPEG clip through the desktop decoder (OpenCV) are exactly the source
    /// frames round(linspace) names: each frame's flat colour identifies its index (VideoFixture paints frame i with its own
    /// colour). Seeking landed frames away on an MP4; the reader steps instead.</summary>
    [VideoTheory]
    [InlineData(VideoFixture.InterCoded)]
    [InlineData(VideoFixture.AllIntra)]
    public void ClipFramesAreTheSampledSourceFrames(string fourcc)
    {
        string dir = Path.Combine(Path.GetTempPath(), "lfm3-video-" + Guid.NewGuid().ToString("N"));
        try
        {
            string path = VideoFixture.TryWrite(Path.Combine(dir, fourcc == VideoFixture.AllIntra ? "clip.avi" : "clip.mp4"), frames: 75, fps: 30, fourcc: fourcc);
            Assert.NotNull(path);
            (Lfm3VideoPlan plan, Lfm3RgbImage[] frames) = Lfm3Video.Read(path, Lfm3Video.MaxFramesOmni, TensorSharp.Models.Media.Desktop.DesktopMediaProvider.Instance);
            Assert.Equal(new[] { 0, 18, 37, 56, 74 }, plan.FrameIndices);
            Assert.Equal((128, 96), (plan.Width, plan.Height));
            for (int f = 0; f < frames.Length; f++)
            {
                // VideoFixture draws a white 24 x 24 block at ((7 i) mod (w - 24), (5 i) mod (h - 24)) on frame i: its corner
                // names the source frame through any codec's colour shifts
                Lfm3RgbImage img = frames[f];
                int minX = int.MaxValue, minY = int.MaxValue;
                for (int y = 0; y < img.Height; y++)
                    for (int x = 0; x < img.Width; x++)
                    {
                        int p = (y * img.Width + x) * 3;
                        if (Math.Min(img.Rgb[p], Math.Min(img.Rgb[p + 1], img.Rgb[p + 2])) > 180)
                        {
                            minX = Math.Min(minX, x);
                            minY = Math.Min(minY, y);
                        }
                    }
                int i = plan.FrameIndices[f];
                (int ex, int ey) = (i * 7 % (img.Width - 24), i * 5 % (img.Height - 24));
                Assert.True(Math.Abs(minX - ex) <= 1 && Math.Abs(minY - ey) <= 1,
                    $"sampled frame {f} should be source frame {i} (block at {ex}, {ey}); its block is at {minX}, {minY}");
            }
        }
        finally
        {
            try { Directory.Delete(dir, recursive: true); } catch { /* best effort */ }
        }
    }

    [Fact]
    public void LayoutsPlaceRowsBetweenMarkers()
    {
        var image = Lfm3VisionLayout.Image(3);
        var spans = new List<Lfm3VisionRowSpan>();
        Assert.Equal(new[] { 125009, 125030, 125030, 125030, 125010 }, image.ToTokens(_ => throw new InvalidOperationException(), rows: spans));
        Assert.Equal(new[] { new Lfm3VisionRowSpan(1, 0, 3) }, spans);
        Assert.Equal(new[] { 125030, 125030, 125030 }, image.ToTokens(_ => Array.Empty<int>(), inner: true));

        var video = Lfm3VisionLayout.Video(new[] { new Lfm3VideoGroup(0, 1, 0.25), new Lfm3VideoGroup(2, 2, 1.0) }, 2);
        spans.Clear();
        List<int> tokens = video.ToTokens(t => t.Select(ch => (int)ch).ToArray(), rowToken: 0, rows: spans);
        string Label(string s) => string.Concat(s.Select(ch => $"{(int)ch},"));
        string expected = "125009," + Label("<0.2 seconds>") + "125009,0,0,125010," + Label("<1.0 seconds>") + "125009,0,0,125010,125010,";
        Assert.Equal(expected, string.Concat(tokens.Select(t => t + ",")));
        Assert.Equal(2, video.Blocks);
        Assert.Equal(4, video.RowCount);
        Assert.Equal(new[] { 1 + 13 + 1, 1 + 13 + 4 + 13 + 1 }, spans.Select(s => s.Offset));
    }

    /// <summary>Every image chunk's graph input, built here, hashes to what llama.cpp fed its graph (bit-identical
    /// resize, normalization and frame pairing), and the chunk layout matches llama.cpp's.</summary>
    [Fact]
    public void PreprocessingIsBitIdenticalToLlamaCpp()
    {
        foreach (Case c in Cases)
        {
            List<(string Text, float[] Pixels, int Rows)> mine = c.Kind == "synthvideo" ? VideoChunks(c) : ImageChunks(c);
            JsonElement[] reference = c.Chunks.EnumerateArray().ToArray();
            Assert.True(reference.Length == mine.Count, $"{c.Name}: {mine.Count} chunks, llama.cpp {reference.Length}");
            for (int i = 0; i < mine.Count; i++)
            {
                if (reference[i].TryGetProperty("text", out JsonElement text))
                {
                    Assert.Equal(text.GetString(), mine[i].Text);
                    continue;
                }
                Assert.Equal(reference[i].GetProperty("rows").GetInt32(), mine[i].Rows);
                Assert.Equal(reference[i].GetProperty("pixels").GetInt32(), mine[i].Pixels.Length);
                Assert.True(reference[i].GetProperty("pixels_sha256").GetString() == Sha256(mine[i].Pixels),
                    $"{c.Name} chunk {i}: the graph input differs from llama.cpp's ({PixelDiff(c, i, mine[i].Pixels)})");
            }
        }
    }

    private static string PixelDiff(Case c, int chunk, float[] pixels) => $"{pixels.Length} values, first {string.Join(", ", pixels.Take(4))}";

    private static string Sha256(float[] values)
    {
        var bytes = new byte[values.Length * 4];
        Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
        return Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
    }

    private static List<(string, float[], int)> ImageChunks(Case c)
    {
        Lfm3RgbImage src = Source(c);
        (int w, int h) = Lfm3ImageProcessor.SmartResize(src.Width, src.Height);
        Lfm3RgbImage resized = Lfm3ImageProcessor.Resize(src, w, h);
        return LayoutChunks(Lfm3VisionLayout.Image(Lfm3ImageProcessor.TokenCount(w, h)), _ => Lfm3ImageProcessor.Normalize(resized));
    }

    private static List<(string, float[], int)> VideoChunks(Case c)
    {
        (Lfm3VideoPlan plan, Lfm3RgbImage[] frames, Lfm3SyntheticVideoDecoder decoder) = Clip(c);
        Assert.Equal(c.Json.GetProperty("served_frames").EnumerateArray().Select(e => e.GetInt32()), plan.FrameIndices);
        // the request steps through the clip (no seek) and includes every sampled frame
        int[] request = Assert.Single(decoder.Requests);
        Assert.True(MediaHelper.PrefersSequentialStepping(request), string.Join(",", request));
        Assert.Empty(plan.FrameIndices.Except(request));
        return LayoutChunks(plan.Layout(), block =>
        {
            Lfm3VideoGroup g = plan.Groups[block];
            float[] a = Lfm3ImageProcessor.Normalize(frames[g.First]);
            return g.Second == g.First ? a : a.Concat(Lfm3ImageProcessor.Normalize(frames[g.Second])).ToArray();
        });
    }

    /// <summary>The layout as llama.cpp's chunks: adjacent tokens and labels as one text chunk, each row block an image
    /// chunk with its graph input (planar frames, a pair's second after its first).</summary>
    private static List<(string, float[], int)> LayoutChunks(Lfm3VisionLayout layout, Func<int, float[]> pixels)
    {
        var chunks = new List<(string, float[], int)>();
        var text = new StringBuilder();
        foreach (Lfm3VisionPiece p in layout.Pieces)
        {
            switch (p.Kind)
            {
                case Lfm3VisionPieceKind.Token:
                    text.Append(p.Token switch { 125009 => "<|vision_start|>", 125010 => "<|vision_end|>", _ => throw new InvalidOperationException() });
                    break;
                case Lfm3VisionPieceKind.Text:
                    text.Append(p.Text);
                    break;
                default:
                    chunks.Add((text.ToString(), null, 0));
                    text.Clear();
                    chunks.Add((null, pixels(p.Block), p.RowCount));
                    break;
            }
        }
        chunks.Add((text.ToString(), null, 0));
        return chunks;
    }

    // ================================================================ the tower against llama.cpp

    /// <summary>One lane's encoder over TS_LFM3_MMPROJ.</summary>
    private sealed class LaneEncoder : IDisposable
    {
        private readonly IDisposable _owner;
        private readonly bool _ggml;

        public LaneEncoder(Lane lane)
        {
            string path = Environment.GetEnvironmentVariable("TS_LFM3_MMPROJ");
            using var gguf = new GgufFile(path);
            MmprojType = gguf.GetUint32("general.file_type") switch { 1 => "F16", 7 => "Q8_0", uint t => throw new InvalidOperationException($"no llama.cpp fixture for an mmproj of file type {t}") };
            IAllocator allocator;
            BackendType backend;
            switch (lane)
            {
                case Lane.Managed:
                case Lane.Reference:
                    allocator = new CpuAllocator(BlasEnum.DotNet);
                    backend = BackendType.Cpu;
                    break;
                case Lane.Mlx:
                    var mlx = new MlxAllocator();
                    _owner = mlx;
                    allocator = mlx;
                    backend = BackendType.Mlx;
                    break;
                case Lane.Cuda:
                    var cuda = new CudaAllocator();
                    _owner = cuda;
                    allocator = cuda;
                    backend = BackendType.Cuda;
                    break;
                default:
                    GgmlBackendType g = lane switch { Lane.GgmlMetal => GgmlBackendType.Metal, Lane.GgmlCuda => GgmlBackendType.Cuda, _ => GgmlBackendType.Cpu };
                    allocator = new GgmlAllocator(new GgmlContext([0], g), 0);
                    backend = lane switch { Lane.GgmlMetal => BackendType.GgmlMetal, Lane.GgmlCuda => BackendType.GgmlCuda, _ => BackendType.GgmlCpu };
                    _ggml = true;
                    break;
            }
            Encoder = lane == Lane.Reference
                ? new Lfm3VisionEncoder(gguf, allocator, backend, new ReferenceExecutor(allocator))
                : new Lfm3VisionEncoder(gguf, allocator, backend);
            switch (lane)
            {
                case Lane.Managed: Assert.IsType<ManagedGraphExecutor>(Encoder.Graph.Executor); break;
                case Lane.Reference: Assert.IsType<ReferenceExecutor>(Encoder.Graph.Executor); break;
                case Lane.Mlx: Assert.IsType<MlxGraphExecutor>(Encoder.Graph.Executor); break;
                case Lane.Cuda: Assert.IsType<CudaGraphExecutor>(Encoder.Graph.Executor); break;
                default: Assert.IsType<GgmlGraphExecutor>(Encoder.Graph.Executor); break;
            }
        }

        public Lfm3VisionEncoder Encoder { get; }

        public string MmprojType { get; }

        public void Dispose()
        {
            Encoder.Dispose();
            _owner?.Dispose();
            if (_ggml)
            {
                GgmlBasicOps.ReleaseReuseComputeBuffers();
                GgmlBasicOps.ClearHostBufferCache();
            }
        }
    }

    /// <summary>The tensor executor without a backend's kernels: the decompositions direct CUDA runs, on the CPU.</summary>
    private sealed class ReferenceExecutor : TensorGraphExecutor
    {
        public ReferenceExecutor(IAllocator allocator) : base(allocator) { }

        protected override Tensor QuantizedMatMul(IGraphWeight weight, Tensor x)
        {
            int rows = (int)x.Sizes[0];
            float[] host = x.GetElementsAsFloat((int)x.ElementCount());
            var y = new float[rows * weight.Ne1];
            HostQuantizedProduct(weight, host, rows, y);
            var result = new Tensor(Allocator, DType.Float32, rows, weight.Ne1);
            result.SetElementsAsFloat(y);
            return result;
        }
    }

    /// <summary>The case's rows as this lane's encoder makes them (all blocks one after the other) and the encode time.</summary>
    private static (float[] Rows, double Ms) Encode(Lfm3VisionEncoder encoder, Case c)
    {
        var sw = Stopwatch.StartNew();
        float[] rows;
        if (c.Kind == "synthvideo")
        {
            (Lfm3VideoPlan plan, Lfm3RgbImage[] frames, _) = Clip(c);
            sw.Restart();
            rows = encoder.EncodeVideo(plan, frames).AllRows();
        }
        else
        {
            Lfm3RgbImage src = Source(c);
            sw.Restart();
            rows = encoder.EncodeImage(src).Rows;
        }
        return (rows, sw.Elapsed.TotalMilliseconds);
    }

    /// <summary>Row-wise agreement with llama.cpp's sample rows (F16-rounded) and its all-row mean / RMS.</summary>
    internal readonly record struct Agreement(double MinCos, double RelRms, double RelMax, double MeanDiff, double RmsRatio)
    {
        public override string ToString() => $"cos min {MinCos:F6}, rel rms {RelRms:G3}, rel max {RelMax:G3}, mean diff {MeanDiff:G3}, rms ratio {RmsRatio:F4}";
    }

    internal static Agreement Compare(Case c, string mmprojType, float[] rows)
    {
        JsonElement m = c.Mmproj(mmprojType);
        int count = m.GetProperty("rows").GetInt32();
        Assert.Equal(count * 2048L, rows.LongLength);
        int[] sample = m.GetProperty("sample_rows").EnumerateArray().Select(e => e.GetInt32()).ToArray();
        byte[] f16 = Convert.FromBase64String(m.GetProperty("sample_f16").GetString());
        double minCos = 1, dd = 0, rr = 0, worst = 0, scale = 0;
        for (int s = 0; s < sample.Length; s++)
        {
            double dot = 0, na = 0, nb = 0;
            for (int i = 0; i < 2048; i++)
            {
                double r = (double)BitConverter.ToHalf(f16, (s * 2048 + i) * 2), a = rows[(long)sample[s] * 2048 + i];
                dot += r * a;
                na += a * a;
                nb += r * r;
                dd += (a - r) * (a - r);
                rr += r * r;
                worst = Math.Max(worst, Math.Abs(a - r));
                scale = Math.Max(scale, Math.Abs(r));
            }
            minCos = Math.Min(minCos, dot / Math.Sqrt(na * nb));
        }
        double mean = 0, sq = 0;
        foreach (float v in rows)
        {
            mean += v;
            sq += (double)v * v;
        }
        mean /= rows.Length;
        double rms = Math.Sqrt(sq / rows.Length);
        return new Agreement(minCos, Math.Sqrt(dd / rr), worst / scale, Math.Abs(mean - m.GetProperty("mean").GetDouble()) / m.GetProperty("rms").GetDouble(),
            rms / m.GetProperty("rms").GetDouble());
    }

    /// <summary>
    /// Each case's rows on <paramref name="lane"/> against llama.cpp's on ggml-metal (same mmproj): within 1% RMS of
    /// llama.cpp's sample rows (cosine >= 0.9999 each), the all-row mean and RMS within 0.1% / 0.3%; the CPU lanes' block
    /// products quantize activations to 8 bits (as ggml-cpu does), so with a Q8_0 mmproj they sit up to 4% away.
    /// </summary>
    private void TowerMatchesLlamaCpp(Lane lane, params string[] names)
    {
        using var lanes = new LaneEncoder(lane);
        bool cpuBlocks = lanes.MmprojType == "Q8_0" && lane is Lane.Managed or Lane.Reference or Lane.GgmlCpu;
        var failures = new List<string>();
        foreach (string name in names)
        {
            Case c = Get(name);
            (float[] rows, double ms) = Encode(lanes.Encoder, c);
            Agreement a = Compare(c, lanes.MmprojType, rows);
            string line = $"lfm3 vision {lane} {lanes.MmprojType} {name}: {a} ({ms:F1} ms; llama.cpp metal {c.Mmproj(lanes.MmprojType).GetProperty("encode_ms").GetDouble():F1} ms)";
            _output.WriteLine(line);
            Console.WriteLine(line);
            // measured (M5 Pro, llama.cpp d3c3e242c): ggml-metal cos >= 0.99998, rel RMS <= 0.18%; MLX >= 0.99993, <= 0.72%;
            // F16 on the CPUs <= 0.22%; Q8_0 on the CPUs >= 0.9988, <= 2.2% (8-bit activations in the block products)
            bool ok = cpuBlocks
                ? a.MinCos >= 0.998 && a.RelRms <= 0.04 && a.MeanDiff <= 0.002 && Math.Abs(a.RmsRatio - 1) <= 0.006
                : a.MinCos >= 0.9999 && a.RelRms <= 0.01 && a.MeanDiff <= 0.001 && Math.Abs(a.RmsRatio - 1) <= 0.003;
            if (!ok) failures.Add(line);
        }
        Assert.True(failures.Count == 0, string.Join("\n", failures));
    }

    private static readonly string[] AllCases = { "small", "wide", "tall", "square", "hd", "cap", "svideo", "alpha" };
    private static readonly string[] CpuCases = { "small", "svideo", "alpha" };

    [Lfm3VisionFact(Lane.GgmlMetal)]
    public void MetalMatchesLlamaCpp() => TowerMatchesLlamaCpp(Lane.GgmlMetal, AllCases);

    [Lfm3VisionFact(Lane.Mlx)]
    public void MlxMatchesLlamaCpp() => TowerMatchesLlamaCpp(Lane.Mlx, AllCases);

    [Lfm3VisionFact(Lane.GgmlCpu)]
    public void GgmlCpuMatchesLlamaCpp() => TowerMatchesLlamaCpp(Lane.GgmlCpu, CpuCases);

    [Lfm3VisionFact(Lane.Managed)]
    public void ManagedCpuMatchesLlamaCpp() => TowerMatchesLlamaCpp(Lane.Managed, CpuCases);

    /// <summary>The decompositions direct CUDA runs (no CUDA device needed), with attention split into query pieces.</summary>
    [Lfm3VisionFact(Lane.Reference)]
    public void ReferenceExecutorMatchesLlamaCpp() => TowerMatchesLlamaCpp(Lane.Reference, "small");

    [Lfm3VisionFact(Lane.Cuda)]
    public void DirectCudaMatchesLlamaCpp() => TowerMatchesLlamaCpp(Lane.Cuda, AllCases);

    [Lfm3VisionFact(Lane.GgmlCuda)]
    public void GgmlCudaMatchesLlamaCpp() => TowerMatchesLlamaCpp(Lane.GgmlCuda, AllCases);

    /// <summary>Attention split into query pieces (the managed CPU's and direct CUDA's bound on score memory) gives the rows
    /// of one piece.</summary>
    [Lfm3VisionFact(Lane.Managed)]
    public void QueryPiecesMatchOnePiece()
    {
        using var lanes = new LaneEncoder(Lane.Managed);
        Lfm3RgbImage img = Lfm3VisionSynthetic.Image(256, 256, 9);
        float[] whole = lanes.Encoder.EncodeImage(img).Rows;
        lanes.Encoder.AttentionChunkRows = 96;
        float[] pieces = lanes.Encoder.EncodeImage(img).Rows;
        double worst = 0;
        for (int i = 0; i < whole.Length; i++) worst = Math.Max(worst, Math.Abs(whole[i] - pieces[i]));
        Assert.True(worst <= 1e-5, $"max |diff| {worst}");
    }
}
