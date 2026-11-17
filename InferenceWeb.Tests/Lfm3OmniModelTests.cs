// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Diagnostics;
using System.Text.Json;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.GGML;
using TensorSharp.MLX;
using TensorSharp.Models;
using TensorSharp.Models.Media;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;
using Xunit.Abstractions;
using Xunit.Sdk;

namespace InferenceWeb.Tests;

/// <summary>[Fact] over LFM3-Omni-2B with its mmproj (TS_TEST_LFM3_MODEL: a directory with LFM3-Omni-2B-*.gguf and
/// mmproj-LFM3-Omni-2B-*.gguf, or TS_LFM3_MMPROJ for the projector) on one backend lane: GGML lanes run in the process
/// that pins their backend (TS_TEST_GGML_BACKEND), MLX where its library loads, the managed CPU everywhere.</summary>
[TraitDiscoverer("InferenceWeb.Tests.RequiresTraitDiscoverer", "InferenceWeb.Tests")]
[AttributeUsage(AttributeTargets.Method)]
public sealed class Lfm3OmniFactAttribute : FactAttribute, ITraitAttribute
{
    public string RequiresValue { get; }

    public Lfm3OmniFactAttribute(BackendType lane, bool video = false)
    {
        RequiresValue = lane switch
        {
            BackendType.Mlx => "Models,Mlx",
            BackendType.GgmlMetal => "Models,GgmlMetal",
            BackendType.GgmlCpu => "Models,GgmlCpu",
            _ => "Models",
        } + (video ? ",Video" : "");
        Skip = TestGates.ModelSkip(Lfm3OmniModelTests.ModelEnv, "lfm3-omni")
            ?? (video ? TestGates.VideoSkip : null)
            ?? lane switch
            {
                BackendType.Mlx => TestGates.MlxSkip,
                BackendType.Cpu => null,
                _ => TestGates.GgmlPinSkip(lane),
            };
    }
}

/// <summary>
/// LFM3-Omni media input end to end on the model (<see cref="Lfm3RuntimeModel"/> + the injector's
/// <c>ProcessLfm3History</c>), real weights: the template's spans filled with the towers' rows, a video file laid out
/// exactly as llama.cpp's mtmd lays it out (Fixtures/Lfm3Vision, the synthetic clip), a soundtrack's rows after each
/// frame pair, host-sampled frames paired by their times, media identity for the prefix cache (a follow-up question
/// about the same image reuses the KV through it, a different image stops at it) and two long clips encoded
/// concurrently while each yields the GPU lock (no deadlock).
/// </summary>
public sealed class Lfm3OmniModelTests : IDisposable
{
    public const string ModelEnv = "TS_TEST_LFM3_MODEL";

    private readonly ITestOutputHelper _output;
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "lfm3-omni-model-" + Guid.NewGuid().ToString("N"));

    public Lfm3OmniModelTests(ITestOutputHelper output)
    {
        _output = output;
        Directory.CreateDirectory(_directory);
    }

    public void Dispose()
    {
        try { Directory.Delete(_directory, true); } catch (IOException) { }
    }

    // ================================================================ model per lane

    private static readonly object s_gate = new();
    private static readonly Dictionary<BackendType, Lfm3RuntimeModel> s_models = new();

    /// <summary>The lane's model, loaded once per process (with the projector) and kept for the class's tests.</summary>
    private static Lfm3RuntimeModel Model(BackendType lane)
    {
        lock (s_gate)
        {
            if (s_models.TryGetValue(lane, out Lfm3RuntimeModel cached))
                return cached;
            string dir = Environment.GetEnvironmentVariable(ModelEnv)!;
            string path = TestGates.FindSmallestGguf(dir, "lfm3-omni")!;
            var model = Assert.IsType<Lfm3RuntimeModel>(ModelBase.Create(path, lane));
            model.MultimodalInjector.LoadProjectors(MmprojPath(dir));
            Assert.NotNull(model.VisionEncoder);
            Assert.NotNull(model.AudioEncoder);
            s_models[lane] = model;
            return model;
        }
    }

    private static string MmprojPath(string modelDir)
    {
        string explicitPath = Environment.GetEnvironmentVariable("TS_LFM3_MMPROJ");
        if (!string.IsNullOrEmpty(explicitPath)) return explicitPath;
        string dir = File.Exists(modelDir) ? Path.GetDirectoryName(modelDir)! : modelDir;
        return Directory.GetFiles(dir, "mmproj-LFM3-Omni-2B-Q8_0.gguf").FirstOrDefault()
            ?? Directory.GetFiles(dir, "*mmproj*LFM3*.gguf").OrderBy(p => p).First();
    }

    private const int VisionStart = 125009, VisionEnd = 125010, ImageToken = 125030, VideoToken = 125029, AudioToken = 125031;

    private static List<int> Render(ModelBase model, List<ChatMessage> messages)
    {
        string text = ChatTemplate.RenderFromGgufTemplate(model.Config.ChatTemplate, messages, addGenerationPrompt: true, architecture: "lfm3");
        return model.Tokenizer.Encode(text, addSpecial: true);
    }

    private static (List<int> Tokens, IReadOnlyList<PromptMediaSpan> Spans) Prepare(ModelBase model, List<ChatMessage> history,
        List<int> tokens, string requestId)
    {
        lock (model.GpuComputeLock)
        {
            List<int> expanded = model.MultimodalInjector.ProcessPromptTokens(history, new List<int>(tokens), requestId);
            return (expanded, model.MultimodalInjector.GetPreparedMediaSpans(requestId));
        }
    }

    private string WritePicture(string name, int width, int height, int seed)
    {
        var rgb = new byte[width * height * 3];
        var rng = new Random(seed);
        // Smooth colour fields with a few blocks: content the tower and the model can tell apart by seed.
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
            {
                int p = (y * width + x) * 3;
                rgb[p] = (byte)(x * 255 / width);
                rgb[p + 1] = (byte)(y * 255 / height);
                rgb[p + 2] = (byte)((seed * 70) & 255);
            }
        for (int b = 0; b < 6; b++)
        {
            int bx = rng.Next(width - 40), by = rng.Next(height - 40);
            byte r = (byte)rng.Next(256), g = (byte)rng.Next(256), bl = (byte)rng.Next(256);
            for (int y = by; y < by + 40; y++)
                for (int x = bx; x < bx + 40; x++)
                {
                    int p = (y * width + x) * 3;
                    (rgb[p], rgb[p + 1], rgb[p + 2]) = (r, g, bl);
                }
        }
        string path = Path.Combine(_directory, name);
        using var image = new ImageMagick.MagickImage(rgb, new ImageMagick.PixelReadSettings((uint)width, (uint)height, ImageMagick.StorageType.Char, "RGB"));
        image.Write(path, ImageMagick.MagickFormat.Png);
        return path;
    }

    // ================================================================ images and audio

    [Lfm3OmniFact(BackendType.GgmlMetal)] public void ImageAndAudio_FillTheTemplateSpans_GgmlMetal() => ImageAndAudio(BackendType.GgmlMetal);
    [Lfm3OmniFact(BackendType.GgmlCpu)] public void ImageAndAudio_FillTheTemplateSpans_GgmlCpu() => ImageAndAudio(BackendType.GgmlCpu);
    [Lfm3OmniFact(BackendType.Mlx)] public void ImageAndAudio_FillTheTemplateSpans_Mlx() => ImageAndAudio(BackendType.Mlx);
    [Lfm3OmniFact(BackendType.Cpu)] public void ImageAndAudio_FillTheTemplateSpans_Cpu() => ImageAndAudio(BackendType.Cpu);

    private void ImageAndAudio(BackendType lane)
    {
        Lfm3RuntimeModel model = Model(lane);
        string image = WritePicture("a.png", 100, 80, 1);
        string copy = Path.Combine(_directory, "a-copy.png");
        File.Copy(image, copy);
        string other = WritePicture("b.png", 100, 80, 2);
        // 1.25 s of a chirp at 16 kHz: 123 frames, 16 rows.
        var samples = new float[20000];
        for (int i = 0; i < samples.Length; i++) samples[i] = 0.3f * MathF.Sin(2 * MathF.PI * (200f + i * 0.05f) * i / 16000f);
        string clip = Path.Combine(_directory, "chirp.wav");
        TensorSharp.Models.Video.WavWriter.Write(clip, new[] { samples }, 16000);
        int audioRows = Lfm3AudioPreprocessor.RowCountForSamples(samples.Length);
        Assert.Equal(16, audioRows);

        var history = new List<ChatMessage>
        {
            new() { Role = "user", Content = "What do you see and hear?", ImagePaths = [image], AudioPaths = [clip] },
        };
        List<int> tokens = Render(model, history);
        int placeholderImage = tokens.IndexOf(ImageToken), placeholderAudio = tokens.IndexOf(AudioToken);
        Assert.True(placeholderImage > 0 && placeholderAudio > placeholderImage);
        var (expanded, spans) = Prepare(model, history, tokens, "img-audio");
        try
        {
            // 100 x 80 -> 288 x 256 (the pixel floor): 9 x 8 = 72 rows.
            const int imageRows = 72;
            Assert.Equal(tokens.Count - 2 + imageRows + audioRows, expanded.Count);
            Assert.Equal(tokens.Take(placeholderImage), expanded.Take(placeholderImage));
            Assert.All(expanded.Skip(placeholderImage).Take(imageRows), t => Assert.Equal(ImageToken, t));
            Assert.Equal(VisionEnd, expanded[placeholderImage + imageRows]);
            int audioAt = placeholderAudio - 1 + imageRows;
            Assert.Equal(VisionStart, expanded[audioAt - 1]);
            Assert.All(expanded.Skip(audioAt).Take(audioRows), t => Assert.Equal(AudioToken, t));
            Assert.Equal(VisionEnd, expanded[audioAt + audioRows]);
            Assert.Equal(tokens.Skip(placeholderAudio + 1), expanded.Skip(audioAt + audioRows));

            // Each span covers its markers and names its content.
            Assert.Equal(2, spans.Count);
            Assert.Equal((placeholderImage - 1, placeholderImage + imageRows + 1), (spans[0].Start, spans[0].End));
            Assert.Equal((audioAt - 1, audioAt + audioRows + 1), (spans[1].Start, spans[1].End));
            Assert.StartsWith("img:", spans[0].ContentId);
            Assert.StartsWith("aud:", spans[1].ContentId);
            Assert.True(model.MultimodalInjector.HasPendingEmbeddings("img-audio"));
        }
        finally
        {
            model.MultimodalInjector.ClearPreparedPromptState("img-audio");
        }

        // The same picture under another name is the same media; another picture is not.
        string IdOf(string path)
        {
            var h = new List<ChatMessage> { new() { Role = "user", Content = "Q", ImagePaths = [path] } };
            var (_, s) = Prepare(model, h, Render(model, h), "id");
            model.MultimodalInjector.ClearPreparedPromptState("id");
            return Assert.Single(s).ContentId;
        }
        Assert.Equal(IdOf(image), IdOf(copy));
        Assert.NotEqual(IdOf(image), IdOf(other));
    }

    // ================================================================ video

    [Lfm3OmniFact(BackendType.GgmlMetal, video: true)] public void VideoFile_IsLaidOutAsLlamaCppsMtmd_GgmlMetal() => VideoFileLayout(BackendType.GgmlMetal);
    [Lfm3OmniFact(BackendType.Mlx, video: true)] public void VideoFile_IsLaidOutAsLlamaCppsMtmd_Mlx() => VideoFileLayout(BackendType.Mlx);
    [Lfm3OmniFact(BackendType.Cpu, video: true)] public void VideoFile_IsLaidOutAsLlamaCppsMtmd_Cpu() => VideoFileLayout(BackendType.Cpu);

    /// <summary>The synthetic clip of the vision fixture (75 frames at 30 fps, 320 x 256; no sound) written as MJPEG and read
    /// through the desktop decoder: its span is token for token llama.cpp's mtmd chunks (labels, markers, three pairs of 80
    /// rows), the frames the model's own 2 fps sampling, not the host's.</summary>
    private void VideoFileLayout(BackendType lane)
    {
        Lfm3RuntimeModel model = Model(lane);
        string clip = VideoFixture.TryWrite(Path.Combine(_directory, "svideo.avi"), frames: 75, fps: 30, width: 320, height: 256,
            fourcc: VideoFixture.AllIntra);
        Assert.NotNull(clip);

        var expected = new List<int>();
        using (Stream s = typeof(Lfm3VisionTests).Assembly.GetManifestResourceStream("InferenceWeb.Tests.Fixtures.Lfm3Vision.reference.json")!)
        using (JsonDocument fixture = JsonDocument.Parse(s))
        {
            JsonElement svideo = fixture.RootElement.GetProperty("cases").EnumerateArray().First(c => c.GetProperty("name").GetString() == "svideo");
            foreach (JsonElement chunk in svideo.GetProperty("chunks").EnumerateArray())
            {
                if (chunk.TryGetProperty("tokens", out JsonElement ids))
                    expected.AddRange(ids.EnumerateArray().Select(t => t.GetInt32()));
                else
                    expected.AddRange(Enumerable.Repeat(ImageToken, chunk.GetProperty("rows").GetInt32()));
            }
        }

        // Host-sampled frames ride along (as the CLI and the server send them); the model reads the file instead.
        var history = new List<ChatMessage>
        {
            new() { Role = "user", Content = "What moves?", IsVideo = true, ImagePaths = [WritePicture("f.png", 64, 64, 3)], VideoFilePaths = [clip] },
        };
        List<int> tokens = Render(model, history);
        int at = tokens.IndexOf(VideoToken);
        var (expanded, spans) = Prepare(model, history, tokens, "svideo");
        model.MultimodalInjector.ClearPreparedPromptState("svideo");
        Assert.Equal(expected, expanded.Skip(at - 1).Take(expected.Count));
        Assert.Equal(tokens.Skip(at + 2), expanded.Skip(at - 1 + expected.Count));
        Assert.Equal(3, spans.Count);
        Assert.All(spans, s => Assert.StartsWith("lfm3v:", s.ContentId));
        Assert.Equal(3, spans.Select(s => s.ContentId).Distinct().Count());
    }

    [Lfm3OmniFact(BackendType.GgmlMetal)] public void VideoSoundtrack_FollowsEachFramePair_GgmlMetal() => VideoSoundtrack(BackendType.GgmlMetal);
    [Lfm3OmniFact(BackendType.Mlx)] public void VideoSoundtrack_FollowsEachFramePair_Mlx() => VideoSoundtrack(BackendType.Mlx);

    /// <summary>A clip with a soundtrack (ffmpeg: a 2.5 s test pattern at 30 fps with a tone): after each pair's
    /// <c>&lt;|vision_end|&gt;</c> come that pair's soundtrack rows, bare, as llama.cpp's helper splits the track
    /// (<see cref="Lfm3AudioPrompt.SplitSoundtrack"/>), and together they are the whole trimmed track's rows.</summary>
    private void VideoSoundtrack(BackendType lane)
    {
        string ffmpeg = MediaExecutableLocator.Find("ffmpeg", "TS_FFMPEG");
        if (ffmpeg == null)
        {
            _output.WriteLine("no ffmpeg: the soundtrack case needs it to write a clip with sound (NOT RUN)");
            Assert.Fail("ffmpeg is required to write a clip with a soundtrack (set TS_FFMPEG)");
        }
        Lfm3RuntimeModel model = Model(lane);
        string clip = Path.Combine(_directory, "tone.mp4");
        var psi = new ProcessStartInfo(ffmpeg)
        {
            RedirectStandardError = true, RedirectStandardOutput = true, UseShellExecute = false,
        };
        foreach (string a in new[] { "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=320x256:rate=30:duration=2.5",
                     "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000:duration=2.5", "-c:v", "mpeg4", "-q:v", "4",
                     "-c:a", "aac", "-shortest", clip })
            psi.ArgumentList.Add(a);
        using (var p = Process.Start(psi)!)
        {
            string err = p.StandardError.ReadToEnd();
            p.WaitForExit();
            Assert.True(p.ExitCode == 0, err);
        }

        Lfm3VideoPlan plan = Lfm3Video.Plan(clip);
        float[] track = Lfm3AudioPreprocessor.LoadAudio(clip);
        var pieces = Lfm3AudioPrompt.SplitSoundtrack(plan.FrameIndices, plan.SourceFps, plan.SourceFrames, track.Length, out long trimmed);
        int trackRows = Lfm3AudioPreprocessor.RowCountForSamples(trimmed);
        _output.WriteLine($"plan: {plan.SourceFrames} frames at {plan.SourceFps} fps, {plan.Groups.Count} pairs of {plan.RowsPerGroup} rows; " +
            $"track {track.Length} samples -> {trimmed} -> {trackRows} rows in {pieces.Count} pieces: {string.Join(" ", pieces.Select(x => x.Rows))}");

        var history = new List<ChatMessage> { new() { Role = "user", Content = "What do you hear?", IsVideo = true, VideoFilePaths = [clip] } };
        List<int> tokens = Render(model, history);
        int at = tokens.IndexOf(VideoToken);
        var (expanded, spans) = Prepare(model, history, tokens, "tone");
        model.MultimodalInjector.ClearPreparedPromptState("tone");

        int cursor = at;   // the first label starts where <|video|> was
        int audioSpans = 0, audioRows = 0;
        for (int g = 0; g < plan.Groups.Count; g++)
        {
            List<int> label = model.Tokenizer.Encode(plan.Groups[g].Label, addSpecial: false);
            Assert.Equal(label, expanded.Skip(cursor).Take(label.Count));
            cursor += label.Count;
            Assert.Equal(VisionStart, expanded[cursor++]);
            Assert.All(expanded.Skip(cursor).Take(plan.RowsPerGroup), t => Assert.Equal(ImageToken, t));
            cursor += plan.RowsPerGroup;
            Assert.Equal(VisionEnd, expanded[cursor++]);
            int rows = pieces.Where(x => x.Group == g).Sum(x => x.Rows);
            Assert.All(expanded.Skip(cursor).Take(rows), t => Assert.Equal(AudioToken, t));
            cursor += rows;
            audioRows += rows;
            if (rows > 0) audioSpans++;
        }
        Assert.Equal(VisionEnd, expanded[cursor]);
        Assert.Equal(trackRows, audioRows);
        Assert.Equal(plan.Groups.Count + audioSpans, spans.Count);
        Assert.Equal(audioSpans, spans.Count(s => s.ContentId.Contains(":s:", StringComparison.Ordinal)));
    }

    [Lfm3OmniFact(BackendType.GgmlMetal)] public void SampledFrames_PairByTheirTimes_GgmlMetal() => SampledFrames(BackendType.GgmlMetal);
    [Lfm3OmniFact(BackendType.Cpu)] public void SampledFrames_PairByTheirTimes_Cpu() => SampledFrames(BackendType.Cpu);

    /// <summary>Frames a client sampled (timed, no file): consecutive frames up to 0.575 s apart share a pair labelled with
    /// their mean time; sparser ones each fill a patch with their own time (<see cref="Lfm3Video.PlanFrames"/>).</summary>
    private void SampledFrames(BackendType lane)
    {
        Lfm3RuntimeModel model = Model(lane);
        string[] frames = Enumerable.Range(0, 4).Select(i => WritePicture($"f{i}.png", 128, 96, 10 + i)).ToArray();
        var history = new List<ChatMessage>
        {
            new() { Role = "user", Content = "Q", IsVideo = true, ImagePaths = frames.ToList(), ImageTimestamps = [0.0, 0.5, 1.0, 3.0] },
        };
        List<int> tokens = Render(model, history);
        int at = tokens.IndexOf(VideoToken);
        var (expanded, spans) = Prepare(model, history, tokens, "frames");
        model.MultimodalInjector.ClearPreparedPromptState("frames");
        Assert.Equal(3, spans.Count);
        string text = model.Tokenizer.Decode(expanded.Skip(at).Take(expanded.Count - tokens.Count).Where(t => t < 124000).ToList());
        Assert.Contains("<0.2 seconds>", text);
        Assert.Contains("<1.0 seconds>", text);
        Assert.Contains("<3.0 seconds>", text);
        Assert.All(spans, s => Assert.StartsWith("lfm3vf:", s.ContentId));
    }

    // ================================================================ split encodes

    [Lfm3OmniFact(BackendType.GgmlMetal)] public void LargeImage_InParts_EqualsOneProgram_GgmlMetal() => LargeImageInParts(BackendType.GgmlMetal);
    [Lfm3OmniFact(BackendType.Mlx)] public void LargeImage_InParts_EqualsOneProgram_Mlx() => LargeImageInParts(BackendType.Mlx);
    [Lfm3OmniFact(BackendType.Cpu)] public void LargeImage_InParts_EqualsOneProgram_Cpu() => LargeImageInParts(BackendType.Cpu);

    /// <summary>A 1920 x 1088 frame (8160 patches, 2040 rows) encoded with a yield callback runs as two programs by default
    /// and as four with YieldPatches 2048, split at layer boundaries with the callback between them: its rows are the
    /// single program's (F32 activations between parts; bit-identical on GGML and the managed CPU, within 5e-3 on MLX,
    /// which compiles and fuses each program on its own).</summary>
    private void LargeImageInParts(BackendType lane)
    {
        Lfm3RuntimeModel model = Model(lane);
        Lfm3VisionEncoder vision = model.VisionEncoder!;
        Lfm3RgbImage image = Lfm3ImageProcessor.LoadImage(WritePicture("hd.png", 1920, 1080, 31));
        (int w, int h) = vision.ImageSize(image.Width, image.Height);
        Lfm3RgbImage frame = Lfm3ImageProcessor.Resize(image, w, h);
        int patches = w / 16 * (h / 16);
        Assert.Equal(2, vision.PartsFor(patches, canYield: true));
        Assert.Equal(1, vision.PartsFor(patches, canYield: false));
        Assert.Equal(1, vision.PartsFor(4096, canYield: true));
        Lfm3VisionEmbedding whole, split;
        int yields = 0;
        int previous = vision.YieldPatches;
        vision.YieldPatches = 2048;
        try
        {
            Assert.Equal(4, vision.PartsFor(patches, canYield: true));
            lock (model.GpuComputeLock)
            {
                whole = vision.EncodeFrame(frame);
                split = vision.EncodeFrame(frame, () => { yields++; model.YieldGpuComputeLock(); });
            }
        }
        finally
        {
            vision.YieldPatches = previous;
        }
        Assert.Equal(3, yields);
        Assert.Equal(whole.RowCount, split.RowCount);
        double num = 0, den = 0, dot = 0, sa = 0, sb = 0;
        for (int i = 0; i < whole.Rows.Length; i++)
        {
            double a = whole.Rows[i], b = split.Rows[i];
            num += (a - b) * (a - b); den += a * a; dot += a * b; sa += a * a; sb += b * b;
        }
        double rel = Math.Sqrt(num / den), cos = dot / Math.Sqrt(sa * sb);
        _output.WriteLine($"[{lane}] {whole.RowCount} rows, split in 4 vs one program: rel RMS {rel:E2}, cos {cos:F8}");
        Assert.True(rel <= (lane == BackendType.Mlx ? 5e-3 : 1e-6), $"rel RMS {rel:E2}");
    }

    // ================================================================ prefix reuse

    [Lfm3OmniFact(BackendType.GgmlMetal)] public Task PrefixReuse_FollowsTheSameImage_StopsAtAnotherOne_GgmlMetal() => PrefixReuse(BackendType.GgmlMetal);
    [Lfm3OmniFact(BackendType.Mlx)] public Task PrefixReuse_FollowsTheSameImage_StopsAtAnotherOne_Mlx() => PrefixReuse(BackendType.Mlx);

    /// <summary>
    /// Through the engine with the prefix cache: turn 2, a follow-up about the image of turn 1, reuses the KV through the
    /// image and the whole previous turn, and its greedy tokens equal a cold engine's; a new conversation whose image is
    /// another one reuses at most the text before that image's span, and answers as a cold engine does.
    /// </summary>
    private async Task PrefixReuse(BackendType lane)
    {
        Lfm3RuntimeModel model = Model(lane);
        // 640 x 480: 300 rows, so turn 1 is longer than a cache block.
        string imageA = WritePicture("ra.png", 640, 480, 21);
        string imageB = WritePicture("rb.png", 640, 480, 22);
        const string system = "You are a helpful assistant. Answer briefly.";
        List<int> Text(string s) => model.Tokenizer.Encode(s, addSpecial: false);

        (List<int> Tokens, List<ChatMessage> History) Turn1(string image)
        {
            var t = new List<int>(model.Tokenizer.Encode("<|im_start|>system\n" + system + "<|im_end|>\n<|im_start|>user\n", addSpecial: true));
            t.Add(VisionStart); t.Add(ImageToken); t.Add(VisionEnd);
            t.AddRange(Text("Describe the picture in one sentence.<|im_end|>\n<|im_start|>assistant\n<think></think>"));
            return (t, new List<ChatMessage> { new() { Role = "user", Content = "Describe", ImagePaths = [image] } });
        }
        (List<int> Tokens, List<ChatMessage> History) Turn2(List<int> previous, List<int> output, List<ChatMessage> history)
        {
            var t = new List<int>(previous);
            t.AddRange(output);
            int imEnd = model.Tokenizer.LookupToken("<|im_end|>");
            if (output.Count == 0 || output[^1] != imEnd) t.Add(imEnd);
            t.AddRange(Text("\n<|im_start|>user\nWhich colour covers most of it?<|im_end|>\n<|im_start|>assistant\n<think></think>"));
            return (t, history);
        }

        var config = new SchedulerConfig
        {
            MaxNumBatchedTokens = 4096, MaxNumRunningSequences = 4, MaxPrefillChunkSize = 1024, SoloPrefillChunkSize = 1024,
            NumBlocks = 64, BlockSize = 256, EnablePrefixCaching = true, DecodeQuantumTokens = 256,
        };
        const int newTokens = 16;
        async Task<(int Reused, List<int> Output, int ImageStart, int ImageEnd)> Generate(InferenceEngine engine,
            (List<int> Tokens, List<ChatMessage> History) turn, string id)
        {
            var (tokens, spans) = Prepare(model, turn.History, turn.Tokens, id);
            try
            {
                var seq = new SequenceState(id, tokens, newTokens, 256, SamplingConfig.Greedy, mediaSpans: spans, cacheScope: "lfm3-omni-reuse");
                var handle = engine.SubmitRequest(seq);
                var output = new List<int>();
                await foreach (int t in handle.Tokens.ReadAllAsync())
                    output.Add(t);
                var completion = await handle.Completion;
                PromptMediaSpan image = Assert.Single(spans);
                return (completion.PrefixCacheReusedTokens, output, image.Start, image.End);
            }
            finally
            {
                model.MultimodalInjector.ClearPreparedPromptState(id);
            }
        }

        var a1 = Turn1(imageA);
        var b1 = Turn1(imageB);
        List<int> outA1, outA2, outB1;
        int reusedA2, reusedB1, imageStart, imageEnd, expandedA1;
        using (var engine = new InferenceEngine(model, config, NullLogger.Instance))
        {
            var r1 = await Generate(engine, a1, "a1");
            outA1 = r1.Output;
            (imageStart, imageEnd) = (r1.ImageStart, r1.ImageEnd);
            expandedA1 = a1.Tokens.Count - 1 + 300;
            var r2 = await Generate(engine, Turn2(a1.Tokens, outA1, a1.History), "a2");
            (reusedA2, outA2) = (r2.Reused, r2.Output);
            var r3 = await Generate(engine, b1, "b1");
            (reusedB1, outB1) = (r3.Reused, r3.Output);
        }
        _output.WriteLine($"[{lane}] image span [{imageStart}, {imageEnd}); turn 1 prompt {expandedA1} + {outA1.Count} generated: " +
            $"turn 2 reused {reusedA2}; another image reused {reusedB1}");
        _output.WriteLine($"A1: {model.Tokenizer.Decode(outA1)}\nA2: {model.Tokenizer.Decode(outA2)}\nB1: {model.Tokenizer.Decode(outB1)}");

        List<int> coldA2, coldB1;
        using (var engine = new InferenceEngine(model, config, NullLogger.Instance))
            coldA2 = (await Generate(engine, Turn2(a1.Tokens, outA1, a1.History), "cold-a2")).Output;
        using (var engine = new InferenceEngine(model, config, NullLogger.Instance))
            coldB1 = (await Generate(engine, b1, "cold-b1")).Output;

        Assert.True(reusedA2 >= imageEnd, $"the follow-up reused {reusedA2} tokens, stopping before the image ends at {imageEnd}");
        Assert.True(reusedA2 >= expandedA1 + outA1.Count - 1 - 1,
            $"the follow-up reused {reusedA2} of the {expandedA1 + outA1.Count - 1} tokens turn 1 left");
        Assert.True(reusedB1 <= imageStart, $"another image reused {reusedB1} tokens, past its span's start {imageStart}");
        Assert.Equal(coldA2, outA2);
        Assert.Equal(coldB1, outB1);
        Assert.NotEqual(outA1, outB1);
    }

    // ================================================================ encoder concurrency

    [Lfm3OmniFact(BackendType.GgmlMetal)] public void LongClips_EncodedConcurrently_WithGpuYields_DoNotDeadlock_GgmlMetal() => LongClips(BackendType.GgmlMetal);
    [Lfm3OmniFact(BackendType.Cpu)] public void LongClips_EncodedConcurrently_WithGpuYields_DoNotDeadlock_Cpu() => LongClips(BackendType.Cpu);

    /// <summary>Two clips longer than one chunk, each encoded under the model's GPU lock and yielding it between chunks
    /// (what preparation does): the second must not wait on the first's shared state while holding the lock the first
    /// needs back. Both finish, with the rows of the clip encoded alone.</summary>
    private void LongClips(BackendType lane)
    {
        Lfm3RuntimeModel model = Model(lane);
        Lfm3AudioEncoder audio = model.AudioEncoder!;
        int previous = audio.MaxChunkFrames;
        audio.MaxChunkFrames = 64;
        try
        {
            var samples = new float[16000];
            for (int i = 0; i < samples.Length; i++) samples[i] = 0.2f * MathF.Sin(2 * MathF.PI * 330f * i / 16000f) * (1 + MathF.Sin(i / 900f));
            float[] alone;
            lock (model.GpuComputeLock)
                alone = audio.Encode(samples, model.YieldGpuComputeLock);

            float[][] results = new float[2][];
            using var bothStarted = new Barrier(2);
            var threads = Enumerable.Range(0, 2).Select(k => new Thread(() =>
            {
                lock (model.GpuComputeLock)
                {
                    bool first = true;
                    results[k] = audio.Encode(samples, () =>
                    {
                        if (first)
                        {
                            first = false;
                            Monitor.Exit(model.GpuComputeLock);
                            try { bothStarted.SignalAndWait(TimeSpan.FromSeconds(60)); }
                            finally { Monitor.Enter(model.GpuComputeLock); }
                        }
                        model.YieldGpuComputeLock();
                    });
                }
            })).ToArray();
            foreach (Thread t in threads) t.Start();
            foreach (Thread t in threads)
                Assert.True(t.Join(TimeSpan.FromMinutes(3)), "two concurrent long clips deadlocked");
            foreach (float[] r in results)
            {
                Assert.Equal(alone.Length, r.Length);
                double num = 0, den = 0;
                for (int i = 0; i < r.Length; i++) { num += (r[i] - alone[i]) * (r[i] - alone[i]); den += alone[i] * alone[i]; }
                Assert.True(Math.Sqrt(num / den) < 1e-4, $"rows differ from the clip encoded alone by {Math.Sqrt(num / den):E2} RMS");
            }
        }
        finally
        {
            audio.MaxChunkFrames = previous;
        }
    }
}
