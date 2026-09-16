// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Text;
using System.Text.Json;
using TensorSharp.Models;
using TensorSharp.Models.Media;
using TensorSharp.Runtime;
using TensorSharp.Server.Hosting;
using TensorSharp.Server.RequestParsers;

namespace InferenceWeb.Tests;

/// <summary>
/// Ordered video-frame input for Qwen 3.8 Flash Next (<c>qwen4exp</c>): the OpenAI
/// <c>video_url</c> part is sampled into timed frames, the prompt renders them as the
/// Qwen3-VL video layout (one <c>&lt;|video_pad|&gt;</c> block per temporal pair, each
/// labelled with its time, wrapped once per clip), and the injector's prompt layout
/// gives consecutive pairs increasing temporal M-RoPE ids. Encoder and model
/// numerics are covered separately.
/// </summary>
public sealed class Qwen4ExpVideoRequestTests : IDisposable
{
    private const string Start = QwenVideoFrames.VisionStart;
    private const string End = QwenVideoFrames.VisionEnd;
    private const string Image = Start + QwenVideoFrames.ImagePad + End;
    private const string VideoPad = QwenVideoFrames.VideoPad;

    private readonly string _directory = Path.Combine(Path.GetTempPath(), "q4e-video-" + Guid.NewGuid().ToString("N"));
    private readonly IVideoDecoder _previous = MediaCodecs.Video;
    private readonly FakeDecoder _decoder = new();

    public Qwen4ExpVideoRequestTests()
    {
        Directory.CreateDirectory(_directory);
        MediaCodecs.Video = _decoder;
    }

    public void Dispose()
    {
        MediaCodecs.Video = _previous;
        if (Directory.Exists(_directory)) Directory.Delete(_directory, true);
    }

    [Fact]
    public void OpenAiVideo_IsAcceptedForQwen4ExpAndRendersTimedPairsOncePerClip()
    {
        var uploads = new UploadStoragePolicy(_directory);
        using var json = JsonDocument.Parse("""
            [{"role":"user","content":[
              {"type":"image_url","image_url":{"url":"data:image/png;base64,AQ=="}},
              {"type":"video_url","video_url":{"url":"data:video/mp4;base64,AQID","fps":1,"max_frames":3}},
              {"type":"image_url","image_url":{"url":"data:image/png;base64,Ag=="}},
              {"type":"text","text":"Describe the numbered frames in time order."}]}]
            """);
        var message = Assert.Single(ChatMessageParser.ParseOpenAI(json.RootElement, uploads, architecture: "qwen4exp"));
        Assert.True(message.IsVideo);
        Assert.Equal(new[] { 0, 2, 4 }, _decoder.Requested);
        Assert.Equal(5, message.ImagePaths!.Count);
        Assert.Equal(new double?[] { null, 0, 1, 2, null }, message.ImageTimestamps);
        Assert.All(message.ImagePaths, path => Assert.True(File.Exists(path)));

        // Three frames at 0, 1, 2 s merge into pairs (0,1) at 0.5 s and (2,2) at 2.0 s:
        // the odd tail repeats its last frame, as the Qwen-VL processor pads a clip.
        string clip = Start
            + "<0.5 seconds>" + Start + VideoPad + End
            + "<2.0 seconds>" + Start + VideoPad + End
            + End;
        string content = ChatTemplate.InjectMultimodalTokens(new() { message }, "qwen4exp")[0].Content!;
        Assert.StartsWith(Image + clip + Image + "Describe the numbered frames", content);
        Assert.Equal(2, Count(content, VideoPad));
        Assert.Equal(2, Count(content, QwenVideoFrames.ImagePad));

        var items = QwenVideoFrames.Layout(message);
        Assert.Equal(3, items.Count);
        Assert.False(items[0].IsVideo);
        Assert.True(items[1].IsVideo);
        Assert.Equal(new[] { new QwenVideoFrames.Group(1, 2, 0.5), new QwenVideoFrames.Group(3, 3, 2.0) }, items[1].Groups);
        Assert.False(items[2].IsVideo);
        Assert.Equal(4, items[2].ImageIndex);
    }

    [Fact]
    public void TwoVideoParts_StayTwoClipsAndKeepTheirOwnTimeOrder()
    {
        var uploads = new UploadStoragePolicy(_directory);
        using var json = JsonDocument.Parse("""
            [{"role":"user","content":[
              {"type":"video_url","video_url":{"url":"data:video/mp4;base64,AQID","fps":1,"max_frames":2}},
              {"type":"video_url","video_url":{"url":"data:video/mp4;base64,AQID","fps":1,"max_frames":2}},
              {"type":"text","text":"Which clip comes first?"}]}]
            """);
        var message = Assert.Single(ChatMessageParser.ParseOpenAI(json.RootElement, uploads, architecture: "qwen4exp"));
        Assert.Equal(new double?[] { 0, 2, 0, 2 }, message.ImageTimestamps);
        var items = QwenVideoFrames.Layout(message);
        Assert.Equal(2, items.Count);
        Assert.All(items, item => Assert.True(item.IsVideo));
        Assert.Equal(new[] { new QwenVideoFrames.Group(0, 1, 1.0) }, items[0].Groups);
        Assert.Equal(new[] { new QwenVideoFrames.Group(2, 3, 1.0) }, items[1].Groups);

        string content = ChatTemplate.InjectMultimodalTokens(new() { message }, "qwen4exp")[0].Content!;
        string clip = Start + "<1.0 seconds>" + Start + VideoPad + End + End;
        Assert.StartsWith(clip + clip + "Which clip", content);
    }

    [Fact]
    public void LegacyFramesWithoutTimes_StayStillImagesForQwen()
    {
        // A Web UI video upload extracts frames without source times; that history
        // must render exactly as it always has (one image span per frame).
        var message = new ChatMessage
        {
            Role = "user",
            Content = "what happened?",
            IsVideo = true,
            ImagePaths = new() { "a.png", "b.png", "c.png" },
        };
        string content = ChatTemplate.InjectMultimodalTokens(new() { message }, "qwen4exp")[0].Content!;
        Assert.Equal(Image + Image + Image + "what happened?", content);
        Assert.DoesNotContain(VideoPad, content);
        Assert.All(QwenVideoFrames.Layout(message), item => Assert.False(item.IsVideo));
    }

    [Fact]
    public void Qwen35_StillRefusesVideoUrlRatherThanSilentlyDroppingIt()
    {
        var uploads = new UploadStoragePolicy(_directory);
        using var json = JsonDocument.Parse("""
            [{"role":"user","content":[
              {"type":"video_url","video_url":{"url":"data:video/mp4;base64,AQID","fps":1,"max_frames":2}}]}]
            """);
        Assert.Throws<JsonException>(() => ChatMessageParser.ParseOpenAI(json.RootElement, uploads, architecture: "qwen35"));
        Assert.Empty(_decoder.Requested);
        Assert.True(ChatProtocolRegistry.For("qwen4exp")!.CapsVideoFrames);
        Assert.False(ChatProtocolRegistry.For("qwen35")!.CapsVideoFrames);
    }

    [Theory]
    [InlineData(0.25, "0.2")]
    [InlineData(0.75, "0.8")]
    [InlineData(1.0, "1.0")]
    [InlineData(2.349, "2.3")]
    public void TimeLabels_UseOneDecimalWithTiesToEven(double seconds, string expected)
        => Assert.Equal(expected, QwenVideoFrames.FormatSeconds(seconds));

    [Fact]
    public void Layout_OddClipRepeatsLastFrameAndNonIncreasingTimeStartsNewClip()
    {
        var message = new ChatMessage
        {
            Role = "user",
            IsVideo = true,
            ImagePaths = new() { "s.png", "f0.png", "f1.png", "f2.png", "f3.png", "f4.png", "g0.png" },
            ImageTimestamps = new() { null, 0, 0.5, 1.0, 1.5, 2.0, 0 },
        };
        var items = QwenVideoFrames.Layout(message);
        Assert.Equal(3, items.Count);
        Assert.Equal(0, items[0].ImageIndex);
        Assert.Equal(new[]
        {
            new QwenVideoFrames.Group(1, 2, 0.25),
            new QwenVideoFrames.Group(3, 4, 1.25),
            new QwenVideoFrames.Group(5, 5, 2.0),
        }, items[1].Groups);
        Assert.Equal(new[] { new QwenVideoFrames.Group(6, 6, 0) }, items[2].Groups);

        var sb = new StringBuilder();
        QwenVideoFrames.AppendPlaceholders(message, sb);
        Assert.Equal(Image
            + Start + "<0.2 seconds>" + Start + VideoPad + End + "<1.2 seconds>" + Start + VideoPad + End
            + "<2.0 seconds>" + Start + VideoPad + End + End
            + Start + "<0.0 seconds>" + Start + VideoPad + End + End, sb.ToString());
    }

    [Fact]
    public void Layout_RejectsNegativeOrNonFiniteFrameTimes()
    {
        var message = new ChatMessage
        {
            Role = "user", IsVideo = true,
            ImagePaths = new() { "f0.png", "f1.png" },
            ImageTimestamps = new() { 0, double.NaN },
        };
        Assert.Throws<ArgumentOutOfRangeException>(() => QwenVideoFrames.Layout(message));
        message.ImageTimestamps = new() { -1, 0 };
        Assert.Throws<ArgumentOutOfRangeException>(() => QwenVideoFrames.Layout(message));
    }

    [Fact]
    public void PromptLayout_ConsecutivePairsGetIncreasingTemporalIdsAndTextResumesAfterTheClip()
    {
        const int imagePad = 100, videoPad = 101;
        var pads = new HashSet<int> { imagePad, videoPad };
        // <|vision_start|> <t0> <|vision_start|> PAIR <|vision_end|> <t1> <|vision_start|> PAIR <|vision_end|> <|vision_end|> text
        var tokens = new List<int> { 7, 8, 7, videoPad, 9, 8, 7, videoPad, 9, 9, 5 };
        var spans = new[]
        {
            new ModelMultimodalInjector.QwenVLVisionSpan(videoPad, 6, 2, 3),
            new ModelMultimodalInjector.QwenVLVisionSpan(videoPad, 6, 2, 3),
        };
        var (expanded, positions, starts) = ModelMultimodalInjector.LayoutQwenVLPrompt(tokens, spans, pads);

        Assert.Equal(new[] { 7, 8, 7, videoPad, videoPad, videoPad, videoPad, videoPad, videoPad, 9, 8, 7,
            videoPad, videoPad, videoPad, videoPad, videoPad, videoPad, 9, 9, 5 }, expanded);
        Assert.Equal(new[] { 3, 12 }, starts);
        Assert.Equal(3 * expanded.Count, positions.Length);

        int[] Axis(int token, int axis) => new[] { positions[3 * token + axis] };
        // First pair at running position 3: (3, 3+h, 3+w); the text after it resumes
        // at 3 + max(2,3) = 6, so the second pair starts at 9 and is strictly later.
        Assert.Equal(new[] { 3, 3, 3, 3, 3, 3 }, Enumerable.Range(3, 6).Select(t => positions[3 * t]));
        Assert.Equal(new[] { 3, 3, 3, 4, 4, 4 }, Enumerable.Range(3, 6).Select(t => positions[3 * t + 1]));
        Assert.Equal(new[] { 3, 4, 5, 3, 4, 5 }, Enumerable.Range(3, 6).Select(t => positions[3 * t + 2]));
        Assert.Equal(6, positions[3 * 9]);
        Assert.Equal(new[] { 9, 9, 9, 9, 9, 9 }, Enumerable.Range(12, 6).Select(t => positions[3 * t]));
        Assert.Equal(new[] { 9, 9, 9, 10, 10, 10 }, Enumerable.Range(12, 6).Select(t => positions[3 * t + 1]));
        Assert.Equal(new[] { 9, 10, 11, 9, 10, 11 }, Enumerable.Range(12, 6).Select(t => positions[3 * t + 2]));
        Assert.True(positions[3 * 12] > positions[3 * 3]);
        // The pairs differ by a pure shift of the running position on every axis.
        for (int i = 0; i < 6; i++)
            for (int axis = 0; axis < 3; axis++)
                Assert.Equal(positions[3 * (3 + i) + axis] + 6, positions[3 * (12 + i) + axis]);
        Assert.Equal(new[] { 12, 12, 12 }, new[] { Axis(18, 0)[0], Axis(18, 1)[0], Axis(18, 2)[0] });
        Assert.Equal(new[] { 14, 14, 14 }, new[] { Axis(20, 0)[0], Axis(20, 1)[0], Axis(20, 2)[0] });
    }

    [Fact]
    public void PromptLayout_ImageAndVideoSpansExpandTheirOwnPadTokens()
    {
        const int imagePad = 100, videoPad = 101;
        var pads = new HashSet<int> { imagePad, videoPad };
        var tokens = new List<int> { imagePad, 1, videoPad, 2 };
        var spans = new[]
        {
            new ModelMultimodalInjector.QwenVLVisionSpan(imagePad, 4, 2, 2),
            new ModelMultimodalInjector.QwenVLVisionSpan(videoPad, 1, 1, 1),
        };
        var (expanded, positions, starts) = ModelMultimodalInjector.LayoutQwenVLPrompt(tokens, spans, pads);
        Assert.Equal(new[] { imagePad, imagePad, imagePad, imagePad, 1, videoPad, 2 }, expanded);
        Assert.Equal(new[] { 0, 5 }, starts);
        Assert.Equal(new[] { 0,0,0, 0,0,1, 0,1,0, 0,1,1, 2,2,2, 3,3,3, 4,4,4 }, positions);
    }

    [Fact]
    public void PromptLayout_RefusesAPlaceholderOfTheWrongKindAndReportsMissingOnes()
    {
        const int imagePad = 100, videoPad = 101;
        var pads = new HashSet<int> { imagePad, videoPad };
        var video = new[] { new ModelMultimodalInjector.QwenVLVisionSpan(videoPad, 1, 1, 1) };
        Assert.Throws<InvalidOperationException>(() =>
            ModelMultimodalInjector.LayoutQwenVLPrompt(new List<int> { imagePad }, video, pads));

        var (expanded, positions, starts) = ModelMultimodalInjector.LayoutQwenVLPrompt(new List<int> { 5, 6 }, video, pads);
        Assert.Equal(new[] { 5, 6 }, expanded);
        Assert.Equal(new[] { -1 }, starts);
        Assert.Equal(new[] { 0,0,0, 1,1,1 }, positions);
    }

    private static int Count(string text, string needle)
    {
        int count = 0;
        for (int i = text.IndexOf(needle, StringComparison.Ordinal); i >= 0;
             i = text.IndexOf(needle, i + needle.Length, StringComparison.Ordinal))
            count++;
        return count;
    }

    private sealed class FakeDecoder : IVideoDecoder
    {
        public List<int> Requested = new();
        public VideoInfo Probe(string path) => new(2, 6, 2, 2);
        public void ReadFrames(string path, IReadOnlyList<int> indices, FrameCallback onFrame)
        {
            foreach (int index in indices)
            {
                Requested.Add(index);
                byte[] rgb = Enumerable.Repeat((byte)(index * 20), 12).ToArray();
                onFrame(index, rgb, 2, 2, 6, PixelLayout.Rgb);
            }
        }
    }
}
