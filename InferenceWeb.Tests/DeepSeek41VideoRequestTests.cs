// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Text.Json;
using TensorSharp.Models;
using TensorSharp.Models.Media;
using TensorSharp.Runtime;
using TensorSharp.Server.Hosting;
using TensorSharp.Server.RequestParsers;

namespace InferenceWeb.Tests;

public sealed class DeepSeek41VideoRequestTests : IDisposable
{
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "dsv41-video-" + Guid.NewGuid().ToString("N"));
    private readonly IVideoDecoder _previous = MediaCodecs.Video;
    private readonly FakeDecoder _decoder = new();
    public DeepSeek41VideoRequestTests()
    {
        Directory.CreateDirectory(_directory);
        MediaCodecs.Video = _decoder;
    }
    public void Dispose()
    {
        MediaCodecs.Video = _previous;
        if (Directory.Exists(_directory)) Directory.Delete(_directory, true);
    }

    [Theory]
    [InlineData("deepseek41")]
    [InlineData("gemma4")]
    public void OpenAiVideo_UsesExistingDecoderAndRetainsSourceTimesBesideOrderedImages(string architecture)
    {
        var uploads = new UploadStoragePolicy(_directory);
        using var json = JsonDocument.Parse("""
            [{"role":"user","content":[
              {"type":"image_url","image_url":{"url":"data:image/png;base64,AQ=="}},
              {"type":"video_url","video_url":{"url":"data:video/mp4;base64,AQID","fps":1,"max_frames":3}},
              {"type":"image_url","image_url":{"url":"data:image/png;base64,Ag=="}},
              {"type":"text","text":"Describe the numbered frames in time order."}]}]
            """);
        var message = Assert.Single(ChatMessageParser.ParseOpenAI(json.RootElement, uploads, architecture: architecture));
        Assert.True(message.IsVideo);
        Assert.Equal(new[] { 0, 2, 4 }, _decoder.Requested);
        Assert.Equal(5, message.ImagePaths.Count);
        Assert.Equal(new double?[] { null, 0, 1, 2, null }, message.ImageTimestamps);
        Assert.All(message.ImagePaths, path => Assert.True(File.Exists(path)));
        Assert.Equal(Directory.EnumerateFiles(_directory).Sum(path => new FileInfo(path).Length), uploads.UsedBytes);
        if (architecture == "deepseek41")
        {
            string prompt = ChatTemplate.RenderDeepSeek41(new() { message });
            string image = ChatTemplate.DeepSeek41ImagePlaceholder;
            Assert.Contains(image + "Frame at 0 seconds: " + image + "\nFrame at 1 seconds: " + image +
                "\nFrame at 2 seconds: " + image + "\n" + image, prompt);
        }
        else
        {
            const string frames = "<|image> 00:00 <|image> 00:01 <|image> 00:02 <|image><|image>";
            string prompt = ChatTemplate.RenderGemma4(new() { message });
            Assert.Contains(frames, prompt);
            Assert.DoesNotContain("<|video>", prompt);
            Assert.StartsWith(frames, ChatTemplate.InjectMultimodalTokens(new() { message }, "gemma4")[0].Content);
        }
        var structured = StructuredOutputPrompt.Apply(new() { message }, StructuredOutputFormat.JsonObject());
        Assert.Equal(message.ImageTimestamps, structured.Last().ImageTimestamps);
    }

    [Fact]
    public void VideoFrameCap_SamplesEntireClipAndKeepsActualSourceTimes()
    {
        var uploads = new UploadStoragePolicy(_directory);
        using var json = Request("data:video/mp4;base64,AQID", "\"fps\":1,\"max_frames\":2");
        var message = Assert.Single(ChatMessageParser.ParseOpenAI(json.RootElement, uploads, architecture: "deepseek41"));
        Assert.Equal(new[] { 0, 4 }, _decoder.Requested);
        Assert.Equal(new double?[] { 0, 2 }, message.ImageTimestamps);
    }

    [Theory]
    [InlineData("http://localhost/private.mp4", "\"fps\":1,\"max_frames\":3")]
    [InlineData("data:video/mp4;base64,?", "\"fps\":1,\"max_frames\":3")]
    [InlineData("data:video/mp4;base64,AQID", "\"fps\":0,\"max_frames\":3")]
    [InlineData("data:video/mp4;base64,AQID", "\"fps\":1,\"max_frames\":65")]
    public void InvalidVideoRequest_FailsBeforeWritingOrDecoding(string url, string options)
    {
        var uploads = new UploadStoragePolicy(_directory);
        using var json = Request(url, options);
        Assert.Throws<JsonException>(() => ChatMessageParser.ParseOpenAI(json.RootElement, uploads, architecture: "deepseek41"));
        Assert.Empty(_decoder.Requested);
        Assert.Empty(Directory.EnumerateFiles(_directory));
        Assert.Equal(0, uploads.UsedBytes);
    }

    [Fact]
    public void DecoderFailure_CleansVideoAndReleasesUploadReservation()
    {
        var uploads = new UploadStoragePolicy(_directory);
        _decoder.Fail = true;
        using var json = Request("data:video/mp4;base64,AQID", "\"fps\":1,\"max_frames\":3");
        Assert.Throws<JsonException>(() => ChatMessageParser.ParseOpenAI(json.RootElement, uploads, architecture: "deepseek41"));
        Assert.Empty(Directory.EnumerateFiles(_directory));
        Assert.Equal(0, uploads.UsedBytes);
    }

    [Fact]
    public void OtherModel_DoesNotSilentlyDiscardUnsupportedVideoPart()
    {
        var uploads = new UploadStoragePolicy(_directory);
        using var json = Request("data:video/mp4;base64,AQID", "\"fps\":1,\"max_frames\":3");
        Assert.Throws<JsonException>(() => ChatMessageParser.ParseOpenAI(json.RootElement, uploads, architecture: "deepseek4"));
        Assert.Empty(_decoder.Requested);
    }

    private static JsonDocument Request(string url, string options)
        => JsonDocument.Parse("[{\"role\":\"user\",\"content\":[{\"type\":\"video_url\",\"video_url\":{\"url\":" +
            JsonSerializer.Serialize(url) + "," + options + "}}]}]");

    private sealed class FakeDecoder : IVideoDecoder
    {
        public bool Fail;
        public List<int> Requested = new();
        public VideoInfo Probe(string path) => Fail ? throw new InvalidDataException("bad video") : new(2, 6, 2, 2);
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
