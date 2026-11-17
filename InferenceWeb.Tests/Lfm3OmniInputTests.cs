// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Text.Json;
using TensorSharp.Models;
using TensorSharp.Models.Architecture;
using TensorSharp.Models.Media;
using TensorSharp.Runtime;
using TensorSharp.Server;
using TensorSharp.Server.Hosting;
using TensorSharp.Server.RequestParsers;

namespace InferenceWeb.Tests;

/// <summary>
/// LFM3-Omni media input without weights: how a message's attachments become media items
/// (<see cref="Lfm3MediaLayout"/>) and template spans, how the request surfaces carry a video's own file
/// (<see cref="ChatMessage.VideoFilePaths"/>: Web UI <c>videoFilePaths</c>, OpenAI <c>video_url</c>, Responses
/// <c>input_video</c>), the refusals of media without a tower (LFM3-1.7B, Omni without its mmproj), and the lock
/// discipline the encoders rely on. The towers themselves are <see cref="Lfm3OmniModelTests"/> (real weights).
/// </summary>
public sealed class Lfm3OmniInputTests : IDisposable
{
    private const string Arch = "lfm3";
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "lfm3-omni-in-" + Guid.NewGuid().ToString("N"));
    private readonly IVideoDecoder _previous = MediaCodecs.Video;
    private readonly FakeDecoder _decoder = new();

    public Lfm3OmniInputTests()
    {
        Directory.CreateDirectory(_directory);
        MediaCodecs.Video = _decoder;
    }

    public void Dispose()
    {
        MediaCodecs.Video = _previous;
        try { Directory.Delete(_directory, true); } catch (IOException) { }
    }

    private static string Kinds(ChatMessage m) => string.Join(",", Lfm3MediaLayout.Items(m).Select(i => i.Kind switch
    {
        Lfm3MediaKind.Image => $"i{i.Index}",
        Lfm3MediaKind.Audio => $"a{i.Index}",
        _ => i.VideoFile != null ? $"v[{i.VideoFile}]" : $"v({string.Join(" ", i.Frames!)})",
    }));

    // ---------------------------------------------------------------- layout

    [Fact]
    public void Layout_StillsAndTimedFrameRunsKeepTheirOrder_AudioFollows()
    {
        var m = new ChatMessage
        {
            Role = "user", IsVideo = true, ImagePaths = ["s0", "f0", "f1", "f2", "s1", "g0", "g1"],
            ImageTimestamps = [null, 0.0, 0.5, 1.0, null, 0.0, 0.5], AudioPaths = ["a", "b"],
        };
        // Two video_url parts stay two videos: the second's times restart.
        Assert.Equal("i0,v(1 2 3),i4,v(5 6),a0,a1", Kinds(m));
        Assert.Equal(new[] { 0.0, 0.5, 1.0 }, Lfm3MediaLayout.Items(m)[1].FrameSeconds);
    }

    [Fact]
    public void Layout_VideoFilesTakeTheirRunsPlace_AndTheFramesAreNotUsed()
    {
        var m = new ChatMessage
        {
            Role = "user", IsVideo = true, ImagePaths = ["s0", "f0", "f1", "g0"], ImageTimestamps = [null, 0.0, 0.5, 0.0],
            VideoFilePaths = ["a.mp4", "b.mp4"],
        };
        Assert.Equal("i0,v[a.mp4],v[b.mp4]", Kinds(m));
        m.ImagePaths = ["f0", "s0", "g0"];
        m.ImageTimestamps = [0.0, null, 0.0];
        Assert.Equal("v[a.mp4],i1,v[b.mp4]", Kinds(m));
    }

    [Fact]
    public void Layout_UntimedFramesAreOneRun_ExceptWhatTheUserAttached()
    {
        // The Web UI: a photo and a clip in one message; imagePaths holds the photo and the clip's extracted frames,
        // attachments names the photo and the clip (never a frame).
        var m = new ChatMessage
        {
            Role = "user", IsVideo = true, ImagePaths = ["/u/photo.png", "/u/c_0001.png", "/u/c_0002.png"],
            AttachmentPaths = ["/u/photo.png", "/u/c.mp4"],
        };
        Assert.Equal("i0,v(1 2)", Kinds(m));
        m.VideoFilePaths = ["/u/c.mp4"];
        Assert.Equal("i0,v[/u/c.mp4]", Kinds(m));
        // Without an attachment list (the CLI) every untimed entry of a video message is a frame.
        m.AttachmentPaths = null;
        Assert.Equal("v[/u/c.mp4]", Kinds(m));
    }

    [Fact]
    public void Layout_FilesThatCannotBeMatchedToRunsFollowTheStills()
    {
        // Two uploads' untimed frames merge into one run; the two files are the videos.
        var m = new ChatMessage
        {
            Role = "user", IsVideo = true, ImagePaths = ["/u/p.png", "/u/a_1.png", "/u/b_1.png"], AttachmentPaths = ["/u/p.png"],
            VideoFilePaths = ["/u/a.mp4", "/u/b.mp4"],
        };
        Assert.Equal("i0,v[/u/a.mp4],v[/u/b.mp4]", Kinds(m));
        // A clip named without frames at all, and a message that is not marked as video.
        Assert.Equal("v[x.mp4]", Kinds(new ChatMessage { Role = "user", VideoFilePaths = ["x.mp4"] }));
        Assert.Equal("i0,i1", Kinds(new ChatMessage { Role = "user", ImagePaths = ["a", "b"] }));
        Assert.Empty(Lfm3MediaLayout.Items(new ChatMessage { Role = "user", Content = "text" }));
    }

    [Fact]
    public void Placeholders_FollowTheLayout_OneTemplateSpanPerItem()
    {
        const string img = ChatTemplate.Lfm3ImagePlaceholder, vid = ChatTemplate.Lfm3VideoPlaceholder, aud = ChatTemplate.Lfm3AudioPlaceholder;
        var mixed = new ChatMessage
        {
            Role = "user", Content = "Q", IsVideo = true, ImagePaths = ["s", "f0", "f1"], ImageTimestamps = [null, 0.0, 0.5],
            AudioPaths = ["a"], VideoFilePaths = ["c.mp4"],
        };
        var fileOnly = new ChatMessage { Role = "user", Content = "Q", VideoFilePaths = ["c.mp4"] };
        List<ChatMessage> injected = ChatTemplate.InjectMultimodalTokens([mixed, fileOnly], Arch);
        Assert.Equal(img + vid + aud + "Q", injected[0].Content);
        Assert.Equal(vid + "Q", injected[1].Content);
        // The copy keeps what the layout reads, so a render of the copy agrees with the expander's view of the original.
        Assert.Equal(mixed.VideoFilePaths, injected[0].VideoFilePaths);
        Assert.Equal(Kinds(mixed), Kinds(injected[0]));
    }

    [Fact]
    public void ElidedVideo_DropsItsFileWithItsFrames()
    {
        var m = new ChatMessage
        {
            Role = "user", Content = "Q", IsVideo = true, ImagePaths = ["f0", "f1"], ImageTimestamps = [0.0, 0.5],
            VideoFilePaths = ["c.mp4"],
        };
        ChatMessage elided = ChatHistoryPreparer.WithMediaReplacedByNote(m, images: 2, audio: 0);
        Assert.Null(elided.ImagePaths);
        Assert.Null(elided.VideoFilePaths);
        Assert.Empty(Lfm3MediaLayout.Items(elided));
    }

    [Fact]
    public void ElidedFileOnlyVideo_IsCountedAndDropped()
    {
        // A video_url for a family that reads the clip itself has no frames at all.
        var m = new ChatMessage { Role = "user", Content = "Q", IsVideo = true, ImagePaths = ["s.png"], VideoFilePaths = ["c.mp4"] };
        ChatMessage elided = ChatHistoryPreparer.WithMediaReplacedByNote(m, images: 1, audio: 0);
        Assert.Null(elided.VideoFilePaths);
        Assert.Contains("earlier video", elided.Content);
        var fileOnly = new ChatMessage { Role = "user", Content = "Q", IsVideo = true, VideoFilePaths = ["c.mp4"] };
        Assert.True(ChatHistoryPreparer.HasMultimodalContent(fileOnly));
        Assert.Null(ChatHistoryPreparer.WithMediaReplacedByNote(fileOnly, images: 1, audio: 0).VideoFilePaths);
    }

    // ---------------------------------------------------------------- request surfaces

    [Fact]
    public void WebUi_ReadsVideoFilePaths_AndResolvesThemInTheUploadDirectory()
    {
        File.WriteAllBytes(Path.Combine(_directory, "clip.mp4"), new byte[] { 1 });
        File.WriteAllBytes(Path.Combine(_directory, "clip_0001.png"), new byte[] { 1 });
        using var json = JsonDocument.Parse("""
            [{"role":"user","content":"What happens?","imagePaths":["clip_0001.png"],"videoFilePaths":["clip.mp4"],"isVideo":true}]
            """);
        var messages = ChatMessageParser.ParseWebUi(json.RootElement);
        Assert.Null(ChatMessageParser.ResolveAttachmentPaths(messages, _directory));
        ChatMessage m = Assert.Single(messages);
        Assert.Equal(Path.Combine(_directory, "clip.mp4"), Assert.Single(m.VideoFilePaths!));
        Assert.Equal("v[" + Path.Combine(_directory, "clip.mp4") + "]", Kinds(m));

        using var escape = JsonDocument.Parse("""[{"role":"user","content":"x","videoFilePaths":["../../etc/passwd"]}]""");
        Assert.NotNull(ChatMessageParser.ResolveAttachmentPaths(ChatMessageParser.ParseWebUi(escape.RootElement), _directory));
    }

    [Fact]
    public void OpenAiVideoUrl_ForLfm3_SavesTheClipWithoutSamplingFrames()
    {
        var uploads = new UploadStoragePolicy(_directory);
        using var json = JsonDocument.Parse("""
            [{"role":"user","content":[
              {"type":"video_url","video_url":{"url":"data:video/mp4;base64,AQID","fps":1,"max_frames":3}},
              {"type":"image_url","image_url":{"url":"data:image/png;base64,AQ=="}},
              {"type":"text","text":"What happens?"}]}]
            """);
        ChatMessage m = Assert.Single(ChatMessageParser.ParseOpenAI(json.RootElement, uploads, architecture: Arch));
        Assert.True(m.IsVideo);
        string clip = Assert.Single(m.VideoFilePaths!);
        Assert.True(File.Exists(clip));
        Assert.Equal(new byte[] { 1, 2, 3 }, File.ReadAllBytes(clip));
        // LFM3 reads the clip itself: it is probed (an unreadable one is refused here), no frame is decoded or written.
        Assert.Equal(1, _decoder.Probes);
        Assert.Empty(_decoder.Requested);
        Assert.Single(m.ImagePaths!);
        Assert.Equal($"i0,v[{clip}]", Kinds(m));

        _decoder.Unreadable = true;
        Assert.Throws<JsonException>(() => ChatMessageParser.ParseOpenAI(json.RootElement, uploads, architecture: Arch));
    }

    [Fact]
    public void OpenAiVideoUrl_ForAFrameFamily_KeepsTheClipBesideItsSampledFrames()
    {
        var uploads = new UploadStoragePolicy(_directory);
        using var json = JsonDocument.Parse("""
            [{"role":"user","content":[
              {"type":"video_url","video_url":{"url":"data:video/mp4;base64,AQID","fps":1,"max_frames":3}},
              {"type":"text","text":"What happens?"}]}]
            """);
        ChatMessage m = Assert.Single(ChatMessageParser.ParseOpenAI(json.RootElement, uploads, architecture: "qwen4exp"));
        Assert.Equal(new[] { 0, 2, 4 }, _decoder.Requested);
        Assert.Equal(new double?[] { 0, 1, 2 }, m.ImageTimestamps);
        Assert.True(File.Exists(Assert.Single(m.VideoFilePaths!)));
    }

    [Fact]
    public void ResponsesInputVideo_IsTheVideoUrlOfTheResponsesShape()
    {
        var uploads = new UploadStoragePolicy(_directory);
        using var json = JsonDocument.Parse("""
            [{"type":"message","role":"user","content":[
              {"type":"input_video","video_url":"data:video/mp4;base64,AQID","fps":1,"max_frames":3},
              {"type":"input_image","image_url":"data:image/png;base64,AQ=="},
              {"type":"input_text","text":"What happens?"}]}]
            """);
        List<ChatMessage> messages = ChatMessageParser.ParseResponsesInput(json.RootElement, null, uploads, architecture: Arch);
        ChatMessage m = Assert.Single(messages);
        string clip = Assert.Single(m.VideoFilePaths!);
        Assert.Empty(_decoder.Requested);
        Assert.Equal($"i0,v[{clip}]", Kinds(m));
        // A frame family samples it, timed, with a still beside it keeping no time.
        ChatMessage q = Assert.Single(ChatMessageParser.ParseResponsesInput(json.RootElement, null, uploads, architecture: "qwen4exp"));
        Assert.Equal(new double?[] { 0, 1, 2, null }, q.ImageTimestamps);

        // A protocol that does not sample video refuses the part, as the chat endpoint refuses video_url.
        using var plain = JsonDocument.Parse("""
            [{"type":"message","role":"user","content":[{"type":"input_video","video_url":"data:video/mp4;base64,AQID"}]}]
            """);
        var ex = Assert.Throws<JsonException>(() => ChatMessageParser.ParseResponsesInput(plain.RootElement, null, uploads, architecture: "llama"));
        Assert.Contains("input_video", ex.Message);
        using var bad = JsonDocument.Parse("""
            [{"type":"message","role":"user","content":[{"type":"input_video","video_url":"https://example.com/a.mp4"}]}]
            """);
        Assert.Throws<JsonException>(() => ChatMessageParser.ParseResponsesInput(bad.RootElement, null, uploads, architecture: Arch));
    }

    [Fact]
    public void AudioRefusal_NamesTheOmniProjector_UntilTheAudioTowerIsLoaded()
    {
        Assert.Equal(AudioInputSupport.Lfm3Message, AudioInputSupport.UnsupportedReasonFor(Arch));
        Assert.Null(AudioInputSupport.UnsupportedReasonFor(Arch, audioEncoderLoaded: true));
        Assert.Contains("mmproj", AudioInputSupport.Lfm3Message);

        var uploads = new UploadStoragePolicy(_directory);
        using var json = JsonDocument.Parse("""
            [{"role":"user","content":[{"type":"input_audio","input_audio":{"data":"UklGRg==","format":"wav"}}]}]
            """);
        var ex = Assert.Throws<JsonException>(() => ChatMessageParser.ParseOpenAI(json.RootElement, uploads, architecture: Arch));
        Assert.Equal(AudioInputSupport.Lfm3Message, ex.Message);
        // Nothing was written before the refusal.
        Assert.Empty(Directory.EnumerateFiles(_directory));
        Assert.NotNull(ChatMessageParser.ParseOpenAI(json.RootElement, uploads, architecture: Arch, audioEncoderLoaded: true));
    }

    // ---------------------------------------------------------------- the model without towers

    [Fact]
    public void ModelWithoutTowers_RefusesEveryMediaKind_AndLeavesTextAlone()
    {
        string path = new Lfm3SyntheticModelBuilder { WeightType = SyntheticGguf.GgmlType.F32 }.Write(Path.Combine(_directory, "lfm3.gguf"));
        using ModelBase model = ModelBase.Create(path, BackendType.Cpu);
        var lfm3 = Assert.IsType<Lfm3RuntimeModel>(model);
        Assert.False(model.HasVisionEncoder());
        Assert.False(lfm3.IsAudioEncoderLoaded);
        Assert.Equal(AudioInputSupport.Lfm3Message, AudioInputSupport.UnsupportedReasonFor(model));

        var tokens = new List<int> { 1, 5, 6, 7 };
        Assert.Same(tokens, model.MultimodalInjector.ProcessPromptTokens(
            [new ChatMessage { Role = "user", Content = "hi" }], tokens, "text"));

        var image = new ChatMessage { Role = "user", Content = "Q", ImagePaths = ["x.png"] };
        var video = new ChatMessage { Role = "user", Content = "Q", IsVideo = true, VideoFilePaths = ["x.mp4"] };
        var audio = new ChatMessage { Role = "user", Content = "Q", AudioPaths = ["x.wav"] };
        foreach (ChatMessage m in new[] { image, video })
        {
            var ex = Assert.Throws<NotSupportedException>(() => model.MultimodalInjector.ProcessPromptTokens([m], tokens, "v"));
            Assert.Equal(Lfm3RuntimeModel.VisionInputUnsupportedMessage, ex.Message);
        }
        var aex = Assert.Throws<NotSupportedException>(() => model.MultimodalInjector.ProcessPromptTokens([audio], tokens, "a"));
        Assert.Equal(AudioInputSupport.Lfm3Message, aex.Message);
        Assert.False(model.MultimodalInjector.HasPendingEmbeddings("a"));
    }

    [Fact]
    public void RunOutsideGpuComputeLock_LetsAnotherThreadIn_AndTakesTheLockBack()
    {
        string path = new Lfm3SyntheticModelBuilder { WeightType = SyntheticGguf.GgmlType.F32 }.Write(Path.Combine(_directory, "lfm3.gguf"));
        using ModelBase model = ModelBase.Create(path, BackendType.Cpu);
        object gate = model.GpuComputeLock;

        // Without the lock (the CLI) the work simply runs.
        Assert.Equal(3, model.RunOutsideGpuComputeLock(() => 3));
        Assert.False(Monitor.IsEntered(gate));

        using var started = new ManualResetEventSlim();
        bool otherRan = false;
        lock (gate)
        {
            int result = model.RunOutsideGpuComputeLock(() =>
            {
                Assert.False(Monitor.IsEntered(gate));
                var other = new Thread(() => { lock (gate) otherRan = true; });
                other.Start();
                Assert.True(other.Join(TimeSpan.FromSeconds(30)), "another thread could not take the released lock");
                return 7;
            });
            Assert.Equal(7, result);
            Assert.True(Monitor.IsEntered(gate));
        }
        Assert.True(otherRan);
    }

    private sealed class FakeDecoder : IVideoDecoder
    {
        public List<int> Requested { get; } = new();
        public int Probes { get; private set; }
        public bool Unreadable { get; set; }

        public VideoInfo Probe(string path)
        {
            Probes++;
            return Unreadable ? throw new InvalidDataException("not a video") : new(2, 6, 2, 2);
        }

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
