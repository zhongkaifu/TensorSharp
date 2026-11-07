// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Runtime.InteropServices;
using System.Text.Json;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp;
using TensorSharp.Models.Media;
using TensorSharp.Models.Media.Desktop;
using TensorSharp.Models.QwenImage;
using TensorSharp.Models.Video;
using TensorSharp.Models.WanVideo;
using TensorSharp.Server.Hosting;
using CliProgram = TensorSharp.Cli.Program;

namespace InferenceWeb.Tests;

/// <summary>
/// A clip that could not be saved is refused before it is generated. OpenCvSharp ships no
/// native library for Windows on Arm, so there, without FFmpeg, no MP4 encoder exists, and the
/// CLI and the server used to find that out only after generating the whole clip (and the CLI
/// lost the soundtrack with it). The encoders and the video model are stand-ins; no model runs
/// and nothing is encoded.
/// </summary>
[Collection(MediaCodecSwaps.Name)]
public sealed class VideoExportCheckTests : IDisposable
{
    private const string NoEncoder = "Video export needs FFmpeg (stand-in encoder).";

    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-video-export-" + Guid.NewGuid().ToString("N"));
    private readonly IVideoEncoder _registered = MediaCodecs.VideoEncoder;
    private WebUiVideoLoraTests.PlainVideoStandIn _model;
    private ModelService _models;
    private WebUiChatService _chat;

    public VideoExportCheckTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        MediaCodecs.VideoEncoder = _registered;
        _models?.Dispose();
        try { Directory.Delete(_dir, recursive: true); } catch { /* best effort */ }
    }

    // ---- the desktop provider's answer ---------------------------------------------------------

    [Fact]
    public void AnFfmpegThatExistsIsEnoughWithoutLoadingOpenCv()
    {
        string ffmpeg = Path.Combine(_dir, OperatingSystem.IsWindows() ? "ffmpeg.exe" : "ffmpeg");
        File.WriteAllText(ffmpeg, "stand-in, never run");
        bool probed = false;
        Assert.Null(DesktopMediaProvider.Mp4ExportUnavailable(ffmpeg, () => probed = true));
        Assert.False(probed);
    }

    [Fact]
    public void OpenCvAloneIsEnough() =>
        Assert.Null(DesktopMediaProvider.Mp4ExportUnavailable(null, () => true));

    [Fact]
    public void WithNeitherEncoderTheReasonNamesBothAndThePlatform()
    {
        string reason = DesktopMediaProvider.Mp4ExportUnavailable(null, () => false);
        Assert.StartsWith("Video export needs FFmpeg, but none was found on PATH, at TS_FFMPEG or beside the application", reason);
        Assert.Contains("OpenCvSharpExtern", reason);
        Assert.Contains($"on {RuntimeInformation.RuntimeIdentifier}.", reason);
    }

    [Fact]
    public void AConfiguredFfmpegThatDoesNotExistIsNamed()
    {
        string missing = Path.Combine(_dir, "no-such-ffmpeg.exe");
        string reason = DesktopMediaProvider.Mp4ExportUnavailable(missing, () => false);
        Assert.Contains($"TS_FFMPEG names '{missing}', which does not exist", reason);
    }

    [Fact]
    public void OnlyTheManagedProviderHasNoEncoderToTry()
    {
        Assert.Null(((IVideoEncoder)new SaveOnlyEncoder()).Mp4ExportUnavailableReason);
        Assert.NotNull(((IVideoEncoder)ManagedMediaProvider.Instance).Mp4ExportUnavailableReason);
    }

    // ---- CLI -----------------------------------------------------------------------------------

    [Fact]
    public void TheCliRefusesAnMp4OutputButNotPngFrames()
    {
        MediaCodecs.VideoEncoder = new StandInVideoEncoder(NoEncoder);
        Assert.Equal(NoEncoder + " Or pass --output <name>.png to save the frames as PNG files.",
            CliProgram.VideoOutputError("clip.mp4"));
        Assert.Null(CliProgram.VideoOutputError("frames.PNG"));

        MediaCodecs.VideoEncoder = new StandInVideoEncoder(null);
        Assert.Null(CliProgram.VideoOutputError("clip.mp4"));
    }

    [Fact]
    public void TheCliKeepsTheSoundtrackWhenTheMp4ExportFails()
    {
        MediaCodecs.VideoEncoder = new StandInVideoEncoder(null);
        string output = Path.Combine(_dir, "out", "clip.mp4");
        var video = new GeneratedVideo
        {
            Frames = [new RgbImage(2, 2, new float[12])], Fps = 24, Seed = 1,
            Audio = new GeneratedVideoAudio { Channels = [new float[800], new float[800]], SampleRate = 32000 },
        };

        var ex = Assert.Throws<InvalidOperationException>(() => CliProgram.SaveGeneratedVideo(video, output, TimeSpan.Zero));
        Assert.Equal(StandInVideoEncoder.Failure, ex.Message);
        Assert.True(File.Exists(Path.Combine(_dir, "out", "clip.wav")));
    }

    // ---- server ----------------------------------------------------------------------------------

    [Fact]
    public async Task ThePlainRouteRefusesWith503BeforeTheModelRuns()
    {
        LoadVideoModel();
        MediaCodecs.VideoEncoder = new StandInVideoEncoder(NoEncoder);
        var ex = await Assert.ThrowsAsync<WebUiRequestRejectedException>(
            () => _chat.VideoGenerateAsync(Body(), CancellationToken.None).WaitAsync(WebUiVideoLoraTests.TestTimeout));
        Assert.Equal(503, ex.StatusCode);
        var payload = JsonSerializer.SerializeToElement(ex.Payload);
        Assert.Equal("video_export_unavailable", payload.GetProperty("code").GetString());
        Assert.Equal(NoEncoder, payload.GetProperty("error").GetString());
        Assert.Equal(0, _model.GenerateCalls);
    }

    [Fact]
    public async Task TheStreamEndsWithTheReasonBeforeTheModelRuns()
    {
        LoadVideoModel();
        MediaCodecs.VideoEncoder = new StandInVideoEncoder(NoEncoder);
        var last = Assert.Single(await Stream(Body()));
        Assert.True(last.GetProperty("done").GetBoolean());
        Assert.Equal(NoEncoder, last.GetProperty("error").GetString());
        Assert.Equal(0, _model.GenerateCalls);
    }

    [Fact]
    public async Task GenerateVideoAsyncRefusesWith503BeforeTheModelRuns()
    {
        // The path Video Studio and /v1/videos/generations take.
        LoadVideoModel();
        MediaCodecs.VideoEncoder = new StandInVideoEncoder(NoEncoder);
        var ex = await Assert.ThrowsAsync<WebUiRequestRejectedException>(() => _chat.GenerateVideoAsync(
            WebUiVideoLoraTests.Prompt, new VideoGenerationParams()).WaitAsync(WebUiVideoLoraTests.TestTimeout));
        Assert.Equal(503, ex.StatusCode);
        Assert.Equal(0, _model.GenerateCalls);
    }

    [Fact]
    public async Task WithAnEncoderTheRequestReachesTheModel()
    {
        LoadVideoModel();
        MediaCodecs.VideoEncoder = new StandInVideoEncoder(null);
        var last = Assert.Single(await Stream(Body()));
        Assert.Equal(WebUiVideoLoraTests.TurboStandIn.Reached, last.GetProperty("error").GetString());
        Assert.Equal(1, _model.GenerateCalls);
    }

    // ---- helpers ---------------------------------------------------------------------------------

    private void LoadVideoModel()
    {
        _models = new ModelService(NullLogger<ModelService>.Instance,
            (path, _, _, _) => _model = new WebUiVideoLoraTests.PlainVideoStandIn(path));
        _chat = new WebUiChatService(_models, new SessionManager(), WebUiVideoLoraTests.Options(_dir),
            new UploadStoragePolicy(Path.Combine(_dir, "uploads")),
            new SkillRegistry(new SkillRegistryOptions { Roots = Array.Empty<string>() }),
            codeRunner: null, workspaces: null, codeArtifacts: null, NullLoggerFactory.Instance);
        _models.LoadModel(WebUiVideoLoraTests.WriteStandInGguf(_dir, "wan.gguf"), null, "cpu");
        Assert.Same(_model, _models.Model);
    }

    private static JsonElement Body() => JsonSerializer.SerializeToElement(new { prompt = WebUiVideoLoraTests.Prompt });

    private async Task<List<JsonElement>> Stream(JsonElement body)
    {
        var frames = _chat.VideoGenerateStreamAsync(body, CancellationToken.None).GetAsyncEnumerator();
        var result = new List<JsonElement>();
        try
        {
            while (await frames.MoveNextAsync().AsTask().WaitAsync(WebUiVideoLoraTests.TestTimeout))
                result.Add(JsonSerializer.SerializeToElement(frames.Current));
        }
        finally
        {
            await frames.DisposeAsync();
        }
        return result;
    }

    /// <summary>An encoder written before the export check existed: it implements SaveMp4 only.</summary>
    private sealed class SaveOnlyEncoder : IVideoEncoder
    {
        public string SaveMp4(string path, RgbImage[] frames, int fps) => "h264";
    }
}

/// <summary>An MP4 encoder that says whether it has an encoder to try, and writes nothing: its
/// export always fails.</summary>
internal sealed class StandInVideoEncoder(string unavailableReason) : IVideoEncoder
{
    internal const string Failure = "Could not encode MP4 (stand-in encoder).";

    public string Mp4ExportUnavailableReason => unavailableReason;

    public string SaveMp4(string path, RgbImage[] frames, int fps) => throw new InvalidOperationException(Failure);
}

/// <summary>For the tests of the video routes, whose stand-in models stop before a clip exists:
/// replaces the machine's encoder with one that reports an encoder to try, so whether FFmpeg or
/// OpenCV is installed here does not decide what they test. Test parallelism is off for the
/// whole assembly (TestAssemblyConfig), so no other test sees the swap.</summary>
internal sealed class VideoExportScope : IDisposable
{
    private readonly IVideoEncoder _registered = MediaCodecs.VideoEncoder;

    internal VideoExportScope() => MediaCodecs.VideoEncoder = new StandInVideoEncoder(null);

    public void Dispose() => MediaCodecs.VideoEncoder = _registered;
}
