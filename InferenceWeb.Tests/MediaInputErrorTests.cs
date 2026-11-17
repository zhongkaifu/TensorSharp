// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System.Text;
using System.Text.Json;
using Microsoft.AspNetCore.Http;
using TensorSharp.Models.Media;
using TensorSharp.Models.Media.Desktop;
using TensorSharp.Server.Logging;
using TensorSharp.Server.RequestParsers;

namespace InferenceWeb.Tests;

/// <summary>
/// Media a request carries that cannot be read is the client's error: ffmpeg opens the uploaded file alone, through the
/// audio and video demuxers (never a playlist or script that names further files), a decode that fails or runs out of time
/// is a <see cref="MediaInputException"/> naming the attachment by file name, the server answers it with a 400 and that
/// message, and an audio upload is stored under the extension its format calls for (a FLAC clip used to be read as MP3).
/// </summary>
public sealed class MediaInputErrorTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-media-input-" + Guid.NewGuid().ToString("N"));

    public MediaInputErrorTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    private string Tone(string name)
    {
        var samples = new float[8000];
        for (int i = 0; i < samples.Length; i++) samples[i] = 0.25f * MathF.Sin(i * 0.05f);
        string path = Path.Combine(_dir, name);
        File.WriteAllBytes(path, Lfm3AudioFrontEndTests.WavF32(16000, samples));
        return path;
    }

    /// <summary>An ffconcat script uploaded as a .flac, and an HLS playlist, name a file beside them in the upload directory;
    /// without the demuxer whitelist ffmpeg read that file (another upload) as the request's audio.</summary>
    [FfmpegFact]
    public void FfmpegReadsTheUploadOnlyNotFilesItNames()
    {
        string other = Tone("other.wav");
        Assert.Equal(8000, DesktopMediaProvider.Instance.Decode(other).SampleCount);
        string script = Path.Combine(_dir, "script.flac");
        File.WriteAllText(script, "ffconcat version 1.0\nfile other.wav\n");
        string playlist = Path.Combine(_dir, "list.m3u8");
        File.WriteAllText(playlist, "#EXTM3U\n#EXT-X-TARGETDURATION:1\n#EXTINF:0.5,\nother.wav\n#EXT-X-ENDLIST\n");
        foreach (string upload in new[] { script, playlist })
        {
            var ex = Assert.Throws<MediaInputException>(() => DesktopMediaProvider.Instance.Decode(upload));
            Assert.Contains(Path.GetFileName(upload), ex.Message);
            Assert.DoesNotContain(_dir, ex.Message);
        }
        Assert.Contains("not an audio or video file this host decodes",
            Assert.Throws<MediaInputException>(() => DesktopMediaProvider.Instance.Decode(script)).Message);
    }

    /// <summary>Bytes no demuxer reads: a short reason that names the file and not where the host keeps it.</summary>
    [FfmpegFact]
    public void UndecodableBytesNameTheFileOnly()
    {
        string junk = Path.Combine(_dir, "junk.flac");
        var bytes = new byte[20000];
        new Random(5).NextBytes(bytes);
        File.WriteAllBytes(junk, bytes);
        var ex = Assert.Throws<MediaInputException>(() => DesktopMediaProvider.Instance.Decode(junk));
        Assert.StartsWith("ffmpeg could not decode 'junk.flac'", ex.Message);
        Assert.DoesNotContain(_dir, ex.Message);
        Assert.DoesNotContain("Terminating thread", ex.Message);
        Assert.True(ex.Message.Length < 600, ex.Message);
    }

    /// <summary>A decode that outlives its deadline is killed and reported as the input's failure.</summary>
    [FfmpegFact]
    public void ADecodeThatTimesOutIsTheInputsFailure()
    {
        string tone = Tone("tone.wav");
        var ex = Assert.Throws<MediaInputException>(() => DesktopMediaProvider.Instance.Decode(tone, TimeSpan.FromMilliseconds(1)));
        Assert.IsType<TimeoutException>(ex.InnerException);
        Assert.Contains("'tone.wav' could not be decoded: ffmpeg did not finish within", ex.Message);
    }

    [Fact]
    public void FfmpegsErrorLogIsSummarized()
    {
        string log = "[af#0:0 @ 0x771cc2c000] Cannot determine format of input 0:0 after EOF\n"
                     + "[af#0:0 @ 0x771cc2c000] Task finished with error code: -1094995529 (Invalid data found when processing input)\n"
                     + "[aost#0:0/pcm_f32le @ 0x7c44c64000] [enc:pcm_f32le @ 0x7c450181c0] Could not open encoder before EOF\n"
                     + "[af#0:0 @ 0x7c44c68000] Terminating thread with return code -1094995529 (Invalid data found when processing input)\n"
                     + "    Last message repeated 1 times\n"
                     + "[out#0/wav @ 0x7c44c58180] Nothing was written into output file, because at least one of its streams received no packets.\n";
        Assert.Equal("Cannot determine format of input 0:0 after EOF", DesktopMediaProvider.SummarizeFfmpegError(log));
        Assert.Equal(400, DesktopMediaProvider.SummarizeFfmpegError(new string('x', 1000)).Length);
    }

    [Fact]
    public void AFailureNamesTheAttachmentNotItsDirectory()
    {
        string path = Path.Combine(_dir, "uploads", "abc.wav");
        MediaInputException ex = MediaInputException.From(new FileNotFoundException($"audio file not found: {path}"), new[] { path });
        Assert.Equal("A media attachment of the request could not be read: audio file not found: abc.wav", ex.Message);
        Assert.Same(ex, MediaInputException.From(ex, new[] { path }));
        Assert.True(MediaInputException.IsDecodeFailure(new InvalidDataException("x")));
        Assert.True(MediaInputException.IsDecodeFailure(new TimeoutException("x")));
        Assert.False(MediaInputException.IsDecodeFailure(new InvalidOperationException("x")));
    }

    [Theory]
    [InlineData("/v1/chat/completions")]
    [InlineData("/api/chat/ollama")]
    public async Task TheServerAnswersUndecodableMediaWith400AndTheReason(string route)
    {
        var ctx = new DefaultHttpContext();
        ctx.Request.Path = route;
        ctx.Response.Body = new MemoryStream();
        await ApiExceptionMiddleware.InvokeAsync(ctx,
            () => throw new MediaInputException("ffmpeg could not decode 'clip.flac' (exit 183): Invalid data"));
        Assert.Equal(400, ctx.Response.StatusCode);
        ctx.Response.Body.Seek(0, SeekOrigin.Begin);
        using var doc = JsonDocument.Parse(Encoding.UTF8.GetString(((MemoryStream)ctx.Response.Body).ToArray()));
        JsonElement error = doc.RootElement.GetProperty("error");
        string message = route.StartsWith("/v1") ? error.GetProperty("message").GetString() : error.GetString();
        Assert.Equal("ffmpeg could not decode 'clip.flac' (exit 183): Invalid data", message);
        if (route.StartsWith("/v1"))
            Assert.Equal("invalid_request_error", error.GetProperty("type").GetString());
    }

    [Theory]
    [InlineData("wav", "", ".wav")]
    [InlineData("mp3", "", ".mp3")]
    [InlineData("flac", "", ".flac")]
    [InlineData("x-flac", "", ".flac")]
    [InlineData("m4a", "", ".m4a")]
    [InlineData("aac", "", ".aac")]
    [InlineData("webm", "", ".webm")]
    [InlineData("opus", "", ".opus")]
    [InlineData("aiff", "", ".aiff")]
    [InlineData(null, "RIFF....WAVE", ".wav")]
    [InlineData(null, "fLaC", ".flac")]
    [InlineData(null, "....ftypM4A ", ".m4a")]
    [InlineData(null, "OggS........OpusHead", ".opus")]
    [InlineData("ogg", "OggS........OpusHead", ".opus")]
    [InlineData(null, "OggS.......vorbis", ".ogg")]
    [InlineData(null, "FORM....AIFF", ".aiff")]
    [InlineData(null, "caff", ".caf")]
    [InlineData(null, "ID3", ".mp3")]
    [InlineData("unknown", "????", ".mp3")]
    public void AnAudioUploadKeepsItsContainersExtension(string format, string head, string expected)
    {
        byte[] data = Encoding.ASCII.GetBytes(head.PadRight(32, '\0'));
        Assert.Equal(expected, ChatMessageParser.AudioExtension(format, data));
    }

    [Fact]
    public void SignaturesWithoutText()
    {
        Assert.Equal(".webm", ChatMessageParser.AudioExtension(null, new byte[] { 0x1A, 0x45, 0xDF, 0xA3, 0, 0, 0, 0 }));
        Assert.Equal(".aac", ChatMessageParser.AudioExtension(null, new byte[] { 0xFF, 0xF1, 0x50, 0x80 }));
        Assert.Equal(".mp3", ChatMessageParser.AudioExtension(null, new byte[] { 0xFF, 0xFB, 0x90, 0x64 }));
    }
}
