// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Runtime.CompilerServices;
using System.Text.Json;
using TensorAgent.Core.Hosting;
using TensorAgent.Core.Sessions;
using TensorSharp.Models.Video;
using TensorSharp.Models.WanVideo;

namespace TensorAgent.Tests;

/// <summary>
/// A chat turn on a video model: which message becomes which request for which
/// checkpoint, how the video service's frames reach the page, and what the saved chat
/// keeps. The generation itself needs the weights and runs in the live checks.
/// </summary>
public sealed class VideoTurnsTests
{
    /// <summary>MiniMax-H3's first-and-last-frame checkpoint, as the service reports it.</summary>
    private static readonly FakeVideoModel Keyframes = new()
    {
        SupportsImageConditioning = true,
        SupportsEndImageConditioning = true,
    };

    /// <summary>MiniMax-H3's reference checkpoint.</summary>
    private static readonly FakeVideoModel References = new()
    {
        SupportsImageConditioning = true,
        SupportsReferenceConditioning = true,
        MaxReferenceImages = 9,
    };

    private static JsonElement Body(object lastUser, params object[] earlier)
    {
        var messages = earlier.Concat(new[] { lastUser }).ToArray();
        return JsonSerializer.SerializeToElement(new { sessionId = "s", messages });
    }

    private static Dictionary<string, object> Payload(VideoTurns.VideoRequest request)
    {
        Assert.Null(request.Error);
        return Assert.IsType<Dictionary<string, object>>(request.Payload);
    }

    [Fact]
    public void ADescriptionAloneIsTextToVideoAtTheAppsClipLength()
    {
        Dictionary<string, object> payload = Payload(VideoTurns.Read(
            Body(new { role = "user", content = "  a red fox in falling snow  " }), Keyframes));

        Assert.Equal("a red fox in falling snow", payload["prompt"]);
        Assert.Equal(VideoTurns.DefaultFrames, payload["frames"]);
        Assert.False(payload.ContainsKey("steps"), "the model's own step count applies");
        Assert.False(payload.ContainsKey("imagePath"));
        // fps is never sent: H3 writes 24 fps whatever is asked.
        Assert.False(payload.ContainsKey("fps"));
    }

    [Fact]
    public void WanUsesItsOwnFrameGridAndAcceptsOneConditioningPhoto()
    {
        var wan = new FakeVideoModel
        {
            VideoModelFamily = "wan", SupportsAudio = false, SupportsImageConditioning = true,
        };
        var payload = Payload(VideoTurns.Read(Body(new
        {
            role = "user", content = "the fox walks", stillImagePaths = new[] { "fox.png" },
        }), wan));
        Assert.Equal(21, payload["frames"]);
        Assert.Equal("fox.png", payload["imagePath"]);
        Assert.False(payload.ContainsKey("steps"));
        Assert.NotNull(VideoTurns.Read(Body(new
        {
            role = "user", content = "the fox walks", stillImagePaths = new[] { "a.png", "b.png" },
        }), wan).Error);
    }

    [Fact]
    public void OnTheKeyframeCheckpointOnePhotoStartsTheClipAndASecondEndsIt()
    {
        Dictionary<string, object> one = Payload(VideoTurns.Read(Body(new
        {
            role = "user", content = "she turns and smiles", stillImagePaths = new[] { "a.png" },
        }), Keyframes));
        Assert.Equal("a.png", one["imagePath"]);
        Assert.False(one.ContainsKey("endImage"));

        Dictionary<string, object> two = Payload(VideoTurns.Read(Body(new
        {
            role = "user", content = "dawn becomes dusk", stillImagePaths = new[] { "a.png", "b.png" },
        }), Keyframes));
        Assert.Equal("a.png", two["imagePath"]);
        Assert.Equal("b.png", two["endImage"]);
    }

    [Fact]
    public void TheKeyframeCheckpointRefusesWhatItCannotUseInTheAppsOwnWords()
    {
        VideoTurns.VideoRequest three = VideoTurns.Read(Body(new
        {
            role = "user", content = "x", stillImagePaths = new[] { "a.png", "b.png", "c.png" },
        }), Keyframes);
        Assert.Null(three.Payload);
        Assert.Contains("two", three.Error);

        VideoTurns.VideoRequest clip = VideoTurns.Read(Body(new
        {
            role = "user", content = "x", videoFilePaths = new[] { "clip.mp4" },
        }), new FakeVideoModel { SupportsImageConditioning = true, SupportsEndImageConditioning = true, VideoModelFamily = "minimax-h3" });
        Assert.Null(clip.Payload);
        // It names the entry that can, which is what the user can act on.
        Assert.Contains("MiniMax-H3 References", clip.Error);

        VideoTurns.VideoRequest sound = VideoTurns.Read(Body(new
        {
            role = "user", content = "x", audioPaths = new[] { "song.wav" },
        }), Keyframes);
        Assert.Null(sound.Payload);
    }

    [Fact]
    public void OnTheReferenceCheckpointEveryAttachmentIsAReference()
    {
        Dictionary<string, object> payload = Payload(VideoTurns.Read(Body(new
        {
            role = "user",
            content = "they dance in the rain",
            stillImagePaths = new[] { "p1.png", "p2.png" },
            // A clip's sampled frames are in imagePaths too; they are not photos.
            imagePaths = new[] { "p1.png", "p2.png", "frame1.png", "frame2.png" },
            videoFilePaths = new[] { "clip.mp4" },
            audioPaths = new[] { "song.wav" },
        }), References));

        Assert.Equal(new[] { "p1.png", "p2.png" }, payload["referenceImages"]);
        Assert.Equal(new[] { "clip.mp4" }, payload["referenceVideos"]);
        Assert.Equal(new[] { "song.wav" }, payload["referenceAudios"]);
        Assert.False(payload.ContainsKey("imagePath"), "a reference is not a first frame");
    }

    [Fact]
    public void TheReferenceCheckpointTakesAtMostItsLimitAllTogether()
    {
        VideoTurns.VideoRequest request = VideoTurns.Read(Body(new
        {
            role = "user",
            content = "x",
            stillImagePaths = Enumerable.Range(0, 8).Select(i => $"p{i}.png").ToArray(),
            audioPaths = new[] { "a.wav", "b.wav" },
        }), References);

        Assert.Null(request.Payload);
        Assert.Contains("up to 9", request.Error);
        Assert.Contains("10 are attached", request.Error);
    }

    [Fact]
    public void ADocumentIsRefusedRatherThanFilmedAsTheScript()
    {
        // The page prepends an attached document's text to the message; filming that would
        // turn a report into the description of a clip.
        VideoTurns.VideoRequest request = VideoTurns.Read(Body(new
        {
            role = "user",
            content = "[File: notes.txt]\nquarterly figures\n[End of file]\n\na red fox in the snow",
            textFilePaths = new[] { "notes.txt" },
        }), Keyframes);

        Assert.Null(request.Payload);
        Assert.Contains("document", request.Error);
    }

    [Fact]
    public void TheNewestUserMessageIsTheRequestAndItNeedsWords()
    {
        // An earlier turn's photo does not carry over into a new clip.
        Dictionary<string, object> payload = Payload(VideoTurns.Read(Body(
            new { role = "user", content = "now at night" },
            new { role = "user", content = "a beach", stillImagePaths = new[] { "a.png" } },
            new { role = "assistant", content = "", videoUrl = "/uploads/v.mp4" }), Keyframes));
        Assert.Equal("now at night", payload["prompt"]);
        Assert.False(payload.ContainsKey("imagePath"));

        VideoTurns.VideoRequest wordless = VideoTurns.Read(Body(new
        {
            role = "user", content = "   ", stillImagePaths = new[] { "a.png" },
        }), Keyframes);
        Assert.Null(wordless.Payload);
        Assert.Equal("Describe the video you want.", wordless.Error);
    }

    [Fact]
    public async Task TheServicesFramesBecomeTheChatStreamsOnceEach()
    {
        List<JsonElement> frames = await Collect(VideoTurns.Translate(ServiceFrames(
            new { videoGen = true, step = 0, total = 20, phase = "text-encode", detail = (string?)null, elapsedSeconds = 0.1, etaSeconds = -1.0 },
            // A step arrives twice: the hook first, with no phase and no timing...
            new { videoGen = true, step = 1, total = 20, phase = (string?)null, detail = (string?)null, elapsedSeconds = 0.0, etaSeconds = -1.0 },
            // ...then the phase report that carries the timing.
            new { videoGen = true, step = 1, total = 20, phase = "denoise", detail = (string?)null, elapsedSeconds = 12.0, etaSeconds = 228.0 },
            new { videoGen = true, step = 2, total = 20, phase = (string?)null, detail = (string?)null, elapsedSeconds = 0.0, etaSeconds = -1.0 },
            new { videoGen = true, step = 2, total = 20, phase = "denoise", detail = (string?)null, elapsedSeconds = 19.0, etaSeconds = 171.0 },
            new { videoGen = true, step = 20, total = 20, phase = "vae-decode", detail = (string?)null, elapsedSeconds = 140.0, etaSeconds = -1.0 },
            // One report per VAE tile is still one phase.
            new { videoGen = true, step = 20, total = 20, phase = "vae-decode", detail = (string?)null, elapsedSeconds = 141.0, etaSeconds = -1.0 },
            new { videoGen = true, step = 20, total = 20, phase = "audio-decode", detail = (string?)null, elapsedSeconds = 146.0, etaSeconds = -1.0 },
            new { videoGen = true, step = 20, total = 20, phase = "done", detail = (string?)null, elapsedSeconds = 147.0, etaSeconds = -1.0 },
            new
            {
                done = true, url = "/uploads/video-1.mp4", audioUrl = "/uploads/video-1.wav", audioMuxed = true,
                width = 640, height = 384, frames = 22, fps = 24, seed = 7L, codec = "h264", elapsedSeconds = 148.5,
            }), "session-1"));

        string[] phases = frames.Where(f => f.TryGetProperty("video_phase", out _))
            .Select(f => $"{f.GetProperty("video_phase").GetString()}:{f.GetProperty("video_step").GetInt32()}")
            .ToArray();
        Assert.Equal(new[]
        {
            "text-encode:0", "denoise:1", "denoise:1", "denoise:2", "denoise:2", "vae-decode:20", "audio-decode:20", "encode:20",
        }, phases);
        JsonElement timed = frames.Last(f => f.TryGetProperty("video_phase", out JsonElement p) && p.GetString() == "denoise");
        Assert.Equal(171.0, timed.GetProperty("eta").GetDouble());
        Assert.Equal(20, timed.GetProperty("video_steps").GetInt32());

        JsonElement clip = Assert.Single(frames, f => f.TryGetProperty("videoUrl", out _));
        Assert.Equal("/uploads/video-1.mp4", clip.GetProperty("videoUrl").GetString());
        // The sound is inside the MP4, so there is no second file for the page to play.
        Assert.Equal(JsonValueKind.Null, clip.GetProperty("audioUrl").ValueKind);
        Assert.True(clip.GetProperty("hasAudio").GetBoolean());
        Assert.Equal(22, clip.GetProperty("frames").GetInt32());
        Assert.Equal(7, clip.GetProperty("seed").GetInt64());

        JsonElement done = frames[^1];
        Assert.True(done.GetProperty("done").GetBoolean());
        Assert.Equal("session-1", done.GetProperty("sessionId").GetString());
        Assert.Equal(148.5, done.GetProperty("elapsed").GetDouble());
    }

    [Fact]
    public async Task ASoundtrackThatIsNotInsideTheClipIsHandedToThePage()
    {
        List<JsonElement> frames = await Collect(VideoTurns.Translate(ServiceFrames(new
        {
            done = true, url = "/uploads/v.mp4", audioUrl = "/uploads/v.wav", audioMuxed = false,
            width = 640, height = 384, frames = 22, fps = 24, seed = 1L, codec = "h264", elapsedSeconds = 1.0,
        }), "s"));

        JsonElement clip = Assert.Single(frames, f => f.TryGetProperty("videoUrl", out _));
        Assert.Equal("/uploads/v.wav", clip.GetProperty("audioUrl").GetString());
        Assert.True(clip.GetProperty("hasAudio").GetBoolean());
    }

    [Fact]
    public async Task AFailureOrAStopEndsTheTurnWithTheSession()
    {
        List<JsonElement> failed = await Collect(VideoTurns.Translate(ServiceFrames(
            new { done = true, error = "MiniMax-H3 DiT forward failed." }), "s-failed"));
        JsonElement error = Assert.Single(failed);
        Assert.Equal("MiniMax-H3 DiT forward failed.", error.GetProperty("error").GetString());
        Assert.Equal("s-failed", error.GetProperty("sessionId").GetString());

        // The service ends without a terminal frame only when the turn was stopped.
        List<JsonElement> stopped = await Collect(VideoTurns.Translate(ServiceFrames(
            new { videoGen = true, step = 3, total = 20, phase = "denoise", detail = (string?)null, elapsedSeconds = 30.0, etaSeconds = 170.0 }),
            "s-stopped"));
        Assert.True(stopped[^1].GetProperty("aborted").GetBoolean());
        Assert.Equal("s-stopped", stopped[^1].GetProperty("sessionId").GetString());
    }

    /// <summary>
    /// The clip is what a video turn makes, often all it makes, and the chat has to keep it:
    /// written down when the turn ends, through the same manager and recorder a text turn
    /// uses, so a clip made while the page was hidden is in the chat when it is reopened.
    /// </summary>
    [Fact]
    public async Task TheClipIsWhatTheConversationKeeps()
    {
        string root = Path.Combine(Path.GetTempPath(), "video-turns-" + Guid.NewGuid().ToString("N")[..8]);
        Directory.CreateDirectory(root);
        try
        {
            var store = new ConversationStore(Path.Combine(root, "conversations"));
            Conversation conversation = store.Create();
            conversation.Messages.Add(new StoredMessage { Role = "user", Content = "a red fox in falling snow" });
            store.Save(conversation);
            var recorder = new ConversationRecorder(store);
            recorder.Bind("session-clip", conversation.Id);
            using var turns = new ChatTurnManager(recorder);

            string turn = turns.Start(conversation.Id, _ => VideoTurns.Translate(ServiceFrames(
                new { videoGen = true, step = 1, total = 20, phase = "denoise", detail = (string?)null, elapsedSeconds = 7.0, etaSeconds = 133.0 },
                new
                {
                    done = true, url = "/uploads/fox.mp4", audioUrl = "/uploads/fox.wav", audioMuxed = false,
                    width = 640, height = 384, frames = 22, fps = 24, seed = 3L, codec = "h264", elapsedSeconds = 150.0,
                }), "session-clip"));
            for (int i = 0; i < 200 && turns.StatusOfId(turn)!.IsRunning; i++)
                await Task.Delay(25);

            Conversation reloaded = Assert.IsType<Conversation>(
                new ConversationStore(Path.Combine(root, "conversations")).Load(conversation.Id));
            StoredMessage assistant = Assert.Single(reloaded.Messages, m => m.Role == "assistant");
            Assert.Equal("/uploads/fox.mp4", assistant.VideoUrl);
            Assert.Equal("/uploads/fox.wav", assistant.AudioUrl);
            Assert.Null(assistant.ImageUrl);
            IEnumerable<string> kept = reloaded.Messages.SelectMany(m => m.ReferencedUploads);
            Assert.Contains("fox.mp4", kept);
            Assert.Contains("fox.wav", kept);
        }
        finally
        {
            try { Directory.Delete(root, true); } catch (IOException) { }
        }
    }

    /// <summary>A muxed clip keeps no separate soundtrack: there is nothing for the page to play beside it.</summary>
    [Fact]
    public async Task AClipWithItsSoundInsideKeepsOnlyTheClip()
    {
        string root = Path.Combine(Path.GetTempPath(), "video-turns-" + Guid.NewGuid().ToString("N")[..8]);
        Directory.CreateDirectory(root);
        try
        {
            var store = new ConversationStore(Path.Combine(root, "conversations"));
            Conversation conversation = store.Create();
            conversation.Messages.Add(new StoredMessage { Role = "user", Content = "waves" });
            store.Save(conversation);
            var recorder = new ConversationRecorder(store);
            recorder.Bind("session-muxed", conversation.Id);
            using var turns = new ChatTurnManager(recorder);

            string turn = turns.Start(conversation.Id, _ => VideoTurns.Translate(ServiceFrames(new
            {
                done = true, url = "/uploads/waves.mp4", audioUrl = "/uploads/waves.wav", audioMuxed = true,
                width = 640, height = 384, frames = 22, fps = 24, seed = 3L, codec = "h264", elapsedSeconds = 150.0,
            }), "session-muxed"));
            for (int i = 0; i < 200 && turns.StatusOfId(turn)!.IsRunning; i++)
                await Task.Delay(25);

            StoredMessage assistant = Assert.Single(
                new ConversationStore(Path.Combine(root, "conversations")).Load(conversation.Id)!.Messages,
                m => m.Role == "assistant");
            Assert.Equal("/uploads/waves.mp4", assistant.VideoUrl);
            Assert.Null(assistant.AudioUrl);
        }
        finally
        {
            try { Directory.Delete(root, true); } catch (IOException) { }
        }
    }

    private static async IAsyncEnumerable<object> ServiceFrames(params object[] frames)
    {
        foreach (object frame in frames)
        {
            await Task.Yield();
            yield return frame;
        }
    }

    private static async Task<List<JsonElement>> Collect(IAsyncEnumerable<object> frames)
    {
        var list = new List<JsonElement>();
        await foreach (object frame in frames)
            list.Add(JsonSerializer.SerializeToElement(frame));
        return list;
    }

    /// <summary>What a video model reports about itself; nothing here generates.</summary>
    private sealed class FakeVideoModel : IVideoGenerationModel
    {
        public string VideoModelFamily { get; init; } = "minimax-h3";
        public bool SupportsAudio { get; init; } = true;
        public bool SupportsImageConditioning { get; init; }
        public bool SupportsEndImageConditioning { get; init; }
        public bool SupportsReferenceConditioning { get; init; }
        public int MaxReferenceImages { get; init; }

        public GeneratedVideo GenerateVideo(string prompt, VideoGenerationParams? p = null) =>
            throw new NotSupportedException("the request shaping under test never generates");
    }
}
