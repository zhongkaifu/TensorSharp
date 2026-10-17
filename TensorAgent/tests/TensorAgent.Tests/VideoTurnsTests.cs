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
using System.Net;
using System.Net.Http.Json;
using System.Reflection;
using System.Text.Json;
using Microsoft.Extensions.Logging.Abstractions;
using TensorAgent.Core.Hosting;
using TensorAgent.Core.Sessions;
using TensorSharp.AgentHost.Skills;
using TensorSharp.Chat;
using TensorSharp.Models.Video;
using TensorSharp.Models.WanVideo;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Server;
using TensorSharp.Server.Hosting;

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
            DefaultPreviewFrames = VideoTurns.DefaultWanFrames,
        };
        var payload = Payload(VideoTurns.Read(Body(new
        {
            role = "user", content = "the fox walks", stillImagePaths = new[] { "fox.png" },
        }), wan));
        Assert.Equal(21, payload["frames"]);
        Assert.Equal("fox.png", payload["imagePath"]);
        Assert.Equal("i2v", payload["videoMode"]);
        Assert.False(payload.ContainsKey("steps"));
        Assert.NotNull(VideoTurns.Read(Body(new
        {
            role = "user", content = "the fox walks", stillImagePaths = new[] { "a.png", "b.png" },
        }), wan).Error);
    }

    [Fact]
    public void NewVideoFamiliesUseTheirOwnPreviewGrid()
    {
        // Request shaping only: this fake does not register or advertise an LTX pipeline.
        var ltx = new FakeVideoModel { VideoModelFamily = "ltx-2.5", DefaultPreviewFrames = 25 };
        var payload = Payload(VideoTurns.Read(Body(new { role = "user", content = "a fox walks" }), ltx));
        Assert.Equal(25, payload["frames"]);
        Assert.Equal(8, payload["steps"]);
        Assert.Equal(1f, payload["cfg"]);
        Assert.Equal("euler_ancestral", payload["sampler"]);
        Assert.Equal(0f, payload["flowShift"]);
        // The two-stage model owns its output size (1024x1024): a 512x512 request
        // would plan motion at 256x256.
        Assert.False(payload.ContainsKey("width"));
        Assert.False(payload.ContainsKey("height"));
        Assert.Equal(24, payload["fps"]);
        Assert.Equal("t2va", payload["videoMode"]);
        Assert.Equal(true, payload["generateAudio"]);
    }

    [Fact]
    public void AnUnspecifiedPreviewLengthLeavesTheModelDefaultIntact()
    {
        var model = new FakeVideoModel { VideoModelFamily = "future-video", DefaultPreviewFrames = 0 };
        var payload = Payload(VideoTurns.Read(Body(new { role = "user", content = "a fox walks" }), model));
        Assert.False(payload.ContainsKey("frames"));
    }

    [Fact]
    public void LtxMultiplePhotosGuideOrderedMomentsInsteadOfIdentityReferences()
    {
        var model = new FakeVideoModel { VideoModelFamily = "ltx-2.5", DefaultPreviewFrames = 17,
            SupportsImageConditioning = true, SupportsEndImageConditioning = true,
            SupportsKeyframeConditioning = true, MaxReferenceImages = 16 };
        var payload = Payload(VideoTurns.Read(Body(new { role = "user", content = "请让图中的女人来个飞吻",
            stillImagePaths = new[] { "a.png", "b.png", "c.png" } }), model));
        Assert.Equal("keyframes", payload["videoMode"]);
        JsonElement conditions = JsonSerializer.SerializeToElement(payload["imageConditions"]);
        Assert.Equal(new[] { 0, 8, 16 }, conditions.EnumerateArray().Select(c => c.GetProperty("frameIndex").GetInt32()));
        Assert.False(payload.ContainsKey("referenceImages"));
        Assert.NotNull(VideoTurns.Read(Body(new { role = "user", content = "x", audioPaths = new[] { "a.wav" } }), model).Error);
        Assert.NotNull(VideoTurns.Read(Body(new { role = "user", content = "x", videoFilePaths = new[] { "a.mp4" } }), model).Error);
        Assert.Contains("Ingredients", VideoTurns.Read(Body(new { role = "user", content = "x", videoMode = "ref",
            stillImagePaths = new[] { "a.png" } }), model).Error);
        Assert.NotNull(VideoTurns.Read(Body(new { role = "user", content = "x",
            imageConditions = new[] { new { imagePath = "a.png", frameIndex = 5 } } }), Keyframes).Error);
    }

    [Fact]
    public void SavedKeyframeRequestsKeepTheirTimelineAndProtectTheirUploads()
    {
        const string json = """
            {"role":"user","content":"a woman blows a kiss",
             "imageConditions":[{"imagePath":"portrait.png","frameIndex":-1,"strength":0.65}],
             "attachments":[{"file":"portrait.png","mediaType":"image","videoFrameIndex":-1,"videoStrength":0.65}]}
            """;
        StoredMessage message = JsonSerializer.Deserialize<StoredMessage>(json)!;
        StoredMessage restored = JsonSerializer.Deserialize<StoredMessage>(JsonSerializer.Serialize(message))!;
        Assert.Contains("portrait.png", restored.ReferencedUploads);
        Assert.Equal(-1, Assert.Single(restored.ImageConditions!).FrameIndex);
        Assert.Equal(.65f, Assert.Single(restored.ImageConditions!).Strength);
        Assert.Equal(-1, Assert.Single(restored.Attachments!).VideoFrameIndex);
        Assert.Equal(.65f, Assert.Single(restored.Attachments!).VideoStrength);
        string empty = JsonSerializer.Serialize(new StoredMessage { Role = "user", Content = "x" });
        Assert.DoesNotContain("imageConditions", empty);
    }

    [Theory]
    [InlineData(97, 48)]
    [InlineData(121, 60)]
    public void SelectedClipLengthPlacesGuidesAcrossTheWholeTimelineAndSurvivesReplay(int frames, int middle)
    {
        var model = new FakeVideoModel { VideoModelFamily = "ltx-2.5", DefaultPreviewFrames = 17,
            SupportsImageConditioning = true, SupportsEndImageConditioning = true,
            SupportsKeyframeConditioning = true, MaxReferenceImages = 16 };
        var message = new StoredMessage { Role = "user", Content = "The woman blows a kiss.",
            VideoFrames = frames, StillImagePaths = ["a.png", "b.png", "c.png"] };
        var restored = JsonSerializer.Deserialize<StoredMessage>(JsonSerializer.Serialize(message))!;
        Assert.Equal(frames, restored.VideoFrames);
        var payload = Payload(VideoTurns.Read(Body(restored), model));
        Assert.Equal(frames, payload["frames"]);
        var conditions = JsonSerializer.SerializeToElement(payload["imageConditions"]);
        Assert.Equal(new[] { 0, middle, frames - 1 }, conditions.EnumerateArray().Select(c => c.GetProperty("frameIndex").GetInt32()));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    [InlineData(96)]
    public void InvalidSelectedClipLengthIsRefusedBeforeGeneration(int frames)
    {
        var model = new FakeVideoModel { VideoModelFamily = "ltx-2.5", DefaultPreviewFrames = 17 };
        var request = VideoTurns.Read(Body(new { role = "user", content = "x", videoFrames = frames }), model);
        Assert.Null(request.Payload);
        Assert.Contains("8n+1", request.Error);
    }

    [Fact]
    public void InstalledIngredientsUsesIdentitySheetRecipeAndPreservesReplayMode()
    {
        var model = new FakeVideoModel { VideoModelFamily = "ltx-2.5", DefaultPreviewFrames = 17,
            SupportsImageConditioning = true, SupportsEndImageConditioning = true,
            SupportsKeyframeConditioning = true, SupportsReferenceConditioning = true, MaxReferenceImages = 16 };
        var payload = Payload(VideoTurns.Read(Body(new { role = "user", content = "The woman waves.",
            videoMode = "ref", stillImagePaths = new[] { "a.png", "b.png" } }), model));
        Assert.Equal("ref", payload["videoMode"]);
        Assert.Equal(121, payload["frames"]);
        Assert.Equal(768, payload["width"]);
        Assert.Equal(448, payload["height"]);
        Assert.Equal(new[] { "a.png", "b.png" }, Assert.IsType<string[]>(payload["referenceImages"]));
        Assert.False(payload.ContainsKey("imageConditions"));
        var stored = new StoredMessage { Role = "user", Content = "The woman waves.", VideoMode = "ref", StillImagePaths = ["a.png"] };
        var restored = JsonSerializer.Deserialize<StoredMessage>(JsonSerializer.Serialize(stored))!;
        Assert.Equal("ref", restored.VideoMode);
        Assert.Contains("a.png", restored.ReferencedUploads);
    }

    [Fact]
    public void OnTheKeyframeCheckpointOnePhotoStartsTheClipAndASecondEndsIt()
    {
        Dictionary<string, object> one = Payload(VideoTurns.Read(Body(new
        {
            role = "user", content = "she turns and smiles", stillImagePaths = new[] { "a.png" },
        }), Keyframes));
        Assert.Equal("a.png", one["imagePath"]);
        Assert.Equal("i2v", one["videoMode"]);
        Assert.False(one.ContainsKey("endImage"));

        Dictionary<string, object> two = Payload(VideoTurns.Read(Body(new
        {
            role = "user", content = "dawn becomes dusk", stillImagePaths = new[] { "a.png", "b.png" },
        }), Keyframes));
        Assert.Equal("a.png", two["imagePath"]);
        Assert.Equal("b.png", two["endImage"]);
        Assert.Equal("fl2v", two["videoMode"]);
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
        Assert.Equal("ref", payload["videoMode"]);
        Assert.False(payload.ContainsKey("imagePath"), "a reference is not a first frame");
    }

    /// <summary>
    /// An actual upload reaches the video model through the default chat route, without
    /// a CLI image path. The startup mode deliberately conflicts with references and
    /// pairs of keyframes: the newest message's attachments choose this turn's mode.
    /// The stand-in captures the parameters before inference, so no weights or GPU are
    /// required and this test does not claim to validate numerical video generation.
    /// </summary>
    [Theory]
    [InlineData(false, 1, "i2v", false, false)]
    [InlineData(false, 2, "fl2v", false, false)]
    [InlineData(true, 1, "ref", false, false)]
    [InlineData(false, 1, "i2v", true, false)]
    [InlineData(false, 2, "fl2v", true, false)]
    [InlineData(false, 3, "keyframes", true, false)]
    [InlineData(false, 1, "keyframes", true, true)]
    public async Task UploadedPhotosReachTheVideoModelThroughChat(bool references, int photoCount, string mode, bool ltx, bool explicitLast)
    {
        string root = Path.Combine(Path.GetTempPath(), "video-upload-chat-" + Guid.NewGuid().ToString("N"));
        string uploads = Path.Combine(root, "uploads");
        Directory.CreateDirectory(uploads);
        try
        {
            string gguf = Path.Combine(root, "stand-in.gguf");
            using (var writer = new BinaryWriter(File.Create(gguf)))
            {
                // An empty, aligned GGUF is sufficient for the managed test model.
                writer.Write(0x46554747u);
                writer.Write(3u);
                writer.Write(0UL);
                writer.Write(0UL);
                writer.Write(new byte[8]);
            }
            using var model = new CapturingVideoModel(gguf, references, ltx);
            using var models = new ModelService();
            FieldInfo modelField = models.LifecycleService.GetType().GetField("_model",
                BindingFlags.Instance | BindingFlags.NonPublic)!;
            modelField.SetValue(models.LifecycleService, model);
            try
            {
                var options = new ServerHostingOptions(
                    startupModelPath: gguf, startupMmProjPath: null, defaultBackend: "cpu",
                    supportedBackends: new[] { new BackendOption("cpu", "CPU") },
                    defaultMaxTokens: 256, maxTokensPinned: false,
                    defaultVideoFrames: 144, defaultVideoFps: 0, defaultVideoWidth: 0,
                    defaultVideoHeight: 0, defaultVideoSteps: 0, defaultVideoMode: "i2v",
                    uploadDirectory: uploads, logDirectory: Path.Combine(root, "logs"),
                    fileLoggingEnabled: false, samplingDefaults: null);
                var chat = new WebUiChatService(models, new SessionManager(), options,
                    new UploadStoragePolicy(uploads), new SkillRegistry(new SkillRegistryOptions()),
                    codeRunner: null, workspaces: null, codeArtifacts: null, NullLoggerFactory.Instance);
                using var server = new LoopbackServer(NullLogger.Instance);
                server.MapWebUi(chat, uploads);
                server.Start();
                using var client = new HttpClient { BaseAddress = new Uri(server.BaseUrl) };
                client.DefaultRequestHeaders.Add("Cookie", $"{LoopbackServer.TokenCookie}={server.Token}");

                byte[] image = MediaFixtures.RedCircleOnWhitePng(32);
                var files = new List<string>();
                for (int i = 0; i < photoCount; i++)
                {
                    using var form = new MultipartFormDataContent();
                    form.Add(new ByteArrayContent(image), "file", $"photo-{i}.png");
                    using HttpResponseMessage upload = await client.PostAsync("/api/upload", form);
                    Assert.Equal(HttpStatusCode.OK, upload.StatusCode);
                    JsonElement attachment = await upload.Content.ReadFromJsonAsync<JsonElement>();
                    Assert.Equal("image", attachment.GetProperty("mediaType").GetString());
                    files.Add(attachment.GetProperty("file").GetString()!);
                }

                var user = new Dictionary<string, object>
                {
                    ["role"] = "user", ["content"] = "the camera follows the red circle",
                    ["stillImagePaths"] = files, ["imagePaths"] = files,
                };
                if (explicitLast)
                    user["imageConditions"] = new[] { new { imagePath = files[0], frameIndex = -1, strength = .65f } };
                using HttpResponseMessage response = await client.PostAsJsonAsync("/api/chat", new { messages = new[] { user } });
                Assert.Equal(HttpStatusCode.OK, response.StatusCode);
                string stream = await response.Content.ReadAsStringAsync();
                // This marker confirms dispatch passed both turn shaping and the shared
                // parser and reached the model, where the test deliberately stops.
                Assert.Contains(CapturingVideoModel.Captured, stream);
                Assert.DoesNotContain("needs a conditioning image", stream);
                Assert.Equal("the camera follows the red circle", model.Prompt);
                VideoGenerationParams parameters = Assert.IsType<VideoGenerationParams>(model.Parameters);
                Assert.Equal(mode, parameters.Mode);
                Assert.Equal(ltx ? 17 : VideoTurns.DefaultFrames, parameters.Frames);
                if (ltx)
                {
                    Assert.Equal(8, parameters.Steps);
                    Assert.Equal(1f, parameters.CfgScale);
                    Assert.True(parameters.GenerateAudio);
                    Assert.Equal(0, parameters.Width);
                    Assert.Equal(0, parameters.Height);
                }
                if (references)
                {
                    Assert.Null(parameters.ImageBytes);
                    Assert.Equal(files.Select(file => Path.Combine(uploads, file)), parameters.ReferenceImagePaths);
                    Assert.Equal(image, File.ReadAllBytes(Assert.Single(parameters.ReferenceImagePaths)));
                }
                else if (mode == "keyframes")
                {
                    Assert.Null(parameters.ImageBytes);
                    Assert.Null(parameters.ReferenceImagePaths);
                    Assert.Equal(explicitLast ? new[] { -1 } : new[] { 0, 8, 16 }, parameters.ImageConditions.Select(c => c.FrameIndex));
                    Assert.Equal(files.Select(file => Path.Combine(uploads, file)), parameters.ImageConditions.Select(c => c.ImagePath));
                    Assert.All(parameters.ImageConditions, c => Assert.Equal(image, File.ReadAllBytes(c.ImagePath)));
                    if (explicitLast) Assert.Equal(.65f, Assert.Single(parameters.ImageConditions).Strength);
                }
                else
                {
                    Assert.Equal(image, parameters.ImageBytes);
                    Assert.Null(parameters.ReferenceImagePaths);
                    Assert.Equal(photoCount > 1 ? Path.Combine(uploads, files[1]) : null, parameters.EndImagePath);
                }
                if (!ltx)
                {
                    // All three service entry points reject keyframes on a checkpoint
                    // that cannot consume them, before a request is silently generated.
                    using JsonDocument invalid = JsonDocument.Parse(JsonSerializer.Serialize(new
                    {
                        prompt = "a red circle moves",
                        imageConditions = new[] { new { imagePath = files[0], frameIndex = 5 } },
                    }));
                    List<JsonElement> rejected = await Collect(chat.VideoGenerateStreamAsync(invalid.RootElement, CancellationToken.None));
                    Assert.Contains("does not accept image keyframes", Assert.Single(rejected).GetProperty("error").GetString());
                    await Assert.ThrowsAsync<WebUiRequestRejectedException>(() => chat.VideoGenerateAsync(invalid.RootElement, CancellationToken.None));
                    await Assert.ThrowsAsync<WebUiRequestRejectedException>(() => chat.GenerateVideoAsync("a red circle moves", new VideoGenerationParams
                    {
                        ImageConditions = new[] { new VideoImageCondition { ImagePath = Path.Combine(uploads, files[0]), FrameIndex = 5 } },
                    }));
                    Assert.Equal(1, model.GenerateCalls);
                }
            }
            finally
            {
                modelField.SetValue(models.LifecycleService, null);
            }
        }
        finally
        {
            try { Directory.Delete(root, true); } catch (IOException) { }
        }
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

    /// <summary>
    /// A MiniMax-H3 request never carries steps, a flow shift, a sampler or a CFG: the base
    /// model's own defaults apply, and a Turbo plug-in's contract fixes them itself (and would
    /// refuse a value it does not fix). LTX's published recipe is the one family that sends them.
    /// </summary>
    [Theory]
    [InlineData(false, 0)]
    [InlineData(false, 1)]
    [InlineData(false, 2)]
    [InlineData(true, 0)]
    [InlineData(true, 2)]
    public void AMiniMaxH3RequestLeavesTheSamplingToTheModel(bool references, int photos)
    {
        string[] stills = Enumerable.Range(0, photos).Select(i => $"photo-{i}.png").ToArray();
        Dictionary<string, object> payload = Payload(VideoTurns.Read(
            Body(new { role = "user", content = "a fox in falling snow", stillImagePaths = stills, imagePaths = stills }),
            references ? References : Keyframes));

        foreach (string sampling in new[] { "steps", "flowShift", "sampler", "cfg" })
            Assert.False(payload.ContainsKey(sampling), $"the H3 request carries {sampling}");
    }

    /// <summary>With LoRA plug-ins, every step frame names them; the clip and the end do not.</summary>
    [Fact]
    public async Task TheStepFramesNameThePlugInsTheClipIsMadeWith()
    {
        object[] service =
        {
            new { videoGen = true, step = 0, total = 8, phase = "text-encode", detail = (string?)null, elapsedSeconds = 0.5, etaSeconds = -1.0 },
            new { videoGen = true, step = 1, total = 8, phase = "denoise", detail = (string?)null, elapsedSeconds = 9.0, etaSeconds = 63.0 },
            new { videoGen = true, step = 8, total = 8, phase = "vae-decode", detail = (string?)null, elapsedSeconds = 72.0, etaSeconds = -1.0 },
            new
            {
                done = true, url = "/uploads/v.mp4", audioUrl = "/uploads/v.wav", audioMuxed = true,
                width = 640, height = 384, frames = 22, fps = 24, seed = 1L, codec = "h264", elapsedSeconds = 80.0,
            },
        };
        List<JsonElement> named = await Collect(VideoTurns.Translate(ServiceFrames(service), "s", default, new[] { "LightX2V Turbo 8-Step" }));

        JsonElement[] steps = named.Where(f => f.TryGetProperty("video_step", out _)).ToArray();
        Assert.Equal(3, steps.Length);
        Assert.All(steps, f => Assert.Equal(new[] { "LightX2V Turbo 8-Step" },
            f.GetProperty("video_loras").EnumerateArray().Select(n => n.GetString())));
        Assert.All(named.Where(f => !f.TryGetProperty("video_step", out _)), f => Assert.False(f.TryGetProperty("video_loras", out _)));
        Assert.Equal(63.0, steps[1].GetProperty("eta").GetDouble());

        // Without plug-ins the frames are what they always were.
        List<JsonElement> plain = await Collect(VideoTurns.Translate(ServiceFrames(service), "s"));
        Assert.All(plain, f => Assert.False(f.TryGetProperty("video_loras", out _)));
        Assert.Equal(named.Count, plain.Count);
    }

    /// <summary>
    /// A chat clip asks the host once, after the request is read and recorded, what plug-ins
    /// it is made with, and the video service applies exactly those to the model before it
    /// runs: through <see cref="ImageTurns.FramesFor"/>, the one route every turn takes. A choice
    /// the host cannot honour ends the turn with the reason, and nothing is applied or made.
    /// </summary>
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task AClipIsMadeWithThePlugInsPreparedOnceForIt(bool refused)
    {
        string root = Path.Combine(Path.GetTempPath(), "video-loras-" + Guid.NewGuid().ToString("N"));
        string uploads = Path.Combine(root, "uploads");
        Directory.CreateDirectory(uploads);
        try
        {
            string gguf = Path.Combine(root, "stand-in.gguf");
            using (var writer = new BinaryWriter(File.Create(gguf)))
            {
                writer.Write(0x46554747u);
                writer.Write(3u);
                writer.Write(0UL);
                writer.Write(0UL);
                writer.Write(new byte[8]);
            }
            string weights = Path.Combine(root, "turbo.safetensors");
            File.WriteAllBytes(weights, new byte[16]);
            using var model = new PlugInVideoModel(gguf);
            using var models = new ModelService();
            FieldInfo modelField = models.LifecycleService.GetType().GetField("_model", BindingFlags.Instance | BindingFlags.NonPublic)!;
            modelField.SetValue(models.LifecycleService, model);
            try
            {
                var options = new ServerHostingOptions(
                    startupModelPath: gguf, startupMmProjPath: null, defaultBackend: "cpu",
                    supportedBackends: new[] { new BackendOption("cpu", "CPU") },
                    defaultMaxTokens: 256, maxTokensPinned: false,
                    defaultVideoFrames: 0, defaultVideoFps: 0, defaultVideoWidth: 0,
                    defaultVideoHeight: 0, defaultVideoSteps: 0, defaultVideoMode: null,
                    uploadDirectory: uploads, logDirectory: Path.Combine(root, "logs"),
                    fileLoggingEnabled: false, samplingDefaults: null);
                var chat = new WebUiChatService(models, new SessionManager(), options,
                    new UploadStoragePolicy(uploads), new SkillRegistry(new SkillRegistryOptions()),
                    codeRunner: null, workspaces: null, codeArtifacts: null, NullLoggerFactory.Instance);
                var recorded = new List<string>();
                chat.OnChatRequest = (session, _) => recorded.Add(session);
                var spec = new LoraSpec(weights, 1f, null);
                var asked = new List<bool>();
                ImageTurns.Preparation Prepare(bool editing)
                {
                    asked.Add(editing);
                    // Asked after the request is recorded, as a picture asks after its plan.
                    Assert.Equal(new[] { "s" }, recorded);
                    return refused
                        ? ImageTurns.Preparation.Refused("LightX2V Turbo 8-Step is turned on but its files are missing.")
                        : ImageTurns.Preparation.Ready(new[] { spec }, new[] { "LightX2V Turbo 8-Step" });
                }

                List<JsonElement> frames = await Collect(ImageTurns.FramesFor(chat, Body(new { role = "user", content = "a fox in falling snow" }),
                    CancellationToken.None, Prepare, ImageTurns.Planner.For(chat, uploads)));

                Assert.Equal(new[] { false }, asked);
                JsonElement last = frames[^1];
                Assert.True(last.GetProperty("done").GetBoolean());
                Assert.Equal("s", last.GetProperty("sessionId").GetString());
                if (refused)
                {
                    Assert.Equal("LightX2V Turbo 8-Step is turned on but its files are missing.", Assert.Single(frames).GetProperty("error").GetString());
                    Assert.Empty(model.Applied);
                    Assert.Equal(0, model.GenerateCalls);
                    return;
                }
                IReadOnlyList<LoraSpec> applied = Assert.Single(model.Applied);
                Assert.Equal(spec, Assert.Single(applied));
                Assert.Equal(1, model.GenerateCalls);
                // The model's own sampling: nothing the request could contradict a plug-in with.
                VideoGenerationParams parameters = model.Parameters!;
                Assert.Equal(0, parameters.Steps);
                Assert.Equal(0f, parameters.FlowShift);
                Assert.Null(parameters.Sampler);
                Assert.Contains(PlugInVideoModel.Captured, last.GetProperty("error").GetString());
            }
            finally
            {
                modelField.SetValue(models.LifecycleService, null);
            }
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
        public int DefaultPreviewFrames { get; init; } = VideoTurns.DefaultFrames;
        public bool SupportsAudio { get; init; } = true;
        public bool SupportsImageConditioning { get; init; }
        public bool SupportsEndImageConditioning { get; init; }
        public bool SupportsKeyframeConditioning { get; init; }
        public bool SupportsReferenceConditioning { get; init; }
        public int MaxReferenceImages { get; init; }

        public GeneratedVideo GenerateVideo(string prompt, VideoGenerationParams? p = null) =>
            throw new NotSupportedException("the request shaping under test never generates");
    }

    private sealed class CapturingVideoModel : ModelBase, IVideoGenerationModel
    {
        public const string Captured = "Video request captured before inference.";
        public string? Prompt { get; private set; }
        public VideoGenerationParams? Parameters { get; private set; }
        public int GenerateCalls { get; private set; }

        public CapturingVideoModel(string path, bool references, bool ltx = false) : base(path, BackendType.Cpu)
        {
            VideoModelFamily = ltx ? "ltx-2.5" : "minimax-h3";
            Config = new ModelConfig { Architecture = VideoModelFamily };
            SupportsReferenceConditioning = references;
        }

        public string VideoModelFamily { get; }
        public int DefaultPreviewFrames => SupportsKeyframeConditioning ? 17 : VideoTurns.DefaultFrames;
        public bool SupportsAudio => true;
        public bool SupportsImageConditioning => true;
        public bool SupportsEndImageConditioning => !SupportsReferenceConditioning;
        public bool SupportsReferenceConditioning { get; }
        public bool SupportsKeyframeConditioning => VideoModelFamily == "ltx-2.5";
        public int MaxReferenceImages => SupportsReferenceConditioning ? 9 : SupportsKeyframeConditioning ? 16 : 0;

        public GeneratedVideo GenerateVideo(string prompt, VideoGenerationParams? parameters = null)
        {
            GenerateCalls++;
            Prompt = prompt;
            Parameters = parameters;
            throw new NotSupportedException(Captured);
        }

        protected override float[] ForwardCore(int[] tokens) => throw new NotSupportedException();
        protected override void ResetKVCacheCore() { }
    }

    /// <summary>A MiniMax-H3 keyframes stand-in that takes LoRA plug-ins and records each set
    /// it is given; it stops at the clip, before any inference.</summary>
    private sealed class PlugInVideoModel : ModelBase, IVideoGenerationModel, ILoraPluginHost
    {
        public const string Captured = "Clip request captured before inference.";
        private IReadOnlyList<LoraSpec> _loras = Array.Empty<LoraSpec>();

        public PlugInVideoModel(string path) : base(path, BackendType.Cpu) =>
            Config = new ModelConfig { Architecture = VideoModelFamily };

        public List<IReadOnlyList<LoraSpec>> Applied { get; } = new();
        public VideoGenerationParams? Parameters { get; private set; }
        public int GenerateCalls { get; private set; }

        public string VideoModelFamily => "minimax-h3";
        public int DefaultPreviewFrames => VideoTurns.DefaultFrames;
        public bool SupportsAudio => true;
        public bool SupportsImageConditioning => true;
        public bool SupportsEndImageConditioning => true;
        public bool SupportsReferenceConditioning => false;
        public bool SupportsKeyframeConditioning => false;
        public int MaxReferenceImages => 0;

        public string LoraPluginType => LoraCliFlags.MiniMaxH3PluginType;
        public IReadOnlyList<LoraSpec> LoraSpecs => _loras;
        public bool AcceptsLora(LoraSpec spec) => true;

        public void SetLoras(IReadOnlyList<LoraSpec> specs)
        {
            Applied.Add(specs.ToArray());
            _loras = specs.ToArray();
        }

        public GeneratedVideo GenerateVideo(string prompt, VideoGenerationParams? parameters = null)
        {
            GenerateCalls++;
            Parameters = parameters;
            throw new NotSupportedException(Captured);
        }

        protected override float[] ForwardCore(int[] tokens) => throw new NotSupportedException();
        protected override void ResetKVCacheCore() { }
    }
}
