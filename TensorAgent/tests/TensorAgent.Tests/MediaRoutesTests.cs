// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Net;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using Microsoft.Extensions.Logging.Abstractions;
using TensorAgent.Core.Hosting;
using TensorSharp.AgentHost.Skills;
using TensorSharp.Chat;
using TensorSharp.Models.Media;
using TensorSharp.Server;
using TensorSharp.Server.Hosting;

namespace TensorAgent.Tests;

/// <summary>
/// Everything the media routes do before a model exists: what an upload becomes, where
/// it lands, whether it can be fetched back, and how the two generation routes refuse.
///
/// <para>
/// None of it needs weights, which is the point — a phone spends most of its time with
/// no model loaded, and every one of these paths is reachable in that state. The
/// fixtures are built by <see cref="MediaFixtures"/> through the same codec seam the
/// app uses, so a machine whose provider cannot encode a PNG or an MP4 fails here
/// rather than halfway through an upload.
/// </para>
/// </summary>
public sealed class MediaRoutesTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "tensoragent-media-" + Guid.NewGuid().ToString("N"));
    private readonly string _uploads;
    private readonly WebUiChatService _chat;
    private readonly ModelService _models;
    private readonly LoopbackServer _server;
    private readonly HttpClient _client;

    public MediaRoutesTests()
    {
        _uploads = Path.Combine(_root, "uploads");
        Directory.CreateDirectory(_uploads);

        var options = new ServerHostingOptions(
            startupModelPath: Path.Combine(_root, "models", "none.gguf"),
            startupMmProjPath: null,
            defaultBackend: "ggml_cpu",
            supportedBackends: new[] { new BackendOption("ggml_cpu", "GGML CPU") },
            defaultMaxTokens: 256,
            maxTokensPinned: false,
            defaultVideoFrames: 0, defaultVideoFps: 0, defaultVideoWidth: 0,
            defaultVideoHeight: 0, defaultVideoSteps: 0, defaultVideoMode: null,
            uploadDirectory: _uploads,
            logDirectory: Path.Combine(_root, "logs"),
            fileLoggingEnabled: false,
            samplingDefaults: null);

        _models = new ModelService();
        _chat = new WebUiChatService(
            _models, new SessionManager(), options,
            new UploadStoragePolicy(_uploads), new SkillRegistry(new SkillRegistryOptions()),
            codeRunner: null, workspaces: null, codeArtifacts: null,
            NullLoggerFactory.Instance);

        _server = new LoopbackServer(NullLogger.Instance);
        _server.MapWebUi(_chat, _uploads);
        _server.Start();

        _client = new HttpClient { BaseAddress = new Uri(_server.BaseUrl), Timeout = TimeSpan.FromMinutes(2) };
        _client.DefaultRequestHeaders.Add("Cookie", $"{LoopbackServer.TokenCookie}={_server.Token}");
    }

    public void Dispose()
    {
        _client.Dispose();
        _server.Dispose();
        try { Directory.Delete(_root, true); } catch (Exception) { /* scratch */ }
    }

    private async Task<HttpResponseMessage> UploadAsync(byte[] content, string fileName)
    {
        using var form = new MultipartFormDataContent();
        var part = new ByteArrayContent(content);
        part.Headers.ContentType = new MediaTypeHeaderValue("application/octet-stream");
        form.Add(part, "file", fileName);
        return await _client.PostAsync("/api/upload", form);
    }

    private static async Task<JsonElement> BodyOf(HttpResponseMessage response)
        => JsonSerializer.Deserialize<JsonElement>(await response.Content.ReadAsStringAsync());

    /// <summary>Read a server-sent-event stream into its frames, as the page does.</summary>
    private Task<List<JsonElement>> StreamAsync(string route, object body) => StreamAsync(_client, route, body);

    private static async Task<List<JsonElement>> StreamAsync(HttpClient client, string route, object body)
    {
        using var request = new HttpRequestMessage(HttpMethod.Post, route)
        {
            Content = new StringContent(JsonSerializer.Serialize(body), Encoding.UTF8, "application/json"),
        };
        using HttpResponseMessage response = await client.SendAsync(request, HttpCompletionOption.ResponseHeadersRead);
        var frames = new List<JsonElement>();
        await using Stream stream = await response.Content.ReadAsStreamAsync();
        using var reader = new StreamReader(stream);
        while (await reader.ReadLineAsync() is { } line)
        {
            if (line.StartsWith("data: ", StringComparison.Ordinal))
                frames.Add(JsonSerializer.Deserialize<JsonElement>(line[6..]));
        }
        return frames;
    }

    // ---- what an upload becomes ----------------------------------------------------

    [Fact]
    public async Task APictureUploadsAsAnImageAndLandsInTheUploadDirectory()
    {
        HttpResponseMessage response = await UploadAsync(MediaFixtures.RedCircleOnWhitePng(96), "circle.png");
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);

        // The member names are the page's, not ours: WebUiChatService.UploadAsync
        // returns { ok, file, url, mediaType, fileName } and the Web UI reads every
        // one of them when it builds the attachment chip.
        JsonElement body = await BodyOf(response);
        Assert.True(body.GetProperty("ok").GetBoolean());
        Assert.Equal("image", body.GetProperty("mediaType").GetString());
        Assert.Equal("circle.png", body.GetProperty("fileName").GetString());

        string saved = body.GetProperty("file").GetString()!;
        Assert.EndsWith(".png", saved, StringComparison.Ordinal);
        Assert.Equal("/uploads/" + saved, body.GetProperty("url").GetString());
        Assert.True(File.Exists(Path.Combine(_uploads, saved)), $"{saved} is not in {_uploads}");
    }

    [Fact]
    public async Task APhotoUploadsAsAnImageAndKeepsItsOwnBytes()
    {
        byte[] jpeg = MediaFixtures.Jpeg(
            MediaFixtures.SideBySideRgb(32, 16, (255, 255, 255), (0, 0, 0)), 32, 16);

        JsonElement body = await BodyOf(await UploadAsync(jpeg, "IMG_0042.JPG"));
        Assert.Equal("image", body.GetProperty("mediaType").GetString());

        // The saved copy has to be byte-identical: it is what the vision and edit
        // pipelines read, and a transcode at upload time would silently change what
        // the model is shown.
        string saved = Path.Combine(_uploads, body.GetProperty("file").GetString()!);
        Assert.Equal(jpeg, await File.ReadAllBytesAsync(saved));
    }

    /// <summary>
    /// The bug behind "Upload failed (400)", through the route that produced it.
    ///
    /// <para>
    /// iOS's photo picker names an asset with <c>NSItemProvider.SuggestedName</c> —
    /// the file name with the extension STRIPPED — and MAUI derives a part's content
    /// type from the extension it does not have, so a photo from the library reached
    /// this route as an unnamed, untyped blob and was refused before its bytes were
    /// looked at. The camera and the document picker carry an extension, which is why
    /// it looked like "images are broken" rather than "uploads are broken".
    /// </para>
    /// </summary>
    [Fact]
    public async Task AnIPhonePhotoWithNoExtensionInItsNameStillUploads()
    {
        byte[] png = MediaFixtures.RedCircleOnWhitePng(64);
        HttpResponseMessage response = await UploadAsync(png, "IMG_0004");
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);

        JsonElement body = await BodyOf(response);
        Assert.Equal("image", body.GetProperty("mediaType").GetString());

        // Saved with an extension /uploads can serve, and byte-identical: what the
        // vision pipeline reads is the file the user picked.
        string saved = body.GetProperty("file").GetString()!;
        Assert.EndsWith(".png", saved, StringComparison.Ordinal);
        Assert.Equal(png, await File.ReadAllBytesAsync(Path.Combine(_uploads, saved)));

        // And it comes back out of the route the bubble renders it from.
        HttpResponseMessage served = await _client.GetAsync("/uploads/" + saved);
        Assert.Equal(HttpStatusCode.OK, served.StatusCode);
        Assert.Equal("image/png", served.Content.Headers.ContentType?.MediaType);
    }

    /// <summary>
    /// The strictness that naming must not have loosened: an upload whose type nobody
    /// can work out is still refused, with the sentence the user should read.
    /// </summary>
    [Fact]
    public async Task AnUploadNothingCanIdentifyIsStillRefusedAndSaysWhy()
    {
        HttpResponseMessage response = await UploadAsync(new byte[512], "mystery");
        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);

        JsonElement body = await BodyOf(response);
        string error = body.GetProperty("error").GetString()!;
        Assert.Contains("not supported", error, StringComparison.OrdinalIgnoreCase);
        Assert.Empty(Directory.GetFiles(_uploads));
    }

    [Fact]
    public async Task ASoundUploadsAsAudio()
    {
        JsonElement body = await BodyOf(await UploadAsync(MediaFixtures.ToneWav(1.0), "memo.wav"));
        Assert.Equal("audio", body.GetProperty("mediaType").GetString());
        Assert.True(File.Exists(Path.Combine(_uploads, body.GetProperty("file").GetString()!)));
    }

    [Fact]
    public async Task AClipUploadsAsVideoAndItsFramesAreWrittenBesideItUnderItsOwnName()
    {
        byte[] mp4 = MediaFixtures.TwoHalvesMp4(_root, out _);
        JsonElement body = await BodyOf(await UploadAsync(mp4, "clip.mp4"));
        Assert.Equal("video", body.GetProperty("mediaType").GetString());

        string saved = body.GetProperty("file").GetString()!;
        Assert.True(File.Exists(Path.Combine(_uploads, saved)));

        // The page refers to a frame by its bare name and the chat pipeline resolves
        // that against the upload root, so frames written anywhere else are a 404 in
        // the bubble and nothing at all in the prompt.
        var frames = body.GetProperty("frames").EnumerateArray().Select(f => f.GetString()!).ToList();
        Assert.NotEmpty(frames);
        string stem = Path.GetFileNameWithoutExtension(saved);
        foreach (string frame in frames)
        {
            Assert.StartsWith(stem, frame, StringComparison.Ordinal);
            Assert.True(File.Exists(Path.Combine(_uploads, frame)), $"{frame} is not beside the clip");
        }
        Assert.Equal(
            frames.Select(f => "/uploads/" + f).ToList(),
            body.GetProperty("frameUrls").EnumerateArray().Select(f => f.GetString()!).ToList());
    }

    [Fact]
    public async Task AFileTypeTheServeSidePolicyDoesNotCoverIsRefusedBeforeItIsWritten()
    {
        HttpResponseMessage response = await UploadAsync([1, 2, 3, 4], "payload.bin");
        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        Assert.Contains("bin", (await BodyOf(response)).GetProperty("error").GetString()!, StringComparison.OrdinalIgnoreCase);
        Assert.Empty(Directory.GetFiles(_uploads));
    }

    [Fact]
    public async Task AnUploadIsServedBackFromTheUrlItsOwnResponseGave()
    {
        // Every picture in the chat, every extracted video frame, every generated
        // image and every generated clip is referenced by this URL and nothing else.
        // Without it the API succeeds and the page shows a row of broken images.
        JsonElement body = await BodyOf(await UploadAsync(MediaFixtures.RedCircleOnWhitePng(64), "circle.png"));
        string url = body.GetProperty("url").GetString()!;

        using HttpResponseMessage fetched = await _client.GetAsync(url);
        Assert.Equal(HttpStatusCode.OK, fetched.StatusCode);
        Assert.Equal("image/png", fetched.Content.Headers.ContentType?.MediaType);
        Assert.Equal(
            MediaFixtures.RedCircleOnWhitePng(64),
            await fetched.Content.ReadAsByteArrayAsync());
    }

    [Fact]
    public async Task AnUploadedPageIsNeverServedBackAsHtml()
    {
        // The upload directory holds whatever a user handed the app, and the page it
        // would run in is the same origin as the API and its launch token. The Server's
        // own policy serves every text and code extension as text/plain for exactly
        // this reason; the app has to agree with it, not with a generic MIME table.
        JsonElement body = await BodyOf(await UploadAsync(
            "<script>fetch('/api/agent/settings')</script>"u8.ToArray(), "note.html"));

        using HttpResponseMessage fetched = await _client.GetAsync(body.GetProperty("url").GetString()!);
        Assert.Equal(HttpStatusCode.OK, fetched.StatusCode);
        Assert.Equal("text/plain", fetched.Content.Headers.ContentType?.MediaType);
    }

    [Fact]
    public async Task NothingOutsideTheUploadDirectoryIsReachableThroughTheUploadsRoute()
    {
        await File.WriteAllTextAsync(Path.Combine(_root, "secret.txt"), "not yours");
        foreach (string escape in new[] { "/uploads/../secret.txt", "/uploads/..%2Fsecret.txt" })
        {
            using HttpResponseMessage response = await _client.GetAsync(escape);
            Assert.True(
                response.StatusCode is HttpStatusCode.NotFound or HttpStatusCode.BadRequest,
                $"{escape} answered {(int)response.StatusCode}");
        }
    }

    [Fact]
    public async Task AnUnreadableHeicStillUploadsInsteadOfFailingTheWholeRequest()
    {
        // The success half of this path — the PNG preview a browser can actually show —
        // cannot be covered on a development machine: no provider on any desktop
        // encodes HEIC (Magick.NET reports Heic read=true, write=false), so there is no
        // way to generate the fixture in code. What is covered is the half that
        // matters when a photo is damaged or truncated in transit: the preview is
        // best-effort, and losing it must cost the user a thumbnail, not the upload.
        byte[] notReallyHeic = [.. "\0\0\0\x18ftypheic"u8, .. new byte[64]];
        JsonElement body = await BodyOf(await UploadAsync(notReallyHeic, "IMG_0100.HEIC"));

        Assert.True(body.GetProperty("ok").GetBoolean());
        Assert.Equal("image", body.GetProperty("mediaType").GetString());
        Assert.False(body.TryGetProperty("previewUrl", out _), "an undecodable HEIC cannot have a preview");
        Assert.True(File.Exists(Path.Combine(_uploads, body.GetProperty("file").GetString()!)));
    }

    // ---- the codec seam under the upload -------------------------------------------

    [Fact]
    public void APhotoStoredSidewaysComesBackUpright()
    {
        // An iPhone writes the sensor's pixels and an orientation tag rather than
        // rotating anything, so the picture a vision model is shown is upside down
        // unless the decoder honours the tag. This is the contract IImageCodec states
        // and the one an iOS provider has to reproduce with ImageIO.
        byte[] stored = MediaFixtures.Jpeg(
            MediaFixtures.SideBySideRgb(32, 16, (255, 255, 255), (0, 0, 0)), 32, 16);
        byte[] sideways = MediaFixtures.WithExifOrientation(stored, orientation: 6);

        Assert.Equal((32, 16), MediaCodecs.Image.ReadDimensions(sideways));

        byte[] rgba = MediaCodecs.Image.DecodeRgba(sideways, out int width, out int height);
        Assert.Equal(16, width);
        Assert.Equal(32, height);

        // Orientation 6 is a quarter turn clockwise, so the white left half becomes the
        // white top half. Asserting the pixels rather than only the dimensions is what
        // separates "rotated" from "transposed the wrong way".
        Assert.Equal(255, rgba[(4 * width + 8) * 4]);
        Assert.Equal(0, rgba[((height - 5) * width + 8) * 4]);
    }

    [Fact]
    public void ATaggedPhotoAndTheSamePhotoUntaggedDifferOnlyInOrientation()
    {
        byte[] rgb = MediaFixtures.SideBySideRgb(32, 16, (255, 255, 255), (0, 0, 0));
        byte[] stored = MediaFixtures.Jpeg(rgb, 32, 16);

        byte[] upright = MediaCodecs.Image.DecodeRgba(stored, out int w, out int h);
        Assert.Equal((32, 16), (w, h));
        Assert.Equal(255, upright[(8 * w + 4) * 4]);
        Assert.Equal(0, upright[(8 * w + 28) * 4]);
    }

    // ---- the generation routes, with nothing loaded --------------------------------

    [Fact]
    public async Task ImageEditWithNoModelLoadedIsRefusedWithAReasonRatherThanAServerError()
    {
        HttpResponseMessage response = await _client.PostAsync("/api/image-edit",
            new StringContent("""{"prompt":"make it blue","imagePaths":[]}""", Encoding.UTF8, "application/json"));

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        string error = (await BodyOf(response)).GetProperty("error").GetString()!;
        Assert.Contains("Qwen-Image-2.1", error, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task VideoGenerateWithNoModelLoadedIsRefusedWithAReasonRatherThanAServerError()
    {
        HttpResponseMessage response = await _client.PostAsync("/api/video-generate",
            new StringContent("""{"prompt":"a cat on a beach"}""", Encoding.UTF8, "application/json"));

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        string error = (await BodyOf(response)).GetProperty("error").GetString()!;
        Assert.Contains("video", error, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task TheStreamingGenerationRoutesEndWithAnErrorFrameTheProgressBarCanShow()
    {
        // The streaming siblings cannot answer with a status the way the plain routes
        // do — by the time the page is reading, the headers are long gone — so the
        // refusal has to arrive as a final frame. An empty stream would leave the
        // progress bar spinning forever with nothing to explain it.
        foreach (string route in new[] { "/api/image-edit/stream", "/api/video-generate/stream" })
        {
            List<JsonElement> frames = await StreamAsync(route, new { prompt = "anything", imagePaths = Array.Empty<string>() });
            Assert.NotEmpty(frames);

            JsonElement last = frames[^1];
            Assert.True(last.GetProperty("done").GetBoolean());
            Assert.False(string.IsNullOrWhiteSpace(last.GetProperty("error").GetString()), $"{route} said nothing");
        }
    }

    /// <summary>
    /// In the app, an edit made through these routes uses the LoRA plug-ins the user chose,
    /// exactly as a picture in the chat does: a choice that cannot be honoured refuses the
    /// edit with the reason, both routes ask for an edit's plug-ins, and a choice that can
    /// be honoured goes on to the image service (which, with nothing loaded, says so).
    /// Without the preparation they used whatever set the last chat picture left behind.
    /// </summary>
    [Fact]
    public async Task TheEditRoutesUseTheHostsLoraChoiceAndRefuseOneItCannotHonour()
    {
        const string reason = "Viggle Turbo is turned on but its files are missing.";
        var asked = new List<bool>();
        bool refuse = true;
        using var server = new LoopbackServer(NullLogger.Instance);
        server.MapWebUi(_chat, _uploads, prepareImage: editing =>
        {
            lock (asked) asked.Add(editing);
            return refuse
                ? ImageTurns.Preparation.Refused(reason)
                : ImageTurns.Preparation.Ready(Array.Empty<TensorSharp.Runtime.LoraSpec>(), Array.Empty<string>());
        });
        server.Start();
        using var client = new HttpClient { BaseAddress = new Uri(server.BaseUrl), Timeout = TimeSpan.FromMinutes(2) };
        client.DefaultRequestHeaders.Add("Cookie", $"{LoopbackServer.TokenCookie}={server.Token}");
        var body = new { prompt = "make it blue", imagePaths = Array.Empty<string>() };

        HttpResponseMessage refused = await client.PostAsync("/api/image-edit",
            new StringContent(JsonSerializer.Serialize(body), Encoding.UTF8, "application/json"));
        Assert.Equal(HttpStatusCode.Conflict, refused.StatusCode);
        Assert.Equal(reason, (await BodyOf(refused)).GetProperty("error").GetString());

        JsonElement frame = Assert.Single(await StreamAsync(client, "/api/image-edit/stream", body));
        Assert.True(frame.GetProperty("done").GetBoolean());
        Assert.Equal(reason, frame.GetProperty("error").GetString());

        refuse = false;
        HttpResponseMessage served = await client.PostAsync("/api/image-edit",
            new StringContent(JsonSerializer.Serialize(body), Encoding.UTF8, "application/json"));
        Assert.Equal(HttpStatusCode.BadRequest, served.StatusCode);
        Assert.Contains("Qwen-Image-2.1", (await BodyOf(served)).GetProperty("error").GetString(), StringComparison.OrdinalIgnoreCase);

        Assert.Equal(new[] { true, true, true }, asked);
    }

    /// <summary>
    /// The route's own frame source reads a picture turn against the whole conversation with
    /// the planner it was given, as the app's GPU gate does (AgentAppHostTests). Before, it
    /// read the newest message alone, and a follow-up to a picture drew an unrelated one.
    /// The stand-in model is unsure, so the turn ends with its question and no image work.
    /// </summary>
    [Fact]
    public async Task TheDefaultChatRoutePlansAPictureTurnFromTheConversation()
    {
        var modelField = _models.LifecycleService.GetType().GetField("_model",
            System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)!;
        modelField.SetValue(_models.LifecycleService,
            System.Runtime.CompilerServices.RuntimeHelpers.GetUninitializedObject(typeof(TensorSharp.Models.QwenImage.QwenImageModel)));
        try
        {
            File.WriteAllBytes(Path.Combine(_uploads, "dog.png"), new byte[] { 1 });
            var asked = new List<ImageTurns.PlanQuestion>();
            using var server = new LoopbackServer(NullLogger.Instance);
            server.MapWebUi(_chat, _uploads, planImage: new ImageTurns.Planner(_uploads, (question, _) =>
            {
                lock (asked) asked.Add(question);
                // Every option alike: the model cannot tell what was meant, so the turn asks.
                float each = 1f / question.Options.Count;
                return Task.FromResult<TensorSharp.Models.QwenImage.ImageIntentChoice?>(
                    new(0, each, 0) { Probabilities = Enumerable.Repeat(each, question.Options.Count).ToArray() });
            }));
            server.Start();
            using var client = new HttpClient { BaseAddress = new Uri(server.BaseUrl) };
            client.DefaultRequestHeaders.Add("Cookie", $"{LoopbackServer.TokenCookie}={server.Token}");

            List<JsonElement> frames = await StreamAsync(client, "/api/chat", new
            {
                messages = new object[]
                {
                    new { role = "user", content = "a dog" },
                    new { role = "assistant", content = "", imageUrl = "/uploads/dog.png" },
                    new { role = "user", content = "with a hat" },
                },
            });

            Assert.Contains("\"with a hat\"", Assert.Single(asked).User, StringComparison.Ordinal);
            Assert.Contains(frames, f => f.TryGetProperty("image_choice", out _));
            Assert.True(frames[^1].GetProperty("done").GetBoolean());
        }
        finally
        {
            modelField.SetValue(_models.LifecycleService, null);
        }
    }

    [Fact]
    public async Task DefaultChatRouteHonoursLoraPreparationForAMaskedEdit()
    {
        // Only the model-type dispatch is needed: preparation refuses before any
        // model member or numerical kernel is used. No weights or GPU are involved.
        var modelField = _models.LifecycleService.GetType().GetField("_model",
            System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)!;
        modelField.SetValue(_models.LifecycleService,
            System.Runtime.CompilerServices.RuntimeHelpers.GetUninitializedObject(typeof(TensorSharp.Models.QwenImage.QwenImageModel)));
        try
        {
            const string reason = "The selected edit plug-in is missing.";
            var asked = new List<bool>();
            using var server = new LoopbackServer(NullLogger.Instance);
            server.MapWebUi(_chat, _uploads, prepareImage: editing =>
            {
                asked.Add(editing);
                return ImageTurns.Preparation.Refused(reason);
            });
            server.Start();
            using var client = new HttpClient { BaseAddress = new Uri(server.BaseUrl) };
            client.DefaultRequestHeaders.Add("Cookie", $"{LoopbackServer.TokenCookie}={server.Token}");
            JsonElement frame = Assert.Single(await StreamAsync(client, "/api/chat", new
            {
                messages = new[] { new { role = "user", content = "edit the selected area", stillImagePaths = new[] { "source.png" }, maskPath = "mask.png" } },
            }));
            Assert.Equal(new[] { true }, asked);
            Assert.True(frame.GetProperty("done").GetBoolean());
            Assert.Equal(reason, frame.GetProperty("error").GetString());
        }
        finally
        {
            modelField.SetValue(_models.LifecycleService, null);
        }
    }
}
