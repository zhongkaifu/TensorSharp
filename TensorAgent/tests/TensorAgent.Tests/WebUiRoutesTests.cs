using System.Net;
using System.Net.Http.Json;
using System.Text;
using System.Text.Json;
using Microsoft.Extensions.Logging.Abstractions;
using TensorAgent.Core.Catalog;
using TensorAgent.Core.Hosting;
using TensorAgent.Core.Localization;
using TensorAgent.Core.Sessions;
using TensorAgent.Core.Settings;
using TensorSharp.AgentHost.Skills;
using TensorSharp.Chat;
using TensorSharp.Server.Hosting;
using TensorSharp.Server;

namespace TensorAgent.Tests;

/// <summary>
/// The API the WebView talks to, exercised over real HTTP against the real chat
/// service.
///
/// <para>
/// The page inside the app is TensorSharp.Server's index.html unchanged, so what
/// matters is not that these handlers do something reasonable but that they answer
/// the same paths with the same shapes as the desktop server. A test that called the
/// handlers directly would miss the parts that actually break: status codes, the
/// event-stream framing, and the token gate.
/// </para>
/// </summary>
public sealed class WebUiRoutesTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "tensoragent-api-" + Guid.NewGuid().ToString("N"));
    private readonly LoopbackServer _server;
    private readonly HttpClient _client;
    private readonly WebUiChatService _chat;
    private readonly ConversationStore _conversations;
    private readonly SettingsStore _settings;
    private readonly ModelStore _models;

    public WebUiRoutesTests()
    {
        Directory.CreateDirectory(_root);
        var options = new ServerHostingOptions(
            startupModelPath: Path.Combine(_root, "models", "none.gguf"),
            startupMmProjPath: null,
            defaultBackend: "ggml_cpu",
            supportedBackends: new[] { new BackendOption("ggml_cpu", "GGML CPU") },
            defaultMaxTokens: 256,
            maxTokensPinned: false,
            defaultVideoFrames: 0, defaultVideoFps: 0, defaultVideoWidth: 0,
            defaultVideoHeight: 0, defaultVideoSteps: 0, defaultVideoMode: null,
            uploadDirectory: _root,
            logDirectory: Path.Combine(_root, "logs"),
            fileLoggingEnabled: false,
            samplingDefaults: null);

        _chat = new WebUiChatService(
            new ModelService(), new SessionManager(), options,
            new UploadStoragePolicy(_root), new SkillRegistry(new SkillRegistryOptions()),
            codeRunner: null, workspaces: null, codeArtifacts: null,
            NullLoggerFactory.Instance);

        _conversations = new ConversationStore(Path.Combine(_root, "chats"));
        _settings = new SettingsStore(Path.Combine(_root, "settings.json"));
        _models = new ModelStore(Path.Combine(_root, "weights"));

        _server = new LoopbackServer(NullLogger.Instance);
        _server.MapWebUi(_chat, _root);
        _server.MapAgent(ModelCatalog.BuiltIn, _models, _conversations, _settings, () => "test engine");
        _server.Start();

        _client = new HttpClient { BaseAddress = new Uri(_server.BaseUrl) };
        _client.DefaultRequestHeaders.Add("Cookie", $"{LoopbackServer.TokenCookie}={_server.Token}");
    }

    public void Dispose()
    {
        _client.Dispose();
        _server.Dispose();
        try { Directory.Delete(_root, true); } catch { }
    }

    private static async Task<JsonElement> BodyOf(HttpResponseMessage response)
        => JsonSerializer.Deserialize<JsonElement>(await response.Content.ReadAsStringAsync());

    // ---- the token gate -------------------------------------------------------------

    [Fact]
    public async Task WithoutTheLaunchTokenTheApiIsClosed()
    {
        using var bare = new HttpClient { BaseAddress = new Uri(_server.BaseUrl) };
        HttpResponseMessage response = await bare.GetAsync("/api/models");
        Assert.Equal(HttpStatusCode.Forbidden, response.StatusCode);
    }

    [Fact]
    public async Task TheEntryUrlHandsOutTheCookieSoTheWebViewNeedsNothingElse()
    {
        Assert.Contains($"?token={_server.Token}", _server.EntryUrl, StringComparison.Ordinal);
        using var bare = new HttpClient { BaseAddress = new Uri(_server.BaseUrl) };
        HttpResponseMessage response = await bare.GetAsync($"/api/models?token={_server.Token}");
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        Assert.Contains(response.Headers.GetValues("Set-Cookie"), v => v.Contains(_server.Token, StringComparison.Ordinal));
    }

    [Fact]
    public async Task UploadBatch_ReturnsEveryFileAndServesItsExactContent()
    {
        using var multipart = new MultipartFormDataContent();
        multipart.Add(new StringContent("first source"), "files", "source.txt");
        multipart.Add(new StringContent("second source"), "files", "source.txt");
        using HttpResponseMessage response = await _client.PostAsync("/api/upload", multipart);
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        JsonElement body = await BodyOf(response);
        Assert.True(body.GetProperty("ok").GetBoolean());
        JsonElement[] files = body.GetProperty("files").EnumerateArray().ToArray();
        Assert.Equal(2, files.Length);
        Assert.Equal("first source", files[0].GetProperty("textContent").GetString());
        Assert.Equal("second source", files[1].GetProperty("textContent").GetString());
        Assert.NotEqual(files[0].GetProperty("file").GetString(), files[1].GetProperty("file").GetString());
        foreach (JsonElement file in files)
        {
            Assert.Equal("source.txt", file.GetProperty("fileName").GetString());
            Assert.Equal(file.GetProperty("textContent").GetString(), await _client.GetStringAsync(file.GetProperty("url").GetString()));
        }
    }

    [Fact]
    public async Task UploadSingle_RetainsExistingResponse()
    {
        using var multipart = new MultipartFormDataContent();
        multipart.Add(new StringContent("one source"), "file", "source.txt");
        using HttpResponseMessage response = await _client.PostAsync("/api/upload", multipart);
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        JsonElement body = await BodyOf(response);
        Assert.Equal("one source", body.GetProperty("textContent").GetString());
        Assert.False(body.TryGetProperty("files", out _));
    }

    [Fact]
    public async Task UploadBatch_RejectsTheWholeRequestWhenALaterExtensionIsInvalid()
    {
        using var multipart = new MultipartFormDataContent();
        multipart.Add(new StringContent("valid source"), "files", "source.txt");
        multipart.Add(new ByteArrayContent(new byte[] { 1, 2, 3 }), "files", "program.exe");
        using HttpResponseMessage response = await _client.PostAsync("/api/upload", multipart);
        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        Assert.Contains(".exe", (await BodyOf(response)).GetProperty("error").GetString());
        Assert.Empty(Directory.GetFiles(_root, "*.txt"));
    }

    // ---- the shared Web UI surface --------------------------------------------------

    [Fact]
    public async Task TheModelsRouteAnswersTheShapeThePageReads()
    {
        JsonElement body = await BodyOf(await _client.GetAsync("/api/models"));
        // The exact member names the page reads; renaming one of these silently
        // empties a control in the UI, which is why they are asserted by name.
        Assert.True(body.TryGetProperty("models", out _));
        Assert.True(body.TryGetProperty("mmProjModels", out _));
        Assert.True(body.TryGetProperty("supportedBackends", out JsonElement backends));
        Assert.Equal("ggml_cpu", backends[0].GetProperty("Value").GetString());
        Assert.True(body.TryGetProperty("loaded", out _));
        Assert.True(body.TryGetProperty("defaultBackend", out _));
        Assert.True(body.TryGetProperty("defaultMaxTokens", out _));
    }

    [Fact]
    public async Task TheQueueStatusRouteAnswers()
    {
        HttpResponseMessage response = await _client.GetAsync("/api/queue/status");
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
    }

    [Fact]
    public async Task ASessionCanBeCreatedAndDisposedThroughTheSameRoutesTheDesktopUses()
    {
        JsonElement created = await BodyOf(await _client.PostAsync("/api/sessions", null));
        string id = created.GetProperty("sessionId").GetString()!;
        Assert.NotEmpty(id);

        HttpResponseMessage disposed = await _client.DeleteAsync($"/api/sessions/{id}");
        Assert.Equal(HttpStatusCode.OK, disposed.StatusCode);
    }

    [Fact]
    public async Task DisposingASessionThatDoesNotExistIsA404AndNotAServerError()
    {
        HttpResponseMessage response = await _client.DeleteAsync("/api/sessions/does-not-exist");
        Assert.Equal(HttpStatusCode.NotFound, response.StatusCode);
    }

    [Fact]
    public async Task AChatRequestWithNoModelLoadedIsRefusedWithAStatusNotAnEmptyStream()
    {
        // This is the regression that matters most: an event stream whose headers went
        // out before the first frame turns every refusal into a 200 with no content,
        // and the page shows nothing at all rather than the reason.
        HttpResponseMessage response = await _client.PostAsync("/api/chat",
            new StringContent("""{"messages":[{"role":"user","content":"hi"}]}""", Encoding.UTF8, "application/json"));

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        Assert.DoesNotContain("event-stream", response.Content.Headers.ContentType?.MediaType ?? string.Empty, StringComparison.Ordinal);
        JsonElement body = await BodyOf(response);
        Assert.False(string.IsNullOrWhiteSpace(body.GetProperty("error").GetString()));
    }

    [Fact]
    public async Task TheAnswerIsWrittenDownWhenTheStreamEndsAndNotWhenTheNextRequestCarriesIt()
    {
        // The recorder has tests of its own; this is about the wiring. /api/chat wraps
        // the service's frames in WebUiRoutes.Recording, and that wrapper is what closes
        // the turn. Drop it and an answer is saved only when the NEXT request happens to
        // carry it — which on a phone means the answer the user is reading is exactly
        // the one that is lost. Everything on this path is the real thing except the
        // frames themselves, because producing those needs a loaded model.
        var recorder = new ConversationRecorder(_conversations);
        using var server = new LoopbackServer(NullLogger.Instance) { RequireToken = false };
        server.MapWebUi(_chat, _root, skills: null, recorder: recorder, chatFrames: (body, _) => Answer(body));
        server.Start();
        using var client = new HttpClient { BaseAddress = new Uri(server.BaseUrl) };

        JsonElement created = await BodyOf(await client.PostAsync("/api/sessions?conversation=new", null));
        string sessionId = created.GetProperty("sessionId").GetString()!;
        string conversationId = created.GetProperty("conversationId").GetString()!;

        HttpResponseMessage streamed = await client.PostAsync("/api/chat", new StringContent(
            $$"""{"sessionId":"{{sessionId}}","messages":[{"role":"user","content":"what is 2 + 2"}]}""",
            Encoding.UTF8, "application/json"));
        string frames = await streamed.Content.ReadAsStringAsync();

        // The page still sees every frame: the wrapper reads them, it does not eat them.
        Assert.Equal("text/event-stream", streamed.Content.Headers.ContentType!.MediaType);
        Assert.Contains("data: {\"thinking\":\"adding them\"}", frames, StringComparison.Ordinal);
        Assert.Contains("data: {\"token\":\"It is \"}", frames, StringComparison.Ordinal);

        // And the answer is on disk with no second request having been made, which is
        // what a relaunch reads.
        Conversation saved = new ConversationStore(_conversations.Root).Load(conversationId)!;
        StoredMessage answer = Assert.Single(saved.Messages);
        Assert.Equal("assistant", answer.Role);
        Assert.Equal("It is 4.", answer.Content);
        Assert.Equal("adding them", answer.Thinking);
        StoredTurnStats stats = Assert.IsType<StoredTurnStats>(answer.Stats);
        Assert.Equal(2, stats.TokenCount);
        Assert.Equal(1.5, stats.Elapsed);
        Assert.Equal(8.0, stats.TokensPerSecond);

        // Shaped like the frames the chat service produces: the session id arrives only
        // on the last one, and it is what tells the wrapper where to file the answer.
        static async IAsyncEnumerable<object> Answer(JsonElement body)
        {
            string sessionId = body.GetProperty("sessionId").GetString()!;
            yield return new { thinking = "adding them" };
            yield return new { token = "It is " };
            await Task.Yield();
            yield return new { token = "4." };
            yield return new
            {
                done = true, tokenCount = 2, elapsed = 1.5, tokPerSec = 8.0,
                aborted = false, error = (string?)null, sessionId,
            };
        }
    }

    /// <summary>
    /// The route's recorder keeps what a picture was made from beside the picture, and a
    /// question's choices beside the question, for the same reason it keeps the answer: the
    /// next picture turn plans from them, and the page may be gone when the turn ends.
    /// </summary>
    [Fact]
    public async Task ThePicturesRecordAndAQuestionsChoicesAreWrittenDownWithTheTurn()
    {
        var recorder = new ConversationRecorder(_conversations);
        using var server = new LoopbackServer(NullLogger.Instance) { RequireToken = false };
        server.MapWebUi(_chat, _root, skills: null, recorder: recorder, chatFrames: (body, _) => Turn(body));
        server.Start();
        using var client = new HttpClient { BaseAddress = new Uri(server.BaseUrl) };

        JsonElement created = await BodyOf(await client.PostAsync("/api/sessions?conversation=new", null));
        string sessionId = created.GetProperty("sessionId").GetString()!;
        string conversationId = created.GetProperty("conversationId").GetString()!;

        foreach (string ask in new[] { "picture", "question" })
        {
            HttpResponseMessage streamed = await client.PostAsync("/api/chat", new StringContent(
                $$"""{"sessionId":"{{sessionId}}","ask":"{{ask}}","messages":[{"role":"user","content":"with a hat"}]}""",
                Encoding.UTF8, "application/json"));
            await streamed.Content.ReadAsStringAsync();

            StoredMessage answer = new ConversationStore(_conversations.Root).Load(conversationId)!.Messages[^1];
            if (ask == "picture")
            {
                Assert.Equal("/uploads/hat.png", answer.ImageUrl);
                Assert.Equal("edit", answer.ImagePlan);
                Assert.Equal(new[] { "dog.png" }, answer.ImageSources);
                Assert.Equal("with a hat", answer.ImagePrompt);
                Assert.Equal(0, answer.ImageSeed);
                Assert.Equal("mask.png", answer.ImageMask!.MaskPath);
                Assert.Contains("mask.png", answer.ReferencedUploads);
            }
            else
            {
                Assert.Equal("Which one?", answer.Content);
                Assert.Equal(new[] { "edit:dog.png", "new:" },
                    answer.ImageChoices!.Select(c => c.Intent + ":" + c.Source));
            }
        }

        static async IAsyncEnumerable<object> Turn(JsonElement body)
        {
            string sessionId = body.GetProperty("sessionId").GetString()!;
            await Task.Yield();
            if (body.GetProperty("ask").GetString() == "picture")
            {
                yield return new { image_plan = "edit", image_sources = new[] { "/uploads/dog.png" }, image_prompt = "with a hat", image_plan_reason = "model" };
                yield return new
                {
                    imageUrl = "/uploads/hat.png", width = 64, height = 64,
                    imagePlan = "edit", imageSources = new[] { "dog.png" }, imagePrompt = "with a hat", imageSeed = 0L,
                    imageMask = new { maskPath = "mask.png", maskMode = "grayscale" },
                };
            }
            else
            {
                yield return new { token = "Which one?" };
                yield return new { image_choice = new object[] { new { intent = "edit", source = "dog.png" }, new { intent = "new", source = (string?)null } } };
            }
            yield return new { done = true, sessionId };
        }
    }

    /// <summary>
    /// What the page asks before it decides which chat to open.
    ///
    /// <para>
    /// Launching the app should show an empty composer. A page that is merely reloading
    /// inside an app that never stopped — WebKit kills the content process of a WebView
    /// whose view left the window — should come back to the chat it was in, which may
    /// still have an answer being generated for it here. Both are the same page load
    /// seen from inside the page, so the host answers for it.
    /// </para>
    /// <para>
    /// The last assertion is the one a bare "is any session bound" check gets wrong:
    /// sessions are released as chats are closed, so a long-running app would report
    /// itself freshly launched the moment the user closed a chat, and then throw away
    /// the next one they opened.
    /// </para>
    /// </summary>
    [Fact]
    public async Task TheHostCallsItAColdLaunchOnlyUntilTheFirstChatIsOpened()
    {
        var recorder = new ConversationRecorder(_conversations);
        using var server = new LoopbackServer(NullLogger.Instance) { RequireToken = false };
        server.MapWebUi(_chat, _root, skills: null, recorder: recorder);
        server.Start();
        using var client = new HttpClient { BaseAddress = new Uri(server.BaseUrl) };

        Assert.True(
            (await BodyOf(await client.GetAsync("/api/agent/launch"))).GetProperty("cold").GetBoolean(),
            "the first page load after the app started is the launch, and gets a clean chat");

        JsonElement created = await BodyOf(await client.PostAsync("/api/sessions?conversation=new", null));
        string sessionId = created.GetProperty("sessionId").GetString()!;
        string conversationId = created.GetProperty("conversationId").GetString()!;

        JsonElement warm = await BodyOf(await client.GetAsync("/api/agent/launch"));
        Assert.False(warm.GetProperty("cold").GetBoolean(),
            "a page reloading inside a running app must resume its chat, not start over");

        // And it says WHICH chat. This one has no messages, so it is not in
        // /api/agent/conversations at all — a page left to guess from that list would
        // send the user to a different conversation than the one it was in.
        Assert.Equal(conversationId, warm.GetProperty("conversation").GetString());
        Assert.DoesNotContain(
            new ConversationStore(_conversations.Root).List(), c => c.Id == conversationId);

        await client.DeleteAsync("/api/sessions/" + sessionId);

        Assert.False(
            (await BodyOf(await client.GetAsync("/api/agent/launch"))).GetProperty("cold").GetBoolean(),
            "closing a chat made a running app look freshly launched");
    }

    [Fact]
    public async Task TheFileATurnProducedIsWrittenDownWithTheAnswerThatMentionsIt()
    {
        // The PDF is usually the point of the turn, and it does not arrive in the
        // answer: a small model repeats the link erratically, so the page renders it
        // from the frames instead. That makes the frames the only record of it, and a
        // record only the page holds is gone the moment the user opens another chat.
        var recorder = new ConversationRecorder(_conversations);
        using var server = new LoopbackServer(NullLogger.Instance) { RequireToken = false };
        server.MapWebUi(_chat, _root, skills: null, recorder: recorder, chatFrames: (body, _) => Answer(body));
        server.Start();
        using var client = new HttpClient { BaseAddress = new Uri(server.BaseUrl) };

        JsonElement created = await BodyOf(await client.PostAsync("/api/sessions?conversation=new", null));
        string sessionId = created.GetProperty("sessionId").GetString()!;
        string conversationId = created.GetProperty("conversationId").GetString()!;

        await client.PostAsync("/api/chat", new StringContent(
            $$"""{"sessionId":"{{sessionId}}","messages":[{"role":"user","content":"make a pdf of this"}]}""",
            Encoding.UTF8, "application/json"));

        Conversation saved = new ConversationStore(_conversations.Root).Load(conversationId)!;
        StoredMessage answer = Assert.Single(saved.Messages);
        StoredArtifact file = Assert.Single(answer.Artifacts!);
        Assert.Equal("photo.pdf", file.Name);
        Assert.Equal(40960, file.Bytes);
        Assert.Equal("/api/code/artifacts/r/photo.pdf", file.Url);

        // The same file announced twice in one turn is still one file.
        static async IAsyncEnumerable<object> Answer(JsonElement body)
        {
            string sessionId = body.GetProperty("sessionId").GetString()!;
            object files = new[] { new { name = "photo.pdf", bytes = 40960, url = "/api/code/artifacts/r/photo.pdf" } };
            yield return new { skill_step = "shell", skill = "documents", ok = true, files };
            await Task.Yield();
            yield return new { skill_step = "shell", skill = "documents", ok = true, files };
            yield return new { token = "Here is your PDF." };
            yield return new { done = true, tokenCount = 4, aborted = false, error = (string?)null, sessionId };
        }
    }

    [Fact]
    public async Task WithNoSkillRegistryTheSkillsRouteStillAnswersThePage()
    {
        HttpResponseMessage response = await _client.GetAsync("/api/skills");
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        JsonElement body = await BodyOf(response);
        Assert.True(body.TryGetProperty("skills", out _));
    }

    // ---- the app's own surface ------------------------------------------------------

    [Fact]
    public async Task TheCatalogListsTheBuiltInModelsWithTheirInstallState()
    {
        JsonElement body = await BodyOf(await _client.GetAsync("/api/agent/catalog"));
        JsonElement models = body.GetProperty("models");
        Assert.Equal(ModelCatalog.BuiltIn.Count, models.GetArrayLength());

        JsonElement first = models[0];
        Assert.False(string.IsNullOrWhiteSpace(first.GetProperty("id").GetString()));
        Assert.Equal("NotInstalled", first.GetProperty("state").GetString());
        Assert.True(first.GetProperty("totalBytes").GetInt64() > 0);
        Assert.True(first.GetProperty("remainingBytes").GetInt64() > 0);
    }

    [Fact]
    public async Task TheCatalogSerializesItsFamiliesAndArchitecturesByName()
    {
        JsonElement body = await BodyOf(await _client.GetAsync("/api/agent/catalog"));
        var families = new HashSet<string>(StringComparer.Ordinal);
        var kinds = new HashSet<string>(StringComparer.Ordinal);
        foreach (JsonElement model in body.GetProperty("models").EnumerateArray())
        {
            families.Add(model.GetProperty("family").GetString()!);
            kinds.Add(model.GetProperty("kind").GetString()!);
        }
        Assert.True(families.SetEquals(new[]
            {
                "Gemma4", "Qwen35", "Bonsai", "Qwen38", "MuseGlimmer", "Qwen38FlashNext", "QwenImage", "MiniMaxH3",
                "Qwen36", "GptOss", "Nemotron", "Mistral3", "HunyuanDense", "DeepSeek4", "DeepSeek41", "Glm5",
                "DiffusionGemma", "Wan",
            }),
            string.Join(", ", families));
        Assert.True(kinds.SetEquals(new[] { "Dense", "MixtureOfExperts", "Diffusion" }), string.Join(", ", kinds));
    }

    // The catalog reports the window a load on THIS device gets: a desktop is given more
    // than the phone's measured window, and the route must not say otherwise.
    [Fact]
    public async Task TheCatalogReportsTheContextWindowOfThisDeviceClass()
    {
        CatalogModel entry = ModelCatalog.BuiltIn.First(m => m.DesktopContextLength > m.ContextLength);
        using var desktop = new LoopbackServer(NullLogger.Instance);
        desktop.MapAgent(ModelCatalog.BuiltIn, _models, _conversations, _settings, () => "test engine",
            device: DeviceClass.Desktop);
        desktop.Start();
        using var client = new HttpClient { BaseAddress = new Uri(desktop.BaseUrl) };
        client.DefaultRequestHeaders.Add("Cookie", $"{LoopbackServer.TokenCookie}={desktop.Token}");

        JsonElement onDesktop = await BodyOf(await client.GetAsync($"/api/agent/catalog/{entry.Id}"));
        JsonElement onPhone = await BodyOf(await _client.GetAsync($"/api/agent/catalog/{entry.Id}"));

        Assert.Equal(entry.DesktopContextLength, onDesktop.GetProperty("contextLength").GetInt32());
        Assert.Equal(EngineMemoryPolicy.DefaultContextLength(entry, DeviceClass.Desktop),
            onDesktop.GetProperty("contextLength").GetInt32());
        Assert.Equal(entry.ContextLength, onPhone.GetProperty("contextLength").GetInt32());
    }

    [Fact]
    public async Task AnUnknownCatalogEntryIs404()
    {
        HttpResponseMessage response = await _client.GetAsync("/api/agent/catalog/not-a-model");
        Assert.Equal(HttpStatusCode.NotFound, response.StatusCode);
    }

    [Fact]
    public async Task TheBonsai2CardIsADownloadForSixteenGigabyteDevices()
    {
        JsonElement body = await BodyOf(await _client.GetAsync("/api/agent/catalog/bonsai-2-27b-ptq1-0"));
        Assert.Equal("Bonsai", body.GetProperty("family").GetString());
        Assert.False(body.GetProperty("sideloadOnly").GetBoolean());
        Assert.Equal(16, body.GetProperty("minDeviceMemoryGB").GetInt32());
        Assert.Equal("NotInstalled", body.GetProperty("state").GetString());
        // The optional vision projector is not part of the required download.
        Assert.Equal(5_946_648_928, body.GetProperty("totalBytes").GetInt64());
    }

    [Fact]
    public async Task ConversationsAreCreatedListedRenamedAndDeleted()
    {
        JsonElement created = await BodyOf(await _client.PostAsync("/api/agent/conversations", null));
        string id = created.GetProperty("id").GetString()!;

        // A conversation with nothing in it is deliberately not listed, so give it a
        // message before asking for the list. See ConversationStore.List.
        Conversation opened = _conversations.Load(id)!;
        opened.Messages.Add(new StoredMessage { Role = "user", Content = "where should I go" });
        _conversations.Save(opened);

        JsonElement listed = await BodyOf(await _client.GetAsync("/api/agent/conversations"));
        Assert.Contains(listed.GetProperty("conversations").EnumerateArray(),
            c => c.GetProperty("id").GetString() == id);

        JsonElement renamed = await BodyOf(await _client.PostAsJsonAsync(
            $"/api/agent/conversations/{id}", new { title = "Trip planning" }));
        Assert.Equal("Trip planning", renamed.GetProperty("title").GetString());

        JsonElement loaded = await BodyOf(await _client.GetAsync($"/api/agent/conversations/{id}"));
        Assert.Equal("Trip planning", loaded.GetProperty("title").GetString());

        JsonElement deleted = await BodyOf(await _client.DeleteAsync($"/api/agent/conversations/{id}"));
        Assert.True(deleted.GetProperty("deleted").GetBoolean());

        HttpResponseMessage gone = await _client.GetAsync($"/api/agent/conversations/{id}");
        Assert.Equal(HttpStatusCode.NotFound, gone.StatusCode);
    }

    [Fact]
    public async Task TheSandboxSwitchesRoundTripAndTheNetworkStaysOffUnlessAsked()
    {
        JsonElement initial = await BodyOf(await _client.GetAsync("/api/agent/settings"));
        Assert.True(initial.GetProperty("allowCodeExecution").GetBoolean());
        Assert.False(initial.GetProperty("allowNetwork").GetBoolean());

        JsonElement saved = await BodyOf(await _client.PostAsJsonAsync("/api/agent/settings", new
        {
            allowCodeExecution = false,
            allowNetwork = true,
            maxTokens = 4096,
        }));
        Assert.False(saved.GetProperty("allowCodeExecution").GetBoolean());
        Assert.True(saved.GetProperty("allowNetwork").GetBoolean());
        Assert.Equal(4096, saved.GetProperty("maxTokens").GetInt32());

        // The store, not just the response, has to have changed: this is what the
        // sandbox reads on the next command.
        Assert.True(_settings.Load().AllowNetwork);
        Assert.False(_settings.Load().AllowCodeExecution);
    }

    [Fact]
    public async Task TheEngineRouteSaysWhatIsActuallyRunning()
    {
        JsonElement body = await BodyOf(await _client.GetAsync("/api/agent/engine"));
        Assert.Equal("test engine", body.GetProperty("engine").GetString());
        Assert.Equal(_models.Root, body.GetProperty("modelRoot").GetString());
    }

    // ---- the page itself -------------------------------------------------------------

    [Fact]
    public async Task TheServersOwnPageIsServedUnchangedApartFromAppendedCompanionAssets()
    {
        string root = Path.Combine(_root, "webui");
        Directory.CreateDirectory(root);
        const string page = "<html><head><title>TensorSharp</title></head><body><div id=\"chat\"></div></body></html>";
        await File.WriteAllTextAsync(Path.Combine(root, "index.html"), page);
        _server.StaticRoot = root;

        byte[] served = await _client!.GetByteArrayAsync("/");
        byte[] source = await File.ReadAllBytesAsync(Path.Combine(root, "index.html"));

        // The test is exact and byte-level on purpose. Reading the page into a string
        // and writing it back would drop a byte-order mark and normalise the
        // encoding, and the served file would no longer be the Server's — which is
        // the identity that makes a second copy of index.html unnecessary.
        string head = "\n<style id=\"tensoragent-language-loading\">body{visibility:hidden}</style>\n"
            + TensorAgent.Core.Localization.PageStrings.Tag() + "\n";
        const string tag = "\n<link rel=\"stylesheet\" href=\"/mask-editor.css\">\n"
            + "<script src=\"/mask-editor.js\"></script>\n<script src=\"/tensoragent.js\"></script>\n";
        string text = Encoding.UTF8.GetString(served);
        Assert.Contains(tag, text, StringComparison.Ordinal);
        Assert.Contains(head, text, StringComparison.Ordinal);
        Assert.Equal(source, Encoding.UTF8.GetBytes(text.Replace(head, string.Empty).Replace(tag, string.Empty)));
        Assert.True(text.IndexOf("i18n.js", StringComparison.Ordinal) < text.IndexOf("<body>", StringComparison.Ordinal));

        // And it goes in before the closing tag, so the page's own top-level bindings
        // already exist by the time it runs.
        Assert.True(text.IndexOf("tensoragent.js", StringComparison.Ordinal) < text.IndexOf("</body>", StringComparison.Ordinal));
        // The strings first: both scripts after them read them as they start.
        Assert.True(text.IndexOf("i18n.js", StringComparison.Ordinal) < text.IndexOf("mask-editor.js", StringComparison.Ordinal));
    }

    [Fact]
    public async Task TheCompanionScriptIsServedFromTheAssemblyAndCarriesTheAppsAdditions()
    {
        _server.StaticRoot = Path.Combine(_root, "webui");
        Directory.CreateDirectory(_server.StaticRoot);

        string script = await _client!.GetStringAsync("/tensoragent.js");

        Assert.Contains("window.TensorAgent", script, StringComparison.Ordinal);
        Assert.Contains("addAttachment", script, StringComparison.Ordinal);
        Assert.Contains("insertText", script, StringComparison.Ordinal);
        Assert.Contains("takeShare", script, StringComparison.Ordinal);
        Assert.Contains("/api/sessions?conversation=", script, StringComparison.Ordinal);
        // The routes it calls must be the ones this server actually maps.
        Assert.Contains("/api/agent/conversations", script, StringComparison.Ordinal);
        Assert.Contains("/api/agent/turns", script, StringComparison.Ordinal);
        Assert.Contains("/api/agent/share/claim", script, StringComparison.Ordinal);
        Assert.Contains("/api/agent/share/discard", script, StringComparison.Ordinal);
        Assert.Contains("shareIds", script, StringComparison.Ordinal);
        Assert.DoesNotContain("/api/agent/share/ack", script, StringComparison.Ordinal);
        Assert.DoesNotContain("/api/tensoragent/", script, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("mask-editor.js", "text/javascript", "window.TensorSharpMaskEditor")]
    [InlineData("mask-editor.css", "text/css", ".ts-mask-modal")]
    public async Task TheSelectionEditorShipsThroughTheSameLoopbackHost(string asset, string contentType, string marker)
    {
        _server.StaticRoot = Path.Combine(_root, "webui");
        Directory.CreateDirectory(_server.StaticRoot);
        using HttpResponseMessage response = await _client!.GetAsync("/" + asset);
        Assert.True(response.IsSuccessStatusCode);
        Assert.Equal(contentType, response.Content.Headers.ContentType?.MediaType);
        Assert.Contains(marker, await response.Content.ReadAsStringAsync(), StringComparison.Ordinal);
    }

    [WebJavaScriptFact]
    public async Task TheCompanionScriptIsValidJavaScriptAndCallsOnlyRoutesThatExist()
    {
        // It is injected into a page in a WebView, where a syntax error is invisible:
        // the page keeps working and session resume, native attachments and dictation
        // all quietly do nothing. The engine that parses it here is the same family
        // as the one that will run it.
        _server.StaticRoot = Path.Combine(_root, "webui");
        Directory.CreateDirectory(_server.StaticRoot);
        string script = await _client!.GetStringAsync("/tensoragent.js");

        string probe = Path.Combine(_root, "probe.js");
        await File.WriteAllTextAsync(probe, script);
        TensorAgent.Core.Sandbox.SyntaxCheckResult syntax = await WebJavaScript.CheckSyntaxAsync(probe);
        Assert.True(syntax.Ok, syntax.Message);

        // Every route it fetches must be one this server maps, or the feature it
        // belongs to fails at run time with a 404 nobody sees.
        foreach (string route in new[]
        {
            "/api/sessions", "/api/agent/conversations", "/api/agent/events",
            "/api/agent/share/claim", "/api/agent/share/discard",
        })
            Assert.Contains(route, script, StringComparison.Ordinal);
        Assert.Equal(HttpStatusCode.OK, (await _client.PostAsJsonAsync("/api/agent/events", new { type = "ready" })).StatusCode);
        Assert.Equal(HttpStatusCode.OK, (await _client.PostAsJsonAsync("/api/agent/share/claim", new { })).StatusCode);
        // There is no share queue in this route fixture, so a well-formed discard is a
        // conflict. That response proves the endpoint exists without pretending a
        // missing durable head was removed.
        Assert.Equal(HttpStatusCode.Conflict,
            (await _client.PostAsJsonAsync("/api/agent/share/discard", new { id = "missing" })).StatusCode);
    }

    /// <summary>
    /// The two things the composer must say about itself, checked where they can be
    /// checked from a terminal: in the script and the page as they are actually served.
    ///
    /// <para>
    /// The behaviour behind them is JavaScript in a WebView and is verified on a real
    /// engine by <c>MainPage</c>'s <c>uicheck</c>, which synthesises the gestures and
    /// prints one line per assertion for <c>scripts/verify-sim.sh</c>. What is worth
    /// pinning here is the contract those checks are written against, so a rename
    /// breaks a fast test rather than a simulator run nobody is about to do.
    /// </para>
    /// </summary>
    [Fact]
    public async Task TheComposerOffersVoiceByGestureAndSaysWhatTheModelIsDoing()
    {
        _server.StaticRoot = Path.Combine(_root, "webui");
        Directory.CreateDirectory(_server.StaticRoot);
        string script = await _client!.GetStringAsync("/tensoragent.js");

        // Voice is a long press on the message box, not a switch that spends a slot on
        // the only row of chrome this design has.
        Assert.DoesNotContain("$('voice')", script, StringComparison.Ordinal);
        Assert.DoesNotContain("voice.checked", script, StringComparison.Ordinal);
        Assert.Contains("pointerdown", script, StringComparison.Ordinal);
        Assert.Contains("setVoice", script, StringComparison.Ordinal);
        // And a way back to typing, or the gesture is a trap.
        Assert.Contains("$('abc')", script, StringComparison.Ordinal);

        // Reasoning is a Settings choice, not a permanent control under the composer,
        // and Skills is a menu item rather than a button on the same row. Both were
        // spending the only row of chrome this page has.
        Assert.DoesNotContain("$('think')", script, StringComparison.Ordinal);
        Assert.DoesNotContain("$('skills-btn')", script, StringComparison.Ordinal);
        Assert.Contains("state.think", script, StringComparison.Ordinal);
        Assert.Contains("refreshSettings", script, StringComparison.Ordinal);
        Assert.Contains("data-sheet", script, StringComparison.Ordinal);

        // The live panel: what it is doing, and the tail of what it is producing —
        // pinned under the message box, not inside the turn, because by the time a
        // program has been run the top of the turn is several screens away.
        Assert.Contains("$('activity')", script, StringComparison.Ordinal);
        Assert.Contains("tailOf", script, StringComparison.Ordinal);
        Assert.Contains("progress(t('page.activity.thinking'))", script, StringComparison.Ordinal);
        Assert.Equal("Thinking…", Loc.Tables.LoadFiles("en")["page"]["page.activity.thinking"]);

        // And the kept trace: the host's own record of each skill lookup and each
        // command, which is what the frame carries and the desktop page deliberately
        // throws away. Without it a phone user watching a minute of silence has no way
        // to tell working from stuck.
        Assert.Contains("skill_step", script, StringComparison.Ordinal);
        Assert.Contains("fileLine", script, StringComparison.Ordinal);

        // The page's half of the same contract. It is the app's own file rather than
        // the Server's, so it is checked in the repository where it lives.
        string page = await File.ReadAllTextAsync(Path.Combine(
            RepoRoot, "TensorAgent", "src", "TensorAgent.Maui", "wwwroot", "index.html"));
        Assert.DoesNotContain("id=\"voice\"", page, StringComparison.Ordinal);
        Assert.Contains("id=\"abc\"", page, StringComparison.Ordinal);
        Assert.Contains("hold to talk", page, StringComparison.Ordinal);
        Assert.DoesNotContain("id=\"think\"", page, StringComparison.Ordinal);
        Assert.DoesNotContain("id=\"skills-btn\"", page, StringComparison.Ordinal);
        Assert.Contains("data-sheet=\"skills-sheet\"", page, StringComparison.Ordinal);
        Assert.Contains("#activity", page, StringComparison.Ordinal);
        Assert.Contains("id=\"activity\"", page, StringComparison.Ordinal);
        // Three lines: enough to follow, too few to bury the answer underneath.
        Assert.Contains("-webkit-line-clamp: 3", page, StringComparison.Ordinal);
    }

    /// <summary>The repository this test assembly was built from.</summary>
    private static string RepoRoot
    {
        get
        {
            var directory = new DirectoryInfo(AppContext.BaseDirectory);
            while (directory is not null && !File.Exists(Path.Combine(directory.FullName, "TensorSharp.slnx")))
                directory = directory.Parent;
            return directory?.FullName
                ?? throw new InvalidOperationException($"no TensorSharp.slnx above {AppContext.BaseDirectory}");
        }
    }

    [Fact]
    public async Task AnUnknownRouteIs404WithJsonRatherThanAnEmptyBody()
    {
        HttpResponseMessage response = await _client.GetAsync("/api/nothing-here");
        Assert.Equal(HttpStatusCode.NotFound, response.StatusCode);
        JsonElement body = await BodyOf(response);
        Assert.True(body.TryGetProperty("error", out _));
    }
    /// <summary>
    /// The page is told the host's own wording for a network refusal.
    ///
    /// <para>
    /// It needs to recognise one in order to offer the switch that fixes it, and the
    /// one thing it must not do is keep a second copy of the sentence: two spellings
    /// drift, and the day they do the offer silently stops appearing and nobody finds
    /// out, because a missing button looks exactly like a refusal that did not happen.
    /// </para>
    /// </summary>
    [Fact]
    public async Task TheEngineRouteTellsThePageHowANetworkRefusalIsWorded()
    {
        JsonElement engine = await _client!.GetFromJsonAsync<JsonElement>("/api/agent/engine");

        Assert.True(engine.TryGetProperty("networkDisabledMessage", out JsonElement wording),
            "the page cannot recognise a refusal it was never told the wording of");
        Assert.Equal(
            TensorAgent.Core.Sandbox.ExecutionPolicy.NetworkDisabledMessage,
            wording.GetString());

        // And the client really looks for it, rather than carrying its own sentence.
        // The companion is only served once a static root exists, as it is in the app.
        _server.StaticRoot = Path.Combine(_root, "webui");
        Directory.CreateDirectory(_server.StaticRoot);
        string script = await _client.GetStringAsync("/tensoragent.js");
        Assert.Contains("networkDisabledMessage", script, StringComparison.Ordinal);
        Assert.DoesNotContain(
            TensorAgent.Core.Sandbox.ExecutionPolicy.NetworkDisabledMessage, script, StringComparison.Ordinal);
    }

}
