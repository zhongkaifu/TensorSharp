// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Net.Http.Headers;
using System.Net.Http.Json;
using System.Text;
using System.Text.Json;
using Microsoft.Extensions.Logging.Abstractions;
using TensorAgent.Core.Catalog;
using TensorAgent.Core.Hosting;
using TensorAgent.Core.Settings;
using TensorSharp.AgentHost.Skills;
using TensorSharp.Chat;
using TensorSharp.Models.Media;
using TensorSharp.Server;
using TensorSharp.Server.Hosting;

namespace TensorAgent.Tests;

/// <summary>
/// Which weights each media scenario needs and where to put them, kept apart from the
/// tests so a skip message can name a file rather than a variable.
///
/// <para>
/// This duplicates the gating idiom in <c>EndToEndChatTests</c> on purpose: at the time
/// of writing there is no shared <c>LiveModelHarness</c> to reuse, and inventing one
/// here would collide with whoever extracts it. When that file appears, the three
/// <c>Unavailable*</c> methods below and <c>EndToEndChatTests.Unavailable</c> should
/// become one.
/// </para>
/// </summary>
internal static class LiveMedia
{
    /// <summary>A directory holding a multimodal catalog entry's weights and projector.</summary>
    public const string ModelDirVariable = "TENSORAGENT_TEST_MODEL_DIR";

    /// <summary>The weights file inside it, when it is not named as the catalog names it.</summary>
    public const string ModelFileVariable = "TENSORAGENT_TEST_MODEL_FILE";

    /// <summary>The projector file inside it, when it is not named as the catalog names it.</summary>
    public const string ProjectorFileVariable = "TENSORAGENT_TEST_MMPROJ_FILE";

    /// <summary>A directory holding a locally supplied Qwen-Image-2.1 DiT and its companion networks.</summary>
    public const string ImageModelDirVariable = "TENSORAGENT_TEST_IMAGE_MODEL_DIR";

    /// <summary>A directory holding a Wan video DiT and its companion networks.</summary>
    public const string VideoModelDirVariable = "TENSORAGENT_TEST_VIDEO_MODEL_DIR";

    /// <summary>Which DiT inside it to use, when the directory holds more than one.</summary>
    public const string VideoModelFileVariable = "TENSORAGENT_TEST_VIDEO_MODEL_FILE";

    /// <summary>
    /// Which backend a live test asks for. The chat scenarios default to the CPU, which
    /// is what a build machine has and what the rest of the live suite already assumes;
    /// the diffusion ones default to Metal, because a single Qwen-Image-2.1 edit or Wan clip
    /// on a CPU is measured in hours and a test nobody can finish is not a test.
    /// </summary>
    public const string BackendVariable = "TENSORAGENT_TEST_BACKEND";

    public static string Backend(string fallback) =>
        Environment.GetEnvironmentVariable(BackendVariable) is { Length: > 0 } chosen ? chosen : fallback;

    /// <summary>Why the vision and audio scenarios cannot run here, or null when they can.</summary>
    internal static string? UnavailableMultimodal(out CatalogModel model, out string weights, out string projector)
    {
        model = null!;
        weights = string.Empty;
        projector = string.Empty;

        string? directory = Environment.GetEnvironmentVariable(ModelDirVariable);
        if (string.IsNullOrWhiteSpace(directory) || !Directory.Exists(directory))
            return $"set {ModelDirVariable} to a directory holding a multimodal catalog model and its mmproj "
                + "(e.g. gemma-4-E4B-it-Q8_0.gguf + mmproj-gemma-4-E4B-it-Q8_0.gguf)";

        string? namedWeights = Environment.GetEnvironmentVariable(ModelFileVariable);
        string? namedProjector = Environment.GetEnvironmentVariable(ProjectorFileVariable);

        // Of the entries whose files are present, take the one whose recorded size is
        // nearest the bytes actually on disk. Which entry is chosen decides the context
        // length, the sampling defaults and the names the files are linked under, so
        // picking the first one that merely exists would run an E4B checkpoint under an
        // E2B entry's settings whenever a local copy is named differently — and a
        // re-quantised or uncensored build is never byte-identical to the catalog's, so
        // demanding an exact match would reject the very case the override exists for.
        CatalogModel? nearest = null;
        string nearestWeights = string.Empty, nearestProjector = string.Empty;
        long nearestDistance = long.MaxValue;
        foreach (CatalogModel candidate in ModelCatalog.BuiltIn)
        {
            if (candidate.Kind == CatalogArchitectureKind.Diffusion || candidate.Projector is null)
                continue;
            if (!candidate.Modalities.HasFlag(CatalogModalities.Image))
                continue;

            string weightsPath = Path.Combine(directory, namedWeights ?? candidate.Weights.FileName);
            string projectorPath = Path.Combine(directory, namedProjector ?? candidate.Projector.FileName);
            if (!File.Exists(weightsPath) || !File.Exists(projectorPath))
                continue;

            long distance = Math.Abs(new FileInfo(weightsPath).Length - candidate.Weights.Bytes);
            if (distance >= nearestDistance)
                continue;
            nearest = candidate;
            nearestWeights = weightsPath;
            nearestProjector = projectorPath;
            nearestDistance = distance;
        }

        if (nearest is null)
            return $"no multimodal catalog model found in {directory}; set {ModelFileVariable} to the weights file name "
                + $"and {ProjectorFileVariable} to the projector file name when the local copies are named differently";

        model = nearest;
        weights = nearestWeights;
        projector = nearestProjector;
        return null;
    }

    /// <summary>The four networks a Qwen-Image-2.1 edit needs, found in one directory.</summary>
    internal sealed record QwenImageFiles(string Dit, string TextEncoder, string Vae, string VisionProjector);

    /// <summary>Why the image-editing scenarios cannot run here, or null when they can.</summary>
    internal static string? UnavailableQwenImage(out QwenImageFiles files)
    {
        files = null!;
        string? directory = Environment.GetEnvironmentVariable(ImageModelDirVariable);
        if (string.IsNullOrWhiteSpace(directory) || !Directory.Exists(directory))
            return $"set {ImageModelDirVariable} to a directory holding a Qwen-Image-2.1 DiT GGUF "
                + "(e.g. qwen_image_2.1_Q4_K_M.gguf), qwen_image_2.1_vae_bf16.safetensors, a Qwen3-VL-8B "
                + "text-encoder GGUF and its mmproj (the files config/qwen-image-2.1.json downloads)";

        // The same names the pipeline's own directory scan accepts, so a directory this
        // finds is one QwenImage21DiffusionModel would also have found.
        string[] entries = Directory.GetFiles(directory);
        string? dit = entries.FirstOrDefault(f =>
            (Named(f, "qwen_image_2.1") || Named(f, "qwen-image-2.1")) && !Named(f, "vae")
            && f.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase));
        string? textEncoder = entries.FirstOrDefault(f =>
            IsQwen3Vl8b(f) && !Named(f, "mmproj") && f.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase));
        string? vae = entries.FirstOrDefault(f =>
            Named(f, "qwen_image_2.1_vae") && f.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase));
        string? projector = entries.FirstOrDefault(f =>
            IsQwen3Vl8b(f) && Named(f, "mmproj") && f.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase));

        if (dit is null)
            return $"no Qwen-Image-2.1 DiT in {directory}: it needs a .gguf whose name contains 'qwen_image_2.1'";
        if (textEncoder is null)
            return $"no text encoder in {directory}: it needs a .gguf whose name contains 'qwen3vl-8b' and not 'mmproj'";
        if (vae is null)
            return $"no VAE in {directory}: it needs qwen_image_2.1_vae_bf16.safetensors";
        // Required, not optional: Qwen-Image-2.1 refuses to edit without its projector.
        if (projector is null)
            return $"no vision projector in {directory}: editing needs mmproj-Qwen3VL-8B-Instruct-F16.gguf";

        files = new QwenImageFiles(dit, textEncoder, vae, projector);
        return null;
    }

    private static bool IsQwen3Vl8b(string path) => Named(path, "qwen3vl-8b") || Named(path, "qwen3-vl-8b");

    /// <summary>The networks a Wan generation needs, found in one directory.</summary>
    internal sealed record VideoFiles(string Dit, string Vae, string TextEncoder);

    /// <summary>Why the video-generation scenario cannot run here, or null when it can.</summary>
    internal static string? UnavailableVideo(out VideoFiles files)
    {
        files = null!;
        string? directory = Environment.GetEnvironmentVariable(VideoModelDirVariable);
        if (string.IsNullOrWhiteSpace(directory) || !Directory.Exists(directory))
            return $"set {VideoModelDirVariable} to a directory holding a Wan DiT GGUF "
                + "(e.g. Wan2.2-TI2V-5B-Turbo-Q8_0.gguf), its VAE safetensors and a umt5-xxl encoder GGUF";

        string[] entries = Directory.GetFiles(directory);
        string[] ggufs = [.. entries.Where(f => f.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase))];

        // A directory of Wan checkpoints holds several generations at once, and picking
        // whichever the file system lists first would silently choose a 14B model, or a
        // schedule that takes a hundred passes per clip. Preference order: whatever the
        // caller named, then a step-distilled TI2V (a Turbo runs four passes rather than
        // a hundred, which is the difference between a test and an afternoon), then any
        // other Wan DiT.
        string? named = Environment.GetEnvironmentVariable(VideoModelFileVariable);
        string[] candidates = [.. ggufs.Where(f => Named(f, "wan") && !IsTextEncoder(f))];
        string? dit = named is { Length: > 0 }
            ? candidates.FirstOrDefault(f => Path.GetFileName(f).Equals(named, StringComparison.OrdinalIgnoreCase))
            : candidates.FirstOrDefault(f => Named(f, "ti2v") && Named(f, "turbo"))
                ?? candidates.FirstOrDefault(f => Named(f, "ti2v"))
                ?? candidates.FirstOrDefault();

        if (dit is null)
            return named is { Length: > 0 }
                ? $"{directory} holds no Wan DiT named '{named}'; clear {VideoModelFileVariable} or correct it"
                : $"no Wan DiT in {directory}: it needs a .gguf whose name contains 'wan'";

        // WanVideoModel pairs a TI2V checkpoint with the Wan2.2 VAE and everything else
        // with the 2.1 one, and looks in a VAE/ subdirectory as well as beside the DiT.
        // Handing it the wrong generation is a load that fails on a tensor name.
        bool ti2v = Named(dit, "ti2v");
        string vaeSubdirectory = Path.Combine(directory, "VAE");
        string[] vaes =
        [
            .. entries,
            .. Directory.Exists(vaeSubdirectory) ? Directory.GetFiles(vaeSubdirectory) : [],
        ];
        string? vae = vaes.FirstOrDefault(f =>
            Named(f, "vae") && f.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase) && Named(f, "2.2") == ti2v);
        string? textEncoder = ggufs.FirstOrDefault(IsTextEncoder);

        if (vae is null)
            return ti2v
                ? $"no Wan2.2 VAE for '{Path.GetFileName(dit)}' in {directory} or {vaeSubdirectory}: a TI2V checkpoint needs Wan2.2_VAE.safetensors"
                : $"no Wan2.1 VAE for '{Path.GetFileName(dit)}' in {directory} or {vaeSubdirectory}: it needs wan_2.1_vae.safetensors";
        if (textEncoder is null)
            return $"no text encoder in {directory}: it needs a umt5-xxl encoder GGUF (city96/umt5-xxl-encoder-gguf)";

        files = new VideoFiles(dit, vae, textEncoder);
        return null;
    }

    private static bool IsTextEncoder(string path) =>
        Named(path, "umt5") || Named(path, "t5xxl") || Named(path, "t5-xxl");

    private static bool Named(string path, string fragment) =>
        Path.GetFileName(path).Contains(fragment, StringComparison.OrdinalIgnoreCase);
}

/// <summary>A fact that needs a multimodal model and its projector on this machine.</summary>
public sealed class MultimodalModelFactAttribute : FactAttribute
{
    public MultimodalModelFactAttribute()
    {
        if (LiveMedia.UnavailableMultimodal(out _, out _, out _) is { } reason)
            Skip = reason;
    }
}

/// <summary>A fact that needs the Qwen-Image-2.1 checkpoint and its companions.</summary>
public sealed class QwenImageModelFactAttribute : FactAttribute
{
    public QwenImageModelFactAttribute()
    {
        if (LiveMedia.UnavailableQwenImage(out _) is { } reason)
            Skip = reason;
    }
}

/// <summary>A fact that needs a Wan video checkpoint and its companions.</summary>
public sealed class VideoModelFactAttribute : FactAttribute
{
    public VideoModelFactAttribute()
    {
        if (LiveMedia.UnavailableVideo(out _) is { } reason)
            Skip = reason;
    }
}

/// <summary>
/// The four things a person actually does with media in this app, each end to end
/// through the routes the page calls: show the model a picture, play it a sound, show
/// it a clip, and ask it to make one.
///
/// <para>
/// Every one of these needs weights, and they are gated separately because they need
/// different weights: understanding is one multimodal chat model, while editing and
/// video generation use explicitly supplied diffusion stacks that the built-in catalog
/// does not carry. A machine with one and not the others runs what it can and says what
/// it is missing.
/// </para>
/// <para>
/// Where an assertion can be hard it is. "The answer mentions red" is a claim about a
/// particular fine-tune's prose; "the prompt grew by hundreds of tokens when the
/// picture was attached" is a claim about whether the encoder ran at all, and a
/// projector that silently failed to load fails that one no matter how fluent the
/// answer is.
/// </para>
/// </summary>
[Collection(ProcessEnvironmentCollection.Name)]
public sealed class MediaScenarioTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "tensoragent-scenario-" + Guid.NewGuid().ToString("N"));
    private AgentAppHost? _host;
    private LoopbackServer? _server;
    private HttpClient? _client;
    private ModelStore? _diffusionStore;

    public void Dispose()
    {
        _client?.Dispose();
        _host?.Dispose();
        _server?.Dispose();
        if (_diffusionStore is not null)
            DiffusionCompanions.Publish(null, _diffusionStore);
        try { Directory.Delete(_root, true); } catch (Exception) { /* scratch */ }
    }

    // ---- harnesses -------------------------------------------------------------------

    /// <summary>
    /// Start the real app around a catalog entry whose files are linked, not copied:
    /// these are gigabytes each and a per-test copy costs more than the inference.
    /// </summary>
    private AgentAppHost StartApp(CatalogModel model, IReadOnlyDictionary<string, string> linkByCatalogName)
    {
        var paths = new AgentPaths(Path.Combine(_root, "data"), Path.Combine(_root, "cache")) { DeviceMemoryGB = 16 };
        paths.EnsureCreated();

        string target = Path.Combine(paths.ModelsDirectory, model.Id);
        Directory.CreateDirectory(target);
        foreach ((string name, string source) in linkByCatalogName)
            File.CreateSymbolicLink(Path.Combine(target, name), source);

        var settings = new SettingsStore(paths.SettingsFile);
        AppSettings chosen = settings.Load();
        chosen.SelectedModelId = model.Id;
        chosen.MaxTokens = 192;
        settings.Save(chosen);

        _host = new AgentAppHost(paths);
        // These scenarios deliberately load through /api/models/load below. Starting
        // the host would also launch its remembered-model background load, leaving two
        // unsynchronised ModelService.LoadModel calls racing over the same model and
        // projector. Start only the loopback server so the API call is the sole load.
        _host.Server.Start();
        _client = Client(_host.Server);
        return _host;
    }

    /// <summary>
    /// The same routes over a model the catalog does not carry. There is no selected-model
    /// path to start from, so the hosting options name the file directly.
    /// </summary>
    private void StartDirect(string weightsPath, string backend)
    {
        string uploads = Path.Combine(_root, "uploads");
        Directory.CreateDirectory(uploads);

        var options = new ServerHostingOptions(
            startupModelPath: weightsPath,
            startupMmProjPath: null,
            defaultBackend: backend,
            supportedBackends: new[] { new BackendOption(backend, backend) },
            defaultMaxTokens: 256,
            maxTokensPinned: false,
            defaultVideoFrames: 0, defaultVideoFps: 0, defaultVideoWidth: 0,
            defaultVideoHeight: 0, defaultVideoSteps: 0, defaultVideoMode: null,
            uploadDirectory: uploads,
            logDirectory: Path.Combine(_root, "logs"),
            fileLoggingEnabled: false,
            samplingDefaults: null);

        var chat = new WebUiChatService(
            new ModelService(), new SessionManager(), options,
            new UploadStoragePolicy(uploads), new SkillRegistry(new SkillRegistryOptions()),
            codeRunner: null, workspaces: null, codeArtifacts: null,
            NullLoggerFactory.Instance);

        _server = new LoopbackServer(NullLogger.Instance);
        _server.MapWebUi(chat, uploads);
        _server.Start();
        _client = Client(_server);
    }

    /// <summary>
    /// Stage explicitly supplied Qwen-Image-2.1 files under the built-in entry's role-aware
    /// names, publish its companions, and host the DiT directly.
    /// </summary>
    private CatalogModel StartQwenImage(LiveMedia.QwenImageFiles files, string backend)
    {
        CatalogModel model = DiffusionModelFixture.QwenImage21;
        _diffusionStore = new ModelStore(Path.Combine(_root, "diffusion-models"));
        Directory.CreateDirectory(_diffusionStore.DirectoryFor(model));
        foreach ((string name, string source) in LinksFor(model, files))
            File.CreateSymbolicLink(Path.Combine(_diffusionStore.DirectoryFor(model), name), source);

        DiffusionCompanions.Publish(model, _diffusionStore);
        StartDirect(_diffusionStore.PathFor(model, model.Weights), backend);
        return model;
    }

    private static HttpClient Client(LoopbackServer server)
    {
        // A Wan clip and a Qwen-Image-2.1 edit are both minutes of work on a laptop, and an
        // HttpClient that gives up at 100 seconds turns that into a cancelled request
        // whose failure looks nothing like the slowness that caused it.
        var client = new HttpClient { BaseAddress = new Uri(server.BaseUrl), Timeout = TimeSpan.FromHours(2) };
        client.DefaultRequestHeaders.Add("Cookie", $"{LoopbackServer.TokenCookie}={server.Token}");
        return client;
    }

    private async Task LoadAsync(string modelFileName, string backend, string? mmproj = null)
    {
        HttpResponseMessage response = await _client!.PostAsJsonAsync("/api/models/load", new
        {
            model = modelFileName,
            backend,
            mmproj,
        });
        Assert.True(response.IsSuccessStatusCode,
            $"loading {modelFileName} on {backend} failed: {(int)response.StatusCode} {await response.Content.ReadAsStringAsync()}");
    }

    private async Task<JsonElement> UploadAsync(byte[] content, string fileName)
    {
        using var form = new MultipartFormDataContent();
        var part = new ByteArrayContent(content);
        part.Headers.ContentType = new MediaTypeHeaderValue("application/octet-stream");
        form.Add(part, "file", fileName);

        using HttpResponseMessage response = await _client!.PostAsync("/api/upload", form);
        string payload = await response.Content.ReadAsStringAsync();
        Assert.True(response.IsSuccessStatusCode, $"uploading {fileName} failed: {(int)response.StatusCode} {payload}");
        return JsonSerializer.Deserialize<JsonElement>(payload);
    }

    private async Task<List<JsonElement>> StreamAsync(string route, object body)
    {
        using var request = new HttpRequestMessage(HttpMethod.Post, route)
        {
            Content = new StringContent(JsonSerializer.Serialize(body), Encoding.UTF8, "application/json"),
        };
        using HttpResponseMessage response = await _client!.SendAsync(request, HttpCompletionOption.ResponseHeadersRead);
        Assert.True(response.IsSuccessStatusCode,
            $"{route} failed: {(int)response.StatusCode} {await response.Content.ReadAsStringAsync()}");

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

    private static string TextOf(IEnumerable<JsonElement> frames)
    {
        var text = new StringBuilder();
        foreach (JsonElement frame in frames)
        {
            if (frame.TryGetProperty("token", out JsonElement token) && token.GetString() is { } piece)
                text.Append(piece);
            else if (frame.TryGetProperty("replace", out JsonElement replace) && replace.GetString() is { } whole)
                text.Clear().Append(whole);
        }
        return text.ToString();
    }

    private static int PromptTokensOf(IEnumerable<JsonElement> frames) =>
        frames.Last(f => f.TryGetProperty("done", out _)).GetProperty("promptTokens").GetInt32();

    private static void AssertMentionsOneOf(string answer, string label, params string[] words)
    {
        Assert.True(
            words.Any(w => answer.Contains(w, StringComparison.OrdinalIgnoreCase)),
            $"nothing in the answer named {label} (looked for {string.Join(", ", words)}); it said: {answer}");
    }

    private async Task<byte[]> FetchAsync(string url)
    {
        using HttpResponseMessage response = await _client!.GetAsync(url);
        Assert.True(response.IsSuccessStatusCode, $"{url} answered {(int)response.StatusCode}");
        return await response.Content.ReadAsByteArrayAsync();
    }

    // ---- understanding ---------------------------------------------------------------

    [MultimodalModelFact]
    public async Task APictureIsDescribedByWhatIsActuallyInIt()
    {
        Assert.Null(LiveMedia.UnavailableMultimodal(out CatalogModel model, out string weights, out string projector));
        StartApp(model, new Dictionary<string, string>
        {
            [model.Weights.FileName] = weights,
            [model.Projector!.FileName] = projector,
        });
        await LoadAsync(model.Weights.FileName, LiveMedia.Backend("ggml_cpu"), model.Projector.FileName);

        // The projector has to be the one the app hosts, not merely present: /api/models
        // reports what was loaded, and this is the field the page reads before it lets
        // anyone attach a photo.
        JsonElement models = JsonSerializer.Deserialize<JsonElement>(await _client!.GetStringAsync("/api/models"));
        Assert.False(string.IsNullOrEmpty(models.GetProperty("loaded").GetString()));
        Assert.Equal(model.Projector.FileName, models.GetProperty("loadedMmProj").GetString());
        Assert.True(models.GetProperty("visionReady").GetBoolean());
        Assert.True(models.GetProperty("acceptsVisionProjector").GetBoolean());

        JsonElement upload = await UploadAsync(MediaFixtures.RedCircleOnWhitePng(), "circle.png");
        Assert.Equal("image", upload.GetProperty("mediaType").GetString());
        string file = upload.GetProperty("file").GetString()!;

        JsonElement session = JsonSerializer.Deserialize<JsonElement>(
            await (await _client.PostAsync("/api/sessions?conversation=new", null)).Content.ReadAsStringAsync());
        string sessionId = session.GetProperty("sessionId").GetString()!;

        List<JsonElement> withPicture = await StreamAsync("/api/chat", new
        {
            sessionId,
            messages = new[]
            {
                new { role = "user", content = "What shape and colour is in this picture? Answer in one short sentence.", imagePaths = new[] { file } },
            },
            maxTokens = 64,
            think = false,
            // Greedy. The catalog's sampling defaults for this family are temperature
            // 1.0 / top-k 64 / top-p 0.95, which makes a test that asserts a specific
            // WORD a coin toss -- and a flaky assertion is worse than no assertion,
            // because it reads as evidence either way. Whether the model can see the
            // picture is a property of the pipeline, not of the sampler.
            temperature = 0.0,
        });

        // The MECHANICAL check goes first, deliberately. "The model did not say red"
        // and "the image never reached the model" are different failures with different
        // owners -- one is the model's, one is this app's -- and asserting the
        // description first reports both as the same red line, which is exactly how an
        // hour goes into blaming a pipeline that was working. The soft-token span an
        // image expands to is hundreds of tokens, so a prompt that did not grow is a
        // projector that did nothing, and that is knowable before any prose is judged.
        List<JsonElement> withoutPicture = await StreamAsync("/api/chat", new
        {
            sessionId,
            messages = new[] { new { role = "user", content = "What shape and colour is in this picture? Answer in one short sentence." } },
            maxTokens = 16,
            think = false,
        });
        int with = PromptTokensOf(withPicture), without = PromptTokensOf(withoutPicture);
        Console.WriteLine($"scenario picture: prompt {with} tokens with the image, {without} without");
        Assert.True(with > without + 32,
            $"the prompt did not grow for the image: {with} tokens with it, {without} without — "
            + "the projector encoded nothing, so this is the app and not the model");

        string answer = TextOf(withPicture);
        AssertMentionsOneOf(answer, "the colour", "red", "crimson", "scarlet");
        AssertMentionsOneOf(answer, "the shape", "circle", "circular", "disc", "disk", "dot", "round", "sphere", "ball");
    }

    [MultimodalModelFact]
    public async Task APictureIsRejectedBeforeGenerationWhenTheProjectorIsMissing()
    {
        Assert.Null(LiveMedia.UnavailableMultimodal(out CatalogModel model, out string weights, out _));
        AgentAppHost host = StartApp(model, new Dictionary<string, string>
        {
            [model.Weights.FileName] = weights,
        });

        // This is an intentional text-only load of a multimodal checkpoint. It is a
        // valid state when optional downloads are disabled; what must never be valid is
        // pretending an image was analyzed in that state.
        await Task.Run(() => host.UseModel(model));
        JsonElement models = JsonSerializer.Deserialize<JsonElement>(await _client!.GetStringAsync("/api/models"));
        Assert.Equal(JsonValueKind.Null, models.GetProperty("loadedMmProj").ValueKind);
        Assert.False(models.GetProperty("visionReady").GetBoolean());
        Assert.True(models.GetProperty("acceptsVisionProjector").GetBoolean());

        JsonElement upload = await UploadAsync(MediaFixtures.RedCircleOnWhitePng(), "circle.png");
        string file = upload.GetProperty("file").GetString()!;
        using var request = new HttpRequestMessage(HttpMethod.Post, "/api/chat")
        {
            Content = JsonContent.Create(new
            {
                messages = new[]
                {
                    new
                    {
                        role = "user",
                        content = "What is in this picture?",
                        imagePaths = new[] { file },
                        // The app host has a file tool, and this exact attachment would
                        // be stageable. A projector-capable checkpoint must still fail
                        // closed instead of silently downgrading its broken vision path.
                        attachments = new[]
                        {
                            new { file, fileName = "circle.png", mediaType = "image" },
                        },
                    },
                },
                maxTokens = 32,
                think = false,
            }),
        };

        using HttpResponseMessage response = await _client.SendAsync(request, HttpCompletionOption.ResponseHeadersRead);
        string payload = await response.Content.ReadAsStringAsync();
        Assert.Equal(System.Net.HttpStatusCode.BadRequest, response.StatusCode);
        JsonElement rejection = JsonSerializer.Deserialize<JsonElement>(payload);
        Assert.Equal("vision_not_ready", rejection.GetProperty("code").GetString());
        Assert.Contains("projector", rejection.GetProperty("error").GetString(), StringComparison.OrdinalIgnoreCase);
    }

    [MultimodalModelFact]
    public async Task ASoundIsHeardRatherThanIgnored()
    {
        // Gemma 4's mmproj carries an audio tower as well as a vision one — its GGUF
        // declares clip.has_audio_encoder alongside clip.has_vision_encoder, and
        // ModelMultimodalInjector.LoadProjectors loads both from the same file — so this
        // is a real path and not a hopeful one. What it is given is a pure tone, because
        // nothing in this repository can synthesise speech, and a test that claimed a
        // transcript for audio that contains no words would be asserting a hallucination.
        Assert.Null(LiveMedia.UnavailableMultimodal(out CatalogModel model, out string weights, out string projector));
        Assert.True(model.Modalities.HasFlag(CatalogModalities.Audio),
            $"{model.Id} is not listed as an audio model; the catalog and the projector disagree");

        StartApp(model, new Dictionary<string, string>
        {
            [model.Weights.FileName] = weights,
            [model.Projector!.FileName] = projector,
        });
        await LoadAsync(model.Weights.FileName, LiveMedia.Backend("ggml_cpu"), model.Projector.FileName);

        JsonElement upload = await UploadAsync(MediaFixtures.ToneWav(seconds: 3.0, hertz: 440), "tone.wav");
        Assert.Equal("audio", upload.GetProperty("mediaType").GetString());
        string file = upload.GetProperty("file").GetString()!;

        JsonElement session = JsonSerializer.Deserialize<JsonElement>(
            await (await _client!.PostAsync("/api/sessions?conversation=new", null)).Content.ReadAsStringAsync());
        string sessionId = session.GetProperty("sessionId").GetString()!;

        const string question = "Describe this audio clip in one short sentence.";
        List<JsonElement> withSound = await StreamAsync("/api/chat", new
        {
            sessionId,
            messages = new[] { new { role = "user", content = question, audioPaths = new[] { file } } },
            maxTokens = 64,
            think = false,
        });
        List<JsonElement> withoutSound = await StreamAsync("/api/chat", new
        {
            sessionId,
            messages = new[] { new { role = "user", content = question } },
            maxTokens = 16,
            think = false,
        });

        // The hard half: three seconds of audio becomes a long span of soft tokens, and a
        // clip that never reached the audio tower cannot have lengthened the prompt.
        Assert.True(PromptTokensOf(withSound) > PromptTokensOf(withoutSound) + 32,
            $"the prompt did not grow for the audio: {PromptTokensOf(withSound)} tokens with it, "
            + $"{PromptTokensOf(withoutSound)} without — the audio tower encoded nothing");

        // The soft half: a steady 440 Hz sine has a small vocabulary of honest
        // descriptions, and "I cannot hear anything" is not among them.
        string answer = TextOf(withSound);
        Assert.False(string.IsNullOrWhiteSpace(answer), "the model said nothing about the clip");
        AssertMentionsOneOf(answer, "the sound",
            "tone", "beep", "hum", "buzz", "sine", "pitch", "note", "frequency", "sound", "audio", "ring");
    }

    [MultimodalModelFact]
    public async Task AClipIsDescribedByWhatChangesAcrossIt()
    {
        // There is no video tower anywhere in this engine. A clip becomes sampled frames
        // at upload time and the page sends those frames as images — which is why the
        // frames the upload reply lists are what this test attaches, exactly as
        // index.html does.
        Assert.Null(LiveMedia.UnavailableMultimodal(out CatalogModel model, out string weights, out string projector));
        StartApp(model, new Dictionary<string, string>
        {
            [model.Weights.FileName] = weights,
            [model.Projector!.FileName] = projector,
        });
        await LoadAsync(model.Weights.FileName, LiveMedia.Backend("ggml_cpu"), model.Projector.FileName);

        byte[] clip = MediaFixtures.TwoHalvesMp4(Path.Combine(_root, "fixtures"), out _);
        JsonElement upload = await UploadAsync(clip, "clip.mp4");
        Assert.Equal("video", upload.GetProperty("mediaType").GetString());

        string[] frames = upload.GetProperty("frames").EnumerateArray().Select(f => f.GetString()!).ToArray();
        Assert.True(frames.Length >= 2, "a clip has to yield at least two frames for a change to be visible");

        JsonElement session = JsonSerializer.Deserialize<JsonElement>(
            await (await _client!.PostAsync("/api/sessions?conversation=new", null)).Content.ReadAsStringAsync());

        List<JsonElement> answered = await StreamAsync("/api/chat", new
        {
            sessionId = session.GetProperty("sessionId").GetString(),
            messages = new[]
            {
                new
                {
                    role = "user",
                    content = "These frames are in order from one short video. What colour does the screen start as, and what colour does it end as?",
                    imagePaths = frames,
                },
            },
            maxTokens = 96,
            think = false,
        });

        string answer = TextOf(answered);
        AssertMentionsOneOf(answer, "the first half", "red", "crimson", "scarlet");
        AssertMentionsOneOf(answer, "the second half", "blue", "azure", "navy");
    }

    // ---- image editing ---------------------------------------------------------------
    // Route tests, not quality tests: each asks for the smallest output Qwen-Image-2.1
    // accepts and two denoising steps, so a run proves the files load, the edit runs end
    // to end and a real PNG comes back, in minutes rather than the default 2048x2048 x 40.

    private const int SmokeSize = 256;
    private const int SmokeSteps = 2;

    [QwenImageModelFact]
    public async Task AnEditedPictureComesBackAsARealImageAtASensibleSize()
    {
        Assert.Null(LiveMedia.UnavailableQwenImage(out LiveMedia.QwenImageFiles files));
        string backend = LiveMedia.Backend("ggml_metal");
        CatalogModel model = StartQwenImage(files, backend);

        await LoadAsync(model.Weights.FileName, backend);

        JsonElement upload = await UploadAsync(MediaFixtures.RedCircleOnWhitePng(512), "circle.png");
        string file = upload.GetProperty("file").GetString()!;

        using HttpResponseMessage response = await _client!.PostAsJsonAsync("/api/image-edit", new
        {
            prompt = "Change the red circle to a blue square. Keep the white background.",
            imagePaths = new[] { file },
            seed = 42,
            width = SmokeSize,
            height = SmokeSize,
            steps = SmokeSteps,
        });
        string payload = await response.Content.ReadAsStringAsync();
        Assert.True(response.IsSuccessStatusCode, $"the edit failed: {(int)response.StatusCode} {payload}");

        JsonElement body = JsonSerializer.Deserialize<JsonElement>(payload);
        Assert.True(body.GetProperty("ok").GetBoolean());
        int width = body.GetProperty("width").GetInt32();
        int height = body.GetProperty("height").GetInt32();
        Assert.Equal((SmokeSize, SmokeSize), (width, height));

        // Fetched through the URL the page would use, then decoded: a reply naming a
        // file that cannot be served or cannot be read is a broken image in the chat,
        // and the width and height above are only the model's own claim about it.
        byte[] png = await FetchAsync(body.GetProperty("url").GetString()!);
        byte[] rgba = MediaCodecs.Image.DecodeRgba(png, out int decodedWidth, out int decodedHeight);
        Assert.Equal((width, height), (decodedWidth, decodedHeight));
        Assert.Equal(decodedWidth * decodedHeight * 4, rgba.Length);

        // A diffusion failure that still writes a file writes a flat one; the result has
        // to have more than one colour in it.
        Assert.True(rgba.Where((_, i) => i % 4 == 0).Distinct().Count() > 4, "the edited image is a flat fill, not a picture");
    }

    [QwenImageModelFact]
    public async Task TheStreamingEditShowsItsWorkAndThenTheFinishedPicture()
    {
        Assert.Null(LiveMedia.UnavailableQwenImage(out LiveMedia.QwenImageFiles files));
        string backend = LiveMedia.Backend("ggml_metal");
        CatalogModel model = StartQwenImage(files, backend);

        await LoadAsync(model.Weights.FileName, backend);

        JsonElement upload = await UploadAsync(MediaFixtures.RedCircleOnWhitePng(512), "circle.png");

        List<JsonElement> frames = await StreamAsync("/api/image-edit/stream", new
        {
            prompt = "Make the circle blue.",
            imagePaths = new[] { upload.GetProperty("file").GetString() },
            seed = 7,
            width = SmokeSize,
            height = SmokeSize,
            steps = SmokeSteps,
        });

        // Denoising takes a minute or more at real sizes. Without progress frames the
        // page has a spinner and no way to tell a slow edit from a hung one, so the
        // frames are the feature, not decoration. Two steps still earn one preview.
        List<JsonElement> progress = frames.Where(f => f.TryGetProperty("imageEdit", out _)).ToList();
        Assert.NotEmpty(progress);

        List<JsonElement> previews = progress
            .Where(f => f.TryGetProperty("image", out JsonElement image) && image.ValueKind == JsonValueKind.String)
            .ToList();
        Assert.True(previews.Count > 0, "no step carried a decoded preview; the page would show progress with nothing in it");
        Assert.StartsWith("data:image/png;base64,", previews[0].GetProperty("image").GetString(), StringComparison.Ordinal);

        JsonElement last = frames[^1];
        Assert.True(last.GetProperty("done").GetBoolean());
        Assert.False(last.TryGetProperty("error", out JsonElement error) && error.ValueKind == JsonValueKind.String,
            "the stream ended with an error: " + (error.ValueKind == JsonValueKind.String ? error.GetString() : ""));

        byte[] png = await FetchAsync(last.GetProperty("url").GetString()!);
        MediaCodecs.Image.DecodeRgba(png, out int width, out int height);
        Assert.Equal(last.GetProperty("width").GetInt32(), width);
        Assert.Equal(last.GetProperty("height").GetInt32(), height);
        Assert.Equal((SmokeSize, SmokeSize), (width, height));
    }

    /// <summary>
    /// Map each file found on disk onto the name the test fixture gives its role, no
    /// matter what the local copy is called. An edit that runs then proves the VAE,
    /// text encoder and vision projector were all found under those names.
    /// </summary>
    private static Dictionary<string, string> LinksFor(CatalogModel model, LiveMedia.QwenImageFiles files) =>
        new(StringComparer.Ordinal)
        {
            [model.Weights.FileName] = files.Dit,
            [NameOf(model, CatalogFileRole.TextEncoder)] = files.TextEncoder,
            [NameOf(model, CatalogFileRole.Vae)] = files.Vae,
            [NameOf(model, CatalogFileRole.VisionProjector)] = files.VisionProjector,
        };

    private static string NameOf(CatalogModel model, CatalogFileRole role) =>
        model.Files.Single(f => f.Role == role).FileName;

    // ---- video generation ------------------------------------------------------------

    [VideoModelFact]
    public async Task AGeneratedClipComesBackAsAnMp4TheAppCanRead()
    {
        Assert.Null(LiveMedia.UnavailableVideo(out LiveMedia.VideoFiles files));

        // The Wan companions have no catalog entry to be published from, so they are
        // named the way the desktop's --video-vae / --video-text-encoder flags name them.
        Environment.SetEnvironmentVariable("TS_VIDEO_VAE", files.Vae);
        Environment.SetEnvironmentVariable("TS_VIDEO_TEXT_ENCODER", files.TextEncoder);
        try
        {
            StartDirect(files.Dit, LiveMedia.Backend("ggml_metal"));
            await LoadAsync(Path.GetFileName(files.Dit), LiveMedia.Backend("ggml_metal"));

            using HttpResponseMessage response = await _client!.PostAsJsonAsync("/api/video-generate", new
            {
                prompt = "a red balloon rising over a calm blue sea",
                // The smallest request the checkpoint will accept: this is a test of the
                // route and the file it produces, not of what the model is capable of.
                width = 256,
                height = 256,
                frames = 17,
                fps = 8,
                seed = 3,
            });
            string payload = await response.Content.ReadAsStringAsync();
            Assert.True(response.IsSuccessStatusCode, $"the generation failed: {(int)response.StatusCode} {payload}");

            JsonElement body = JsonSerializer.Deserialize<JsonElement>(payload);
            Assert.True(body.GetProperty("ok").GetBoolean());
            Assert.True(body.GetProperty("frames").GetInt32() > 1);

            // "h264" is the only codec a WebView will play; mp4v writes a file that
            // opens in VLC and shows nothing in the app, which is worth knowing about
            // here rather than from a user.
            Assert.Equal("h264", body.GetProperty("codec").GetString());

            byte[] mp4 = await FetchAsync(body.GetProperty("url").GetString()!);
            string path = Path.Combine(_root, "generated.mp4");
            await File.WriteAllBytesAsync(path, mp4);

            VideoInfo probed = MediaCodecs.Video.Probe(path);
            Assert.Equal(body.GetProperty("frames").GetInt32(), probed.FrameCount);
            Assert.True(probed.Fps > 0, "the generated clip has no frame rate");
        }
        finally
        {
            Environment.SetEnvironmentVariable("TS_VIDEO_VAE", null);
            Environment.SetEnvironmentVariable("TS_VIDEO_TEXT_ENCODER", null);
        }
    }
}
